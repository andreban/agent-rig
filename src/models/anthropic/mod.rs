// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

//! Anthropic Claude provider adapter.
//!
//! Implements [`LlmModel`] against the Claude Messages API using the
//! [`antropologia`](https://github.com/andreban/antropologia) client.
//! Requires the `anthropic` Cargo feature.

/// Re-export of the underlying [`antropologia`] crate.
pub use antropologia;
/// Re-exports of [`antropologia`]'s effort and thinking settings, used by
/// [`AnthropicModelBuilder::effort`] and [`AnthropicModelBuilder::thinking`].
pub use antropologia::prelude::{Effort, ThinkingConfig, ThinkingDisplay};

use std::pin::Pin;

use antropologia::prelude::{
    AntropologiaClient, ContentBlock, ContentDelta, Message as ClaudeMessage, MessageAccumulator,
    MessagesRequest, MessagesResponse, Role as ClaudeRole, StopReason, StreamEvent,
    Tool as ClaudeTool, ToolResultContent as ClaudeToolResultContent, Usage as ClaudeUsage,
};
use async_trait::async_trait;
use futures_util::{Stream, StreamExt};
use serde_json::{Value, json};

use crate::{
    error::Error,
    model::{
        LlmModel, Media, Message, MessageContent, ModelRequest, ModelResponse, ModelStreamChunk,
        Role, TokenUsage, ToolCall, ToolResultContent, tool_result_json,
    },
    tools::ToolDefinition,
};

/// LLM provider backed by Anthropic Claude.
///
/// Each assistant turn's content blocks (thinking blocks with their signatures, text, and
/// tool calls) are emitted as opaque provider metadata (`{"anthropic": {"content": [...]}}`)
/// that the runner stores on the assistant [`Message`]. This adapter replays them unchanged on
/// later requests, as Claude requires when thinking precedes tool calls. Summarized thinking,
/// when enabled with [`AnthropicModelBuilder::thinking`], is also surfaced as
/// [`ModelResponse::thinking`] / [`ModelStreamChunk::Thinking`].
///
/// A refusal (`stop_reason: "refusal"`) is returned as [`Error::Provider`].
///
/// The underlying [`AntropologiaClient`] can be accessed via [`AnthropicModel::client`].
///
/// # Examples
///
/// ```no_run
/// use agent_rig::models::anthropic::{AnthropicModel, Effort, ThinkingConfig};
///
/// // Simple
/// let model = AnthropicModel::new("ANTHROPIC_API_KEY", "claude-opus-5-5");
///
/// // With settings
/// let model = AnthropicModel::builder("ANTHROPIC_API_KEY", "claude-opus-5-5")
///     .max_tokens(32_000)
///     .effort(Effort::High)
///     .thinking(ThinkingConfig::adaptive_summarized())
///     .build();
/// ```
pub struct AnthropicModel {
    client: AntropologiaClient,
    model: String,
    settings: Settings,
}

/// Generation settings applied to every request.
#[derive(Debug, Clone, Default)]
struct Settings {
    max_tokens: Option<u32>,
    effort: Option<Effort>,
    thinking: Option<ThinkingConfig>,
    temperature: Option<f32>,
    top_p: Option<f32>,
}

impl AnthropicModel {
    /// Creates a new `AnthropicModel` with default generation settings.
    ///
    /// - `api_key` — your Anthropic API key.
    /// - `model` — the Claude model ID (e.g. `"claude-opus-5-5"`).
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self::builder(api_key, model).build()
    }

    /// Returns an [`AnthropicModelBuilder`] for constructing an `AnthropicModel` with custom
    /// settings.
    pub fn builder(api_key: impl Into<String>, model: impl Into<String>) -> AnthropicModelBuilder {
        AnthropicModelBuilder::new(api_key, model)
    }

    /// Returns a reference to the underlying [`AntropologiaClient`].
    pub fn client(&self) -> &AntropologiaClient {
        &self.client
    }

    /// Returns the model identifier.
    pub fn model(&self) -> &str {
        &self.model
    }
}

/// Builder for [`AnthropicModel`].
pub struct AnthropicModelBuilder {
    api_key: String,
    model: String,
    base_url: Option<String>,
    settings: Settings,
}

impl AnthropicModelBuilder {
    /// Creates a new builder.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: model.into(),
            base_url: None,
            settings: Settings::default(),
        }
    }

    /// Sets a custom API base URL (e.g. for a proxy).
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Sets the maximum tokens per response, thinking included. Defaults to
    /// [`antropologia::DEFAULT_MAX_TOKENS`].
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.settings.max_tokens = Some(max_tokens);
        self
    }

    /// Sets the effort level (`output_config.effort`). When unset, the model's default applies.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use agent_rig::models::anthropic::{AnthropicModel, Effort};
    ///
    /// let model = AnthropicModel::builder("ANTHROPIC_API_KEY", "claude-opus-5-5")
    ///     .effort(Effort::Low)
    ///     .build();
    /// ```
    pub fn effort(mut self, effort: Effort) -> Self {
        self.settings.effort = Some(effort);
        self
    }

    /// Sets extended thinking. Current models think adaptively by default, with thinking text
    /// omitted; use [`ThinkingConfig::adaptive_summarized`] to receive a summary.
    pub fn thinking(mut self, thinking: ThinkingConfig) -> Self {
        self.settings.thinking = Some(thinking);
        self
    }

    /// Sets sampling temperature. Recent models reject non-default sampling parameters.
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.settings.temperature = Some(temperature);
        self
    }

    /// Sets nucleus sampling. Recent models reject non-default sampling parameters.
    pub fn top_p(mut self, top_p: f32) -> Self {
        self.settings.top_p = Some(top_p);
        self
    }

    /// Builds the [`AnthropicModel`].
    pub fn build(self) -> AnthropicModel {
        let mut client_builder = AntropologiaClient::builder(self.api_key);
        if let Some(base_url) = self.base_url {
            client_builder = client_builder.base_url(base_url);
        }
        AnthropicModel {
            client: client_builder.build(),
            model: self.model,
            settings: self.settings,
        }
    }
}

/// Translates a [`ToolDefinition`] into a Claude [`ClaudeTool`].
impl From<&ToolDefinition> for ClaudeTool {
    fn from(def: &ToolDefinition) -> ClaudeTool {
        ClaudeTool::new(
            def.name.clone(),
            def.description.clone(),
            def.parameters.clone().to_value(),
        )
    }
}

/// Key under which this adapter stores its state in [`Message::provider_metadata`]. Other
/// adapters' keys are ignored.
const METADATA_KEY: &str = "anthropic";

/// Wraps a turn's content blocks as provider metadata.
fn content_metadata(content: &[ContentBlock]) -> Option<Value> {
    serde_json::to_value(content)
        .ok()
        .map(|content| serde_json::json!({ METADATA_KEY: { "content": content } }))
}

/// Reads back the content blocks this adapter stored on an assistant message.
fn replayed_content(msg: &Message) -> Option<Vec<ContentBlock>> {
    let content = msg
        .provider_metadata
        .as_ref()?
        .get(METADATA_KEY)?
        .get("content")?;
    serde_json::from_value(content.clone()).ok()
}

/// Converts a JSON value into text for a Claude tool-result block.
fn json_to_text(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Converts a single [`Media`] into a Claude content block (`image` for supported image types,
/// `document` for `application/pdf`). Returns an error for unsupported MIME types.
fn media_block(media: &Media) -> Result<ContentBlock, Error> {
    match media {
        Media::Inline { mime_type, data } => {
            let block_type = match mime_type.as_str() {
                "image/jpeg" | "image/png" | "image/gif" | "image/webp" => "image",
                "application/pdf" => "document",
                other => {
                    return Err(Error::Provider(format!(
                        "Anthropic adapter does not support tool-result media type: {other}"
                    )));
                }
            };
            Ok(ContentBlock::Other(json!({
                "type": block_type,
                "source": {
                    "type": "base64",
                    "media_type": mime_type,
                    "data": data,
                }
            })))
        }
    }
}

/// Converts tool-result content blocks into a Claude tool-result block. agent-rig wraps results
/// as `{"success": ...}` or `{"error": ...}`; the latter sets `is_error`. When media blocks are
/// present, `content` is sent as an array of `text` and `image`/`document` blocks; otherwise it
/// is sent as a string.
fn tool_result_block(
    tool_use_id: &str,
    content: &[ToolResultContent],
) -> Result<ContentBlock, Error> {
    let result = tool_result_json(content);
    let is_error = result
        .as_object()
        .is_some_and(|map| map.len() == 1 && map.contains_key("error"))
        .then_some(true);
    let has_media = content
        .iter()
        .any(|part| matches!(part, ToolResultContent::Media(_)));
    if !has_media {
        return Ok(ContentBlock::ToolResult {
            tool_use_id: tool_use_id.to_string(),
            content: ClaudeToolResultContent::Text(json_to_text(&result)),
            is_error,
            cache_control: None,
        });
    }

    let blocks = content
        .iter()
        .map(|part| match part {
            ToolResultContent::Json(v) => Ok(ContentBlock::text(json_to_text(v))),
            ToolResultContent::Media(m) => media_block(m),
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ContentBlock::ToolResult {
        tool_use_id: tool_use_id.to_string(),
        content: ClaudeToolResultContent::Blocks(blocks),
        is_error,
        cache_control: None,
    })
}

/// Content blocks for one agent-rig message.
fn content_blocks(msg: &Message) -> Result<Vec<ContentBlock>, Error> {
    if msg.role == Role::Assistant
        && let Some(content) = replayed_content(msg)
    {
        return Ok(content);
    }
    match &msg.content {
        MessageContent::Text(text) => Ok(vec![ContentBlock::text(text.clone())]),
        MessageContent::ToolCalls(calls) => Ok(calls
            .iter()
            .map(|call| {
                ContentBlock::tool_use(call.id.clone(), call.name.clone(), call.args.clone())
            })
            .collect()),
        MessageContent::ToolResult { tool_call, content } => {
            Ok(vec![tool_result_block(&tool_call.id, content)?])
        }
    }
}

/// Builds a Claude [`MessagesRequest`] from a [`ModelRequest`].
fn build_messages_request(
    model: &str,
    settings: &Settings,
    request: ModelRequest,
) -> Result<MessagesRequest, Error> {
    let mut messages: Vec<ClaudeMessage> = Vec::new();
    for msg in request.messages.iter() {
        let role = match msg.role {
            Role::User => ClaudeRole::User,
            Role::Assistant => ClaudeRole::Assistant,
        };
        let blocks = content_blocks(msg)?;
        // Claude requires alternating roles, so consecutive same-role messages (e.g. one tool
        // result per call) are merged into one.
        match messages.last_mut() {
            Some(last) if last.role == role => last.content.extend(blocks),
            _ => messages.push(ClaudeMessage::with_blocks(role, blocks)),
        }
    }

    let mut builder = MessagesRequest::builder(model).messages(messages);
    if let Some(system) = request.system.filter(|s| !s.is_empty()) {
        builder = builder.system(system);
    }
    if !request.tools.is_empty() {
        builder = builder.tools(request.tools.iter().map(ClaudeTool::from).collect());
    }
    if let Some(schema) = request.output_schema {
        builder = builder.json_schema(schema.to_value());
    }
    if let Some(max_tokens) = settings.max_tokens {
        builder = builder.max_tokens(max_tokens);
    }
    if let Some(effort) = settings.effort {
        builder = builder.effort(effort);
    }
    if let Some(thinking) = settings.thinking.clone() {
        builder = builder.thinking(thinking);
    }
    if let Some(temperature) = settings.temperature {
        builder = builder.temperature(temperature);
    }
    if let Some(top_p) = settings.top_p {
        builder = builder.top_p(top_p);
    }
    Ok(builder.build())
}

/// Converts Claude usage into [`TokenUsage`]. Claude reports cache reads and writes apart from
/// `input_tokens`; agent-rig counts them inside `input_tokens`.
fn to_token_usage(usage: &ClaudeUsage) -> TokenUsage {
    TokenUsage {
        input_tokens: Some(usage.total_input_tokens()),
        output_tokens: Some(usage.output_tokens),
        cached_input_tokens: usage.cache_read_input_tokens,
        thinking_tokens: None,
        tool_use_prompt_tokens: None,
    }
}

/// Fails on a refusal, which otherwise looks like an empty or truncated answer.
fn check_refusal(response: &MessagesResponse) -> Result<(), Error> {
    if response.stop_reason == Some(StopReason::Refusal) {
        let details = response
            .stop_details
            .as_ref()
            .map_or_else(String::new, |d| format!(": {d}"));
        return Err(Error::Provider(format!(
            "Claude declined the request{details}"
        )));
    }
    Ok(())
}

fn tool_calls(response: &MessagesResponse) -> Vec<ToolCall> {
    response
        .tool_uses()
        .into_iter()
        .map(|(id, name, input)| ToolCall::new(id.to_string(), name.to_string(), input.clone()))
        .collect()
}

/// Converts a complete Claude response into a [`ModelResponse`].
fn model_response(response: MessagesResponse) -> Result<ModelResponse, Error> {
    check_refusal(&response)?;
    let tool_calls = tool_calls(&response);
    let text = response.text();
    Ok(ModelResponse {
        text: (tool_calls.is_empty() && !text.is_empty()).then_some(text),
        tool_calls,
        thinking: response.thinking(),
        provider_metadata: content_metadata(&response.content),
        token_usage: Some(to_token_usage(&response.usage)),
    })
}

/// Converts an [`antropologia`] error into an [`Error::ProviderTransient`] if
/// [retrying may succeed](antropologia::AntropologiaError::is_retryable), or
/// an [`Error::Provider`] if not.
fn provider_error(error: antropologia::AntropologiaError) -> Error {
    if error.is_retryable() {
        Error::ProviderTransient(error.to_string())
    } else {
        Error::Provider(error.to_string())
    }
}

#[async_trait]
impl LlmModel for AnthropicModel {
    async fn generate(&self, request: ModelRequest) -> Result<ModelResponse, Error> {
        let request = build_messages_request(&self.model, &self.settings, request)?;
        let response = self
            .client
            .messages(&request)
            .await
            .map_err(provider_error)?;
        model_response(response)
    }

    fn generate_stream(
        &self,
        request: ModelRequest,
    ) -> Pin<Box<dyn Stream<Item = Result<ModelStreamChunk, Error>> + Send + '_>> {
        Box::pin(async_stream::stream! {
            let request = match build_messages_request(&self.model, &self.settings, request) {
                Ok(request) => request,
                Err(e) => {
                    yield Err(e);
                    return;
                }
            };
            let mut stream = match self.client.messages_stream(&request).await {
                Ok(stream) => Box::pin(stream),
                Err(e) => {
                    yield Err(provider_error(e));
                    return;
                }
            };

            // Rebuilds the full message so its blocks can be replayed on later turns.
            let mut accumulator = MessageAccumulator::new();
            while let Some(event) = stream.next().await {
                let event = match event {
                    Ok(event) => event,
                    Err(e) => {
                        yield Err(provider_error(e));
                        return;
                    }
                };
                if let StreamEvent::ContentBlockDelta { delta, .. } = &event {
                    match delta {
                        ContentDelta::ThinkingDelta { thinking } if !thinking.is_empty() => {
                            yield Ok(ModelStreamChunk::Thinking(thinking.clone()));
                        }
                        ContentDelta::TextDelta { text } if !text.is_empty() => {
                            yield Ok(ModelStreamChunk::TextDelta(text.clone()));
                        }
                        _ => {}
                    }
                }
                if let Err(e) = accumulator.push(event) {
                    yield Err(provider_error(e));
                    return;
                }
            }

            let response = match accumulator.finish() {
                Ok(response) => response,
                Err(e) => {
                    yield Err(provider_error(e));
                    return;
                }
            };
            if let Err(e) = check_refusal(&response) {
                yield Err(e);
                return;
            }
            for call in tool_calls(&response) {
                yield Ok(ModelStreamChunk::ToolCall(call));
            }
            if let Some(metadata) = content_metadata(&response.content) {
                yield Ok(ModelStreamChunk::ProviderMetadata(metadata));
            }
            yield Ok(ModelStreamChunk::Usage(to_token_usage(&response.usage)));
        })
    }
}

#[cfg(test)]
mod tests;
