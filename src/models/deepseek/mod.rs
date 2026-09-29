// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

//! DeepSeek provider adapter.
//!
//! Implements [`LlmModel`] against the DeepSeek API using the
//! [`cetologia`](https://github.com/andreban/cetologia) client.
//! Requires the `deepseek` Cargo feature.

use std::pin::Pin;

use async_trait::async_trait;
use cetologia::prelude::{
    CetologiaClient, ChatCompletionRequest, ChatMessage, FunctionCall, ResponseFormat,
    Role as CetologiaRole, Tool as CetologiaTool, ToolCall as CetologiaToolCall,
    ToolCallAccumulator, ToolType, Usage as CetologiaUsage,
};
use futures_util::{Stream, StreamExt};
use serde_json::Value;

use crate::{
    error::Error,
    model::{
        LlmModel, MessageContent, ModelRequest, ModelResponse, ModelStreamChunk, Role, TokenUsage,
        ToolCall,
    },
    tools::ToolDefinition,
};

/// LLM provider backed by DeepSeek.
///
/// Supports both `deepseek-chat` (DeepSeek-V3) and `deepseek-reasoner` (DeepSeek-R1).
///
/// # Examples
///
/// ```no_run
/// use agent_rig::models::deepseek::DeepSeekModel;
///
/// // Simple
/// let model = DeepSeekModel::new("DEEPSEEK_API_KEY", "deepseek-chat");
///
/// // With settings
/// let model = DeepSeekModel::builder("DEEPSEEK_API_KEY", "deepseek-reasoner")
///     .temperature(0.6)
///     .max_tokens(4096)
///     .build();
/// ```
pub struct DeepSeekModel {
    client: CetologiaClient,
    model: String,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    top_p: Option<f32>,
}

impl DeepSeekModel {
    /// Creates a new `DeepSeekModel` with default generation settings.
    ///
    /// - `api_key` — your DeepSeek API key.
    /// - `model` — the DeepSeek model name (e.g. `"deepseek-chat"`, `"deepseek-reasoner"`).
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: CetologiaClient::new(api_key),
            model: model.into(),
            temperature: None,
            max_tokens: None,
            top_p: None,
        }
    }

    /// Returns a [`DeepSeekModelBuilder`] for constructing a `DeepSeekModel` with custom settings.
    pub fn builder(api_key: impl Into<String>, model: impl Into<String>) -> DeepSeekModelBuilder {
        DeepSeekModelBuilder::new(api_key, model)
    }
}

/// Builder for [`DeepSeekModel`].
pub struct DeepSeekModelBuilder {
    api_key: String,
    model: String,
    base_url: Option<String>,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    top_p: Option<f32>,
}

impl DeepSeekModelBuilder {
    /// Creates a new builder.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: model.into(),
            base_url: None,
            temperature: None,
            max_tokens: None,
            top_p: None,
        }
    }

    /// Sets a custom API base URL (e.g. for proxy or compatible endpoints).
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Sets sampling temperature.
    pub fn temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    /// Sets max tokens.
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Sets nucleus sampling threshold.
    pub fn top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Builds the [`DeepSeekModel`].
    pub fn build(self) -> DeepSeekModel {
        let mut client_builder = CetologiaClient::builder(self.api_key);
        if let Some(base_url) = self.base_url {
            client_builder = client_builder.base_url(base_url);
        }

        DeepSeekModel {
            client: client_builder.build(),
            model: self.model,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            top_p: self.top_p,
        }
    }
}

/// Translates a [`ToolDefinition`] into a DeepSeek [`CetologiaTool`].
impl From<&ToolDefinition> for CetologiaTool {
    fn from(def: &ToolDefinition) -> CetologiaTool {
        CetologiaTool::function(
            def.name.clone(),
            Some(def.description.clone()),
            Some(def.parameters.clone().into()),
        )
    }
}

/// Converts token usage from Cetologia into [`TokenUsage`].
fn to_token_usage(usage: &CetologiaUsage) -> Option<TokenUsage> {
    let cached = usage.prompt_cache_hit_tokens.or_else(|| {
        usage
            .prompt_tokens_details
            .as_ref()
            .and_then(|d| d.cached_tokens)
    });

    Some(TokenUsage {
        input_tokens: Some(usage.prompt_tokens),
        output_tokens: Some(usage.completion_tokens),
        cached_input_tokens: cached,
        thinking_tokens: None,
        tool_use_prompt_tokens: None,
    })
}

/// Helper to build a DeepSeek [`ChatCompletionRequest`] from a [`ModelRequest`].
fn build_chat_request(
    model: &str,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    top_p: Option<f32>,
    request: ModelRequest,
) -> ChatCompletionRequest {
    let mut messages: Vec<ChatMessage> = Vec::new();

    let mut system_prompt = request.system.unwrap_or_default();

    // If an output schema is requested, append JSON formatting instructions to system prompt
    if let Some(schema) = &request.output_schema {
        let schema_json = serde_json::to_string_pretty(&schema.clone().to_value()).unwrap_or_default();
        if !system_prompt.is_empty() {
            system_prompt.push_str("\n\n");
        }
        system_prompt.push_str("Respond with a valid JSON object matching this schema:\n");
        system_prompt.push_str(&schema_json);
    }

    if !system_prompt.is_empty() {
        messages.push(ChatMessage::system(system_prompt));
    }

    for msg in request.messages {
        match &msg.content {
            MessageContent::Text(text) => match msg.role {
                Role::User => messages.push(ChatMessage::user(text.clone())),
                Role::Assistant => messages.push(ChatMessage::assistant(text.clone())),
            },
            MessageContent::ToolCalls(calls) => {
                let tool_calls: Vec<CetologiaToolCall> = calls
                    .iter()
                    .map(|call| CetologiaToolCall {
                        id: call.id.clone(),
                        tool_type: ToolType::Function,
                        function: FunctionCall {
                            name: call.name.clone(),
                            arguments: call.args.to_string(),
                        },
                    })
                    .collect();
                messages.push(ChatMessage {
                    role: CetologiaRole::Assistant,
                    content: None,
                    reasoning_content: None,
                    tool_calls: Some(tool_calls),
                    ..Default::default()
                });
            }
            MessageContent::ToolResult { tool_call, result } => {
                messages.push(ChatMessage::tool(tool_call.id.clone(), result.to_string()));
            }
        }
    }

    let mut builder = ChatCompletionRequest::builder(model).messages(messages);

    if let Some(temp) = temperature {
        builder = builder.temperature(temp);
    }
    if let Some(max_t) = max_tokens {
        builder = builder.max_tokens(max_t);
    }
    if let Some(p) = top_p {
        builder = builder.top_p(p);
    }

    if !request.tools.is_empty() {
        let tools: Vec<CetologiaTool> = request.tools.iter().map(CetologiaTool::from).collect();
        builder = builder.tools(tools);
    }

    if request.output_schema.is_some() {
        builder = builder.response_format(ResponseFormat::JsonObject);
    }

    builder.build()
}

#[async_trait]
impl LlmModel for DeepSeekModel {
    async fn generate(&self, request: ModelRequest) -> Result<ModelResponse, Error> {
        let chat_request = build_chat_request(
            &self.model,
            self.temperature,
            self.max_tokens,
            self.top_p,
            request,
        );

        let response = self
            .client
            .chat(&chat_request)
            .await
            .map_err(|e| Error::Provider(e.to_string()))?;

        let choice = response
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| Error::Provider("empty choices in DeepSeek response".into()))?;

        let thinking = choice.message.reasoning_content;
        let token_usage = response.usage.as_ref().and_then(to_token_usage);

        let tool_calls: Vec<ToolCall> = choice
            .message
            .tool_calls
            .unwrap_or_default()
            .into_iter()
            .map(|tc| {
                let args: Value = serde_json::from_str(&tc.function.arguments)
                    .unwrap_or(Value::String(tc.function.arguments));
                ToolCall {
                    id: tc.id,
                    name: tc.function.name,
                    args,
                    provider_metadata: None,
                }
            })
            .collect();

        if !tool_calls.is_empty() {
            return Ok(ModelResponse {
                text: None,
                tool_calls,
                thinking,
                token_usage,
            });
        }

        Ok(ModelResponse {
            text: choice.message.content,
            tool_calls: vec![],
            thinking,
            token_usage,
        })
    }

    fn generate_stream(
        &self,
        request: ModelRequest,
    ) -> Pin<Box<dyn Stream<Item = Result<ModelStreamChunk, Error>> + Send + '_>> {
        Box::pin(async_stream::stream! {
            let chat_request = build_chat_request(
                &self.model,
                self.temperature,
                self.max_tokens,
                self.top_p,
                request,
            );

            let mut stream = match self.client.chat_stream(&chat_request).await {
                Ok(stream) => stream,
                Err(e) => {
                    yield Err(Error::Provider(e.to_string()));
                    return;
                }
            };

            let mut tool_accumulator = ToolCallAccumulator::new();
            let mut latest_usage: Option<TokenUsage> = None;

            while let Some(chunk_res) = stream.next().await {
                let chunk = match chunk_res {
                    Ok(chunk) => chunk,
                    Err(e) => {
                        yield Err(Error::Provider(e.to_string()));
                        return;
                    }
                };

                if let Some(usage) = chunk.usage.as_ref().and_then(to_token_usage) {
                    latest_usage = Some(usage);
                }

                for choice in chunk.choices {
                    if let Some(reasoning) = choice.delta.reasoning_content {
                        if !reasoning.is_empty() {
                            yield Ok(ModelStreamChunk::Thinking(reasoning));
                        }
                    }

                    if let Some(text) = choice.delta.content {
                        if !text.is_empty() {
                            yield Ok(ModelStreamChunk::TextDelta(text));
                        }
                    }

                    if let Some(tool_calls) = &choice.delta.tool_calls {
                        tool_accumulator.update_all(tool_calls);
                    }
                }
            }

            for tc in tool_accumulator.finish() {
                let args: Value = serde_json::from_str(&tc.function.arguments)
                    .unwrap_or(Value::String(tc.function.arguments));
                yield Ok(ModelStreamChunk::ToolCall(ToolCall {
                    id: tc.id,
                    name: tc.function.name,
                    args,
                    provider_metadata: None,
                }));
            }

            if let Some(usage) = latest_usage {
                yield Ok(ModelStreamChunk::Usage(usage));
            }
        })
    }
}

#[cfg(test)]
mod tests;
