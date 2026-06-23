// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

//! [Ollama](https://ollama.com/) provider adapter.
//!
//! Implements [`LlmModel`] against a local or remote
//! Ollama server using the [`ollama-rs`](https://crates.io/crates/ollama-rs)
//! client. Requires the `ollama` Cargo feature.

use std::pin::Pin;

use async_trait::async_trait;
use futures_util::{Stream, StreamExt};
use ollama_rs::{
    OllamaClient,
    types::chat::{
        ChatRequest, ChatResponse, Function, Message as OllamaMessage, Role as OllamaRole,
        Tool as OllamaTool, ToolCall as OllamaToolCall, ToolCallFunction, ToolType,
    },
    types::common::{Options, Stop},
};

pub use ollama_rs::types::common::{Think, ThinkLevel};

use crate::{
    error::Error,
    model::{
        LlmModel, MessageContent, ModelRequest, ModelResponse, ModelStreamChunk, Role, TokenUsage,
        ToolCall,
    },
    tools::ToolDefinition,
};

/// LLM provider backed by an [Ollama](https://ollama.com/) server.
///
/// Use [`OllamaModel::new`] for the simple case, or [`OllamaModel::builder`]
/// to configure generation settings such as temperature.
///
/// # Examples
///
/// ```no_run
/// use agent_rig::models::ollama::OllamaModel;
///
/// // Simple
/// let model = OllamaModel::new("http://localhost:11434", "llama3");
///
/// // With settings
/// let model = OllamaModel::builder("http://localhost:11434", "llama3")
///     .temperature(0.7)
///     .num_predict(512)
///     .build();
/// ```
pub struct OllamaModel {
    client: OllamaClient,
    model: String,
    options: Option<Options>,
    think: Option<Think>,
}

impl OllamaModel {
    /// Creates a new `OllamaModel` with default generation settings.
    ///
    /// - `server_address` — the base URL of the Ollama server (e.g. `"http://localhost:11434"`).
    /// - `model` — the model name to use (e.g. `"llama3"`, `"mistral"`).
    pub fn new(server_address: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: OllamaClient::new(server_address.into()),
            model: model.into(),
            options: None,
            think: None,
        }
    }

    /// Returns an [`OllamaModelBuilder`] for constructing an `OllamaModel` with custom settings.
    ///
    /// - `server_address` — the base URL of the Ollama server (e.g. `"http://localhost:11434"`).
    /// - `model` — the model name to use (e.g. `"llama3"`, `"mistral"`).
    pub fn builder(
        server_address: impl Into<String>,
        model: impl Into<String>,
    ) -> OllamaModelBuilder {
        OllamaModelBuilder {
            server_address: server_address.into(),
            model: model.into(),
            options: Options::builder(),
            think: None,
        }
    }
}

/// Builder for [`OllamaModel`] with generation settings.
pub struct OllamaModelBuilder {
    server_address: String,
    model: String,
    options: ollama_rs::types::common::OptionsBuilder,
    think: Option<Think>,
}

impl OllamaModelBuilder {
    /// Sets the sampling temperature. Higher values produce more random output.
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.options = self.options.temperature(temperature);
        self
    }

    /// Sets the random seed for reproducible outputs.
    pub fn seed(mut self, seed: u64) -> Self {
        self.options = self.options.seed(seed);
        self
    }

    /// Limits the next-token selection to the K most likely tokens.
    pub fn top_k(mut self, top_k: u32) -> Self {
        self.options = self.options.top_k(top_k);
        self
    }

    /// Sets the nucleus sampling probability threshold.
    pub fn top_p(mut self, top_p: f32) -> Self {
        self.options = self.options.top_p(top_p);
        self
    }

    /// Sets the context window size in tokens.
    pub fn num_ctx(mut self, num_ctx: u32) -> Self {
        self.options = self.options.num_ctx(num_ctx);
        self
    }

    /// Sets the maximum number of tokens to generate.
    pub fn num_predict(mut self, num_predict: u32) -> Self {
        self.options = self.options.num_predict(num_predict);
        self
    }

    /// Sets one or more stop sequences that halt generation when produced.
    pub fn stop(mut self, stop: Stop) -> Self {
        self.options = self.options.stop(stop);
        self
    }

    /// Configures extended-thinking (reasoning) mode for supported models.
    ///
    /// Accepts either a boolean toggle ([`Think::Bool`]) or a named
    /// intensity level ([`Think::Level`]). Models that do not support
    /// thinking ignore this setting.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use agent_rig::models::ollama::{OllamaModel, Think, ThinkLevel};
    ///
    /// let model = OllamaModel::builder("http://localhost:11434", "qwen3:8b")
    ///     .think(Think::Level(ThinkLevel::High))
    ///     .build();
    /// ```
    pub fn think(mut self, think: Think) -> Self {
        self.think = Some(think);
        self
    }

    /// Builds the [`OllamaModel`].
    pub fn build(self) -> OllamaModel {
        OllamaModel {
            client: OllamaClient::new(self.server_address),
            model: self.model,
            options: Some(self.options.build()),
            think: self.think,
        }
    }
}

/// Translates a [`ToolDefinition`] into an Ollama [`OllamaTool`].
impl From<&ToolDefinition> for OllamaTool {
    fn from(def: &ToolDefinition) -> OllamaTool {
        OllamaTool {
            tool_type: ToolType::Function,
            function: Function {
                name: def.name.clone(),
                description: def.description.clone(),
                parameters: def.parameters.clone().into(),
            },
        }
    }
}

/// Converts an Ollama tool call into the crate's [`ToolCall`] type.
///
/// Ollama has no call ID; the function name is used as a stable identifier
/// (sufficient since we don't need to echo an ID back to Ollama).
impl From<&OllamaToolCall> for ToolCall {
    fn from(tc: &OllamaToolCall) -> ToolCall {
        ToolCall {
            id: tc.function.name.clone(),
            name: tc.function.name.clone(),
            args: tc.function.arguments.clone(),
            provider_metadata: None,
        }
    }
}

/// Extracts token usage from the final Ollama [`ChatResponse`] chunk.
///
/// Returns `None` when neither `prompt_eval_count` nor `eval_count` is
/// set, so consumers see "the provider didn't report usage" rather
/// than a [`TokenUsage`] of all-`None`. The Ollama API uses `u64` for
/// counts; we cast to `u32` saturating because per-call token counts
/// above `u32::MAX` (~4.2B) are implausible.
///
/// A `From<&ChatResponse> for Option<TokenUsage>` impl would be more
/// idiomatic but the orphan rule rejects it — `&ChatResponse` is
/// foreign and appears before the local `TokenUsage` in
/// `Option<TokenUsage>`, so no local type is "uncovered" first.
fn to_token_usage(response: &ChatResponse) -> Option<TokenUsage> {
    if response.prompt_eval_count.is_none() && response.eval_count.is_none() {
        return None;
    }
    Some(TokenUsage {
        input_tokens: response.prompt_eval_count.map(saturating_u64_to_u32),
        output_tokens: response.eval_count.map(saturating_u64_to_u32),
        cached_input_tokens: None,
        thinking_tokens: None,
        tool_use_prompt_tokens: None,
    })
}

fn saturating_u64_to_u32(v: u64) -> u32 {
    u32::try_from(v).unwrap_or(u32::MAX)
}

/// Builds an Ollama [`ChatRequest`] from a [`ModelRequest`].
fn build_chat_request(
    model: &str,
    options: Option<Options>,
    think: Option<Think>,
    request: ModelRequest,
) -> Result<ChatRequest, Error> {
    let mut messages: Vec<OllamaMessage> = Vec::new();

    if let Some(system) = request.system {
        messages.push(OllamaMessage::system(system));
    }

    for msg in request.messages {
        let ollama_msg = match &msg.content {
            MessageContent::Text(text) => match msg.role {
                Role::User => OllamaMessage::user(text),
                Role::Assistant => OllamaMessage {
                    content: text.clone(),
                    role: OllamaRole::Assistant,
                    thinking: None,
                    tool_calls: vec![],
                },
            },
            MessageContent::ToolCalls(calls) => OllamaMessage {
                content: String::new(),
                role: OllamaRole::Assistant,
                thinking: None,
                tool_calls: calls
                    .iter()
                    .enumerate()
                    .map(|(i, call)| OllamaToolCall {
                        function: ToolCallFunction {
                            name: call.name.clone(),
                            arguments: call.args.clone(),
                            index: i,
                        },
                    })
                    .collect(),
            },
            MessageContent::ToolResult { result, .. } => {
                OllamaMessage::tool_response(&result).map_err(|e| Error::Provider(e.to_string()))?
            }
        };
        messages.push(ollama_msg);
    }

    let mut builder = ChatRequest::builder(model).messages(messages);

    if let Some(opts) = options {
        builder = builder.options(opts);
    }

    if let Some(think) = think {
        builder = builder.think(think);
    }

    if !request.tools.is_empty() {
        let ollama_tools: Vec<OllamaTool> = request.tools.iter().map(OllamaTool::from).collect();
        // Ollama requires streaming to be disabled when using tools.
        builder = builder.tools(ollama_tools).stream(false);
    }

    if let Some(schema) = request.output_schema {
        builder = builder.format(schema.to_value());
    }

    Ok(builder.build())
}

#[async_trait]
impl LlmModel for OllamaModel {
    async fn generate(&self, request: ModelRequest) -> Result<ModelResponse, Error> {
        let chat_request = build_chat_request(
            &self.model,
            self.options.clone(),
            self.think.clone(),
            request,
        )?;
        let mut stream = self.client.chat(chat_request);
        let mut output = String::new();
        let mut tool_calls: Vec<ToolCall> = Vec::new();
        let mut token_usage: Option<TokenUsage> = None;

        while let Some(chunk) = stream.next().await {
            let response = chunk.map_err(|e| Error::Provider(e.to_string()))?;

            if !response.message.tool_calls.is_empty() {
                tool_calls = response.message.tool_calls.iter().map(From::from).collect();
            }

            output.push_str(&response.message.content);

            if response.done {
                token_usage = to_token_usage(&response);
                break;
            }
        }

        if !tool_calls.is_empty() {
            return Ok(ModelResponse {
                text: None,
                tool_calls,
                thinking: None,
                token_usage,
            });
        }

        Ok(ModelResponse {
            text: Some(output),
            tool_calls: vec![],
            thinking: None,
            token_usage,
        })
    }

    /// Streams the Ollama response as [`ModelStreamChunk`] values.
    ///
    /// When the request has no tools, text chunks are emitted as
    /// [`ModelStreamChunk::TextDelta`] as they arrive from the server.
    /// When the request includes tools, Ollama requires non-streaming mode;
    /// tool calls are collected from the single response and emitted as
    /// [`ModelStreamChunk::ToolCall`] chunks.
    fn generate_stream(
        &self,
        request: ModelRequest,
    ) -> Pin<Box<dyn Stream<Item = Result<ModelStreamChunk, Error>> + Send + '_>> {
        Box::pin(async_stream::stream! {
            let chat_request = match build_chat_request(
                &self.model, self.options.clone(), self.think.clone(), request) {
                    Ok(chat_request) => chat_request,
                    Err(e) => {
                        yield Err(Error::Provider(e.to_string()));
                        return;
                    }
                };
            let mut stream = self.client.chat(chat_request);

            while let Some(chunk) = stream.next().await {
                let response = match chunk {
                    Ok(response) => response,
                    Err(e) => {
                        yield Err(Error::Provider(e.to_string()));
                        return;
                    }
                };

                // Tool calls come as a batch in non-streaming mode.
                for tc in &response.message.tool_calls {
                    yield Ok(ModelStreamChunk::ToolCall(ToolCall::from(tc)));
                }

                // Emit text content as a delta (empty strings are skipped).
                if !response.message.content.is_empty() {
                    yield Ok(ModelStreamChunk::TextDelta(response.message.content.clone()));
                }

                if response.done {
                    if let Some(usage) = to_token_usage(&response) {
                        yield Ok(ModelStreamChunk::Usage(usage));
                    }
                    break;
                }
            }
        })
    }
}

#[cfg(test)]
mod tests;
