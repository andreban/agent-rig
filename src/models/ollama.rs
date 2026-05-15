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
        ChatRequest, Function, Message as OllamaMessage, Role as OllamaRole, Tool as OllamaTool,
        ToolCall as OllamaToolCall, ToolCallFunction, ToolType,
    },
    types::common::{Options, Stop},
};

use crate::{
    error::Error,
    model::{
        LlmModel, MessageContent, ModelRequest, ModelResponse, ModelStreamChunk, Role, ToolCall,
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
        }
    }
}

/// Builder for [`OllamaModel`] with generation settings.
pub struct OllamaModelBuilder {
    server_address: String,
    model: String,
    options: ollama_rs::types::common::OptionsBuilder,
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

    /// Builds the [`OllamaModel`].
    pub fn build(self) -> OllamaModel {
        OllamaModel {
            client: OllamaClient::new(self.server_address),
            model: self.model,
            options: Some(self.options.build()),
        }
    }
}

/// Translates a [`ToolDefinition`] into an Ollama [`OllamaTool`].
fn to_ollama_tool(def: &ToolDefinition) -> OllamaTool {
    OllamaTool {
        tool_type: ToolType::Function,
        function: Function {
            name: def.name.clone(),
            description: def.description.clone(),
            parameters: def.parameters.clone(),
        },
    }
}

/// Converts an Ollama tool call into the crate's [`ToolCall`] type.
///
/// Ollama has no call ID; the function name is used as a stable identifier
/// (sufficient since we don't need to echo an ID back to Ollama).
fn to_tool_call(tc: &OllamaToolCall) -> ToolCall {
    ToolCall {
        id: tc.function.name.clone(),
        name: tc.function.name.clone(),
        args: tc.function.arguments.clone(),
        provider_metadata: None,
    }
}

/// Builds an Ollama [`ChatRequest`] from a [`ModelRequest`].
fn build_chat_request(
    model: &str,
    options: Option<Options>,
    request: ModelRequest,
) -> Result<ChatRequest, Error> {
    let mut messages: Vec<OllamaMessage> = Vec::new();

    if let Some(system) = request.system {
        messages.push(OllamaMessage::system(system));
    }

    for msg in request.messages {
        let ollama_msg = match msg.content {
            MessageContent::Text(text) => match msg.role {
                Role::User => OllamaMessage::user(text),
                Role::Assistant => OllamaMessage {
                    content: text,
                    role: OllamaRole::Assistant,
                    tool_calls: vec![],
                },
            },
            MessageContent::ToolCalls(calls) => OllamaMessage {
                content: String::new(),
                role: OllamaRole::Assistant,
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

    if !request.tools.is_empty() {
        let ollama_tools: Vec<OllamaTool> = request.tools.iter().map(to_ollama_tool).collect();
        // Ollama requires streaming to be disabled when using tools.
        builder = builder.tools(ollama_tools).stream(false);
    }

    if let Some(schema) = request.output_schema {
        builder = builder.format(schema);
    }

    Ok(builder.build())
}

#[async_trait]
impl LlmModel for OllamaModel {
    async fn generate(&self, request: ModelRequest) -> Result<ModelResponse, Error> {
        let chat_request = build_chat_request(&self.model, self.options.clone(), request)?;
        let mut stream = self.client.chat(chat_request);
        let mut output = String::new();
        let mut tool_calls: Vec<ToolCall> = Vec::new();

        while let Some(chunk) = stream.next().await {
            let response = chunk.map_err(|e| Error::Provider(e.to_string()))?;

            if !response.message.tool_calls.is_empty() {
                tool_calls = response
                    .message
                    .tool_calls
                    .iter()
                    .map(to_tool_call)
                    .collect();
            }

            output.push_str(&response.message.content);

            if response.done {
                break;
            }
        }

        if !tool_calls.is_empty() {
            return Ok(ModelResponse {
                text: None,
                tool_calls,
                thinking: None,
            });
        }

        Ok(ModelResponse {
            text: Some(output),
            tool_calls: vec![],
            thinking: None,
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
            let chat_request = build_chat_request(&self.model, self.options.clone(), request)?;
            let mut stream = self.client.chat(chat_request);

            while let Some(chunk) = stream.next().await {
                let response = chunk.map_err(|e| Error::Provider(e.to_string()))?;

                // Tool calls come as a batch in non-streaming mode.
                for tc in &response.message.tool_calls {
                    yield Ok(ModelStreamChunk::ToolCall(to_tool_call(tc)));
                }

                // Emit text content as a delta (empty strings are skipped).
                if !response.message.content.is_empty() {
                    yield Ok(ModelStreamChunk::TextDelta(response.message.content.clone()));
                }

                if response.done {
                    break;
                }
            }
        })
    }
}
