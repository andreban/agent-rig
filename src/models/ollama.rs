use async_trait::async_trait;
use futures_util::StreamExt;
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
    model::{LlmModel, MessageContent, ModelRequest, ModelResponse, Role, ToolCall},
    tool::ToolDefinition,
};

/// LLM provider backed by an [Ollama](https://ollama.com/) server.
///
/// Use [`OllamaModel::new`] for the simple case, or [`OllamaModel::builder`]
/// to configure generation settings such as temperature.
///
/// # Examples
///
/// ```no_run
/// use rust_agent_kit::models::ollama::OllamaModel;
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

#[async_trait]
impl LlmModel for OllamaModel {
    async fn generate(&self, request: ModelRequest) -> Result<ModelResponse, Error> {
        let mut messages: Vec<OllamaMessage> = Vec::new();

        if let Some(system) = &request.system {
            messages.push(OllamaMessage::system(system.clone()));
        }

        for msg in &request.messages {
            let ollama_msg = match &msg.content {
                MessageContent::Text(text) => match msg.role {
                    Role::User => OllamaMessage::user(text.clone()),
                    Role::Assistant => OllamaMessage {
                        content: text.clone(),
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
                        .collect::<Vec<_>>(),
                },
                MessageContent::ToolResult { result, .. } => {
                    OllamaMessage::tool_response(result)
                        .map_err(|e| Error::Provider(e.to_string()))?
                }
            };
            messages.push(ollama_msg);
        }

        let mut builder = ChatRequest::builder(&self.model).messages(messages);

        if let Some(options) = self.options.clone() {
            builder = builder.options(options);
        }

        // Attach tools when the request includes them.
        if !request.tools.is_empty() {
            let ollama_tools: Vec<OllamaTool> =
                request.tools.iter().map(to_ollama_tool).collect();
            builder = builder.tools(ollama_tools).stream(false);
        }

        if let Some(schema) = request.output_schema {
            builder = builder.format(schema);
        }

        let chat_request = builder.build();

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
                    .map(|tc| ToolCall {
                        // Ollama has no call ID; use the function name as a
                        // stable identifier (sufficient since we don't need to
                        // echo an ID back to Ollama).
                        id: tc.function.name.clone(),
                        name: tc.function.name.clone(),
                        args: tc.function.arguments.clone(),
                        provider_metadata: None,
                    })
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
            });
        }

        Ok(ModelResponse {
            text: Some(output),
            tool_calls: vec![],
        })
    }
}
