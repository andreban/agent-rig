use std::sync::Arc;

use tracing::{debug, instrument};

use crate::{
    agent::Agent,
    error::Error,
    model::{LlmModel, Message, ModelRequest},
    tool::ToolRegistry,
};
use serde::de::DeserializeOwned;

/// The result of a completed agent run.
#[derive(Debug, Clone)]
pub struct AgentResult {
    /// The final text output produced by the agent.
    pub output: String,
}

/// Executes [`Agent`]s against an LLM model.
///
/// `AgentRunner` owns the model and holds a reference to a [`ToolRegistry`].
/// The same runner can execute multiple agents; a [`ToolRegistry`] can be
/// shared across multiple runners via [`Arc`].
///
/// # Examples
///
/// ```no_run
/// use rust_agent_kit::{Agent, AgentRunner};
/// use rust_agent_kit::models::gemini::GeminiModel;
///
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let model = GeminiModel::builder("API_KEY", "gemini-2.5-pro-preview-03-25")
///     .temperature(0.8)
///     .build();
///
/// let agent = Agent::builder()
///     .name("Assistant")
///     .instructions("You are a helpful assistant.")
///     .build();
///
/// let runner = AgentRunner::new(Box::new(model));
/// let result = runner.run(&agent, "Hello!").await?;
/// println!("{}", result.output);
/// # Ok(())
/// # }
/// ```
pub struct AgentRunner {
    model: Box<dyn LlmModel>,
    registry: Arc<ToolRegistry>,
}

impl AgentRunner {
    /// Creates a new `AgentRunner` powered by the given model with an empty
    /// tool registry. Suitable for agents that use no tools.
    pub fn new(model: Box<dyn LlmModel>) -> Self {
        AgentRunner {
            model,
            registry: ToolRegistry::empty(),
        }
    }

    /// Creates a new `AgentRunner` powered by the given model and sharing the
    /// given [`ToolRegistry`].
    ///
    /// Multiple runners can share the same registry by passing [`Arc::clone`]
    /// of the same instance.
    pub fn with_registry(model: Box<dyn LlmModel>, registry: Arc<ToolRegistry>) -> Self {
        AgentRunner { model, registry }
    }

    /// Runs the given agent with the provided user input and returns the result.
    ///
    /// If the agent declares tools, the runner validates that every tool name
    /// is present in its registry before making any network call, then executes
    /// the agentic loop: model call → tool execution → model call, repeating
    /// until the model produces a final text response.
    ///
    /// The output is returned as a raw string. Use [`run_typed`] when the agent
    /// produces structured JSON that should be deserialized into a concrete type.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Agent`] if a tool name declared by the agent has no
    /// registered implementation, or if the model loop ends without a text
    /// response.
    ///
    /// [`run_typed`]: AgentRunner::run_typed
    #[instrument(skip(self, input), fields(agent = agent.name()))]
    pub async fn run(&self, agent: &Agent, input: &str) -> Result<AgentResult, Error> {
        debug!("starting run");

        // Validate that every declared tool name is registered.
        for name in agent.tool_names() {
            if !self.registry.contains(name) {
                return Err(Error::Agent(format!(
                    "tool '{name}' is declared by the agent but not registered in the registry"
                )));
            }
        }

        // Resolve tool definitions from the registry.
        let tools: Vec<_> = agent
            .tool_names()
            .iter()
            .map(|name| self.registry.get(name).unwrap().definition())
            .collect();

        let mut messages = vec![Message::user(input)];
        let mut turn = 0u32;

        loop {
            turn += 1;
            debug!(turn, "calling model");

            let request = ModelRequest {
                messages: messages.clone(),
                system: Some(agent.instructions().to_string()),
                output_schema: agent.output_schema().cloned(),
                tools: tools.clone(),
            };

            let response = self.model.generate(request).await?;

            if response.tool_calls.is_empty() {
                // Final text response — the loop is complete.
                let text = response.text.ok_or_else(|| {
                    Error::Agent("model returned neither text nor tool calls".to_string())
                })?;
                debug!("run complete");
                return Ok(AgentResult { output: text });
            }

            debug!(count = response.tool_calls.len(), "model requested tool calls");

            // Append all tool calls as a single assistant turn.
            messages.push(Message::tool_calls(response.tool_calls.clone()));

            // Execute each tool and append one result message per call.
            for call in &response.tool_calls {
                debug!(tool = call.name, "executing tool");
                let tool = self.registry.get(&call.name).ok_or_else(|| {
                    Error::Agent(format!("model called unregistered tool '{}'", call.name))
                })?;
                let result = tool.call(call.args.clone()).await?;
                debug!(tool = call.name, "tool call complete");
                messages.push(Message::tool_result(
                    call.id.clone(),
                    call.name.clone(),
                    result,
                    call.provider_metadata.clone(),
                ));
            }
        }
    }

    /// Runs the given agent and deserializes the output into `T`.
    ///
    /// This is a typed convenience wrapper around [`run`]. Set `output_schema`
    /// on the agent via [`AgentBuilder::output_schema`] to constrain the model
    /// to produce JSON that matches `T`; this method handles the deserialization.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Agent`] if the model response cannot be deserialized
    /// into `T`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rust_agent_kit::{Agent, AgentRunner};
    /// use rust_agent_kit::models::gemini::GeminiModel;
    /// use schemars::JsonSchema;
    /// use serde::Deserialize;
    ///
    /// #[derive(Debug, Deserialize, JsonSchema)]
    /// struct Summary { headline: String, body: String }
    ///
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let model = GeminiModel::builder("API_KEY", "gemini-2.5-flash").build();
    /// let schema = schemars::schema_for!(Summary);
    /// let agent = Agent::builder()
    ///     .name("Summariser")
    ///     .instructions("Summarise the text.")
    ///     .output_schema(schema)
    ///     .build();
    ///
    /// let summary: Summary = AgentRunner::new(Box::new(model))
    ///     .run_typed(&agent, "Long article text…")
    ///     .await?;
    /// println!("{}", summary.headline);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// [`run`]: AgentRunner::run
    /// [`AgentBuilder::output_schema`]: crate::AgentBuilder::output_schema
    pub async fn run_typed<T>(&self, agent: &Agent, input: &str) -> Result<T, Error>
    where
        T: DeserializeOwned,
    {
        let result = self.run(agent, input).await?;
        serde_json::from_str::<T>(&result.output)
            .map_err(|e| Error::Agent(format!("failed to deserialize structured output: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ModelRequest, ModelResponse};
    use async_trait::async_trait;
    use serde::Deserialize;

    struct EchoModel;

    #[async_trait]
    impl LlmModel for EchoModel {
        async fn generate(&self, request: ModelRequest) -> Result<ModelResponse, Error> {
            let echo = request.messages.last().and_then(|m| {
                if let crate::model::MessageContent::Text(t) = &m.content {
                    Some(t.clone())
                } else {
                    None
                }
            });
            Ok(ModelResponse { text: echo, tool_calls: vec![] })
        }
    }

    fn echo_runner() -> AgentRunner {
        AgentRunner::new(Box::new(EchoModel))
    }

    #[tokio::test]
    async fn run_returns_model_output() {
        let agent = Agent::builder()
            .name("Test")
            .instructions("Be helpful.")
            .build();

        let result = echo_runner().run(&agent, "hello").await.unwrap();
        assert_eq!(result.output, "hello");
    }

    #[derive(Debug, Deserialize, PartialEq)]
    struct Point {
        x: i32,
        y: i32,
    }

    struct JsonModel(String);

    #[async_trait]
    impl LlmModel for JsonModel {
        async fn generate(&self, _request: ModelRequest) -> Result<ModelResponse, Error> {
            Ok(ModelResponse { text: Some(self.0.clone()), tool_calls: vec![] })
        }
    }

    #[tokio::test]
    async fn run_typed_deserializes_response() {
        let runner = AgentRunner::new(Box::new(JsonModel(r#"{"x":1,"y":2}"#.to_string())));
        let agent = Agent::builder()
            .name("Test")
            .instructions("Return a point.")
            .build();

        let point: Point = runner.run_typed(&agent, "give me a point").await.unwrap();
        assert_eq!(point, Point { x: 1, y: 2 });
    }

    #[tokio::test]
    async fn run_typed_returns_error_on_bad_json() {
        let runner = AgentRunner::new(Box::new(JsonModel("not json".to_string())));
        let agent = Agent::builder()
            .name("Test")
            .instructions("Return a point.")
            .build();

        let result: Result<Point, _> = runner.run_typed(&agent, "give me a point").await;
        assert!(matches!(result, Err(Error::Agent(_))));
    }

    #[tokio::test]
    async fn run_errors_when_tool_not_in_registry() {
        let agent = Agent::builder()
            .name("Test")
            .instructions("Use tools.")
            .tool("missing_tool")
            .build();

        let result = AgentRunner::new(Box::new(EchoModel)).run(&agent, "go").await;
        assert!(matches!(result, Err(Error::Agent(_))));
    }

    #[tokio::test]
    async fn run_executes_tool_loop_and_returns_text() {
        use crate::model::{MessageContent, ToolCall as ModelToolCall};
        use crate::tool::{Tool, ToolDefinition, ToolRegistry};
        use serde_json::json;
        use std::sync::atomic::{AtomicU32, Ordering};

        // Model: first call returns a tool call, second returns text.
        static CALL_COUNT: AtomicU32 = AtomicU32::new(0);

        struct ToolLoopModel;

        #[async_trait]
        impl LlmModel for ToolLoopModel {
            async fn generate(&self, request: ModelRequest) -> Result<ModelResponse, Error> {
                let n = CALL_COUNT.fetch_add(1, Ordering::SeqCst);
                if n == 0 {
                    Ok(ModelResponse {
                        text: None,
                        tool_calls: vec![ModelToolCall {
                            id: "call-1".to_string(),
                            name: "add".to_string(),
                            args: json!({"a": 1, "b": 2}),
                            provider_metadata: None,
                        }],
                    })
                } else {
                    // The last message should be the tool result.
                    let last = request.messages.last().unwrap();
                    let answer = if let MessageContent::ToolResult { result, .. } = &last.content {
                        result["sum"].as_i64().unwrap_or(0).to_string()
                    } else {
                        "no result".to_string()
                    };
                    Ok(ModelResponse { text: Some(answer), tool_calls: vec![] })
                }
            }
        }

        struct AddTool;

        #[async_trait]
        impl Tool for AddTool {
            fn definition(&self) -> ToolDefinition {
                ToolDefinition {
                    name: "add".to_string(),
                    description: "Adds two numbers.".to_string(),
                    parameters: json!({"type": "object"}),
                }
            }
            async fn call(&self, args: serde_json::Value) -> Result<serde_json::Value, Error> {
                let a = args["a"].as_i64().unwrap_or(0);
                let b = args["b"].as_i64().unwrap_or(0);
                Ok(json!({"sum": a + b}))
            }
        }

        CALL_COUNT.store(0, Ordering::SeqCst);
        let registry = Arc::new(ToolRegistry::new().register(Box::new(AddTool)));
        let runner = AgentRunner::with_registry(Box::new(ToolLoopModel), registry);
        let agent = Agent::builder()
            .name("Test")
            .instructions("Use tools.")
            .tool("add")
            .build();

        let result = runner.run(&agent, "add 1 and 2").await.unwrap();
        assert_eq!(result.output, "3");
    }

    #[tokio::test]
    async fn run_passes_system_instructions() {
        struct CaptureModel {
            captured: std::sync::Mutex<Option<ModelRequest>>,
        }

        #[async_trait]
        impl LlmModel for CaptureModel {
            async fn generate(&self, request: ModelRequest) -> Result<ModelResponse, Error> {
                *self.captured.lock().unwrap() = Some(request.clone());
                Ok(ModelResponse { text: Some(String::new()), tool_calls: vec![] })
            }
        }

        let capture = std::sync::Arc::new(CaptureModel {
            captured: std::sync::Mutex::new(None),
        });

        struct ArcModel(std::sync::Arc<CaptureModel>);

        #[async_trait]
        impl LlmModel for ArcModel {
            async fn generate(&self, request: ModelRequest) -> Result<ModelResponse, Error> {
                self.0.generate(request).await
            }
        }

        let runner = AgentRunner::new(Box::new(ArcModel(capture.clone())));
        let agent = Agent::builder()
            .name("Test")
            .instructions("System prompt.")
            .build();

        runner.run(&agent, "input").await.unwrap();

        let req = capture.captured.lock().unwrap().clone().unwrap();
        assert_eq!(req.system.as_deref(), Some("System prompt."));
    }
}
