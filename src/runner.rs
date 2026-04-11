use crate::{
    agent::Agent,
    error::Error,
    model::{LlmModel, Message, ModelRequest},
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
/// `AgentRunner` owns the model and acts as the execution engine. The same
/// runner can be used to run multiple agents, or the same agent multiple times.
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
}

impl AgentRunner {
    /// Creates a new `AgentRunner` powered by the given model.
    pub fn new(model: Box<dyn LlmModel>) -> Self {
        AgentRunner { model }
    }

    /// Runs the given agent with the provided user input and returns the result.
    ///
    /// The output is returned as a raw string. Use [`run_typed`] when the agent
    /// produces structured JSON that should be deserialized into a concrete type.
    ///
    /// [`run_typed`]: AgentRunner::run_typed
    pub async fn run(&self, agent: &Agent, input: &str) -> Result<AgentResult, Error> {
        let request = ModelRequest {
            messages: vec![Message::user(input)],
            system: Some(agent.instructions().to_string()),
            output_schema: agent.output_schema().cloned(),
        };

        let response = self.model.generate(request).await?;

        Ok(AgentResult {
            output: response.text,
        })
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
            let echo = request
                .messages
                .last()
                .map(|m| m.content.clone())
                .unwrap_or_default();
            Ok(ModelResponse { text: echo })
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
            Ok(ModelResponse { text: self.0.clone() })
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
    async fn run_passes_system_instructions() {
        struct CaptureModel {
            captured: std::sync::Mutex<Option<ModelRequest>>,
        }

        #[async_trait]
        impl LlmModel for CaptureModel {
            async fn generate(&self, request: ModelRequest) -> Result<ModelResponse, Error> {
                *self.captured.lock().unwrap() = Some(request.clone());
                Ok(ModelResponse { text: String::new() })
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
