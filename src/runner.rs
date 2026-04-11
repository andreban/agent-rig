use crate::{
    agent::Agent,
    error::Error,
    model::{LlmModel, Message, ModelRequest},
};

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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ModelRequest, ModelResponse};
    use async_trait::async_trait;

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
