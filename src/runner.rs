use crate::{
    agent::Agent,
    error::Error,
    model::{Message, ModelRequest},
};

/// The result of a completed agent run.
#[derive(Debug, Clone)]
pub struct AgentResult {
    /// The final text output produced by the agent.
    pub output: String,
}

/// Executes [`Agent`]s, translating a run into [`LlmModel::generate`] calls.
///
/// [`LlmModel::generate`]: crate::model::LlmModel::generate
///
/// # Examples
///
/// ```no_run
/// use rust_agent_kit::{Agent, AgentRunner};
/// use rust_agent_kit::models::gemini::GeminiModel;
///
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let agent = Agent::builder()
///     .name("Assistant")
///     .instructions("You are a helpful assistant.")
///     .model(Box::new(GeminiModel::new("API_KEY", "gemini-2.5-pro-preview-03-25")))
///     .build();
///
/// let result = AgentRunner::new().run(&agent, "Hello!").await?;
/// println!("{}", result.output);
/// # Ok(())
/// # }
/// ```
pub struct AgentRunner;

impl AgentRunner {
    /// Creates a new `AgentRunner`.
    pub fn new() -> Self {
        AgentRunner
    }

    /// Runs the given agent with the provided user input and returns the result.
    pub async fn run(&self, agent: &Agent, input: &str) -> Result<AgentResult, Error> {
        let request = ModelRequest {
            messages: vec![Message::user(input)],
            system: Some(agent.instructions().to_string()),
        };

        let response = agent.model.generate(request).await?;

        Ok(AgentResult {
            output: response.text,
        })
    }
}

impl Default for AgentRunner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        error::Error,
        model::{LlmModel, ModelRequest, ModelResponse},
    };
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

    #[tokio::test]
    async fn run_returns_model_output() {
        let agent = Agent::builder()
            .name("Test")
            .instructions("Be helpful.")
            .model(Box::new(EchoModel))
            .build();

        let result = AgentRunner::new().run(&agent, "hello").await.unwrap();
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
                *self.captured.lock().unwrap() = Some(request);
                Ok(ModelResponse { text: String::new() })
            }
        }

        let model = std::sync::Arc::new(CaptureModel {
            captured: std::sync::Mutex::new(None),
        });

        struct ArcModel(std::sync::Arc<CaptureModel>);

        #[async_trait]
        impl LlmModel for ArcModel {
            async fn generate(&self, request: ModelRequest) -> Result<ModelResponse, Error> {
                self.0.generate(request).await
            }
        }

        let captured = model.clone();
        let agent = Agent::builder()
            .name("Test")
            .instructions("System prompt.")
            .model(Box::new(ArcModel(model)))
            .build();

        AgentRunner::new().run(&agent, "input").await.unwrap();

        let req = captured.captured.lock().unwrap().clone().unwrap();
        assert_eq!(req.system.as_deref(), Some("System prompt."));
    }
}
