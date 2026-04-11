use crate::model::LlmModel;

/// An agent comprising a name, system instructions, and an LLM model.
///
/// # Examples
///
/// ```no_run
/// use rust_agent_kit::Agent;
/// use rust_agent_kit::models::gemini::GeminiModel;
///
/// let agent = Agent::builder()
///     .name("Summariser")
///     .instructions("Summarise the provided text in one sentence.")
///     .model(Box::new(GeminiModel::new("API_KEY", "gemini-2.5-pro-preview-03-25")))
///     .build();
/// ```
pub struct Agent {
    pub(crate) name: String,
    pub(crate) instructions: String,
    pub(crate) model: Box<dyn LlmModel>,
}

impl Agent {
    /// Returns a builder for constructing an [`Agent`].
    pub fn builder() -> AgentBuilder {
        AgentBuilder::default()
    }

    /// The agent's display name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The system instructions passed to the model on every run.
    pub fn instructions(&self) -> &str {
        &self.instructions
    }
}

/// Builder for [`Agent`].
#[derive(Default)]
pub struct AgentBuilder {
    name: Option<String>,
    instructions: Option<String>,
    model: Option<Box<dyn LlmModel>>,
}

impl AgentBuilder {
    /// Sets the agent's display name.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Sets the system instructions for the agent.
    pub fn instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    /// Sets the LLM model the agent will use.
    pub fn model(mut self, model: Box<dyn LlmModel>) -> Self {
        self.model = Some(model);
        self
    }

    /// Builds the [`Agent`].
    ///
    /// # Panics
    ///
    /// Panics if `name`, `instructions`, or `model` have not been set.
    pub fn build(self) -> Agent {
        Agent {
            name: self.name.expect("Agent::builder requires a name"),
            instructions: self.instructions.expect("Agent::builder requires instructions"),
            model: self.model.expect("Agent::builder requires a model"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        error::Error,
        model::{ModelRequest, ModelResponse},
    };
    use async_trait::async_trait;

    struct StubModel;

    #[async_trait]
    impl LlmModel for StubModel {
        async fn generate(&self, _request: ModelRequest) -> Result<ModelResponse, Error> {
            Ok(ModelResponse { text: String::new() })
        }
    }

    fn stub_model() -> Box<dyn LlmModel> {
        Box::new(StubModel)
    }

    #[test]
    fn builder_sets_fields() {
        let agent = Agent::builder()
            .name("Test Agent")
            .instructions("Do stuff.")
            .model(stub_model())
            .build();

        assert_eq!(agent.name(), "Test Agent");
        assert_eq!(agent.instructions(), "Do stuff.");
    }

    #[test]
    #[should_panic(expected = "requires a name")]
    fn builder_panics_without_name() {
        Agent::builder()
            .instructions("Do stuff.")
            .model(stub_model())
            .build();
    }

    #[test]
    #[should_panic(expected = "requires instructions")]
    fn builder_panics_without_instructions() {
        Agent::builder()
            .name("Test Agent")
            .model(stub_model())
            .build();
    }

    #[test]
    #[should_panic(expected = "requires a model")]
    fn builder_panics_without_model() {
        Agent::builder()
            .name("Test Agent")
            .instructions("Do stuff.")
            .build();
    }
}
