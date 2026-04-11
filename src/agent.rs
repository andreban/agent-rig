use serde_json::Value;

/// A blueprint describing an agent's identity and behaviour.
///
/// `Agent` holds only static, serializable configuration: a name, system
/// instructions, and an optional JSON Schema for structured output. It carries
/// no model or runtime state. Pair it with an [`AgentRunner`] that owns the
/// model to execute it.
///
/// [`AgentRunner`]: crate::AgentRunner
///
/// # Examples
///
/// ```
/// use rust_agent_kit::Agent;
///
/// let agent = Agent::builder()
///     .name("Summariser")
///     .instructions("Summarise the provided text in one sentence.")
///     .build();
/// ```
#[derive(Debug, Clone)]
pub struct Agent {
    pub(crate) name: String,
    pub(crate) instructions: String,
    pub(crate) output_schema: Option<serde_json::Value>,
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

    /// The JSON Schema the agent's output must conform to, if any.
    pub fn output_schema(&self) -> Option<&serde_json::Value> {
        self.output_schema.as_ref()
    }
}

/// Builder for [`Agent`].
#[derive(Default)]
pub struct AgentBuilder {
    name: Option<String>,
    instructions: Option<String>,
    output_schema: Option<serde_json::Value>,
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

    /// Constrains the agent's output to the given JSON Schema.
    ///
    /// When set, the runner forwards the schema to the model provider so that
    /// responses conform to the schema. Providers that do not support structured
    /// output ignore this field.
    pub fn output_schema<S: Into<Value>>(mut self, schema: S) -> Self {
        self.output_schema = Some(schema.into());
        self
    }

    /// Builds the [`Agent`].
    ///
    /// # Panics
    ///
    /// Panics if `name` or `instructions` have not been set.
    pub fn build(self) -> Agent {
        Agent {
            name: self.name.expect("Agent::builder requires a name"),
            instructions: self
                .instructions
                .expect("Agent::builder requires instructions"),
            output_schema: self.output_schema,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_sets_fields() {
        let agent = Agent::builder()
            .name("Test Agent")
            .instructions("Do stuff.")
            .build();

        assert_eq!(agent.name(), "Test Agent");
        assert_eq!(agent.instructions(), "Do stuff.");
    }

    #[test]
    #[should_panic(expected = "requires a name")]
    fn builder_panics_without_name() {
        Agent::builder().instructions("Do stuff.").build();
    }

    #[test]
    #[should_panic(expected = "requires instructions")]
    fn builder_panics_without_instructions() {
        Agent::builder().name("Test Agent").build();
    }
}
