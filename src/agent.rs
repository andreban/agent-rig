// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

use schemars::Schema;
use serde::{Deserialize, Serialize};

/// A blueprint describing an agent's identity and behaviour.
///
/// `Agent` holds only static, serializable configuration: a name, system
/// instructions, an optional JSON Schema for structured output, and the names
/// of tools the agent is permitted to use. It carries no model or runtime
/// state. Pair it with an [`AgentRunner`] that owns the model and a
/// [`ToolRegistry`] that owns the implementations.
///
/// Because `Agent` derives [`Serialize`] and [`Deserialize`], configurations
/// can be saved to and loaded from JSON, YAML, or any other `serde`-compatible
/// format. Tool definitions are intentionally not serialized — they are
/// resolved from the [`ToolRegistry`] at runtime.
///
/// [`AgentRunner`]: crate::runner::AgentRunner
/// [`ToolRegistry`]: crate::tools::ToolRegistry
///
/// # Examples
///
/// ```
/// use agent_rig::Agent;
///
/// let agent = Agent::builder()
///     .name("Summariser")
///     .instructions("Summarise the provided text in one sentence.")
///     .build();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Agent {
    pub(crate) name: String,
    pub(crate) instructions: String,
    pub(crate) output_schema: Option<Schema>,
    /// Names of tools this agent may use. Each name must match a key in the
    /// [`ToolRegistry`] supplied to the runner.
    ///
    /// [`ToolRegistry`]: crate::tool::ToolRegistry
    #[serde(default)]
    pub(crate) tool_names: Vec<String>,
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
    pub fn output_schema(&self) -> Option<&Schema> {
        self.output_schema.as_ref()
    }

    /// The names of tools this agent is permitted to use.
    pub fn tool_names(&self) -> &[String] {
        &self.tool_names
    }
}

/// Builder for [`Agent`].
#[derive(Default)]
pub struct AgentBuilder {
    name: Option<String>,
    instructions: Option<String>,
    output_schema: Option<Schema>,
    tool_names: Vec<String>,
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
    pub fn output_schema<S: Into<Schema>>(mut self, schema: S) -> Self {
        self.output_schema = Some(schema.into());
        self
    }

    /// Declares that this agent may use the tool with the given name.
    ///
    /// The name must match a key registered in the [`ToolRegistry`] supplied
    /// to the runner. Call this method once per tool.
    ///
    /// [`ToolRegistry`]: crate::tools::ToolRegistry
    pub fn tool(mut self, name: impl Into<String>) -> Self {
        self.tool_names.push(name.into());
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
            tool_names: self.tool_names,
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
