use async_trait::async_trait;
use tracing::{debug, instrument};

use crate::{
    agent::Agent,
    error::Error,
    runner::AgentRunner,
    tool::{Tool, ToolDefinition},
};

/// A [`Tool`] implementation that delegates to a child [`Agent`].
///
/// `AgentTool` wraps an [`AgentRunner`] and an [`Agent`] so that a parent
/// agent can invoke a child agent as a regular tool call. The parent model
/// sees the [`ToolDefinition`] supplied at construction time; when it calls
/// the tool, the child agent runs its own full agentic loop (including its
/// own tools) and the output is returned to the parent.
///
/// # Input convention
///
/// The `args` JSON object from the parent model is serialized to a JSON
/// string and passed as the child agent's input. The child agent's
/// instructions should describe how to interpret it.
///
/// # Output convention
///
/// The child's final text output is returned as `{ "output": "<text>" }`.
///
/// # Examples
///
/// ```no_run
/// use std::sync::Arc;
/// use rust_agent_kit::{Agent, AgentRunner};
/// use rust_agent_kit::agent_tool::AgentTool;
/// use rust_agent_kit::tool::{ToolDefinition, ToolRegistry};
/// use rust_agent_kit::models::gemini::GeminiModel;
/// use serde_json::json;
///
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let child_model = GeminiModel::builder("API_KEY", "gemini-2.5-flash").build();
/// let child_agent = Agent::builder()
///     .name("Summariser")
///     .instructions("Summarise the text in the 'text' field of the JSON input.")
///     .build();
/// let child_runner = AgentRunner::new(Box::new(child_model));
///
/// let summarise_tool = AgentTool::new(
///     ToolDefinition {
///         name: "summarise".to_string(),
///         description: "Summarises a long piece of text.".to_string(),
///         parameters: json!({
///             "type": "object",
///             "properties": { "text": { "type": "string" } },
///             "required": ["text"]
///         }),
///     },
///     child_agent,
///     child_runner,
/// );
///
/// let registry = Arc::new(ToolRegistry::new().register(Box::new(summarise_tool)));
/// # Ok(())
/// # }
/// ```
pub struct AgentTool {
    definition: ToolDefinition,
    agent: Agent,
    runner: AgentRunner,
}

impl AgentTool {
    /// Creates a new `AgentTool` from a [`ToolDefinition`], an [`Agent`], and
    /// an [`AgentRunner`].
    ///
    /// The `definition` controls how the parent model sees this tool (name,
    /// description, parameters). The `agent` and `runner` are the child
    /// agent's blueprint and execution engine respectively.
    pub fn new(definition: ToolDefinition, agent: Agent, runner: AgentRunner) -> Self {
        Self { definition, agent, runner }
    }
}

#[async_trait]
impl Tool for AgentTool {
    fn definition(&self) -> ToolDefinition {
        self.definition.clone()
    }

    /// Runs the child agent with the serialized `args` as input.
    ///
    /// `args` is serialized to a JSON string and passed to the child runner.
    /// Returns `{ "output": "<child output>" }` on success.
    #[instrument(skip(self, args), fields(tool = self.definition.name))]
    async fn call(&self, args: serde_json::Value) -> Result<serde_json::Value, Error> {
        debug!("delegating to child agent");
        let input = serde_json::to_string(&args)
            .map_err(|e| Error::Agent(format!("failed to serialize args: {e}")))?;
        let result = self.runner.run(&self.agent, &input).await?;
        debug!("child agent complete");
        Ok(serde_json::json!({ "output": result.output }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{LlmModel, ModelRequest, ModelResponse};
    use async_trait::async_trait;
    use serde_json::json;

    /// A stub model that echoes the last user message text as its response.
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

    /// A stub model that always returns an error.
    struct ErrorModel;

    #[async_trait]
    impl LlmModel for ErrorModel {
        async fn generate(&self, _request: ModelRequest) -> Result<ModelResponse, Error> {
            Err(Error::Provider("provider failure".to_string()))
        }
    }

    fn make_tool(runner: AgentRunner) -> AgentTool {
        AgentTool::new(
            ToolDefinition {
                name: "child".to_string(),
                description: "A child agent.".to_string(),
                parameters: json!({ "type": "object" }),
            },
            Agent::builder()
                .name("Child")
                .instructions("Process the input.")
                .build(),
            runner,
        )
    }

    #[test]
    fn definition_returns_supplied_definition() {
        let tool = make_tool(AgentRunner::new(Box::new(EchoModel)));
        let def = tool.definition();
        assert_eq!(def.name, "child");
        assert_eq!(def.description, "A child agent.");
    }

    #[tokio::test]
    async fn call_passes_serialized_args_as_input() {
        // EchoModel echoes the input back; we check the output wraps it correctly.
        let tool = make_tool(AgentRunner::new(Box::new(EchoModel)));
        let args = json!({ "text": "hello" });
        let result = tool.call(args.clone()).await.unwrap();

        let expected_input = serde_json::to_string(&args).unwrap();
        assert_eq!(result, json!({ "output": expected_input }));
    }

    #[tokio::test]
    async fn call_wraps_output_in_output_field() {
        let tool = make_tool(AgentRunner::new(Box::new(EchoModel)));
        let result = tool.call(json!({ "x": 1 })).await.unwrap();
        assert!(result.get("output").is_some());
    }

    #[tokio::test]
    async fn call_propagates_child_error() {
        let tool = make_tool(AgentRunner::new(Box::new(ErrorModel)));
        let err = tool.call(json!({})).await.unwrap_err();
        assert!(matches!(err, Error::Provider(_)));
    }
}
