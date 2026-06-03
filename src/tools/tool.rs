// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

use async_trait::async_trait;

use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::error::Error;

/// Describes a tool to the model: its name, purpose, and parameter schema.
///
/// `ToolDefinition` is the contract between the agent and the LLM. It is
/// returned by [`Tool::definition`] and forwarded to the model on every run.
/// It is never stored in [`Agent`] — definitions live in the [`ToolRegistry`]
/// alongside their implementations.
///
/// [`Agent`]: crate::Agent
/// [`ToolRegistry`]: crate::tools::ToolRegistry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// The tool name the model uses to invoke it. Must match the key in the
    /// [`ToolRegistry`].
    ///
    /// [`ToolRegistry`]: crate::tools::ToolRegistry
    pub name: String,
    /// A human-readable description that helps the model decide when to call
    /// this tool.
    pub description: String,
    /// JSON Schema object describing the arguments the model must pass.
    pub parameters: serde_json::Value,
}

/// A callable tool that an agent can invoke during inference.
///
/// Implement this trait to expose executable logic to the agentic loop.
/// The [`definition`](Tool::definition) method tells the model what the tool
/// does; [`call`](Tool::call) executes it when the model requests it.
///
/// # Examples
///
/// ```no_run
/// use async_trait::async_trait;
/// use agent_rig::error::Error;
/// use agent_rig::tools::{Tool, ToolDefinition};
/// use serde_json::{Value, json};
/// use tokio_util::sync::CancellationToken;
///
/// struct AddTool;
///
/// #[async_trait]
/// impl Tool for AddTool {
///     fn definition(&self) -> ToolDefinition {
///         ToolDefinition {
///             name: "add".to_string(),
///             description: "Adds two integers and returns the sum.".to_string(),
///             parameters: json!({
///                 "type": "object",
///                 "properties": {
///                     "a": { "type": "integer" },
///                     "b": { "type": "integer" }
///                 },
///                 "required": ["a", "b"]
///             }),
///         }
///     }
///
///     async fn call(&self, args: Value, _cancel: CancellationToken) -> Result<Value, Error> {
///         let a = args["a"].as_i64().unwrap_or(0);
///         let b = args["b"].as_i64().unwrap_or(0);
///         Ok(json!({ "result": a + b }))
///     }
/// }
/// ```
#[async_trait]
pub trait Tool: Send + Sync {
    /// Returns the definition that describes this tool to the model.
    fn definition(&self) -> ToolDefinition;

    /// Executes the tool with the JSON arguments the model provided.
    ///
    /// `args` is the raw JSON object from the model's tool call. Returns a
    /// JSON value that is sent back to the model as the tool result.
    ///
    /// `cancel` fires when the surrounding
    /// [`AgentRunner`](crate::runner::AgentRunner) run is cancelled — either
    /// because the consumer dropped the event stream or because an
    /// externally supplied token fired. Long-running tools should
    /// `select!` on `cancel.cancelled()` or pass the token down to the
    /// libraries they call. Tools that ignore `cancel` still terminate
    /// the run correctly (the runner races each call against `cancel` and
    /// drops the future on cancellation), but any side effects already in
    /// flight may continue in the background until they finish on their
    /// own.
    async fn call(
        &self,
        args: serde_json::Value,
        cancel: CancellationToken,
    ) -> Result<serde_json::Value, Error>;
}
