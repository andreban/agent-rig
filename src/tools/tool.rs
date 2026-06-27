// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0
use async_trait::async_trait;

use schemars::Schema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{fmt::Display, sync::Arc};
use tokio_util::sync::CancellationToken;

use crate::model::ToolCall;

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
    pub parameters: Schema,
}

/// The outcome of a [`Tool`] call: a success payload or an error, each
/// carrying arbitrary JSON.
///
/// Tool errors are *soft*. An [`Err`](ToolResult::Err) is not a control-flow
/// failure that aborts the run — it is sent back to the model as data so it
/// can react (retry, apologise, pick a different approach). Converting a
/// `ToolResult` into a [`Value`] — which happens before the result reaches the
/// model — wraps the payload in an envelope so the model can tell the two
/// apart:
///
/// - [`Ok(v)`](ToolResult::Ok) becomes `{"success": v}`
/// - [`Err(v)`](ToolResult::Err) becomes `{"error": v}`
///
/// The [`Display`] impl produces the same envelope as compact JSON.
#[derive(Debug)]
#[must_use]
pub enum ToolResult {
    /// A successful call. The payload is the tool's output and reaches the
    /// model under a `"success"` key.
    Ok(Value),
    /// A failed call. The payload describes the error and reaches the model
    /// under an `"error"` key; it does **not** abort the run.
    Err(Value),
}

impl ToolResult {
    /// Builds an [`Ok`](ToolResult::Ok) from anything convertible into a [`Value`].
    pub fn ok<T: Into<Value>>(result: T) -> Self {
        ToolResult::Ok(result.into())
    }

    /// Builds an [`Err`](ToolResult::Err) from anything convertible into a [`Value`].
    pub fn error<T: Into<Value>>(error: T) -> Self {
        ToolResult::Err(error.into())
    }

    fn key(&self) -> &'static str {
        match self {
            ToolResult::Ok(_) => "success",
            ToolResult::Err(_) => "error",
        }
    }
}

impl From<ToolResult> for Value {
    fn from(result: ToolResult) -> Value {
        let key = result.key(); // borrow ends here (&'static str)
        let (ToolResult::Ok(v) | ToolResult::Err(v)) = result; // now free to move v out
        let mut map = serde_json::Map::new();
        map.insert(key.to_string(), v);
        Value::Object(map)
    }
}

impl Display for ToolResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (ToolResult::Ok(v) | ToolResult::Err(v)) = self;
        write!(f, r#"{{"{}": {}}}"#, self.key(), v)
    }
}

/// A callable tool that an agent can invoke during inference.
///
/// `Tool` is the object-safe trait the [`ToolRegistry`](crate::tools::ToolRegistry)
/// actually stores: arguments and results are untyped [`serde_json::Value`]s, so
/// a single registry can hold tools with wildly different shapes behind
/// `Box<dyn Tool>`.
///
/// Tools work in raw JSON (for example, a passthrough tool, or
/// [`AgentTool`](crate::tools::AgentTool), which serializes whatever the model
/// sends into a child run).
///
/// # Examples
///
/// ```no_run
/// use std::sync::Arc;
/// use async_trait::async_trait;
/// use agent_rig::model::ToolCall;
/// use agent_rig::tools::{Tool, ToolDefinition, ToolResult};
/// use schemars::json_schema;
/// use serde_json::json;
/// use tokio_util::sync::CancellationToken;
///
/// struct EchoTool {
///     definition: ToolDefinition,
/// }
///
/// impl Default for EchoTool {
///     fn default() -> Self {
///         Self {
///             definition: ToolDefinition {
///                 name: "echo".to_string(),
///                 description: "Echoes the arguments back unchanged.".to_string(),
///                 parameters: json_schema!({ "type": "object" }),
///             },
///         }
///     }
/// }
///
/// #[async_trait]
/// impl Tool for EchoTool {
///     fn definition(&self) -> &ToolDefinition {
///         &self.definition
///     }
///
///     async fn call(
///         &self,
///         tool_call: Arc<ToolCall>,
///         _cancel: CancellationToken,
///     ) -> ToolResult {
///         ToolResult::ok(json!({ "echo": tool_call.args.clone() }))
///     }
/// }
/// ```
#[async_trait]
pub trait Tool: Send + Sync {
    /// Returns the definition that describes this tool to the model.
    fn definition(&self) -> &ToolDefinition;

    /// Returns a short, human-readable title for a specific invocation.
    ///
    /// `args` is the raw tool-call JSON. The default returns the tool's name
    /// and ignores `args`; override it to derive a more descriptive label (for
    /// example, `"Read foo.rs"` instead of `"read_file"`). The client can use
    /// this title when rendering or logging the tool call.
    fn title(&self, _args: &Value) -> String {
        self.definition().name.clone()
    }

    /// Executes the tool call and returns the result sent back to the model.
    ///
    /// `tool_call` carries the model-supplied [`id`](ToolCall::id),
    /// [`name`](ToolCall::name), and [`args`](ToolCall::args). Deserialize
    /// `tool_call.args` into whatever parameters the tool expects; returning a
    /// [`ToolResult::Err`] on malformed input is fine — a soft error is sent to
    /// the model as data so it can react, not a hard failure that aborts the
    /// run.
    ///
    /// Gating a call behind user approval is the consumer's concern, handled in
    /// the event loop before `call` is invoked (see
    /// [`AgentEvent::ToolCall`](crate::runner::AgentEvent::ToolCall)); a tool
    /// that reaches `call` may assume it is authorized to run.
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
    async fn call(&self, tool_call: Arc<ToolCall>, cancel: CancellationToken) -> ToolResult;
}
