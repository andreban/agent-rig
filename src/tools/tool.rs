// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0
use async_trait::async_trait;

use schemars::Schema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{fmt::Display, sync::Arc};
use tokio_util::sync::CancellationToken;

use crate::model::{IntoToolResultContents, Media, ToolCall, ToolResultContent, tool_result_json};

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

/// The outcome of a [`Tool`] call: a list of [`ToolResultContent`] blocks on success,
/// or an error [`Value`].
///
/// Tool errors are *soft*. An [`Err`](ToolResult::Err) is not a control-flow
/// failure that aborts the run — it is sent back to the model as data so it
/// can react (retry, apologise, pick a different approach). Converting a
/// `ToolResult` into `Vec<ToolResultContent>` — which happens before the result
/// reaches the model — wraps JSON payloads in an envelope so the model can tell
/// success and error apart:
///
/// - [`Ok(blocks)`](ToolResult::Ok) wraps each [`ToolResultContent::Json(v)`](ToolResultContent::Json) in
///   `{"success": v}` and passes [`ToolResultContent::Media`] blocks through unchanged.
/// - [`Err(v)`](ToolResult::Err) becomes `[ToolResultContent::Json({"error": v})]`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[must_use]
pub enum ToolResult {
    /// A successful call carrying one or more [`ToolResultContent`] blocks (JSON,
    /// [`Media`], or both).
    Ok(Vec<ToolResultContent>),
    /// A failed call. The payload describes the error and reaches the model
    /// under an `"error"` key; it does **not** abort the run.
    Err(Value),
}

impl ToolResult {
    /// Builds an [`Ok`](ToolResult::Ok) from anything convertible into
    /// `Vec<ToolResultContent>` (a single [`Value`], [`Media`], [`ToolResultContent`],
    /// `String`, `&str`, `bool`, or a `Vec<ToolResultContent>`).
    pub fn ok(content: impl IntoToolResultContents) -> Self {
        ToolResult::Ok(content.into_tool_result_contents())
    }

    /// Builds an [`Err`](ToolResult::Err) from anything convertible into a [`Value`].
    pub fn error<T: Into<Value>>(error: T) -> Self {
        ToolResult::Err(error.into())
    }
}

impl From<ToolResult> for Vec<ToolResultContent> {
    fn from(result: ToolResult) -> Self {
        match result {
            ToolResult::Ok(blocks) => blocks
                .into_iter()
                .map(|block| match block {
                    ToolResultContent::Json(v) => {
                        let mut map = serde_json::Map::new();
                        map.insert("success".to_string(), v);
                        ToolResultContent::Json(Value::Object(map))
                    }
                    ToolResultContent::Media(m) => ToolResultContent::Media(m),
                })
                .collect(),
            ToolResult::Err(v) => {
                let mut map = serde_json::Map::new();
                map.insert("error".to_string(), v);
                vec![ToolResultContent::Json(Value::Object(map))]
            }
        }
    }
}

impl IntoToolResultContents for ToolResult {
    fn into_tool_result_contents(self) -> Vec<ToolResultContent> {
        Vec::<ToolResultContent>::from(self)
    }
}

impl From<Value> for ToolResult {
    fn from(value: Value) -> Self {
        ToolResult::ok(value)
    }
}

impl From<Media> for ToolResult {
    fn from(media: Media) -> Self {
        ToolResult::ok(media)
    }
}

impl From<ToolResultContent> for ToolResult {
    fn from(content: ToolResultContent) -> Self {
        ToolResult::ok(content)
    }
}

impl From<Vec<ToolResultContent>> for ToolResult {
    fn from(content: Vec<ToolResultContent>) -> Self {
        ToolResult::Ok(content)
    }
}

impl<const N: usize> From<[ToolResultContent; N]> for ToolResult {
    fn from(content: [ToolResultContent; N]) -> Self {
        ToolResult::Ok(Vec::from(content))
    }
}

impl From<String> for ToolResult {
    fn from(value: String) -> Self {
        ToolResult::ok(value)
    }
}

impl From<&str> for ToolResult {
    fn from(value: &str) -> Self {
        ToolResult::ok(value)
    }
}

impl From<bool> for ToolResult {
    fn from(value: bool) -> Self {
        ToolResult::ok(value)
    }
}

impl From<ToolResult> for Value {
    fn from(result: ToolResult) -> Value {
        tool_result_json(&Vec::<ToolResultContent>::from(result))
    }
}

impl Display for ToolResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ToolResult::Ok(blocks) => match blocks.as_slice() {
                [ToolResultContent::Json(v)] => write!(f, r#"{{"success": {v}}}"#),
                _ => {
                    let converted = Vec::<ToolResultContent>::from(self.clone());
                    write!(
                        f,
                        "{}",
                        serde_json::to_string(&converted).unwrap_or_default()
                    )
                }
            },
            ToolResult::Err(v) => write!(f, r#"{{"error": {v}}}"#),
        }
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
