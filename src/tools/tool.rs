// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

use async_trait::async_trait;

use schemars::Schema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::{error::Error, runner::AgentEvent};

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

/// Receives incremental progress updates emitted by a tool mid-call.
///
/// The runner passes a `&dyn ProgressReporter` into [`Tool::propose`] and
/// [`Tool::apply`]; each
/// [`update`](ProgressReporter::update) emits a `ToolCallUpdate` event on the
/// run's event stream. Tool authors do not implement this — the runner
/// supplies the implementation.
#[async_trait]
pub trait ProgressReporter: Send + Sync {
    /// Emits a progress update carrying a [`ProgressDetails`] payload.
    ///
    /// Delivery is guaranteed but applies backpressure: if the run's event
    /// channel is full, this awaits until the consumer drains it, so a slow
    /// consumer can throttle a chatty tool.
    async fn update(&self, details: ProgressDetails);
}

/// The payload carried by a progress update emitted through
/// [`ProgressReporter::update`] and surfaced on
/// [`AgentEvent::ToolCallUpdate`](crate::runner::AgentEvent::ToolCallUpdate).
///
/// Most tools report progress with [`Other`](ProgressDetails::Other), wrapping
/// whatever JSON best describes their current state. The
/// [`AgentUpdate`](ProgressDetails::AgentUpdate) variant is produced by
/// [`AgentTool`](crate::tools::AgentTool) to relay a nested child agent's own
/// events upward, giving the parent consumer visibility into the child run.
#[derive(Debug)]
pub enum ProgressDetails {
    /// An event from a nested child agent, forwarded by
    /// [`AgentTool`](crate::tools::AgentTool) so the parent run can observe the
    /// child's progress. Boxed because [`AgentEvent`] embeds `ProgressDetails`,
    /// which would otherwise make the type infinitely sized.
    AgentUpdate(Box<AgentEvent>),
    /// A tool-defined JSON payload describing the tool's current state. This is
    /// what most tools emit from [`Tool::apply`].
    Other(Value),
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
/// use async_trait::async_trait;
/// use agent_rig::error::Error;
/// use agent_rig::tools::{ProgressReporter, Tool, ToolDefinition};
/// use schemars::json_schema;
/// use serde_json::{Value, json};
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
///     // `propose` is left as the default (it returns the args unchanged), so
///     // only `apply` needs implementing.
///     async fn apply(
///         &self,
///         proposal: Value,
///         _progress: &dyn ProgressReporter,
///         _cancel: CancellationToken,
///     ) -> Result<Value, Error> {
///         Ok(json!({ "echo": proposal }))
///     }
/// }
/// ```
#[async_trait]
pub trait Tool: Send + Sync {
    /// Returns the definition that describes this tool to the model.
    fn definition(&self) -> &ToolDefinition;

    /// Returns a short, human-readable title for a specific invocation,
    /// surfaced on [`AgentEvent::ToolCallStart`].
    ///
    /// `args` is the raw tool-call JSON. The default returns the tool's name
    /// and ignores `args`; override it to derive a more descriptive label (for
    /// example, `"Read foo.rs"` instead of `"read_file"`). Returning `Err`
    /// signals the arguments could not be interpreted — the runner falls back
    /// to the tool name.
    ///
    /// [`AgentEvent::ToolCallStart`]: crate::runner::AgentEvent::ToolCallStart
    fn title(&self, _args: &Value) -> Result<String, Error> {
        Ok(self.definition().name.clone())
    }

    /// Plans the call without committing side effects, returning a *proposal*:
    /// a JSON value that both the [`AuthManager`](crate::auth::AuthManager) and
    /// [`apply`](Tool::apply) read from.
    ///
    /// A tool call runs in two phases. `propose` resolves the model's raw
    /// `args` into the concrete thing that will happen — for an edit tool, it
    /// reads the file and computes the new contents, returning something like
    /// `{ "path": …, "old_text": …, "new_text": … }`. The runner shows that
    /// proposal to the `AuthManager` (which can render a diff from it), and if
    /// approved hands the *same* value to `apply` (which writes `new_text`).
    /// Because one value drives both, what the approver sees and what executes
    /// can never drift apart.
    ///
    /// `propose` runs on **every** call — before authorization, and even when
    /// no authorization is required — because its result is what `apply`
    /// consumes. It must therefore be **side-effect-free**: it may read
    /// (resolve a path, compute a diff, expand a command), but it must not
    /// mutate. Returning `Err` aborts the call before authorization is ever
    /// requested and surfaces the error to the model.
    ///
    /// The default returns the `args` unchanged, which is right for any tool
    /// whose call needs no planning. Override it to resolve `args` into a
    /// richer proposal.
    ///
    /// `progress` and `cancel` behave as described on [`apply`](Tool::apply).
    async fn propose(
        &self,
        args: &Value,
        _progress: &dyn ProgressReporter,
        _cancel: CancellationToken,
    ) -> Result<Value, Error> {
        Ok(args.clone())
    }

    /// Executes an approved proposal.
    ///
    /// `proposal` is the value [`propose`](Tool::propose) returned for this
    /// call, handed back verbatim once it is authorized; the returned value is
    /// sent back to the model as the tool result. For the default `propose`,
    /// `proposal` is just the raw tool-call JSON.
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
    ///
    /// `progress` reports incremental progress: call
    /// [`progress.update(details)`](ProgressReporter::update) to emit a
    /// `ToolCallUpdate` event for this call. Delivery is guaranteed but
    /// awaits, so it applies backpressure under a slow consumer.
    async fn apply(
        &self,
        proposal: Value,
        progress: &dyn ProgressReporter,
        cancel: CancellationToken,
    ) -> Result<Value, Error>;
}
