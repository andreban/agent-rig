// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

use std::marker::PhantomData;

use async_trait::async_trait;

use schemars::Schema;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
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

/// A callable tool that an agent can invoke during inference.
///
/// Implement this trait to expose executable logic to the agentic loop.
/// The [`definition`](Tool::definition) method tells the model what the tool
/// does; [`call`](Tool::call) executes it when the model requests it.
///
/// `I` is the argument type the tool receives. The runner deserializes the
/// model's JSON tool-call payload into `I` before invoking [`call`], so `I`
/// must implement [`DeserializeOwned`]. `O` is the result type the tool
/// returns; the runner serializes it back to JSON when forwarding the
/// tool result to the model, so `O` must implement [`Serialize`].
///
/// For tools that don't benefit from typed args, use
/// `Tool<serde_json::Value, serde_json::Value>` and operate on raw JSON.
///
/// # Examples
///
/// ```no_run
/// use async_trait::async_trait;
/// use agent_rig::error::Error;
/// use agent_rig::tools::{ProgressReporter, Tool, ToolDefinition};
/// use schemars::json_schema;
/// use serde::{Deserialize, Serialize};
/// use tokio_util::sync::CancellationToken;
///
/// #[derive(Deserialize)]
/// struct AddArgs { a: i64, b: i64 }
///
/// #[derive(Serialize)]
/// struct AddResult { result: i64 }
///
/// struct AddTool {
///     definition: ToolDefinition,
/// }
///
/// impl Default for AddTool {
///     fn default() -> Self {
///         Self {
///             definition: ToolDefinition {
///                 name: "add".to_string(),
///                 description: "Adds two integers and returns the sum.".to_string(),
///                 parameters: json_schema!({
///                     "type": "object",
///                     "properties": {
///                         "a": { "type": "integer" },
///                         "b": { "type": "integer" }
///                     },
///                     "required": ["a", "b"]
///                 }),
///             },
///         }
///     }
/// }
///
/// #[async_trait]
/// impl Tool<AddArgs, AddResult> for AddTool {
///     fn definition(&self) -> &ToolDefinition {
///         &self.definition
///     }
///
///     async fn call(
///         &self,
///         args: AddArgs,
///         _progress: &dyn ProgressReporter,
///         _cancel: CancellationToken,
///     ) -> Result<AddResult, Error> {
///         Ok(AddResult { result: args.a + args.b })
///     }
/// }
/// ```
/// Receives incremental progress updates emitted by a tool mid-call.
///
/// The runner passes a `&dyn ProgressReporter` into [`Tool::call`]; each
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
#[derive(Debug, Clone)]
pub enum ProgressDetails {
    /// An event from a nested child agent, forwarded by
    /// [`AgentTool`](crate::tools::AgentTool) so the parent run can observe the
    /// child's progress. Boxed because [`AgentEvent`] embeds `ProgressDetails`,
    /// which would otherwise make the type infinitely sized.
    AgentUpdate(Box<AgentEvent>),
    /// A tool-defined JSON payload describing the tool's current state. This is
    /// what most tools emit from [`Tool::call`].
    Other(Value),
}

#[async_trait]
pub trait Tool<I, O>: Send + Sync
where
    I: DeserializeOwned + Send,
    O: Serialize + Send,
{
    /// Returns a short, human-readable title for a specific invocation,
    /// surfaced on [`AgentEvent::ToolCallStarted`].
    ///
    /// The default returns the tool's name. Override it to derive a more
    /// descriptive label from the decoded `args` (for example,
    /// `"Read foo.rs"` instead of `"read_file"`).
    ///
    /// [`AgentEvent::ToolCallStarted`]: crate::runner::AgentEvent::ToolCallStarted
    fn title(&self, _args: &I) -> String {
        self.definition().name.clone()
    }

    /// Returns the definition that describes this tool to the model.
    fn definition(&self) -> &ToolDefinition;

    /// Executes the tool with the arguments the model provided.
    ///
    /// `args` is decoded from the model's tool-call JSON before this method
    /// is invoked. The returned value is encoded back to JSON and sent to
    /// the model as the tool result.
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
    async fn call(
        &self,
        args: I,
        progress: &dyn ProgressReporter,
        cancel: CancellationToken,
    ) -> Result<O, Error>;
}

/// Object-safe view of a [`Tool`] that hides the typed argument and result
/// behind `serde_json::Value`. This is what the [`ToolRegistry`](crate::tools::ToolRegistry)
/// actually stores: every concrete `Tool<I, O>` is wrapped in a
/// [`ToolBridge`] before being inserted, so a single registry can hold
/// tools with different `I`/`O` types.
///
/// Not intended for direct implementation — implement [`Tool`] instead.
///
/// [`ToolRegistry`]: crate::tools::ToolRegistry
#[doc(hidden)]
#[async_trait]
pub trait ErasedTool: Send + Sync {
    fn definition(&self) -> &ToolDefinition;
    fn title(&self, args: &Value) -> Result<String, Error>;
    async fn call(
        &self,
        args: serde_json::Value,
        progress: &dyn ProgressReporter,
        cancel: CancellationToken,
    ) -> Result<serde_json::Value, Error>;
}

/// Wraps a typed [`Tool`] in an object-safe [`ErasedTool`] by serializing
/// arguments and results at the boundary. The registry builds this
/// internally; tool authors never name it.
#[doc(hidden)]
pub struct ToolBridge<T, I, O> {
    tool: T,
    _phantom: PhantomData<fn(I) -> O>,
}

impl<T, I, O> ToolBridge<T, I, O> {
    pub fn new(tool: T) -> Self {
        Self {
            tool,
            _phantom: PhantomData,
        }
    }
}

#[async_trait]
impl<T, I, O> ErasedTool for ToolBridge<T, I, O>
where
    T: Tool<I, O> + Send + Sync,
    I: DeserializeOwned + Send,
    O: Serialize + Send,
{
    // Returns the name of this tool.
    fn title(&self, args: &Value) -> Result<String, Error> {
        let typed: I = serde_json::from_value(args.clone())
            .map_err(|e| Error::Agent(format!("invalid tool arguments: {e}")))?;
        let result = self.tool.title(&typed);
        Ok(result)
    }

    fn definition(&self) -> &ToolDefinition {
        Tool::definition(&self.tool)
    }

    async fn call(
        &self,
        args: serde_json::Value,
        progress: &dyn ProgressReporter,
        cancel: CancellationToken,
    ) -> Result<serde_json::Value, Error> {
        let typed: I = serde_json::from_value(args)
            .map_err(|e| Error::Agent(format!("invalid tool arguments: {e}")))?;
        let output = Tool::call(&self.tool, typed, progress, cancel).await?;
        serde_json::to_value(output)
            .map_err(|e| Error::Agent(format!("failed to serialize tool result: {e}")))
    }
}
