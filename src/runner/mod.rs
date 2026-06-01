//! Drives an [`Agent`] against an [`LlmModel`] until it produces a final reply.
//!
//! [`AgentRunner`] owns the model, the [`ToolRegistry`], and (optionally) an
//! [`AuthManager`]. Calling [`AgentRunner::run`] spawns the agentic loop on a
//! background task and returns a [`Stream`] of [`AgentEvent`]s — text and
//! reasoning deltas, plus the lifecycle of each tool call.
//!
//! The loop keeps calling the model until a turn comes back with no tool
//! calls. Within a single turn, tool calls are executed concurrently; their
//! result messages are appended to the thread in the same order the model
//! issued them, even though the corresponding events may interleave.

use std::{
    pin::Pin,
    sync::{Arc, atomic::AtomicUsize},
};

use futures_util::{Stream, StreamExt, future::join_all};
use tokio::sync::mpsc::{self, Sender, error::SendError};
use tracing::debug;

use crate::{
    Agent,
    auth::AuthManager,
    model::{LlmModel, Message, ModelRequest, ModelStreamChunk, ToolCall},
    tools::{ToolDefinition, ToolRegistry, ToolRegistryEntry},
};

mod events;
pub use events::{AgentEvent, RunEvent, ToolCallResult};

/// Buffer size of the per-run mpsc channel that carries [`RunEvent`]s from
/// the spawned agentic loop to the consumer's stream. Sized to absorb a
/// typical token burst from a streaming provider without forcing the
/// worker to round-trip on every event.
const EVENT_CHANNEL_CAPACITY: usize = 100;

#[derive(Debug)]
pub struct RunEmitter {
    pub run_id: usize,
    pub parent: Option<usize>,
    pub tx: Sender<RunEvent>,
}

impl RunEmitter {
    fn next_run_id() -> usize {
        static RUN_ID_FACTORY: AtomicUsize = AtomicUsize::new(0);
        RUN_ID_FACTORY.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }

    pub fn new(tx: Sender<RunEvent>, parent: Option<usize>) -> Self {
        let run_id = RunEmitter::next_run_id();
        Self { tx, run_id, parent }
    }

    pub async fn send(&self, event: AgentEvent) -> Result<(), SendError<RunEvent>> {
        let event = RunEvent {
            run_id: self.run_id,
            parent: self.parent,
            agent_event: event,
        };
        self.tx.send(event).await
    }

    pub fn child(&self) -> Self {
        Self {
            parent: Some(self.run_id),
            run_id: RunEmitter::next_run_id(),
            tx: self.tx.clone(),
        }
    }
}

/// Drives an agent against an [`LlmModel`] and a [`ToolRegistry`].
///
/// Construct one with [`AgentRunner::new`] (no tools) or
/// [`AgentRunner::with_registry`] (with tools), optionally chaining
/// [`AgentRunner::with_auth_manager`] to gate tool execution. Call
/// [`AgentRunner::run`] to start the agentic loop and consume the returned
/// stream until it ends.
///
/// `AgentRunner` is cheap to clone — internals are behind [`Arc`] — so a
/// single runner can be shared across tasks.
#[derive(Clone)]
pub struct AgentRunner {
    model: Arc<dyn LlmModel>,
    registry: Arc<ToolRegistry>,
    auth_manager: Option<Arc<dyn AuthManager>>,
}

impl AgentRunner {
    /// Creates a runner that uses `model` and has no tools registered.
    pub fn new(model: Arc<dyn LlmModel>) -> Self {
        AgentRunner {
            model,
            registry: Arc::new(ToolRegistry::new()),
            auth_manager: None,
        }
    }

    /// Creates a runner that uses `model` and the supplied [`ToolRegistry`].
    pub fn with_registry(model: Arc<dyn LlmModel>, registry: Arc<ToolRegistry>) -> Self {
        AgentRunner {
            model,
            registry,
            auth_manager: None,
        }
    }

    /// Sets the [`AuthManager`] consulted before every tool call.
    ///
    /// With no manager set, no authorization is performed and all calls run.
    /// The manager decides which calls require approval and how to obtain it.
    pub fn with_auth_manager(mut self, auth_manager: Arc<dyn AuthManager>) -> Self {
        self.auth_manager = Some(auth_manager);
        self
    }

    /// Runs `agent` starting from `thread` and returns the event stream.
    ///
    /// The agentic loop runs on a background tokio task; events are delivered
    /// through an mpsc channel as they happen. The stream ends after the
    /// model produces a turn with no tool calls, or after an
    /// [`AgentEvent::Error`].
    ///
    /// `thread` is the conversation so far — typically a single
    /// [`Message::user`](crate::model::Message::user) for the first turn, or a
    /// previously accumulated history for follow-ups. The thread is consumed;
    /// the resulting thread is not returned (each call starts a fresh loop).
    pub fn run(
        &self,
        agent: &Agent,
        thread: Vec<Message>,
    ) -> Pin<Box<dyn Stream<Item = RunEvent> + Send>> {
        debug!(agent = agent.name(), "starting run");
        // Clone `self` and `agent` outside the `stream!` macro block: the
        // generator is spawned with `tokio::spawn` and must therefore be
        // `'static`, so it can't capture the non-`'static` `&self` or
        // `&Agent` references that `run` received.
        let cloned = self.clone();
        let agent = agent.clone();

        let stream = async_stream::stream! {
          let (tx, mut rx) = mpsc::channel::<RunEvent>(EVENT_CHANNEL_CAPACITY);
          let tx = RunEmitter::new(tx, None);
          tokio::spawn(cloned.main_loop(tx, agent, thread));

          while let Some(message) = rx.recv().await {
            yield message;
          }
        };
        Box::pin(stream)
    }

    async fn main_loop(self, tx: RunEmitter, agent: Agent, mut thread: Vec<Message>) {
        let tools: Vec<ToolDefinition> = self.registry.definitions();

        loop {
            let request = ModelRequest {
                messages: thread.clone(),
                system: Some(agent.instructions().to_string()),
                output_schema: agent.output_schema().cloned(),
                tools: tools.clone(),
            };

            let mut model_stream = self.model.generate_stream(request);
            let mut tool_calls: Vec<ToolCall> = Vec::new();
            while let Some(chunk) = model_stream.next().await {
                match chunk {
                    Ok(ModelStreamChunk::Thinking(t)) => {
                        let _ = tx.send(AgentEvent::ThinkingDelta(t)).await;
                    }
                    Ok(ModelStreamChunk::TextDelta(t)) => {
                        let _ = tx.send(AgentEvent::TextDelta(t)).await;
                    }
                    Ok(ModelStreamChunk::ToolCall(call)) => {
                        tool_calls.push(call);
                    }
                    Ok(ModelStreamChunk::Usage(usage)) => {
                        let _ = tx.send(AgentEvent::Usage(usage)).await;
                    }
                    Err(error) => {
                        let _ = tx.send(AgentEvent::Error(error)).await;
                        return;
                    }
                }
            }

            if tool_calls.is_empty() {
                break;
            }

            self.handle_tool_calls(&tx, tool_calls, &mut thread).await;
        }
    }

    async fn handle_tool_calls(
        &self,
        tx: &RunEmitter,
        tool_calls: Vec<ToolCall>,
        thread: &mut Vec<Message>,
    ) {
        // Append the tool calls as a single assistant turn.
        thread.push(Message::tool_calls(tool_calls.clone()));

        // Each future runs the full lifecycle for one call: authorization check
        // (if required), emit Started, execute, emit Finished / Error / Denied.
        // Hallucinated calls skip Started but still produce a synthetic result
        // so the assistant turn and tool-result messages remain paired.
        let tool_futures = tool_calls.into_iter().map(|call| async move {
            // Hallucinated tool: produce a synthetic message, no events.
            let Some(tool) = self.registry.get(&call.name) else {
                return (call, ToolCallResult::Unknown);
            };

            // Authorization gate: the sync check decides whether to consult
            // the async decision path. If no manager is configured, no gating.
            if let Some(auth) = &self.auth_manager
                && auth.requires_authorization(&call.name, &call.args)
                && !auth.authorize(&call.name, &call.args).await
            {
                let result = ToolCallResult::Denied;
                let _ = tx
                    .send(AgentEvent::ToolCallFinished {
                        name: call.name.clone(),
                        result: result.clone(),
                    })
                    .await;
                return (call, result);
            }

            let _ = tx
                .send(AgentEvent::ToolCallStarted {
                    name: call.name.clone(),
                    args: call.args.clone(),
                })
                .await;

            let event: ToolCallResult = match tool {
                ToolRegistryEntry::Tool(t) => t.call(call.args.clone()).await,
                ToolRegistryEntry::Agent(a) => a.call(tx.child(), call.args.clone()).await,
            }
            .into();

            debug!(tool = call.name, "tool call complete");
            let _ = tx
                .send(AgentEvent::ToolCallFinished {
                    name: call.name.clone(),
                    result: event.clone(),
                })
                .await;
            (call, event)
        });

        // Run all calls concurrently. `join_all` preserves input order in the
        // returned Vec, so tool-result messages are appended in the same order
        // the model requested them — even though events may interleave.
        let results = join_all(tool_futures).await;
        for (call, result) in results {
            thread.push(Message::tool_result(
                call.id,
                call.name,
                result.into(),
                call.provider_metadata,
            ));
        }
    }
}

#[cfg(test)]
mod tests;
