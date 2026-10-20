// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

//! Drives an [`Agent`] against an [`LlmModel`] until it produces a final reply.
//!
//! [`AgentRunner`] owns the model and the tool definitions. Calling
//! [`AgentRunner::run`] returns a [`Stream`] of [`AgentEvent`]s — text and
//! reasoning deltas, plus the lifecycle of each tool call. The agentic loop
//! runs inside the stream itself: it makes progress only while the stream is
//! polled, and dropping the stream stops it.
//!
//! The loop keeps calling the model until a turn comes back with no tool
//! calls. Within a single turn, every tool call is emitted before the runner
//! waits on any of them, so the consumer can resolve them concurrently; their
//! result messages are appended to the thread in the same order the model
//! issued them, regardless of the order in which they are resolved.

use std::{
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use futures_util::{Stream, StreamExt, future::join_all};
use serde_json::Value;
use tokio::sync::oneshot;
use tokio_util::sync::{CancellationToken, DropGuard};
use tracing::{debug, info};

use crate::{
    Agent,
    model::{LlmModel, Message, MessageList, ModelRequest, ModelStreamChunk, ToolCall},
    tools::{ToolCallRequest, ToolDefinition, ToolResult},
};

mod events;
pub use events::{AgentEvent, RunEvent, ToolCallResult};

/// Returns a process-unique id for a new run.
fn next_run_id() -> usize {
    static RUN_ID_FACTORY: AtomicUsize = AtomicUsize::new(0);
    RUN_ID_FACTORY.fetch_add(1, Ordering::Relaxed)
}

/// Drives an agent against an [`LlmModel`] and a set of [`ToolDefinition`]s.
///
/// Construct one with [`AgentRunner::new`] (no tools) or
/// [`AgentRunner::with_tools`] (with tools). Call [`AgentRunner::run`] to
/// start the agentic loop and consume the returned stream until it ends.
///
/// `AgentRunner` is cheap to clone — internals are behind [`Arc`] — so a
/// single runner can be shared across tasks.
#[derive(Clone)]
pub struct AgentRunner {
    model: Arc<dyn LlmModel>,
    tools: Vec<ToolDefinition>,
}

impl AgentRunner {
    /// Creates a runner that uses `model` and has no tools registered.
    pub fn new(model: Arc<dyn LlmModel>) -> Self {
        AgentRunner {
            model,
            tools: vec![],
        }
    }

    /// Creates a runner that uses `model` and the supplied tool definitions.
    pub fn with_tools(model: Arc<dyn LlmModel>, tools: Vec<ToolDefinition>) -> Self {
        AgentRunner { model, tools }
    }

    /// Runs `agent` starting from `thread` and returns the event stream.
    ///
    /// The agentic loop runs inside the returned stream: nothing happens —
    /// no model request is sent — until the stream is first polled, and the
    /// loop only advances while the consumer keeps polling. The stream ends
    /// after the model produces a turn with no tool calls, or after a
    /// terminal [`AgentEvent::Error`] / [`AgentEvent::Cancelled`].
    ///
    /// `thread` is the conversation so far — typically a single
    /// [`Message::user`](crate::model::Message::user) for the first turn, or a
    /// previously accumulated history for follow-ups. The thread is consumed;
    /// the resulting thread is not returned (each call starts a fresh loop).
    ///
    /// **Dropping the returned stream cancels the run.** The in-flight
    /// provider HTTP call is dropped, and the cancellation token handed to
    /// each pending [`ToolCallRequest`] fires so the consumer's tool work can
    /// stop. Consumers that need to share a cancel token with a sibling task
    /// (deadline timer, multi-run coordination) should use
    /// [`AgentRunner::run_with_cancellation`] instead.
    pub fn run(
        &self,
        agent: &Agent,
        thread: MessageList,
    ) -> Pin<Box<dyn Stream<Item = RunEvent> + Send>> {
        self.run_with_cancellation(agent, thread, CancellationToken::new())
    }

    /// Runs `agent` like [`run`](Self::run), but also cancels when `cancel`
    /// fires.
    ///
    /// The supplied token is the caller's; the runner does not cancel it.
    /// Internally the runner derives a child token from `cancel` and binds
    /// the drop-on-stream-drop guard to that child, so dropping the returned
    /// stream cancels the run without cancelling the caller's token (which
    /// may be shared with siblings). Either trigger — external cancel or
    /// stream drop — terminates the run; whichever fires first wins.
    ///
    /// When cancellation is triggered while the consumer is still draining
    /// the stream, the runner emits a terminal [`AgentEvent::Cancelled`]
    /// before the stream ends. When cancellation is triggered by dropping
    /// the stream, the loop is dropped with it and no `Cancelled` event is
    /// produced.
    pub fn run_with_cancellation(
        &self,
        agent: &Agent,
        thread: MessageList,
        cancel: CancellationToken,
    ) -> Pin<Box<dyn Stream<Item = RunEvent> + Send>> {
        debug!(agent = agent.name(), "starting run");
        // Child of the caller's token: external `cancel` firing still
        // propagates, but dropping the stream (which drops the guard) only
        // cancels the child, leaving the caller's token alone.
        let cancel = cancel.child_token();
        let guard = cancel.clone().drop_guard();
        // The returned stream must be `'static`, so it owns clones of the
        // runner and agent rather than borrowing `&self` / `&Agent`.
        Box::pin(
            self.clone()
                .event_stream(agent.clone(), thread, cancel, guard),
        )
    }

    /// The agentic loop, as a stream of events for one run.
    ///
    /// `guard` is moved into the stream's state, so dropping the stream —
    /// even before it is first polled — cancels `cancel`.
    fn event_stream(
        self,
        agent: Agent,
        mut thread: MessageList,
        cancel: CancellationToken,
        guard: DropGuard,
    ) -> impl Stream<Item = RunEvent> + Send {
        let run_id = next_run_id();
        let event = move |agent_event| RunEvent {
            run_id,
            agent_event,
        };

        async_stream::stream! {
            let _guard = guard;
            yield event(AgentEvent::TurnStart);

            loop {
                // Top-of-loop checkpoint: short-circuit before building the
                // next request if cancellation already fired (e.g. right
                // after the previous tool-call phase completed).
                if cancel.is_cancelled() {
                    yield event(AgentEvent::Cancelled);
                    return;
                }

                let request = ModelRequest {
                    messages: thread.clone(),
                    system: Some(agent.instructions().to_string()),
                    output_schema: agent.output_schema().cloned(),
                    tools: self.tools.clone(),
                };

                let mut model_stream = self.model.generate_stream(request);
                let mut tool_calls: Vec<Arc<ToolCall>> = Vec::new();
                let mut reply = String::new();
                let mut provider_metadata: Option<Value> = None;
                loop {
                    tokio::select! {
                        biased;
                        _ = cancel.cancelled() => {
                            // Drop the model stream — this drops the
                            // underlying reqwest future and aborts the
                            // in-flight HTTP call.
                            drop(model_stream);
                            yield event(AgentEvent::Cancelled);
                            return;
                        }
                        next = model_stream.next() => {
                            let Some(chunk) = next else { break };
                            match chunk {
                                Ok(ModelStreamChunk::Thinking(t)) => {
                                    yield event(AgentEvent::ThinkingDelta(t));
                                }
                                Ok(ModelStreamChunk::TextDelta(t)) => {
                                    reply.push_str(&t);
                                    yield event(AgentEvent::TextDelta(t));
                                }
                                Ok(ModelStreamChunk::ToolCall(call)) => {
                                    tool_calls.push(Arc::new(call));
                                }
                                Ok(ModelStreamChunk::Usage(usage)) => {
                                    yield event(AgentEvent::Usage(usage));
                                }
                                Ok(ModelStreamChunk::ProviderMetadata(metadata)) => {
                                    provider_metadata = Some(metadata);
                                }
                                Err(error) => {
                                    yield event(AgentEvent::Error(error));
                                    return;
                                }
                            }
                        }
                    }
                }

                if tool_calls.is_empty() {
                    if !reply.is_empty() {
                        let mut message = Message::assistant(reply);
                        message.provider_metadata = provider_metadata;
                        thread.push(message);
                    }
                    yield event(AgentEvent::TurnFinish { thread });
                    return;
                }

                // Opaque adapter state rides on the assistant message so the
                // adapter can replay it on later turns.
                let mut message = Message::tool_calls(tool_calls.clone());
                message.provider_metadata = provider_metadata;
                thread.push(message);

                // Emit every request before waiting on any of them, so the
                // consumer sees the whole turn and can resolve the calls
                // concurrently.
                let mut pending = Vec::with_capacity(tool_calls.len());
                for call in tool_calls {
                    info!("Invoking tool '{}' with args '{:?}'", call.name, call.args);
                    let (resolve_tx, resolve_rx) = oneshot::channel();
                    yield event(AgentEvent::ToolCall(ToolCallRequest::new(
                        call.clone(),
                        cancel.clone(),
                        resolve_tx,
                    )));
                    pending.push((call, resolve_rx));
                }

                // `join_all` preserves input order in the returned Vec, so
                // tool-result messages are appended in the same order the
                // model requested them, whatever order they resolve in.
                let replies = join_all(pending.into_iter().map(|(call, resolve_rx)| async move {
                    let result = resolve_rx.await.unwrap_or_else(|_| {
                        ToolResult::error(format!("Tool call {} failed", call.name))
                    });
                    info!("Tool '{}' responded with result '{:?}'", call.name, result);
                    (call, result)
                }));
                let results = tokio::select! {
                    biased;
                    _ = cancel.cancelled() => {
                        yield event(AgentEvent::Cancelled);
                        return;
                    }
                    results = replies => results,
                };
                for (call, result) in results {
                    thread.push(Message::tool_result(call, result));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
