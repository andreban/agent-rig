// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

//! Drives an [`Agent`] against an [`LlmModel`] until it produces a final reply.
//!
//! [`AgentRunner`] owns the model and the [`ToolRegistry`]. Calling
//! [`AgentRunner::run`] spawns the agentic loop on a
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
use serde_json::Value;
use tokio::sync::{
    mpsc::{self, Sender, error::SendError},
    oneshot,
};
use tokio_util::sync::CancellationToken;
use tracing::{debug, info};

use crate::{
    Agent,
    model::{LlmModel, Message, ModelRequest, ModelStreamChunk, ToolCall},
    tools::{ToolCallRequest, ToolDefinition},
};

mod events;
pub use events::{AgentEvent, RunEvent, ToolCallResult};

/// Buffer size of the per-run mpsc channel that carries [`RunEvent`]s from
/// the spawned agentic loop to the consumer's stream. Sized to absorb a
/// typical token burst from a streaming provider without forcing the
/// worker to round-trip on every event.
const EVENT_CHANNEL_CAPACITY: usize = 100;

#[derive(Debug)]
pub(crate) struct RunEmitter {
    pub run_id: usize,
    tx: Sender<RunEvent>,
}

impl RunEmitter {
    fn next_run_id() -> usize {
        static RUN_ID_FACTORY: AtomicUsize = AtomicUsize::new(0);
        RUN_ID_FACTORY.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }

    fn new(tx: Sender<RunEvent>) -> Self {
        let run_id = RunEmitter::next_run_id();
        Self { tx, run_id }
    }

    pub async fn send(&self, event: AgentEvent) -> Result<(), SendError<RunEvent>> {
        let event = RunEvent {
            run_id: self.run_id,
            agent_event: event,
        };
        self.tx.send(event).await
    }
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

    /// Creates a runner that uses `model` and the supplied [`ToolRegistry`].
    pub fn with_tools(model: Arc<dyn LlmModel>, tools: Vec<ToolDefinition>) -> Self {
        AgentRunner { model, tools }
    }

    /// Runs `agent` starting from `thread` and returns the event stream.
    ///
    /// The agentic loop runs on a background tokio task; events are delivered
    /// through an mpsc channel as they happen. The stream ends after the
    /// model produces a turn with no tool calls, or after a terminal
    /// [`AgentEvent::Error`] / [`AgentEvent::Cancelled`].
    ///
    /// `thread` is the conversation so far — typically a single
    /// [`Message::user`](crate::model::Message::user) for the first turn, or a
    /// previously accumulated history for follow-ups. The thread is consumed;
    /// the resulting thread is not returned (each call starts a fresh loop).
    ///
    /// **Dropping the returned stream cancels the run.** The in-flight
    /// provider HTTP call and any concurrently running tool futures are
    /// dropped at their next await point. Consumers that need to share a
    /// cancel token with a sibling task (deadline timer, multi-run
    /// coordination) should use [`AgentRunner::run_with_cancellation`]
    /// instead.
    pub fn run(
        &self,
        agent: &Agent,
        thread: Vec<Arc<Message>>,
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
    /// the stream, the `Cancelled` event is best-effort and typically not
    /// observed (the receiver is already gone).
    pub fn run_with_cancellation(
        &self,
        agent: &Agent,
        thread: Vec<Arc<Message>>,
        cancel: CancellationToken,
    ) -> Pin<Box<dyn Stream<Item = RunEvent> + Send>> {
        debug!(agent = agent.name(), "starting run");
        // Clone `self` and `agent` outside the `stream!` macro block: the
        // generator is spawned with `tokio::spawn` and must therefore be
        // `'static`, so it can't capture the non-`'static` `&self` or
        // `&Agent` references that `run` received.
        let cloned = self.clone();
        let agent = agent.clone();

        // Child of the caller's token: external `cancel` firing still
        // propagates, but dropping the stream (which fires the DropGuard
        // below) only cancels the child, leaving the caller's token alone.
        let internal = cancel.child_token();
        let token_for_loop = internal.clone();

        // Spawn and bind the DropGuard eagerly — not inside the
        // `async_stream::stream!` body — so that dropping the returned
        // stream before it is ever polled still fires the guard and
        // cancels the spawned loop.
        let (tx, mut rx) = mpsc::channel::<RunEvent>(EVENT_CHANNEL_CAPACITY);
        let tx = RunEmitter::new(tx);
        tokio::spawn(cloned.main_loop(tx, agent, thread, token_for_loop));
        let guard = internal.drop_guard();

        let stream = async_stream::stream! {
          // Reference `guard` inside the generator so the macro captures
          // it by move into the generator's state. The guard then lives
          // for the lifetime of the returned stream; dropping the stream
          // drops the guard and cancels the internal token.
          let _guard = guard;

          while let Some(event) = rx.recv().await {
            yield event
          }
        };
        Box::pin(stream)
    }

    async fn main_loop(
        self,
        tx: RunEmitter,
        agent: Agent,
        mut thread: Vec<Arc<Message>>,
        cancel: CancellationToken,
    ) {
        let _ = tx.send(AgentEvent::TurnStart).await;

        loop {
            // Top-of-loop checkpoint: short-circuit before building the
            // next request if cancellation already fired (e.g. during the
            // previous tool-call phase).
            if cancel.is_cancelled() {
                let _ = tx.send(AgentEvent::Cancelled).await;
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
            loop {
                tokio::select! {
                    biased;
                    _ = cancel.cancelled() => {
                        // Drop the model stream — this drops the
                        // underlying reqwest future and aborts the
                        // in-flight HTTP call.
                        drop(model_stream);
                        let _ = tx.send(AgentEvent::Cancelled).await;
                        return;
                    }
                    next = model_stream.next() => {
                        let Some(chunk) = next else { break };
                        match chunk {
                            Ok(ModelStreamChunk::Thinking(t)) => {
                                let _ = tx.send(AgentEvent::ThinkingDelta(t)).await;
                            }
                            Ok(ModelStreamChunk::TextDelta(t)) => {
                                reply.push_str(&t);
                                let _ = tx.send(AgentEvent::TextDelta(t)).await;
                            }
                            Ok(ModelStreamChunk::ToolCall(call)) => {
                                tool_calls.push(Arc::new(call));
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
                }
            }

            if tool_calls.is_empty() {
                if !reply.is_empty() {
                    thread.push(Arc::new(Message::assistant(reply)));
                }
                let _ = tx.send(AgentEvent::TurnFinish { thread }).await;
                return;
            }

            self.handle_tool_calls(&tx, tool_calls, &mut thread, &cancel)
                .await;

            // The tool-call phase races every per-call future against
            // `cancel`; if cancellation fired during that phase, emit the
            // terminal event here (rather than waiting for the next
            // top-of-loop check, which would build a request we'd
            // immediately throw away).
            if cancel.is_cancelled() {
                let _ = tx.send(AgentEvent::Cancelled).await;
                return;
            }
        }
    }

    async fn handle_tool_calls(
        &self,
        tx: &RunEmitter,
        tool_calls: Vec<Arc<ToolCall>>,
        thread: &mut Vec<Arc<Message>>,
        cancel: &CancellationToken,
    ) {
        thread.push(Arc::new(Message::tool_calls(tool_calls.clone())));
        let tool_futures = tool_calls.into_iter().map(|call| {
            info!("Invoking tool '{}' with args '{:?}'", call.name, call.args);
            let cancel = cancel.clone();
            async move {
                let (resolve_tx, resolve_rx) = oneshot::channel();
                let _ = tx
                    .send(AgentEvent::ToolCall(ToolCallRequest::new(
                        call.clone(),
                        cancel.clone(),
                        resolve_tx,
                    )))
                    .await;

                let result = tokio::select! {
                    biased;
                    _ = cancel.cancelled() => {
                        return (call, Value::from("Cancelled"));
                    }
                    res = resolve_rx => res.unwrap_or(Value::from(format!("Tool call {} failed", call.name))),
                };

                info!("Tool '{}' responded with result '{:?}'", call.name, result);
                (call, result)
            }
        });

        // Run all calls concurrently. `join_all` preserves input order in the
        // returned Vec, so tool-result messages are appended in the same order
        // the model requested them — even though events may interleave.
        let results = join_all(tool_futures).await;
        for (call, result) in results {
            thread.push(Arc::new(Message::tool_result(call, result)));
        }
    }
}

#[cfg(test)]
mod tests;
