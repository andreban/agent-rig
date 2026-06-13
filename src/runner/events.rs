// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

//! Public event types yielded by the [`AgentRunner`].
//!
//! [`AgentEvent`] is the union of things the runner reports as it drives the
//! agentic loop; [`RunEvent`] tags one of those with the identity of the
//! run that produced it ([`run_id`](RunEvent::run_id), optional
//! [`parent`](RunEvent::parent)). [`ToolCallResult`] is the outcome carried
//! by [`AgentEvent::ToolCallFinished`].
//!
//! [`AgentRunner`]: super::AgentRunner

use serde_json::Value;

use crate::error::Error;
use crate::model::{Message, TokenUsage};

/// Outcome of executing a single tool call.
///
/// Reported back to the model as the tool result on the next turn (via
/// [`From<ToolCallResult> for Value`](#impl-From<ToolCallResult>-for-Value))
/// and surfaced to the consumer inside [`AgentEvent::ToolCallFinished`].
#[derive(Clone, Debug)]
pub enum ToolCallResult {
    /// The tool ran and returned this JSON value.
    Ok(Value),
    /// The tool failed. The error is surfaced to the model as a string.
    Err(Error),
    /// The [`AuthManager`](crate::auth::AuthManager) denied the call; the tool
    /// was not invoked.
    Denied,
    /// The model called a tool that is not registered. No `Started` /
    /// `Finished` events are emitted in this case, but a synthetic
    /// result is still sent back to the model so the assistant turn and
    /// tool-result messages stay paired.
    Unknown,
}

impl From<ToolCallResult> for Value {
    fn from(value: ToolCallResult) -> Self {
        match value {
            ToolCallResult::Denied => Value::from("Tool call denied"),
            ToolCallResult::Ok(result) => result,
            ToolCallResult::Err(error) => Value::from(format!("Tool call error: {error}")),
            ToolCallResult::Unknown => Value::from("Unknown tool"),
        }
    }
}

impl From<Result<Value, Error>> for ToolCallResult {
    fn from(res: Result<Value, Error>) -> Self {
        match res {
            Ok(result) => ToolCallResult::Ok(result),
            Err(error) => ToolCallResult::Err(error),
        }
    }
}

/// An event yielded by [`AgentRunner::run`](super::AgentRunner::run) as the
/// agent loop progresses.
///
/// Variants fall into two groups:
///
/// - Model output: [`ThinkingDelta`](AgentEvent::ThinkingDelta) and
///   [`TextDelta`](AgentEvent::TextDelta) carry chunks as the provider streams
///   them. Concatenating every `TextDelta` reconstructs the final reply.
/// - Tool lifecycle:
///   [`ToolCallStarted`](AgentEvent::ToolCallStarted) fires before a tool
///   runs (after authorization, if any) and
///   [`ToolCallFinished`](AgentEvent::ToolCallFinished) fires once it
///   resolves. Hallucinated tool calls (no matching registry entry) emit
///   *neither* event; see [`ToolCallResult::Unknown`].
/// - [`Usage`](AgentEvent::Usage) reports token counts for one model call.
///   A run that performs `N` model calls produces up to `N` `Usage`
///   events; consumers sum across them to derive per-run totals.
/// - [`StartTurn`](AgentEvent::StartTurn) is emitted as the first event of
///   every run, before any model output.
/// - [`EndTurn`](AgentEvent::EndTurn) is emitted as the last event on normal
///   completion and carries the full conversation thread (including any
///   [`ToolCalls`](crate::model::MessageContent::ToolCalls) and tool-result
///   messages appended during the run). Use it to carry state forward into
///   the next multi-turn prompt.
/// - [`Cancelled`](AgentEvent::Cancelled) and [`Error`](AgentEvent::Error)
///   are terminal: the stream ends after either of them, and they are
///   mutually exclusive with the loop's normal completion (no tool calls
///   in the final model turn).
#[derive(Clone, Debug)]
pub enum AgentEvent {
    /// A registered tool is about to run with these arguments.
    ToolCallStarted {
        /// Provider-assigned call identifier, matching
        /// [`ToolCall::id`](crate::model::ToolCall::id). Use it to correlate
        /// this event with its [`ToolCallFinished`](AgentEvent::ToolCallFinished)
        /// — events from parallel calls in the same turn may interleave.
        tool_id: String,
        /// Name of the tool being invoked.
        name: String,
        /// The JSON arguments the model passed.
        args: serde_json::Value,
        /// The display information for the tool being invoked.
        title: String,
    },
    /// A tool call resolved with [`ToolCallResult`]. Fires after the tool
    /// returns, errors, or is denied.
    ToolCallFinished {
        /// Provider-assigned call identifier, matching the
        /// [`ToolCallStarted`](AgentEvent::ToolCallStarted) `id` (and
        /// [`ToolCall::id`](crate::model::ToolCall::id)). Use it to pair this
        /// event with its `Started` — events from parallel calls in the same
        /// turn may interleave.
        tool_id: String,
        /// Name of the tool that resolved.
        name: String,
        /// Outcome of the call.
        result: ToolCallResult,
    },
    /// A chunk of the model's reasoning/thinking output, if the provider
    /// supports extended thinking.
    ThinkingDelta(String),
    /// A chunk of the model's text output.
    TextDelta(String),
    /// Token counts reported by the provider for one model call.
    ///
    /// Emitted at most once per model call (a run that issues multiple
    /// tool-calling turns produces multiple `Usage` events). Provider
    /// adapters that do not report usage never produce this event.
    Usage(TokenUsage),
    /// The run has begun. Emitted as the first event of every run, before any
    /// model output.
    StartTurn,
    /// The run completed normally (no tool calls in the final model turn).
    ///
    /// Carries the full conversation thread as it stood when the loop exited,
    /// including any tool-call and tool-result messages appended during the
    /// run. Callers that maintain multi-turn state should capture this to
    /// pass as the initial `thread` for the next [`AgentRunner::run`] call.
    ///
    /// Not emitted on [`Cancelled`](AgentEvent::Cancelled) or
    /// [`Error`](AgentEvent::Error) paths.
    ///
    /// [`AgentRunner::run`]: super::AgentRunner::run
    EndTurn {
        /// The conversation thread at the point the loop exited.
        thread: Vec<Message>,
    },
    /// The run was cancelled — either because the consumer dropped the
    /// returned stream, or because an externally supplied
    /// [`CancellationToken`](tokio_util::sync::CancellationToken) fired.
    /// The stream ends after this event.
    ///
    /// Delivery is best-effort: when cancellation is triggered by the
    /// consumer dropping the stream, the receiver is already gone and
    /// the event is silently discarded. Consumers that supply their own
    /// token via
    /// [`AgentRunner::run_with_cancellation`](super::AgentRunner::run_with_cancellation)
    /// and keep draining the stream will observe this event.
    Cancelled,
    /// The provider returned an error. The stream ends after this event.
    Error(crate::error::Error),
}

/// An [`AgentEvent`] tagged with the identity of the run that produced it.
///
/// `run_id` is unique per process; `parent` points at the `run_id` of the
/// run that invoked this one (a sub-agent invocation), or `None` for a
/// root run. For a flat single-run consumer the extra fields can be
/// ignored — destructure or read `event.agent_event`.
#[derive(Debug)]
pub struct RunEvent {
    /// Unique identifier of the run that produced this event.
    pub run_id: usize,
    /// `run_id` of the run that invoked this one. `None` for a root run.
    pub parent: Option<usize>,
    /// The wrapped event.
    pub agent_event: AgentEvent,
}
