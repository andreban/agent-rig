// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

//! Authorization policy for tool calls.

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::oneshot;

/// A request, carried on the agent's own event stream, for the consumer to
/// approve or deny a tool call before it runs.
///
/// When an [`AuthManager`] reports that a call
/// [`requires_authorization`](AuthManager::requires_authorization), the runner
/// emits this as [`AgentEvent::ApprovalRequest`](crate::runner::AgentEvent::ApprovalRequest)
/// and then blocks the call until the consumer answers. Because the request
/// travels the same FIFO stream as
/// [`ToolCallStarted`](crate::runner::AgentEvent::ToolCallStarted), the
/// consumer is guaranteed to have already seen the `ToolCallStarted` for the
/// same [`tool_id`](Self::tool_id) — the two can be correlated by id without
/// any out-of-band coordination.
///
/// The consumer **must** consume the request, by calling [`respond`](Self::respond)
/// or by dropping it. Dropping (for example, abandoning the stream) is treated
/// as a denial. Holding the request without answering blocks the tool call
/// indefinitely.
#[derive(Debug)]
pub struct ApprovalRequest {
    /// The tool call's identifier — the same id reported on
    /// [`ToolCallStarted`](crate::runner::AgentEvent::ToolCallStarted) and
    /// `ToolCallFinished`. Use it to correlate the prompt with the announced
    /// call.
    pub tool_id: String,
    /// The name of the tool the model wants to invoke.
    pub name: String,
    /// The raw JSON arguments the model requested.
    pub args: Value,
    /// What the tool resolved [`args`](Self::args) into via
    /// [`Tool::propose`](crate::tools::Tool::propose): the concrete thing that
    /// will happen if approved (for an edit tool, the path plus old and new
    /// contents — enough to render a diff). The approved proposal is handed
    /// verbatim to [`Tool::apply`](crate::tools::Tool::apply), so what the
    /// prompt shows is exactly what runs. For a tool that does no planning the
    /// proposal equals `args`.
    pub proposal: Value,
    resolver: oneshot::Sender<bool>,
}

impl ApprovalRequest {
    /// Builds a request whose decision is delivered through `resolver`.
    /// Constructed by the runner; consumers receive ready-made requests on the
    /// event stream.
    pub(crate) fn new(
        tool_id: String,
        name: String,
        args: Value,
        proposal: Value,
        resolver: oneshot::Sender<bool>,
    ) -> Self {
        Self {
            tool_id,
            name,
            args,
            proposal,
            resolver,
        }
    }

    /// Answers the request: `true` allows the tool call to run, `false` denies
    /// it (the runner reports denial via
    /// [`ToolCallResult::Denied`](crate::runner::ToolCallResult::Denied), with
    /// no accompanying reason). Consumes the request — it can be answered once.
    pub fn respond(self, allowed: bool) {
        let _ = self.resolver.send(allowed);
    }
}

/// Decides which tool calls require the consumer's approval before they run.
///
/// A runner that holds an `AuthManager` consults it for **every** tool call it
/// is about to make. For each call that
/// [`requires_authorization`](AuthManager::requires_authorization) returns
/// `true`, the runner emits an
/// [`AgentEvent::ApprovalRequest`](crate::runner::AgentEvent::ApprovalRequest)
/// on its event stream and waits for the consumer to
/// [`respond`](ApprovalRequest::respond). The allow/deny decision and the
/// machinery to obtain it (CLI prompt, UI dialog, policy file, remote service)
/// live with the consumer, not in this trait — keeping the prompt in-stream is
/// what lets a frontend correlate it with the already-announced
/// [`ToolCallStarted`](crate::runner::AgentEvent::ToolCallStarted).
#[async_trait]
pub trait AuthManager: Send + Sync {
    /// Cheap, synchronous gate: should this call prompt the consumer for
    /// approval?
    ///
    /// Defaults to `true` (always prompt). Override to fast-path calls that
    /// don't need approval — a `HashSet<String>` lookup, an args inspection.
    ///
    /// Must be non-blocking — no I/O, no locks, no awaits.
    fn requires_authorization(&self, _name: &str, _args: &Value) -> bool {
        true
    }
}
