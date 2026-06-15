// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

use serde_json::Value;
use tokio::sync::oneshot;

/// A request, carried on the agent's own event stream, for the consumer to
/// approve or deny a tool call before it runs.
///
/// When a tool's [`requires_approval`](crate::tools::Tool::requires_approval) returns
/// `true`, the runner emits this as
/// [`AgentEvent::ApprovalRequest`](crate::runner::AgentEvent::ApprovalRequest)
/// and then blocks the call until the consumer answers. Because the request
/// travels the same FIFO stream as
/// [`ToolCallStart`](crate::runner::AgentEvent::ToolCallStart), the
/// consumer is guaranteed to have already seen the `ToolCallStart` for the
/// same [`tool_call_id`](Self::tool_call_id) — the two can be correlated by id without
/// any out-of-band coordination.
///
/// The consumer **must** consume the request, by calling [`respond`](Self::respond)
/// or by dropping it. Dropping (for example, abandoning the stream) is treated
/// as a denial. Holding the request without answering blocks the tool call
/// indefinitely.
#[derive(Debug)]
pub struct ApprovalRequest {
    /// The tool call's identifier — the same id reported on
    /// [`ToolCallStart`](crate::runner::AgentEvent::ToolCallStart) and
    /// [`ToolCallFinish`](crate::runner::AgentEvent::ToolCallFinish). Use it to correlate the prompt with the announced
    /// call.
    pub tool_call_id: String,
    /// The name of the tool the model wants to invoke.
    pub tool_name: String,
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
        tool_call_id: String,
        tool_name: String,
        args: Value,
        proposal: Value,
        resolver: oneshot::Sender<bool>,
    ) -> Self {
        Self {
            tool_call_id,
            tool_name,
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
