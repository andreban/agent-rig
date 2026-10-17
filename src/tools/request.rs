// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

use crate::model::ToolCall;
use crate::tools::ToolResult;

/// A request emitted by [`AgentRunner`](crate::runner::AgentRunner) asking the
/// consumer to execute a tool call and resolve its result.
#[derive(Debug)]
pub struct ToolCallRequest {
    /// The tool call requested by the model (`id`, `name`, `args`).
    pub details: Arc<ToolCall>,
    /// Cancellation token that fires if the run is cancelled while the tool is executing.
    pub cancellation_token: CancellationToken,
    resolver: oneshot::Sender<ToolResult>,
}

impl ToolCallRequest {
    /// Constructs a new `ToolCallRequest` wired to `resolver`.
    pub fn new(
        tool_call: Arc<ToolCall>,
        cancellation_token: CancellationToken,
        resolver: oneshot::Sender<ToolResult>,
    ) -> Self {
        Self {
            details: tool_call,
            cancellation_token,
            resolver,
        }
    }

    /// Sends the tool result back to the runner, resuming the agentic loop.
    ///
    /// Accepts a [`ToolResult`] directly or any type that converts into one
    /// (such as [`ToolCallResult`](crate::runner::ToolCallResult) or
    /// [`serde_json::Value`]).
    pub fn resolve(self, result: impl Into<ToolResult>) {
        let _ = self.resolver.send(result.into());
    }
}
