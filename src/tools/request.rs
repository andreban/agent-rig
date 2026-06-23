// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use serde_json::Value;
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

use crate::model::ToolCall;

#[derive(Debug)]
pub struct ToolCallRequest {
    pub details: Arc<ToolCall>,
    pub cancellation_token: CancellationToken,
    resolver: oneshot::Sender<Value>,
}

impl ToolCallRequest {
    pub fn new(
        tool_call: Arc<ToolCall>,
        cancellation_token: CancellationToken,
        resolver: oneshot::Sender<Value>,
    ) -> Self {
        Self {
            details: tool_call,
            cancellation_token,
            resolver,
        }
    }
    pub fn resolve(self, result: impl Into<Value>) {
        let _ = self.resolver.send(result.into());
    }
}
