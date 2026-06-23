// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use async_trait::async_trait;
use futures_util::StreamExt;
use serde_json::Value;
use tokio_util::sync::CancellationToken;
use tracing::instrument;

use crate::{
    agent::Agent,
    model::Message,
    runner::{AgentEvent, AgentRunner},
    tools::{
        Tool,
        tool::{ToolDefinition, ToolResult},
    },
};

/// Wraps a child [`Agent`] (plus its [`AgentRunner`]) so it can be invoked
/// as a tool by a parent agent.
///
/// Register one with
/// [`ToolRegistry::register`](crate::tools::ToolRegistry::register), like any
/// other [`Tool`].
/// When the parent model calls this tool, the runner serialises the JSON
/// arguments into a single user message and runs the child agent against its
/// own runner. The child's stream is consumed internally and its accumulated
/// `TextDelta` output becomes the tool result; the child's events are not
/// forwarded to the parent stream.
pub struct AgentTool {
    definition: ToolDefinition,
    agent: Agent,
    runner: AgentRunner,
}

impl AgentTool {
    /// Builds an `AgentTool` from the public tool definition, the child agent,
    /// and the runner that will execute it.
    pub fn new(definition: ToolDefinition, agent: Agent, runner: AgentRunner) -> Self {
        Self {
            definition,
            agent,
            runner,
        }
    }

    pub fn name(&self) -> &str {
        self.agent.name()
    }
}

#[async_trait]
impl Tool for AgentTool {
    /// The [`ToolDefinition`] this child agent exposes to the parent model.
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    /// Invokes the child agent with the proposal and consumes its event stream.
    ///
    /// `proposal` is the resolved tool-call JSON (the default
    /// [`Tool::propose`] passes the model's arguments through unchanged); it is
    /// serialized to JSON and passed as the user message of the new run. The
    /// child run is consumed internally and its accumulated text is returned as
    /// the tool result; the child's events are not forwarded to the parent
    /// stream.
    ///
    /// `cancel` is propagated into the child run via
    /// [`AgentRunner::run_with_cancellation`], so cancelling the parent run
    /// cancels every nested agent in the tree.
    #[instrument(skip(self, proposal, cancel), fields(tool = self.definition.name))]
    async fn apply(&self, proposal: Value, cancel: CancellationToken) -> ToolResult {
        let input = match serde_json::to_string(&proposal) {
            Ok(input) => input,
            Err(e) => return ToolResult::Err(e.to_string().into()),
        };

        let mut result = String::new();
        let mut stream =
            self.runner
                .run_with_cancellation(&self.agent, vec![Arc::new(Message::user(input))], cancel);
        while let Some(next) = stream.next().await {
            if let AgentEvent::TextDelta(text) = &next.agent_event {
                result += text;
            }
            match next.agent_event {
                AgentEvent::TurnStart | AgentEvent::TurnFinish { .. } | AgentEvent::Cancelled => {}
                AgentEvent::Error(e) => return ToolResult::Err(e.to_string().into()),
                _ => {}
            }
        }
        ToolResult::Ok(result.into())
    }
}

#[cfg(test)]
mod tests;
