// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

use futures_util::StreamExt;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;
use tracing::instrument;

use crate::{
    agent::Agent,
    error::Error,
    model::Message,
    runner::{AgentEvent, AgentRunner, RunEmitter},
    tools::tool::ToolDefinition,
};

/// Wraps a child [`Agent`] (plus its [`AgentRunner`]) so it can be invoked
/// as a tool by a parent agent.
///
/// Register one with
/// [`ToolRegistry::register_agent`](crate::tools::ToolRegistry::register_agent).
/// When the parent model calls this tool, the runner serialises the JSON
/// arguments into a single user message, runs the child agent against its
/// own runner, and forwards the child's [`AgentEvent`]s through to the parent
/// stream. The accumulated `TextDelta` output becomes the tool result.
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
}

impl AgentTool {
    /// The [`ToolDefinition`] this child agent exposes to the parent model.
    pub fn definition(&self) -> ToolDefinition {
        self.definition.clone()
    }

    /// Invokes the child agent with `args` and returns the child's event stream.
    ///
    /// `args` is serialized to JSON and passed as the user message of the new
    /// run. The caller (the parent runner) consumes the stream, forwards each
    /// event, and uses the accumulated text as the tool result.
    ///
    /// `cancel` is propagated into the child run via
    /// [`AgentRunner::run_with_cancellation`], so cancelling the parent run
    /// cancels every nested agent in the tree.
    #[instrument(skip(self, args, cancel), fields(tool = self.definition.name))]
    pub async fn call(
        &self,
        tx: RunEmitter,
        args: serde_json::Value,
        cancel: CancellationToken,
    ) -> Result<Value, Error> {
        let input = serde_json::to_string(&args)
            .map_err(|e| Error::Agent(format!("failed to serialize args: {e}")))?;

        let mut result = String::new();
        let mut stream =
            self.runner
                .run_with_cancellation(&self.agent, vec![Message::user(input)], cancel);
        while let Some(next) = stream.next().await {
            if let AgentEvent::TextDelta(text) = &next.agent_event {
                result += text;
            }
            let _ = tx.send(next.agent_event).await;
        }
        Ok(json!({"output": result }))
    }
}

#[cfg(test)]
mod tests;
