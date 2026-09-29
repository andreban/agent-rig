// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use async_trait::async_trait;
use futures_util::{StreamExt, stream::FuturesUnordered};
use serde_json::Value;
use tokio_util::sync::CancellationToken;
use tracing::instrument;

use crate::{
    agent::Agent,
    model::{LlmModel, Message, MessageContent, MessageList, Role, TokenUsage, ToolCall},
    runner::{AgentEvent, AgentRunner, ToolCallResult},
    tools::{
        Tool, ToolCallRequest, ToolRegistry,
        tool::{ToolDefinition, ToolResult},
    },
};

/// Callback invoked with the [`TokenUsage`] of each of the child's model calls.
type UsageObserver = Box<dyn Fn(&TokenUsage) + Send + Sync>;

/// Wraps a child [`Agent`] (plus its [`AgentRunner`] and tools) so it can be
/// invoked as a tool by a parent agent.
///
/// Register one with
/// [`ToolRegistry::register`](crate::tools::ToolRegistry::register), like any
/// other [`Tool`].
/// When the parent model calls this tool, the JSON arguments are serialised
/// into a single user message and the child agent runs against its own
/// runner. `AgentTool` drives the child's stream internally: it executes the
/// child's tool calls against the child's [`ToolRegistry`], reports the
/// child's token usage to the [`on_usage`](Self::on_usage) observer, and
/// returns the child's final reply as the tool result. The child's events
/// are not forwarded to the parent stream.
///
/// Use [`new`](Self::new) for a child without tools and
/// [`with_tools`](Self::with_tools) for a child that calls tools of its own.
///
/// # Examples
///
/// ```no_run
/// use std::sync::Arc;
/// use agent_rig::Agent;
/// use agent_rig::model::LlmModel;
/// use agent_rig::tools::{AgentTool, ToolDefinition, ToolRegistry};
/// use schemars::json_schema;
///
/// # fn example(model: Arc<dyn LlmModel>, child_tools: ToolRegistry) {
/// let agent = Agent::builder()
///     .name("Verifier")
///     .instructions("Check the claim in the `claim` field using your tools.")
///     .build();
/// let verifier = AgentTool::with_tools(
///     ToolDefinition {
///         name: "verify".to_string(),
///         description: "Verifies a claim.".to_string(),
///         parameters: json_schema!({"type": "object"}),
///     },
///     agent,
///     model,
///     Arc::new(child_tools),
/// )
/// .on_usage(|usage| println!("child usage: {usage:?}"));
///
/// let parent_tools = ToolRegistry::new().register(verifier);
/// # }
/// ```
pub struct AgentTool {
    definition: ToolDefinition,
    agent: Agent,
    runner: AgentRunner,
    tools: Arc<ToolRegistry>,
    on_usage: Option<UsageObserver>,
}

impl AgentTool {
    /// Builds an `AgentTool` for a child agent without tools, from the public
    /// tool definition, the child agent, and the runner that will execute it.
    ///
    /// The child's tool calls are resolved against an empty registry, so any
    /// call the child makes resolves as [`ToolCallResult::Unknown`]. Use
    /// [`with_tools`](Self::with_tools) for a child that needs tools.
    pub fn new(definition: ToolDefinition, agent: Agent, runner: AgentRunner) -> Self {
        Self {
            definition,
            agent,
            runner,
            tools: Arc::new(ToolRegistry::new()),
            on_usage: None,
        }
    }

    /// Builds an `AgentTool` for a child agent that calls tools of its own.
    ///
    /// The child runner is built from `model` and `tools.definitions()`, so
    /// the definitions the child model sees always match the tools
    /// `AgentTool` executes. Each child tool call runs with the child run's
    /// cancellation token, and calls from the same turn run concurrently. A
    /// call to a tool that isn't in `tools` resolves as
    /// [`ToolCallResult::Unknown`].
    pub fn with_tools(
        definition: ToolDefinition,
        agent: Agent,
        model: Arc<dyn LlmModel>,
        tools: Arc<ToolRegistry>,
    ) -> Self {
        let runner = AgentRunner::with_tools(model, tools.definitions());
        Self {
            definition,
            agent,
            runner,
            tools,
            on_usage: None,
        }
    }

    /// Sets an observer called with the [`TokenUsage`] of each of the child's
    /// model calls, as the child reports it.
    ///
    /// Use it to count the child's tokens toward the parent run's usage,
    /// cost, or limits. Replaces any previously set observer.
    pub fn on_usage(mut self, observer: impl Fn(&TokenUsage) + Send + Sync + 'static) -> Self {
        self.on_usage = Some(Box::new(observer));
        self
    }

    /// Returns the child agent's name.
    pub fn name(&self) -> &str {
        self.agent.name()
    }

    /// Runs one of the child's tool calls and resolves its request.
    async fn execute(&self, request: ToolCallRequest) {
        let result: Value = match self.tools.get(&request.details.name) {
            Some(tool) => tool
                .call(request.details.clone(), request.cancellation_token.clone())
                .await
                .into(),
            None => ToolCallResult::Unknown.into(),
        };
        request.resolve(result);
    }
}

/// Returns the text of the last assistant message in `thread`, or an empty
/// string when the last assistant message isn't text.
fn final_reply(thread: &MessageList) -> String {
    thread
        .iter()
        .rev()
        .find(|message| message.role == Role::Assistant)
        .and_then(|message| match &message.content {
            MessageContent::Text(text) => Some(text.clone()),
            _ => None,
        })
        .unwrap_or_default()
}

#[async_trait]
impl Tool for AgentTool {
    /// The [`ToolDefinition`] this child agent exposes to the parent model.
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    /// Invokes the child agent with the call's arguments and drives its event
    /// stream to completion.
    ///
    /// The model's `tool_call.args` are serialized to JSON and passed as the
    /// user message of the new run. While the child runs, its
    /// [`AgentEvent::ToolCall`] requests are executed concurrently against
    /// the child's [`ToolRegistry`] and its [`AgentEvent::Usage`] events are
    /// passed to the [`on_usage`](AgentTool::on_usage) observer. The child's
    /// events are not forwarded to the parent stream.
    ///
    /// Returns [`ToolResult::Ok`] with the text of the last assistant message
    /// in the child's final thread. Returns [`ToolResult::Err`] if the child
    /// run errors, is cancelled, or ends without finishing.
    ///
    /// `cancel` is propagated into the child run via
    /// [`AgentRunner::run_with_cancellation`], so cancelling the parent run
    /// cancels every nested agent in the tree.
    #[instrument(skip(self, tool_call, cancel), fields(tool = self.definition.name))]
    async fn call(&self, tool_call: Arc<ToolCall>, cancel: CancellationToken) -> ToolResult {
        let input = match serde_json::to_string(&tool_call.args) {
            Ok(input) => input,
            Err(e) => return ToolResult::Err(e.to_string().into()),
        };

        let mut stream = self.runner.run_with_cancellation(
            &self.agent,
            vec![Message::user(input)].into(),
            cancel,
        );
        // The runner waits for all of a turn's calls before emitting more
        // events, so in-flight calls are polled alongside the stream.
        let mut in_flight = FuturesUnordered::new();
        loop {
            tokio::select! {
                Some(()) = in_flight.next(), if !in_flight.is_empty() => {}
                next = stream.next() => {
                    let Some(next) = next else {
                        return ToolResult::error("child run ended without a final reply");
                    };
                    match next.agent_event {
                        AgentEvent::ToolCall(request) => in_flight.push(self.execute(request)),
                        AgentEvent::Usage(usage) => {
                            if let Some(on_usage) = &self.on_usage {
                                on_usage(&usage);
                            }
                        }
                        AgentEvent::TurnFinish { thread } => {
                            return ToolResult::ok(final_reply(&thread));
                        }
                        AgentEvent::Error(e) => return ToolResult::error(e.to_string()),
                        AgentEvent::Cancelled => {
                            return ToolResult::error("child run was cancelled");
                        }
                        AgentEvent::TurnStart
                        | AgentEvent::TextDelta(_)
                        | AgentEvent::ThinkingDelta(_) => {}
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
