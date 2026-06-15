// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::error::Error;
use crate::model::{MessageContent, ModelResponse, Role, TokenUsage};
use crate::tools::{ProgressReporter, Tool};
use async_trait::async_trait;
use futures_util::StreamExt;
use schemars::json_schema;
use serde_json::{Value, json};
use std::sync::Mutex;
use tokio_util::sync::CancellationToken;

/// Model that returns scripted [`ModelResponse`]s in order and records the
/// request it was called with each turn. Used to drive the runner without
/// hitting a real provider.
struct ScriptedModel {
    responses: Mutex<std::collections::VecDeque<Result<ModelResponse, Error>>>,
    requests: Mutex<Vec<ModelRequest>>,
}

impl ScriptedModel {
    fn new(responses: Vec<Result<ModelResponse, Error>>) -> Arc<Self> {
        Arc::new(Self {
            responses: Mutex::new(responses.into()),
            requests: Mutex::new(Vec::new()),
        })
    }

    fn requests(&self) -> Vec<ModelRequest> {
        self.requests.lock().unwrap().clone()
    }
}

#[async_trait]
impl LlmModel for ScriptedModel {
    async fn generate(&self, request: ModelRequest) -> Result<ModelResponse, Error> {
        self.requests.lock().unwrap().push(request);
        self.responses
            .lock()
            .unwrap()
            .pop_front()
            .expect("ScriptedModel: response queue exhausted")
    }
}

fn final_response(text: &str) -> Result<ModelResponse, Error> {
    Ok(ModelResponse {
        text: Some(text.to_string()),
        tool_calls: vec![],
        thinking: None,
        token_usage: None,
    })
}

fn tool_call_response(
    id: &str,
    name: &str,
    args: serde_json::Value,
) -> Result<ModelResponse, Error> {
    Ok(ModelResponse {
        text: None,
        tool_calls: vec![ToolCall::new(id.into(), name.into(), args)],
        thinking: None,
        token_usage: None,
    })
}

fn agent(name: &str) -> Agent {
    Agent::builder()
        .name(name)
        .instructions("test instructions")
        .build()
}

/// Drives the runner to completion and returns the inner [`AgentEvent`]s.
/// Existing assertions match on `AgentEvent` directly; identity-aware
/// tests use [`collect_run_events`] instead.
async fn collect(runner: &AgentRunner, agent: Agent, prompt: &str) -> Vec<AgentEvent> {
    collect_run_events(runner, agent, prompt)
        .await
        .into_iter()
        .map(|e| e.agent_event)
        .collect()
}

async fn collect_run_events(runner: &AgentRunner, agent: Agent, prompt: &str) -> Vec<RunEvent> {
    let mut stream = runner.run(&agent, vec![Message::user(prompt)]);
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }
    events
}

/// Tool that records every invocation and returns a configurable result.
struct EchoTool {
    definition: ToolDefinition,
    result: Result<serde_json::Value, Error>,
    calls: Arc<Mutex<Vec<serde_json::Value>>>,
}

impl EchoTool {
    fn definition(name: &str) -> ToolDefinition {
        ToolDefinition {
            name: name.to_string(),
            description: "echo".to_string(),
            parameters: json_schema!({"type": "object"}),
        }
    }

    fn ok(name: &'static str) -> (Self, Arc<Mutex<Vec<serde_json::Value>>>) {
        let calls = Arc::new(Mutex::new(Vec::new()));
        (
            Self {
                definition: Self::definition(name),
                result: Ok(json!({"ok": true})),
                calls: calls.clone(),
            },
            calls,
        )
    }

    fn failing(name: &'static str, msg: &str) -> Self {
        Self {
            definition: Self::definition(name),
            result: Err(Error::Agent(msg.to_string())),
            calls: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

#[async_trait]
impl Tool for EchoTool {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn apply(
        &self,
        proposal: serde_json::Value,
        _progress: &dyn ProgressReporter,
        _cancel: CancellationToken,
    ) -> Result<serde_json::Value, Error> {
        self.calls.lock().unwrap().push(proposal);
        self.result.clone()
    }
}

#[tokio::test]
async fn text_only_response_emits_text_delta_and_stops() {
    let model = ScriptedModel::new(vec![final_response("hi there")]);
    let runner = AgentRunner::new(model.clone());

    let events = collect(&runner, agent("Greeter"), "hello").await;
    let text: String = events
        .iter()
        .filter_map(|e| match e {
            AgentEvent::TextDelta(t) => Some(t.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(text, "hi there");
    // Single turn — the runner stops once no tool calls are returned.
    assert_eq!(model.requests().len(), 1);
}

#[tokio::test]
async fn tool_call_then_text_completes_the_loop() {
    let (tool, calls) = EchoTool::ok("echo");
    let registry = Arc::new(ToolRegistry::new().register(tool));
    let model = ScriptedModel::new(vec![
        tool_call_response("c1", "echo", json!({"x": 1})),
        final_response("done"),
    ]);
    let runner = AgentRunner::with_registry(model.clone(), registry);

    let events = collect(&runner, agent("Looper"), "go").await;

    // TurnStart, then ToolCallStart + ToolCallFinish + TextDelta. Both lifecycle events
    // carry the model's call id ("c1") so consumers can correlate them.
    assert!(matches!(events[0], AgentEvent::TurnStart));
    assert!(matches!(
        events[1],
        AgentEvent::ToolCallStart { ref tool_call_id, ref tool_name, .. } if tool_call_id == "c1" && tool_name == "echo"
    ));
    assert!(matches!(
        events[2],
        AgentEvent::ToolCallFinish {
            ref tool_call_id,
            ref tool_name,
            result: ToolCallResult::Ok(_),
        } if tool_call_id == "c1" && tool_name == "echo"
    ));
    assert!(matches!(events[3], AgentEvent::TextDelta(ref t) if t == "done"));

    // The tool was actually invoked, and the second turn sent the tool
    // result back to the model.
    assert_eq!(calls.lock().unwrap().len(), 1);
    let second_request = &model.requests()[1];
    let last_msg = second_request.messages.last().unwrap();
    assert!(matches!(
        &last_msg.content,
        MessageContent::ToolResult { name, .. } if name == "echo"
    ));
}

#[tokio::test]
async fn unknown_tool_produces_synthetic_result_with_no_events() {
    let model = ScriptedModel::new(vec![
        tool_call_response("c1", "nope", json!({})),
        final_response("ok"),
    ]);
    let runner = AgentRunner::new(model.clone());

    let events = collect(&runner, agent("a"), "go").await;

    // Hallucinated tool calls are silent — no ToolCallStart, no ToolCallFinish — but
    // the synthetic tool-result message is still appended to the thread
    // so the assistant turn and the tool-result remain paired.
    assert!(!events.iter().any(|e| matches!(
        e,
        AgentEvent::ToolCallStart { .. } | AgentEvent::ToolCallFinish { .. }
    )));
    let second_request = &model.requests()[1];
    let last_msg = second_request.messages.last().unwrap();
    assert!(matches!(
        &last_msg.content,
        MessageContent::ToolResult { name, .. } if name == "nope"
    ));
}

#[tokio::test]
async fn tool_error_is_reported_via_finished_event() {
    let tool = EchoTool::failing("boom", "kaboom");
    let registry = Arc::new(ToolRegistry::new().register(tool));
    let model = ScriptedModel::new(vec![
        tool_call_response("c1", "boom", json!({})),
        final_response("ok"),
    ]);
    let runner = AgentRunner::with_registry(model, registry);

    let events = collect(&runner, agent("a"), "go").await;
    let finished = events
        .iter()
        .find(|e| matches!(e, AgentEvent::ToolCallFinish { .. }))
        .unwrap();
    assert!(matches!(
        finished,
        AgentEvent::ToolCallFinish {
            result: ToolCallResult::Err(_),
            ..
        }
    ));
}

/// Auth manager whose `requires_authorization` returns a fixed answer. The
/// allow/deny decision now lives with the consumer, which responds to each
/// [`AgentEvent::ApprovalRequest`] — see [`collect_with_approvals`].
struct ScriptedAuth {
    required: bool,
}

impl ScriptedAuth {
    fn new(required: bool) -> Arc<Self> {
        Arc::new(Self { required })
    }
}

#[async_trait]
impl AuthManager for ScriptedAuth {
    fn requires_authorization(&self, _name: &str, _args: &Value) -> bool {
        self.required
    }
}

/// What [`collect_with_approvals`] observed: the non-approval events, plus the
/// name and proposal of every [`AgentEvent::ApprovalRequest`] seen, in order.
struct ApprovalOutcome {
    events: Vec<AgentEvent>,
    approval_names: Vec<String>,
    approval_proposals: Vec<Value>,
}

/// Drives the runner to completion, responding to each `ApprovalRequest` with
/// the next decision from `decisions` (defaulting to allow once the queue is
/// exhausted). Responding consumes the request, so its name and proposal are
/// recorded into the returned [`ApprovalOutcome`] rather than re-emitted.
async fn collect_with_approvals(
    runner: &AgentRunner,
    agent: Agent,
    prompt: &str,
    decisions: Vec<bool>,
) -> ApprovalOutcome {
    let mut decisions: std::collections::VecDeque<bool> = decisions.into();
    let mut outcome = ApprovalOutcome {
        events: Vec::new(),
        approval_names: Vec::new(),
        approval_proposals: Vec::new(),
    };
    let mut stream = runner.run(&agent, vec![Message::user(prompt)]);
    while let Some(event) = stream.next().await {
        match event.agent_event {
            AgentEvent::ApprovalRequest(req) => {
                outcome.approval_names.push(req.tool_name.clone());
                outcome.approval_proposals.push(req.proposal.clone());
                req.respond(decisions.pop_front().unwrap_or(true));
            }
            other => outcome.events.push(other),
        }
    }
    outcome
}

#[tokio::test]
async fn auth_denial_skips_tool_execution() {
    let (tool, calls) = EchoTool::ok("echo");
    let registry = Arc::new(ToolRegistry::new().register(tool));
    let auth = ScriptedAuth::new(true);
    let model = ScriptedModel::new(vec![
        tool_call_response("c1", "echo", json!({})),
        final_response("ok"),
    ]);
    let runner = AgentRunner::with_registry(model, registry).with_auth_manager(auth.clone());

    let outcome = collect_with_approvals(&runner, agent("a"), "go", vec![false]).await;

    // ToolCallStart is emitted before the approval gate, so a denied call still
    // produces ToolCallStart followed by ToolCallFinish(Denied); only the underlying tool
    // is skipped.
    assert!(
        outcome
            .events
            .iter()
            .any(|e| matches!(e, AgentEvent::ToolCallStart { .. }))
    );
    let finished = outcome
        .events
        .iter()
        .find(|e| matches!(e, AgentEvent::ToolCallFinish { .. }))
        .unwrap();
    assert!(matches!(
        finished,
        AgentEvent::ToolCallFinish {
            result: ToolCallResult::Denied,
            ..
        }
    ));
    assert!(calls.lock().unwrap().is_empty());
    assert_eq!(
        outcome.approval_names.as_slice(),
        &["echo".to_string()],
        "the runner requested approval for the echo call"
    );
}

#[tokio::test]
async fn started_reaches_consumer_before_approval_request() {
    // Regression test for #64: the consumer must observe `ToolCallStart`
    // before the `ApprovalRequest` for the same call, so a frontend can
    // correlate the approval prompt with the announced tool call. Because both
    // events now travel the same FIFO stream, this ordering holds structurally
    // — no out-of-band flush barrier needed.
    let (tool, _calls) = EchoTool::ok("echo");
    let registry = Arc::new(ToolRegistry::new().register(tool));
    let auth = ScriptedAuth::new(true);
    let model = ScriptedModel::new(vec![
        tool_call_response("c1", "echo", json!({})),
        final_response("ok"),
    ]);
    let runner = AgentRunner::with_registry(model, registry).with_auth_manager(auth);

    let mut order = Vec::<&str>::new();
    let mut stream = runner.run(&agent("a"), vec![Message::user("go")]);
    while let Some(event) = stream.next().await {
        match event.agent_event {
            AgentEvent::ToolCallStart { .. } => order.push("started"),
            AgentEvent::ApprovalRequest(req) => {
                order.push("approval");
                req.respond(true);
            }
            _ => {}
        }
    }

    let started = order.iter().position(|m| *m == "started");
    let approval = order.iter().position(|m| *m == "approval");
    assert_eq!(
        (started, approval),
        (Some(0), Some(1)),
        "ToolCallStart must reach the consumer before the ApprovalRequest: {order:?}"
    );
}

#[tokio::test]
async fn auth_fast_path_skips_approval() {
    let (tool, calls) = EchoTool::ok("echo");
    let registry = Arc::new(ToolRegistry::new().register(tool));
    // requires_authorization returns false — no ApprovalRequest must be emitted.
    let auth = ScriptedAuth::new(false);
    let model = ScriptedModel::new(vec![
        tool_call_response("c1", "echo", json!({})),
        final_response("ok"),
    ]);
    let runner = AgentRunner::with_registry(model, registry).with_auth_manager(auth.clone());

    let outcome = collect_with_approvals(&runner, agent("a"), "go", vec![]).await;

    assert!(outcome.approval_names.is_empty());
    assert_eq!(calls.lock().unwrap().len(), 1);
}

/// Tool with a custom [`Tool::propose`]: it resolves the raw args into a
/// distinct proposal and records the proposal its `apply` actually receives.
struct ProposingTool {
    definition: ToolDefinition,
    /// The proposal `propose` returns; `Err` makes it fail before auth.
    proposal: Result<Value, Error>,
    applied: Arc<Mutex<Vec<Value>>>,
}

impl ProposingTool {
    fn new(proposal: Result<Value, Error>) -> (Self, Arc<Mutex<Vec<Value>>>) {
        let applied = Arc::new(Mutex::new(Vec::new()));
        (
            Self {
                definition: ToolDefinition {
                    name: "proposing".to_string(),
                    description: "proposing".to_string(),
                    parameters: json_schema!({"type": "object"}),
                },
                proposal,
                applied: applied.clone(),
            },
            applied,
        )
    }
}

#[async_trait]
impl Tool for ProposingTool {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn propose(
        &self,
        _args: &Value,
        _progress: &dyn ProgressReporter,
        _cancel: CancellationToken,
    ) -> Result<Value, Error> {
        self.proposal
            .clone()
            .map_err(|e| Error::Agent(e.to_string()))
    }

    async fn apply(
        &self,
        proposal: Value,
        _progress: &dyn ProgressReporter,
        _cancel: CancellationToken,
    ) -> Result<Value, Error> {
        self.applied.lock().unwrap().push(proposal);
        Ok(json!({"ok": true}))
    }
}

#[tokio::test]
async fn proposal_is_forwarded_to_approval_and_apply() {
    let resolved = json!({"path": "f.rs", "old_text": "a", "new_text": "b"});
    let (tool, applied) = ProposingTool::new(Ok(resolved.clone()));
    let registry = Arc::new(ToolRegistry::new().register(tool));
    let auth = ScriptedAuth::new(true);
    let model = ScriptedModel::new(vec![
        tool_call_response("c1", "proposing", json!({"raw": "args"})),
        final_response("ok"),
    ]);
    let runner = AgentRunner::with_registry(model, registry).with_auth_manager(auth.clone());

    let outcome = collect_with_approvals(&runner, agent("a"), "go", vec![true]).await;

    // The ApprovalRequest carried the resolved proposal, not the raw args ...
    assert_eq!(
        outcome.approval_proposals.as_slice(),
        std::slice::from_ref(&resolved)
    );
    // ... and `apply` ran with the same approved proposal.
    assert_eq!(
        applied.lock().unwrap().as_slice(),
        std::slice::from_ref(&resolved)
    );
}

#[tokio::test]
async fn propose_error_skips_approval_and_apply() {
    let (tool, applied) = ProposingTool::new(Err(Error::Agent("cannot plan".to_string())));
    let registry = Arc::new(ToolRegistry::new().register(tool));
    let auth = ScriptedAuth::new(true);
    let model = ScriptedModel::new(vec![
        tool_call_response("c1", "proposing", json!({})),
        final_response("ok"),
    ]);
    let runner = AgentRunner::with_registry(model, registry).with_auth_manager(auth.clone());

    let outcome = collect_with_approvals(&runner, agent("a"), "go", vec![]).await;

    // A propose failure surfaces as ToolCallFinish(Err) ...
    let finished = outcome
        .events
        .iter()
        .find(|e| matches!(e, AgentEvent::ToolCallFinish { .. }))
        .unwrap();
    assert!(matches!(
        finished,
        AgentEvent::ToolCallFinish {
            result: ToolCallResult::Err(_),
            ..
        }
    ));
    // ... without ever requesting approval or running `apply`.
    assert!(outcome.approval_names.is_empty());
    assert!(applied.lock().unwrap().is_empty());
}

#[tokio::test]
async fn thinking_chunks_are_forwarded() {
    let model = ScriptedModel::new(vec![Ok(ModelResponse {
        text: Some("answer".into()),
        tool_calls: vec![],
        thinking: Some("reasoning".into()),
        token_usage: None,
    })]);
    let runner = AgentRunner::new(model);

    let events = collect(&runner, agent("a"), "q").await;
    let kinds: Vec<&'static str> = events
        .iter()
        .map(|e| match e {
            AgentEvent::ThinkingDelta(_) => "thinking",
            AgentEvent::TextDelta(_) => "text",
            AgentEvent::ToolCallStart { .. } => "started",
            AgentEvent::ToolCallUpdate { .. } => "updated",
            AgentEvent::ToolCallFinish { .. } => "finished",
            AgentEvent::Usage(_) => "usage",
            AgentEvent::Cancelled => "cancelled",
            AgentEvent::Error(_) => "error",
            AgentEvent::TurnStart => "start_turn",
            AgentEvent::TurnFinish { .. } => "end_turn",
            AgentEvent::ApprovalRequest(_) => "approval_request",
        })
        .collect();
    // start_turn fires first, then default `generate_stream` yields thinking
    // before text, then end_turn.
    assert_eq!(kinds, vec!["start_turn", "thinking", "text", "end_turn"]);
}

#[tokio::test]
async fn token_usage_is_forwarded_as_agent_event() {
    let usage = TokenUsage {
        input_tokens: Some(11),
        output_tokens: Some(22),
        cached_input_tokens: Some(3),
        thinking_tokens: None,
        tool_use_prompt_tokens: None,
    };
    let model = ScriptedModel::new(vec![Ok(ModelResponse {
        text: Some("hello".into()),
        tool_calls: vec![],
        thinking: None,
        token_usage: Some(usage.clone()),
    })]);
    let runner = AgentRunner::new(model);

    let events = collect(&runner, agent("a"), "q").await;
    let usages: Vec<&TokenUsage> = events
        .iter()
        .filter_map(|e| match e {
            AgentEvent::Usage(u) => Some(u),
            _ => None,
        })
        .collect();
    assert_eq!(usages.len(), 1);
    assert_eq!(usages[0], &usage);
}

#[tokio::test]
async fn one_usage_event_per_model_call() {
    let (tool, _) = EchoTool::ok("echo");
    let registry = Arc::new(ToolRegistry::new().register(tool));
    let first_usage = TokenUsage {
        input_tokens: Some(10),
        output_tokens: Some(5),
        ..Default::default()
    };
    let second_usage = TokenUsage {
        input_tokens: Some(20),
        output_tokens: Some(8),
        ..Default::default()
    };
    let model = ScriptedModel::new(vec![
        Ok(ModelResponse {
            text: None,
            tool_calls: vec![ToolCall::new("c1".into(), "echo".into(), json!({"x": 1}))],
            thinking: None,
            token_usage: Some(first_usage.clone()),
        }),
        Ok(ModelResponse {
            text: Some("done".into()),
            tool_calls: vec![],
            thinking: None,
            token_usage: Some(second_usage.clone()),
        }),
    ]);
    let runner = AgentRunner::with_registry(model, registry);

    let events = collect(&runner, agent("a"), "go").await;
    let usages: Vec<&TokenUsage> = events
        .iter()
        .filter_map(|e| match e {
            AgentEvent::Usage(u) => Some(u),
            _ => None,
        })
        .collect();
    assert_eq!(usages, vec![&first_usage, &second_usage]);
}

#[tokio::test]
async fn no_usage_event_when_provider_does_not_report() {
    let model = ScriptedModel::new(vec![Ok(ModelResponse {
        text: Some("hi".into()),
        tool_calls: vec![],
        thinking: None,
        token_usage: None,
    })]);
    let runner = AgentRunner::new(model);

    let events = collect(&runner, agent("a"), "q").await;
    assert!(!events.iter().any(|e| matches!(e, AgentEvent::Usage(_))));
}

#[tokio::test]
async fn model_error_is_emitted_and_stops_the_loop() {
    let model = ScriptedModel::new(vec![Err(Error::Provider("boom".into()))]);
    let runner = AgentRunner::new(model.clone());

    let events = collect(&runner, agent("a"), "q").await;
    assert!(matches!(events.last(), Some(AgentEvent::Error(_))));
    // The runner must not keep calling the model after an error.
    assert_eq!(model.requests().len(), 1);
}

#[tokio::test]
async fn parallel_tool_results_are_paired_in_request_order() {
    let (tool, _) = EchoTool::ok("echo");
    let registry = Arc::new(ToolRegistry::new().register(tool));
    let model = ScriptedModel::new(vec![
        Ok(ModelResponse {
            text: None,
            tool_calls: vec![
                ToolCall::new("c1".into(), "echo".into(), json!({"i": 1})),
                ToolCall::new("c2".into(), "echo".into(), json!({"i": 2})),
                ToolCall::new("c3".into(), "echo".into(), json!({"i": 3})),
            ],
            thinking: None,
            token_usage: None,
        }),
        final_response("done"),
    ]);
    let runner = AgentRunner::with_registry(model.clone(), registry);

    let _ = collect(&runner, agent("a"), "go").await;

    // Second turn must contain three tool results in the same order the
    // model issued the calls, regardless of which finished first.
    let second = &model.requests()[1];
    let ids: Vec<&str> = second
        .messages
        .iter()
        .filter_map(|m| match &m.content {
            MessageContent::ToolResult { id, .. } => Some(id.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(ids, vec!["c1", "c2", "c3"]);
}

/// Parks forever in `generate`, notifying `dropped` when the generate
/// future is dropped. Used to verify that cancellation actually drops the
/// in-flight model future rather than just stopping at the next checkpoint.
struct PendingModel {
    dropped: Arc<tokio::sync::Notify>,
}

#[async_trait]
impl LlmModel for PendingModel {
    async fn generate(&self, _request: ModelRequest) -> Result<ModelResponse, Error> {
        struct NotifyOnDrop(Arc<tokio::sync::Notify>);
        impl Drop for NotifyOnDrop {
            fn drop(&mut self) {
                // `notify_one` stores a permit when no waiter is registered,
                // so the test's later `.notified().await` is robust to the
                // ordering of when the future is first polled.
                self.0.notify_one();
            }
        }
        let _guard = NotifyOnDrop(self.dropped.clone());
        std::future::pending::<Result<ModelResponse, Error>>().await
    }
}

/// Tool that captures its `cancel` token (so the test can inspect it),
/// signals it has started, then parks on `cancel.cancelled().await`.
struct CancellableTool {
    definition: ToolDefinition,
    started: Arc<tokio::sync::Notify>,
    captured: Arc<Mutex<Option<CancellationToken>>>,
}

#[async_trait]
impl Tool for CancellableTool {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn apply(
        &self,
        _proposal: Value,
        _progress: &dyn ProgressReporter,
        cancel: CancellationToken,
    ) -> Result<Value, Error> {
        *self.captured.lock().unwrap() = Some(cancel.clone());
        self.started.notify_one();
        cancel.cancelled().await;
        Ok(json!({"ran": true}))
    }
}

#[tokio::test]
async fn dropping_stream_drops_inflight_model_future() {
    let dropped = Arc::new(tokio::sync::Notify::new());
    let model = Arc::new(PendingModel {
        dropped: dropped.clone(),
    });
    let runner = AgentRunner::new(model);

    let notified = dropped.notified();
    let stream = runner.run(&agent("a"), vec![Message::user("hi")]);

    // Give the spawned task a moment to enter `generate`.
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    drop(stream);

    tokio::time::timeout(std::time::Duration::from_secs(1), notified)
        .await
        .expect("dropping the stream must drop the in-flight model future");
}

#[tokio::test]
async fn external_cancellation_emits_cancelled_and_ends_stream() {
    let dropped = Arc::new(tokio::sync::Notify::new());
    let model = Arc::new(PendingModel {
        dropped: dropped.clone(),
    });
    let runner = AgentRunner::new(model);
    let cancel = CancellationToken::new();

    let mut stream =
        runner.run_with_cancellation(&agent("a"), vec![Message::user("hi")], cancel.clone());

    let consumer = tokio::spawn(async move {
        let mut events = Vec::new();
        while let Some(e) = stream.next().await {
            events.push(e.agent_event);
        }
        events
    });

    // Park briefly so the runner is awaiting `model_stream.next()` before
    // we cancel — otherwise the top-of-loop check might short-circuit
    // without ever exercising the in-flight cancel path.
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    cancel.cancel();

    let events = tokio::time::timeout(std::time::Duration::from_secs(1), consumer)
        .await
        .expect("stream should end after cancellation")
        .expect("consumer task should not panic");

    assert!(
        matches!(events.last(), Some(AgentEvent::Cancelled)),
        "expected terminal Cancelled, got {events:?}"
    );
}

#[tokio::test]
async fn external_token_is_not_cancelled_when_stream_is_dropped() {
    let dropped = Arc::new(tokio::sync::Notify::new());
    let model = Arc::new(PendingModel {
        dropped: dropped.clone(),
    });
    let runner = AgentRunner::new(model);
    let external = CancellationToken::new();

    let stream =
        runner.run_with_cancellation(&agent("a"), vec![Message::user("hi")], external.clone());
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    drop(stream);

    // The spawned task's child token fires (the model future was dropped),
    // but the caller's token must not — they may be sharing it with other
    // tasks.
    tokio::time::timeout(std::time::Duration::from_secs(1), dropped.notified())
        .await
        .expect("the model future is dropped via the internal child token");
    assert!(
        !external.is_cancelled(),
        "external token must not be cancelled when the stream is dropped"
    );
}

#[tokio::test]
async fn cancellation_during_tool_phase_skips_finished_and_emits_cancelled() {
    let started = Arc::new(tokio::sync::Notify::new());
    let captured: Arc<Mutex<Option<CancellationToken>>> = Arc::new(Mutex::new(None));
    let tool = CancellableTool {
        definition: ToolDefinition {
            name: "slow".to_string(),
            description: "cancellable".to_string(),
            parameters: json_schema!({"type": "object"}),
        },
        started: started.clone(),
        captured: captured.clone(),
    };
    let registry = Arc::new(ToolRegistry::new().register(tool));
    let model = ScriptedModel::new(vec![
        tool_call_response("c1", "slow", json!({})),
        final_response("never reached"),
    ]);
    let runner = AgentRunner::with_registry(model, registry);
    let cancel = CancellationToken::new();

    let mut stream =
        runner.run_with_cancellation(&agent("a"), vec![Message::user("go")], cancel.clone());

    let notified = started.notified();
    let consumer = tokio::spawn(async move {
        let mut events = Vec::new();
        while let Some(e) = stream.next().await {
            events.push(e.agent_event);
        }
        events
    });

    // Wait until the tool has captured the token and is parked.
    tokio::time::timeout(std::time::Duration::from_secs(1), notified)
        .await
        .expect("tool should reach its parked state");
    cancel.cancel();

    let events = tokio::time::timeout(std::time::Duration::from_secs(1), consumer)
        .await
        .expect("stream should end")
        .expect("consumer task should not panic");

    assert!(
        events.iter().any(
            |e| matches!(e, AgentEvent::ToolCallStart { tool_name, .. } if tool_name == "slow")
        ),
        "ToolCallStart must have fired before cancellation: {events:?}"
    );
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, AgentEvent::ToolCallFinish { .. })),
        "ToolCallFinish must NOT fire on cancellation: {events:?}"
    );
    assert!(
        matches!(events.last(), Some(AgentEvent::Cancelled)),
        "terminal event must be Cancelled: {events:?}"
    );

    let captured = captured.lock().unwrap();
    let token = captured
        .as_ref()
        .expect("tool must have captured its cancel token");
    assert!(
        token.is_cancelled(),
        "tool's CancellationToken must fire when the run is cancelled"
    );
}

#[tokio::test]
async fn nested_agent_tool_propagates_cancellation() {
    use crate::tools::AgentTool;

    // Child runs against a PendingModel — it will park indefinitely
    // unless cancelled.
    let child_dropped = Arc::new(tokio::sync::Notify::new());
    let child_model: Arc<dyn LlmModel> = Arc::new(PendingModel {
        dropped: child_dropped.clone(),
    });
    let child_runner = AgentRunner::new(child_model);
    let child_agent = Agent::builder()
        .name("Child")
        .instructions("test instructions")
        .build();
    let agent_tool = AgentTool::new(
        ToolDefinition {
            name: "delegate".to_string(),
            description: "delegate".to_string(),
            parameters: json_schema!({"type": "object"}),
        },
        child_agent,
        child_runner,
    );

    let parent_registry = Arc::new(ToolRegistry::new().register(agent_tool));
    let parent_model = ScriptedModel::new(vec![tool_call_response(
        "c1",
        "delegate",
        json!({"task": "anything"}),
    )]);
    let parent_runner = AgentRunner::with_registry(parent_model, parent_registry);
    let cancel = CancellationToken::new();

    let child_notified = child_dropped.notified();
    let mut stream = parent_runner.run_with_cancellation(
        &agent("Parent"),
        vec![Message::user("delegate")],
        cancel.clone(),
    );

    let consumer = tokio::spawn(async move {
        let mut events = Vec::new();
        while let Some(e) = stream.next().await {
            events.push(e.agent_event);
        }
        events
    });

    // Park briefly so the child's model future is in flight, then cancel
    // from the parent's external token.
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    cancel.cancel();

    tokio::time::timeout(std::time::Duration::from_secs(1), child_notified)
        .await
        .expect("cancelling the parent must drop the child's in-flight model future");
    let _ = tokio::time::timeout(std::time::Duration::from_secs(1), consumer).await;
}

#[tokio::test]
async fn user_message_is_first_in_initial_request() {
    let model = ScriptedModel::new(vec![final_response("hi")]);
    let runner = AgentRunner::new(model.clone());

    let _ = collect(&runner, agent("a"), "hello there").await;

    let first = &model.requests()[0];
    let first_msg = &first.messages[0];
    assert_eq!(first_msg.role, Role::User);
    assert!(matches!(
        &first_msg.content,
        MessageContent::Text(t) if t == "hello there"
    ));
    assert_eq!(first.system.as_deref(), Some("test instructions"));
}
