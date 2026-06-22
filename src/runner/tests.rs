// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::Agent;
use crate::error::Error;
use crate::model::{LlmModel, MessageContent, ModelRequest, ModelResponse, TokenUsage, ToolCall};
use async_trait::async_trait;
use futures_util::StreamExt;
use serde_json::json;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use tokio_util::sync::CancellationToken;

// --- test infrastructure ---

/// Scripted model that returns queued responses and records every request it
/// receives. Returning `Err` from `generate` simulates a provider failure.
struct ScriptedModel {
    responses: Mutex<VecDeque<Result<ModelResponse, Error>>>,
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

fn text(s: &str) -> Result<ModelResponse, Error> {
    Ok(ModelResponse {
        text: Some(s.to_string()),
        tool_calls: vec![],
        thinking: None,
        token_usage: None,
    })
}

fn tool_call(id: &str, name: &str, args: serde_json::Value) -> Result<ModelResponse, Error> {
    Ok(ModelResponse {
        text: None,
        tool_calls: vec![ToolCall::new(id.to_string(), name.to_string(), args)],
        thinking: None,
        token_usage: None,
    })
}

fn agent() -> Agent {
    Agent::builder().name("Test").instructions("noop").build()
}

// --- tests ---

#[tokio::test]
async fn text_only_emits_start_delta_finish() {
    let model = ScriptedModel::new(vec![text("hello")]);
    let runner = AgentRunner::new(model);

    let mut events = vec![];
    let mut stream = runner.run(&agent(), vec![Message::user("hi")]);
    while let Some(ev) = stream.next().await {
        events.push(ev.agent_event);
    }

    assert!(matches!(events[0], AgentEvent::TurnStart));
    assert!(matches!(&events[1], AgentEvent::TextDelta(t) if t == "hello"));
    assert!(matches!(events[2], AgentEvent::TurnFinish { .. }));
    assert_eq!(events.len(), 3);
}

#[tokio::test]
async fn thinking_delta_is_forwarded() {
    let model = ScriptedModel::new(vec![Ok(ModelResponse {
        text: Some("answer".to_string()),
        tool_calls: vec![],
        thinking: Some("reasoning".to_string()),
        token_usage: None,
    })]);
    let runner = AgentRunner::new(model);

    let mut events = vec![];
    let mut stream = runner.run(&agent(), vec![Message::user("hi")]);
    while let Some(ev) = stream.next().await {
        events.push(ev.agent_event);
    }

    // TurnStart, ThinkingDelta, TextDelta, TurnFinish
    assert!(matches!(&events[1], AgentEvent::ThinkingDelta(t) if t == "reasoning"));
    assert!(matches!(&events[2], AgentEvent::TextDelta(t) if t == "answer"));
}

#[tokio::test]
async fn usage_event_is_forwarded() {
    let model = ScriptedModel::new(vec![Ok(ModelResponse {
        text: Some("hi".to_string()),
        tool_calls: vec![],
        thinking: None,
        token_usage: Some(TokenUsage {
            input_tokens: Some(10),
            output_tokens: Some(5),
            ..Default::default()
        }),
    })]);
    let runner = AgentRunner::new(model);

    let mut events = vec![];
    let mut stream = runner.run(&agent(), vec![Message::user("hi")]);
    while let Some(ev) = stream.next().await {
        events.push(ev.agent_event);
    }

    let usage = events.iter().find_map(|e| {
        if let AgentEvent::Usage(u) = e {
            Some(u)
        } else {
            None
        }
    });
    assert!(usage.is_some());
    let u = usage.unwrap();
    assert_eq!(u.input_tokens, Some(10));
    assert_eq!(u.output_tokens, Some(5));
}

#[tokio::test]
async fn model_error_emits_error_event_and_closes_stream() {
    let model = ScriptedModel::new(vec![Err(Error::Provider("boom".to_string()))]);
    let runner = AgentRunner::new(model);

    let mut events = vec![];
    let mut stream = runner.run(&agent(), vec![Message::user("hi")]);
    while let Some(ev) = stream.next().await {
        events.push(ev.agent_event);
    }

    assert!(matches!(events[0], AgentEvent::TurnStart));
    assert!(matches!(&events[1], AgentEvent::Error(Error::Provider(s)) if s == "boom"));
    assert_eq!(events.len(), 2);
}

#[tokio::test]
async fn tool_call_resolved_triggers_second_turn() {
    let model = ScriptedModel::new(vec![
        tool_call("c1", "add", json!({"a": 1, "b": 2})),
        text("result is 3"),
    ]);
    let runner = AgentRunner::new(model.clone());

    let mut text_chunks = vec![];
    let mut stream = runner.run(&agent(), vec![Message::user("what is 1+2?")]);
    while let Some(ev) = stream.next().await {
        match ev.agent_event {
            AgentEvent::ToolCall(call) => call.resolve(json!({"sum": 3})),
            AgentEvent::TextDelta(t) => text_chunks.push(t),
            _ => {}
        }
    }

    assert_eq!(text_chunks, vec!["result is 3"]);
    assert_eq!(model.requests().len(), 2);
}

#[tokio::test]
async fn turn_finish_thread_contains_full_history() {
    let model = ScriptedModel::new(vec![
        tool_call("c1", "greet", json!({"name": "Alice"})),
        text("done"),
    ]);
    let runner = AgentRunner::new(model);

    let user_msg = Message::user("hello");
    let mut finish_thread: Option<Vec<Message>> = None;
    let mut stream = runner.run(&agent(), vec![user_msg]);
    while let Some(ev) = stream.next().await {
        match ev.agent_event {
            AgentEvent::ToolCall(call) => call.resolve(json!({"greeting": "Hello, Alice!"})),
            AgentEvent::TurnFinish { thread } => finish_thread = Some(thread),
            _ => {}
        }
    }

    let thread = finish_thread.unwrap();
    // user → tool_calls → tool_result → assistant
    assert_eq!(thread.len(), 4);
    assert!(matches!(&thread[0].content, MessageContent::Text(t) if t == "hello"));
    assert!(matches!(&thread[1].content, MessageContent::ToolCalls(_)));
    assert!(
        matches!(&thread[2].content, MessageContent::ToolResult { tool_call, .. } if tool_call.name == "greet")
    );
    assert!(matches!(&thread[3].content, MessageContent::Text(t) if t == "done"));
}

#[tokio::test]
async fn multiple_tool_calls_all_resolved_in_one_turn() {
    let model = ScriptedModel::new(vec![
        Ok(ModelResponse {
            text: None,
            tool_calls: vec![
                ToolCall::new("c1".to_string(), "a".to_string(), json!({})),
                ToolCall::new("c2".to_string(), "b".to_string(), json!({})),
            ],
            thinking: None,
            token_usage: None,
        }),
        text("all done"),
    ]);
    let runner = AgentRunner::new(model.clone());

    let mut tool_calls_seen = 0usize;
    let mut stream = runner.run(&agent(), vec![Message::user("do both")]);
    while let Some(ev) = stream.next().await {
        if let AgentEvent::ToolCall(call) = ev.agent_event {
            tool_calls_seen += 1;
            call.resolve(json!({"ok": true}));
        }
    }

    assert_eq!(tool_calls_seen, 2);
    assert_eq!(model.requests().len(), 2);
}

#[tokio::test]
async fn dropped_tool_call_sends_error_string_to_model() {
    let model = ScriptedModel::new(vec![tool_call("c1", "noop", json!({})), text("done")]);
    let runner = AgentRunner::new(model.clone());

    let mut stream = runner.run(&agent(), vec![Message::user("hi")]);
    while let Some(ev) = stream.next().await {
        // Intentionally drop AgentEvent::ToolCall without resolving
        drop(ev);
    }

    let requests = model.requests();
    assert_eq!(requests.len(), 2);
    // The tool result sent to the model on the second request should carry the
    // fallback error string rather than a useful result.
    let tool_result = requests[1]
        .messages
        .iter()
        .find(|m| matches!(&m.content, MessageContent::ToolResult { .. }));
    assert!(tool_result.is_some());
    if let MessageContent::ToolResult { result, .. } = &tool_result.unwrap().content {
        assert!(result.as_str().unwrap_or("").contains("failed"));
    }
}

#[tokio::test]
async fn runner_tools_forwarded_to_model_request() {
    use schemars::json_schema;
    let model = ScriptedModel::new(vec![text("hi")]);
    let tool_def = ToolDefinition {
        name: "my_tool".to_string(),
        description: "desc".to_string(),
        parameters: json_schema!({"type": "object"}),
    };
    let runner = AgentRunner::with_tools(model.clone(), vec![tool_def]);

    let mut stream = runner.run(&agent(), vec![Message::user("hi")]);
    while stream.next().await.is_some() {}

    let requests = model.requests();
    assert_eq!(requests[0].tools.len(), 1);
    assert_eq!(requests[0].tools[0].name, "my_tool");
}

#[tokio::test]
async fn pre_cancelled_token_emits_cancelled_without_calling_model() {
    let model = ScriptedModel::new(vec![]); // no responses queued — must not be called
    let runner = AgentRunner::new(model.clone());

    let cancel = CancellationToken::new();
    cancel.cancel();

    let mut events = vec![];
    let mut stream = runner.run_with_cancellation(&agent(), vec![Message::user("hi")], cancel);
    while let Some(ev) = stream.next().await {
        events.push(ev.agent_event);
    }

    assert!(matches!(events[0], AgentEvent::TurnStart));
    assert!(matches!(events[1], AgentEvent::Cancelled));
    assert_eq!(events.len(), 2);
    assert_eq!(
        model.requests().len(),
        0,
        "model must not be called when pre-cancelled"
    );
}

#[tokio::test]
async fn run_id_is_consistent_across_all_events_in_a_run() {
    let model = ScriptedModel::new(vec![text("hello")]);
    let runner = AgentRunner::new(model);

    let mut run_ids = vec![];
    let mut stream = runner.run(&agent(), vec![Message::user("hi")]);
    while let Some(ev) = stream.next().await {
        run_ids.push(ev.run_id);
    }

    assert!(!run_ids.is_empty());
    let first = run_ids[0];
    assert!(run_ids.iter().all(|id| *id == first));
}
