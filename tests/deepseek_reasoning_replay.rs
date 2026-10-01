// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

//! Offline tests for the DeepSeek adapter against a mocked API: the model's
//! `reasoning_content` is replayed on later turns, and `reasoning_effort`
//! reaches the request body.

use std::sync::Arc;

use agent_rig::Agent;
use agent_rig::model::{Message, MessageList};
use agent_rig::models::deepseek::{DeepSeekModel, ReasoningEffort};
use agent_rig::runner::{AgentEvent, AgentRunner};
use agent_rig::tools::ToolDefinition;
use futures_util::StreamExt;
use schemars::json_schema;
use serde_json::{Value, json};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Builds an SSE body from a list of chunk deltas, terminated by `[DONE]`.
fn sse(deltas: &[Value]) -> ResponseTemplate {
    let mut body = String::new();
    for delta in deltas {
        let chunk = json!({
            "id": "chunk",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "deepseek-reasoner",
            "choices": [{ "index": 0, "delta": delta }],
        });
        body.push_str(&format!("data: {chunk}\n\n"));
    }
    body.push_str("data: [DONE]\n\n");
    ResponseTemplate::new(200)
        .insert_header("content-type", "text/event-stream")
        .set_body_string(body)
}

fn tool_call_turn(reasoning: &str, id: &str) -> ResponseTemplate {
    sse(&[
        json!({ "role": "assistant", "reasoning_content": reasoning }),
        json!({ "tool_calls": [{
            "index": 0,
            "id": id,
            "type": "function",
            "function": { "name": "add", "arguments": "{\"a\":1,\"b\":2}" },
        }]}),
    ])
}

fn text_turn(reasoning: &str, text: &str) -> ResponseTemplate {
    sse(&[
        json!({ "role": "assistant", "reasoning_content": reasoning }),
        json!({ "content": text }),
    ])
}

/// Mounts `responses` so they are served once each, in order.
async fn mount_in_order(server: &MockServer, responses: Vec<ResponseTemplate>) {
    for response in responses {
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(response)
            .up_to_n_times(1)
            .mount(server)
            .await;
    }
}

/// Bodies of every request the mock server received, in order.
async fn request_bodies(server: &MockServer) -> Vec<Value> {
    server
        .received_requests()
        .await
        .unwrap()
        .iter()
        .map(|r| r.body_json().unwrap())
        .collect()
}

/// Assistant messages of a request body, in order.
fn assistant_messages(body: &Value) -> Vec<Value> {
    body["messages"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|m| m["role"] == "assistant")
        .cloned()
        .collect()
}

fn add_tool() -> ToolDefinition {
    ToolDefinition {
        name: "add".to_string(),
        description: "Adds two integers.".to_string(),
        parameters: json_schema!({
            "type": "object",
            "properties": { "a": { "type": "integer" }, "b": { "type": "integer" } },
            "required": ["a", "b"]
        }),
    }
}

/// Runs the agent on `thread`, resolving every tool call, and returns the
/// final thread.
async fn run(runner: &AgentRunner, thread: MessageList) -> MessageList {
    let agent = Agent::builder()
        .name("test")
        .instructions("Use the add tool.")
        .tool("add")
        .build();
    let mut stream = runner.run(&agent, thread);
    let mut finished = None;
    while let Some(event) = stream.next().await {
        match event.agent_event {
            AgentEvent::ToolCall(call) => call.resolve(json!({ "result": 3 })),
            AgentEvent::TurnFinish { thread } => finished = Some(thread),
            AgentEvent::Error(e) => panic!("run failed: {e}"),
            _ => {}
        }
    }
    finished.expect("run did not finish")
}

#[tokio::test]
async fn reasoning_is_replayed_on_tool_call_and_text_turns() {
    let server = MockServer::start().await;
    mount_in_order(
        &server,
        vec![
            tool_call_turn("first reasoning", "call_1"),
            tool_call_turn("second reasoning", "call_2"),
            text_turn("third reasoning", "The answer is 3."),
            text_turn("fourth reasoning", "Still 3."),
        ],
    )
    .await;

    let model = Arc::new(
        DeepSeekModel::builder("test-key", "deepseek-reasoner")
            .base_url(server.uri())
            .build(),
    );
    let runner = AgentRunner::with_tools(model, vec![add_tool()]);

    let mut thread = run(&runner, vec![Message::user("What is 1+2?")].into()).await;
    thread.push(Message::user("Are you sure?"));
    run(&runner, thread).await;

    let bodies = request_bodies(&server).await;
    assert_eq!(bodies.len(), 4);

    // First request: no assistant history yet.
    assert!(assistant_messages(&bodies[0]).is_empty());

    // Second request replays the first tool-call turn's reasoning verbatim.
    let assistant = assistant_messages(&bodies[1]);
    assert_eq!(assistant.len(), 1);
    assert_eq!(assistant[0]["reasoning_content"], "first reasoning");
    assert_eq!(assistant[0]["tool_calls"][0]["id"], "call_1");

    // Third request replays both tool-call turns.
    let assistant = assistant_messages(&bodies[2]);
    assert_eq!(assistant.len(), 2);
    assert_eq!(assistant[0]["reasoning_content"], "first reasoning");
    assert_eq!(assistant[1]["reasoning_content"], "second reasoning");
    assert_eq!(assistant[1]["tool_calls"][0]["id"], "call_2");

    // Fourth request (next user turn) also replays the assistant text turn.
    let assistant = assistant_messages(&bodies[3]);
    assert_eq!(assistant.len(), 3);
    assert_eq!(assistant[2]["content"], "The answer is 3.");
    assert_eq!(assistant[2]["reasoning_content"], "third reasoning");
}

#[tokio::test]
async fn reasoning_effort_reaches_request_body() {
    let server = MockServer::start().await;
    mount_in_order(&server, vec![text_turn("r", "hi"), text_turn("r", "hi")]).await;

    let with_effort = Arc::new(
        DeepSeekModel::builder("test-key", "deepseek-reasoner")
            .base_url(server.uri())
            .reasoning_effort(ReasoningEffort::Max)
            .build(),
    );
    let default = Arc::new(
        DeepSeekModel::builder("test-key", "deepseek-reasoner")
            .base_url(server.uri())
            .build(),
    );

    run(
        &AgentRunner::new(with_effort),
        vec![Message::user("hi")].into(),
    )
    .await;
    run(&AgentRunner::new(default), vec![Message::user("hi")].into()).await;

    let bodies = request_bodies(&server).await;
    assert_eq!(bodies[0]["reasoning_effort"], "max");
    assert!(bodies[1].get("reasoning_effort").is_none());
}
