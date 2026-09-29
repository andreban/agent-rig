// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::error::Error;
use crate::model::{LlmModel, MessageContent, ModelRequest, ModelResponse, TokenUsage, ToolCall};
use async_trait::async_trait;
use schemars::json_schema;
use serde_json::json;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::{Barrier, Notify};
use tokio_util::sync::CancellationToken;

/// Upper bound for tests that would hang if a tool call were never resolved
/// or calls ran sequentially.
const TIMEOUT: Duration = Duration::from_secs(5);

/// Wraps raw JSON args in an [`Arc<ToolCall>`] for invoking [`AgentTool::call`].
fn tool_call(args: serde_json::Value) -> Arc<ToolCall> {
    Arc::new(ToolCall::new(
        "test-call".to_string(),
        "child_tool".to_string(),
        args,
    ))
}

/// Minimal scripted [`LlmModel`] returning queued responses and recording
/// every [`ModelRequest`] it received. Kept local to this test module so
/// it doesn't depend on the runner's test helpers.
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

fn text_only(text: &str) -> Result<ModelResponse, Error> {
    Ok(ModelResponse {
        text: Some(text.to_string()),
        tool_calls: vec![],
        thinking: None,
        token_usage: None,
    })
}

fn build_agent_tool(model: Arc<ScriptedModel>) -> AgentTool {
    let agent = Agent::builder()
        .name("Child")
        .instructions("test instructions")
        .build();
    let runner = AgentRunner::new(model);
    AgentTool::new(
        ToolDefinition {
            name: "child_tool".to_string(),
            description: "test child".to_string(),
            parameters: json_schema!({"type": "object"}),
        },
        agent,
        runner,
    )
}

/// The child's final reply is returned as a successful tool result.
#[tokio::test]
async fn call_returns_final_reply_as_success() {
    let model = ScriptedModel::new(vec![text_only("hello world")]);
    let tool = build_agent_tool(model);

    let result = tool
        .call(
            tool_call(json!({"text": "anything"})),
            CancellationToken::new(),
        )
        .await;
    let ToolResult::Ok(output) = result else {
        panic!("expected Ok, got {result:?}");
    };
    assert_eq!(output, json!("hello world"));
}

/// The JSON args become the child run's user message verbatim (after
/// `serde_json::to_string`).
#[tokio::test]
async fn call_passes_args_as_serialized_json_user_message() {
    let model = ScriptedModel::new(vec![text_only("ok")]);
    let tool = build_agent_tool(model.clone());

    let _ = tool
        .call(
            tool_call(json!({"text": "hello", "n": 42})),
            CancellationToken::new(),
        )
        .await;

    let requests = model.requests();
    assert_eq!(requests.len(), 1);
    let first_msg = &requests[0].messages[0];
    let MessageContent::Text(raw) = &first_msg.content else {
        panic!("expected text content, got {:?}", first_msg.content);
    };
    // Round-trip through `serde_json` so object-field order can't
    // make this flaky.
    let parsed: serde_json::Value = serde_json::from_str(raw).unwrap();
    assert_eq!(parsed, json!({ "text": "hello", "n": 42 }));
}

fn tool_calls(calls: &[(&str, &str)]) -> Result<ModelResponse, Error> {
    Ok(ModelResponse {
        text: None,
        tool_calls: calls
            .iter()
            .map(|(id, name)| ToolCall::new(id.to_string(), name.to_string(), json!({})))
            .collect(),
        thinking: None,
        token_usage: None,
    })
}

fn with_usage(
    mut response: Result<ModelResponse, Error>,
    input: u32,
    output: u32,
) -> Result<ModelResponse, Error> {
    if let Ok(response) = &mut response {
        response.token_usage = Some(TokenUsage {
            input_tokens: Some(input),
            output_tokens: Some(output),
            ..Default::default()
        });
    }
    response
}

fn definition(name: &str) -> ToolDefinition {
    ToolDefinition {
        name: name.to_string(),
        description: format!("test tool {name}"),
        parameters: json_schema!({"type": "object"}),
    }
}

fn build_agent_tool_with_tools(model: Arc<ScriptedModel>, tools: ToolRegistry) -> AgentTool {
    let agent = Agent::builder()
        .name("Child")
        .instructions("test instructions")
        .build();
    AgentTool::with_tools(definition("child_tool"), agent, model, Arc::new(tools))
}

/// Returns the result the child model received for the tool call `name`.
fn tool_result_in(request: &ModelRequest, name: &str) -> Option<serde_json::Value> {
    request.messages.iter().find_map(|m| match &m.content {
        MessageContent::ToolResult { tool_call, result } if tool_call.name == name => {
            Some(result.clone())
        }
        _ => None,
    })
}

/// Returns a fixed JSON payload.
struct AnswerTool {
    definition: ToolDefinition,
}

#[async_trait]
impl Tool for AnswerTool {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn call(&self, _: Arc<ToolCall>, _: CancellationToken) -> ToolResult {
        ToolResult::ok(json!({"answer": 42}))
    }
}

/// Waits on a shared barrier, so it only completes if every tool sharing the
/// barrier is running at the same time.
struct BarrierTool {
    definition: ToolDefinition,
    barrier: Arc<Barrier>,
}

#[async_trait]
impl Tool for BarrierTool {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn call(&self, _: Arc<ToolCall>, _: CancellationToken) -> ToolResult {
        self.barrier.wait().await;
        ToolResult::ok(self.definition.name.clone())
    }
}

/// Signals `started`, then blocks until its cancellation token fires.
struct BlockingTool {
    definition: ToolDefinition,
    started: Arc<Notify>,
}

#[async_trait]
impl Tool for BlockingTool {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn call(&self, _: Arc<ToolCall>, cancel: CancellationToken) -> ToolResult {
        self.started.notify_one();
        cancel.cancelled().await;
        ToolResult::error("cancelled")
    }
}

/// The child's tool call runs against its registry and the result reaches
/// the child model on its next request.
#[tokio::test]
async fn child_tool_call_runs_and_result_reaches_child() {
    let model = ScriptedModel::new(vec![tool_calls(&[("c1", "answer")]), text_only("done")]);
    let tool = build_agent_tool_with_tools(
        model.clone(),
        ToolRegistry::new().register(AnswerTool {
            definition: definition("answer"),
        }),
    );

    let result = tool
        .call(tool_call(json!({})), CancellationToken::new())
        .await;
    let ToolResult::Ok(output) = result else {
        panic!("expected Ok, got {result:?}");
    };
    assert_eq!(output, json!("done"));

    let requests = model.requests();
    assert_eq!(requests.len(), 2);
    // The child model sees the tools from the registry.
    assert_eq!(requests[0].tools.len(), 1);
    assert_eq!(requests[0].tools[0].name, "answer");
    assert_eq!(
        tool_result_in(&requests[1], "answer"),
        Some(json!({"success": {"answer": 42}}))
    );
}

/// Two tool calls in one turn run concurrently: each waits on a barrier that
/// only opens once both are running.
#[tokio::test]
async fn child_tool_calls_in_one_turn_run_concurrently() {
    let barrier = Arc::new(Barrier::new(2));
    let model = ScriptedModel::new(vec![
        tool_calls(&[("c1", "first"), ("c2", "second")]),
        text_only("done"),
    ]);
    let tool = build_agent_tool_with_tools(
        model.clone(),
        ToolRegistry::new()
            .register(BarrierTool {
                definition: definition("first"),
                barrier: barrier.clone(),
            })
            .register(BarrierTool {
                definition: definition("second"),
                barrier,
            }),
    );

    let result = tokio::time::timeout(
        TIMEOUT,
        tool.call(tool_call(json!({})), CancellationToken::new()),
    )
    .await
    .expect("tool calls did not run concurrently");
    assert!(matches!(result, ToolResult::Ok(_)), "got {result:?}");

    let requests = model.requests();
    assert_eq!(
        tool_result_in(&requests[1], "first"),
        Some(json!({"success": "first"}))
    );
    assert_eq!(
        tool_result_in(&requests[1], "second"),
        Some(json!({"success": "second"}))
    );
}

/// A call to a tool the child doesn't have resolves as unknown instead of
/// hanging the run.
#[tokio::test]
async fn unknown_child_tool_resolves_instead_of_hanging() {
    let model = ScriptedModel::new(vec![tool_calls(&[("c1", "missing")]), text_only("done")]);
    let tool = build_agent_tool(model.clone());

    let result = tokio::time::timeout(
        TIMEOUT,
        tool.call(tool_call(json!({})), CancellationToken::new()),
    )
    .await
    .expect("unknown tool call was never resolved");
    let ToolResult::Ok(output) = result else {
        panic!("expected Ok, got {result:?}");
    };
    assert_eq!(output, json!("done"));
    assert_eq!(
        tool_result_in(&model.requests()[1], "missing"),
        Some(serde_json::Value::from(ToolCallResult::Unknown))
    );
}

/// `on_usage` is called with the usage of every child model call.
#[tokio::test]
async fn on_usage_sees_every_child_model_call() {
    let model = ScriptedModel::new(vec![
        with_usage(tool_calls(&[("c1", "answer")]), 10, 2),
        with_usage(text_only("done"), 20, 3),
    ]);
    let seen = Arc::new(Mutex::new(Vec::new()));
    let tool = build_agent_tool_with_tools(
        model,
        ToolRegistry::new().register(AnswerTool {
            definition: definition("answer"),
        }),
    )
    .on_usage({
        let seen = seen.clone();
        move |usage| seen.lock().unwrap().push(usage.clone())
    });

    let result = tool
        .call(tool_call(json!({})), CancellationToken::new())
        .await;
    assert!(matches!(result, ToolResult::Ok(_)), "got {result:?}");

    let seen = seen.lock().unwrap();
    let tokens: Vec<_> = seen
        .iter()
        .map(|u| (u.input_tokens, u.output_tokens))
        .collect();
    assert_eq!(tokens, vec![(Some(10), Some(2)), (Some(20), Some(3))]);
}

/// Text the child writes before a tool call is not part of the result; only
/// the final reply is.
#[tokio::test]
async fn result_is_only_the_final_reply() {
    let mut first = tool_calls(&[("c1", "answer")]);
    if let Ok(response) = &mut first {
        response.text = Some("Let me look that up. ".to_string());
    }
    let model = ScriptedModel::new(vec![first, text_only("The answer is 42.")]);
    let tool = build_agent_tool_with_tools(
        model,
        ToolRegistry::new().register(AnswerTool {
            definition: definition("answer"),
        }),
    );

    let result = tool
        .call(tool_call(json!({})), CancellationToken::new())
        .await;
    let ToolResult::Ok(output) = result else {
        panic!("expected Ok, got {result:?}");
    };
    assert_eq!(output, json!("The answer is 42."));
}

/// Cancelling while a child tool is running ends the call with an error.
#[tokio::test]
async fn cancel_returns_err() {
    let started = Arc::new(Notify::new());
    let model = ScriptedModel::new(vec![tool_calls(&[("c1", "block")])]);
    let tool = build_agent_tool_with_tools(
        model,
        ToolRegistry::new().register(BlockingTool {
            definition: definition("block"),
            started: started.clone(),
        }),
    );

    let cancel = CancellationToken::new();
    let (result, ()) = tokio::time::timeout(TIMEOUT, async {
        tokio::join!(tool.call(tool_call(json!({})), cancel.clone()), async {
            started.notified().await;
            cancel.cancel();
        })
    })
    .await
    .expect("cancelled call did not return");
    assert!(matches!(result, ToolResult::Err(_)), "got {result:?}");
}

/// An error from the child model ends the call with an error.
#[tokio::test]
async fn child_error_returns_err() {
    let model = ScriptedModel::new(vec![Err(Error::Provider("boom".to_string()))]);
    let tool = build_agent_tool(model);

    let result = tool
        .call(tool_call(json!({})), CancellationToken::new())
        .await;
    assert!(matches!(result, ToolResult::Err(_)), "got {result:?}");
}
