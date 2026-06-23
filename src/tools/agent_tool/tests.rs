// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::error::Error;
use crate::model::{LlmModel, MessageContent, ModelRequest, ModelResponse};
use async_trait::async_trait;
use schemars::json_schema;
use serde_json::json;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use tokio_util::sync::CancellationToken;

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

/// Regression test for a Gemini-specific bug: `function_response.response`
/// is typed as a protobuf `Struct`, so the value MUST be a JSON object.
/// `AgentTool::apply` always wraps the child's text reply in
/// `{"output": "..."}` regardless of what the child agent actually emitted.
#[tokio::test]
async fn call_returns_accumulated_text_as_success() {
    let model = ScriptedModel::new(vec![text_only("hello world")]);
    let tool = build_agent_tool(model);

    let result = tool
        .apply(json!({"text": "anything"}), CancellationToken::new())
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
        .apply(json!({"text": "hello", "n": 42}), CancellationToken::new())
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
