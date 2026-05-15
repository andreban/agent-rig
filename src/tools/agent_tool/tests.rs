use super::*;
use crate::model::{LlmModel, MessageContent, ModelRequest, ModelResponse};
use crate::runner::RunEvent;
use async_trait::async_trait;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

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
            parameters: json!({"type": "object"}),
        },
        agent,
        runner,
    )
}

fn fresh_emitter() -> (RunEmitter, mpsc::Receiver<RunEvent>) {
    let (tx, rx) = mpsc::channel(16);
    (RunEmitter::new(tx, None), rx)
}

/// Drains the receiver. All senders end up dropped once `call` returns
/// (the emitter is moved into `call`; the child runner's spawned task
/// drops its own senders when it exits), so `recv` eventually yields
/// `None`.
async fn drain(mut rx: mpsc::Receiver<RunEvent>) -> Vec<RunEvent> {
    let mut events = Vec::new();
    while let Some(e) = rx.recv().await {
        events.push(e);
    }
    events
}

/// Regression test for a Gemini-specific bug: `function_response.response`
/// is typed as a protobuf `Struct`, so the value MUST be a JSON object.
/// `AgentTool::call` always wraps the child's text reply in
/// `{"output": "..."}` regardless of what the child agent actually emitted.
#[tokio::test]
async fn call_wraps_accumulated_text_in_output_object() {
    let model = ScriptedModel::new(vec![text_only("hello world")]);
    let tool = build_agent_tool(model);
    let (emitter, rx) = fresh_emitter();

    let result = tool
        .call(emitter, json!({"text": "anything"}))
        .await
        .unwrap();
    assert_eq!(result, json!({ "output": "hello world" }));
    let _ = drain(rx).await;
}

/// The JSON args become the child run's user message verbatim (after
/// `serde_json::to_string`).
#[tokio::test]
async fn call_passes_args_as_serialized_json_user_message() {
    let model = ScriptedModel::new(vec![text_only("ok")]);
    let tool = build_agent_tool(model.clone());
    let (emitter, rx) = fresh_emitter();

    let _ = tool
        .call(emitter, json!({"text": "hello", "n": 42}))
        .await
        .unwrap();
    let _ = drain(rx).await;

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

/// Every event the child run produces is forwarded through the supplied
/// emitter. Forwarding goes via `tx.send(next.agent_event)`, which
/// re-stamps the event with the *parent* emitter's `run_id` — that's the
/// observable contract.
#[tokio::test]
async fn call_forwards_child_events_through_parent_emitter() {
    let model = ScriptedModel::new(vec![text_only("hi")]);
    let tool = build_agent_tool(model);
    let (emitter, rx) = fresh_emitter();
    let parent_run_id = emitter.run_id;

    let _ = tool.call(emitter, json!({})).await.unwrap();
    let events = drain(rx).await;

    let text: String = events
        .iter()
        .filter_map(|e| match &e.agent_event {
            AgentEvent::TextDelta(t) => Some(t.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(text, "hi");
    assert!(
        events.iter().all(|e| e.run_id == parent_run_id),
        "forwarded events should be stamped with the parent emitter's run_id"
    );
}
