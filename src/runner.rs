//! Drives an [`Agent`] against an [`LlmModel`] until it produces a final reply.
//!
//! [`AgentRunner`] owns the model, the [`ToolRegistry`], and (optionally) an
//! [`AuthManager`]. Calling [`AgentRunner::run`] spawns the agentic loop on a
//! background task and returns a [`Stream`] of [`AgentEvent`]s — text and
//! reasoning deltas, plus the lifecycle of each tool call.
//!
//! The loop keeps calling the model until a turn comes back with no tool
//! calls. Within a single turn, tool calls are executed concurrently; their
//! result messages are appended to the thread in the same order the model
//! issued them, even though the corresponding events may interleave.

use std::{pin::Pin, sync::Arc};

use futures_util::{Stream, StreamExt, future::join_all};
use serde_json::Value;
use tokio::sync::mpsc::{self, Sender};
use tracing::debug;

use crate::{
    Agent,
    auth::AuthManager,
    error::Error,
    model::{LlmModel, Message, ModelRequest, ModelStreamChunk, ToolCall},
    tools::{AgentTool, Tool, ToolDefinition, ToolRegistry, ToolRegistryEntry},
};

/// Outcome of executing a single tool call.
///
/// Reported back to the model as the tool result on the next turn (via
/// [`From<ToolCallResult> for Value`](#impl-From<ToolCallResult>-for-Value))
/// and surfaced to the consumer inside [`AgentEvent::ToolCallFinished`].
#[derive(Clone, Debug)]
pub enum ToolCallResult {
    /// The tool ran and returned this JSON value.
    Ok(Value),
    /// The tool failed. The error is surfaced to the model as a string.
    Err(Error),
    /// The [`AuthManager`] denied the call; the tool was not invoked.
    Denied,
    /// The model called a tool that is not registered. No `Started` /
    /// `Finished` events are emitted in this case, but a synthetic
    /// result is still sent back to the model so the assistant turn and
    /// tool-result messages stay paired.
    Unknown,
}

impl From<ToolCallResult> for Value {
    fn from(value: ToolCallResult) -> Self {
        match value {
            ToolCallResult::Denied => Value::from("Tool call denied"),
            ToolCallResult::Ok(result) => result,
            ToolCallResult::Err(error) => Value::from(format!("Tool call error: {error}")),
            ToolCallResult::Unknown => Value::from("Unknown tool"),
        }
    }
}

/// An event yielded by [`AgentRunner::run`] as the agent loop progresses.
///
/// Variants fall into two groups:
///
/// - Model output: [`ThinkingDelta`](AgentEvent::ThinkingDelta) and
///   [`TextDelta`](AgentEvent::TextDelta) carry chunks as the provider streams
///   them. Concatenating every `TextDelta` reconstructs the final reply.
/// - Tool lifecycle:
///   [`ToolCallStarted`](AgentEvent::ToolCallStarted) fires before a tool
///   runs (after authorization, if any) and
///   [`ToolCallFinished`](AgentEvent::ToolCallFinished) fires once it
///   resolves. Hallucinated tool calls (no matching registry entry) emit
///   *neither* event; see [`ToolCallResult::Unknown`].
/// - [`Error`](AgentEvent::Error) terminates the stream early when the
///   underlying provider fails.
#[derive(Clone, Debug)]
pub enum AgentEvent {
    /// A registered tool is about to run with these arguments.
    ToolCallStarted {
        /// Name of the tool being invoked.
        name: String,
        /// The JSON arguments the model passed.
        args: serde_json::Value,
    },
    /// A tool call resolved with [`ToolCallResult`]. Fires after the tool
    /// returns, errors, or is denied.
    ToolCallFinished {
        /// Name of the tool that resolved.
        name: String,
        /// Outcome of the call.
        result: ToolCallResult,
    },
    /// A chunk of the model's reasoning/thinking output, if the provider
    /// supports extended thinking.
    ThinkingDelta(String),
    /// A chunk of the model's text output.
    TextDelta(String),
    /// The provider returned an error. The stream ends after this event.
    Error(crate::error::Error),
}

/// Per-event metadata for nested agent runs.
///
/// Reserved for sub-agent dispatch — `thread_id` identifies the agent's
/// thread and `depth` records how deeply nested it is below the root run.
/// Not yet wired through [`AgentRunner::run`]; see
/// `docs/PLAN-subagents-mpsc.md`.
pub struct RunnerEvent {
    /// Identifier of the thread that produced this event.
    pub thread_id: usize,
    /// Nesting depth below the root run (root is 0).
    pub depth: usize,
    /// The wrapped event.
    pub agent_event: AgentEvent,
}

/// Drives an agent against an [`LlmModel`] and a [`ToolRegistry`].
///
/// Construct one with [`AgentRunner::new`] (no tools) or
/// [`AgentRunner::with_registry`] (with tools), optionally chaining
/// [`AgentRunner::with_auth_manager`] to gate tool execution. Call
/// [`AgentRunner::run`] to start the agentic loop and consume the returned
/// stream until it ends.
///
/// `AgentRunner` is cheap to clone — internals are behind [`Arc`] — so a
/// single runner can be shared across tasks.
#[derive(Clone)]
pub struct AgentRunner {
    model: Arc<dyn LlmModel>,
    registry: Arc<ToolRegistry>,
    auth_manager: Option<Arc<dyn AuthManager>>,
}

impl AgentRunner {
    /// Creates a runner that uses `model` and has no tools registered.
    pub fn new(model: Arc<dyn LlmModel>) -> Self {
        AgentRunner {
            model,
            registry: Arc::new(ToolRegistry::new()),
            auth_manager: None,
        }
    }

    /// Creates a runner that uses `model` and the supplied [`ToolRegistry`].
    pub fn with_registry(model: Arc<dyn LlmModel>, registry: Arc<ToolRegistry>) -> Self {
        AgentRunner {
            model,
            registry,
            auth_manager: None,
        }
    }

    /// Sets the [`AuthManager`] consulted before every tool call.
    ///
    /// With no manager set, no authorization is performed and all calls run.
    /// The manager decides which calls require approval and how to obtain it.
    pub fn with_auth_manager(mut self, auth_manager: Arc<dyn AuthManager>) -> Self {
        self.auth_manager = Some(auth_manager);
        self
    }

    /// Runs `agent` starting from `thread` and returns the event stream.
    ///
    /// The agentic loop runs on a background tokio task; events are delivered
    /// through an mpsc channel as they happen. The stream ends after the
    /// model produces a turn with no tool calls, or after an
    /// [`AgentEvent::Error`].
    ///
    /// `thread` is the conversation so far — typically a single
    /// [`Message::user`](crate::model::Message::user) for the first turn, or a
    /// previously accumulated history for follow-ups. The thread is consumed;
    /// the resulting thread is not returned (each call starts a fresh loop).
    pub fn run(
        &self,
        agent: Agent,
        thread: Vec<Message>,
    ) -> Pin<Box<dyn Stream<Item = AgentEvent> + Send>> {
        debug!(agent = agent.name(), "starting run");
        // Clone `self` outside the `stream!` macro block to prevent the generator from
        // capturing the non-'static `&self` reference, satisfying `'static` for the trait object.
        let cloned = self.clone();

        let stream = async_stream::stream! {
          let (tx, mut rx) = mpsc::channel::<AgentEvent>(100);
          tokio::spawn(cloned.main_loop(tx, agent, thread));

          while let Some(message) = rx.recv().await {
            yield message;
          }
        };
        Box::pin(stream)
    }

    async fn main_loop(self, tx: Sender<AgentEvent>, agent: Agent, mut thread: Vec<Message>) {
        let tools: Vec<ToolDefinition> = self.registry.definitions();

        loop {
            let request = ModelRequest {
                messages: thread.clone(),
                system: Some(agent.instructions().to_string()),
                output_schema: agent.output_schema().cloned(),
                tools: tools.clone(),
            };

            let mut model_stream = self.model.generate_stream(request);
            let mut tool_calls: Vec<ToolCall> = Vec::new();
            while let Some(chunk) = model_stream.next().await {
                match chunk {
                    Ok(ModelStreamChunk::Thinking(t)) => {
                        let _ = tx.send(AgentEvent::ThinkingDelta(t)).await;
                    }
                    Ok(ModelStreamChunk::TextDelta(t)) => {
                        let _ = tx.send(AgentEvent::TextDelta(t)).await;
                    }
                    Ok(ModelStreamChunk::ToolCall(call)) => {
                        tool_calls.push(call);
                    }
                    Err(error) => {
                        let _ = tx.send(AgentEvent::Error(error)).await;
                        return;
                    }
                }
            }

            if tool_calls.is_empty() {
                break;
            }

            self.handle_tool_calls(&tx, tool_calls, &mut thread).await;
        }
    }

    async fn handle_tool_calls(
        &self,
        tx: &Sender<AgentEvent>,
        tool_calls: Vec<ToolCall>,
        thread: &mut Vec<Message>,
    ) {
        // Append the tool calls as a single assistant turn.
        thread.push(Message::tool_calls(tool_calls.clone()));

        // Each future runs the full lifecycle for one call: authorization check
        // (if required), emit Started, execute, emit Finished / Error / Denied.
        // Hallucinated calls skip Started but still produce a synthetic result
        // so the assistant turn and tool-result messages remain paired.
        let tool_futures = tool_calls.into_iter().map(|call| async move {
            // Hallucinated tool: produce a synthetic message, no events.
            let Some(tool) = self.registry.get(&call.name) else {
                return (call, ToolCallResult::Unknown);
            };

            // Authorization gate: the sync check decides whether to consult
            // the async decision path. If no manager is configured, no gating.
            if let Some(auth) = &self.auth_manager
                && auth.requires_authorization(&call.name, &call.args)
                && !auth.authorize(&call.name, &call.args).await
            {
                let result = ToolCallResult::Denied;
                let _ = tx
                    .send(AgentEvent::ToolCallFinished {
                        name: call.name.clone(),
                        result: result.clone(),
                    })
                    .await;
                return (call, result);
            }

            let _ = tx
                .send(AgentEvent::ToolCallStarted {
                    name: call.name.clone(),
                    args: call.args.clone(),
                })
                .await;

            let event = match tool {
                ToolRegistryEntry::Tool(t) => handle_tool(t.as_ref(), &call).await,
                ToolRegistryEntry::Agent(a) => handle_agent(a.as_ref(), &call, tx.clone()).await,
            };

            debug!(tool = call.name, "tool call complete");
            let _ = tx
                .send(AgentEvent::ToolCallFinished {
                    name: call.name.clone(),
                    result: event.clone(),
                })
                .await;
            (call, event)
        });

        // Run all calls concurrently. `join_all` preserves input order in the
        // returned Vec, so tool-result messages are appended in the same order
        // the model requested them — even though events may interleave.
        let results = join_all(tool_futures).await;
        for (call, result) in results {
            thread.push(Message::tool_result(
                call.id,
                call.name,
                result.into(),
                call.provider_metadata,
            ));
        }
    }
}

async fn handle_tool(tool: &dyn Tool, tool_call: &ToolCall) -> ToolCallResult {
    let result = tool.call(tool_call.args.clone()).await;
    match result {
        Ok(result) => ToolCallResult::Ok(result),
        Err(error) => ToolCallResult::Err(error),
    }
}

async fn handle_agent(
    tool: &AgentTool,
    tool_call: &ToolCall,
    tx: Sender<AgentEvent>,
) -> ToolCallResult {
    let mut result = String::new();
    let stream = tool.call(tool_call.args.clone()).await;
    let mut stream = match stream {
        Ok(stream) => stream,
        Err(error) => return ToolCallResult::Err(error),
    };

    while let Some(next) = stream.next().await {
        if let AgentEvent::TextDelta(text) = &next {
            result += text;
        }
        let _ = tx.send(next).await;
    }
    ToolCallResult::Ok(Value::String(result))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{MessageContent, ModelResponse, Role};
    use async_trait::async_trait;
    use futures_util::StreamExt;
    use serde_json::json;
    use std::sync::Mutex;

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
        })
    }

    fn agent(name: &str) -> Agent {
        Agent::builder()
            .name(name)
            .instructions("test instructions")
            .build()
    }

    async fn collect(runner: &AgentRunner, agent: Agent, prompt: &str) -> Vec<AgentEvent> {
        let mut stream = runner.run(agent, vec![Message::user(prompt)]);
        let mut events = Vec::new();
        while let Some(event) = stream.next().await {
            events.push(event);
        }
        events
    }

    /// Tool that records every invocation and returns a configurable result.
    struct EchoTool {
        name: &'static str,
        result: Result<serde_json::Value, Error>,
        calls: Arc<Mutex<Vec<serde_json::Value>>>,
    }

    impl EchoTool {
        fn ok(name: &'static str) -> (Self, Arc<Mutex<Vec<serde_json::Value>>>) {
            let calls = Arc::new(Mutex::new(Vec::new()));
            (
                Self {
                    name,
                    result: Ok(json!({"ok": true})),
                    calls: calls.clone(),
                },
                calls,
            )
        }

        fn failing(name: &'static str, msg: &str) -> Self {
            Self {
                name,
                result: Err(Error::Agent(msg.to_string())),
                calls: Arc::new(Mutex::new(Vec::new())),
            }
        }
    }

    #[async_trait]
    impl Tool for EchoTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: self.name.to_string(),
                description: "echo".to_string(),
                parameters: json!({"type": "object"}),
            }
        }

        async fn call(&self, args: serde_json::Value) -> Result<serde_json::Value, Error> {
            self.calls.lock().unwrap().push(args);
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
        let registry = Arc::new(ToolRegistry::new().register(Box::new(tool)));
        let model = ScriptedModel::new(vec![
            tool_call_response("c1", "echo", json!({"x": 1})),
            final_response("done"),
        ]);
        let runner = AgentRunner::with_registry(model.clone(), registry);

        let events = collect(&runner, agent("Looper"), "go").await;

        // Started + Finished + TextDelta
        assert!(matches!(
            events[0],
            AgentEvent::ToolCallStarted { ref name, .. } if name == "echo"
        ));
        assert!(matches!(
            events[1],
            AgentEvent::ToolCallFinished {
                ref name,
                result: ToolCallResult::Ok(_),
            } if name == "echo"
        ));
        assert!(matches!(events[2], AgentEvent::TextDelta(ref t) if t == "done"));

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

        // Hallucinated tool calls are silent — no Started, no Finished — but
        // the synthetic tool-result message is still appended to the thread
        // so the assistant turn and the tool-result remain paired.
        assert!(!events.iter().any(|e| matches!(
            e,
            AgentEvent::ToolCallStarted { .. } | AgentEvent::ToolCallFinished { .. }
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
        let registry = Arc::new(ToolRegistry::new().register(Box::new(tool)));
        let model = ScriptedModel::new(vec![
            tool_call_response("c1", "boom", json!({})),
            final_response("ok"),
        ]);
        let runner = AgentRunner::with_registry(model, registry);

        let events = collect(&runner, agent("a"), "go").await;
        let finished = events
            .iter()
            .find(|e| matches!(e, AgentEvent::ToolCallFinished { .. }))
            .unwrap();
        assert!(matches!(
            finished,
            AgentEvent::ToolCallFinished {
                result: ToolCallResult::Err(_),
                ..
            }
        ));
    }

    /// Auth manager that records its decisions and returns a scripted vector
    /// of allow/deny responses.
    struct ScriptedAuth {
        decisions: Mutex<std::collections::VecDeque<bool>>,
        required: bool,
        calls: Arc<Mutex<Vec<String>>>,
    }

    impl ScriptedAuth {
        fn new(required: bool, decisions: Vec<bool>) -> Arc<Self> {
            Arc::new(Self {
                decisions: Mutex::new(decisions.into()),
                required,
                calls: Arc::new(Mutex::new(Vec::new())),
            })
        }
    }

    #[async_trait]
    impl AuthManager for ScriptedAuth {
        fn requires_authorization(&self, _name: &str, _args: &Value) -> bool {
            self.required
        }

        async fn authorize(&self, name: &str, _args: &Value) -> bool {
            self.calls.lock().unwrap().push(name.to_string());
            self.decisions
                .lock()
                .unwrap()
                .pop_front()
                .expect("ScriptedAuth: decision queue exhausted")
        }
    }

    #[tokio::test]
    async fn auth_denial_skips_tool_execution() {
        let (tool, calls) = EchoTool::ok("echo");
        let registry = Arc::new(ToolRegistry::new().register(Box::new(tool)));
        let auth = ScriptedAuth::new(true, vec![false]);
        let model = ScriptedModel::new(vec![
            tool_call_response("c1", "echo", json!({})),
            final_response("ok"),
        ]);
        let runner = AgentRunner::with_registry(model, registry).with_auth_manager(auth.clone());

        let events = collect(&runner, agent("a"), "go").await;

        // Denial skips Started and the underlying tool, but still emits
        // Finished so the consumer sees the outcome.
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, AgentEvent::ToolCallStarted { .. }))
        );
        let finished = events
            .iter()
            .find(|e| matches!(e, AgentEvent::ToolCallFinished { .. }))
            .unwrap();
        assert!(matches!(
            finished,
            AgentEvent::ToolCallFinished {
                result: ToolCallResult::Denied,
                ..
            }
        ));
        assert!(calls.lock().unwrap().is_empty());
        assert_eq!(auth.calls.lock().unwrap().as_slice(), &["echo".to_string()]);
    }

    #[tokio::test]
    async fn auth_fast_path_skips_authorize() {
        let (tool, calls) = EchoTool::ok("echo");
        let registry = Arc::new(ToolRegistry::new().register(Box::new(tool)));
        // requires_authorization returns false — authorize must never run.
        let auth = ScriptedAuth::new(false, vec![]);
        let model = ScriptedModel::new(vec![
            tool_call_response("c1", "echo", json!({})),
            final_response("ok"),
        ]);
        let runner = AgentRunner::with_registry(model, registry).with_auth_manager(auth.clone());

        let _ = collect(&runner, agent("a"), "go").await;

        assert!(auth.calls.lock().unwrap().is_empty());
        assert_eq!(calls.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn thinking_chunks_are_forwarded() {
        let model = ScriptedModel::new(vec![Ok(ModelResponse {
            text: Some("answer".into()),
            tool_calls: vec![],
            thinking: Some("reasoning".into()),
        })]);
        let runner = AgentRunner::new(model);

        let events = collect(&runner, agent("a"), "q").await;
        let kinds: Vec<&'static str> = events
            .iter()
            .map(|e| match e {
                AgentEvent::ThinkingDelta(_) => "thinking",
                AgentEvent::TextDelta(_) => "text",
                AgentEvent::ToolCallStarted { .. } => "started",
                AgentEvent::ToolCallFinished { .. } => "finished",
                AgentEvent::Error(_) => "error",
            })
            .collect();
        // Default `generate_stream` yields thinking before text.
        assert_eq!(kinds, vec!["thinking", "text"]);
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
        let registry = Arc::new(ToolRegistry::new().register(Box::new(tool)));
        let model = ScriptedModel::new(vec![
            Ok(ModelResponse {
                text: None,
                tool_calls: vec![
                    ToolCall::new("c1".into(), "echo".into(), json!({"i": 1})),
                    ToolCall::new("c2".into(), "echo".into(), json!({"i": 2})),
                    ToolCall::new("c3".into(), "echo".into(), json!({"i": 3})),
                ],
                thinking: None,
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
}
