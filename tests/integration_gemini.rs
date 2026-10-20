// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::sync::Arc;

use agent_rig::error::Error;
use agent_rig::model::{LlmModel, Message, MessageContent, MessageList, ModelRequest, ToolCall};
use agent_rig::runner::{AgentEvent, AgentRunner};
use agent_rig::tools::{Tool, ToolDefinition, ToolRegistry, ToolResult};
use agent_rig::{Agent, models::gemini::GeminiModel};
use async_trait::async_trait;
use futures_util::StreamExt;
use schemars::{JsonSchema, json_schema};
use serde::Deserialize;
use serde_json::json;
use tokio_util::sync::CancellationToken;

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

struct AddTool {
    definition: ToolDefinition,
}

impl Default for AddTool {
    fn default() -> Self {
        Self {
            definition: ToolDefinition {
                name: "add".to_string(),
                description: "Adds two integers and returns their sum.".to_string(),
                parameters: json_schema!({
                    "type": "object",
                    "properties": {
                        "a": { "type": "integer", "description": "First operand" },
                        "b": { "type": "integer", "description": "Second operand" }
                    },
                    "required": ["a", "b"]
                }),
            },
        }
    }
}

#[async_trait]
impl Tool for AddTool {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn call(&self, tool_call: Arc<ToolCall>, _cancel: CancellationToken) -> ToolResult {
        let a = tool_call.args["a"].as_i64().unwrap_or(0);
        let b = tool_call.args["b"].as_i64().unwrap_or(0);
        ToolResult::ok(json!({ "result": a + b }))
    }
}

const MODEL: &str = "gemini-3.1-flash-lite";

/// A model that returns function calls without IDs, so the runner has to
/// make them up.
const NO_CALL_ID_MODEL: &str = "gemini-2.5-flash";

fn api_key() -> Option<String> {
    let _ = dotenvy::dotenv();
    std::env::var("GEMINI_API_KEY").ok()
}

/// Drives the runner to completion and concatenates the streamed text.
async fn collect_text(runner: AgentRunner, agent: Agent, prompt: &str) -> String {
    let mut text = String::new();
    let mut stream = runner.run(&agent, vec![Message::user(prompt)].into());
    while let Some(event) = stream.next().await {
        if let AgentEvent::TextDelta(chunk) = event.agent_event {
            text.push_str(&chunk);
        }
    }
    text
}

/// What a tool-calling run produced.
struct ToolRun {
    /// Concatenated streamed text.
    text: String,
    /// Every tool call the runner asked the consumer to resolve, in event order.
    calls: Vec<Arc<ToolCall>>,
    /// The final conversation thread from `TurnFinish`.
    thread: MessageList,
}

/// Drives the runner to completion, running each requested tool from
/// `registry` and resolving the call with its result, like a real consumer
/// would. Fails on an `AgentEvent::Error` or if the run ends without
/// `TurnFinish`.
async fn run_with_tools(
    runner: AgentRunner,
    agent: Agent,
    registry: &ToolRegistry,
    prompt: &str,
) -> TestResult<ToolRun> {
    let mut text = String::new();
    let mut calls = Vec::new();
    let mut thread = None;
    let mut stream = runner.run(&agent, vec![Message::user(prompt)].into());
    while let Some(event) = stream.next().await {
        match event.agent_event {
            AgentEvent::ToolCall(call) => {
                calls.push(call.details.clone());
                let result = match registry.get(&call.details.name) {
                    Some(tool) => {
                        tool.call(call.details.clone(), call.cancellation_token.clone())
                            .await
                    }
                    None => ToolResult::error("Unknown tool"),
                };
                call.resolve(result);
            }
            AgentEvent::TextDelta(chunk) => text.push_str(&chunk),
            AgentEvent::TurnFinish { thread: t } => thread = Some(t),
            AgentEvent::Error(error) => return Err(error.into()),
            _ => {}
        }
    }
    let thread = thread.ok_or("run ended without TurnFinish")?;
    Ok(ToolRun {
        text,
        calls,
        thread,
    })
}

/// Returns `true` if `call` is an `add` call on `a` and `b`, in either order.
fn is_add(call: &ToolCall, a: i64, b: i64) -> bool {
    let (x, y) = (call.args["a"].as_i64(), call.args["b"].as_i64());
    call.name == "add" && ((x, y) == (Some(a), Some(b)) || (x, y) == (Some(b), Some(a)))
}

#[tokio::test]
async fn agent_run_returns_non_empty_output() {
    let Some(api_key) = api_key() else { return };

    let model = GeminiModel::new(api_key, MODEL);
    let agent = Agent::builder()
        .name("Greeter")
        .instructions("Reply with exactly one sentence.")
        .build();

    let runner = AgentRunner::new(Arc::new(model));
    let output = collect_text(runner, agent, "Say hello.").await;
    assert!(!output.is_empty());
}

#[tokio::test]
async fn agent_follows_system_instructions() {
    let Some(api_key) = api_key() else { return };

    let model = GeminiModel::new(api_key, MODEL);
    let agent = Agent::builder()
        .name("Pirate")
        .instructions("You are a pirate. Always respond with 'Arrr' somewhere in your reply.")
        .build();

    let runner = AgentRunner::new(Arc::new(model));
    let output = collect_text(runner, agent, "How are you?").await;
    assert!(output.to_lowercase().contains("arrr"));
}

#[tokio::test]
async fn agent_output_schema_returns_valid_json() {
    let Some(api_key) = api_key() else { return };

    #[derive(Deserialize, JsonSchema)]
    struct Sentiment {
        label: String,
        score: f32,
    }

    let model = GeminiModel::new(api_key, MODEL);
    let agent = Agent::builder()
        .name("Classifier")
        .instructions("Classify the sentiment of the input. Return a label (positive/negative/neutral) and a confidence score between 0 and 1.")
        .output_schema(schemars::schema_for!(Sentiment))
        .build();

    let runner = AgentRunner::new(Arc::new(model));
    let output = collect_text(runner, agent, "I love sunny days!").await;

    let parsed: Sentiment = serde_json::from_str(&output).unwrap();
    assert!(!parsed.label.is_empty());
    assert!((0.0..=1.0).contains(&parsed.score));
}

#[tokio::test]
async fn agent_tool_calling_returns_correct_result() -> TestResult {
    let Some(api_key) = api_key() else {
        return Ok(());
    };

    let model = GeminiModel::new(api_key, MODEL);
    let registry = ToolRegistry::new().register(AddTool::default());
    let agent = Agent::builder()
        .name("Calculator")
        .instructions("You are a calculator. Use the add tool to compute sums. When asked to add numbers, call the tool and report the result.")
        .tool("add")
        .build();

    let runner = AgentRunner::with_tools(Arc::new(model), registry.definitions());
    let run = run_with_tools(runner, agent, &registry, "What is 17 + 25?").await?;

    assert!(
        run.calls.iter().any(|call| is_add(call, 17, 25)),
        "expected an add(17, 25) tool call, got: {:?}",
        run.calls
    );
    assert!(
        run.text.contains("42"),
        "expected '42' in output, got: {}",
        run.text
    );
    Ok(())
}

/// Parallel tool calls on a model that returns no call IDs, so the runner's
/// made-up IDs (#96) and the merged function-response turn (#97) are
/// exercised together in a real round trip.
#[tokio::test]
async fn agent_parallel_tool_calls_without_call_ids() -> TestResult {
    let Some(api_key) = api_key() else {
        return Ok(());
    };

    let model = GeminiModel::new(api_key, NO_CALL_ID_MODEL);
    let registry = ToolRegistry::new().register(AddTool::default());
    let agent = Agent::builder()
        .name("Calculator")
        .instructions("You are a calculator. Never add numbers yourself; always use the add tool. When asked for several sums, call the add tool for all of them in parallel, in a single turn, then report every result.")
        .tool("add")
        .build();

    let runner = AgentRunner::with_tools(Arc::new(model), registry.definitions());
    let run = run_with_tools(
        runner,
        agent,
        &registry,
        "Compute 17 + 25, 3 + 4 and 100 + 1.",
    )
    .await?;

    let parallel_turn = run
        .thread
        .iter()
        .find_map(|message| match &message.content {
            MessageContent::ToolCalls(calls) if calls.len() > 1 => Some(calls),
            _ => None,
        })
        .ok_or_else(|| {
            format!(
                "expected a turn with several tool calls, got: {:?}",
                run.calls
            )
        })?;
    let ids: HashSet<&str> = parallel_turn.iter().map(|call| call.id.as_str()).collect();
    assert!(
        !ids.contains(""),
        "expected non-empty tool call IDs, got: {parallel_turn:?}"
    );
    assert_eq!(
        ids.len(),
        parallel_turn.len(),
        "expected distinct tool call IDs, got: {parallel_turn:?}"
    );
    for (a, b) in [(17, 25), (3, 4), (100, 1)] {
        assert!(
            run.calls.iter().any(|call| is_add(call, a, b)),
            "expected an add({a}, {b}) tool call, got: {:?}",
            run.calls
        );
    }
    for sum in ["42", "7", "101"] {
        assert!(
            run.text.contains(sum),
            "expected '{sum}' in output, got: {}",
            run.text
        );
    }
    Ok(())
}

#[tokio::test]
async fn agent_run_reports_token_usage() {
    let Some(api_key) = api_key() else { return };

    let model = GeminiModel::new(api_key, MODEL);
    let agent = Agent::builder()
        .name("Greeter")
        .instructions("Reply with exactly one sentence.")
        .build();

    let runner = AgentRunner::new(Arc::new(model));
    let mut stream = runner.run(&agent, vec![Message::user("Say hello.")].into());

    let mut got_usage = None;
    while let Some(event) = stream.next().await {
        if let AgentEvent::Usage(usage) = event.agent_event {
            got_usage = Some(usage);
        }
    }

    let usage = got_usage.expect("Gemini must report token usage");
    assert!(usage.input_tokens.unwrap_or(0) > 0);
    assert!(usage.output_tokens.unwrap_or(0) > 0);
}

/// Regression test for #43: when the provider rejects a streaming request, the
/// failure must surface as an `Err` stream item — never a silently-empty stream
/// that the runner reads as a normal empty turn.
///
/// This reproduces the exact scenario from the issue: a function-response turn
/// with no preceding function-call turn (what a denied/replayed tool call
/// produces). Gemini rejects it with `400 INVALID_ARGUMENT`. The contract under
/// test is that `generate_stream` yields a [`Error::Provider`] rather than
/// ending with zero items.
///
/// Gated on a real `GEMINI_API_KEY` because the rejection comes from the live
/// API; it is a no-op pass when no key is configured, matching the other tests
/// in this file.
#[tokio::test]
async fn generate_stream_surfaces_rejected_request() {
    let Some(api_key) = api_key() else { return };

    let model = GeminiModel::new(api_key, MODEL);
    // A tool result with no preceding tool-call turn — the malformed replay
    // thread described in #43. Gemini responds 400 INVALID_ARGUMENT:
    // "function response turn must come immediately after a function call turn".
    let request = ModelRequest {
        messages: vec![Message::tool_result(
            Arc::new(ToolCall {
                id: "ghost-call".to_string(),
                name: "ghost".to_string(),
                args: json!({ "ok": true }),
                provider_metadata: None,
            }),
            json!({ "ok": true }),
        )]
        .into(),
        system: None,
        output_schema: None,
        tools: vec![],
    };

    let mut stream = model.generate_stream(request);
    let mut items = Vec::new();
    while let Some(item) = stream.next().await {
        items.push(item);
    }

    assert!(
        !items.is_empty(),
        "stream ended silently with zero items; the rejected request was \
         swallowed and is indistinguishable from a successful empty turn (#43)"
    );
    assert!(
        items.iter().any(|i| matches!(i, Err(Error::Provider(_)))),
        "expected a Provider error to be yielded for the rejected request, \
         got: {items:?}"
    );
}

#[tokio::test]
async fn agent_run_with_temperature_setting() {
    let Some(api_key) = api_key() else { return };

    let model = GeminiModel::builder(api_key, MODEL)
        .temperature(0.1)
        .max_output_tokens(64)
        .build();

    let agent = Agent::builder()
        .name("Assistant")
        .instructions("Be concise.")
        .build();

    let runner = AgentRunner::new(Arc::new(model));
    let output = collect_text(runner, agent, "What is 2 + 2?").await;
    assert!(!output.is_empty());
}
