// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use agent_rig::error::Error;
use agent_rig::model::{
    LlmModel, Message, MessageContent, MessageList, ModelRequest, ToolCall, ToolResultContent,
};
use agent_rig::runner::{AgentEvent, AgentRunner};
use agent_rig::tools::{Tool, ToolDefinition, ToolRegistry, ToolResult};
use agent_rig::{Agent, models::ollama::OllamaModel};
use async_trait::async_trait;
use futures_util::StreamExt;
use schemars::{JsonSchema, json_schema};
use serde::Deserialize;
use serde_json::{Value, json};
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

fn ollama_url() -> String {
    let _ = dotenvy::dotenv();
    std::env::var("OLLAMA_URL").unwrap_or_else(|_| "http://localhost:11434".to_string())
}

fn ollama_model() -> String {
    std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "qwen3:8b".to_string())
}

/// Returns `true` only when the Ollama server is reachable *and* the model
/// returned by [`ollama_model`] is actually pulled. When either is missing the
/// tests skip (return early as a pass), so the suite is a clean no-op on a
/// machine without Ollama or without the configured model — instead of failing
/// with confusing empty-output assertions. Set `OLLAMA_MODEL` to a model you
/// have pulled (e.g. `OLLAMA_MODEL=gemma4:e2b`) to actually exercise them.
async fn ollama_available(url: &str) -> bool {
    let Ok(resp) = reqwest::get(format!("{url}/api/tags")).await else {
        return false;
    };
    let Ok(body) = resp.json::<Value>().await else {
        return false;
    };
    let want = ollama_model();
    body["models"].as_array().is_some_and(|models| {
        models.iter().any(|m| {
            m["name"].as_str().is_some_and(|name| {
                // Match the full tag (`qwen3:8b`) or a bare name (`qwen3`).
                name == want.as_str() || name.split(':').next() == Some(want.as_str())
            })
        })
    })
}

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

/// Checks, on the final thread, that every `add` call was answered with its
/// real sum, so the tool's result (not an error, and not another call's
/// result) is what went back to the model.
fn check_add_results(run: &ToolRun) -> TestResult {
    for call in run.calls.iter().filter(|call| call.name == "add") {
        let sum = call.args["a"].as_i64().unwrap_or(0) + call.args["b"].as_i64().unwrap_or(0);
        let expected = Vec::<ToolResultContent>::from(ToolResult::ok(json!({ "result": sum })));
        let sent = run
            .thread
            .iter()
            .find_map(|message| match &message.content {
                MessageContent::ToolResult { tool_call, content } if tool_call.id == call.id => {
                    Some(content)
                }
                _ => None,
            });
        if sent != Some(&expected) {
            return Err(format!("expected {expected:?} for call {call:?}, got: {sent:?}").into());
        }
    }
    Ok(())
}

/// Returns `true` if `text` contains `n` as a whole number, so `7` doesn't
/// match inside `17`.
fn contains_number(text: &str, n: i64) -> bool {
    let n = n.to_string();
    text.split(|c: char| !c.is_ascii_digit())
        .any(|token| token == n)
}

#[tokio::test]
async fn agent_run_returns_non_empty_output() {
    let url = ollama_url();
    if !ollama_available(&url).await {
        return;
    }

    let model = OllamaModel::new(&url, ollama_model());
    let agent = Agent::builder()
        .name("Greeter")
        .instructions("Reply with exactly one sentence.")
        .build();

    let runner = AgentRunner::new(Arc::new(model));
    let output = collect_text(runner, agent, "Say hello.").await;
    assert!(!output.is_empty());
}

#[tokio::test]
async fn agent_run_reports_token_usage() {
    let url = ollama_url();
    if !ollama_available(&url).await {
        return;
    }

    let model = OllamaModel::new(&url, ollama_model());
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

    let usage = got_usage.expect("Ollama must report token usage on done chunk");
    assert!(usage.input_tokens.unwrap_or(0) > 0);
    assert!(usage.output_tokens.unwrap_or(0) > 0);
}

#[tokio::test]
async fn agent_follows_system_instructions() {
    let url = ollama_url();
    if !ollama_available(&url).await {
        return;
    }

    let model = OllamaModel::new(&url, ollama_model());
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
    let url = ollama_url();
    if !ollama_available(&url).await {
        return;
    }

    #[derive(Deserialize, JsonSchema)]
    struct Sentiment {
        label: String,
        score: f32,
    }

    let model = OllamaModel::new(&url, ollama_model());
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
    let url = ollama_url();
    if !ollama_available(&url).await {
        return Ok(());
    }

    let model = OllamaModel::new(&url, ollama_model());
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
    check_add_results(&run)?;
    assert!(
        contains_number(&run.text, 42),
        "expected '42' in output, got: {}",
        run.text
    );
    Ok(())
}

/// Regression test for #43: a provider failure in the streaming path must
/// surface as an `Err` stream item, not a silently-empty stream.
///
/// Pointing the model at a port with nothing listening makes the underlying
/// `chat` request fail with a connection error. This needs no live Ollama
/// server and is fully deterministic. Before the fix the `?` inside the
/// `async_stream::stream!` block dropped the error and ended the generator
/// with zero items; now the error is yielded.
#[tokio::test]
async fn generate_stream_surfaces_provider_error() {
    // Port 1 has nothing listening, so the connection is refused immediately.
    let model = OllamaModel::new("http://127.0.0.1:1", "no-such-model");
    let request = ModelRequest {
        messages: vec![Message::user("hello")].into(),
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
        "stream ended silently with no items; the provider error was swallowed (#43)"
    );
    assert!(
        items.iter().any(|i| matches!(i, Err(Error::Provider(_)))),
        "expected a Provider error to be yielded, got: {items:?}"
    );
}

#[tokio::test]
async fn agent_run_with_generation_options() {
    let url = ollama_url();
    if !ollama_available(&url).await {
        return;
    }

    let model = OllamaModel::builder(&url, ollama_model())
        .temperature(0.1)
        .num_predict(512)
        .build();

    let agent = Agent::builder()
        .name("Assistant")
        .instructions("Be concise.")
        .build();

    let runner = AgentRunner::new(Arc::new(model));
    let output = collect_text(runner, agent, "What is 2 + 2?").await;
    assert!(!output.is_empty());
}
