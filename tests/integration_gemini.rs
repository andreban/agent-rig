// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use agent_rig::error::Error;
use agent_rig::model::{LlmModel, Message, ModelRequest};
use agent_rig::runner::{AgentEvent, AgentRunner};
use agent_rig::tools::{ProgressReporter, Tool, ToolDefinition, ToolRegistry};
use agent_rig::{Agent, models::gemini::GeminiModel};
use async_trait::async_trait;
use futures_util::StreamExt;
use schemars::{JsonSchema, json_schema};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

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

    async fn apply(&self, args: Value, _progress: &dyn ProgressReporter, _cancel: CancellationToken) -> Result<Value, Error> {
        let a = args["a"].as_i64().unwrap_or(0);
        let b = args["b"].as_i64().unwrap_or(0);
        Ok(json!({ "result": a + b }))
    }
}

const MODEL: &str = "gemini-3.1-flash-lite";

fn api_key() -> Option<String> {
    let _ = dotenvy::dotenv();
    std::env::var("GEMINI_API_KEY").ok()
}

/// Drives the runner to completion and concatenates the streamed text.
async fn collect_text(runner: AgentRunner, agent: Agent, prompt: &str) -> String {
    let mut text = String::new();
    let mut stream = runner.run(&agent, vec![Message::user(prompt)]);
    while let Some(event) = stream.next().await {
        if let AgentEvent::TextDelta(chunk) = event.agent_event {
            text.push_str(&chunk);
        }
    }
    text
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
async fn agent_tool_calling_returns_correct_result() {
    let Some(api_key) = api_key() else { return };

    let model = GeminiModel::new(api_key, MODEL);
    let registry = Arc::new(ToolRegistry::new().register(AddTool::default()));
    let agent = Agent::builder()
        .name("Calculator")
        .instructions("You are a calculator. Use the add tool to compute sums. When asked to add numbers, call the tool and report the result.")
        .tool("add")
        .build();

    let runner = AgentRunner::with_registry(Arc::new(model), registry);
    let output = collect_text(runner, agent, "What is 17 + 25?").await;

    assert!(
        output.contains("42"),
        "expected '42' in output, got: {output}"
    );
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
    let mut stream = runner.run(&agent, vec![Message::user("Say hello.")]);

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
            "ghost-call".to_string(),
            "ghost".to_string(),
            json!({ "ok": true }),
            None,
        )],
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
