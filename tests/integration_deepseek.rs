// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use agent_rig::Agent;
use agent_rig::error::Error;
use agent_rig::model::{LlmModel, Message, ModelRequest, ToolCall};
use agent_rig::models::deepseek::DeepSeekModel;
use agent_rig::runner::{AgentEvent, AgentRunner};
use agent_rig::tools::{Tool, ToolDefinition, ToolRegistry, ToolResult};
use async_trait::async_trait;
use futures_util::StreamExt;
use schemars::{JsonSchema, json_schema};
use serde::Deserialize;
use serde_json::json;
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

    async fn call(&self, tool_call: Arc<ToolCall>, _cancel: CancellationToken) -> ToolResult {
        let a = tool_call.args["a"].as_i64().unwrap_or(0);
        let b = tool_call.args["b"].as_i64().unwrap_or(0);
        ToolResult::ok(json!({ "result": a + b }))
    }
}

const MODEL: &str = "deepseek-chat";

fn api_key() -> Option<String> {
    let _ = dotenvy::dotenv();
    std::env::var("DEEPSEEK_API_KEY").ok()
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

#[tokio::test]
async fn agent_run_returns_non_empty_output() {
    let Some(key) = api_key() else {
        println!("SKIPPED: DEEPSEEK_API_KEY not set");
        return;
    };

    let model = Arc::new(DeepSeekModel::new(key, MODEL));
    let runner = AgentRunner::new(model);

    let agent = Agent::builder()
        .name("test-agent")
        .instructions("Answer in one sentence.")
        .build();

    let output = collect_text(runner, agent, "What is Rust?").await;
    assert!(!output.is_empty(), "expected non-empty output");
}

#[tokio::test]
async fn agent_follows_system_instructions() {
    let Some(key) = api_key() else {
        println!("SKIPPED: DEEPSEEK_API_KEY not set");
        return;
    };

    let model = Arc::new(DeepSeekModel::new(key, MODEL));
    let runner = AgentRunner::new(model);

    let agent = Agent::builder()
        .name("pirate-agent")
        .instructions(
            "You are a pirate. Speak only in pirate dialect. Use words like 'ahoy' and 'matey'.",
        )
        .build();

    let output = collect_text(runner, agent, "Hello, who are you?")
        .await
        .to_lowercase();
    assert!(
        output.contains("ahoy") || output.contains("matey") || output.contains("pirate"),
        "expected pirate language, got: {output}"
    );
}

#[tokio::test]
async fn agent_run_reports_token_usage() {
    let Some(key) = api_key() else {
        println!("SKIPPED: DEEPSEEK_API_KEY not set");
        return;
    };

    let model = Arc::new(DeepSeekModel::new(key, MODEL));
    let runner = AgentRunner::new(model);

    let agent = Agent::builder()
        .name("usage-agent")
        .instructions("Say hello.")
        .build();

    let mut stream = runner.run(&agent, vec![Message::user("Hi")].into());
    let mut usage_found = false;
    while let Some(event) = stream.next().await {
        if let AgentEvent::Usage(usage) = event.agent_event {
            assert!(usage.input_tokens.is_some());
            assert!(usage.output_tokens.is_some());
            usage_found = true;
        }
    }
    assert!(usage_found, "expected at least one Usage event");
}

#[tokio::test]
async fn agent_tool_calling_returns_correct_result() {
    let Some(key) = api_key() else {
        println!("SKIPPED: DEEPSEEK_API_KEY not set");
        return;
    };

    let model = Arc::new(DeepSeekModel::new(key, MODEL));
    let registry = Arc::new(ToolRegistry::new().register(AddTool::default()));
    let runner = AgentRunner::with_tools(model, registry.definitions());

    let agent = Agent::builder()
        .name("calc-agent")
        .instructions("You have an `add` tool. Always use the `add` tool to perform addition. Never calculate in your head.")
        .tool("add")
        .build();

    let mut text = String::new();
    let mut stream = runner.run(
        &agent,
        vec![Message::user("What is 42 + 58? Use the add tool.")].into(),
    );
    while let Some(event) = stream.next().await {
        match event.agent_event {
            AgentEvent::ToolCall(call) => {
                let result = if let Some(tool) = registry.get(&call.details.name) {
                    tool.call(call.details.clone(), call.cancellation_token.clone())
                        .await
                } else {
                    ToolResult::error("Unknown tool")
                };
                call.resolve(result);
            }
            AgentEvent::TextDelta(chunk) => {
                text.push_str(&chunk);
            }
            _ => {}
        }
    }
    assert!(
        text.contains("100"),
        "expected tool result 100 in output, got: {text}"
    );
}

#[derive(Debug, Deserialize, JsonSchema)]
struct MathAnswer {
    result: i64,
    explanation: String,
}

#[tokio::test]
async fn agent_output_schema_returns_valid_json() {
    let Some(key) = api_key() else {
        println!("SKIPPED: DEEPSEEK_API_KEY not set");
        return;
    };

    let model = Arc::new(DeepSeekModel::new(key, MODEL));
    let runner = AgentRunner::new(model);

    let agent = Agent::builder()
        .name("schema-agent")
        .instructions("Solve the math problem.")
        .output_schema(schemars::schema_for!(MathAnswer))
        .build();

    let output = collect_text(runner, agent, "What is 6 * 7?").await;
    let parsed: MathAnswer = serde_json::from_str(&output)
        .unwrap_or_else(|e| panic!("expected valid JSON matching MathAnswer, got `{output}`: {e}"));

    assert_eq!(parsed.result, 42);
    assert!(!parsed.explanation.is_empty());
}

#[tokio::test]
async fn generate_stream_surfaces_provider_error() {
    let bad_model = DeepSeekModel::new("invalid_key_for_testing", MODEL);
    let request = ModelRequest {
        messages: vec![Message::user("Hello")].into(),
        system: None,
        output_schema: None,
        tools: vec![],
    };

    let mut stream = bad_model.generate_stream(request);
    let mut saw_error = false;
    while let Some(chunk_result) = stream.next().await {
        if let Err(Error::Provider(_)) = chunk_result {
            saw_error = true;
            break;
        }
    }
    assert!(saw_error, "expected Error::Provider from invalid API key");
}
