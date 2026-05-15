// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use agent_rig::error::Error;
use agent_rig::model::Message;
use agent_rig::runner::{AgentEvent, AgentRunner};
use agent_rig::tools::{Tool, ToolDefinition, ToolRegistry};
use agent_rig::{Agent, models::ollama::OllamaModel};
use async_trait::async_trait;
use futures_util::StreamExt;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

struct AddTool;

#[async_trait]
impl Tool for AddTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "add".to_string(),
            description: "Adds two integers and returns their sum.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "a": { "type": "integer", "description": "First operand" },
                    "b": { "type": "integer", "description": "Second operand" }
                },
                "required": ["a", "b"]
            }),
        }
    }

    async fn call(&self, args: Value) -> Result<Value, Error> {
        let a = args["a"].as_i64().unwrap_or(0);
        let b = args["b"].as_i64().unwrap_or(0);
        Ok(json!({ "result": a + b }))
    }
}

fn ollama_url() -> String {
    let _ = dotenvy::dotenv();
    std::env::var("OLLAMA_URL").unwrap_or_else(|_| "http://localhost:11434".to_string())
}

fn ollama_model() -> String {
    std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "qwen3:8b".to_string())
}

async fn ollama_available(url: &str) -> bool {
    reqwest::get(format!("{url}/api/version")).await.is_ok()
}

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
async fn agent_tool_calling_returns_correct_result() {
    let url = ollama_url();
    if !ollama_available(&url).await {
        return;
    }

    let model = OllamaModel::new(&url, ollama_model());
    let registry = Arc::new(ToolRegistry::new().register(Box::new(AddTool)));
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
