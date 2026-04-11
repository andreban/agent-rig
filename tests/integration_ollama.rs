use std::sync::Arc;

use async_trait::async_trait;
use rust_agent_kit::{
    Agent, AgentRunner,
    error::Error,
    models::ollama::OllamaModel,
    tool::{Tool, ToolDefinition, ToolRegistry},
};
use serde_json::{Value, json};
use schemars::JsonSchema;
use serde::Deserialize;

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

    let result = AgentRunner::new(Box::new(model))
        .run(&agent, "Say hello.")
        .await
        .unwrap();
    assert!(!result.output.is_empty());
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

    let result = AgentRunner::new(Box::new(model))
        .run(&agent, "How are you?")
        .await
        .unwrap();
    assert!(result.output.to_lowercase().contains("arrr"));
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

    let schema = schemars::schema_for!(Sentiment);

    let model = OllamaModel::new(&url, ollama_model());
    let agent = Agent::builder()
        .name("Classifier")
        .instructions("Classify the sentiment of the input. Return a label (positive/negative/neutral) and a confidence score between 0 and 1.")
        .output_schema(schema)
        .build();

    let result = AgentRunner::new(Box::new(model))
        .run(&agent, "I love sunny days!")
        .await
        .unwrap();

    let parsed: Sentiment = serde_json::from_str(&result.output).unwrap();
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

    let result = AgentRunner::with_registry(Box::new(model), registry)
        .run(&agent, "What is 17 + 25?")
        .await
        .unwrap();

    assert!(result.output.contains("42"), "expected '42' in output, got: {}", result.output);
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

    let result = AgentRunner::new(Box::new(model))
        .run(&agent, "What is 2 + 2?")
        .await
        .unwrap();
    assert!(!result.output.is_empty());
}
