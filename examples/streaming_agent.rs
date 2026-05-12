// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

//! Demonstrates `AgentRunner::run_stream` with tool calls and thinking events.
//!
//! Run with:
//! ```text
//! GEMINI_API_KEY=... cargo run --example streaming_agent
//! ```
//!
//! The example wires up a calculator agent that must use an `add` tool to
//! answer the question. `run_stream` is used so every event is visible as it
//! happens:
//!
//! - `ToolCallStarted` / `ToolCallCompleted` — printed when the agent invokes
//!   the tool.
//! - `TextDelta` — printed incrementally as the model generates its answer.
//! - `Thinking` — printed if the model emits reasoning tokens. This requires
//!   both a model with extended thinking enabled *and* a provider adapter with
//!   a native `generate_stream` implementation. With the current `GeminiModel`
//!   (which uses the default stream wrapper), thinking tokens from the model
//!   are not yet surfaced; they will appear once native Gemini streaming is
//!   added.

use std::sync::Arc;

use agent_rig::{
    Agent, AgentEvent, AgentRunner,
    error::Error,
    models::gemini::GeminiModel,
    tool::{Tool, ToolDefinition, ToolRegistry},
};
use async_trait::async_trait;
use futures_util::StreamExt;
use geologia::prelude::{ThinkingConfig, ThinkingLevel};
use serde_json::{Value, json};
use tracing_subscriber::EnvFilter;

const MODEL: &str = "gemini-3.1-flash-lite";

// ---------------------------------------------------------------------------
// Tool: add two integers
// ---------------------------------------------------------------------------

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
        println!("[tool]  add({a}, {b}) = {}", a + b);
        Ok(json!({ "result": a + b }))
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = dotenvy::dotenv();
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let api_key = std::env::var("GEMINI_API_KEY")?;

    let model = GeminiModel::builder(api_key, MODEL)
        .thinking_config(ThinkingConfig {
            include_thoughts: true,
            thinking_level: Some(ThinkingLevel::High),
            ..Default::default()
        })
        .build();
    let registry = Arc::new(ToolRegistry::new().register(Box::new(AddTool)));

    let agent = Agent::builder()
        .name("Calculator")
        .instructions(
            "You are a calculator assistant. \
             Always use the `add` tool to compute sums — never calculate mentally.",
        )
        .tool("add")
        .build();

    let runner = AgentRunner::with_registry(Box::new(model), registry);

    println!(
        "Question: What is 1234 + 5678?
"
    );

    let stream = runner.run_stream(&agent, "What is 1234 + 5678?");
    futures_util::pin_mut!(stream);

    while let Some(event) = stream.next().await {
        match event? {
            AgentEvent::Thinking(token) => {
                // Reasoning tokens — only emitted by models with extended
                // thinking enabled once native Gemini streaming is in place.
                print!("[2m{token}[0m"); // dim text
            }
            AgentEvent::TextDelta(chunk) => {
                print!("{chunk}");
            }
            AgentEvent::ToolCallStarted { name, args } => {
                println!("[runner] tool call started: {name}({args})");
            }
            AgentEvent::ToolCallCompleted { name, result } => {
                println!("[runner] tool call completed: {name} → {result}");
            }
        }
    }

    println!(); // newline after streamed output
    Ok(())
}
