// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

//! Demonstrates streaming with tool calls and thinking events on top of
//! [`MpscRunner`].
//!
//! Run with:
//! ```text
//! GEMINI_API_KEY=... cargo run --example streaming_agent
//! ```
//!
//! The example wires up a calculator agent that must use an `add` tool to
//! answer the question. Every event is printed as it happens:
//!
//! - `ToolCallStart` / `ToolCallFinish` — printed when the agent invokes
//!   the tool.
//! - `TextDelta` — printed incrementally as the model generates its answer.
//! - `ThinkingDelta` — printed if the model emits reasoning tokens (requires
//!   extended thinking enabled and a provider with native streaming).

use std::sync::Arc;

use agent_rig::model::Message;
use agent_rig::runner::{AgentEvent, AgentRunner};
use agent_rig::tools::{Tool, ToolDefinition, ToolRegistry, ToolResult};
use agent_rig::{Agent, models::gemini::GeminiModel};
use async_trait::async_trait;
use futures_util::StreamExt;
use geologia::prelude::{ThinkingConfig, ThinkingLevel};
use schemars::json_schema;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;
use tracing_subscriber::EnvFilter;

const MODEL: &str = "gemini-3.1-flash-lite";

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

    async fn apply(&self, args: Value, _cancel: CancellationToken) -> ToolResult {
        let a = args["a"].as_i64().unwrap_or(0);
        let b = args["b"].as_i64().unwrap_or(0);
        println!("[tool]  add({a}, {b}) = {}", a + b);
        ToolResult::ok(json!({ "result": a + b }))
    }
}

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
    let registry = Arc::new(ToolRegistry::new().register(AddTool::default()));

    let agent = Agent::builder()
        .name("Calculator")
        .instructions(
            "You are a calculator assistant. \
             Always use the `add` tool to compute sums — never calculate mentally.",
        )
        .tool("add")
        .build();

    let runner = AgentRunner::with_tools(Arc::new(model), registry.definitions());

    println!("Question: What is 1234 + 5678?\n");

    let mut stream = runner.run(&agent, vec![Message::user("What is 1234 + 5678?")].into());
    while let Some(event) = stream.next().await {
        match event.agent_event {
            AgentEvent::ThinkingDelta(token) => {
                print!("\x1b[2m{token}\x1b[0m");
            }
            AgentEvent::TextDelta(chunk) => {
                print!("{chunk}");
            }
            AgentEvent::ToolCall(call) => {
                println!("[runner] tool call started: {:?}", call.details);

                let Some(tool) = registry.get(&call.details.name) else {
                    call.resolve(ToolResult::error("Unknown tool"));
                    continue;
                };

                let result = tool
                    .apply(call.details.args.clone(), call.cancellation_token.clone())
                    .await;
                println!(
                    "[runner] tool call finished: {} → {result}",
                    call.details.name
                );
                call.resolve(result);
            }
            AgentEvent::Usage(usage) => println!("[runner] token usage: {usage:?}"),
            AgentEvent::Error(error) => eprintln!("\n[runner] stream error: {error}"),
            AgentEvent::Cancelled => println!("\n[runner] cancelled"),
            AgentEvent::TurnStart => {}
            AgentEvent::TurnFinish { .. } => {}
        }
    }

    println!();
    Ok(())
}
