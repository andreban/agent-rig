// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

//! Demonstrates parallel tool execution in `agent-rig`.
//!
//! When a model returns multiple tool calls in a single response, the runner
//! executes them all concurrently rather than one at a time. This example
//! makes the parallelism visible by:
//!
//! - Giving each tool call an artificial 500 ms delay (simulating a slow API).
//! - Asking the model for temperatures in three cities at once, so it (ideally)
//!   batches all three `get_temperature` calls into one model turn.
//! - Measuring wall-clock time: ~500 ms total instead of the ~1 500 ms that
//!   sequential execution would take.
//!
//! All `ToolCallStart` events fire before any `ToolCallFinish` event,
//! confirming that the calls are in flight simultaneously.
//!
//! Run with:
//! ```bash
//! GEMINI_API_KEY=your_key cargo run --example parallel_tool_calls
//! ```

use std::sync::Arc;
use std::time::Instant;

use agent_rig::model::Message;
use agent_rig::runner::{AgentEvent, AgentRunner};
use agent_rig::tools::{Tool, ToolDefinition, ToolRegistry, ToolResult};
use agent_rig::{Agent, models::gemini::GeminiModel};
use async_trait::async_trait;
use futures_util::StreamExt;
use schemars::json_schema;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;
use tracing_subscriber::EnvFilter;

const MODEL: &str = "gemini-3.1-flash-lite";

struct GetTemperatureTool {
    definition: ToolDefinition,
}

impl Default for GetTemperatureTool {
    fn default() -> Self {
        Self {
            definition: ToolDefinition {
                name: "get_temperature".to_string(),
                description: "Returns the current temperature in Celsius for the given city."
                    .to_string(),
                parameters: json_schema!({
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "The name of the city"
                        }
                    },
                    "required": ["city"]
                }),
            },
        }
    }
}

#[async_trait]
impl Tool for GetTemperatureTool {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn apply(&self, args: Value, _cancel: CancellationToken) -> ToolResult {
        let city = args["city"].as_str().unwrap_or("unknown").to_string();

        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        let celsius = match city.to_lowercase().as_str() {
            "london" => 12.0_f64,
            "tokyo" => 27.0_f64,
            "sydney" => 21.0_f64,
            _ => 18.0_f64,
        };

        println!("[tool]  get_temperature({city}) → {celsius}°C  (after 500 ms delay)");
        ToolResult::ok(json!({ "city": city, "celsius": celsius }))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = dotenvy::dotenv();
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let api_key = std::env::var("GEMINI_API_KEY")?;

    let registry = Arc::new(ToolRegistry::new().register(GetTemperatureTool::default()));

    let agent = Agent::builder()
        .name("Weather Assistant")
        .instructions(
            "You are a weather assistant. \
             When asked about multiple cities, call get_temperature for ALL of them \
             in a single response (do not wait for one result before requesting the next). \
             Report all temperatures in Celsius.",
        )
        .tool("get_temperature")
        .build();

    let runner = AgentRunner::with_tools(
        Arc::new(GeminiModel::new(api_key, MODEL)),
        registry.definitions(),
    );

    let question = "What are the current temperatures in London, Tokyo, and Sydney?";
    println!("Question: {question}");
    println!("(Each tool call has a simulated 500 ms delay — parallel = ~500 ms total)\n");

    let start = Instant::now();
    let mut stream = runner.run(&agent, vec![Message::user(question)]);

    while let Some(event) = stream.next().await {
        match event.agent_event {
            AgentEvent::ToolCall(call) => {
                println!(
                    "[runner] started:   #{} {}({})",
                    call.details.id, call.details.name, call.details.args
                );
                let tool_name = call.details.name.clone();
                let tool_call_id = call.details.id.clone();
                let registry = registry.clone();
                // Spawn each tool call so they run in parallel.
                tokio::spawn(async move {
                    let Some(tool) = registry.get(&tool_name) else {
                        call.resolve(ToolResult::error("Uknown Tool"));
                        return;
                    };

                    let result = tool
                        .apply(call.details.args.clone(), call.cancellation_token.clone())
                        .await;
                    println!("[runner] finished:  #{tool_call_id} {tool_name} → {result}");
                    call.resolve(result);
                });
            }
            AgentEvent::TextDelta(chunk) => print!("{chunk}"),
            AgentEvent::ThinkingDelta(_) => {}
            AgentEvent::Usage(usage) => println!("\n[runner] usage:     {usage:?}"),
            AgentEvent::Error(error) => eprintln!("\n[runner] stream error: {error}"),
            AgentEvent::Cancelled => println!("\n[runner] cancelled"),
            AgentEvent::TurnStart => {}
            AgentEvent::TurnFinish { .. } => {}
        }
    }

    println!();
    println!("\nTotal elapsed: {:.0?}", start.elapsed());
    println!("  3 × 500 ms in parallel ≈ 500 ms  (sequential would be ≈ 1 500 ms)");

    Ok(())
}
