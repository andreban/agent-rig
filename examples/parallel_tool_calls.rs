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
//! The `run_stream` output also illustrates the new event ordering:
//! all `ToolCallStarted` events fire before any `ToolCallCompleted` event,
//! confirming that the calls are in flight simultaneously.
//!
//! Run with:
//! ```bash
//! GEMINI_API_KEY=your_key cargo run --example parallel_tool_calls
//! ```

use std::sync::Arc;
use std::time::Instant;

use agent_rig::{
    Agent, AgentEvent, AgentRunner,
    error::Error,
    models::gemini::GeminiModel,
    tool::{Tool, ToolDefinition, ToolRegistry},
};
use async_trait::async_trait;
use futures_util::StreamExt;
use serde_json::{Value, json};
use tracing_subscriber::EnvFilter;

const MODEL: &str = "gemini-3.1-flash-lite";

// ---------------------------------------------------------------------------
// Tool: get_temperature
// Simulates a slow remote weather API (500 ms round-trip).
// ---------------------------------------------------------------------------

struct GetTemperatureTool;

#[async_trait]
impl Tool for GetTemperatureTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "get_temperature".to_string(),
            description: "Returns the current temperature in Celsius for the given city."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The name of the city"
                    }
                },
                "required": ["city"]
            }),
        }
    }

    async fn call(&self, args: Value) -> Result<Value, Error> {
        let city = args["city"].as_str().unwrap_or("unknown").to_string();

        // Simulate a 500 ms network round-trip.
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        let celsius = match city.to_lowercase().as_str() {
            "london" => 12.0_f64,
            "tokyo" => 27.0_f64,
            "sydney" => 21.0_f64,
            _ => 18.0_f64,
        };

        println!("[tool]  get_temperature({city}) → {celsius}°C  (after 500 ms delay)");
        Ok(json!({ "city": city, "celsius": celsius }))
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

    let registry = Arc::new(ToolRegistry::new().register(Box::new(GetTemperatureTool)));

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

    let runner = AgentRunner::with_registry(Box::new(GeminiModel::new(api_key, MODEL)), registry);

    let question = "What are the current temperatures in London, Tokyo, and Sydney?";
    println!("Question: {question}");
    println!(
        "(Each tool call has a simulated 500 ms delay — parallel = ~500 ms total)
"
    );

    let start = Instant::now();
    let stream = runner.run_stream(&agent, question);
    futures_util::pin_mut!(stream);

    while let Some(event) = stream.next().await {
        match event? {
            AgentEvent::ToolCallStarted { name, args } => {
                println!("[runner] started:   {name}({})", args);
            }
            AgentEvent::ToolCallCompleted { name, result } => {
                println!("[runner] completed: {name} → {result}");
            }
            AgentEvent::TextDelta(chunk) => {
                print!("{chunk}");
            }
            AgentEvent::Thinking(_) => {}
        }
    }

    println!();
    println!(
        "
Total elapsed: {:.0?}",
        start.elapsed()
    );
    println!("  3 × 500 ms in parallel ≈ 500 ms  (sequential would be ≈ 1 500 ms)");

    Ok(())
}
