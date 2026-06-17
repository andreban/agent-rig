// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

//! Combines streaming, tool calls, thinking events, and structured output on
//! top of [`MpscRunner`].
//!
//! The agent is asked about temperatures in several cities. It must call the
//! `get_temperature` tool for each city, then respond with a structured JSON
//! summary that matches the `WeatherReport` schema.
//!
//! Every event is printed as it arrives:
//! - `ThinkingDelta`      — dim grey reasoning tokens
//! - `ToolCallStart`      — printed before the tool runs
//! - `ToolCallFinish`     — printed after the tool returns
//! - `TextDelta`          — the (JSON) answer arriving incrementally
//!
//! After the stream ends, the accumulated text is deserialized into
//! `WeatherReport` for typed access.
//!
//! Run with:
//! ```text
//! GEMINI_API_KEY=<key> cargo run --example streaming_structured
//! ```

use std::sync::Arc;

use agent_rig::model::Message;
use agent_rig::runner::{AgentEvent, AgentRunner};
use agent_rig::tools::{Tool, ToolDefinition, ToolRegistry, ToolResult};
use agent_rig::{Agent, models::gemini::GeminiModel};
use async_trait::async_trait;
use futures_util::StreamExt;
use geologia::prelude::{ThinkingConfig, ThinkingLevel};
use schemars::JsonSchema;
use schemars::json_schema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;
use tracing::info;
use tracing_subscriber::EnvFilter;

const MODEL: &str = "gemini-3.1-flash-lite";

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct CityTemperature {
    city: String,
    celsius: f64,
    fahrenheit: f64,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct WeatherReport {
    cities: Vec<CityTemperature>,
    summary: String,
}

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
        let city = args["city"].as_str().unwrap_or("unknown");
        let celsius = match city.to_lowercase().as_str() {
            "london" => 12.0,
            "tokyo" => 27.0,
            "sydney" => 21.0,
            "new york" => 18.0,
            _ => 20.0,
        };
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

    let model = GeminiModel::builder(api_key, MODEL)
        .thinking_config(ThinkingConfig {
            include_thoughts: true,
            thinking_level: Some(ThinkingLevel::Low),
            ..Default::default()
        })
        .build();

    let registry = Arc::new(ToolRegistry::new().register(GetTemperatureTool::default()));

    let agent = Agent::builder()
        .name("Weather Reporter")
        .instructions(
            "You are a weather reporting assistant. \
             Use the `get_temperature` tool to look up the current temperature for each city \
             requested. Convert Celsius to Fahrenheit yourself (F = C × 9/5 + 32). \
             Return a structured report.",
        )
        .tool("get_temperature")
        .output_schema(schemars::schema_for!(WeatherReport))
        .build();

    let runner = AgentRunner::with_tools(Arc::new(model), registry.definitions());

    let question = "What are the current temperatures in London, Tokyo, and Sydney?";
    println!("Question: {question}\n");

    let mut output = String::new();
    let mut in_thinking = false;

    let mut stream = runner.run(&agent, vec![Message::user(question)]);
    while let Some(event) = stream.next().await {
        match event.agent_event {
            AgentEvent::ThinkingDelta(token) => {
                if !in_thinking {
                    print!("\x1b[2m[thinking] ");
                    in_thinking = true;
                }
                print!("\x1b[2m{token}\x1b[0m");
            }
            AgentEvent::TextDelta(chunk) => {
                if in_thinking {
                    println!("\x1b[0m");
                    in_thinking = false;
                }
                print!("{chunk}");
                output.push_str(&chunk);
            }
            AgentEvent::ToolCall(call) => {
                info!(?call, "AgentEvent::ToolCall");
                let Some(tool) = registry.get(&call.tool_name) else {
                    call.resolve(ToolResult::error("Unknown Tool"));
                    continue;
                };
                let result = tool
                    .apply(call.args.clone(), call.cancellation_token.clone())
                    .await;
                call.resolve(result);
            }
            AgentEvent::Usage(usage) => println!("[usage] {usage:?}"),
            AgentEvent::Error(error) => eprintln!("\n[runner] stream error: {error}"),
            AgentEvent::Cancelled => println!("\n[runner] cancelled"),
            AgentEvent::TurnStart => {}
            AgentEvent::TurnFinish { .. } => {}
        }
    }

    println!("\n");

    let report: WeatherReport = serde_json::from_str(&output)?;
    println!("--- Typed WeatherReport ---");
    for city in &report.cities {
        println!(
            "  {}: {:.1}°C / {:.1}°F",
            city.city, city.celsius, city.fahrenheit
        );
    }
    println!("Summary: {}", report.summary);

    Ok(())
}
