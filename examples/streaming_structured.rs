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

use agent_rig::error::Error;
use agent_rig::model::Message;
use agent_rig::runner::{AgentEvent, AgentRunner, ToolCallResult};
use agent_rig::tools::{ProgressReporter, Tool, ToolDefinition, ToolRegistry};
use agent_rig::{Agent, models::gemini::GeminiModel};
use async_trait::async_trait;
use futures_util::StreamExt;
use geologia::prelude::{ThinkingConfig, ThinkingLevel};
use schemars::JsonSchema;
use schemars::json_schema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;
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

    async fn apply(
        &self,
        args: Value,
        _progress: &dyn ProgressReporter,
        _cancel: CancellationToken,
    ) -> Result<Value, Error> {
        let city = args["city"].as_str().unwrap_or("unknown");
        let celsius = match city.to_lowercase().as_str() {
            "london" => 12.0,
            "tokyo" => 27.0,
            "sydney" => 21.0,
            "new york" => 18.0,
            _ => 20.0,
        };
        Ok(json!({ "city": city, "celsius": celsius }))
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

    let runner = AgentRunner::with_registry(Arc::new(model), registry);

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
            AgentEvent::ToolCallStart { tool_name, args, .. } => {
                if in_thinking {
                    println!("\x1b[0m");
                    in_thinking = false;
                }
                println!("[tool →] {name}({args})");
            }
            AgentEvent::ToolCallUpdate {
                tool_name, details, ..
            } => {
                println!("[tool →] {name}({details:?})");
            }

            AgentEvent::ToolCallFinish {
                tool_name, result, ..
            } => match result {
                ToolCallResult::Ok(value) => println!("[tool ←] {name} = {value}"),
                ToolCallResult::Err(error) => println!("[tool ✗] {name} → {error:?}"),
                ToolCallResult::Denied => println!("[tool ⨯] {name} denied"),
                ToolCallResult::Unknown => println!("[tool ?] {name} unknown"),
            },
            AgentEvent::Usage(usage) => println!("[usage] {usage:?}"),
            AgentEvent::Error(error) => eprintln!("\n[runner] stream error: {error}"),
            AgentEvent::Cancelled => println!("\n[runner] cancelled"),
            AgentEvent::TurnStart => {}
            AgentEvent::TurnFinish { .. } => {}
            AgentEvent::ApprovalRequest(request) => {
                request.respond(true);
            }
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
