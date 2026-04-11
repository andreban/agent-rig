//! Demonstrates `run_stream` with thinking tokens, tool invocation events, and
//! structured output all in one example.
//!
//! The agent is asked about temperatures in several cities. It must call the
//! `get_temperature` tool for each city, then respond with a structured JSON
//! summary that matches the `WeatherReport` schema.
//!
//! Every `AgentEvent` is printed as it arrives:
//! - `Thinking`           — dim grey reasoning tokens
//! - `ToolCallStarted`    — printed before the tool runs
//! - `ToolCallCompleted`  — printed after the tool returns
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

use async_trait::async_trait;
use futures_util::StreamExt;
use google_genai::prelude::{ThinkingConfig, ThinkingLevel};
use rust_agent_kit::{
    Agent, AgentEvent, AgentRunner,
    error::Error,
    models::gemini::GeminiModel,
    tool::{Tool, ToolDefinition, ToolRegistry},
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tracing_subscriber::EnvFilter;

const MODEL: &str = "gemini-3.1-flash-lite-preview";

// ---------------------------------------------------------------------------
// Output schema
// ---------------------------------------------------------------------------

/// Temperature reading for a single city.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct CityTemperature {
    /// City name.
    city: String,
    /// Temperature in Celsius.
    celsius: f64,
    /// Temperature in Fahrenheit.
    fahrenheit: f64,
}

/// Structured weather report returned by the agent.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct WeatherReport {
    /// List of cities with their current temperatures.
    cities: Vec<CityTemperature>,
    /// One-sentence summary of the overall conditions.
    summary: String,
}

// ---------------------------------------------------------------------------
// Tool: get_temperature
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
            thinking_level: Some(ThinkingLevel::Low),
            ..Default::default()
        })
        .build();

    let registry = Arc::new(ToolRegistry::new().register(Box::new(GetTemperatureTool)));

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

    let runner = AgentRunner::with_registry(Box::new(model), registry);

    let question = "What are the current temperatures in London, Tokyo, and Sydney?";
    println!("Question: {question}\n");

    let stream = runner.run_stream(&agent, question);
    futures_util::pin_mut!(stream);

    let mut output = String::new();
    let mut in_thinking = false;

    while let Some(event) = stream.next().await {
        match event? {
            AgentEvent::Thinking(token) => {
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
            AgentEvent::ToolCallStarted { name, args } => {
                if in_thinking {
                    println!("\x1b[0m");
                    in_thinking = false;
                }
                println!("[tool →] {name}({args})");
            }
            AgentEvent::ToolCallCompleted { name, result } => {
                println!("[tool ←] {name} = {result}");
            }
        }
    }

    println!("\n");

    // Deserialize the accumulated JSON into the typed struct.
    let report: WeatherReport = serde_json::from_str(&output)?;
    println!("--- Typed WeatherReport ---");
    for city in &report.cities {
        println!("  {}: {:.1}°C / {:.1}°F", city.city, city.celsius, city.fahrenheit);
    }
    println!("Summary: {}", report.summary);

    Ok(())
}
