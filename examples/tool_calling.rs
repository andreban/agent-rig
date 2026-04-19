// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

//! Demonstrates tool calling with `agent-rig`.
//!
//! The agent is given two tools — `get_temperature` and `celsius_to_fahrenheit`
//! — and asked a question that requires calling both in sequence. The runner
//! handles the agentic loop automatically.
//!
//! Run with:
//! ```bash
//! GEMINI_API_KEY=your_key cargo run --example tool_calling
//! ```

use std::sync::Arc;

use agent_rig::{
    Agent, AgentRunner,
    error::Error,
    models::gemini::GeminiModel,
    tool::{Tool, ToolDefinition, ToolRegistry},
};
use async_trait::async_trait;
use serde_json::{Value, json};
use tracing_subscriber::EnvFilter;

const MODEL: &str = "gemini-3.1-flash-lite-preview";

// ---------------------------------------------------------------------------
// Tool: get_temperature
// Returns the current temperature in Celsius for a given city (stubbed).
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
        // Stubbed data — a real implementation would call a weather API.
        let celsius = match city.to_lowercase().as_str() {
            "london" => 15.0,
            "tokyo" => 28.0,
            "sydney" => 22.0,
            _ => 20.0,
        };
        println!("[tool] get_temperature({city}) → {celsius}°C");
        Ok(json!({ "city": city, "celsius": celsius }))
    }
}

// ---------------------------------------------------------------------------
// Tool: celsius_to_fahrenheit
// Converts a Celsius value to Fahrenheit.
// ---------------------------------------------------------------------------

struct CelsiusToFahrenheitTool;

#[async_trait]
impl Tool for CelsiusToFahrenheitTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "celsius_to_fahrenheit".to_string(),
            description: "Converts a temperature from Celsius to Fahrenheit.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "celsius": {
                        "type": "number",
                        "description": "Temperature in Celsius"
                    }
                },
                "required": ["celsius"]
            }),
        }
    }

    async fn call(&self, args: Value) -> Result<Value, Error> {
        let celsius = args["celsius"].as_f64().unwrap_or(0.0);
        let fahrenheit = celsius * 9.0 / 5.0 + 32.0;
        println!("[tool] celsius_to_fahrenheit({celsius}) → {fahrenheit}°F");
        Ok(json!({ "fahrenheit": fahrenheit }))
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

    let registry = Arc::new(
        ToolRegistry::new()
            .register(Box::new(GetTemperatureTool))
            .register(Box::new(CelsiusToFahrenheitTool)),
    );

    let agent = Agent::builder()
        .name("Weather Assistant")
        .instructions(
            "You are a helpful weather assistant. \
             Use the available tools to answer questions about current temperatures. \
             Always convert to Fahrenheit when asked.",
        )
        .tool("get_temperature")
        .tool("celsius_to_fahrenheit")
        .build();

    let runner = AgentRunner::with_registry(Box::new(GeminiModel::new(api_key, MODEL)), registry);

    let question = "What is the current temperature in Tokyo in Fahrenheit?";
    println!(
        "Question: {question}
"
    );

    let result = runner.run(&agent, question).await?;
    println!(
        "
Answer: {}",
        result.output
    );

    Ok(())
}
