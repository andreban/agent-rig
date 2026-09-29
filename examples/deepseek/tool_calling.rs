// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

//! Demonstrates tool calling with DeepSeek (`deepseek-chat`).
//!
//! The agent is given a `get_weather` tool and asked a question that triggers tool execution.
//! The consumer resolves the tool call and streams back the synthesized response.
//!
//! Run with:
//! ```bash
//! cargo run --example deepseek_tool_calling --features deepseek
//! ```

use std::error::Error;
use std::sync::Arc;

use agent_rig::model::{Message, ToolCall};
use agent_rig::models::deepseek::DeepSeekModel;
use agent_rig::runner::{AgentEvent, AgentRunner};
use agent_rig::tools::{Tool, ToolDefinition, ToolRegistry, ToolResult};
use agent_rig::Agent;
use async_trait::async_trait;
use futures_util::StreamExt;
use schemars::json_schema;
use serde_json::json;
use tokio_util::sync::CancellationToken;
use tracing_subscriber::EnvFilter;

const MODEL: &str = "deepseek-chat";

struct GetWeatherTool {
    definition: ToolDefinition,
}

impl Default for GetWeatherTool {
    fn default() -> Self {
        Self {
            definition: ToolDefinition {
                name: "get_weather".to_string(),
                description: "Returns the current weather and temperature in Celsius for a given city.".to_string(),
                parameters: json_schema!({
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "Name of the city, e.g. London, Paris, Tokyo"
                        }
                    },
                    "required": ["city"]
                }),
            },
        }
    }
}

#[async_trait]
impl Tool for GetWeatherTool {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn call(&self, tool_call: Arc<ToolCall>, _cancel: CancellationToken) -> ToolResult {
        let city = tool_call.args["city"].as_str().unwrap_or("unknown");
        println!("[tool] Fetching weather for {city}...");
        let temp = match city.to_lowercase().as_str() {
            "tokyo" => 24.0,
            "london" => 16.0,
            "paris" => 19.0,
            _ => 20.0,
        };
        ToolResult::ok(json!({ "city": city, "temperature_celsius": temp, "condition": "Sunny" }))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let _ = dotenvy::dotenv();
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let api_key = std::env::var("DEEPSEEK_API_KEY")
        .expect("DEEPSEEK_API_KEY must be set in .env or environment");

    let registry = Arc::new(ToolRegistry::new().register(GetWeatherTool::default()));

    let agent = Agent::builder()
        .name("Weather Guide")
        .instructions("You are a helpful travel assistant. Always use the `get_weather` tool when asked about weather in any city.")
        .tool("get_weather")
        .build();

    let model = Arc::new(DeepSeekModel::new(api_key, MODEL));
    let runner = AgentRunner::with_tools(model, registry.definitions());

    let prompt = "What is the weather like in Tokyo right now?";
    println!("User: {prompt}\n");

    let mut stream = runner.run(&agent, vec![Message::user(prompt)].into());

    while let Some(event) = stream.next().await {
        match event.agent_event {
            AgentEvent::ToolCall(call) => {
                println!("[runner] tool call requested: {}({})", call.details.name, call.details.args);
                let result = if let Some(tool) = registry.get(&call.details.name) {
                    tool.call(call.details.clone(), call.cancellation_token.clone()).await
                } else {
                    ToolResult::error("Unknown tool")
                };
                println!("[runner] resolving tool call with result: {result}");
                call.resolve(result);
            }
            AgentEvent::TextDelta(chunk) => {
                print!("{chunk}");
            }
            AgentEvent::Usage(usage) => {
                println!("\n\n--- Token Usage ---");
                println!("Input tokens: {:?}", usage.input_tokens);
                println!("Output tokens: {:?}", usage.output_tokens);
                println!("Cached prompt tokens: {:?}", usage.cached_input_tokens);
            }
            AgentEvent::Error(err) => eprintln!("\n[runner] error: {err}"),
            _ => {}
        }
    }

    println!();
    Ok(())
}
