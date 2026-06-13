use std::sync::Arc;
use std::time::Instant;

use agent_rig::error::Error;
use agent_rig::model::Message;
use agent_rig::runner::{AgentEvent, AgentRunner, ToolCallResult};
use agent_rig::tools::ToolRegistry;
use agent_rig::tools::{ProgressReporter, Tool, ToolDefinition};
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

    async fn call(&self, args: Value, _progress: &dyn ProgressReporter, _cancel: CancellationToken) -> Result<Value, Error> {
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

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = dotenvy::dotenv();
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let api_key = std::env::var("GEMINI_API_KEY")?;
    let model = GeminiModel::new(api_key, MODEL);
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

    let runner = AgentRunner::with_registry(Arc::new(model), registry);

    let question = "What are the current temperatures in London, Tokyo, and Sydney?";
    println!("Question: {question}");
    println!("(Each tool call has a simulated 500 ms delay — parallel = ~500 ms total)\n");

    let start = Instant::now();
    let mut stream = runner.run(&agent, vec![Message::user(question)]);

    while let Some(event) = stream.next().await {
        let run_id = event.run_id;
        match event.agent_event {
            AgentEvent::ToolCallStarted { name, args, .. } => {
                println!("[runner[{run_id}]] started:   {name}({args})");
            }
            AgentEvent::ToolCallUpdate { name, details, .. } => {
                println!("[runner[{run_id}]] started:   {name}({details:?})");
            }
            AgentEvent::ToolCallFinished { name, result, .. } => match result {
                ToolCallResult::Ok(result) => {
                    println!("[runner[{run_id}]] finished:  {name} → {result}");
                }
                ToolCallResult::Err(error) => {
                    println!("[runner[{run_id}]] finished:  {name} → {error:?}");
                }
                ToolCallResult::Denied => {
                    println!("[runner[{run_id}]] denied:    {name}");
                }
                ToolCallResult::Unknown => {
                    println!("[runner[{run_id}]] unknown:    {name}");
                }
            },
            AgentEvent::TextDelta(chunk) => {
                print!("{chunk}");
            }
            AgentEvent::ThinkingDelta(thinking) => {
                print!("{thinking}")
            }
            AgentEvent::Usage(usage) => {
                println!("\n[runner[{run_id}]] usage:     {usage:?}");
            }
            AgentEvent::Error(error) => {
                eprintln!("\n[runner[{run_id}]] stream error: {error}");
            }
            AgentEvent::Cancelled => {
                println!("\n[runner[{run_id}]] cancelled");
            }
            AgentEvent::StartTurn => {}
            AgentEvent::EndTurn { .. } => {}
        }
    }

    println!();
    println!("\nTotal elapsed: {:.0?}", start.elapsed());
    println!("  3 × 500 ms in parallel ≈ 500 ms  (sequential would be ≈ 1 500 ms)");

    Ok(())
}
