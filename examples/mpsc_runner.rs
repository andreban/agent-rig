use std::sync::Arc;
use std::time::Instant;

use agent_rig::error::Error;
use agent_rig::model::Message;
use agent_rig::mpsc_runner::{AgentEvent, MpscRunner};
use agent_rig::tool::{Tool, ToolDefinition, ToolRegistry};
use agent_rig::{Agent, models::gemini::GeminiModel};
use async_trait::async_trait;
use futures_util::StreamExt;
use serde_json::{Value, json};
use tracing_subscriber::EnvFilter;

const MODEL: &str = "gemini-3.1-flash-lite";

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

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = dotenvy::dotenv();
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let api_key = std::env::var("GEMINI_API_KEY")?;
    let model = GeminiModel::new(api_key, MODEL);
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

    let runner = MpscRunner::with_registry(Arc::new(model), registry);

    let question = "What are the current temperatures in London, Tokyo, and Sydney?";
    println!("Question: {question}");
    println!("(Each tool call has a simulated 500 ms delay — parallel = ~500 ms total)\n");

    let start = Instant::now();
    let mut stream = runner.run(agent, vec![Message::user(question)]);

    while let Some(event) = stream.next().await {
        match event {
            AgentEvent::ToolCallStarted { name, args } => {
                println!("[runner] started:   {name}({args})");
            }
            AgentEvent::ToolCallFinished { name, result } => {
                println!("[runner] finished:  {name} → {result}");
            }
            AgentEvent::ToolCallError { name, error } => {
                println!("[runner] error:     {name} → {error}");
            }
            AgentEvent::ToolCallDenied { name, reason } => {
                println!("[runner] denied:    {name} → {reason}");
            }
            AgentEvent::TextDelta(chunk) => {
                print!("{chunk}");
            }
            AgentEvent::ThinkingDelta(_) => {}
            AgentEvent::Error(error) => {
                eprintln!("\n[runner] stream error: {error}");
            }
        }
    }

    println!();
    println!("\nTotal elapsed: {:.0?}", start.elapsed());
    println!("  3 × 500 ms in parallel ≈ 500 ms  (sequential would be ≈ 1 500 ms)");

    Ok(())
}
