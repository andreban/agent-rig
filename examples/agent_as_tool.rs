// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

use agent_rig::models::gemini::GeminiModel;
use agent_rig::tool::{ToolDefinition, ToolRegistry};
use agent_rig::{Agent, AgentRunner, AgentTool};
use serde_json::json;
use std::error::Error;
use std::sync::Arc;
use tracing_subscriber::EnvFilter;

const MODEL: &str = "gemini-3.1-flash-lite";

/// Child agent: summarises text passed in the `text` field of its JSON input.
fn summariser_tool(api_key: &str) -> AgentTool {
    let model = GeminiModel::builder(api_key, MODEL).build();
    let agent = Agent::builder()
        .name("Summariser")
        .instructions(
            "You receive a JSON object with a `text` field. \
             Summarise the text in two sentences or fewer.",
        )
        .build();
    let runner = AgentRunner::new(Box::new(model));
    AgentTool::new(
        ToolDefinition {
            name: "summarise".to_string(),
            description: "Summarises a long piece of text into two sentences or fewer. \
                          Pass the text in the `text` field."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "text": { "type": "string", "description": "The text to summarise." }
                },
                "required": ["text"]
            }),
        },
        agent,
        runner,
    )
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let _ = dotenvy::dotenv();
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();
    let api_key = std::env::var("GEMINI_API_KEY")?;

    // Build the parent runner with the summariser child agent registered as a tool.
    let registry = Arc::new(ToolRegistry::new().register(Box::new(summariser_tool(&api_key))));

    let parent_model = GeminiModel::builder(&api_key, MODEL).build();
    let parent_runner = AgentRunner::with_registry(Box::new(parent_model), registry);

    let parent_agent = Agent::builder()
        .name("Orchestrator")
        .instructions(
            "You are a research assistant. When asked to summarise something, \
             use the `summarise` tool. Return only the summary.",
        )
        .tool("summarise")
        .build();

    let input = "Please summarise the following article: \
        Rust is a systems programming language focused on three goals: safety, speed, and \
        concurrency. It accomplishes these goals without a garbage collector, making it useful \
        for a number of use cases other languages aren't good at: embedding in other languages, \
        programs with specific space and time requirements, and writing low-level code, like \
        device drivers and operating systems.";

    let result = parent_runner.run(&parent_agent, input).await?;
    println!("{}", result.output);
    Ok(())
}
