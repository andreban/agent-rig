// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

//! Demonstrates running an agent as a tool of another agent on top of
//! [`MpscRunner`].
//!
//! A `Summariser` child agent is wrapped in an [`AgentTool`] and registered
//! with the parent runner via [`ToolRegistry::register_agent`]. The parent
//! `Orchestrator` agent calls the `summarise` tool to delegate work; the
//! parent runner streams the child's events through to the consumer.

use std::sync::Arc;

use agent_rig::Agent;
use agent_rig::model::Message;
use agent_rig::models::gemini::GeminiModel;
use agent_rig::runner::{AgentEvent, AgentRunner, ToolCallResult};
use agent_rig::tools::{AgentTool, ToolDefinition, ToolRegistry};
use futures_util::StreamExt;
use serde_json::json;
use std::error::Error;
use tracing_subscriber::EnvFilter;

const MODEL: &str = "gemini-3.1-flash-lite";

fn summariser_tool(api_key: &str) -> AgentTool {
    let model = GeminiModel::builder(api_key, MODEL).build();
    let agent = Agent::builder()
        .name("Summariser")
        .instructions(
            "You receive a JSON object with a `text` field. \
             Summarise the text in two sentences or fewer.",
        )
        .build();
    let runner = AgentRunner::new(Arc::new(model));
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

    let registry = Arc::new(ToolRegistry::new().register_agent(summariser_tool(&api_key)));

    let parent_model = GeminiModel::builder(&api_key, MODEL).build();
    let parent_runner = AgentRunner::with_registry(Arc::new(parent_model), registry);

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

    let mut answer = String::new();
    let mut stream = parent_runner.run(parent_agent, vec![Message::user(input)]);
    while let Some(event) = stream.next().await {
        match event {
            AgentEvent::TextDelta(chunk) => answer.push_str(&chunk),
            AgentEvent::ToolCallStarted { name, args } => {
                println!("[runner] started:   {name}({args})");
            }
            AgentEvent::ToolCallFinished { name, result } => match result {
                ToolCallResult::Ok(value) => println!("[runner] finished:  {name} → {value}"),
                ToolCallResult::Err(error) => println!("[runner] error:     {name} → {error:?}"),
                ToolCallResult::Denied => println!("[runner] denied:    {name}"),
                ToolCallResult::Unknown => println!("[runner] unknown:   {name}"),
            },
            AgentEvent::Error(error) => eprintln!("[runner] stream error: {error}"),
            AgentEvent::ThinkingDelta(_) => {}
        }
    }
    println!("\n{answer}");
    Ok(())
}
