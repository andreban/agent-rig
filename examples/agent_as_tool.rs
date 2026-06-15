// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

//! Demonstrates running an agent as a tool of another agent.
//!
//! A `Summariser` child agent is wrapped in an [`AgentTool`] and registered
//! with the parent runner via [`ToolRegistry::register`], like any other tool.
//! The parent `Orchestrator` agent calls the `summarise` tool to delegate
//! work; the child run is consumed internally and only its accumulated text
//! is returned as the tool result.

use std::sync::Arc;

use agent_rig::Agent;
use agent_rig::model::Message;
use agent_rig::models::gemini::GeminiModel;
use agent_rig::runner::{AgentEvent, AgentRunner};
use agent_rig::tools::{AgentTool, ToolDefinition, ToolRegistry};
use futures_util::StreamExt;
use schemars::json_schema;
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
            parameters: json_schema!({
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

    let registry = Arc::new(ToolRegistry::new().register(summariser_tool(&api_key)));

    let parent_model = GeminiModel::builder(&api_key, MODEL).build();
    let parent_runner = AgentRunner::with_tools(Arc::new(parent_model), registry.definitions());

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

    // Every event carries `run_id` and an `Option<usize>` parent. The root
    // run has `parent = None`; sub-agent runs have `parent = Some(...)`
    // pointing at the run that invoked them. We log every event with both
    // fields and accumulate the *root* run's TextDelta into the final
    // answer (so the child summariser's own tokens aren't double-counted).
    let mut answer = String::new();
    let mut stream = parent_runner.run(&parent_agent, vec![Message::user(input)]);
    while let Some(event) = stream.next().await {
        let run_id = event.run_id;
        let prefix = format!("[run={run_id}]");

        match event.agent_event {
            AgentEvent::ThinkingDelta(chunk) => {
                println!("{prefix} thinking: {chunk:?}");
            }
            AgentEvent::TextDelta(chunk) => {
                println!("{prefix} text:     {chunk:?}");
                answer.push_str(&chunk);
            }
            AgentEvent::ToolCall(call) => {
                println!("{prefix} started:  {}({})", call.tool_name, call.args);
                let tool_name = call.tool_name.clone();
                let result = match registry.get(&call.tool_name) {
                    Some(tool) => tool
                        .apply(call.args.clone(), call.cancellation_token.clone())
                        .await
                        .unwrap_or_else(|e| serde_json::Value::from(format!("Tool error: {e}"))),
                    None => serde_json::Value::from("Unknown tool"),
                };
                println!("{prefix} ok:       {tool_name} → {result}");
                call.resolve(result);
            }
            AgentEvent::Usage(usage) => {
                println!("{prefix} usage:    {usage:?}")
            }
            AgentEvent::Error(error) => {
                eprintln!("{prefix} error:    {error}")
            }
            AgentEvent::Cancelled => {
                println!("{prefix} cancelled")
            }
            AgentEvent::TurnStart => {}
            AgentEvent::TurnFinish { .. } => {}
        }
    }
    println!("\n--- final answer ---\n{answer}");
    Ok(())
}
