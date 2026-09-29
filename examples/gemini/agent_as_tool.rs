// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

//! Demonstrates running an agent as a tool of another agent.
//!
//! A `Summariser` child agent is wrapped in an [`AgentTool`] and registered
//! with the parent runner via [`ToolRegistry::register`], like any other tool.
//! The child has a `word_count` tool of its own, passed through
//! [`AgentTool::with_tools`]; `AgentTool` runs the child's tool calls and
//! returns only the child's final reply as the tool result. The child's token
//! usage is reported through [`AgentTool::on_usage`].

use std::sync::Arc;

use agent_rig::Agent;
use agent_rig::model::{Message, ToolCall};
use agent_rig::models::gemini::GeminiModel;
use agent_rig::runner::{AgentEvent, AgentRunner, ToolCallResult};
use agent_rig::tools::{AgentTool, Tool, ToolDefinition, ToolRegistry, ToolResult};
use async_trait::async_trait;
use futures_util::StreamExt;
use schemars::json_schema;
use serde_json::{Value, json};
use std::error::Error;
use tokio_util::sync::CancellationToken;
use tracing_subscriber::EnvFilter;

const MODEL: &str = "gemini-3.1-flash-lite";

/// Counts the words in the `text` argument.
struct WordCountTool {
    definition: ToolDefinition,
}

impl Default for WordCountTool {
    fn default() -> Self {
        Self {
            definition: ToolDefinition {
                name: "word_count".to_string(),
                description: "Counts the words in a piece of text.".to_string(),
                parameters: json_schema!({
                    "type": "object",
                    "properties": {
                        "text": { "type": "string", "description": "The text to count." }
                    },
                    "required": ["text"]
                }),
            },
        }
    }
}

#[async_trait]
impl Tool for WordCountTool {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn call(&self, tool_call: Arc<ToolCall>, _cancel: CancellationToken) -> ToolResult {
        match tool_call.args.get("text").and_then(Value::as_str) {
            Some(text) => ToolResult::ok(json!({ "words": text.split_whitespace().count() })),
            None => ToolResult::error("missing `text` argument"),
        }
    }
}

fn summariser_tool(api_key: &str) -> AgentTool {
    let model = GeminiModel::builder(api_key, MODEL).build();
    let agent = Agent::builder()
        .name("Summariser")
        .instructions(
            "You receive a JSON object with a `text` field. \
             Summarise the text in two sentences or fewer. \
             Use the `word_count` tool to check the original length and \
             mention it in your summary.",
        )
        .tool("word_count")
        .build();
    let tools = Arc::new(ToolRegistry::new().register(WordCountTool::default()));
    AgentTool::with_tools(
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
        Arc::new(model),
        tools,
    )
    .on_usage(|usage| println!("[summariser] usage: {usage:?}"))
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

    // Only the parent run's events reach this stream: the summariser's run
    // (including its `word_count` calls) is driven inside `AgentTool::call`.
    let mut answer = String::new();
    let mut stream = parent_runner.run(&parent_agent, vec![Message::user(input)].into());
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
                println!("{prefix} started:  {:?}", call.details);
                let tool_name = call.details.name.clone();
                let result: Value = match registry.get(&call.details.name) {
                    Some(tool) => tool
                        .call(call.details.clone(), call.cancellation_token.clone())
                        .await
                        .into(),
                    None => ToolCallResult::Unknown.into(),
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
