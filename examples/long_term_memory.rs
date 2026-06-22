// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

//! Demonstrates long-term memory via tools in `agent-rig`.
//!
//! "Long-term memory" is not an inherent trait of the LLM — it is simply an
//! I/O operation facilitated by the [`Tool`] trait. The agent is given two
//! tools:
//!
//! - `remember_fact`: writes a fact to a shared in-memory store.
//! - `recall_fact`:  searches the store for facts matching a keyword.
//!
//! Two simulated sessions are run back-to-back with an empty conversation
//! history between them (the way a real multi-session system works). The
//! model relies solely on its tools to persist and retrieve information
//! across the session boundary.
//!
//! Run with:
//! ```bash
//! GEMINI_API_KEY=your_key cargo run --example long_term_memory
//! ```

use std::sync::{Arc, Mutex};

use agent_rig::model::Message;
use agent_rig::runner::{AgentEvent, AgentRunner};
use agent_rig::tools::{Tool, ToolDefinition, ToolRegistry, ToolResult};
use agent_rig::{Agent, models::gemini::GeminiModel};
use async_trait::async_trait;
use futures_util::StreamExt;
use schemars::json_schema;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;
use tracing_subscriber::EnvFilter;

const MODEL: &str = "gemini-3.1-flash-lite";

type MemoryStore = Arc<Mutex<Vec<String>>>;

struct RememberFactTool {
    definition: ToolDefinition,
    store: MemoryStore,
}

impl RememberFactTool {
    fn new(store: MemoryStore) -> Self {
        Self {
            definition: ToolDefinition {
                name: "remember_fact".to_string(),
                description:
                    "Saves an important fact about the user or the world for later recall."
                        .to_string(),
                parameters: json_schema!({
                    "type": "object",
                    "properties": {
                        "fact": {
                            "type": "string",
                            "description": "The self-contained fact to remember"
                        }
                    },
                    "required": ["fact"]
                }),
            },
            store,
        }
    }
}

#[async_trait]
impl Tool for RememberFactTool {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn apply(&self, args: Value, _cancel: CancellationToken) -> ToolResult {
        let fact = match args.get("fact") {
            Some(fact) => fact,
            None => return ToolResult::error("missing 'fact' argument"),
        };

        println!("[memory] storing: \"{fact}\"");
        match self.store.lock() {
            Ok(mut lock) => {
                lock.push(fact.to_string());
                ToolResult::ok(json!({ "status": "saved" }))
            }
            Err(e) => ToolResult::error(e.to_string()),
        }
    }
}

struct RecallFactTool {
    definition: ToolDefinition,
    store: MemoryStore,
}

impl RecallFactTool {
    fn new(store: MemoryStore) -> Self {
        Self {
            definition: ToolDefinition {
                name: "recall_fact".to_string(),
                description: "Searches long-term memory for relevant facts. \
                          Pass one or more specific keywords (e.g. 'dog name' or 'favourite colour'). \
                          Each word is matched independently — a fact is returned if it contains \
                          any of the supplied words."
                    .to_string(),
                parameters: json_schema!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "One or more space-separated keywords to search for (e.g. 'dog name')"
                        }
                    },
                    "required": ["query"]
                }),
            },
            store,
        }
    }
}

#[async_trait]
impl Tool for RecallFactTool {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn apply(&self, args: Value, _cancel: CancellationToken) -> ToolResult {
        let query = match args.get("query") {
            Some(query) => query.to_string().to_lowercase(),
            None => return ToolResult::error("missing 'query' argument"),
        };

        let terms: Vec<&str> = query.split_whitespace().collect();
        let store = match self.store.lock() {
            Ok(store) => store,
            Err(e) => return ToolResult::error(format!("lock poisoned: {e}")),
        };
        let results: Vec<&str> = store
            .iter()
            .filter(|fact| {
                let lower = fact.to_lowercase();
                terms.iter().any(|term| lower.contains(term))
            })
            .map(String::as_str)
            .collect();
        println!("[memory] recall(\"{query}\") → {} result(s)", results.len());
        ToolResult::ok(json!({ "results": results }))
    }
}

async fn run_once(
    runner: &AgentRunner,
    agent: &Agent,
    registry: &Arc<ToolRegistry>,
    input: &str,
) -> String {
    let mut reply = String::new();
    let mut stream = runner.run(agent, vec![Message::user(input)]);
    while let Some(event) = stream.next().await {
        match event.agent_event {
            AgentEvent::TextDelta(chunk) => reply.push_str(&chunk),
            AgentEvent::Usage(usage) => println!("[runner] usage: {usage:?}"),
            AgentEvent::Error(error) => eprintln!("[runner] stream error: {error}"),
            AgentEvent::ToolCall(call) => {
                let result = match registry.get(&call.details.name) {
                    Some(tool) => {
                        tool.apply(call.details.args.clone(), call.cancellation_token.clone())
                            .await
                    }
                    None => ToolResult::error("Unknown tool"),
                };
                call.resolve(result);
            }
            _ => {}
        }
    }
    reply
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = dotenvy::dotenv();
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();
    let api_key = std::env::var("GEMINI_API_KEY")?;

    let store: MemoryStore = Arc::new(Mutex::new(Vec::new()));

    let registry = Arc::new(
        ToolRegistry::new()
            .register(RememberFactTool::new(store.clone()))
            .register(RecallFactTool::new(store.clone())),
    );

    let agent = Agent::builder()
        .name("Memory Assistant")
        .instructions(
            "You are a helpful assistant with access to a long-term memory store. \
             Whenever the user tells you something worth remembering — a personal fact, \
             a preference, or any detail they mention — proactively call `remember_fact` \
             to save it. \
             When the user asks about something you might have stored, call `recall_fact` \
             before answering. Use specific, concrete keywords in your recall queries \
             (e.g. 'dog', 'cat', 'name', 'colour') rather than abstract words like 'pet' \
             or 'animal'. If the first search returns no results, try again with a \
             different keyword.",
        )
        .tool("remember_fact")
        .tool("recall_fact")
        .build();

    let runner = AgentRunner::with_tools(
        Arc::new(GeminiModel::new(api_key, MODEL)),
        registry.definitions(),
    );

    println!("=== Session 1 ===\n");
    let input1 = "My dog's name is Barnaby.";
    println!("User: {input1}");
    let reply1 = run_once(&runner, &agent, &registry, input1).await;
    println!("Assistant: {reply1}\n");

    println!("=== Session 2 (new session — no conversation history) ===\n");
    let input2 = "Do you remember what kind of pet I have and its name?";
    println!("User: {input2}");
    let reply2 = run_once(&runner, &agent, &registry, input2).await;
    println!("Assistant: {reply2}\n");

    Ok(())
}
