// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

//! Demonstrates long-term memory via tools in `agent-rig`.
//!
//! This example shows that "long-term memory" is not an inherent trait of the
//! LLM — it is simply an I/O operation facilitated by the [`Tool`] trait. The
//! agent is given two tools:
//!
//! - `remember_fact`: writes a fact to a shared in-memory store.
//! - `recall_fact`: searches the store for facts matching a keyword.
//!
//! Two simulated sessions are run back-to-back with an empty conversation
//! history between them (the way a real multi-session system works). The
//! model relies solely on its tools to persist and retrieve information across
//! the session boundary.
//!
//! Run with:
//! ```bash
//! GEMINI_API_KEY=your_key cargo run --example long_term_memory
//! ```

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use agent_rig::{
    Agent, AgentRunner,
    error::Error,
    models::gemini::GeminiModel,
    tool::{Tool, ToolDefinition, ToolRegistry},
};
use serde_json::{Value, json};
use tracing_subscriber::EnvFilter;

const MODEL: &str = "gemini-3.1-flash-lite-preview";

// ---------------------------------------------------------------------------
// Shared memory store
// ---------------------------------------------------------------------------

/// A thread-safe list of fact strings shared between the two memory tools.
type MemoryStore = Arc<Mutex<Vec<String>>>;

// ---------------------------------------------------------------------------
// Tool: remember_fact
// ---------------------------------------------------------------------------

/// Saves a self-contained fact to the shared [`MemoryStore`].
struct RememberFactTool {
    store: MemoryStore,
}

#[async_trait]
impl Tool for RememberFactTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "remember_fact".to_string(),
            description: "Saves an important fact about the user or the world for later recall."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "fact": {
                        "type": "string",
                        "description": "The self-contained fact to remember"
                    }
                },
                "required": ["fact"]
            }),
        }
    }

    async fn call(&self, args: Value) -> Result<Value, Error> {
        let fact = args["fact"]
            .as_str()
            .ok_or_else(|| Error::Agent("missing 'fact' argument".to_string()))?
            .to_string();
        println!("[memory] storing: "{fact}"");
        self.store
            .lock()
            .map_err(|e| Error::Agent(format!("lock poisoned: {e}")))?
            .push(fact);
        Ok(json!({ "status": "saved" }))
    }
}

// ---------------------------------------------------------------------------
// Tool: recall_fact
// ---------------------------------------------------------------------------

/// Searches the shared [`MemoryStore`] for facts that contain the given query
/// string (case-insensitive substring match).
struct RecallFactTool {
    store: MemoryStore,
}

#[async_trait]
impl Tool for RecallFactTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "recall_fact".to_string(),
            description: "Searches long-term memory for relevant facts. \
                          Pass one or more specific keywords (e.g. 'dog name' or 'favourite colour'). \
                          Each word is matched independently — a fact is returned if it contains \
                          any of the supplied words."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "One or more space-separated keywords to search for (e.g. 'dog name')"
                    }
                },
                "required": ["query"]
            }),
        }
    }

    async fn call(&self, args: Value) -> Result<Value, Error> {
        let query = args["query"]
            .as_str()
            .ok_or_else(|| Error::Agent("missing 'query' argument".to_string()))?
            .to_lowercase();
        // Split into individual words so that a query like "dog name" matches
        // facts containing *either* "dog" or "name".
        let terms: Vec<&str> = query.split_whitespace().collect();
        let store = self
            .store
            .lock()
            .map_err(|e| Error::Agent(format!("lock poisoned: {e}")))?;
        let results: Vec<&str> = store
            .iter()
            .filter(|fact| {
                let lower = fact.to_lowercase();
                terms.iter().any(|term| lower.contains(term))
            })
            .map(String::as_str)
            .collect();
        println!("[memory] recall("{query}") → {} result(s)", results.len());
        Ok(json!({ "results": results }))
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

    // Shared store — the only state that survives across "sessions".
    let store: MemoryStore = Arc::new(Mutex::new(Vec::new()));

    let registry = Arc::new(
        ToolRegistry::new()
            .register(Box::new(RememberFactTool { store: store.clone() }))
            .register(Box::new(RecallFactTool { store: store.clone() })),
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

    let runner = AgentRunner::with_registry(
        Box::new(GeminiModel::new(api_key, MODEL)),
        registry,
    );

    // -----------------------------------------------------------------------
    // Session 1 — establishing memory
    // -----------------------------------------------------------------------
    println!("=== Session 1 ===
");
    let input1 = "My dog's name is Barnaby.";
    println!("User: {input1}");
    let result1 = runner.run(&agent, input1).await?;
    println!("Assistant: {}
", result1.output);

    // -----------------------------------------------------------------------
    // Session 2 — retrieving memory with a fresh (empty) conversation history
    // -----------------------------------------------------------------------
    println!("=== Session 2 (new session — no conversation history) ===
");
    let input2 = "Do you remember what kind of pet I have and its name?";
    println!("User: {input2}");
    let result2 = runner.run(&agent, input2).await?;
    println!("Assistant: {}
", result2.output);

    Ok(())
}