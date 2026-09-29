// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

//! Demonstrates a basic agent using Ollama.
//!
//! Run with:
//! ```bash
//! cargo run --example ollama_simple_agent --features ollama
//! ```

use std::error::Error;
use std::sync::Arc;

use agent_rig::model::Message;
use agent_rig::models::ollama::OllamaModel;
use agent_rig::runner::{AgentEvent, AgentRunner};
use agent_rig::Agent;
use futures_util::StreamExt;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let _ = dotenvy::dotenv();
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let server_url = std::env::var("OLLAMA_URL").unwrap_or_else(|_| "http://localhost:11434".into());
    let model_name = std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "llama3".into());

    let model = OllamaModel::builder(server_url, model_name)
        .temperature(0.7)
        .build();

    let agent = Agent::builder()
        .name("Ollama Assistant")
        .instructions("You are a concise, helpful assistant. Answer in 2-3 sentences.")
        .build();

    let runner = AgentRunner::new(Arc::new(model));

    let prompt = "What are the key benefits of Rust's ownership system?";
    println!("Prompt: {prompt}\n");

    let mut stream = runner.run(&agent, vec![Message::user(prompt)].into());
    while let Some(event) = stream.next().await {
        match event.agent_event {
            AgentEvent::TextDelta(chunk) => print!("{chunk}"),
            AgentEvent::Usage(usage) => println!("\n[runner] usage: {usage:?}"),
            AgentEvent::Error(err) => eprintln!("\n[runner] error: {err}"),
            _ => {}
        }
    }
    println!();
    Ok(())
}
