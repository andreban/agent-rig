// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

//! Demonstrates a basic streaming agent using Claude Haiku 5.5 (`claude-haiku-5-5`).
//!
//! Run with:
//! ```bash
//! cargo run --example anthropic_chat --features anthropic
//! ```

use std::error::Error;
use std::sync::Arc;

use agent_rig::Agent;
use agent_rig::model::Message;
use agent_rig::models::anthropic::{AnthropicModel, Effort};
use agent_rig::runner::{AgentEvent, AgentRunner};
use futures_util::StreamExt;
use tracing_subscriber::EnvFilter;

const MODEL: &str = "claude-haiku-5-5";

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let _ = dotenvy::dotenv();
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .expect("ANTHROPIC_API_KEY must be set in .env or environment");

    let model = AnthropicModel::builder(api_key, MODEL)
        .effort(Effort::Low)
        .build();

    let agent = Agent::builder()
        .name("Claude Assistant")
        .instructions("You are a concise, helpful coding assistant. Answer questions clearly with code examples.")
        .build();

    let runner = AgentRunner::new(Arc::new(model));

    let prompt = "Explain Rust's RAII pattern in three bullet points with a short code snippet.";
    println!("Prompt: {prompt}\n");

    let mut stream = runner.run(&agent, vec![Message::user(prompt)].into());
    while let Some(event) = stream.next().await {
        match event.agent_event {
            AgentEvent::TextDelta(chunk) => print!("{chunk}"),
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
