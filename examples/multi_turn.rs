// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

//! A streaming REPL demonstrating multi-turn conversation on top of
//! [`AgentRunner`].
//!
//! Each turn appends the user input to the thread and passes it to the runner.
//! When the run completes, [`AgentEvent::TurnFinish`] delivers the full updated
//! thread — including any tool-call / tool-result pairs and the final assistant
//! message — ready to pass straight back on the next turn. No manual message
//! reconstruction needed.
//!
//! Run with:
//!   GEMINI_API_KEY=... cargo run --example multi_turn

use std::io::{self, BufRead, Write};
use std::sync::Arc;

use agent_rig::model::Message;
use agent_rig::runner::{AgentEvent, AgentRunner};
use agent_rig::{Agent, models::gemini::GeminiModel};
use futures_util::StreamExt;
use geologia::prelude::{ThinkingConfig, ThinkingLevel};
use std::error::Error;
use tracing_subscriber::EnvFilter;

const MODEL: &str = "gemini-3.1-flash-lite";

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let _ = dotenvy::dotenv();
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();
    let api_key = std::env::var("GEMINI_API_KEY")?;

    let model = GeminiModel::builder(api_key, MODEL)
        .thinking_config(ThinkingConfig {
            include_thoughts: true,
            thinking_level: Some(ThinkingLevel::High),
            ..Default::default()
        })
        .build();
    let agent = Agent::builder()
        .name("Assistant")
        .instructions("You are a helpful assistant. Keep replies concise.")
        .build();
    let runner = AgentRunner::new(Arc::new(model));

    let mut thread: Vec<Arc<Message>> = Vec::new();
    let stdin = io::stdin();

    println!("Multi-turn chat (Ctrl-C or Ctrl-D to quit)\n");
    print!("You: ");
    io::stdout().flush()?;

    for line in stdin.lock().lines() {
        let input = line?;
        let input = input.trim().to_string();
        if input.is_empty() {
            print!("You: ");
            io::stdout().flush()?;
            continue;
        }

        thread.push(Arc::new(Message::user(input)));

        print!("Assistant: ");
        io::stdout().flush()?;

        let mut stream = runner.run(&agent, thread);
        thread = Vec::new();
        while let Some(event) = stream.next().await {
            match event.agent_event {
                AgentEvent::ThinkingDelta(token) => {
                    print!("\x1b[2m{token}\x1b[0m");
                    io::stdout().flush()?;
                }
                AgentEvent::TextDelta(chunk) => {
                    print!("{chunk}");
                    io::stdout().flush()?;
                }
                AgentEvent::TurnFinish {
                    thread: updated, ..
                } => {
                    thread = updated;
                }
                AgentEvent::Usage(usage) => {
                    println!("\n[runner] token usage: {usage:?}");
                }
                AgentEvent::Error(error) => {
                    eprintln!("\n[runner] stream error: {error}");
                }
                _ => {}
            }
        }
        println!("\n");

        print!("You: ");
        io::stdout().flush()?;
    }

    Ok(())
}
