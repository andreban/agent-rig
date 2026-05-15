// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

//! A streaming REPL demonstrating multi-turn conversation on top of
//! [`MpscRunner`].
//!
//! `MpscRunner::run` takes ownership of the message thread for one run, so this
//! example keeps the running history locally: each turn appends the user input
//! to the thread, runs the agent, accumulates the assistant's reply from
//! `TextDelta` events, and finally pushes a single
//! `Message::assistant(reply)` so the next turn sees the full context.
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

    let mut thread: Vec<Message> = Vec::new();
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

        thread.push(Message::user(input));

        print!("Assistant: ");
        io::stdout().flush()?;

        let mut reply = String::new();
        let mut stream = runner.run(agent.clone(), thread.clone());
        while let Some(event) = stream.next().await {
            match event {
                AgentEvent::ThinkingDelta(token) => {
                    print!("\x1b[2m{token}\x1b[0m");
                    io::stdout().flush()?;
                }
                AgentEvent::TextDelta(chunk) => {
                    print!("{chunk}");
                    io::stdout().flush()?;
                    reply.push_str(&chunk);
                }
                AgentEvent::Error(error) => {
                    eprintln!("\n[runner] stream error: {error}");
                }
                _ => {}
            }
        }
        thread.push(Message::assistant(reply));
        println!("\n");

        print!("You: ");
        io::stdout().flush()?;
    }

    Ok(())
}
