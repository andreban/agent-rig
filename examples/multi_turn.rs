// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

use agent_rig::{Agent, AgentEvent, AgentRunner, models::gemini::GeminiModel};
use futures_util::StreamExt;
use geologia::prelude::{ThinkingConfig, ThinkingLevel};
use std::{
    error::Error,
    io::{self, BufRead, Write},
};
use tracing_subscriber::EnvFilter;

const MODEL: &str = "gemini-3.1-flash-lite-preview";

/// A simple streaming REPL that demonstrates multi-turn conversation using
/// [`Conversation::run_stream`].
///
/// History is managed automatically by the [`Conversation`] — no manual
/// message tracking needed. Each completed stream updates the internal history
/// so subsequent turns see the full context.
///
/// Run with:
///   GEMINI_API_KEY=... cargo run --example multi_turn
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
    let runner = AgentRunner::new(Box::new(model));
    let mut conv = runner.conversation(&agent);

    let stdin = io::stdin();

    println!(
        "Multi-turn chat (Ctrl-C or Ctrl-D to quit)
"
    );
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

        print!("Assistant: ");
        io::stdout().flush()?;

        let stream = conv.run_stream(&input);
        futures_util::pin_mut!(stream);

        while let Some(event) = stream.next().await {
            match event? {
                AgentEvent::Thinking(token) => {
                    print!("[2m{token}[0m"); // dim text
                    io::stdout().flush()?;
                }
                AgentEvent::TextDelta(chunk) => {
                    print!("{chunk}");
                    io::stdout().flush()?;
                }
                _ => {}
            }
        }
        println!(
            "
"
        );

        print!("You: ");
        io::stdout().flush()?;
    }

    Ok(())
}
