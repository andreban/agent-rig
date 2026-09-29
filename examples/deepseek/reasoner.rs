// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

//! Demonstrates DeepSeek-R1 (`deepseek-reasoner`) reasoning mode with real-time thinking tokens.
//!
//! Run with:
//! ```bash
//! cargo run --example deepseek_reasoner --features deepseek
//! ```

use std::error::Error;
use std::sync::Arc;

use agent_rig::model::Message;
use agent_rig::models::deepseek::DeepSeekModel;
use agent_rig::runner::{AgentEvent, AgentRunner};
use agent_rig::Agent;
use futures_util::StreamExt;
use tracing_subscriber::EnvFilter;

const MODEL: &str = "deepseek-reasoner";

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let _ = dotenvy::dotenv();
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let api_key = std::env::var("DEEPSEEK_API_KEY")
        .expect("DEEPSEEK_API_KEY must be set in .env or environment");

    let model = DeepSeekModel::new(api_key, MODEL);

    let agent = Agent::builder()
        .name("DeepSeek Reasoner")
        .instructions("You are a rigorous reasoning model. Carefully think step-by-step through puzzles and logic questions.")
        .build();

    let runner = AgentRunner::new(Arc::new(model));

    let question = "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?";
    println!("Question: {question}\n");

    let mut in_thinking = false;
    let mut stream = runner.run(&agent, vec![Message::user(question)].into());

    while let Some(event) = stream.next().await {
        match event.agent_event {
            AgentEvent::ThinkingDelta(token) => {
                if !in_thinking {
                    println!("\x1b[36;1m[Reasoning Trace]\x1b[0m\x1b[2m");
                    in_thinking = true;
                }
                print!("{token}");
            }
            AgentEvent::TextDelta(chunk) => {
                if in_thinking {
                    println!("\x1b[0m\n\n\x1b[32;1m[Final Answer]\x1b[0m");
                    in_thinking = false;
                }
                print!("{chunk}");
            }
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
