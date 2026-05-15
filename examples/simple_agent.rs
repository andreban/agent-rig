// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use agent_rig::model::Message;
use agent_rig::runner::{AgentEvent, AgentRunner};
use agent_rig::{Agent, models::gemini::GeminiModel};
use futures_util::StreamExt;
use std::error::Error;
use tracing_subscriber::EnvFilter;

const MODEL: &str = "gemini-3.1-flash-lite";

const INSTRUCTIONS: &str = r#"
You are a research planning assistant

**TASK INSTRUCTIONS**
- You will be given a research topic.
- Your task is to provide a plan on how to research this topic.
- Output 5 concise tasks (5 words or less) to your plan.
"#;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let _ = dotenvy::dotenv();
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();
    let api_key = std::env::var("GEMINI_API_KEY")?;

    let model = GeminiModel::builder(api_key, MODEL)
        .temperature(0.8)
        .build();

    let agent = Agent::builder()
        .name("Research Planner")
        .instructions(INSTRUCTIONS)
        .build();

    let runner = AgentRunner::new(Arc::new(model));

    let mut stream = runner.run(agent, vec![Message::user("learn about AI agents")]);
    while let Some(event) = stream.next().await {
        match event {
            AgentEvent::TextDelta(chunk) => print!("{chunk}"),
            AgentEvent::Error(error) => eprintln!("\n[runner] stream error: {error}"),
            _ => {}
        }
    }
    println!();
    Ok(())
}
