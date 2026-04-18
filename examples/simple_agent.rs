// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

use agent_rig::{Agent, AgentRunner, models::gemini::GeminiModel};
use std::error::Error;
use tracing_subscriber::EnvFilter;

const MODEL: &str = "gemini-3.1-flash-lite-preview";

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
    tracing_subscriber::fmt().with_env_filter(EnvFilter::from_default_env()).init();
    let api_key = std::env::var("GEMINI_API_KEY")?;

    let model = GeminiModel::builder(api_key, MODEL)
        .temperature(0.8)
        .build();

    let agent = Agent::builder()
        .name("Research Planner")
        .instructions(INSTRUCTIONS)
        .build();

    let input = "learn about AI agents";
    let result = AgentRunner::new(Box::new(model)).run(&agent, input).await?;
    println!("{}", result.output);
    Ok(())
}