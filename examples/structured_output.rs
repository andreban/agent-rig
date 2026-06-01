// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

//! Demonstrates structured output on top of [`MpscRunner`].
//!
//! `schemars` generates the JSON Schema from `ResearchPlan` and it is set on
//! the agent so the model is constrained to produce matching JSON. The example
//! accumulates the streamed `TextDelta` chunks into a single string and
//! deserializes it into `ResearchPlan`.
//!
//! Run with:
//! ```text
//! GEMINI_API_KEY=<key> cargo run --example structured_output
//! ```

use std::sync::Arc;

use agent_rig::model::Message;
use agent_rig::runner::{AgentEvent, AgentRunner};
use agent_rig::{Agent, models::gemini::GeminiModel};
use futures_util::StreamExt;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::error::Error;
use tracing_subscriber::EnvFilter;

const MODEL: &str = "gemini-3.1-flash-lite";

const INSTRUCTIONS: &str = "\
You are a research planning assistant. \
Given a research topic, produce a structured plan with a title and exactly 5 concise tasks.";

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ResearchTask {
    /// Short imperative description of the task (5 words or less).
    task: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ResearchPlan {
    /// Title summarising the research topic.
    title: String,
    /// Ordered list of research tasks.
    tasks: Vec<ResearchTask>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let _ = dotenvy::dotenv();
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();
    let api_key = std::env::var("GEMINI_API_KEY")?;

    let model = GeminiModel::builder(api_key, MODEL)
        .temperature(0.4)
        .build();

    let agent = Agent::builder()
        .name("Research Planner")
        .instructions(INSTRUCTIONS)
        .output_schema(schemars::schema_for!(ResearchPlan))
        .build();

    let runner = AgentRunner::new(Arc::new(model));

    let mut output = String::new();
    let mut stream = runner.run(&agent, vec![Message::user("learn about AI agents")]);
    while let Some(event) = stream.next().await {
        match event.agent_event {
            AgentEvent::TextDelta(chunk) => output.push_str(&chunk),
            AgentEvent::Usage(usage) => println!("[runner] token usage: {usage:?}"),
            AgentEvent::Error(error) => eprintln!("[runner] stream error: {error}"),
            _ => {}
        }
    }

    let plan: ResearchPlan = serde_json::from_str(&output)?;
    println!("Title: {}", plan.title);
    println!("Tasks:");
    for (i, t) in plan.tasks.iter().enumerate() {
        println!("  {}. {}", i + 1, t.task);
    }

    Ok(())
}
