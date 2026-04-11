//! Demonstrates typed structured output using `AgentRunner::run_typed`.
//!
//! `schemars` generates the JSON Schema from `ResearchPlan` and it is set on the
//! agent so the model is constrained to produce matching JSON. `run_typed` then
//! deserializes the response directly into `ResearchPlan` — no `serde_json::from_str`
//! at the call site.
//!
//! Run with:
//! ```text
//! GEMINI_API_KEY=<key> cargo run --example structured_output
//! ```

use rust_agent_kit::{Agent, AgentRunner, models::gemini::GeminiModel};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::error::Error;

const MODEL: &str = "gemini-2.5-flash";

const INSTRUCTIONS: &str = "\
You are a research planning assistant. \
Given a research topic, produce a structured plan with a title and exactly 5 concise tasks.";

/// A single task in the research plan.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ResearchTask {
    /// Short imperative description of the task (5 words or less).
    task: String,
}

/// A structured research plan produced by the agent.
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
    let api_key = std::env::var("GEMINI_API_KEY")?;

    let model = GeminiModel::builder(api_key, MODEL)
        .temperature(0.4)
        .build();

    let agent = Agent::builder()
        .name("Research Planner")
        .instructions(INSTRUCTIONS)
        .output_schema(schemars::schema_for!(ResearchPlan))
        .build();

    let plan: ResearchPlan = AgentRunner::new(Box::new(model))
        .run_typed(&agent, "learn about AI agents")
        .await?;

    println!("Title: {}", plan.title);
    println!("Tasks:");
    for (i, t) in plan.tasks.iter().enumerate() {
        println!("  {}. {}", i + 1, t.task);
    }

    Ok(())
}
