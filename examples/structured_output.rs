//! Demonstrates structured output using a JSON schema derived from a Rust type.
//!
//! The agent extracts a research plan as a strongly-typed `ResearchPlan` struct.
//! `schemars` generates the JSON schema from the type; the schema is set on the
//! agent via `output_schema` so the model is constrained to produce JSON that
//! matches it.  The response text is then deserialized back into `ResearchPlan`
//! using `serde_json`.
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

    let schema = schemars::schema_for!(ResearchPlan);

    let model = GeminiModel::builder(api_key, MODEL)
        .temperature(0.4)
        .build();

    let agent = Agent::builder()
        .name("Research Planner")
        .instructions(INSTRUCTIONS)
        .output_schema(schema)
        .model(Box::new(model))
        .build();

    let input = "learn about AI agents";
    let result = AgentRunner::new().run(&agent, input).await?;

    // The model is constrained to return valid JSON matching ResearchPlan.
    let plan: ResearchPlan = serde_json::from_str(&result.output)?;

    println!("Title: {}", plan.title);
    println!("Tasks:");
    for (i, t) in plan.tasks.iter().enumerate() {
        println!("  {}. {}", i + 1, t.task);
    }

    Ok(())
}
