//! Demonstrates structured output using a JSON schema derived from a Rust type.
//!
//! The agent extracts a research plan as a strongly-typed `ResearchPlan` struct.
//! `schemars` generates the JSON schema from the type; the schema is forwarded
//! to Gemini via `response_schema` so the model is constrained to produce JSON
//! that matches it.  The response text is then deserialized back into
//! `ResearchPlan` using `serde_json`.
//!
//! Run with:
//! ```text
//! GEMINI_API_KEY=<key> cargo run --example structured_output
//! ```

use rust_agent_kit::{Agent, AgentRunner, models::gemini::GeminiModel};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
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

/// Converts a schemars-generated JSON Schema into a Gemini-compatible schema.
///
/// The Gemini API accepts a subset of JSON Schema and rejects meta-fields like
/// `$schema`, `title`, and `definitions`.  This function strips those fields and
/// inlines every `$ref` reference so the result is self-contained.
fn to_gemini_schema(mut root: Value) -> Value {
    let definitions = root
        .as_object_mut()
        .and_then(|o| o.remove("definitions"))
        .unwrap_or(Value::Null);

    resolve_refs(&mut root, &definitions);
    root
}

fn resolve_refs(value: &mut Value, definitions: &Value) {
    match value {
        Value::Object(obj) => {
            // Inline $ref before doing anything else with this node.
            if let Some(ref_val) = obj.get("$ref").cloned() {
                if let Some(ref_str) = ref_val.as_str() {
                    if let Some(def_name) = ref_str.strip_prefix("#/definitions/") {
                        if let Some(def) = definitions.get(def_name) {
                            let mut resolved = def.clone();
                            resolve_refs(&mut resolved, definitions);
                            *value = resolved;
                            return;
                        }
                    }
                }
            }
            // Strip meta-fields the Gemini API does not recognise.
            obj.remove("$schema");
            for v in obj.values_mut() {
                resolve_refs(v, definitions);
            }
        }
        Value::Array(arr) => {
            for v in arr.iter_mut() {
                resolve_refs(v, definitions);
            }
        }
        _ => {}
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let _ = dotenvy::dotenv();
    let api_key = std::env::var("GEMINI_API_KEY")?;

    // Derive the JSON schema from `ResearchPlan`, then strip schemars-specific
    // meta-fields and inline $ref references so Gemini can parse the schema.
    let schema = to_gemini_schema(serde_json::to_value(schemars::schema_for!(ResearchPlan))?);

    let model = GeminiModel::builder(api_key, MODEL)
        .temperature(0.4)
        .response_mime_type("application/json")
        .response_schema(schema)
        .build();

    let agent = Agent::builder()
        .name("Research Planner")
        .instructions(INSTRUCTIONS)
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
