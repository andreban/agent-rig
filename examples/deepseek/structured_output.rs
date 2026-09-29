// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

//! Demonstrates structured output with DeepSeek (`deepseek-chat`).
//!
//! Run with:
//! ```bash
//! cargo run --example deepseek_structured_output --features deepseek
//! ```

use std::error::Error;
use std::sync::Arc;

use agent_rig::Agent;
use agent_rig::model::Message;
use agent_rig::models::deepseek::DeepSeekModel;
use agent_rig::runner::{AgentEvent, AgentRunner};
use futures_util::StreamExt;
use schemars::JsonSchema;
use serde::Deserialize;
use tracing_subscriber::EnvFilter;

const MODEL: &str = "deepseek-chat";

#[derive(Debug, Deserialize, JsonSchema)]
#[allow(dead_code)]
struct RecipeAnalysis {
    recipe_name: String,
    prep_time_minutes: u32,
    difficulty: String,
    key_ingredients: Vec<String>,
    chef_tip: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let _ = dotenvy::dotenv();
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let api_key = std::env::var("DEEPSEEK_API_KEY")
        .expect("DEEPSEEK_API_KEY must be set in .env or environment");

    let model = Arc::new(DeepSeekModel::new(api_key, MODEL));
    let runner = AgentRunner::new(model);

    let agent = Agent::builder()
        .name("Recipe Analyst")
        .instructions("Analyze recipes and provide structured nutritional and preparation details.")
        .output_schema(schemars::schema_for!(RecipeAnalysis))
        .build();

    let prompt = "Analyze the classic Italian Spaghetti Carbonara recipe.";
    println!("Prompt: {prompt}\n");

    let mut json_text = String::new();
    let mut stream = runner.run(&agent, vec![Message::user(prompt)].into());

    while let Some(event) = stream.next().await {
        match event.agent_event {
            AgentEvent::TextDelta(chunk) => {
                print!("{chunk}");
                json_text.push_str(&chunk);
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

    println!("\n--- Deserialized Struct ---");
    let analysis: RecipeAnalysis = serde_json::from_str(&json_text)?;
    println!("{:#?}", analysis);

    Ok(())
}
