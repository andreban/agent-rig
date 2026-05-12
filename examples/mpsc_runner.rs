use std::sync::Arc;

use agent_rig::mpsc_runner::MpscRunner;
use agent_rig::{Agent, models::gemini::GeminiModel};
use futures_util::StreamExt;
use geologia::prelude::{ThinkingConfig, ThinkingLevel};
use tracing_subscriber::EnvFilter;

const MODEL: &str = "gemini-3.1-flash-lite";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
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

    let runner = MpscRunner::new(Arc::new(model));
    let mut stream = runner.run("hello".to_string());
    while let Some(next) = stream.next().await {
        println!("{next}")
    }

    Ok(())
}
