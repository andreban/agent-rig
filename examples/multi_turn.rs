use futures_util::StreamExt;
use google_genai::prelude::{ThinkingConfig, ThinkingLevel};
use rust_agent_kit::{Agent, AgentEvent, AgentRunner, model::Message, models::gemini::GeminiModel};
use std::{
    error::Error,
    io::{self, BufRead, Write},
};
use tracing_subscriber::EnvFilter;

const MODEL: &str = "gemini-3.1-flash-lite-preview";

/// A simple streaming REPL that demonstrates multi-turn conversation using
/// `RunBuilder::run_stream`.
///
/// Each turn:
///   1. `run_builder(&agent).history(history.clone()).run_stream(input)` sends
///      the full context to the model and returns a stream of [`AgentEvent`]s.
///   2. `TextDelta` chunks are printed as they arrive, so the reply appears
///      incrementally rather than all at once.
///   3. The completed user message and assistant reply are appended to `history`
///      so subsequent turns remember the whole conversation.
///
/// Run with:
///   GEMINI_API_KEY=... cargo run --example multi_turn
#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let _ = dotenvy::dotenv();
    tracing_subscriber::fmt().with_env_filter(EnvFilter::from_default_env()).init();
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

    let stdin = io::stdin();
    let mut history: Vec<Message> = Vec::new();

    println!("Multi-turn chat (Ctrl-C or Ctrl-D to quit)\n");
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

        let mut reply = String::new();
        let stream = runner.run_builder(&agent).history(history.clone()).run_stream(&input);
        futures_util::pin_mut!(stream);

        while let Some(event) = stream.next().await {
            match event? {
                AgentEvent::Thinking(token) => {
                    print!("\x1b[2m{token}\x1b[0m"); // dim text
                    io::stdout().flush()?;
                }
                AgentEvent::TextDelta(chunk) => {
                    print!("{chunk}");
                    io::stdout().flush()?;
                    reply.push_str(&chunk);
                }
                _ => {}
            }
        }
        println!("\n");

        // Extend history with this turn so the next call has full context.
        history.push(Message::user(&input));
        history.push(Message::assistant(&reply));

        print!("You: ");
        io::stdout().flush()?;
    }

    Ok(())
}
