//! Demonstrates the authorization flow in the `MpscRunner`.
//!
//! The runner consults its configured [`AuthManager`] for every tool call;
//! managers decide which ones actually need approval. Here we plug in a CLI
//! `StdinPromptAuthManager` that holds a set of protected tool names — it
//! fast-paths `true` for anything not in the set and prompts y/N on stdin
//! for the rest.
//!
//! Run with:
//! ```bash
//! GEMINI_API_KEY=your_key cargo run --example mpsc_auth_flow
//! ```

use std::collections::HashSet;
use std::sync::Arc;

use agent_rig::auth::AuthManager;
use agent_rig::error::Error;
use agent_rig::model::Message;
use agent_rig::runner::{AgentEvent, AgentRunner, ToolCallResult};
use agent_rig::tools::{Tool, ToolDefinition, ToolRegistry};
use agent_rig::{Agent, models::gemini::GeminiModel};
use async_trait::async_trait;
use futures_util::StreamExt;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, BufReader};
use tracing_subscriber::EnvFilter;

const MODEL: &str = "gemini-3.1-flash-lite";

struct SendEmailTool;

#[async_trait]
impl Tool for SendEmailTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "send_email".to_string(),
            description: "Sends an email to the given recipient.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "to":      { "type": "string", "description": "Recipient email address" },
                    "subject": { "type": "string" },
                    "body":    { "type": "string" }
                },
                "required": ["to", "subject", "body"]
            }),
        }
    }

    async fn call(&self, args: Value) -> Result<Value, Error> {
        let to = args["to"].as_str().unwrap_or("");
        let subject = args["subject"].as_str().unwrap_or("");
        println!("[tool]  pretending to send email to {to} (subject: {subject:?})");
        Ok(json!({ "status": "sent", "to": to }))
    }
}

/// Prompts via stdin for tool calls whose name is in `protected`. Calls to
/// any other tool are filtered out by `requires_authorization` and never
/// reach `authorize`.
///
/// `authorize` may be called concurrently when the model returns multiple
/// protected tool calls in one turn; stdin can't be safely read by multiple
/// tasks at once and interleaved prompts would be unreadable, so the whole
/// prompt-and-read sequence runs under a mutex.
struct StdinPromptAuthManager {
    protected: HashSet<String>,
    prompt_lock: tokio::sync::Mutex<()>,
}

impl StdinPromptAuthManager {
    fn new<I, S>(protected: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            protected: protected.into_iter().map(Into::into).collect(),
            prompt_lock: tokio::sync::Mutex::new(()),
        }
    }
}

#[async_trait]
impl AuthManager for StdinPromptAuthManager {
    fn requires_authorization(&self, name: &str, _args: &Value) -> bool {
        self.protected.contains(name)
    }

    async fn authorize(&self, name: &str, args: &Value) -> bool {
        let _guard = self.prompt_lock.lock().await;

        println!("\n[auth]  Tool '{name}' wants to run with args:");
        println!("[auth]    {args}");
        print!("[auth]  Approve? [y/N]: ");
        use std::io::Write;
        let _ = std::io::stdout().flush();

        let mut line = String::new();
        let mut stdin = BufReader::new(tokio::io::stdin());
        if let Err(e) = stdin.read_line(&mut line).await {
            eprintln!("[auth]  stdin error: {e}");
            return false;
        }

        matches!(line.trim().to_lowercase().as_str(), "y" | "yes")
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = dotenvy::dotenv();
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let api_key = std::env::var("GEMINI_API_KEY")?;
    let model = GeminiModel::new(api_key, MODEL);
    let registry = Arc::new(ToolRegistry::new().register(Box::new(SendEmailTool)));

    let agent = Agent::builder()
        .name("Email Assistant")
        .instructions(
            "You are an assistant that can send email on the user's behalf. \
             When the user asks you to send a message, call send_email exactly once. \
             After the call returns, confirm what was sent in one short sentence.",
        )
        .tool("send_email")
        .build();

    let auth_manager = Arc::new(StdinPromptAuthManager::new(["send_email"]));
    let runner =
        AgentRunner::with_registry(Arc::new(model), registry).with_auth_manager(auth_manager);

    let question =
        "Send an email to bob@example.com with subject 'Lunch' and body 'See you at noon.'";
    println!("Question: {question}");
    println!("(The runner will pause and ask for approval before send_email runs.)\n");

    let mut stream = runner.run(&agent, vec![Message::user(question)]);

    while let Some(event) = stream.next().await {
        match event.agent_event {
            AgentEvent::ToolCallStarted { name, args } => {
                println!("\n[runner] started:   {name}({args})");
            }
            AgentEvent::ToolCallFinished { name, result } => match result {
                ToolCallResult::Ok(value) => println!("[runner] finished:  {name} → {value}"),
                ToolCallResult::Err(error) => println!("[runner] error:     {name} → {error:?}"),
                ToolCallResult::Denied => println!("[runner] denied:    {name}"),
                ToolCallResult::Unknown => println!("[runner] unknown:   {name}"),
            },
            AgentEvent::TextDelta(chunk) => print!("{chunk}"),
            AgentEvent::ThinkingDelta(_) => {}
            AgentEvent::Usage(usage) => println!("\n[runner] usage:     {usage:?}"),
            AgentEvent::Error(error) => eprintln!("\n[runner] stream error: {error}"),
        }
    }

    println!();
    Ok(())
}
