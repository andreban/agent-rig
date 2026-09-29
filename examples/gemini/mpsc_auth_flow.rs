// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

//! Demonstrates a consumer-side approval flow.
//!
//! Whether a tool call needs approval is the consumer's policy, applied in the
//! event loop — not something the tool's contract advertises. Here the client
//! keeps a small allowlist ([`TOOLS_NEEDING_APPROVAL`]); when an
//! [`AgentEvent::ToolCall`] names a gated tool, it previews the call arguments
//! and prompts the user on stdin for a y/N decision before invoking
//! [`Tool::call`]. Denying resolves the call with a soft error instead of
//! running the tool.
//!
//! Run with:
//! ```bash
//! GEMINI_API_KEY=your_key cargo run --example mpsc_auth_flow
//! ```

use std::sync::Arc;

use agent_rig::model::{Message, ToolCall};
use agent_rig::runner::{AgentEvent, AgentRunner};
use agent_rig::tools::{Tool, ToolDefinition, ToolRegistry, ToolResult};
use agent_rig::{Agent, models::gemini::GeminiModel};
use async_trait::async_trait;
use futures_util::StreamExt;
use schemars::json_schema;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio_util::sync::CancellationToken;
use tracing::info;
use tracing_subscriber::EnvFilter;

const MODEL: &str = "gemini-3.1-flash-lite";

struct SendEmailTool {
    definition: ToolDefinition,
}

impl Default for SendEmailTool {
    fn default() -> Self {
        Self {
            definition: ToolDefinition {
                name: "send_email".to_string(),
                description: "Sends an email to the given recipient.".to_string(),
                parameters: json_schema!({
                    "type": "object",
                    "properties": {
                        "to":      { "type": "string", "description": "Recipient email address" },
                        "subject": { "type": "string" },
                        "body":    { "type": "string" }
                    },
                    "required": ["to", "subject", "body"]
                }),
            },
        }
    }
}

#[async_trait]
impl Tool for SendEmailTool {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn call(&self, tool_call: Arc<ToolCall>, _cancel: CancellationToken) -> ToolResult {
        let to = tool_call.args["to"].as_str().unwrap_or("");
        let subject = tool_call.args["subject"].as_str().unwrap_or("");
        println!("[tool]  pretending to send email to {to} (subject: {subject:?})");
        ToolResult::ok(json!({ "status": "sent", "to": to }))
    }
}

/// Tool names this client gates behind a confirmation prompt. In a real app
/// this might come from configuration, a policy engine, or a tool category —
/// the point is that it lives with the consumer, not the tool.
const TOOLS_NEEDING_APPROVAL: &[&str] = &["send_email"];

/// Renders the `send_email` arguments as a short, human-readable preview for
/// the approval prompt.
fn email_preview(args: &Value) -> String {
    let to = args["to"].as_str().unwrap_or("");
    let subject = args["subject"].as_str().unwrap_or("");
    let body = args["body"].as_str().unwrap_or("");
    format!("To: {to}\n  Subject: {subject}\n  {body}")
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = dotenvy::dotenv();
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let api_key = std::env::var("GEMINI_API_KEY")?;
    let model = GeminiModel::new(api_key, MODEL);
    let registry = Arc::new(ToolRegistry::new().register(SendEmailTool::default()));

    let agent = Agent::builder()
        .name("Email Assistant")
        .instructions(
            "You are an assistant that can send email on the user's behalf. \
             When the user asks you to send a message, call send_email exactly once. \
             After the call returns, confirm what was sent in one short sentence.",
        )
        .tool("send_email")
        .build();

    let runner = AgentRunner::with_tools(Arc::new(model), registry.definitions());

    let question =
        "Send an email to bob@example.com with subject 'Lunch' and body 'See you at noon.'";
    println!("Question: {question}");
    println!("(The runner will pause and ask for approval before send_email runs.)\n");

    let mut stream = runner.run(&agent, vec![Message::user(question)].into());

    while let Some(event) = stream.next().await {
        match event.agent_event {
            AgentEvent::TextDelta(chunk) => print!("{chunk}"),
            AgentEvent::ThinkingDelta(_) => {}
            AgentEvent::Usage(usage) => println!("\n[runner] usage:     {usage:?}"),
            AgentEvent::Error(error) => eprintln!("\n[runner] stream error: {error}"),
            AgentEvent::Cancelled => println!("\n[runner] cancelled"),
            AgentEvent::TurnStart => {}
            AgentEvent::TurnFinish { .. } => {}
            AgentEvent::ToolCall(tool_call) => {
                info!(?tool_call, "AgentEvent::ToolCall");
                let Some(tool) = registry.get(&tool_call.details.name) else {
                    tool_call.resolve(ToolResult::error("Unknown Tool"));
                    continue;
                };

                // Gate execution on the client's own approval policy. For a
                // gated tool, show a preview of the raw call arguments and
                // prompt before invoking it.
                if TOOLS_NEEDING_APPROVAL.contains(&tool_call.details.name.as_str()) {
                    println!(
                        "\n[auth]  Tool '{}' (id {}) wants to run:",
                        tool_call.details.name, tool_call.details.id
                    );
                    println!("[auth]    {}", email_preview(&tool_call.details.args));
                    print!("[auth]  Approve? [y/N]: ");
                    use std::io::Write;
                    let _ = std::io::stdout().flush();

                    let mut line = String::new();
                    let mut stdin = BufReader::new(tokio::io::stdin());

                    if let Err(e) = stdin.read_line(&mut line).await {
                        eprintln!("[auth]  stdin error: {e}");
                        tool_call.resolve(Value::from("Tool call authorization failed."));
                        continue;
                    };

                    let approved = matches!(line.trim().to_lowercase().as_str(), "y" | "yes");
                    if !approved {
                        tool_call.resolve(Value::from("User rejected approval of the tool call"));
                        continue;
                    }
                }

                let result = tool
                    .call(tool_call.details.clone(), tool_call.cancellation_token.clone())
                    .await;

                tool_call.resolve(result);
            }
        }
    }

    println!();
    Ok(())
}
