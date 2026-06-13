// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

//! Demonstrates the three ways to cancel an [`AgentRunner::run`] in flight.
//!
//! 1. **Drop the stream.** The default ergonomics: just drop the returned
//!    stream and the runner (and any tool that was running) shuts down at
//!    its next await point.
//! 2. **External [`CancellationToken`].** Use `run_with_cancellation` when
//!    you need to share the cancel signal with a sibling task (deadline
//!    timer, server shutdown coordinator, …).
//! 3. **Deadline.** Composed with `tokio::time::timeout` on top of (2).
//!
//! The example uses a tool that intentionally sleeps for a few seconds.
//! Run with:
//! ```bash
//! GEMINI_API_KEY=your_key cargo run --example cancellation
//! ```

use std::sync::Arc;
use std::time::Duration;

use agent_rig::error::Error;
use agent_rig::model::Message;
use agent_rig::runner::{AgentEvent, AgentRunner, ToolCallResult};
use agent_rig::tools::{ProgressReporter, Tool, ToolDefinition, ToolRegistry};
use agent_rig::{Agent, models::gemini::GeminiModel};
use async_trait::async_trait;
use futures_util::StreamExt;
use schemars::json_schema;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;
use tracing_subscriber::EnvFilter;

const MODEL: &str = "gemini-3.1-flash-lite";

/// Tool that "uploads" a file by sleeping. Cooperates with cancellation:
/// the `tokio::select!` against `cancel.cancelled()` aborts the upload
/// cleanly instead of running to completion.
struct SlowUploadTool {
    definition: ToolDefinition,
}

impl Default for SlowUploadTool {
    fn default() -> Self {
        Self {
            definition: ToolDefinition {
                name: "upload".to_string(),
                description: "Uploads a file. Takes about 5 seconds.".to_string(),
                parameters: json_schema!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" }
                    },
                    "required": ["path"]
                }),
            },
        }
    }
}

#[async_trait]
impl Tool for SlowUploadTool {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn apply(&self, args: Value, _progress: &dyn ProgressReporter, cancel: CancellationToken) -> Result<Value, Error> {
        let path = args["path"].as_str().unwrap_or("unknown");
        println!("[tool]  upload({path}) starting (5s)…");
        tokio::select! {
            _ = cancel.cancelled() => {
                println!("[tool]  upload({path}) aborted on cancellation");
                Err(Error::Agent("upload cancelled".into()))
            }
            _ = tokio::time::sleep(Duration::from_secs(5)) => {
                println!("[tool]  upload({path}) finished");
                Ok(json!({ "uploaded": path }))
            }
        }
    }
}

fn build_runner(api_key: String) -> (AgentRunner, Agent) {
    let model = GeminiModel::new(api_key, MODEL);
    let registry = Arc::new(ToolRegistry::new().register(SlowUploadTool::default()));
    let runner = AgentRunner::with_registry(Arc::new(model), registry);
    let agent = Agent::builder()
        .name("Uploader")
        .instructions(
            "You upload files. When asked to upload one, call the `upload` tool \
             with the requested path and report the result.",
        )
        .tool("upload")
        .build();
    (runner, agent)
}

async fn drain<S>(label: &str, mut stream: S)
where
    S: StreamExt<Item = agent_rig::runner::RunEvent> + Unpin,
{
    while let Some(event) = stream.next().await {
        match event.agent_event {
            AgentEvent::ToolCallStarted { name, args, .. } => {
                println!("[{label}] started:   {name}({args})");
            }
            AgentEvent::ToolCallUpdate { name, details, .. } => {
                println!("[{label}] update:   {name}({details:?})");
            }
            AgentEvent::ToolCallFinished { name, result, .. } => match result {
                ToolCallResult::Ok(value) => println!("[{label}] finished:  {name} → {value}"),
                ToolCallResult::Err(error) => println!("[{label}] error:     {name} → {error:?}"),
                ToolCallResult::Denied => println!("[{label}] denied:    {name}"),
                ToolCallResult::Unknown => println!("[{label}] unknown:   {name}"),
            },
            AgentEvent::TextDelta(chunk) => print!("{chunk}"),
            AgentEvent::ThinkingDelta(_) => {}
            AgentEvent::Usage(usage) => println!("\n[{label}] usage:     {usage:?}"),
            AgentEvent::Error(error) => eprintln!("\n[{label}] stream error: {error}"),
            AgentEvent::Cancelled => println!("\n[{label}] cancelled"),
            AgentEvent::StartTurn => {}
            AgentEvent::EndTurn { .. } => {}
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = dotenvy::dotenv();
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();
    let api_key = std::env::var("GEMINI_API_KEY")?;

    // (1) Drop the stream after a short delay.
    println!("=== (1) Cancel by dropping the returned stream ===");
    let (runner, agent) = build_runner(api_key.clone());
    let stream = runner.run(&agent, vec![Message::user("upload /tmp/report.pdf")]);
    let drainer = tokio::spawn(drain("drop", stream));
    tokio::time::sleep(Duration::from_millis(1500)).await;
    // Abort the consumer task; dropping the JoinHandle does NOT drop the
    // stream — we need to drop the future the task owns. The cleanest
    // way is to `abort` the task, which drops its future.
    drainer.abort();
    // Give the tool a beat to print its cancellation message.
    tokio::time::sleep(Duration::from_millis(200)).await;
    println!();

    // (2) Explicit CancellationToken from a sibling task.
    println!("=== (2) Cancel via an external CancellationToken ===");
    let (runner, agent) = build_runner(api_key.clone());
    let cancel = CancellationToken::new();
    let stream = runner.run_with_cancellation(
        &agent,
        vec![Message::user("upload /var/log/app.log")],
        cancel.clone(),
    );
    let drainer = tokio::spawn(drain("token", stream));
    tokio::time::sleep(Duration::from_millis(1500)).await;
    cancel.cancel();
    let _ = drainer.await;
    println!();

    // (3) Deadline via tokio::time::timeout composed with (2).
    println!("=== (3) Deadline via tokio::time::timeout ===");
    let (runner, agent) = build_runner(api_key);
    let cancel = CancellationToken::new();
    let stream = runner.run_with_cancellation(
        &agent,
        vec![Message::user("upload /etc/hosts")],
        cancel.clone(),
    );
    // Fire cancel after a 1.5s deadline; the drainer task observes the
    // terminal Cancelled event.
    let deadline_cancel = cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(1500)).await;
        deadline_cancel.cancel();
    });
    drain("deadline", stream).await;

    Ok(())
}
