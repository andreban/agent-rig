// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

//! Demonstrates returning an image from a tool run by `AgentRunner` against
//! DeepSeek (`deepseek-chat`).
//!
//! The model calls a `take_snapshot` tool registered in a `ToolRegistry`,
//! which returns a solid square in a colour picked at random via
//! `ToolResult::ok(Media::inline(...))`. The model then names the colour, which
//! shows it received the image. The example exits with an error if it names the
//! wrong one.
//!
//! Run with:
//! ```bash
//! DEEPSEEK_API_KEY=your_key cargo run --example deepseek_tool_result_image --features deepseek
//! ```
//!
//! Set `COLOR` to `red`, `green`, `blue` or `yellow` to pick the colour, and
//! `MODEL` to override the model.

use std::io::Cursor;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use agent_rig::Agent;
use agent_rig::model::{Media, Message, ToolCall};
use agent_rig::models::deepseek::DeepSeekModel;
use agent_rig::runner::{AgentEvent, AgentRunner};
use agent_rig::tools::{Tool, ToolDefinition, ToolRegistry, ToolResult};
use async_trait::async_trait;
use base64::Engine;
use futures_util::StreamExt;
use image::{ImageFormat, Rgb, RgbImage};
use schemars::json_schema;
use tokio_util::sync::CancellationToken;

const MODEL: &str = "deepseek-chat";

const COLORS: [(&str, [u8; 3]); 4] = [
    ("red", [220, 20, 20]),
    ("green", [20, 180, 20]),
    ("blue", [20, 40, 220]),
    ("yellow", [240, 220, 20]),
];

/// A 64×64 PNG filled with `rgb`, base64-encoded.
fn square_png(rgb: [u8; 3]) -> Result<String, Box<dyn std::error::Error>> {
    let mut bytes = Cursor::new(Vec::new());
    RgbImage::from_pixel(64, 64, Rgb(rgb)).write_to(&mut bytes, ImageFormat::Png)?;
    Ok(base64::engine::general_purpose::STANDARD.encode(bytes.into_inner()))
}

/// The colour from `COLOR`, or one picked by time (in microseconds, since some
/// platforms' clocks have 100 ns resolution).
fn pick_color() -> Result<(&'static str, [u8; 3]), Box<dyn std::error::Error>> {
    if let Ok(name) = std::env::var("COLOR") {
        return COLORS
            .iter()
            .find(|(c, _)| *c == name)
            .copied()
            .ok_or_else(|| format!("unknown COLOR {name}").into());
    }
    let micros = SystemTime::now().duration_since(UNIX_EPOCH)?.as_micros();
    Ok(COLORS[(micros % COLORS.len() as u128) as usize])
}

struct TakeSnapshotTool {
    definition: ToolDefinition,
    png_base64: String,
}

#[async_trait]
impl Tool for TakeSnapshotTool {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn call(&self, _tool_call: Arc<ToolCall>, _cancel: CancellationToken) -> ToolResult {
        ToolResult::ok(Media::inline("image/png", self.png_base64.clone()))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().ok();
    let api_key = std::env::var("DEEPSEEK_API_KEY")?;
    let model_name = std::env::var("MODEL").unwrap_or_else(|_| MODEL.to_string());
    let model = Arc::new(DeepSeekModel::new(api_key, model_name));

    let (color, rgb) = pick_color()?;
    let registry = ToolRegistry::new().register(TakeSnapshotTool {
        definition: ToolDefinition {
            name: "take_snapshot".into(),
            description: "Takes a snapshot with the camera and returns it as an image.".into(),
            parameters: json_schema!({"type": "object", "properties": {}}),
        },
        png_base64: square_png(rgb)?,
    });

    let agent = Agent::builder()
        .name("SnapshotInspector")
        .instructions("Use the take_snapshot tool and report the colour shown in the image.")
        .tool("take_snapshot")
        .build();
    let runner = AgentRunner::with_tools(model, registry.definitions());

    let mut answer = String::new();
    let mut stream = runner.run(
        &agent,
        vec![Message::user(
            "Take a snapshot with the camera and tell me, in one word, the colour it shows.",
        )]
        .into(),
    );
    while let Some(event) = stream.next().await {
        match event.agent_event {
            AgentEvent::ToolCall(call) => {
                println!("Model called: {}", call.details.name);
                println!("Returning a {color} image");
                let result = match registry.get(&call.details.name) {
                    Some(tool) => {
                        tool.call(call.details.clone(), call.cancellation_token.clone())
                            .await
                    }
                    None => ToolResult::error("Unknown tool"),
                };
                call.resolve(result);
            }
            AgentEvent::TextDelta(chunk) => answer.push_str(&chunk),
            AgentEvent::Error(e) => return Err(e.into()),
            _ => {}
        }
    }

    println!("Model answered: {answer}");
    if answer.to_lowercase().contains(color) {
        println!("OK: the model saw the {color} image");
        Ok(())
    } else {
        Err(format!("expected the answer to mention {color}").into())
    }
}
