// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

//! Demonstrates returning an image from a tool with `Message::tool_result_with_media`.
//!
//! The model calls a `take_snapshot` tool, which returns a solid square in a
//! colour picked at random. The model then names the colour, which shows it
//! received the image. The example exits with an error if it names the wrong
//! one.
//!
//! Run with:
//! ```bash
//! GEMINI_API_KEY=your_key cargo run --example tool_result_image
//! ```
//!
//! Set `COLOR` to `red`, `green`, `blue` or `yellow` to pick the colour, and
//! `MODEL` to override the model.

use std::io::Cursor;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use agent_rig::model::{LlmModel, Media, Message, MessageList, ModelRequest};
use agent_rig::models::gemini::GeminiModel;
use agent_rig::tools::ToolDefinition;
use base64::Engine;
use image::{ImageFormat, Rgb, RgbImage};
use schemars::json_schema;
use serde_json::json;

const MODEL: &str = "gemini-3.1-flash-lite";

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

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().ok();
    let api_key = std::env::var("GEMINI_API_KEY")?;
    let model_name = std::env::var("MODEL").unwrap_or_else(|_| MODEL.to_string());
    let model = GeminiModel::new(api_key, model_name);

    let tools = vec![ToolDefinition {
        name: "take_snapshot".into(),
        description: "Takes a snapshot with the camera and returns it as an image.".into(),
        parameters: json_schema!({"type": "object", "properties": {}}),
    }];
    let mut messages = vec![Message::user(
        "Take a snapshot with the camera and tell me, in one word, the colour it shows.",
    )];

    let first = model
        .generate(ModelRequest {
            messages: MessageList::from(messages.clone()),
            system: None,
            output_schema: None,
            tools: tools.clone(),
        })
        .await?;
    let call = first
        .tool_calls
        .first()
        .cloned()
        .ok_or("the model did not call take_snapshot")?;
    println!("Model called: {}", call.name);

    let (color, rgb) = pick_color()?;
    println!("Returning a {color} image");
    let call = Arc::new(call);
    messages.push(Message::tool_calls(vec![Arc::clone(&call)]));
    messages.push(Message::tool_result_with_media(
        call,
        json!({"status": "ok"}),
        vec![Media::inline("image/png", square_png(rgb)?)],
    ));

    let second = model
        .generate(ModelRequest {
            messages: MessageList::from(messages),
            system: None,
            output_schema: None,
            tools,
        })
        .await?;
    let answer = second.text.unwrap_or_default();
    println!("Model answered: {answer}");

    if answer.to_lowercase().contains(color) {
        println!("OK: the model saw the {color} image");
        Ok(())
    } else {
        Err(format!("expected the answer to mention {color}").into())
    }
}
