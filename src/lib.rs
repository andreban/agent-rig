// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

//! # agent-rig
//!
//! A provider-agnostic toolkit for building AI agents in Rust.
//!
//! ## Quick Start
//!
//! ```no_run
//! # // Compile-tested only when the `gemini` feature is enabled.
//! # #[cfg(feature = "gemini")]
//! # mod doc {
//! // Requires the `gemini` feature: `cargo add agent-rig --features gemini`
//! use std::sync::Arc;
//! use agent_rig::{Agent, model::Message, models::gemini::GeminiModel,
//!     runner::{AgentEvent, AgentRunner}};
//! use futures_util::StreamExt;
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let model = GeminiModel::builder("API_KEY", "gemini-2.5-pro")
//!     .temperature(0.8)
//!     .build();
//!
//! let agent = Agent::builder()
//!     .name("Assistant")
//!     .instructions("You are a helpful assistant.")
//!     .build();
//!
//! let runner = AgentRunner::new(Arc::new(model));
//! let mut stream = runner.run(&agent, vec![Arc::new(Message::user("Hello!"))]);
//! while let Some(event) = stream.next().await {
//!     if let AgentEvent::TextDelta(chunk) = event.agent_event {
//!         print!("{chunk}");
//!     }
//! }
//! # Ok(())
//! # }
//! # }
//! ```

pub mod error;
pub mod model;
pub mod models;
pub mod tools;

mod agent;
pub mod runner;

pub use agent::{Agent, AgentBuilder};
