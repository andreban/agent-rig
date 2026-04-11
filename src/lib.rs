//! # rust-agent-kit
//!
//! A provider-agnostic toolkit for building AI agents in Rust.
//!
//! ## Quick Start
//!
//! ```no_run
//! use rust_agent_kit::{Agent, AgentRunner};
//! use rust_agent_kit::models::gemini::GeminiModel;
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let agent = Agent::builder()
//!     .name("Assistant")
//!     .instructions("You are a helpful assistant.")
//!     .model(Box::new(GeminiModel::new("API_KEY", "gemini-2.5-pro-preview-03-25")))
//!     .build();
//!
//! let result = AgentRunner::new().run(&agent, "Hello!").await?;
//! println!("{}", result.output);
//! # Ok(())
//! # }
//! ```

pub mod error;
pub mod model;
pub mod models;

mod agent;
mod runner;

pub use agent::{Agent, AgentBuilder};
pub use runner::{AgentResult, AgentRunner};
