//! # rust-agent-kit
//!
//! A provider-agnostic toolkit for building AI agents in Rust.
//!
//! ## Quick Start
//!
//! ```no_run,ignore
//! // Requires the `gemini` feature: `cargo add rust-agent-kit --features gemini`
//! use rust_agent_kit::{Agent, AgentRunner};
//! use rust_agent_kit::models::gemini::GeminiModel;
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let model = GeminiModel::builder("API_KEY", "gemini-2.5-pro-preview-03-25")
//!     .temperature(0.8)
//!     .build();
//!
//! let agent = Agent::builder()
//!     .name("Assistant")
//!     .instructions("You are a helpful assistant.")
//!     .build();
//!
//! let runner = AgentRunner::new(Box::new(model));
//! let result = runner.run(&agent, "Hello!").await?;
//! println!("{}", result.output);
//! # Ok(())
//! # }
//! ```

pub mod agent_tool;
pub mod error;
pub mod model;
pub mod models;
pub mod tool;

mod agent;
mod runner;

pub use agent::{Agent, AgentBuilder};
pub use agent_tool::AgentTool;
pub use runner::{AgentEvent, AgentResult, AgentRunner, RunBuilder};
