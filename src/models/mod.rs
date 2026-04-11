//! Built-in LLM provider implementations.
//!
//! Each submodule provides a concrete [`LlmModel`](crate::model::LlmModel)
//! implementation for a specific provider.

pub mod gemini;
pub mod ollama;
