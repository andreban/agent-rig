// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

//! Built-in LLM provider implementations.
//!
//! Each submodule provides a concrete [`LlmModel`](crate::model::LlmModel)
//! implementation for a specific provider. Modules are gated behind Cargo
//! features; enable `gemini` or `ollama` (or `full`) to include them.

#[cfg(feature = "gemini")]
pub mod gemini;

#[cfg(feature = "ollama")]
pub mod ollama;
