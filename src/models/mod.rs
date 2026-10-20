// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

//! Built-in LLM provider implementations.
//!
//! Each submodule provides a concrete [`LlmModel`](crate::model::LlmModel)
//! implementation for a specific provider. Modules are gated behind Cargo
//! features; enable `gemini`, `ollama`, `deepseek`, or `anthropic` (or `full`) to include them.

#[cfg(feature = "gemini")]
pub mod gemini;

#[cfg(feature = "ollama")]
pub mod ollama;

#[cfg(feature = "deepseek")]
pub mod deepseek;

#[cfg(feature = "anthropic")]
pub mod anthropic;

/// Generates a random, opaque tool-call ID of the form `call_<16 hex digits>`.
///
/// Used by adapters whose provider omits the call ID (Gemini) or has none
/// (Ollama), so every [`ToolCall`](crate::model::ToolCall) still has a
/// non-empty ID that is unique within a run. IDs are random rather than
/// derived from the message history, so trimming or rebuilding a thread
/// can't cause collisions.
#[cfg(any(feature = "gemini", feature = "ollama"))]
pub(crate) fn synthesize_tool_call_id() -> String {
    use std::hash::{BuildHasher, Hasher};
    use std::sync::atomic::{AtomicU64, Ordering};

    // `RandomState` is seeded from OS randomness, so hashing a process-wide
    // counter yields IDs that are unique within the process and don't repeat
    // across restarts.
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
    hasher.write_u64(COUNTER.fetch_add(1, Ordering::Relaxed));
    format!("call_{:016x}", hasher.finish())
}
