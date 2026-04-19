// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

/// Errors that can occur when using the agent kit.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// An error returned by the underlying LLM provider.
    #[error("LLM provider error: {0}")]
    Provider(String),

    /// An error during agent execution.
    #[error("Agent error: {0}")]
    Agent(String),
}
