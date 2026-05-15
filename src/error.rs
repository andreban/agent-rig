// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

//! Error types returned by the library.
//!
//! All fallible operations across the crate return [`enum@Error`]. Provider
//! adapters wrap transport- and API-level failures into [`Error::Provider`];
//! agent-side failures (serialization, lock poisoning, user-defined tool
//! errors) use [`Error::Agent`].

/// Errors that can occur when using the agent kit.
#[derive(Clone, Debug, thiserror::Error)]
pub enum Error {
    /// An error returned by the underlying LLM provider.
    #[error("LLM provider error: {0}")]
    Provider(String),

    /// An error during agent execution.
    #[error("Agent error: {0}")]
    Agent(String),
}
