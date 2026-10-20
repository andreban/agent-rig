// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

//! Error types returned by the library.
//!
//! All fallible operations across the crate return [`enum@Error`]. Provider
//! adapters wrap transport- and API-level failures into [`Error::Provider`],
//! or [`Error::ProviderTransient`] when sending the same request again may
//! succeed; agent-side failures (serialization, lock poisoning, user-defined
//! tool errors) use [`Error::Agent`].

/// Errors that can occur when using the agent kit.
///
/// New variants may be added, so `match`es on it need a wildcard arm.
#[derive(Clone, Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// An error returned by the underlying LLM provider.
    #[error("LLM provider error: {0}")]
    Provider(String),

    /// A provider error that may go away if the same request is sent again:
    /// rate limits and exhausted quotas (`429`), server errors and overload
    /// (`5xx`), and failures to connect or timeouts.
    ///
    /// The adapters don't retry; callers decide whether and when to (see
    /// [`Error::is_transient`]).
    #[error("LLM provider error: {0}")]
    ProviderTransient(String),

    /// An error during agent execution.
    #[error("Agent error: {0}")]
    Agent(String),
}

impl Error {
    /// Whether sending the same request again may succeed, i.e. this is an
    /// [`Error::ProviderTransient`].
    pub fn is_transient(&self) -> bool {
        matches!(self, Error::ProviderTransient(_))
    }
}
