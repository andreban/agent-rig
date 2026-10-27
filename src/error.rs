// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

//! Error types returned by the library.
//!
//! All fallible operations across the crate return [`enum@Error`]. Provider
//! adapters wrap transport- and API-level failures into [`Error::Provider`],
//! or [`Error::ProviderTransient`] when sending the same request again may
//! succeed; agent-side failures (serialization, lock poisoning, user-defined
//! tool errors) use [`Error::Agent`].

use std::time::Duration;

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
    /// [`Error::is_transient`] and [`Error::retry_after`]).
    #[error("LLM provider error: {message}")]
    ProviderTransient {
        /// Human-readable error message from the provider client.
        message: String,
        /// How long the provider asked the caller to wait before retrying, when
        /// the response included a hint (Anthropic's `retry-after` header or
        /// Gemini's `google.rpc.RetryInfo`); `None` otherwise.
        retry_after: Option<Duration>,
    },

    /// An error during agent execution.
    #[error("Agent error: {0}")]
    Agent(String),
}

impl Error {
    /// Whether sending the same request again may succeed, i.e. this is an
    /// [`Error::ProviderTransient`].
    pub fn is_transient(&self) -> bool {
        matches!(self, Error::ProviderTransient { .. })
    }

    /// How long the provider asked the caller to wait before retrying, if it
    /// gave a hint (see [`Error::ProviderTransient`]).
    ///
    /// Returns `None` for transient errors without a hint (or from providers
    /// that don't expose one, like DeepSeek and Ollama) and for
    /// [`Error::Provider`] and [`Error::Agent`].
    pub fn retry_after(&self) -> Option<Duration> {
        match self {
            Error::ProviderTransient { retry_after, .. } => *retry_after,
            Error::Provider(_) | Error::Agent(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retry_after_returns_hint_only_for_transient_errors_with_one() {
        let with_hint = Error::ProviderTransient {
            message: "rate limited".into(),
            retry_after: Some(Duration::from_secs(30)),
        };
        assert!(with_hint.is_transient());
        assert_eq!(with_hint.retry_after(), Some(Duration::from_secs(30)));
        assert_eq!(with_hint.to_string(), "LLM provider error: rate limited");

        let without_hint = Error::ProviderTransient {
            message: "overloaded".into(),
            retry_after: None,
        };
        assert!(without_hint.is_transient());
        assert_eq!(without_hint.retry_after(), None);

        let provider = Error::Provider("bad request".into());
        assert!(!provider.is_transient());
        assert_eq!(provider.retry_after(), None);

        let agent = Error::Agent("cancelled".into());
        assert!(!agent.is_transient());
        assert_eq!(agent.retry_after(), None);
    }
}
