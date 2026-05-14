// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

//! Authorization policy for tool calls.

use async_trait::async_trait;
use serde_json::Value;

/// Decides whether a tool call should be executed.
///
/// A runner that holds an `AuthManager` consults it for **every** tool call
/// it is about to make. Implementations are the single source of truth for
/// authorization policy: which calls require approval, how that approval is
/// obtained (CLI prompt, UI dialog, policy file, remote service), and
/// whether to allow or deny.
///
/// The trait has two methods so the filter and the decision can live apart:
///
/// - [`requires_authorization`] — sync, must be cheap. The runner calls it
///   first; if it returns `false`, [`authorize`] is skipped entirely. Use
///   it to fast-path uninteresting calls (a `HashSet<String>` lookup, an
///   args inspection). No I/O, no locks, no awaits — the sync signature is
///   the contract.
/// - [`authorize`] — async, may block on user input, RPC, dialogs, etc.
///   Returns `Ok(())` to allow, `Err(reason)` to deny. The reason is
///   surfaced as a denial event by the runner and fed back to the model
///   as the tool result so it can react.
///
/// `authorize` may be called concurrently when the model returns multiple
/// tool calls in one turn. Implementations sharing UI resources (stdin, a
/// modal dialog) must serialize internally — typically with a
/// [`tokio::sync::Mutex`]. The lock belongs in `authorize`, not in
/// `requires_authorization`.
///
/// [`requires_authorization`]: AuthManager::requires_authorization
/// [`authorize`]: AuthManager::authorize
#[async_trait]
pub trait AuthManager: Send + Sync {
    /// Cheap, synchronous gate: should this call go through [`authorize`]?
    ///
    /// Defaults to `true` (always gate) so a minimal impl only has to
    /// provide [`authorize`]. Override to fast-path calls that don't need
    /// approval.
    ///
    /// Must be non-blocking — no I/O, no locks, no awaits.
    ///
    /// [`authorize`]: AuthManager::authorize
    fn requires_authorization(&self, _name: &str, _args: &Value) -> bool {
        true
    }

    /// Decides whether to allow this tool call. Only invoked when
    /// [`requires_authorization`] returned `true`.
    ///
    /// [`requires_authorization`]: AuthManager::requires_authorization
    async fn authorize(&self, name: &str, args: &Value) -> Result<(), String>;
}
