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
///   Returns `true` to allow the call, `false` to deny it. Denial is a
///   binary outcome (modelled after an accept/decline approval prompt);
///   the runner reports it via
///   [`ToolCallResult::Denied`](crate::runner::ToolCallResult::Denied) with
///   no accompanying reason.
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
    /// Decides whether the call with these arguments may run.
    ///
    /// Returning `true` allows the runner to execute the tool; returning
    /// `false` surfaces a [`ToolCallResult::Denied`](crate::runner::ToolCallResult::Denied)
    /// without invoking the tool.
    ///
    /// `id` is the tool call's identifier — the same id the runner later
    /// reports on [`AgentEvent::ToolCallStarted`](crate::runner::AgentEvent::ToolCallStarted)
    /// and `ToolCallFinished`. Implementations that surface the approval
    /// prompt out-of-process (an editor permission request, a GUI dialog
    /// keyed by id, a remote approval service) can use it to correlate the
    /// prompt with the tool call the runner reports.
    ///
    /// `args` is the raw JSON the model requested. `proposal` is what the tool
    /// resolved those args into via
    /// [`Tool::propose`](crate::tools::Tool::propose): the concrete thing that
    /// will happen if approved (for an edit tool, the path plus old and new
    /// contents — enough to render a diff). The approved proposal is handed
    /// verbatim to [`Tool::apply`](crate::tools::Tool::apply), so what the
    /// prompt shows is exactly what runs. For a tool that does no planning the
    /// proposal equals `args`. Rendering it for a human is the manager's job:
    /// a tool-aware manager matches on `name` and renders accordingly; a
    /// generic one can display the JSON.
    ///
    /// This is the async decision path — block on user input, RPCs, or any
    /// other I/O here. See the trait docs for concurrency requirements.
    async fn authorize(&self, id: &str, name: &str, args: &Value, proposal: &Value) -> bool;
}
