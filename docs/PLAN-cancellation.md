# Plan: Cancellation Support for AgentRunner

Tracking issue: [#29](https://github.com/andreban/rust-agent-kit/issues/29)

## 1. Executive Summary

`AgentRunner::run` spawns the agentic loop on `tokio::spawn` and returns a `Stream<Item = RunEvent>` fed by an mpsc channel. Today, dropping the returned stream only closes the receiver; the spawned task notices when it next tries to `tx.send`, which can be after the next full provider response or after a long-running tool finishes. The in-flight HTTP call and the tool future keep running.

This plan adds **cooperative cancellation** that propagates from the consumer down through the runner, the model HTTP call, and the tool execution — including sub-agent runs invoked via `AgentTool`. The default ergonomics are deliberately quiet: dropping the returned stream cancels everything, so consumers that already drop their stream on Ctrl-C / client disconnect get correct behaviour with no API change. Consumers that need to hand a token to someone else (timer task, sibling runner) get `run_with_cancellation`.

---

## 2. User Stories

- **As a TUI / editor adapter author**, I want dropping the stream on Ctrl-C to immediately abort the provider call and any running tool, so a billed API call doesn't keep running after the user cancelled.
- **As an HTTP server author**, I want the agent to stop when the client disconnects (the stream is dropped on the response future), without writing a token-plumbing layer myself.
- **As a tool author**, I want a `CancellationToken` so my long-running tool (file scan, RPC, child process) can shut down cleanly when the agent is cancelled.
- **As an advanced consumer**, I want to supply my own `CancellationToken` so I can cancel the run from a sibling task (deadline timer, multi-run coordination) and share the same token across nested agents.
- **As an integrator of nested agents**, I want cancellation to flow into every sub-agent invoked via `AgentTool`, so cancelling the root agent cancels the whole tree.

---

## 3. Public API

### 3.1 `AgentRunner::run` — behaviour change, no signature change

```rust
pub fn run(
    &self,
    agent: &Agent,
    thread: Vec<Message>,
) -> Pin<Box<dyn Stream<Item = RunEvent> + Send>>;
```

Same signature. New behaviour: **dropping the returned stream cancels the run**. Internally, `run` delegates to `run_with_cancellation` with a fresh `CancellationToken`.

### 3.2 `AgentRunner::run_with_cancellation` — new

```rust
pub fn run_with_cancellation(
    &self,
    agent: &Agent,
    thread: Vec<Message>,
    cancel: CancellationToken,
) -> Pin<Box<dyn Stream<Item = RunEvent> + Send>>;
```

Same semantics as `run`, plus: when `cancel` fires the runner cancels too. The two trigger paths compose — whichever happens first wins.

The supplied `cancel` is the caller's token; the runner does **not** cancel it. The runner derives an internal child token (`cancel.child_token()`) and binds the drop-on-stream-drop guard to that child, so dropping the stream cancels the run without cancelling the caller's token (which may be shared with siblings).

### 3.3 `Tool::call` — breaking signature change

```rust
#[async_trait]
pub trait Tool: Send + Sync {
    fn definition(&self) -> ToolDefinition;
    async fn call(
        &self,
        args: serde_json::Value,
        cancel: CancellationToken,   // NEW
    ) -> Result<serde_json::Value, Error>;
}
```

Every tool receives a token clone tied to the run. Tools that don't care about cancellation ignore the parameter; long-running tools `select!` on `cancel.cancelled()` or pass it down to the libraries they call. See §4.4 for what the runner does when a tool returns a value after cancellation.

### 3.4 `AgentTool::call` — signature change for parity

```rust
impl AgentTool {
    pub async fn call(
        &self,
        tx: RunEmitter,
        args: serde_json::Value,
        cancel: CancellationToken,   // NEW
    ) -> Result<serde_json::Value, Error>;
}
```

The child agent's runner is invoked via `run_with_cancellation(child_agent, thread, cancel)`, so cancellation flows transparently into nested runs without the parent runner having to special-case the registry entry.

### 3.5 `AgentEvent::Cancelled` — new terminal variant

```rust
pub enum AgentEvent {
    // ... existing variants ...
    /// The run was cancelled (via dropped stream or external token). The
    /// stream ends after this event. May not be observed if the consumer
    /// triggered cancellation by dropping the stream — the send is racy
    /// and consumer-side delivery requires the receiver to still be alive.
    Cancelled,
    Error(crate::error::Error),
}
```

Emitted at most once per run, after the loop notices cancellation and before the spawned task exits. A run that ends normally (model produced no more tool calls) emits no `Cancelled`. A run that ends via `AgentEvent::Error` emits no `Cancelled`. The three terminators are mutually exclusive.

---

## 4. Functional Requirements

### 4.1 Cancellation propagation points

The runner checks the cancel token at every meaningful await:

| Point                                | Mechanism                                                                                                  |
|--------------------------------------|------------------------------------------------------------------------------------------------------------|
| Top of the main loop                 | `if cancel.is_cancelled() { emit Cancelled; return }` before constructing the next `ModelRequest`.        |
| Awaiting `model_stream.next()`       | `tokio::select! { _ = cancel.cancelled() => …, chunk = model_stream.next() => … }`. On cancel, drop the stream — this drops the underlying `reqwest` future, aborting the in-flight HTTP call. |
| Awaiting tool futures (`join_all`)   | Each per-call future selects on `cancel.cancelled()`; on cancel, returns a synthetic `ToolCallResult` so `join_all`'s output stays paired with the assistant turn. See §4.4. |
| Awaiting `auth.authorize`            | `select!` against `cancel.cancelled()`; on cancel, the call short-circuits with `Cancelled` and skips both the tool invocation and any `Started`/`Finished` emission. |

The token is also cloned into each `Tool::call` and `AgentTool::call`, so cooperating tools can abort early; the runner does not rely on tools cooperating.

### 4.2 Drop-the-stream wiring

The stream returned by `run` / `run_with_cancellation` is a wrapper that owns a `DropGuard` for the internal cancel token. Sketch:

```rust
pub fn run_with_cancellation(
    &self,
    agent: &Agent,
    thread: Vec<Message>,
    external: CancellationToken,
) -> Pin<Box<dyn Stream<Item = RunEvent> + Send>> {
    let internal = external.child_token();
    let cloned = self.clone();
    let agent = agent.clone();
    let token_for_loop = internal.clone();

    let stream = async_stream::stream! {
        // Moved into the generator: dropping the stream drops `_guard`,
        // which fires `internal` and cancels the spawned task.
        let _guard = internal.drop_guard();

        let (tx, mut rx) = mpsc::channel::<RunEvent>(EVENT_CHANNEL_CAPACITY);
        let tx = RunEmitter::new(tx, None);
        tokio::spawn(cloned.main_loop(tx, agent, thread, token_for_loop));

        while let Some(message) = rx.recv().await {
            yield message;
        }
    };
    Box::pin(stream)
}
```

Why a child token, not the caller's token directly:

- Dropping the stream must not cancel the caller's `external` token (it may be shared with siblings).
- The caller cancelling `external` must still propagate — which a child token does for free.

Why `DropGuard` lives inside the `async_stream!` block: the generator future is what `Pin<Box<dyn Stream>>` actually owns. When the consumer drops the boxed stream, the generator future is dropped, and any owned values (including `_guard`) drop with it.

### 4.3 Order-of-operations for in-flight HTTP

When `cancel` fires while the loop is parked on `model_stream.next().await`:

1. `select!` resolves the `cancel.cancelled()` branch.
2. The other branch (the `next()` future) is dropped.
3. Dropping the `next()` future drops the underlying provider stream — `reqwest`'s response future, which the adapter's `chat`/`generate_content` call constructs. This aborts the HTTP request at the TCP layer.
4. The runner emits `AgentEvent::Cancelled` and returns from `main_loop`.

This works for both adapters today because:
- `GeminiModel::generate_stream` uses the default impl, which awaits `self.generate(request)` — a single reqwest future. Cancel-drop aborts it.
- `OllamaModel::generate_stream` returns an `async_stream` that polls a streaming reqwest body. Cancel-drop on `.next()` ends the stream and drops the body.

No adapter-level changes are required for cancellation to interrupt the HTTP call. Adapters are free to add explicit `select!` if they ever do work after the body finishes streaming, but that's not the case today.

### 4.4 Cancellation during tool execution

Each tool future runs the full per-call lifecycle (auth check → Started → call → Finished). With cancellation, each future races against `cancel.cancelled()`:

```rust
tokio::select! {
    _ = cancel.cancelled() => {
        // No Finished event — the run is cancelling, and we want the
        // sequence to terminate cleanly with a single Cancelled.
        (call, ToolCallResult::Err(Error::Agent("cancelled".into())))
    }
    result = tool.call(call.args.clone(), cancel.clone()) => {
        // Existing path: emit Finished, return result.
        ...
    }
}
```

The synthetic `Err("cancelled")` result keeps the thread paired (each assistant tool-call has a matching tool-result Message) — useful only if some future code path replays the thread; in practice the runner emits `Cancelled` and exits before the thread is used again, so the synthetic result is bookkeeping, not user-visible.

**`Started` / `Finished` emission under cancellation:**

- Cancel fires *before* `Started` is emitted (during auth gating): no `Started`, no `Finished`. The runner emits `Cancelled` and exits.
- Cancel fires *after* `Started` but *before* `tool.call` returns: no `Finished` for that call. Loose `Started` events are part of the cancellation semantics — consumers should treat a trailing `Started` without `Finished` as cancelled (or just stop rendering when `Cancelled` arrives).

Tools that ignore `cancel` continue running to completion; their result is discarded by the `select!` arm that took the `cancel.cancelled()` branch first. The detached future is dropped (not awaited), so any side effects already in flight complete or not depending on the tool's own behaviour. Documenting this in `Tool::call`'s rustdoc: **a tool that ignores cancel does not block the runner from terminating, but cannot prevent its own side effects from continuing in the background**.

### 4.5 Sub-agent (`AgentTool`) propagation

`AgentTool::call` receives the parent runner's `cancel` token (§3.4). Inside, it invokes the child runner via `run_with_cancellation(child_agent, thread, cancel)`. The child runner makes its own child token from the parent's, so:

- Dropping the parent's stream cancels the child's internal token via the parent's `DropGuard`.
- Cancelling the parent's external token cancels the child via the child-token chain.
- Dropping the child's stream (which the AgentTool internally consumes) cancels only the child run, not the parent — but in practice `AgentTool::call` is itself one of the futures the parent runner is racing against `cancel`, so this scenario doesn't happen.

The child's emitter is still `tx.child()` — the run-id attribution behaviour is unchanged.

### 4.6 Cancelled emission timing

The runner emits `AgentEvent::Cancelled` immediately after observing cancel at any of the four checkpoints (§4.1) and before returning from `main_loop`. The emission is best-effort:

```rust
let _ = tx.send(AgentEvent::Cancelled).await;
return;
```

If the consumer dropped the stream (the trigger), the receiver is gone and the send fails silently. If the consumer cancelled externally (token-triggered), the receiver is alive and the event arrives. Either way the spawned task exits and the stream ends.

### 4.7 What the runner does NOT do

- It does not call `JoinHandle::abort()` on the spawned task. Cooperative shutdown only — non-cancel-safe code (CPU-bound `block_on`, locks held across `.await`) keeps running until its next checkpoint. Recommended at V1; revisit if a future contributor adds work that genuinely can't be interrupted cooperatively.
- It does not expose a `CancellationReason` distinguishing user-cancel vs deadline vs upstream-cancel. The `Cancelled` variant is reason-less. Consumers that need attribution own the token they pass in and know why they fired it.
- It does not surface the partial thread (messages accumulated so far) on cancellation. Full rollback — consumers reconstruct partial state from the `TextDelta` events they already received.

---

## 5. Implementation Steps

1. **Dependency check** (`Cargo.toml`)
   - `tokio-util` is already `0.7` with the `rt` feature. Confirm `CancellationToken` and `DropGuard` are reachable via `tokio_util::sync::*` under that feature set; if not, add the minimal feature needed (likely no change required).

2. **`AgentEvent::Cancelled`** (`src/runner/events.rs`)
   - Add the variant with rustdoc noting (a) it's terminal, (b) delivery is best-effort under stream-drop cancellation, (c) it's mutually exclusive with `Error` as a terminator.
   - Update the module rustdoc bullet list accordingly.

3. **`Tool::call` signature** (`src/tools/tool.rs`)
   - Add `cancel: CancellationToken` as the second parameter.
   - Update the trait's rustdoc example to show ignoring the token (the common case).
   - Update the doctest accordingly.

4. **`AgentTool::call` signature** (`src/tools/agent_tool/mod.rs`)
   - Add `cancel: CancellationToken` as the third parameter.
   - Inside, change `self.runner.run(&self.agent, vec![…])` to `self.runner.run_with_cancellation(&self.agent, vec![…], cancel)`.

5. **`AgentRunner::run_with_cancellation`** (`src/runner/mod.rs`)
   - Implement as in §4.2.
   - Re-implement `run` as a one-liner: `self.run_with_cancellation(agent, thread, CancellationToken::new())`.

6. **`main_loop` cancellation** (`src/runner/mod.rs`)
   - Take `cancel: CancellationToken` as a parameter.
   - Top-of-loop check (§4.1): `if cancel.is_cancelled() { let _ = tx.send(AgentEvent::Cancelled).await; return; }`.
   - Wrap the `while let Some(chunk) = model_stream.next().await` in `tokio::select!` against `cancel.cancelled()`; on cancel, drop `model_stream`, emit `Cancelled`, return.
   - Pass `cancel.clone()` into `handle_tool_calls`.

7. **`handle_tool_calls` cancellation** (`src/runner/mod.rs`)
   - Per-call future races `cancel.cancelled()` against the body (§4.4). On cancel, return synthetic `Err("cancelled")` result without emitting `Finished`.
   - After `join_all`, check `cancel.is_cancelled()`. If set, emit `Cancelled` and return without re-entering the loop. (The `Cancelled` emission in `main_loop`'s top-of-loop check would otherwise duplicate; consolidate into a single emission site.)
   - Pass `cancel.clone()` into each `tool.call(...)` and `agent_tool.call(...)`.
   - Auth path: `select!` `cancel.cancelled()` against `auth.authorize(...)`; on cancel, return synthetic `Err("cancelled")` (no `Started` / `Finished`).

8. **Update every `Tool` impl in the crate** (test doubles + agent_tool tests + registry tests + doctests)
   - `src/runner/tests.rs::EchoTool`: add `_cancel` parameter, ignore.
   - `src/tools/registry.rs::tests::StubTool`: same.
   - Any `Tool` doctest in `tool.rs`: same.
   - No external Tool impl is in-tree besides examples; see step 10.

9. **Update examples** (`examples/`)
   - Every example that defines a `Tool` impl needs the new parameter. Grep for `impl Tool for` and add `_cancel: tokio_util::sync::CancellationToken` (or pass it down to a real lib).
   - Add a new `examples/cancellation.rs` demonstrating:
     - dropping the stream mid-run cancels;
     - passing an external token + a deadline cancels;
     - a tool that respects the token aborts cleanly.
   - Add the example to `Cargo.toml` with `required-features = ["gemini"]`.

10. **Tests** (`src/runner/tests.rs`)
    - Add a test: dropping the stream after one event cancels the loop within a bounded wait. (Use a `ScriptedModel` whose `generate` future selects on a `tokio::sync::Notify` so the runner is parked when the test drops the stream; assert the notify is never released, i.e. the future was dropped — or simpler: have `generate` return a pending future and `tokio::time::timeout` the test.)
    - Add a test: `run_with_cancellation` with an externally-fired token emits `AgentEvent::Cancelled` and ends.
    - Add a test: cancellation during the tool-call phase emits `Cancelled` and does **not** emit `Finished` for in-flight tools.
    - Add a test: `Tool::call` receives a token that fires when the run cancels (use a tool that records `cancel.is_cancelled()` after a short await).
    - Add a test: nested `AgentTool` propagates cancel — cancelling the parent's external token aborts the child's model call.
    - Update `kinds` vector in `thinking_chunks_are_forwarded` (and any other exhaustive `match` on `AgentEvent`) to include the new `Cancelled` arm.

11. **Documentation** (`docs/SPEC.md`)
    - Update `AgentEvent` listing to include `Cancelled` with the terminator note.
    - Update `AgentRunner` listing to show both `run` and `run_with_cancellation`, and document the drop-cancels-the-run contract.
    - Update `Tool` trait signature in the Core Types section.
    - Update `AgentTool::call` signature.
    - Add a "Cancellation" subsection under `AgentRunner` summarising the four checkpoints (§4.1) and the drop-the-stream semantics.

12. **Skill update** (`skills/agent-rig.md`)
    - One-line note on the new `Tool::call` signature and on dropping the stream as the simple cancel mechanism. Don't over-explain — link to SPEC.md.

---

## 6. Out of Scope

- **Preemptive cancellation.** No `JoinHandle::abort()`. Cooperative only at V1 (§4.7).
- **Per-tool cancel.** A single token covers the whole run; no way to cancel one tool call without cancelling the others. Revisit if a real consumer needs it.
- **`CancellationReason`.** No attribution on `Cancelled`. Consumers own their tokens and know why they fired.
- **Partial thread on cancel.** Full rollback; the runner does not expose the in-progress thread on `Cancelled`. Consumers reconstruct from received `TextDelta`s.
- **Pause / resume.** Just abort.
- **Timeout convenience method.** No `run_with_deadline`. Consumers compose `tokio::time::timeout` with `run_with_cancellation` themselves (the example will show the pattern).

---

## 7. Resolved Decisions

- **Drop-the-stream is the primary trigger.** Simple consumers (TUI, HTTP server) get correct behaviour without any token plumbing. The original issue listed this as option (3) and noted "depends on the consumer holding the stream, easy to lose track of who owns it"; that risk is real, but for this crate's audience — agents inside event-driven UIs and request handlers — the consumer almost always owns the stream tightly, and the alternative (forcing every consumer to construct and plumb a token) is worse boilerplate.
- **Both triggers coexist via `run_with_cancellation`.** Caller's token + DropGuard via a child token. No either-or; whichever fires first wins.
- **`Tool::call` takes the token directly, not as `Option`.** No backwards-compat shim. The crate has no external `Tool` impls in v0.1, migration is a one-parameter addition per impl, and `Option<CancellationToken>` would force every cooperating tool to write the same `if let Some(c) = cancel { … }` wrapper.
- **Single shared token, not per-tool child tokens.** The runner clones the same token into every concurrent tool future. Per-tool isolation would let one tool's cancellation finish that tool without cancelling siblings, which is a feature we don't have a user for; if it lands later, swap the clone for a `child_token()` at the per-tool site without changing the trait.
- **`Cancelled` is the terminal event; no `CancellationReason`.** Matches the issue's V1 recommendation.

---

## 8. Future Considerations

- **Per-tool cancellation handle.** Once a real consumer needs to cancel one tool without cancelling the run, derive a child token per tool call inside `handle_tool_calls`. The `Tool::call` signature already takes a token, so the change is local to the runner.
- **`AgentEvent::Cancelled { reason }`.** If multiple distinct cancel sources (deadline vs user vs upstream) need to be told apart in observability, add a `CancellationReason` enum. Until then, callers attribute via the token they own.
- **Preemptive abort.** If a future contributor adds genuinely uncancellable work (sync CPU loop, blocking lock), the runner could spawn that work on a dedicated task and `abort()` it. Not needed today.
- **Stream-drop discoverability.** "Dropping the stream cancels" is documented in rustdoc but subtle. If users hit confusion, add a short SPEC.md callout and link from the README.
