# Plan: Automatic Conversation History Management

## 1. Executive Summary

Multi-turn conversation previously required callers to manually maintain a `Vec<Message>` and pass it via `RunBuilder::history` on every turn — easy to get wrong and unnecessarily verbose for the common case. The fix is a `Conversation` type that wraps an `AgentRunner` + `Agent` pair and automatically appends user and assistant messages after each completed turn. The history remains fully accessible and mutable so callers can implement compression, trimming, or synthetic-message injection without abandoning automatic bookkeeping for future turns.

---

## 2. User Stories

- **As a developer**, I want to build a multi-turn chatbot without tracking `Vec<Message>` myself.
- **As a developer**, I want to trim or compress conversation history (e.g., to stay within a context window) without having to switch to manual history management.
- **As a developer**, I want streaming multi-turn conversations where history is updated automatically when each stream completes.
- **As a developer**, I want the existing `RunBuilder` API to remain unchanged so existing code is not broken.

---

## 3. Functional Requirements

### 3.1 `Conversation` (new, in `src/conversation.rs`)

```rust
pub struct Conversation<'a> {
    runner: &'a AgentRunner,
    agent: &'a Agent,
    history: Vec<Message>,
}

impl<'a> Conversation<'a> {
    pub fn history(&self) -> &[Message];
    pub fn history_mut(&mut self) -> &mut Vec<Message>;
    pub async fn run(&mut self, input: &str) -> Result<AgentResult, Error>;
    pub fn run_stream<'b>(&'b mut self, input: &'b str) -> ConversationStream<'b>;
}
```

Created via `AgentRunner::conversation(&agent)` — not constructed directly by callers.

**`run`**: delegates to `RunBuilder` with `self.history.clone()`, awaits the result, then pushes `Message::user(input)` and `Message::assistant(&result.output)` onto `self.history` before returning.

**`run_stream`**: returns a `ConversationStream<'b>` that borrows `&'b mut self.history`. History is updated when the stream is exhausted; not modified if dropped early.

### 3.2 `ConversationStream` (new, in `src/conversation.rs`)

```rust
pub struct ConversationStream<'a> {
    inner: Pin<Box<dyn Stream<Item = Result<AgentEvent, Error>> + Send + 'a>>,
    history: &'a mut Vec<Message>,
    input: String,
    reply: String,
    done: bool,
}

impl Stream for ConversationStream<'_> {
    type Item = Result<AgentEvent, Error>;
    // Polls inner; accumulates TextDelta chunks into `reply`.
    // On Poll::Ready(None): pushes Message::user + Message::assistant to history.
    // On error: sets done = true; does not update history.
}
```

The `done` flag prevents re-polling after the stream ends or after an error.

### 3.3 `AgentRunner::conversation` (new method in `src/runner.rs`)

```rust
pub fn conversation<'a>(&'a self, agent: &'a Agent) -> Conversation<'a> {
    Conversation::new(self, agent)
}
```

---

## 4. Technical Architecture

### 4.1 Data Flow — `Conversation::run`

```
caller
  │  conv.run(input)
  ▼
Conversation::run
  │  RunBuilder::history(self.history.clone()).run(input).await
  ▼
AgentRunner (agentic loop)
  │  AgentResult { output }
  ▼
Conversation::run
  │  self.history.push(Message::user(input))
  │  self.history.push(Message::assistant(&output))
  └─► return AgentResult
```

### 4.2 Data Flow — `Conversation::run_stream`

```
caller
  │  let stream = conv.run_stream(input)   [borrows &mut conv.history]
  ▼
ConversationStream { inner, history, input, reply, done }
  │
  │  caller polls stream:
  │
  ├─ Poll::Ready(Some(Ok(TextDelta(chunk))))  → accumulate in reply, yield event
  ├─ Poll::Ready(Some(Ok(other event)))       → yield event unchanged
  ├─ Poll::Ready(Some(Err(e)))               → done = true, yield error (no history update)
  └─ Poll::Ready(None)
       │  history.push(Message::user(input))
       │  history.push(Message::assistant(reply))
       └─► done = true, yield None

  [stream dropped before None] → history unchanged
```

### 4.3 Borrowing Strategy

`ConversationStream<'b>` holds `&'b mut Vec<Message>` — a direct mutable borrow of `Conversation::history`. This means:

- While `ConversationStream` is alive, no other access to the `Conversation` is possible (the borrow checker enforces this).
- No `Arc<Mutex<…>>` is needed; the cost is zero at runtime.
- The constraint maps naturally to how the API is used: a caller cannot start a second turn while still consuming the first turn's stream.

The inner event stream is typed as `Pin<Box<dyn Stream<…> + Send + 'a>>` to avoid requiring the concrete stream type as a generic parameter on `ConversationStream`. This adds one allocation per `run_stream` call, accepted as a reasonable trade-off for API simplicity.

### 4.4 Early-Drop Behaviour

If `ConversationStream` is dropped before being fully consumed, no history update happens. This is intentional:

- The assistant reply was never fully received, so appending a partial reply to history would corrupt the conversation context.
- Callers that need history updated must consume the stream to completion.

### 4.5 Module Changes

| File | Change |
| :--- | :--- |
| `src/conversation.rs` | New file: `Conversation`, `ConversationStream` |
| `src/runner.rs` | Add `AgentRunner::conversation` method |
| `src/lib.rs` | Add `pub mod conversation`; re-export `Conversation`, `ConversationStream` |
| `examples/multi_turn.rs` | Rewrite to use `Conversation::run_stream` |
| `docs/PRD.md` | Add automatic history management as an explicit goal |
| `docs/SPEC.md` | Add `Conversation` / `ConversationStream` section; update module layout; update roadmap |
| `README.md` | Update multi-turn section; update core types table |
| `skills/agent-rig.md` | Update core types table; rewrite multi-turn section; update pitfalls |

---

## 5. Implementation Plan

| Phase | Task | Description |
| :--- | :--- | :--- |
| **Phase 1** | **`src/conversation.rs`** | Implement `Conversation` and `ConversationStream`. Add rustdoc for all public items. Add unit tests: `run_accumulates_history`, `history_mut_allows_trimming`, `run_stream_accumulates_history`, `run_stream_dropped_early_does_not_update_history`. |
| **Phase 2** | **`src/runner.rs`** | Add `AgentRunner::conversation` method with rustdoc. |
| **Phase 3** | **`src/lib.rs`** | Declare `pub mod conversation`; re-export `Conversation` and `ConversationStream`. |
| **Phase 4** | **`examples/multi_turn.rs`** | Replace manual `history` vec and `reply` accumulation with `runner.conversation(&agent)` and `conv.run_stream(input)`. |
| **Phase 5** | **Docs** | Update `PRD.md`, `SPEC.md`, `README.md`, `skills/agent-rig.md`. Create `docs/PLAN-conversation.md`. |

---

## 6. Design Constraints & Notes

- **No breaking changes.** `RunBuilder::history` remains unchanged. Existing multi-turn code continues to compile and behave identically.
- **`Conversation` is a thin wrapper.** All agentic loop logic stays in `RunBuilder`. `Conversation` adds only history tracking — no duplication of runner logic.
- **History cloned per turn.** `run` and `run_stream` clone `self.history` to pass to `RunBuilder`. This is consistent with the existing `RunBuilder::history(Vec<Message>)` API which takes ownership. For very long histories, callers can periodically trim via `history_mut()` to bound the clone cost.
- **Error handling.** If `run` returns an error, `self.history` is not modified — the failed turn leaves no trace. If `run_stream` yields an error mid-stream, `done` is set to `true` and history is not updated; the conversation is left in a consistent state matching the last successful turn.
- **`run_typed` not added to `Conversation`.** Multi-turn structured output is an unusual pattern. `RunBuilder::run_typed` remains available for callers who need it; adding it to `Conversation` would complicate the API without clear demand.
