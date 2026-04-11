# Plan: Streaming Response Support

## 1. Executive Summary

`AgentRunner::run` is opaque: callers get no visibility into tool calls, reasoning traces, or incremental text while the agent loop runs. The fix is a unified event stream API. `run_stream` yields a rich `AgentEvent` enum covering every observable thing the agent loop produces — tool call lifecycle, thinking/reasoning tokens, and incremental text deltas — while `run` is refactored to delegate to it, keeping the two in sync.

---

## 2. User Stories

- **As a developer**, I want to display tokens to the user as they are generated, without waiting for the full response.
- **As a developer**, I want to show a progress indicator when the agent is executing a tool call.
- **As a developer**, I want to log every tool call (name, args, result) for debugging without patching the runner.
- **As a developer**, I want to capture reasoning/thinking tokens from models that support extended thinking.

---

## 3. Functional Requirements

### 3.1 `ModelStreamChunk` (new, in `src/model.rs`)

Provider adapters stream `ModelStreamChunk` values. The runner wraps these into `AgentEvent`, adding `ToolCallStarted`/`ToolCallCompleted` itself.

```rust
pub enum ModelStreamChunk {
    /// A reasoning/thinking token from a model that supports extended thinking.
    Thinking(String),
    /// An incremental chunk of the model's text output.
    TextDelta(String),
    /// A complete tool call (not streamed mid-call).
    ToolCall(ToolCall),
}
```

### 3.2 `generate_stream` on `LlmModel` (new default method in `src/model.rs`)

```rust
fn generate_stream(
    &self,
    request: ModelRequest,
) -> Pin<Box<dyn Stream<Item = Result<ModelStreamChunk, Error>> + Send + '_>> {
    Box::pin(async_stream::stream! {
        let response = self.generate(request).await?;
        for call in response.tool_calls {
            yield Ok(ModelStreamChunk::ToolCall(call));
        }
        if let Some(text) = response.text {
            yield Ok(ModelStreamChunk::TextDelta(text));
        }
    })
}
```

The default implementation wraps `generate`, so all existing adapters continue to compile and work without changes. Adapters that want true streaming override this method.

### 3.3 `AgentEvent` (new, in `src/runner.rs`)

```rust
pub enum AgentEvent {
    /// The model requested a tool call. Emitted before the tool executes.
    ToolCallStarted {
        name: String,
        args: serde_json::Value,
    },
    /// A tool call completed. Emitted after the tool returns.
    ToolCallCompleted {
        name: String,
        result: serde_json::Value,
    },
    /// A reasoning/thinking token from a model that supports extended thinking.
    Thinking(String),
    /// An incremental chunk of the model's final text response.
    TextDelta(String),
}
```

Tool call events (`ToolCallStarted`, `ToolCallCompleted`) are emitted by the runner. `Thinking` and `TextDelta` are forwarded from the model's `generate_stream`.

### 3.4 `AgentRunner::run_stream` (new method in `src/runner.rs`)

```rust
pub fn run_stream<'a>(
    &'a self,
    agent: &'a Agent,
    input: &'a str,
) -> impl Stream<Item = Result<AgentEvent, Error>> + Send + 'a
```

Drives the full agentic loop using `model.generate_stream` per turn:

1. Validate that every tool name in `agent.tool_names()` is registered.
2. Consume the model stream for the current turn, collecting `ToolCall` chunks and forwarding `Thinking` / `TextDelta` events to the caller.
3. If the turn produced tool calls: emit `ToolCallStarted`, execute the tool, emit `ToolCallCompleted`, append result messages, and loop.
4. If the turn produced no tool calls and no `TextDelta` events: yield `Err(Error::Agent("model returned neither text nor tool calls"))` to preserve the existing error behaviour of `run`.
5. If the turn produced no tool calls but did produce text: terminate the outer loop normally.

### 3.5 `AgentRunner::run` delegates to `run_stream`

```rust
pub async fn run(&self, agent: &Agent, input: &str) -> Result<AgentResult, Error> {
    let mut output = String::new();
    let stream = self.run_stream(agent, input);
    futures_util::pin_mut!(stream);
    while let Some(event) = stream.next().await {
        if let AgentEvent::TextDelta(chunk) = event? {
            output.push_str(&chunk);
        }
    }
    Ok(AgentResult { output })
}
```

This keeps `run` and `run_stream` automatically in sync — any change to the loop logic benefits both.

---

## 4. Technical Architecture

### 4.1 Data Flow

```
caller
  │  run_stream(agent, input)
  ▼
AgentRunner::run_stream
  │
  │  loop per turn:
  │
  ├─► model.generate_stream(request)
  │       │
  │       ├─ ModelStreamChunk::Thinking(t)   ──► AgentEvent::Thinking(t)
  │       ├─ ModelStreamChunk::TextDelta(t)  ──► AgentEvent::TextDelta(t)
  │       └─ ModelStreamChunk::ToolCall(c)   ──► collected, not forwarded yet
  │
  │  if tool calls were collected:
  │
  ├─► yield AgentEvent::ToolCallStarted { name, args }
  ├─► tool.call(args)
  ├─► yield AgentEvent::ToolCallCompleted { name, result }
  └─► append tool result messages → loop
```

### 4.2 Module Changes

| File | Change |
| :--- | :--- |
| `Cargo.toml` | Add `async-stream = "0.3"` dependency |
| `src/model.rs` | Add `ModelStreamChunk` enum; add `generate_stream` default method to `LlmModel`; add `use std::pin::Pin` and `use futures_util::stream::Stream` |
| `src/runner.rs` | Add `AgentEvent` enum; add `run_stream` method; refactor `run` to delegate to `run_stream`; carry `#[instrument]` and `debug!` tracing through the new structure; add `use futures_util::StreamExt` |
| `src/models/ollama.rs` | Add native `generate_stream` override (streams `TextDelta` chunks as they arrive) |
| `src/lib.rs` | Re-export `AgentEvent` at the crate root |
| `docs/SPEC.md` | Document new types and methods; update roadmap |

### 4.3 Native Streaming: `OllamaModel`

`OllamaModel` already uses the Ollama streaming chat API internally — it just buffers all chunks before returning. The native `generate_stream` override will instead emit each `TextDelta` chunk as it arrives:

- When the request has no tools: yield `TextDelta` for each streamed chunk.
- When the request has tools (Ollama requires `stream(false)` for tool calls): collect the single-shot response and emit `ToolCall` chunks at the end.

`GeminiModel` keeps the default implementation for now. A native Gemini streaming implementation (using `generate_content_stream`, including `Thinking` token support) is tracked as a follow-on in issue #6.

### 4.4 Return Type Choice

`Pin<Box<dyn Stream<...>>>` is used as the `generate_stream` return type. This avoids RPIT-in-traits (`impl Trait`) in the `LlmModel` trait, which while stable in Rust 1.75+ can cause subtle issues when combined with `async_trait` and lifetime bounds. `Pin<Box<...>>` is the established safe choice for object-safe async streaming.

For `run_stream`, the return type is `impl Stream<...>` (RPIT on an inherent method, not a trait), which is fully stable and avoids unnecessary boxing at the runner level.

---

## 5. Implementation Plan

| Phase | Task | Description |
| :--- | :--- | :--- |
| **Phase 1** | **`Cargo.toml`** | Add `async-stream = "0.3"` to `[dependencies]`. |
| **Phase 2** | **`src/model.rs`** | Add `ModelStreamChunk` enum. Add `use std::pin::Pin` and `use futures_util::stream::Stream`. Add `generate_stream` default method to `LlmModel` using `async_stream::stream!`. Add rustdoc for all new items. |
| **Phase 3** | **`src/runner.rs`** | Add `AgentEvent` enum with rustdoc. Add `run_stream` method using `async_stream::try_stream!`. Refactor `run` to delegate to `run_stream`. Keep `#[instrument(skip(self, input), fields(agent = agent.name()))]` on `run`; add equivalent `debug!` tracing inside `run_stream` (starting run, turn counter, tool call count, per-tool start/complete, run complete). Add `use futures_util::StreamExt`. |
| **Phase 4** | **`src/models/ollama.rs`** | Add native `generate_stream` override. Emit `TextDelta` per streamed chunk (no-tools path); emit `ToolCall` chunks for the tool-calling path. |
| **Phase 5** | **`src/lib.rs`** | Re-export `AgentEvent` from the crate root. |
| **Phase 6** | **Unit tests** | In `src/runner.rs`, add four tests: (1) text delta events, (2) tool call events (ToolCallStarted + ToolCallCompleted + TextDelta), (3) thinking events (stub model overrides `generate_stream` directly), (4) error mid-stream. |
| **Phase 7** | **`docs/SPEC.md`** | Add `ModelStreamChunk` and `AgentEvent` to the Core Types section. Document `generate_stream` on `LlmModel`. Document `run_stream` on `AgentRunner`. Update the roadmap to mark streaming as done. |

---

## 6. Design Constraints & Notes

- **No breaking changes to `generate`.** The existing `generate` method and all provider adapters are unchanged. `generate_stream` is purely additive with a working default.
- **Thinking tokens deferred for Gemini.** `GeminiModel::generate` currently discards thought parts. Emitting `Thinking` events from Gemini requires a native `generate_stream` using `generate_content_stream`, which is a non-trivial change tracked separately in issue #6. The default implementation will not emit `Thinking` events for Gemini.
- **`run_typed` unchanged.** It calls `run`, which calls `run_stream`. No changes needed.
- **`AgentTool` unchanged.** It calls `runner.run(...)`, which delegates to `run_stream` internally. No changes needed.
- **Tool call ordering.** Within a single turn, all tool calls are collected before any `ToolCallStarted` event is emitted. This matches the existing behavior where all tool calls from a turn are grouped into one assistant message before execution begins.
- **Tracing preserved.** `run` retains its `#[instrument]` attribute. The `debug!` log points currently in `run` — starting run, turn number, tool call count, per-tool start/complete, run complete — move into `run_stream` so they fire regardless of whether the caller uses `run` or `run_stream`.
- **Empty-response error preserved.** `run_stream` yields `Err(Error::Agent("model returned neither text nor tool calls"))` when a model turn produces neither tool calls nor any `TextDelta`. This matches the current `run` behaviour and prevents silent empty-string returns.
