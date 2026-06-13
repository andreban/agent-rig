# Technical Specification — agent-rig

## Architecture

```
┌──────────────┐     runs      ┌──────────────┐
│  AgentRunner │──────────────▶│    Agent     │
│              │               └──────────────┘
│  holds:      │               (pure blueprint)
│   model      │
│   registry   │
│   auth?      │
└──────┬───────┘
       │ holds
┌──────▼───────────┐
│  Arc<dyn LlmModel>│
└──────┬───────────┘
       │ implements
┌──────┴──────────────┐
▼                     ▼
GeminiModel     OllamaModel    (more providers …)
```

The library is a single crate (`agent-rig`). Provider adapters live in `src/models/`. Agent logic lives in `src/agent.rs` and `src/runner/`. The `LlmModel` trait in `src/model.rs` is the extension point. Tool types live under `src/tools/`. The authorization hook lives in `src/auth.rs`.

`Agent` is a pure data blueprint (name, instructions, optional output schema, optional tool name list) with no model reference. `AgentRunner` owns an `Arc<dyn LlmModel>` (cheap to clone), a `ToolRegistry` (also `Arc`), and an optional `AuthManager`. The runner is cheap to `Clone` so a single runner can be shared across tasks; the same runner can execute multiple agents, and the same agent can be run by different runners backed by different models.

The runner streams: `AgentRunner::run` spawns the agentic loop on a background tokio task and returns a `Stream<Item = RunEvent>`. A `RunEvent` wraps an `AgentEvent` together with the `run_id` of the run that produced it and an optional `parent` run id (set when the event came from a sub-agent invoked via `AgentTool`). Provider errors are surfaced as `AgentEvent::Error(Error)` and terminate the stream; the stream item type is bare `RunEvent`, not `Result`.

Cancellation is cooperative. Dropping the returned stream cancels the run — the in-flight provider call and any running tool futures are dropped at their next await point. Callers that need to share a cancel signal with a sibling task use `AgentRunner::run_with_cancellation(agent, thread, cancel)`; the runner derives an internal child token from `cancel`, so dropping the stream cancels the run without cancelling the caller's token. A cancelled run emits a terminal `AgentEvent::Cancelled` (best-effort under stream-drop) before the stream ends.

## Core Types

### `LlmModel` (`src/model.rs`)

```rust
#[async_trait]
pub trait LlmModel: Send + Sync {
    async fn generate(&self, request: ModelRequest) -> Result<ModelResponse, Error>;

    fn generate_stream(
        &self,
        request: ModelRequest,
    ) -> Pin<Box<dyn Stream<Item = Result<ModelStreamChunk, Error>> + Send + '_>> { ... }
}
```

All provider adapters implement this trait. The runner holds an `Arc<dyn LlmModel>` so providers are interchangeable at runtime without generics on the public API.

`generate_stream` has a default implementation that calls `generate` and emits the result as a sequence of `ModelStreamChunk` values (one `Thinking`, the tool calls in order, then one `TextDelta`), so existing adapters compile without changes. Adapters that support true streaming override this method.

### `ModelStreamChunk` (`src/model.rs`)

```rust
pub enum ModelStreamChunk {
    Thinking(String),       // reasoning/thinking token (extended thinking models)
    TextDelta(String),      // incremental text output chunk
    ToolCall(ToolCall),     // a complete tool call (not streamed mid-call)
    Usage(TokenUsage),      // per-call token counts (at most one per stream)
}
```

Emitted by [`LlmModel::generate_stream`]. The runner wraps these into [`AgentEvent`], adding tool call lifecycle events on top. `Usage` is yielded at most once per `generate_stream` invocation — providers that do not report token counts simply never yield it.

### `AgentEvent` and `RunEvent` (`src/runner/events.rs`)

```rust
pub enum AgentEvent {
    /// Emitted before authorization and execution. A denied call therefore
    /// still emits this, followed by `ToolCallFinished { Denied }`.
    /// Hallucinated tool calls (no matching registry entry) do not emit this.
    /// `tool_id` is the provider-assigned call identifier (matching `ToolCall::id`);
    /// use it to correlate with the matching `ToolCallFinished`, since events
    /// from parallel calls in a turn may interleave. `title` is a
    /// human-readable display label for the call, derived from the tool's
    /// `Tool::title(&args)` (defaulting to the tool name).
    ToolCallStarted { tool_id: String, name: String, args: serde_json::Value, title: String },
    /// Emitted after a tool resolves, errors, or is denied. `tool_id` matches the
    /// corresponding `ToolCallStarted`.
    ToolCallFinished { tool_id: String, name: String, result: ToolCallResult },
    /// Reasoning token forwarded from the model stream.
    ThinkingDelta(String),
    /// Incremental text chunk forwarded from the model stream.
    TextDelta(String),
    /// Token counts reported by the provider for one model call.
    /// A run that issues N model calls produces up to N `Usage` events.
    Usage(TokenUsage),
    /// First event of every run, before any model output.
    StartTurn,
    /// Last event on normal completion (no tool calls in the final model turn).
    /// `thread` is the full conversation thread as it stood when the loop
    /// exited, for carrying multi-turn state forward. Not emitted on the
    /// `Cancelled` or `Error` paths.
    EndTurn { thread: Vec<Message> },
    /// The run was cancelled via dropped stream or external token.
    /// Terminal — the stream ends after this event. Delivery is
    /// best-effort under stream-drop (the receiver may already be gone).
    Cancelled,
    /// The provider returned an error. The stream ends after this event.
    Error(Error),
}

pub struct RunEvent {
    /// Unique-per-process identifier of the run that produced this event.
    pub run_id: usize,
    /// `run_id` of the run that invoked this one (sub-agent invocation),
    /// or `None` for a root run.
    pub parent: Option<usize>,
    pub agent_event: AgentEvent,
}
```

`AgentEvent` is the union of things the runner reports as it drives the agentic loop. Tool-call lifecycle events are generated by the runner; `ThinkingDelta` and `TextDelta` are forwarded from the model. Concatenating every `TextDelta` reconstructs the model's final reply.

`RunEvent` is what [`AgentRunner::run`] actually yields — every event is tagged with the identity of the run that produced it, so consumers can distinguish events from a root run and from any sub-agents invoked via `AgentTool`. For a flat single-run consumer the extra fields can be ignored.

### `ToolCallResult` (`src/runner/events.rs`)

```rust
pub enum ToolCallResult {
    Ok(serde_json::Value),  // tool returned successfully
    Err(Error),             // tool returned an error
    Denied,                 // AuthManager denied the call
    Unknown,                // model called a tool not registered in the registry
}
```

The outcome carried by `AgentEvent::ToolCallFinished`. `Unknown` is a special case: hallucinated tool calls emit *no* `ToolCallStarted` event, but the runner still appends a synthetic tool-result message to the thread so the assistant turn and tool-result messages stay paired. A `Denied` call, by contrast, *does* emit `ToolCallStarted` (emitted before the authorization gate) followed by `ToolCallFinished { Denied }`.

### `ModelRequest` / `ModelResponse` / `ToolCall` (`src/model.rs`)

```rust
pub struct ModelRequest {
    pub messages: Vec<Message>,                    // conversation history
    pub system: Option<String>,                    // system prompt
    pub output_schema: Option<schemars::Schema>,   // JSON Schema for structured output, typed
    pub tools: Vec<ToolDefinition>,                // tool definitions for this turn
}

pub struct ToolCall {
    pub id: String,                                    // provider-supplied call ID
    pub name: String,
    pub args: serde_json::Value,
    pub provider_metadata: Option<serde_json::Value>,  // opaque, round-tripped (e.g. Gemini thought_signature)
}

pub struct ModelResponse {
    pub text: Option<String>,          // None when the model issued tool calls
    pub tool_calls: Vec<ToolCall>,     // empty on a final text response
    pub thinking: Option<String>,      // reasoning trace; only set by Gemini when include_thoughts is enabled
    pub token_usage: Option<TokenUsage>, // per-call token counts; None when the provider did not report usage
}

pub struct TokenUsage {
    pub input_tokens: Option<u32>,            // prompt / input tokens
    pub output_tokens: Option<u32>,           // generated / output tokens
    pub cached_input_tokens: Option<u32>,     // subset of input_tokens served from cache
    pub thinking_tokens: Option<u32>,         // reasoning tokens billed separately
    pub tool_use_prompt_tokens: Option<u32>,  // tool-use prompt tokens billed separately (Gemini)
}
```

`Message` carries a `Role` (`User` | `Assistant`) and a `content: MessageContent`. `MessageContent` is an enum with three variants:
- `Text(String)` — a plain text turn
- `ToolCalls(Vec<ToolCall>)` — all tool calls from one model turn (one assistant message)
- `ToolResult { id, name, result, provider_metadata }` — the result of one tool execution (one user message)

`output_schema`, when set, instructs the provider adapter to constrain the response to the supplied JSON Schema. Providers that do not support structured output ignore the field silently.

`text` and `tool_calls` on `ModelResponse` are mutually exclusive per turn.

`ToolCall::provider_metadata` is opaque metadata the runner round-trips back to the model on the next turn (used by Gemini to echo `thought_signature` on both the replayed `FunctionCall` and the matching `FunctionResponse` parts).

### `Agent` (`src/agent.rs`)

```rust
#[derive(Serialize, Deserialize)]
pub struct Agent {
    name: String,
    instructions: String,
    output_schema: Option<schemars::Schema>,
    tool_names: Vec<String>,
}
```

Constructed via `Agent::builder()`. A pure data blueprint: holds the system instructions used on every run, an optional JSON Schema for structured output, and the names of tools the agent is permitted to use. Carries no model or runtime state. Derives `Serialize`/`Deserialize` so agent configurations can be saved to and loaded from files. Tool definitions (description, parameters schema) are *not* serialized with the agent — they live in the `ToolRegistry` alongside their `Tool` implementation and are resolved at runtime.

`output_schema` is set via `AgentBuilder::output_schema(schema)`. The runner copies it into every `ModelRequest`, and each provider adapter applies it using provider-specific mechanisms.

### `ToolDefinition` / `Tool` / `ToolRegistry` (`src/tools/`)

```rust
// src/tools/tool.rs
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: schemars::Schema,    // JSON Schema, typed
}

// The object-safe trait the registry stores. Arguments and results are
// untyped JSON, so a single registry can hold tools of any shape behind
// `Box<dyn Tool>`.
//
// A call runs in two phases: `propose` resolves the args into a proposal
// (side-effect free, before authorization), then `apply` executes it.
#[async_trait]
pub trait Tool: Send + Sync {
    /// Borrows the tool's definition; tools own a `ToolDefinition` and return
    /// a reference to it rather than rebuilding one per call.
    fn definition(&self) -> &ToolDefinition;
    /// Human-readable display label for a specific invocation. Defaults to the
    /// tool name; override to surface argument-dependent labels. Returns `Err`
    /// when the args cannot be interpreted (the runner falls back to the name).
    fn title(&self, args: &serde_json::Value) -> Result<String, Error> {
        Ok(self.definition().name.clone())
    }
    /// Resolves `args` into a proposal — the concrete thing that will happen —
    /// without side effects, before authorization. The returned value is shown
    /// to the `AuthManager` and then handed verbatim to `apply`. The default
    /// returns `args` unchanged; override it to do real planning (e.g. read a
    /// file and return `{ path, old_text, new_text }` so a diff can be shown).
    async fn propose(
        &self,
        args: &serde_json::Value,
        progress: &dyn ProgressReporter,
        cancel: tokio_util::sync::CancellationToken,
    ) -> Result<serde_json::Value, Error> { Ok(args.clone()) }
    /// Executes the approved `proposal` (the value `propose` returned).
    async fn apply(
        &self,
        proposal: serde_json::Value,
        progress: &dyn ProgressReporter,
        cancel: tokio_util::sync::CancellationToken,
    ) -> Result<serde_json::Value, Error>;
}

// src/tools/registry.rs
pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>,    // AgentTool is just another `Tool`
}

impl ToolRegistry {
    pub fn new() -> Self;
    pub fn register<T: Tool + 'static>(self, tool: T) -> Self;     // builder-style
    pub fn definitions(&self) -> Vec<ToolDefinition>;
}
```

`ToolDefinition` is the runtime-only contract between agent and model. It is never stored in `Agent` — it lives in the `ToolRegistry` alongside its implementation and is resolved at run time. `parameters` is a `schemars::Schema`, typically built with the `schemars::json_schema!` macro.

`Tool` is the object-safe trait the registry stores: its arguments and results are raw `serde_json::Value`s. A call runs in two phases. `propose` resolves the model's raw `args` into a *proposal* — the concrete thing that will happen — without side effects and before authorization; it returns a single JSON value. The runner shows that proposal to the `AuthManager` and, if approved, hands the *same* value to `apply`. Because one value drives both the prompt and the execution, what the approver sees and what runs can never drift. An edit tool's `propose` reads the file and returns `{ path, old_text, new_text }`; the auth manager renders a diff from it and `apply` writes `new_text`. `propose` has a default that returns `args` unchanged, so a tool that needs no planning only implements `apply`. The proposal is opaque to the runner — rendering it for a human is the `AuthManager`'s job (a tool-aware manager matches on the tool name; a generic one displays the JSON).

A single registry can hold tools of different shapes because everything is stored as `Box<dyn Tool>`. `register` accepts any `Tool` and boxes it directly — there is no separate wrapper type.

The registry holds two kinds of callables: plain [`Tool`] implementations and sub-agents wrapped in [`AgentTool`]. The runner dispatches each variant differently — plain tools resolve to a single JSON value, while agents produce a stream of events that the parent forwards.

`ToolRegistry` is independent of any runner; share it across multiple runners via `Arc<ToolRegistry>`.

### `AgentRunner` (`src/runner/mod.rs`)

```rust
#[derive(Clone)]
pub struct AgentRunner {
    model: Arc<dyn LlmModel>,
    registry: Arc<ToolRegistry>,
    auth_manager: Option<Arc<dyn AuthManager>>,
}

impl AgentRunner {
    pub fn new(model: Arc<dyn LlmModel>) -> Self;
    pub fn with_registry(model: Arc<dyn LlmModel>, registry: Arc<ToolRegistry>) -> Self;
    pub fn with_auth_manager(self, auth: Arc<dyn AuthManager>) -> Self;

    pub fn run(
        &self,
        agent: &Agent,
        thread: Vec<Message>,
    ) -> Pin<Box<dyn Stream<Item = RunEvent> + Send>>;

    pub fn run_with_cancellation(
        &self,
        agent: &Agent,
        thread: Vec<Message>,
        cancel: tokio_util::sync::CancellationToken,
    ) -> Pin<Box<dyn Stream<Item = RunEvent> + Send>>;
}
```

Owns the LLM model, a shared reference to a `ToolRegistry`, and an optional `AuthManager`. The runner is `Clone` (internals are `Arc`) so it can be shared across tasks. The same runner can execute multiple agents; the same agent can be run by different runners backed by different models.

`run` is the simple entry point. It takes the agent by reference and the message thread by value, and returns a `Stream<Item = RunEvent>`. Internally, `run` clones the agent and `self` before spawning the agentic loop on a background tokio task — the spawned generator must be `'static`, so it cannot capture the `&Agent` or `&self` references directly. Events are delivered through an internal mpsc channel as they happen. The stream ends after the model produces a turn with no tool calls, or after a terminal `AgentEvent::Error` / `AgentEvent::Cancelled`.

`run_with_cancellation` is the same as `run` but also fires when the supplied `cancel` token fires. The runner derives a child token from `cancel` and binds the drop-on-stream-drop guard to that child, so dropping the stream cancels the run without cancelling the caller's token (which may be shared with siblings). `run` is sugar for `run_with_cancellation(agent, thread, CancellationToken::new())`.

The caller is responsible for maintaining conversation history across turns: each `run` call starts a fresh loop with the supplied `thread`. For multi-turn dialogue, append the user input to the thread before calling `run`, drive the stream to completion, then append a single `Message::assistant(reply)` constructed from the accumulated `TextDelta` chunks. See `examples/multi_turn.rs` for a complete REPL.

**Agentic loop semantics:**

1. Build a `ModelRequest` from the current thread, agent instructions, output schema, and registry definitions.
2. Drive `model.generate_stream(request)` — forward `ThinkingDelta`/`TextDelta` events; collect any tool calls.
3. If no tool calls were issued, the loop ends.
4. Append the tool calls as a single assistant turn (`Message::tool_calls`).
5. For each tool call, run a future that: looks up the tool in the registry; if missing, returns `Unknown` with no events. Otherwise emits `ToolCallStarted`, then calls `Tool::propose` to resolve the call (a propose error short-circuits to `ToolCallFinished { Err }` without consulting auth). It then consults the `AuthManager` (if any), passing the resulting proposal; on denial, emits `ToolCallFinished { Denied }`. On allow, calls `Tool::apply` with the approved proposal (or runs the sub-agent), then emits `ToolCallFinished` with the result.
6. All tool futures run **concurrently** via `futures_util::future::join_all`. `join_all` preserves input order in its return value, so tool-result messages are appended in the same order the model issued them — even though events may interleave.
7. Repeat from step 1.

**Cancellation checkpoints.** The runner races against the cancel token at these points: (a) the top of the loop, before building the next `ModelRequest`; (b) `model_stream.next()` (a `tokio::select!` against `cancel.cancelled()` — winning the cancel arm drops the model stream, aborting the in-flight reqwest); (c) `Tool::propose`; (d) `auth.authorize` when an `AuthManager` is configured; and (e) `Tool::apply`. On cancellation the runner emits a single terminal `AgentEvent::Cancelled` and exits; no `ToolCallFinished` is emitted for in-flight tools (a trailing `ToolCallStarted` may be observed without its matching `Finished`). Both `Tool::propose` and `Tool::apply` receive a clone of the run's cancel token; cooperating tools select on it to abort cleanly. Tools that ignore the token do not block the runner from terminating — the runner drops the future on cancel — but their side effects may continue in the background until they finish on their own.

### `AgentTool` (`src/tools/agent_tool.rs`)

`AgentTool` wraps an `AgentRunner` + `Agent` pair so any agent can delegate to a child agent through the tool-call mechanism.

```rust
pub struct AgentTool {
    definition: ToolDefinition,   // name, description, parameters exposed to the parent model
    agent: Agent,
    runner: AgentRunner,
}

impl AgentTool {
    pub fn new(definition: ToolDefinition, agent: Agent, runner: AgentRunner) -> Self;
    pub fn definition(&self) -> &ToolDefinition;
    pub fn name(&self) -> &str;
    pub async fn call(
        &self,
        tx: RunEmitter,
        args: serde_json::Value,
        cancel: tokio_util::sync::CancellationToken,
    ) -> Result<serde_json::Value, Error>;
}
```

Registered via `ToolRegistry::register_agent`. `AgentTool` lives in `src/tools/agent_tool/mod.rs`.

**How `apply` works:**

1. Serializes the proposal to a JSON string and passes it as the user message of a fresh run on the inner `AgentRunner`. `AgentTool` keeps the default `propose`, so the proposal is the model's raw args.
2. Drives the child's event stream internally: every event is forwarded through the parent's `RunEmitter` (a child emitter is created so events carry the child's `run_id` and the parent's `parent` link), and `TextDelta` chunks are accumulated into a string.
3. Returns the accumulated text wrapped as `{"output": "..."}` — this JSON value becomes the tool result the parent model sees on its next turn.

The `cancel` token is forwarded to the child via `run_with_cancellation`, so cancelling the parent (either by dropping its stream or by firing its external token) cancels every nested agent in the tree.

**Design rationale:**

- `AgentTool` owns its `AgentRunner` (not a shared reference). Each sub-agent maintains its own model binding. Multiple concurrent `apply` invocations are safe because `AgentRunner::run` takes `&self`.
- The caller supplies the `ToolDefinition` explicitly: the `name` is what the parent model uses to invoke the sub-agent, the `description` guides the parent model's routing decision, and `parameters` describes what args the parent model should pass.
- `AgentTool` lives in its own module (`src/tools/agent_tool/`) to avoid a circular dependency: `tool.rs` must not import the runner, and the runner must not import `agent_tool` directly (it imports it via `crate::tools`).

### `AuthManager` (`src/auth.rs`)

```rust
#[async_trait]
pub trait AuthManager: Send + Sync {
    fn requires_authorization(&self, name: &str, args: &Value) -> bool { true }
    async fn authorize(&self, id: &str, name: &str, args: &Value, proposal: &Value) -> bool;
}
```

Optional hook on `AgentRunner` (set via `with_auth_manager`). With no manager set, no authorization is performed and every tool call runs.

The trait has two methods so the cheap filter and the async decision can live apart:

- `requires_authorization` — sync, must be cheap. The runner calls it first; if it returns `false`, `authorize` is skipped entirely. No I/O, no locks, no awaits.
- `authorize` — async; may block on user input, RPC, dialogs, etc. Returns `true` to allow, `false` to deny. Denial is binary — the runner reports it via `ToolCallResult::Denied` with no accompanying reason. (This matches a user-facing accept/decline approval prompt.) `id` is the tool call's identifier — the same id the runner later reports on `ToolCallStarted`/`ToolCallFinished` — so an out-of-process approver (editor permission request, GUI dialog, remote service) can correlate the prompt with the call. `args` is the model's raw request; `proposal` is what `Tool::propose` resolved it into — the concrete action that will run if approved (an edit tool's proposal carries the path and old/new contents, enough to render a diff). Rendering it is the manager's job: a tool-aware manager matches on `name`, a generic one can display the JSON.

`authorize` may be called concurrently when the model returns multiple tool calls in one turn. Implementations sharing UI resources (stdin, a modal dialog) must serialize internally — typically with a `tokio::sync::Mutex`. The lock belongs in `authorize`, not in `requires_authorization`.

### `Error` (`src/error.rs`)

```rust
#[derive(Clone, Debug, thiserror::Error)]
pub enum Error {
    #[error("LLM provider error: {0}")]
    Provider(String),
    #[error("Agent error: {0}")]
    Agent(String),
}
```

Provider adapters wrap transport- and API-level failures into `Error::Provider`; agent-side failures (serialization, lock poisoning, user-defined tool errors) use `Error::Agent`.

## Provider Adapters

### `GeminiModel` (`src/models/gemini.rs`)

- Wraps the `geologia` crate (`GeminiClient`).
- Translates `ModelRequest` → `GenerateContentRequest`, mapping `Role::User → Role::User` and `Role::Assistant → Role::Model`.
- System instructions become `system_instruction` on the Gemini request.
- Optional `GenerationConfig` (temperature, max_output_tokens, top_p, top_k, stop_sequences, thinking_config) configurable via `GeminiModel::builder(…)`.
- Structured output: when `ModelRequest::output_schema` is set, a `GenerationConfig` with `response_mime_type("application/json")` and the normalised schema is applied, overriding any model-level config (with `thinking_config` carried over). Schema normalisation (stripping `$schema`/`$defs`, inlining `$ref`) is performed internally.
- Response parts are split by `thought` flag: parts where `thought == Some(true)` are concatenated into `ModelResponse::thinking`; remaining text parts form `ModelResponse::text`. This surfaces reasoning tokens as `AgentEvent::ThinkingDelta` via the runner.
- `provider_metadata` carries Gemini's `thought_signature`, which the adapter echoes back on both the replayed `FunctionCall` parts and the matching `FunctionResponse` parts.
- Token usage: `usage_metadata` is mapped via `From<&UsageMetadata> for TokenUsage` — `prompt_token_count` → `input_tokens`, `candidates_token_count` → `output_tokens`, `cached_content_token_count` → `cached_input_tokens`, `thoughts_token_count` → `thinking_tokens`, `tool_use_prompt_token_count` → `tool_use_prompt_tokens`. Per-modality breakdowns (`*_tokens_details`) and `service_tier` are not propagated.
- Implements `generate_stream` natively on top of `GeminiClient::stream_generate_content`. Each streamed `Candidate` part is converted to a chunk in order: thought `Text` parts become `Thinking`, regular `Text` parts become `TextDelta`, and `FunctionCall` parts become `ToolCall` (with `thought_signature` preserved in `provider_metadata`). The final `UsageMetadata` reported by the provider is emitted as a trailing `Usage` chunk.

### `OllamaModel` (`src/models/ollama.rs`)

- Wraps the `ollama-rs` crate (`OllamaClient`).
- System prompt becomes a synthetic `OllamaMessage::system(…)` prepended to the message list.
- Optional `Options` (temperature, seed, top_k, top_p, num_ctx, num_predict, stop) and extended-thinking config (`think`, accepting a boolean toggle or a `ThinkLevel`) configurable via `OllamaModel::builder(…)`.
- Structured output: when `ModelRequest::output_schema` is set, the schema is passed to the Ollama `format` field (requires Ollama ≥ 0.5 and a model that supports structured output).
- Implements `generate_stream` natively: emits `TextDelta` chunks as they arrive (no-tools path); emits `ToolCall` chunks from the single-shot response when tools are present (Ollama requires `stream(false)` for tool calls).
- Ollama has no call ID; the function name is currently reused as the `ToolCall::id` (sufficient because no part of the codebase keys on the id for Ollama).
- Token usage: `prompt_eval_count` → `input_tokens`, `eval_count` → `output_tokens` (saturating `u64` → `u32` cast). The Ollama API reports these on the final chunk (`done: true`) only — non-final chunks carry no usage. When neither field is populated the adapter emits `token_usage: None` rather than an all-`None` [`TokenUsage`]. Ollama does not report cache, thinking, or tool-use prompt tokens.

## Cargo Features

Provider adapters are opt-in via Cargo features. The core types (`LlmModel`, `Agent`, `AgentRunner`, `Tool`, `Error`, `AuthManager`, etc.) are always available regardless of which features are enabled.

| Feature    | Enables                          |
|------------|----------------------------------|
| `gemini`   | `GeminiModel` (`geologia`)       |
| `ollama`   | `OllamaModel` (`ollama-rs`)      |
| `full`     | All providers (`gemini`, `ollama`) |

The `default` feature set is empty — no provider is compiled unless explicitly requested.

**Usage in `Cargo.toml`:**

```toml
# Only Gemini
agent-rig = { version = "...", features = ["gemini"] }

# All providers
agent-rig = { version = "...", features = ["full"] }
```

New provider adapters must follow the same pattern: add an `optional` dependency and a feature flag; gate the module and its re-exports with `#[cfg(feature = "...")]`.

## Module Layout

```
src/
  lib.rs              — crate root, public re-exports
  error.rs            — Error enum
  model.rs            — LlmModel trait, ModelStreamChunk, Message, MessageContent,
                        ModelRequest, ModelResponse, TokenUsage, ToolCall, Role
  agent.rs            — Agent, AgentBuilder
  auth.rs             — AuthManager trait
  runner/
    mod.rs            — AgentRunner, RunEmitter, agentic loop
    events.rs         — AgentEvent, RunEvent, ToolCallResult
    tests.rs          — runner unit tests (with scripted LlmModel)
  tools/
    mod.rs            — re-exports Tool, ToolDefinition, ToolRegistry, AgentTool
    tool.rs           — Tool trait, ToolDefinition
    registry.rs       — ToolRegistry, ToolRegistryEntry
    agent_tool/
      mod.rs          — AgentTool (wraps AgentRunner + Agent as a tool)
      tests.rs        — AgentTool unit tests
  models/
    mod.rs            — feature-gated: #[cfg(feature="gemini")] pub mod gemini; etc.
    gemini.rs         — GeminiModel, GeminiModelBuilder  (feature: gemini)
    ollama.rs         — OllamaModel, OllamaModelBuilder  (feature: ollama)
examples/
  simple_agent.rs         — single-turn Gemini example
  tool_calling.rs         — Tool trait + ToolRegistry
  structured_output.rs    — output_schema + schemars
  streaming_agent.rs      — thinking deltas + tool calls
  streaming_structured.rs — streaming with structured output
  multi_turn.rs           — manual history multi-turn REPL
  parallel_tool_calls.rs  — concurrent tool execution
  agent_as_tool.rs        — AgentTool composition (parent/child runs distinguished by run_id)
  long_term_memory.rs     — memory via tools
  mpsc_auth_flow.rs       — AuthManager CLI prompt
  mpsc_runner.rs          — runner basics
  cancellation.rs         — drop-the-stream, external CancellationToken, deadline
tests/
  integration_gemini.rs   — live Gemini integration tests
  integration_ollama.rs   — live Ollama integration tests
docs/
  PRD.md                  — product requirements
  SPEC.md                 — this document
  PLAN-*.md               — historical implementation plans (kept for design context)
skills/
  agent-rig.md            — Claude skill: how to write code against this crate
```

## Testing Strategy

- **Unit tests** live in `#[cfg(test)]` modules inside each source file (and in dedicated `tests.rs` siblings under `src/runner/` and `src/tools/agent_tool/`). Provider calls are replaced with stub/echo `LlmModel` implementations in those test modules.
- **Integration tests** in `tests/` hit real provider endpoints. They require environment variables (`GEMINI_API_KEY`, running Ollama server) and short-circuit to a no-op return when the required environment is unavailable. Each test target is gated behind the matching Cargo feature.
- All public items must have rustdoc comments; examples in doc comments are compiled as `no_run` doctests.

## Roadmap

The following capabilities are planned but not yet implemented:

1. **Additional providers.** OpenAI-compatible endpoints and Anthropic Claude are natural next targets given the trait abstraction.
2. **Automatic conversation history.** All multi-turn flows today require the caller to maintain `Vec<Message>` and pass it on each `run`. A higher-level `Conversation` wrapper that records the thread automatically (with explicit access for trimming/compression) is sketched in `docs/PLAN-conversation.md` but not yet implemented.
