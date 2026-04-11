# Technical Specification — rust-agent-kit

## Architecture

```
┌──────────────┐     runs      ┌──────────────┐
│  AgentRunner │──────────────▶│    Agent     │
│              │               └──────────────┘
│  holds       │               (pure blueprint)
│              │
└──────┬───────┘
       │ holds
┌──────▼───────┐
│  LlmModel    │  (trait object)
└──────┬───────┘
       │ implements
┌──────┴──────────────┐
▼                     ▼
GeminiModel     OllamaModel    (more providers …)
```

The library is a single crate (`rust-agent-kit`). All provider types live in `src/models/`. Agent logic lives in `src/agent.rs` and `src/runner.rs`. The `LlmModel` trait in `src/model.rs` is the extension point.

`Agent` is a pure data blueprint (name, instructions, optional output schema) with no model reference. `AgentRunner` owns the `Box<dyn LlmModel>` and is the execution engine. The same runner can execute multiple agents; the same agent can be run by different runners backed by different models.

## Core Types

### `LlmModel` (`src/model.rs`)

```rust
#[async_trait]
pub trait LlmModel: Send + Sync {
    async fn generate(&self, request: ModelRequest) -> Result<ModelResponse, Error>;
}
```

All provider adapters implement this trait. The runner holds a `Box<dyn LlmModel>` so providers are interchangeable at runtime without generics on the public API.

### `ModelRequest` / `ModelResponse`

```rust
pub struct ModelRequest {
    pub messages: Vec<Message>,                    // conversation history
    pub system: Option<String>,                    // system prompt
    pub output_schema: Option<serde_json::Value>,  // JSON Schema for structured output
    pub tools: Vec<ToolDefinition>,                // tool definitions for this turn
}

pub struct ToolCall {
    pub id: String,                    // provider-supplied call ID (echoed in response)
    pub name: String,
    pub args: serde_json::Value,
}

pub struct ModelResponse {
    pub text: Option<String>,          // None when the model issued tool calls
    pub tool_calls: Vec<ToolCall>,     // empty on a final text response
}
```

`Message` carries a `Role` (`User` | `Assistant`) and a `content: MessageContent`. `MessageContent` is an enum with three variants:
- `Text(String)` — a plain text turn
- `ToolCalls(Vec<ToolCall>)` — all tool calls from one model turn (one assistant message)
- `ToolResult { id, name, result }` — the result of one tool execution (one user message)

`output_schema`, when set, instructs the provider adapter to constrain the response to the supplied JSON Schema. Providers that do not support structured output ignore the field silently.

`text` and `tool_calls` on `ModelResponse` are mutually exclusive per turn.

### `Agent` (`src/agent.rs`)

```rust
#[derive(Serialize, Deserialize)]
pub struct Agent {
    name: String,
    instructions: String,
    output_schema: Option<serde_json::Value>,
    tool_names: Vec<String>,
}
```

Constructed via `Agent::builder()`. A pure data blueprint: holds the system instructions used on every run, an optional JSON Schema for structured output, and the names of tools the agent is permitted to use. Carries no model or runtime state. Derives `Serialize`/`Deserialize` so agent configurations can be saved to and loaded from files (JSON, YAML, etc.). Tool definitions (description, parameters schema) are not serialized with the agent — they are owned by each `Tool` implementation in the `AgentRunner` registry and resolved at runtime.

`output_schema` is set via `AgentBuilder::output_schema(schema)`. The runner copies it into every `ModelRequest`, and each provider adapter applies it using provider-specific mechanisms.

### `ToolDefinition` / `Tool` / `ToolRegistry` (`src/tool.rs`)

```rust
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,   // JSON Schema
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn definition(&self) -> ToolDefinition;
    async fn call(&self, args: serde_json::Value) -> Result<serde_json::Value, Error>;
}

pub struct ToolRegistry { ... }

impl ToolRegistry {
    pub fn new() -> Self;
    pub fn register(self, tool: Box<dyn Tool>) -> Self;   // builder-style
}
```

`ToolDefinition` is the runtime-only contract between agent and model. It is never stored in `Agent` — it lives in the `ToolRegistry` alongside its `Tool` implementation and is resolved at run time.

`ToolRegistry` is independent of any runner; share it across multiple runners via `Arc<ToolRegistry>`.

### `AgentRunner` (`src/runner.rs`)

```rust
pub struct AgentRunner {
    model: Box<dyn LlmModel>,
    registry: Arc<ToolRegistry>,
}

impl AgentRunner {
    pub fn new(model: Box<dyn LlmModel>) -> Self;   // empty registry
    pub fn with_registry(model: Box<dyn LlmModel>, registry: Arc<ToolRegistry>) -> Self;
    pub async fn run(&self, agent: &Agent, input: &str) -> Result<AgentResult, Error>;
    pub async fn run_typed<T: DeserializeOwned>(&self, agent: &Agent, input: &str) -> Result<T, Error>;
}
```

Owns the LLM model and a shared reference to a `ToolRegistry`. The same runner can execute multiple agents; the same agent can be run by different runners backed by different models.

`run` validates that every tool name in `agent.tool_names()` is registered, then executes the agentic loop: resolve definitions → `model.generate` → execute tool calls → append results → repeat until the model returns a text response. Returns `AgentResult { output: String }`.

`run_typed<T>` is a thin typed wrapper over `run` that deserializes the final text output into `T` via `serde_json`. Deserialization failure returns `Error::Agent`.

### `AgentTool` (`src/agent_tool.rs`)

`AgentTool` wraps an `AgentRunner` + `Agent` pair into a value that implements `Tool`, so any agent can delegate to a child agent as if it were a regular tool.

```rust
pub struct AgentTool {
    definition: ToolDefinition,   // name, description, parameters exposed to the parent model
    agent: Agent,
    runner: AgentRunner,
}

impl AgentTool {
    /// Creates an `AgentTool` from a pre-built `ToolDefinition`, an `Agent`, and an `AgentRunner`.
    pub fn new(definition: ToolDefinition, agent: Agent, runner: AgentRunner) -> Self;
}

#[async_trait]
impl Tool for AgentTool {
    fn definition(&self) -> ToolDefinition { /* returns self.definition.clone() */ }
    async fn call(&self, args: serde_json::Value) -> Result<serde_json::Value, Error>;
}
```

**How `call` works:**

1. Serializes `args` to a JSON string (via `serde_json::to_string`) and passes it as the input to `self.runner.run(&self.agent, &input)`.
2. The sub-agent processes the input through its own agentic loop (which may invoke its own tools).
3. Returns `json!({ "output": result.output })` so the parent model receives a structured result it can read.

**Design rationale:**

- `AgentTool` **owns** its `AgentRunner` (not a shared reference). Each distinct sub-agent tool maintains its own model binding. Multiple concurrent `call` invocations are safe because `AgentRunner::run` takes `&self`.
- The caller supplies the `ToolDefinition` explicitly: the `name` is what the parent model uses to invoke the sub-agent, the `description` guides the parent model's routing decision, and `parameters` describes what args the parent model should pass (e.g., `{ "query": "string" }`).
- `AgentTool` lives in its own module (`src/agent_tool.rs`) to avoid a circular dependency: `tool.rs` must not import `runner.rs`, and `runner.rs` must not import `agent_tool.rs`.

**Usage pattern:**

```rust
// Child agent + its own model
let child_model = GeminiModel::builder(api_key, "gemini-3.1-flash-lite-preview").build();
let child_agent = Agent::builder()
    .name("Summariser")
    .instructions("Summarise the text provided in the 'text' field of the JSON input.")
    .build();
let child_runner = AgentRunner::new(Box::new(child_model));

// Wrap as a tool for the parent
let summarise_tool = AgentTool::new(
    ToolDefinition {
        name: "summarise".to_string(),
        description: "Summarises a long piece of text.".to_string(),
        parameters: json!({
            "type": "object",
            "properties": { "text": { "type": "string" } },
            "required": ["text"]
        }),
    },
    child_agent,
    child_runner,
);

// Register with the parent runner
let registry = Arc::new(ToolRegistry::new().register(Box::new(summarise_tool)));
let parent_runner = AgentRunner::with_registry(Box::new(parent_model), registry);
```

### `Error` (`src/error.rs`)

```rust
pub enum Error {
    Provider(String),   // error from the upstream LLM provider
    Agent(String),      // error during agent execution
}
```

## Provider Adapters

### `GeminiModel` (`src/models/gemini.rs`)

- Wraps the `google-genai` crate (`GeminiClient`).
- Translates `ModelRequest` → `GenerateContentRequest`, mapping `Role::User → Role::User` and `Role::Assistant → Role::Model`.
- System instructions become `system_instruction` on the Gemini request.
- Optional `GenerationConfig` (temperature, max_output_tokens, top_p, top_k, stop_sequences) configurable via `GeminiModel::builder(…)`.
- Structured output: when `ModelRequest::output_schema` is set, a `GenerationConfig` with `response_mime_type("application/json")` and the normalised schema is applied, overriding any model-level config. Schema normalisation (stripping `$schema`/`$defs`, inlining `$ref`) is performed internally.
- Response text extracted from `candidates[0].get_text()`.

### `OllamaModel` (`src/models/ollama.rs`)

- Wraps the `ollama-rs` crate (`OllamaClient`).
- System prompt becomes a synthetic `OllamaMessage::system(…)` prepended to the message list.
- Uses the streaming chat API (`client.chat` returns a stream); chunks are concatenated until `done == true`.
- Optional `Options` (temperature, seed, top_k, top_p, num_ctx, num_predict, stop) configurable via `OllamaModel::builder(…)`.
- Structured output: when `ModelRequest::output_schema` is set, the schema is passed to the Ollama `format` field (requires Ollama ≥ 0.5 and a model that supports structured output).

## Module Layout

```
src/
  lib.rs           — crate root, public re-exports
  error.rs         — Error enum
  model.rs         — LlmModel trait, Message, MessageContent, ModelRequest, ModelResponse, ToolCall, Role
  tool.rs          — ToolDefinition, Tool trait, ToolRegistry
  agent.rs         — Agent, AgentBuilder
  runner.rs        — AgentRunner, AgentResult
  agent_tool.rs    — AgentTool (wraps AgentRunner + Agent as a Tool)
  models/
    mod.rs         — pub mod gemini; pub mod ollama;
    gemini.rs      — GeminiModel, GeminiModelBuilder
    ollama.rs      — OllamaModel, OllamaModelBuilder
examples/
  simple_agent.rs  — runnable Gemini example
tests/
  integration_gemini.rs   — live Gemini integration tests
  integration_ollama.rs   — live Ollama integration tests
```

## Testing Strategy

- **Unit tests** live in `#[cfg(test)]` modules inside each source file. Provider calls are replaced with stub/echo `LlmModel` implementations.
- **Integration tests** in `tests/` hit real provider endpoints. They require environment variables (`GEMINI_API_KEY`, running Ollama server) and are meant to be run explicitly, not in CI by default.
- All public items must have rustdoc comments; examples in doc comments are compiled as `no_run` doctests.

## Roadmap

The following capabilities are planned but not yet implemented:

1. **Agent as a tool (`AgentTool`).** Wrap an `AgentRunner` + `Agent` pair as a `Tool` so a parent agent can delegate to a child agent. See the `AgentTool` section above for the full design.
2. **Multi-turn conversations.** Allow callers to pass existing conversation history into `AgentRunner::run` for stateful dialogue.
3. **Streaming responses.** Expose a streaming variant of `AgentRunner::run` that yields tokens incrementally.
4. **Additional providers.** OpenAI-compatible endpoints and Anthropic Claude are natural next targets given the trait abstraction.
