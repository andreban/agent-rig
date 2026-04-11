# Technical Specification — rust-agent-kit

## Architecture

```
┌──────────────┐     runs      ┌──────────────┐
│  AgentRunner │──────────────▶│    Agent     │
└──────────────┘               └──────┬───────┘
                                      │ holds
                               ┌──────▼───────┐
                               │  LlmModel    │  (trait object)
                               └──────┬───────┘
                         implements   │
              ┌──────────────┬────────┘
              ▼              ▼
       GeminiModel     OllamaModel    (more providers …)
```

The library is a single crate (`rust-agent-kit`). All provider types live in `src/models/`. Agent logic lives in `src/agent.rs` and `src/runner.rs`. The `LlmModel` trait in `src/model.rs` is the extension point.

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
}

pub struct ModelResponse {
    pub text: String,
}
```

`Message` carries a `Role` (`User` | `Assistant`) and a `content: String`. This is the canonical representation that provider adapters translate to and from their SDK types.

`output_schema`, when set, instructs the provider adapter to constrain the response to the supplied JSON Schema. Providers that do not support structured output ignore the field silently.

### `Agent` (`src/agent.rs`)

```rust
pub struct Agent {
    name: String,
    instructions: String,
    model: Box<dyn LlmModel>,
    output_schema: Option<serde_json::Value>,
}
```

Constructed via `Agent::builder()`. Holds the model, the system instructions used on every run, and an optional JSON Schema for structured output. Does not hold conversation state — the runner owns the request being built.

`output_schema` is set via `AgentBuilder::output_schema(schema)`. The runner copies it into every `ModelRequest`, and each provider adapter applies it using provider-specific mechanisms.

### `AgentRunner` (`src/runner.rs`)

```rust
pub struct AgentRunner;

impl AgentRunner {
    pub async fn run(&self, agent: &Agent, input: &str) -> Result<AgentResult, Error>;
}
```

Translates a user input string into a `ModelRequest` — including the agent's `output_schema` if set — calls `agent.model.generate`, and returns `AgentResult { output: String }`. Will be extended to handle the function-calling loop (see Roadmap).

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
  lib.rs          — crate root, public re-exports
  error.rs        — Error enum
  model.rs        — LlmModel trait, Message, ModelRequest, ModelResponse, Role
  agent.rs        — Agent, AgentBuilder
  runner.rs       — AgentRunner, AgentResult
  models/
    mod.rs        — pub mod gemini; pub mod ollama;
    gemini.rs     — GeminiModel, GeminiModelBuilder
    ollama.rs     — OllamaModel, OllamaModelBuilder
examples/
  simple_agent.rs — runnable Gemini example
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

1. **Tool / function calling.** Extend `ModelRequest` to carry `FunctionDeclaration` definitions and `ModelResponse` to return `FunctionCall` parts. `AgentRunner` will loop: detect function calls → execute registered tools → push `FunctionResponse` → repeat until a text response is produced.
2. **Multi-turn conversations.** Allow callers to pass existing conversation history into `AgentRunner::run` for stateful dialogue.
3. **Streaming responses.** Expose a streaming variant of `AgentRunner::run` that yields tokens incrementally.
4. **Additional providers.** OpenAI-compatible endpoints and Anthropic Claude are natural next targets given the trait abstraction.
