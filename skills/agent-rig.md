---
name: agent-rig
description: >
  Guide and write code using the `agent-rig` library — a provider-agnostic AI agent toolkit for Rust.
  Use this skill whenever the user is writing, reading, debugging, or extending code that involves
  `agent_rig`, `AgentRunner`, `Agent`, `LlmModel`, `ToolRegistry`, `AgentTool`, `GeminiModel`,
  `OllamaModel`, or any type from this crate. Also trigger when the user asks how to add a new LLM
  provider, implement a custom tool, wire up multi-turn conversations, stream agent output, or use
  structured output in the context of this project.
---

You are helping the user write code using the `agent_rig` crate — a provider-agnostic toolkit
for building AI agents in Rust. Apply the patterns and constraints below precisely.

---

## Dependency Setup

The library is **not on crates.io**. It must be added from the git repository. Provider adapters are
opt-in features — core types are always compiled regardless.

```toml
[dependencies]
agent-rig = { git = "https://github.com/andreban/agent-rig.git", features = ["gemini"] }

# All providers:
agent-rig = { git = "https://github.com/andreban/agent-rig.git", features = ["full"] }

# Common companions:
async-trait = "0.1"
serde_json = "1"
futures-util = "0.3"
tokio = { version = "1", features = ["full"] }
dotenvy = "0.15"
```

| Feature   | Enables                    |
|-----------|----------------------------|
| `gemini`  | `GeminiModel` (Google)     |
| `ollama`  | `OllamaModel` (local)      |
| `full`    | All providers              |

---

## Core Types

| Type | Description |
|------|-------------|
| `Agent` / `AgentBuilder` | Pure data blueprint: name, instructions, optional output schema, allowed tool names. Implements `Serialize`/`Deserialize` so configs can be stored as JSON/YAML. |
| `AgentRunner` | Execution engine; owns `Box<dyn LlmModel>` and a shared `ToolRegistry`. Stateless — callers own conversation history. |
| `RunBuilder` | Fluent per-run builder produced by `runner.run_builder(&agent)`. Accepts optional conversation history. |
| `AgentResult` | Returned by `run` / `run_typed`. Has a single field: `output: String`. |
| `LlmModel` | Async trait all provider adapters implement. The extension point for new providers. |
| `Tool` / `ToolDefinition` | Async trait for callable tools; parameters are a JSON Schema `Value`. |
| `ToolRegistry` | Thread-safe registry of `Tool` implementations, shared via `Arc`. |
| `AgentTool` | Wraps an `AgentRunner` + `Agent` as a `Tool` for agent composition. |
| `AgentEvent` | Stream event: `TextDelta`, `Thinking`, `ToolCallStarted`, `ToolCallCompleted`. |
| `Error` | `Provider(String)` or `Agent(String)`. |

**Key design rule**: `Agent` carries no model reference — the same blueprint can run on any
`AgentRunner`. The runner is stateless; callers own and extend conversation history between turns.

---

## Single-Turn (Basic)

```rust
use agent_rig::{Agent, AgentRunner};
use agent_rig::models::gemini::GeminiModel;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = dotenvy::dotenv();
    let api_key = std::env::var("GEMINI_API_KEY")?;

    let model = GeminiModel::builder(api_key, "gemini-3.1-flash-lite-preview")
        .temperature(0.7)
        .build();

    let agent = Agent::builder()
        .name("Assistant")
        .instructions("You are a helpful assistant.")
        .build();

    let runner = AgentRunner::new(Box::new(model));
    let result = runner.run(&agent, "What is the capital of France?").await?;
    println!("{}", result.output);
    Ok(())
}
```

`run` drives the full agentic loop (including any tool calls) and returns the final text in
`result.output`.

---

## Multi-Turn Conversations

The runner is stateless. The caller owns and extends the history between turns. Use
`run_builder(&agent).history(history)` for each turn.

```rust
use agent_rig::{Agent, AgentRunner};
use agent_rig::model::Message;
use agent_rig::models::gemini::GeminiModel;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = dotenvy::dotenv();
    let api_key = std::env::var("GEMINI_API_KEY")?;

    let model = GeminiModel::builder(api_key, "gemini-3.1-flash-lite-preview").build();
    let agent = Agent::builder()
        .name("Assistant")
        .instructions("You are a helpful assistant.")
        .build();
    let runner = AgentRunner::new(Box::new(model));

    // First turn — no history.
    let first = runner.run(&agent, "My name is Alice.").await?;
    println!("Turn 1: {}", first.output);

    // Build history from the completed turn.
    let mut history = vec![
        Message::user("My name is Alice."),
        Message::assistant(&first.output),
    ];

    // Second turn — pass accumulated history.
    let second = runner
        .run_builder(&agent)
        .history(history.clone())
        .run("What is my name?")
        .await?;
    println!("Turn 2: {}", second.output); // "Your name is Alice."

    // Extend for the next turn.
    history.push(Message::user("What is my name?"));
    history.push(Message::assistant(&second.output));

    Ok(())
}
```

---

## Tool Calling

Implement `Tool`, register in `ToolRegistry`, declare names on the agent via `.tool("name")`, and
pass the registry to the runner. The agentic loop runs automatically until the model produces a
final text response.

```rust
use std::sync::Arc;
use async_trait::async_trait;
use agent_rig::{Agent, AgentRunner};
use agent_rig::error::Error;
use agent_rig::models::gemini::GeminiModel;
use agent_rig::tool::{Tool, ToolDefinition, ToolRegistry};
use serde_json::{Value, json};

struct GetWeatherTool;

#[async_trait]
impl Tool for GetWeatherTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "get_weather".to_string(),
            description: "Returns the current temperature in Celsius for a city.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "city": { "type": "string", "description": "The city name." }
                },
                "required": ["city"]
            }),
        }
    }

    async fn call(&self, args: Value) -> Result<Value, Error> {
        let city = args["city"].as_str().unwrap_or("unknown");
        Ok(json!({ "city": city, "celsius": 22.0 }))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = dotenvy::dotenv();
    let api_key = std::env::var("GEMINI_API_KEY")?;

    let registry = Arc::new(ToolRegistry::new().register(Box::new(GetWeatherTool)));

    let agent = Agent::builder()
        .name("Weather Bot")
        .instructions("Answer weather questions using the available tools.")
        .tool("get_weather")  // must match ToolDefinition::name exactly
        .build();

    let model = GeminiModel::builder(api_key, "gemini-3.1-flash-lite-preview").build();
    let runner = AgentRunner::with_registry(Box::new(model), registry);

    let result = runner.run(&agent, "What is the temperature in Tokyo?").await?;
    println!("{}", result.output);
    Ok(())
}
```

**Rules:**
- `.tool("name")` on the agent must match `ToolDefinition::name` exactly, or the runner returns
  `Error::Agent` at runtime.
- Multiple tool calls in a single model turn are executed **concurrently** by the runner.
- `ToolRegistry` is `Arc`-wrapped so it can be shared across multiple runners.

---

## Streaming

`run_stream` returns `impl Stream<Item = Result<AgentEvent, Error>>`. Pin with
`futures_util::pin_mut!` before driving it.

```rust
use futures_util::StreamExt;
use agent_rig::{Agent, AgentEvent, AgentRunner};
use agent_rig::models::gemini::GeminiModel;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = dotenvy::dotenv();
    let api_key = std::env::var("GEMINI_API_KEY")?;

    let model = GeminiModel::builder(api_key, "gemini-3.1-flash-lite-preview").build();
    let agent = Agent::builder()
        .name("Assistant")
        .instructions("You are a helpful assistant.")
        .build();
    let runner = AgentRunner::new(Box::new(model));

    let stream = runner.run_stream(&agent, "Explain Rust ownership in three points.");
    futures_util::pin_mut!(stream);

    while let Some(event) = stream.next().await {
        match event? {
            AgentEvent::Thinking(token) => print!("\x1b[2m{token}\x1b[0m"),
            AgentEvent::TextDelta(chunk) => print!("{chunk}"),
            AgentEvent::ToolCallStarted { name, args } => {
                println!("[calling {name}({args})]");
            }
            AgentEvent::ToolCallCompleted { name, result } => {
                println!("[{name} → {result}]");
            }
        }
    }
    println!();
    Ok(())
}
```

**Notes:**
- `Thinking` tokens are only emitted when the model has extended thinking enabled **and** the
  provider has a native `generate_stream` implementation. `OllamaModel` has native streaming;
  `GeminiModel` currently emits output as a single `TextDelta`.
- For multi-turn streaming, use `runner.run_builder(&agent).history(history).run_stream(input)`.

---

## Structured Output

Set `output_schema` on the agent to constrain the model's response to a JSON Schema. Use
`run_typed<T>` to deserialize directly into a Rust type. The
[`schemars`](https://crates.io/crates/schemars) crate can generate the schema from a struct.

```toml
schemars = "0.8"
serde = { version = "1", features = ["derive"] }
```

```rust
use agent_rig::{Agent, AgentRunner};
use agent_rig::models::gemini::GeminiModel;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ResearchPlan {
    title: String,
    tasks: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = dotenvy::dotenv();
    let api_key = std::env::var("GEMINI_API_KEY")?;

    let model = GeminiModel::builder(api_key, "gemini-3.1-flash-lite-preview")
        .temperature(0.4)
        .build();

    let agent = Agent::builder()
        .name("Planner")
        .instructions("Produce a structured research plan.")
        .output_schema(schemars::schema_for!(ResearchPlan))
        .build();

    let plan: ResearchPlan = AgentRunner::new(Box::new(model))
        .run_typed(&agent, "AI agents in Rust")
        .await?;

    println!("{}: {:?}", plan.title, plan.tasks);
    Ok(())
}
```

**Notes:**
- `run_typed<T>` deserializes `result.output` via `serde_json`; failure returns `Error::Agent`.
- Providers that don't support structured output (e.g., older Ollama models) silently ignore
  `output_schema`.
- `output_schema` and tool calling can be combined on the same agent; the schema is applied to the
  *final* text response after all tool calls are resolved.

---

## Agent Composition (`AgentTool`)

Wrap an `AgentRunner` + `Agent` as a `Tool` using `AgentTool`. Register it with a parent runner.
The parent model delegates work to the child agent as if it were a regular tool call.

```rust
use std::sync::Arc;
use agent_rig::{Agent, AgentRunner, AgentTool};
use agent_rig::models::gemini::GeminiModel;
use agent_rig::tool::{ToolDefinition, ToolRegistry};
use serde_json::json;

const MODEL: &str = "gemini-3.1-flash-lite-preview";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = dotenvy::dotenv();
    let api_key = std::env::var("GEMINI_API_KEY")?;

    // --- Child agent ---
    let child_model = GeminiModel::builder(&api_key, MODEL).build();
    let child_agent = Agent::builder()
        .name("Summariser")
        .instructions(
            "You receive a JSON object with a `text` field. \
             Summarise the text in two sentences or fewer.",
        )
        .build();
    let child_runner = AgentRunner::new(Box::new(child_model));

    let summarise_tool = AgentTool::new(
        ToolDefinition {
            name: "summarise".to_string(),
            description: "Summarises a long piece of text. Pass the text in the `text` field."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "text": { "type": "string", "description": "The text to summarise." }
                },
                "required": ["text"]
            }),
        },
        child_agent,
        child_runner,
    );

    // --- Parent runner ---
    let registry = Arc::new(ToolRegistry::new().register(Box::new(summarise_tool)));
    let parent_model = GeminiModel::builder(&api_key, MODEL).build();
    let parent_runner = AgentRunner::with_registry(Box::new(parent_model), registry);

    let parent_agent = Agent::builder()
        .name("Orchestrator")
        .instructions("Use the `summarise` tool when asked to summarise text.")
        .tool("summarise")
        .build();

    let result = parent_runner
        .run(&parent_agent, "Summarise: Rust focuses on safety, speed, and concurrency…")
        .await?;
    println!("{}", result.output);
    Ok(())
}
```

**Notes:**
- `AgentTool` **owns** its `AgentRunner` (not shared). Each child tool has its own model binding.
- `AgentTool::call` serialises the parent's `args` JSON to a string and passes it as the child's
  input. The child's output is returned as `{ "output": "..." }`.
- Child agents can have their own tools — nesting is unlimited.

---

## Provider Configuration

### Google Gemini (`feature = "gemini"`)

Requires `GEMINI_API_KEY` environment variable.

```rust
use agent_rig::models::gemini::GeminiModel;
use geologia::prelude::{ThinkingConfig, ThinkingLevel};

// Minimal
let model = GeminiModel::builder(api_key, "gemini-3.1-flash-lite-preview").build();

// Full configuration
let model = GeminiModel::builder(api_key, "gemini-3.1-flash-lite-preview")
    .temperature(0.7)
    .max_output_tokens(2048)
    .top_p(0.9)
    .top_k(40)
    .thinking_config(ThinkingConfig {
        include_thoughts: true,
        thinking_level: Some(ThinkingLevel::High),
        ..Default::default()
    })
    .build();
```

### Ollama (`feature = "ollama"`)

Requires a running Ollama server (default: `http://localhost:11434`).

```rust
use agent_rig::models::ollama::OllamaModel;

let model = OllamaModel::builder("llama3.2")
    .temperature(0.8)
    .num_ctx(4096)
    .top_p(0.9)
    .seed(42)
    .build();
```

Structured output requires Ollama ≥ 0.5 and a model that supports it.

---

## Implementing a Custom Provider

Implement `LlmModel` to add any provider. Only `generate` is required; `generate_stream` has a
default that calls `generate` and emits the result as a single `TextDelta`.

```rust
use async_trait::async_trait;
use agent_rig::error::Error;
use agent_rig::model::{
    LlmModel, Message, MessageContent, ModelRequest, ModelResponse, Role, ToolCall,
};

struct MyModel { api_key: String }

#[async_trait]
impl LlmModel for MyModel {
    async fn generate(&self, request: ModelRequest) -> Result<ModelResponse, Error> {
        // 1. Translate request.messages → your provider's format.
        //    Role::User → user, Role::Assistant → assistant/model
        //    MessageContent::Text(s)                   → plain text turn
        //    MessageContent::ToolCalls(calls)           → assistant tool-call turn
        //    MessageContent::ToolResult { id, name, result } → tool result turn (echo `id`)

        // 2. Map request.system → system prompt (if supported).
        // 3. Map request.tools → provider function declarations (if non-empty).
        // 4. Map request.output_schema → structured output constraint (if supported).
        // 5. Call your API and await the response.
        // 6. Return either tool_calls (text = None) or text (tool_calls = vec![]).

        Ok(ModelResponse {
            text: Some("Hello from MyModel!".to_string()),
            tool_calls: vec![],
            thinking: None,
        })
    }
}
```

**`ModelRequest` fields:**

| Field | Type | Notes |
|-------|------|-------|
| `messages` | `Vec<Message>` | Full conversation history including the new user turn |
| `system` | `Option<String>` | System prompt from `Agent::instructions` |
| `tools` | `Vec<ToolDefinition>` | Active tool definitions filtered to the agent's allowed names |
| `output_schema` | `Option<serde_json::Value>` | JSON Schema; ignore if your provider does not support it |

**Returning tool calls:**

```rust
Ok(ModelResponse {
    text: None,
    tool_calls: vec![ToolCall {
        id: "call_abc123".to_string(),  // echo this id in the ToolResult
        name: "get_weather".to_string(),
        args: serde_json::json!({ "city": "Tokyo" }),
    }],
    thinking: None,
})
```

**Adding a feature flag** (for library contributors):

```toml
# Cargo.toml
[features]
myprovider = ["dep:my-provider-crate"]

[dependencies]
my-provider-crate = { version = "...", optional = true }
```

```rust
// src/models/mod.rs
#[cfg(feature = "myprovider")]
pub mod myprovider;
```

Add `"myprovider"` to the `full` feature alias in `Cargo.toml`.

---

## Common Pitfalls

- **Tool name mismatch**: `.tool("name")` on the agent must match `ToolDefinition::name` exactly, or
  the runner panics / returns `Error::Agent` at runtime. Check spelling and casing.
- **History ownership**: The runner is stateless. After each turn, append `Message::user(input)` and
  `Message::assistant(&result.output)` to your vec manually before the next call.
- **`text` and `tool_calls` are mutually exclusive in `ModelResponse`**: return one or the other,
  never both. The runner loops until it receives a text-only response.
- **Stream must be pinned**: Always `futures_util::pin_mut!(stream)` before calling `.next().await`
  on a stream from `run_stream`.
- **Don't double-wrap `ToolRegistry` in `Arc`**: Construct it with `Arc::new(...)` once, then use
  `Arc::clone(&registry)` to share it across runners.
- **`run_typed` schema is on the final response**: The schema is validated after all tool call
  rounds complete, not on intermediate turns.
