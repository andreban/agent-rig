---
name: agent-rig
description: >
  Guide and write code using the `agent-rig` library — a provider-agnostic AI agent toolkit for Rust.
  Use this skill whenever the user is writing, reading, debugging, or extending code that involves
  `agent_rig`, `AgentRunner`, `Agent`, `LlmModel`, `ToolRegistry`, `AgentTool`, `ApprovalRequest`,
  `GeminiModel`, `OllamaModel`, or any type from this crate. Also trigger when the user asks how to
  add a new LLM provider, implement a custom tool, wire up multi-turn conversations, stream agent
  output, gate tool calls behind user approval, or use structured output in the context of this
  project.
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
async-trait    = "0.1"
serde          = { version = "1", features = ["derive"] }
serde_json     = "1"
futures-util   = "0.3"
tokio          = { version = "1", features = ["full"] }
dotenvy        = "0.15"
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
| `Agent` / `AgentBuilder` | Pure data blueprint: name, instructions, optional output schema, allowed tool names. Implements `Serialize`/`Deserialize` so configs can be stored as JSON/YAML. **Carries no model reference** — the same blueprint can run on any `AgentRunner`. |
| `AgentRunner` | Execution engine; owns an `Arc<dyn LlmModel>` and a list of tool definitions. Cheap to `Clone` (everything is behind `Arc`). |
| `LlmModel` | Async trait every provider implements. Has `generate` (required) and `generate_stream` (default impl wraps `generate`). The extension point for new providers. |
| `Message` / `MessageContent` | Conversation history elements. `MessageContent` is either `Text`, `ToolCalls(Vec<Arc<ToolCall>>)`, or `ToolResult` |
| `ModelRequest` / `ModelResponse` / `ToolCall` | Provider-agnostic request/response envelope. `ModelResponse::text` and `tool_calls` are mutually exclusive per turn. |
| `Tool` / `ToolDefinition` | Async trait for callable tools. `parameters` is a JSON Schema `serde_json::Value`. |
| `ToolRegistry` | Client-side registry of `Tool` and `AgentTool` entries, keyed by name, used to resolve tool calls. |
| `AgentTool` | Wraps an `AgentRunner` + `Agent` so it can be invoked through the same tool-call mechanism. Registered via `ToolRegistry::register`. |
| `AgentEvent` | Stream event: `TurnStart`, `ThinkingDelta`, `TextDelta`, `ToolCall`, `Usage`, `TurnFinish`, `Cancelled`, `Error`. |
| `RunEvent` | An `AgentEvent` tagged with a unique `run_id`. **This is what the runner stream actually yields.** |
| `ToolCallResult` | Outcome of a tool call: `Ok(Value)`, `Err(Error)`, `Denied`, `Unknown`. |
| `Error` | `Provider(String)` or `Agent(String)`. |

**Key design rules:**

- `Agent` is pure data — no model, no runtime state. It holds tool *names*; the matching `Tool`
  implementations live in the client's `ToolRegistry`.
- `AgentRunner::run` is **stateless and streaming**. It takes `&Agent` and `Vec<Arc<Message>>` and returns
  a `Stream<Item = RunEvent>`. The caller maintains conversation history across turns.
- Every event in the stream is a `RunEvent`. Read `event.agent_event` to match the underlying
  `AgentEvent`. The `run_id` field uniquely identifies the run; child agent streams are encapsulated by `AgentTool` and do not leak to the parent stream.
- There is **no** `run_typed`, `Conversation`, `RunBuilder`, or `AgentResult` type. If you see these
  names in older code or docs, they were never shipped — replace them with the streaming API.

---

## Single-Turn (Basic)

```rust
use std::sync::Arc;
use agent_rig::{Agent, model::Message, models::gemini::GeminiModel,
    runner::{AgentEvent, AgentRunner}};
use futures_util::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = dotenvy::dotenv();
    let api_key = std::env::var("GEMINI_API_KEY")?;

    let model = GeminiModel::builder(api_key, "gemini-3.1-flash-lite")
        .temperature(0.7)
        .build();

    let agent = Agent::builder()
        .name("Assistant")
        .instructions("You are a helpful assistant.")
        .build();

    let runner = AgentRunner::new(Arc::new(model));

    let mut answer = String::new();
    let mut stream = runner.run(&agent, vec![Arc::new(Message::user("What is the capital of France?"))]);
    while let Some(event) = stream.next().await {
        match event.agent_event {
            AgentEvent::TextDelta(chunk) => answer.push_str(&chunk),
            AgentEvent::Error(e) => eprintln!("[error] {e}"),
            _ => {}
        }
    }
    println!("{answer}");
    Ok(())
}
```

Concatenating every `TextDelta` reconstructs the model's final reply. The stream ends when the model
produces a turn with no tool calls, or after an `AgentEvent::Error`.

---

## Multi-Turn Conversations (Manual History)

`AgentRunner::run` is stateless: each call takes the full thread of reference-counted `Message`s (`Vec<Arc<Message>>`). The caller is
responsible for appending the user's input and the assistant's reply between turns.

```rust
use std::sync::Arc;
use agent_rig::model::Message;

let mut thread: Vec<Arc<Message>> = Vec::new();

// Turn 1
thread.push(Arc::new(Message::user("My name is Alice.")));
let mut reply = String::new();
let mut stream = runner.run(&agent, thread.clone());
while let Some(event) = stream.next().await {
    if let AgentEvent::TextDelta(chunk) = event.agent_event { reply.push_str(&chunk); }
}
thread.push(Arc::new(Message::assistant(reply)));

// Turn 2 — the runner sees the full history
thread.push(Arc::new(Message::user("What is my name?")));
let mut stream = runner.run(&agent, thread.clone());
// drive the stream and push the new reply onto the thread again
```

The thread is `Vec<Arc<Message>>` — trim, compress, or inject synthetic messages directly between turns.
See `examples/multi_turn.rs` for a complete REPL.

---

## Tool Calling

Under the **client-resolved tool execution model**, the runner does not execute tools internally. Instead, when the model requests a tool call, the runner yields an `AgentEvent::ToolCall(call)` event. The consumer is responsible for looking up the tool in their registry, executing it, and resolving the call.

To give your agent callable functions, implement the `Tool` trait, register it in a client-side `ToolRegistry`, and configure the runner with the registry's definitions:

```rust
use std::sync::Arc;
use async_trait::async_trait;
use agent_rig::{Agent, model::Message, models::gemini::GeminiModel,
    runner::{AgentEvent, AgentRunner},
    tools::{Tool, ToolDefinition, ToolRegistry, ToolResult}};
use futures_util::StreamExt;
use serde_json::{Value, json};

struct GetWeatherTool {
    definition: ToolDefinition,
}

impl Default for GetWeatherTool {
    fn default() -> Self {
        Self {
            definition: ToolDefinition {
                name: "get_weather".to_string(),
                description: "Returns the current temperature in Celsius for a city.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "city": { "type": "string", "description": "The city name." }
                    },
                    "required": ["city"]
                }),
            },
        }
    }
}

#[async_trait]
impl Tool for GetWeatherTool {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn apply(
        &self,
        proposal: Value,
        _cancel: tokio_util::sync::CancellationToken,
    ) -> ToolResult {
        let city = proposal["city"].as_str().unwrap_or("unknown");
        ToolResult::ok(json!({ "city": city, "celsius": 22.0 }))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = dotenvy::dotenv();
    let api_key = std::env::var("GEMINI_API_KEY")?;

    let registry = ToolRegistry::new().register(GetWeatherTool::default());

    let agent = Agent::builder()
        .name("Weather Bot")
        .instructions("Answer weather questions using the available tools.")
        .tool("get_weather")  // must match ToolDefinition::name exactly
        .build();

    let model = GeminiModel::builder(api_key, "gemini-3.1-flash-lite").build();
    let runner = AgentRunner::with_tools(Arc::new(model), registry.definitions());

    let mut answer = String::new();
    let mut stream = runner.run(&agent, vec![Arc::new(Message::user("What is the temperature in Tokyo?"))]);
    while let Some(event) = stream.next().await {
        match event.agent_event {
            AgentEvent::ToolCall(tool_call) => {
                println!("[start] {}({})", tool_call.details.name, tool_call.details.args);
                let Some(tool) = registry.get(&tool_call.details.name) else {
                    tool_call.resolve(ToolResult::error("Unknown tool"));
                    continue;
                };
                let result = tool.apply(
                    tool_call.details.args.clone(),
                    tool_call.cancellation_token.clone()
                ).await;
                println!("[done]  {} → {result}", tool_call.details.name);
                tool_call.resolve(result);
            }
            AgentEvent::TextDelta(chunk) => answer.push_str(&chunk),
            AgentEvent::Error(e) => eprintln!("[runner] {e}"),
            _ => {}
        }
    }
    println!("{answer}");
    Ok(())
}
```

**Rules:**

- `.tool("name")` on the agent must match `ToolDefinition::name` exactly. (Currently the agent's
  `tool_names` are not enforced against the registry, but the model will only see tools the registry
  declares — so a missing registration silently does nothing.)
- Multiple tool calls in a single model turn are executed **concurrently** via `futures_util::future::join_all`.
  `join_all` preserves input order, so the tool-result messages are appended to the thread in the
  same order the model issued them — even though `ToolCallStarted` / `ToolCallFinished` events may
  interleave.
- A `ToolCallResult::Unknown` outcome is generated for hallucinated tool names (no matching
  registry entry); no `ToolCallStarted` is emitted, but a synthetic result message is appended so
  the assistant turn and tool-result messages stay paired.
- `ToolRegistry` is shared via `Arc` so a single registry can be reused across runners.

---

## Streaming

Streaming is the only mode — `AgentRunner::run` already returns a stream. There is no separate
`run` (blocking) entry point.

```rust
use futures_util::StreamExt;
use agent_rig::runner::AgentEvent;

let mut stream = runner.run(&agent, vec![Arc::new(Message::user("Explain Rust ownership."))]);
while let Some(event) = stream.next().await {
    match event.agent_event {
        AgentEvent::ThinkingDelta(token) => print!("\x1b[2m{token}\x1b[0m"),  // dim
        AgentEvent::TextDelta(chunk) => print!("{chunk}"),
        AgentEvent::ToolCall(call) => println!("\n[tool call: {}]", call.details.name),
        AgentEvent::Usage(u) => println!("[usage {u:?}]"),
        AgentEvent::Cancelled => println!("[cancelled]"),
        AgentEvent::Error(e) => eprintln!("[error] {e}"),
        _ => {}
    }
}
```

**Notes:**

- `ThinkingDelta` chunks only arrive when the provider supports extended thinking *and* it is
  enabled. Currently that means Gemini with `thinking_config(ThinkingConfig { include_thoughts:
  true, .. })`.
- `GeminiModel` uses the default `generate_stream` (single batched `TextDelta` after the full
  response arrives). `OllamaModel` implements `generate_stream` natively and emits incremental
  text deltas.
- Ollama disables streaming automatically when tools are present (provider requirement); tool
  calls arrive as a single batch in that case.

---

## Authorization (Gating Tool Calls)

Because tool execution is resolved by the client, gating tool calls behind user approval is highly straightforward. You can orchestrate a two-phase execution flow (`Tool::propose` followed by `Tool::apply`) directly in your event loop:

1. **Propose**: Call `tool.propose(...)` to resolve the raw arguments into a *proposal* — a side-effect-free description of what will happen (e.g., an edit tool reading the file and returning a diff).
2. **Prompt**: Show this proposal/diff to the user and prompt for confirmation.
3. **Apply**: If approved, call `tool.apply(...)` with the proposal and resolve the call. If denied, resolve the call with a soft error.

```rust
use std::sync::Arc;
use agent_rig::tools::{Tool, ToolDefinition, ToolResult};
use serde_json::Value;

struct SendEmailTool { definition: ToolDefinition }

impl Tool for SendEmailTool {
    fn definition(&self) -> &ToolDefinition { &self.definition }

    fn requires_approval(&self, _args: &Value) -> bool {
        true  // always prompt before sending
    }

    async fn propose(
        &self,
        tool_call: Arc<crate::model::ToolCall>,
        _cancel: tokio_util::sync::CancellationToken,
    ) -> ToolResult {
        ToolResult::ok(json!({
            "to": tool_call.args["to"],
            "preview": format!("Subject: {}\n\n{}", tool_call.args["subject"], tool_call.args["body"])
        }))
    }

    async fn apply(
        &self,
        proposal: Value,
        _cancel: tokio_util::sync::CancellationToken,
    ) -> ToolResult {
        // Only reached after the user approves
        ToolResult::ok(json!({ "status": "sent" }))
    }
}
```

In your event loop, orchestrate the approval logic like this:

```rust
AgentEvent::ToolCall(tool_call) => {
    let Some(tool) = registry.get(&tool_call.details.name) else {
        tool_call.resolve(ToolResult::error("Unknown tool"));
        continue;
    };

    // 1. Generate the proposal
    let proposal = tool.propose(
        tool_call.details.clone(),
        tool_call.cancellation_token.clone()
    ).await;

    let ToolResult::Ok(proposal_val) = proposal else {
        tool_call.resolve(proposal); // propose failed, resolve immediately
        continue;
    };

    // 2. Check if approval is needed and prompt the user
    let approved = if tool.requires_approval(&tool_call.details.args) {
        let preview = proposal_val["preview"].as_str().unwrap_or("");
        prompt_user_for_email_send(preview) // your custom UI prompt logic
    } else {
        true
    };

    if !approved {
        tool_call.resolve(ToolResult::error("User rejected approval"));
        continue;
    }

    // 3. Apply the approved proposal
    let result = tool.apply(proposal_val, tool_call.cancellation_token.clone()).await;
    tool_call.resolve(result);
}
```

When the model returns multiple tool calls in one turn, approval requests may arrive concurrently.
Consumers sharing UI resources (stdin, a modal dialog) must serialize internally — typically with
a `tokio::sync::Mutex`.

See `examples/mpsc_auth_flow.rs` for a working CLI prompt.

---

## Cancellation

Cancellation is cooperative. The simplest pattern: **drop the returned stream**. The runner
aborts the in-flight provider call and any running tool futures at their next await point.

```rust
let stream = runner.run(&agent, vec![Message::user(input)]);
// drop(stream) anywhere — typically on Ctrl-C, client disconnect, or
// when a wrapping task is aborted — cancels everything.
```

For deadlines or sharing a cancel signal across sibling tasks, use
`run_with_cancellation`:

```rust
use tokio_util::sync::CancellationToken;

let cancel = CancellationToken::new();
let mut stream = runner.run_with_cancellation(&agent, thread, cancel.clone());

// Cancel from anywhere:
//   cancel.cancel();
// — or compose with a deadline:
//   tokio::spawn(async move {
//       tokio::time::sleep(Duration::from_secs(30)).await;
//       cancel.cancel();
//   });
```

Both `Tool::propose` and `Tool::apply` receive the cancel token; long-running tools
should `select!` on it or pass it down to the libraries they call. Tools that ignore it
still terminate the run correctly — the runner races each phase against `cancel` — but their
side effects may continue in the background until they finish on their own.

A cancelled run emits a terminal `AgentEvent::Cancelled` before the stream ends.
Under drop-the-stream cancellation the event is best-effort (the receiver may
already be gone). See `examples/cancellation.rs`.

---

## Structured Output

Set `output_schema` on the agent to constrain the model's response to a JSON Schema. The
[`schemars`](https://crates.io/crates/schemars) crate (≥ 1.0) can generate the schema from a Rust
struct. Accumulate the streamed text and deserialize it.

```toml
[dev-dependencies]
schemars = "1"
serde    = { version = "1", features = ["derive"] }
```

```rust
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ResearchPlan {
    title: String,
    tasks: Vec<String>,
}

let agent = Agent::builder()
    .name("Planner")
    .instructions("Produce a structured research plan.")
    .output_schema(schemars::schema_for!(ResearchPlan))
    .build();

let mut output = String::new();
let mut stream = runner.run(&agent, vec![Message::user("AI agents in Rust")]);
while let Some(event) = stream.next().await {
    if let AgentEvent::TextDelta(chunk) = event.agent_event { output.push_str(&chunk); }
}
let plan: ResearchPlan = serde_json::from_str(&output)?;
```

**Notes:**

- Providers that don't support structured output silently ignore `output_schema`.
- `GeminiModel` normalises schemars output internally (strips `$schema`, inlines `$ref`/`$defs`)
  to satisfy the Gemini API's restricted JSON Schema subset.
- `output_schema` and tool calling can be combined; the schema is applied to the *final* text
  response after all tool-call rounds resolve.

---

## Agent Composition (`AgentTool`)

Wrap an `AgentRunner` + `Agent` pair as an `AgentTool` and register it with a parent runner using the standard `ToolRegistry::register` method. The parent model invokes the child agent as if it were a regular tool. The child's run is driven and fully encapsulated internally within `AgentTool::apply`, yielding a single flat text response back to the parent model.

```rust
use std::sync::Arc;
use agent_rig::{Agent, model::Message, runner::AgentRunner,
    tools::{AgentTool, ToolDefinition, ToolRegistry}, models::gemini::GeminiModel};
use serde_json::json;

const MODEL: &str = "gemini-3.1-flash-lite";

// --- Child agent ---
let child_model  = GeminiModel::builder(&api_key, MODEL).build();
let child_agent  = Agent::builder()
    .name("Summariser")
    .instructions("You receive a JSON object with a `text` field. Summarise it in two sentences.")
    .build();
let child_runner = AgentRunner::new(Arc::new(child_model));

let summarise_tool = AgentTool::new(
    ToolDefinition {
        name: "summarise".to_string(),
        description: "Summarises a long piece of text. Pass the text in the `text` field.".to_string(),
        parameters: json!({
            "type": "object",
            "properties": { "text": { "type": "string" } },
            "required": ["text"]
        }),
    },
    child_agent,
    child_runner,
);

// --- Parent runner ---
let registry = ToolRegistry::new().register(summarise_tool);
let parent_model  = GeminiModel::builder(&api_key, MODEL).build();
let parent_runner = AgentRunner::with_tools(Arc::new(parent_model), registry.definitions());

let parent_agent = Agent::builder()
    .name("Orchestrator")
    .instructions("Use the `summarise` tool when asked to summarise text.")
    .tool("summarise")
    .build();

let mut stream = parent_runner.run(&parent_agent,
    vec![Arc::new(Message::user("Summarise: Rust is …"))]);
while let Some(event) = stream.next().await {
    // ... handle event.agent_event ...
}
```

**Notes:**

- `AgentTool` **owns** its `AgentRunner` (not a shared reference). Each child has its own model
  binding. Multiple concurrent `apply` invocations are safe.
- Internally, `AgentTool::apply` serialises the proposal JSON to a string and passes it as the
  child's user message. The child's `TextDelta` chunks are accumulated; the tool result returned to
  the parent model is the raw accumulated string (returned as `ToolResult::Ok`).
- Child agents can have their own tools and even their own sub-agents. Nesting is unlimited.
- Because `AgentTool` encapsulates the child run, the parent's event stream remains completely flat. Child events (such as the child's `ThinkingDelta` or its own tool calls) do not pollute the parent's event stream.

---

## Provider Configuration

### Google Gemini (`feature = "gemini"`)

Requires `GEMINI_API_KEY` environment variable.

```rust
use agent_rig::models::gemini::GeminiModel;
use geologia::prelude::{ThinkingConfig, ThinkingLevel};

// Minimal
let model = GeminiModel::new(api_key, "gemini-3.1-flash-lite");

// Full configuration
let model = GeminiModel::builder(api_key, "gemini-3.1-flash-lite")
    .temperature(0.7)
    .max_output_tokens(2048)
    .top_p(0.9)
    .top_k(40)
    .stop_sequences(vec!["END".into()])
    .thinking_config(ThinkingConfig {
        include_thoughts: true,
        thinking_level: Some(ThinkingLevel::High),
        ..Default::default()
    })
    .build();
```

Builder method names: `temperature`, `max_output_tokens`, `top_p`, `top_k`, `stop_sequences`,
`thinking_config`. The `ThinkingConfig` and `ThinkingLevel` types come from `geologia::prelude`.

### Ollama (`feature = "ollama"`)

Requires a running Ollama server. The first builder argument is the **server URL**.

```rust
use agent_rig::models::ollama::OllamaModel;

// Minimal
let model = OllamaModel::new("http://localhost:11434", "llama3.2");

// Full configuration
let model = OllamaModel::builder("http://localhost:11434", "llama3.2")
    .temperature(0.8)
    .seed(42)
    .top_k(40)
    .top_p(0.9)
    .num_ctx(4096)
    .num_predict(512)
    .build();
```

Structured output requires Ollama ≥ 0.5 and a model that supports it. Ollama disables streaming
when tools are present (provider requirement).

---

## Implementing a Custom Provider

Implement `LlmModel`. Only `generate` is required; `generate_stream` has a default that calls
`generate` and emits the result as a sequence of chunks.

```rust
use async_trait::async_trait;
use agent_rig::{error::Error,
    model::{LlmModel, MessageContent, ModelRequest, ModelResponse, Role, ToolCall}};

struct MyModel { api_key: String }

#[async_trait]
impl LlmModel for MyModel {
    async fn generate(&self, request: ModelRequest) -> Result<ModelResponse, Error> {
        // 1. Translate request.messages → your provider's format.
        //    Role::User → user, Role::Assistant → assistant/model
        //    MessageContent::Text(s)                                    → plain text turn
        //    MessageContent::ToolCalls(calls)                            → assistant tool-call turn (Vec<Arc<ToolCall>>)
        //    MessageContent::ToolResult { tool_call, result }            → tool result (echo `tool_call.id`)
        //
        // 2. Map request.system → system prompt (if supported).
        // 3. Map request.tools → provider function declarations (if non-empty).
        // 4. Map request.output_schema → structured output constraint (if supported; otherwise ignore).
        // 5. Call your API.
        // 6. Return either tool_calls (text = None) or text (tool_calls = vec![]).
        //    `text` and `tool_calls` are mutually exclusive per turn.

        Ok(ModelResponse {
            text: Some("Hello from MyModel!".to_string()),
            tool_calls: vec![],
            thinking: None,
            token_usage: None,
        })
    }
}
```

**Returning tool calls:**

```rust
Ok(ModelResponse {
    text: None,
    tool_calls: vec![ToolCall::new(
        "call_abc123".to_string(),                 // echo this id in the ToolResult
        "get_weather".to_string(),
        serde_json::json!({ "city": "Tokyo" }),
    )],
    thinking: None,
    token_usage: None,
})
```

**Reporting token usage:** populate `token_usage` with a `TokenUsage { input_tokens, output_tokens, cached_input_tokens, thinking_tokens, tool_use_prompt_tokens }` when the provider returns per-call token counts. Leave dimensions the provider does not report as `None` (distinct from `Some(0)`). The runner forwards it as `AgentEvent::Usage` — one event per model call. Cache semantics are subset: `cached_input_tokens ⊆ input_tokens`.

If your provider has opaque per-call metadata that must be round-tripped (e.g. Gemini's
`thought_signature`), set it via `ToolCall { provider_metadata: Some(...), .. }`. The runner
preserves it on the matching tool-result message; you read it back from
`MessageContent::ToolResult { tool_call, .. }`'s `tool_call.provider_metadata` on the next turn.

**Overriding `generate_stream`** (for true token-by-token output):

```rust
use std::pin::Pin;
use futures_util::Stream;
use agent_rig::model::ModelStreamChunk;

fn generate_stream(
    &self,
    request: ModelRequest,
) -> Pin<Box<dyn Stream<Item = Result<ModelStreamChunk, Error>> + Send + '_>> {
    Box::pin(async_stream::stream! {
        // Yield Ok(ModelStreamChunk::Thinking(s))   for reasoning tokens.
        // Yield Ok(ModelStreamChunk::TextDelta(s))  for text tokens.
        // Yield Ok(ModelStreamChunk::ToolCall(tc))  once for each complete tool call.
        // Tool calls are NEVER streamed mid-call — emit the full ToolCall in one chunk.
    })
}
```

**Adding a feature flag** (for library contributors):

```toml
# Cargo.toml
[features]
myprovider = ["dep:my-provider-crate"]
full       = ["gemini", "ollama", "myprovider"]   # add to the alias

[dependencies]
my-provider-crate = { version = "...", optional = true }
```

```rust
// src/models/mod.rs
#[cfg(feature = "myprovider")]
pub mod myprovider;
```

---

## Common Pitfalls

- **Don't unwrap `event` as `AgentEvent`**: The stream yields `RunEvent`, not `AgentEvent`. Match on
  `event.agent_event` (or destructure `RunEvent { agent_event, .. }`).
- **`Box::new(model)` vs `Arc::new(model)`**: The runner takes `Arc<dyn LlmModel>`. Always wrap
  models in `Arc`, never `Box`. Older drafts of the API used `Box`; that's gone.
- **Tool name mismatch**: `.tool("name")` on the agent must match `ToolDefinition::name` exactly.
- **`text` and `tool_calls` are mutually exclusive in `ModelResponse`**: return one or the other,
  never both. The runner loops until it receives a text-only response.
- **`AgentTool` is a standard `Tool`**: register it with `register(agent_tool)` just like any other tool. There is no `register_agent` method.
- **Don't double-wrap `ToolRegistry` in `Arc`**: build the registry first. In this version, `AgentRunner` does not take a `ToolRegistry` — it only takes `registry.definitions()` — so you do not need to wrap it in `Arc` for the runner.
- **History is the caller's job**: each `runner.run(...)` call starts fresh. For multi-turn,
  accumulate `Arc::new(Message::user(input))` and `Arc::new(Message::assistant(reply))` in a `Vec<Arc<Message>>` between calls.
- **`run` borrows the agent**: `runner.run(&agent, thread)` — take a reference. The thread is moved.
- **Gemini text arrives as one batch**: `GeminiModel` uses the default `generate_stream` today, so
  `TextDelta` is a single chunk after the whole response is received. Reasoning tokens and tool
  calls still stream correctly. Ollama streams natively.
