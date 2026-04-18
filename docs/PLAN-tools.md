# Plan: Decoupled Tool Support for agent_rig

## 1. Executive Summary
The goal is to implement a tool-calling system that keeps `Agent` fully serializable. `Agent` declares only the *names* of tools it uses; the actual `ToolDefinition` (description, parameters schema) is owned by each `Tool` implementation and lives in the `AgentRunner`'s registry. At runtime the runner resolves names to definitions, forwards them to the model, and executes returned tool calls. The `LlmModel` trait and both provider adapters must also be extended to carry tool definitions into requests and surface tool calls in responses.

---

## 2. Updated User Stories
* **As a developer**, I want to save my Agent's configuration (including which tools it uses by name) to a file and reload it without redefining tool logic.
* **As a developer**, I want to register "global" tools that any agent can use.
* **As a developer**, I want to ensure that only authorized tool logic is executed by the runner.

---

## 3. Functional Requirements

### 3.1 Tool Identity (`ToolDefinition`)
A struct returned by `Tool::definition()` that describes a tool to the model.
* **Fields:** `name: String`, `description: String`, `parameters: serde_json::Value` (JSON Schema).
* **Derives:** `Clone`.
* **Role:** Owned by each `Tool` implementation; resolved from the registry at runtime; forwarded to the model via `ModelRequest::tools`; translated by each provider adapter into its SDK-specific format. Never stored in `Agent`.

### 3.2 Tool Logic (`Tool` Trait)
The executable part of a tool. Lives on the `AgentRunner`, never in `Agent`.

```rust
#[async_trait]
pub trait Tool: Send + Sync {
    fn definition(&self) -> ToolDefinition;
    async fn call(&self, args: serde_json::Value) -> Result<serde_json::Value, Error>;
}
```

`call` receives the raw JSON arguments object from the model and returns a JSON value sent back as the tool response.

### 3.3 The Tool Registry
A standalone public struct (`ToolRegistry`) wrapping a `HashMap<String, Box<dyn Tool>>`. Independent of any runner so a single registry can be shared across multiple `AgentRunner` instances via `Arc<ToolRegistry>`.

```rust
pub struct ToolRegistry { ... }

impl ToolRegistry {
    pub fn new() -> Self;
    pub fn register(mut self, tool: Box<dyn Tool>) -> Self;  // builder-style
}
```

`AgentRunner` holds an `Arc<ToolRegistry>`. The default (no tools) is an empty registry wrapped in `Arc`.

**Validation timing:** At the start of `run()`, before any network call, verify that every name in `agent.tool_names` has a matching key in the registry. Return `Error::Agent` if any are missing.

### 3.4 `ModelRequest` / `ModelResponse` Extensions
Tool support requires extending the model layer — the protocol bridge between runner and providers:

```rust
pub struct ModelRequest {
    pub messages: Vec<Message>,
    pub system: Option<String>,
    pub output_schema: Option<serde_json::Value>,
    pub tools: Vec<ToolDefinition>,      // NEW
}

pub struct ToolCall {
    pub id: String,                      // provider-supplied; must be echoed in the response
    pub name: String,
    pub args: serde_json::Value,
}

pub struct ModelResponse {
    pub text: Option<String>,            // None when the model issued tool calls
    pub tool_calls: Vec<ToolCall>,       // empty on a final text response
}
```

`text` and `tool_calls` are mutually exclusive per turn. `AgentResult::output` remains `String`; `run` returns it only after the loop produces a text response.

### 3.5 Provider Adapter Changes
Both adapters must be updated:

* **`GeminiModel`:** Translate `ModelRequest::tools` → `FunctionDeclaration` / `Tools`. Detect `PartData::FunctionCall { id, name, args }` in the response and map to `ToolCall`. Construct `PartData::FunctionResponse` echoing `id` when returning tool results.
* **`OllamaModel`:** Translate `ModelRequest::tools` → Ollama tool format. Map Ollama tool call parts to `ToolCall`. Return results via Ollama's tool response message format.

---

## 4. Technical Architecture

### 4.1 Data Flow
1. **Serialization:** `Agent` (with `tool_names: Vec<String>`) ↔ JSON/YAML file. Definitions are not serialized.
2. **Initialization:** Developer loads `Agent`, creates `AgentRunner`, calls `register_tool` for each implementation.
3. **Validation:** `run()` checks every name in `agent.tool_names` exists in the registry. Fails fast with `Error::Agent` if not.
4. **Inference:** `AgentRunner` resolves each name to a `ToolDefinition` via `registry[name].definition()`, builds a `ModelRequest` with `tools`, and calls `model.generate`.
5. **Tool loop:** If `response.tool_calls` is non-empty, execute each tool, append the results to conversation history, and call `model.generate` again.
6. **Completion:** Loop exits when `response.tool_calls` is empty; return `AgentResult { output: response.text.unwrap() }`.

### 4.2 Agentic Loop (pseudocode)

```
messages = [user_message(input)]
loop:
    response = model.generate(ModelRequest { messages, system, tools, output_schema })
    if response.tool_calls is empty:
        return AgentResult { output: response.text.unwrap() }
    append assistant tool-call turn to messages
    for each call in response.tool_calls:
        result = registry[call.name].call(call.args).await
        append tool-response turn (id, name, result) to messages
```

### 4.3 Proposed Type Signatures

```rust
// src/tool.rs  (new file)
#[derive(Debug, Clone)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn definition(&self) -> ToolDefinition;
    async fn call(&self, args: serde_json::Value) -> Result<serde_json::Value, Error>;
}

// src/model.rs  (additions)
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub args: serde_json::Value,
}
// ModelRequest gains:  pub tools: Vec<ToolDefinition>
// ModelResponse gains: pub text: Option<String>,  pub tool_calls: Vec<ToolCall>

// src/agent.rs  (updated)
#[derive(Serialize, Deserialize)]
pub struct Agent {
    name: String,
    instructions: String,
    output_schema: Option<serde_json::Value>,
    tool_names: Vec<String>,             // names only — definitions live in the registry
}

// src/tool.rs  (additions)
pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self;
    pub fn register(mut self, tool: Box<dyn Tool>) -> Self;  // builder-style, consumes self
}

// src/runner.rs  (updated)
pub struct AgentRunner {
    model: Box<dyn LlmModel>,
    registry: Arc<ToolRegistry>,
}

impl AgentRunner {
    pub fn new(model: Box<dyn LlmModel>) -> Self;                         // empty registry
    pub fn with_registry(model: Box<dyn LlmModel>, registry: Arc<ToolRegistry>) -> Self;
    pub async fn run(&self, agent: &Agent, input: &str) -> Result<AgentResult, Error>;
    pub async fn run_typed<T: DeserializeOwned>(&self, agent: &Agent, input: &str) -> Result<T, Error>;
}
```

---

## 5. Implementation Plan

| Phase | Task | Description |
| :--- | :--- | :--- |
| **Phase 1** | **Core Types** | Add `src/tool.rs` with `ToolDefinition` and `Tool` trait. Add `ToolCall` to `src/model.rs`. Update `ModelRequest` (add `tools: Vec<ToolDefinition>`) and `ModelResponse` (change `text` to `Option<String>`, add `tool_calls: Vec<ToolCall>`). |
| **Phase 2** | **Agent Update** | Add `tool_names: Vec<String>` to `Agent` and `AgentBuilder`. Add `#[derive(Serialize, Deserialize)]` to `Agent`. |
| **Phase 3** | **Registry** | Add `ToolRegistry` to `src/tool.rs` with `new()` and builder-style `register()`. `AgentRunner` gains `registry: Arc<ToolRegistry>` and a `with_registry` constructor. |
| **Phase 4** | **Provider Adapters** | Update `GeminiModel`: translate `ToolDefinition` ↔ `FunctionDeclaration`, map `FunctionCall` parts to `ToolCall`, echo `id` in `FunctionResponse`. Update `OllamaModel` analogously. |
| **Phase 5** | **Agentic Loop** | Rewrite `AgentRunner::run`: validate registry at entry, loop on tool calls (execute → append history → re-generate) until a text response is produced. |
| **Phase 6** | **Tests** | Unit tests with a stub `LlmModel` that emits one round of tool calls then a text response. Integration tests for both providers. |

---

## 6. Design Constraints & Notes
* **Type safety:** `serde_json::Value` is used at the tool boundary so the bridge between the model's JSON strings and Rust types is explicit.
* **Async execution:** `Tool::call` is async via `async-trait`, consistent with `LlmModel::generate`.
* **Tool call ID:** `ToolCall::id` must be echoed in the tool response. Gemini requires it; Ollama may not — adapters handle the difference internally.
* **`text` nullability:** `ModelResponse::text` is `Option<String>`. A response with tool calls has no text; a final response has no tool calls. `run` returns `Error::Agent` if the loop exits without a text response.
* **Validation timing:** Registry validation at the start of `run()`, before any network call, so misconfigurations fail fast.
* **Registry sharing:** `ToolRegistry` is independent of `AgentRunner`. Multiple runners sharing the same registry pass `Arc::clone(&registry)` to each. `AgentRunner::new` defaults to an empty registry for tool-free agents.
* **Name matching:** `ToolRegistry::register` keys on `tool.definition().name`; this must exactly match names in `agent.tool_names`.
