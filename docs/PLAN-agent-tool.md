# Plan: Agent as a Tool (`AgentTool`)

## 1. Executive Summary

The goal is to allow a developer to wrap an `Agent` + `AgentRunner` pair into a value that implements `Tool`, so a parent agent can delegate sub-tasks to a child agent as if it were a regular tool. The parent model sees a `ToolDefinition` (name, description, parameters) supplied by the caller; when the parent invokes it, the child agent executes its own full agentic loop (including its own tools) and returns the result. This enables hierarchical multi-agent pipelines with no changes to the existing `Tool`, `ToolRegistry`, or `AgentRunner` interfaces.

---

## 2. User Stories

- **As a developer**, I want to build a parent agent that can call a specialist child agent (e.g., a summariser, a classifier) without needing to know that the child is itself an agent.
- **As a developer**, I want to control how the parent model sees the child agent: its name, description, and the parameters it expects.
- **As a developer**, I want the child agent to be able to use its own tools independently of the parent's registry.

---

## 3. Functional Requirements

### 3.1 `AgentTool` Struct

A new public struct in `src/agent_tool.rs` that implements `Tool`:

```rust
pub struct AgentTool {
    definition: ToolDefinition,  // name/description/parameters exposed to the parent model
    agent: Agent,
    runner: AgentRunner,
}
```

### 3.2 Constructor

```rust
impl AgentTool {
    pub fn new(definition: ToolDefinition, agent: Agent, runner: AgentRunner) -> Self;
}
```

The caller is responsible for constructing the `AgentRunner` with whatever model and registry the child agent needs.

### 3.3 `Tool` Implementation

```rust
#[async_trait]
impl Tool for AgentTool {
    fn definition(&self) -> ToolDefinition {
        self.definition.clone()
    }

    async fn call(&self, args: serde_json::Value) -> Result<serde_json::Value, Error> {
        let input = serde_json::to_string(&args)
            .map_err(|e| Error::Agent(format!("failed to serialize args: {e}")))?;
        let result = self.runner.run(&self.agent, &input).await?;
        Ok(serde_json::json!({ "output": result.output }))
    }
}
```

**Input convention:** `args` is serialized to a JSON string and passed as the child agent's input. The child agent's instructions should describe how to interpret it (e.g., "You receive a JSON object with a `text` field — summarise it.").

**Output convention:** The child's final text output is returned as `{ "output": "<text>" }` so the parent model receives a structured, readable result.

---

## 4. Technical Architecture

### 4.1 Module Placement

`AgentTool` lives in `src/agent_tool.rs`. This avoids a circular dependency:

```
tool.rs       — no dependency on runner.rs
runner.rs     — depends on tool.rs
agent_tool.rs — depends on tool.rs + runner.rs + agent.rs
```

`lib.rs` re-exports `AgentTool` from `agent_tool.rs`.

### 4.2 Ownership Model

`AgentTool` **owns** its `AgentRunner`. Each distinct sub-agent tool maintains its own model binding. Concurrent `call` invocations are safe because `AgentRunner::run` takes `&self`.

### 4.3 Data Flow

```
parent model
    │  tool call: { "text": "..." }
    ▼
AgentTool::call(args)
    │  serde_json::to_string(&args) → input string
    ▼
AgentRunner::run(&self.agent, &input)
    │  child agentic loop (may invoke child's own tools)
    ▼
AgentResult { output }
    │  json!({ "output": result.output })
    ▼
parent model receives tool result
```

### 4.4 Usage Pattern

```rust
// Child agent
let child_runner = AgentRunner::with_registry(
    Box::new(GeminiModel::builder(api_key, "gemini-3.1-flash-lite-preview").build()),
    Arc::new(ToolRegistry::new().register(Box::new(SomeChildTool))),
);
let child_agent = Agent::builder()
    .name("Summariser")
    .instructions("Summarise the text in the 'text' field of the JSON input.")
    .build();

// Wrap as a tool
let summarise_tool = AgentTool::new(
    ToolDefinition {
        name: "summarise".to_string(),
        description: "Summarises a long piece of text. Pass the text in the 'text' field.".to_string(),
        parameters: json!({
            "type": "object",
            "properties": { "text": { "type": "string" } },
            "required": ["text"]
        }),
    },
    child_agent,
    child_runner,
);

// Register with the parent
let registry = Arc::new(ToolRegistry::new().register(Box::new(summarise_tool)));
let parent_runner = AgentRunner::with_registry(Box::new(parent_model), registry);
let parent_agent = Agent::builder()
    .name("Orchestrator")
    .instructions("Use the summarise tool when given a long document.")
    .tool("summarise")
    .build();

let result = parent_runner.run(&parent_agent, "Summarise this: ...").await?;
```

---

## 5. Implementation Plan

| Phase | Task | Description |
| :--- | :--- | :--- |
| **Phase 1** | **`src/agent_tool.rs`** | Create the file. Define `AgentTool` with fields `definition`, `agent`, `runner`. Implement `AgentTool::new`. Implement `Tool` for `AgentTool` (`definition` returns a clone; `call` serializes args, runs the child agent, wraps the output). Add rustdoc with an example. |
| **Phase 2** | **`lib.rs` re-export** | Add `pub mod agent_tool;` and re-export `AgentTool` at the crate root. |
| **Phase 3** | **Unit tests** | In `src/agent_tool.rs`, add `#[cfg(test)]` tests using a stub `LlmModel` for the child runner: verify `call` passes serialized args as input, verify the output is wrapped as `{ "output": "..." }`, verify child agent errors propagate as `Error::Agent`. |
| **Phase 4** | **Integration test** | Add a test in `tests/` that wires a real child `AgentRunner` (Gemini or Ollama) into a parent runner via `AgentTool` and verifies the parent can invoke the child. |
| **Phase 5** | **Example** | Add `examples/agent_as_tool.rs`: a runnable Gemini example with a parent orchestrator agent that delegates to a child summariser agent via `AgentTool`. Should mirror the structure of `examples/simple_agent.rs`. |
| **Phase 6** | **Docs update** | Update `docs/SPEC.md` and `docs/PRD.md` to reflect the implemented state (move `AgentTool` out of Roadmap). |

---

## 6. Design Constraints & Notes

- **No changes to existing interfaces.** `Tool`, `ToolRegistry`, `AgentRunner`, and `Agent` are unchanged. `AgentTool` is purely additive.
- **Child isolation.** The child runner has its own model and registry; it is completely independent of the parent's. This is intentional — the parent should not need to know or manage the child's tools.
- **Input as JSON string.** Passing serialized JSON rather than a raw string field keeps the child agent's input self-describing and avoids inventing a new convention. The child's instructions tell it how to parse the input.
- **Error propagation.** Errors from the child's agentic loop (`Error::Provider`, `Error::Agent`) surface directly to the parent runner, which will return them to the caller without wrapping.
- **`AgentRunner` is not `Clone`.** Ownership is the natural model; share child runners by wrapping in `Arc` and implementing `Tool` manually if the same child must be registered in multiple parent registries.
