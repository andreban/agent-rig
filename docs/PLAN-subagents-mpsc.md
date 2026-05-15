# Plan: Subagent tools in `MpscRunner`

## 1. Executive summary

A parent agent can register a child agent under a `ToolDefinition` and
invoke it as a tool call. The child agent's event stream is forwarded
upstream so callers of `MpscRunner::run` observe the full nested run,
not just the final string.

Mechanics:

- `AgentTool` stops implementing `Tool`. It becomes a streaming type
  that owns its own `(definition, Agent, MpscRunner)`.
- `ToolRegistry` switches from `HashMap<String, Box<dyn Tool>>` to
  `HashMap<String, RegistryEntry>` where `RegistryEntry` is an enum of
  `Tool(Box<dyn Tool>)` or `Agent(AgentTool)`.
- `MpscRunner::run` yields `RunnerEvent { thread_id, depth, agent_event
  }` instead of `AgentEvent`.
- The runner's tool-call handler branches on the entry kind: tools
  follow the existing auth → call → result flow; agent entries are
  driven as nested streams whose events are forwarded with depth and
  thread_id adjusted, then collapsed into a synthetic tool-result
  message in the parent thread.

## 2. Decisions

| # | Decision |
| :--- | :--- |
| Stream item | `RunnerEvent { thread_id, depth, agent_event }` |
| Registry shape | Single `ToolRegistry`, holds `enum RegistryEntry { Tool, Agent }` |
| Registration | Two methods: `register(Box<dyn Tool>)`, `register_agent(AgentTool)` |
| `AgentTool` as a `Tool` impl | Removed |
| `AgentTool` ownership of runner | Owns its own `MpscRunner` (independent model/registry/auth) |
| Legacy `AgentRunner` | Treats `RegistryEntry::Agent` as unsupported; the legacy `agent_as_tool` example moves to `MpscRunner` |

## 3. Concrete shapes

### 3.1 `ToolRegistry`

```rust
pub enum RegistryEntry {
    Tool(Box<dyn Tool>),
    Agent(AgentTool),
}

impl RegistryEntry {
    pub fn definition(&self) -> ToolDefinition { /* delegates */ }
}

pub struct ToolRegistry {
    entries: HashMap<String, RegistryEntry>,
}

impl ToolRegistry {
    pub fn register(mut self, tool: Box<dyn Tool>) -> Self;
    pub fn register_agent(mut self, agent_tool: AgentTool) -> Self;
    pub fn definitions(&self) -> Vec<ToolDefinition>;        // both kinds
    pub(crate) fn get(&self, name: &str) -> Option<&RegistryEntry>;
    pub(crate) fn contains(&self, name: &str) -> bool;
}
```

### 3.2 `AgentTool`

```rust
pub struct AgentTool {
    definition: ToolDefinition,
    agent: Agent,
    runner: MpscRunner,
}

impl AgentTool {
    pub fn new(definition: ToolDefinition, agent: Agent, runner: MpscRunner) -> Self;
    pub fn definition(&self) -> &ToolDefinition;

    /// Runs the child agent. `args` is serialized to JSON and passed as
    /// the child's first user message. The returned stream emits the
    /// child's full sequence of `RunnerEvent`s (with the child's own
    /// `thread_id` / `depth` space, starting at depth 0).
    pub fn invoke(
        &self,
        args: serde_json::Value,
    ) -> Pin<Box<dyn Stream<Item = RunnerEvent> + Send>>;
}
```

`invoke` is a thin wrapper around `self.runner.run(self.agent.clone(),
vec![Message::user(serde_json::to_string(&args)?)])` — the child runner
behaves as if it were a fresh top-level run; rebasing is the parent's
job.

### 3.3 `RunnerEvent`

```rust
pub struct RunnerEvent {
    pub thread_id: usize,   // unique within the runner that yielded this event
    pub depth: usize,       // 0 at the root of this runner's run; +1 per nested subagent layer
    pub agent_event: AgentEvent,
}
```

Plus an internal `AtomicUsize` thread-id counter on `MpscRunner` (or on
the run state).

### 3.4 `MpscRunner::run`

Signature becomes:

```rust
pub fn run(
    &self,
    agent: Agent,
    thread: Vec<Message>,
) -> Pin<Box<dyn Stream<Item = RunnerEvent> + Send>>;
```

## 4. Behaviour: agent-tool branch in `handle_tool_calls`

When the parent model emits a tool call whose registry entry is
`RegistryEntry::Agent(agent_tool)`:

1. **Authorize the invocation by name** using the parent's
   `AuthManager`, exactly like a regular tool. On deny → emit
   `ToolCallDenied` at parent depth, synthesize the standard
   `"authorization denied: ..."` tool-result in the parent thread,
   skip the run.
2. **Emit `ToolCallStarted`** at the parent's depth/thread_id (same
   lifecycle marker as for a regular tool call).
3. **Allocate a fresh `thread_id`** from the parent runner's counter
   for this child invocation.
4. **Drive `agent_tool.invoke(args)`**. For every child `RunnerEvent`:
   - Rewrite `thread_id` to the parent-allocated id for this
     invocation. *Child internal thread structure is collapsed into one
     parent-side thread.* (Documented limitation; see §6.)
   - Set `depth = parent_depth + 1 + child.depth`.
   - Forward the event upstream.
   - Side-effect: accumulate `TextDelta` chunks into an output buffer;
     observe `AgentEvent::Error` for early exit.
5. **On child completion** (stream ends, no `Error` observed):
   - Emit `ToolCallFinished` at the parent's depth, with `result =
     json!({ "output": <buffered text> })` — same envelope today's
     `AgentTool` produces.
   - Append `Message::tool_result(call.id, call.name, json!({ "output":
     <text> }), None)` to the parent thread.
6. **On child error** (`AgentEvent::Error(err)` observed in the child's
   stream):
   - Emit `ToolCallError { name, error }` at the parent's depth.
   - Append a synthetic `Message::tool_result(call.id, call.name,
     json!(format!("Error: {err}")), None)` so the parent model can
     react and the assistant turn / tool-result pairing stays intact.

Parallel agent-tool invocations and parallel mixed (tool + agent)
invocations run concurrently via `join_all`, same as today. Events
interleave on the parent stream and are distinguishable by
`thread_id`.

## 5. Implementation phases

| Phase | Task |
| :--- | :--- |
| 1 | `RegistryEntry` enum in `src/tool.rs`. Switch `ToolRegistry` to `HashMap<String, RegistryEntry>`. Add `register_agent`. Update `definitions`, `get`, `contains`. |
| 2 | Rewrite `src/agent_tool.rs`: remove `impl Tool for AgentTool`, swap `AgentRunner` for `MpscRunner` in the struct, add `invoke`. Update unit tests to drive the new stream-shaped API. |
| 3 | Change `MpscRunner::run` return type to `Stream<Item = RunnerEvent>`. Add the per-runner `AtomicUsize` thread-id allocator. Wrap existing event emissions with `RunnerEvent { thread_id, depth: 0, ... }`. |
| 4 | Extend `handle_tool_calls` in `src/mpsc_runner.rs` to branch on `RegistryEntry`. Implement the agent-tool path per §4. |
| 5 | Patch the legacy `AgentRunner` (`src/runner.rs`) to error cleanly when it encounters `RegistryEntry::Agent` (registry lookup returns it; runner reports `Error::Agent("agent tools require MpscRunner")`). Don't add features — this is the legacy path. |
| 6 | Update `examples/mpsc_runner.rs` consumer loop for the new `RunnerEvent` envelope. |
| 7 | Migrate `examples/agent_as_tool.rs` to `MpscRunner` + `register_agent`. This becomes the showcase example. Delete the legacy version once the new one runs end-to-end. |
| 8 | Unit tests: stub `LlmModel` driving a child whose stream produces text deltas and a tool call; assert events are forwarded with bumped depth, fresh thread_id, and that the parent thread gets a synthesized tool-result. Add a test for child-error propagation and one for parallel agent + tool invocations. |
| 9 | Update `docs/SPEC.md` and `docs/PRD.md` to reflect the new shapes. |

## 6. Known limitations (v1, accepted)

- **Collapsed child thread structure.** A child invocation is one
  parent-side `thread_id`. Grandchildren and parallel children inside a
  child are *not* individually identifiable from the parent's stream;
  they show up under the same forwarded `thread_id` with varying
  `depth`. Recoverable later by adding a `parent_thread_id: Option<usize>`
  field to `RunnerEvent` and per-event remapping — not worth the
  bookkeeping until a consumer needs it.
- **Legacy `AgentRunner` cannot run agent tools.** It returns
  `Error::Agent` if a registry entry resolves to an `Agent`. Goes away
  with the legacy runner deletion.
- **No streaming of partial child output to the parent model.** The
  parent model sees the child's final concatenated text as the tool
  result, not deltas. Same as today's `AgentTool`. Future work.
- **No cross-layer auth.** Parent's `AuthManager` only gates the
  *invocation* of the subagent by name. The child's own tool calls are
  gated by the child's own runner's `AuthManager`. Layers are
  independent on purpose.

## 7. Out of scope

- Renaming `MpscRunner` → `AgentRunner` and deleting `src/runner.rs`
  (the broader migration tracked by the memory note).
- Cross-process / remote subagents.
- Adding `parent_thread_id` to `RunnerEvent` (deferred per §6).
