# Plan: Token Usage Reporting

Tracking issue: [#28](https://github.com/andreban/rust-agent-kit/issues/28)

## 1. Executive Summary

Consumers of `AgentRunner::run` cannot observe per-turn token counts. Provider responses carry `usageMetadata` (Gemini) and `prompt_eval_count` / `eval_count` (Ollama), but neither `ModelResponse` nor any `AgentEvent` variant surfaces them, so downstream tools (`arnes` and its `PricingRegistry::compute_cost(&ModelKey, &TokenCounts)`) cannot compute cost, track quotas, or attribute usage.

The fix is a provider-agnostic `TokenUsage` struct that flows from provider adapters through the model layer up to the runner's event stream, with one usage report per model call (not per run).

---

## 2. User Stories

- **As a developer**, I want to know how many tokens each model call consumed so I can compute cost per request.
- **As a developer**, I want input vs. output tokens separated so I can attribute usage by direction.
- **As a developer integrating cached prompts**, I want to see how many tokens hit the cache so cache effectiveness is observable.
- **As a stream-only consumer** (no runner), I still want to read usage from the raw model stream.

---

## 3. Naming

The struct is named **`TokenUsage`**. Rationale:

- `Usage` alone is too generic — every crate uses it for something different.
- `TokenUsage` makes the unit (tokens, not requests or seconds) explicit at the call site.
- **Alternative under consideration**: `TokenCounts`. The downstream `arnes` crate already uses `TokenCounts` in `PricingRegistry::compute_cost(&ModelKey, &TokenCounts)`. If matching arnes is more valuable than standing alone, rename. The field shape is the same either way; only the type identifier changes.

A small follow-up subsection on this question is **not** worth a public type alias — pick one.

---

## 4. Functional Requirements

### 4.1 `TokenUsage` (new, in `src/model.rs`)

```rust
/// Token counts reported by a provider for one model call.
///
/// Every field is `Option<u32>` so providers that do not report a given
/// dimension leave it `None` — distinct from `Some(0)`, which means
/// "the provider reported zero tokens in this dimension".
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenUsage {
    /// Prompt / input tokens billed for this call.
    pub input_tokens: Option<u32>,
    /// Generated / output tokens billed for this call.
    pub output_tokens: Option<u32>,
    /// Tokens served from the provider's prompt cache.
    ///
    /// **Contract:** subset of `input_tokens`. That is, `input_tokens`
    /// includes cached tokens; `cached_input_tokens` is the portion of
    /// that count served from the cache. The non-cached portion is
    /// `input_tokens - cached_input_tokens`.
    ///
    /// This matches Gemini's `promptTokenCount` / `cachedContentTokenCount`
    /// and OpenAI's `prompt_tokens` / `prompt_tokens_details.cached_tokens`.
    /// Providers that report cache tokens additively (e.g. Anthropic's
    /// `cache_read_input_tokens`) must normalise inside the adapter so
    /// this invariant holds.
    pub cached_input_tokens: Option<u32>,
    /// Reasoning / thinking tokens, when the provider bills them
    /// separately from `output_tokens`.
    pub thinking_tokens: Option<u32>,
    /// Tokens consumed by tool-use prompt parts, when the provider bills
    /// them separately from `input_tokens` (Gemini's
    /// `toolUsePromptTokenCount`).
    pub tool_use_prompt_tokens: Option<u32>,
}
```

Five fields cover Gemini and Ollama with room for OpenAI/Anthropic. Fields explicitly **not** included:

- **Model identifier.** The runner already knows which model produced the event (it owns the `Arc<dyn LlmModel>`); attaching it to every `TokenUsage` would be redundant and would force every test double to invent one. Consumers attribute usage to a model via the runner they constructed.
- **Cost / pricing.** Out of scope — owned by downstream (`arnes`).
- **Total tokens.** Derivable as `input + output` when both are `Some`. Storing it invites drift between the sum and the reported total when providers round.

### 4.2 `ModelResponse` carries `TokenUsage` (modify `src/model.rs`)

```rust
pub struct ModelResponse {
    pub text: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub thinking: Option<String>,
    pub token_usage: Option<TokenUsage>,   // NEW
}
```

`Option<TokenUsage>` rather than `TokenUsage` because:
- Test doubles and the existing scripted `LlmModel` in `src/runner/tests.rs` shouldn't be forced to manufacture token counts.
- A provider that fails to report usage on a given response (transient API quirk) is honestly represented as `None` rather than a `TokenUsage` full of `None` fields.

### 4.3 `ModelStreamChunk::Usage` variant (modify `src/model.rs`)

```rust
pub enum ModelStreamChunk {
    Thinking(String),
    TextDelta(String),
    ToolCall(ToolCall),
    Usage(TokenUsage),   // NEW
}
```

Emitted at most once per `generate_stream` call, typically as the final chunk before the stream ends. Provider adapters that do not report usage simply never yield this variant.

The default `generate_stream` implementation forwards `response.token_usage` as a trailing `Usage` chunk if `Some`:

```rust
fn generate_stream(&self, request: ModelRequest) -> ... {
    Box::pin(async_stream::stream! {
        let response = self.generate(request).await?;
        if let Some(thinking) = response.thinking {
            yield Ok(ModelStreamChunk::Thinking(thinking));
        }
        for call in response.tool_calls {
            yield Ok(ModelStreamChunk::ToolCall(call));
        }
        if let Some(text) = response.text {
            yield Ok(ModelStreamChunk::TextDelta(text));
        }
        if let Some(token_usage) = response.token_usage {   // NEW
            yield Ok(ModelStreamChunk::Usage(token_usage));
        }
    })
}
```

### 4.4 `AgentEvent::Usage` variant (modify `src/runner/events.rs`)

```rust
pub enum AgentEvent {
    ToolCallStarted { name: String, args: serde_json::Value },
    ToolCallFinished { name: String, result: ToolCallResult },
    ThinkingDelta(String),
    TextDelta(String),
    Usage(TokenUsage),   // NEW
    Error(crate::error::Error),
}
```

The runner forwards every `ModelStreamChunk::Usage` it sees as `AgentEvent::Usage` — no aggregation, no synthesis. Semantics: **one `Usage` event per model call**. A run that performs `N` model calls (one initial + one after each batch of tool results) produces up to `N` `Usage` events. Consumers that want a per-run total sum across them.

**Rejected: `AgentEvent::TurnEnd { usage: TokenUsage }` as a terminal-per-turn event.** Tempting because it doubles as a "model call boundary" signal, but it conflates two concerns: turn boundaries are already inferrable from the absence of `ToolCallStarted`/`ToolCallFinished` after a `TextDelta`, and consumers that don't care about usage shouldn't have to handle an unrelated terminal event. Keeping `Usage` as a peer of the other model-output events is simpler.

### 4.5 Documentation

- Update `docs/SPEC.md`:
  - Add `TokenUsage` under "Core Types".
  - Add `usage` field to the `ModelResponse` listing.
  - Add `Usage(TokenUsage)` variant to the `ModelStreamChunk` and `AgentEvent` listings.
  - Note in the Gemini and Ollama adapter sections what dimensions they populate vs. leave `None`.
- Rustdoc on `TokenUsage` calls out the `None` vs `Some(0)` distinction.
- Add an `examples/token_usage.rs` that prints usage for each model call in a multi-turn tool-calling run, and mention it in `README.md`'s example list.

---

## 5. Provider Mapping

### 5.0 Field-by-field mapping

| `TokenUsage` field        | Gemini (`geologia@08ad998`)          | Ollama (`ollama-rs@08c2dcc`)             | Notes                                                                                       |
|---------------------------|--------------------------------------|------------------------------------------|---------------------------------------------------------------------------------------------|
| `input_tokens`            | `prompt_token_count` ✅              | `prompt_eval_count` ⚠️ (not yet on `ChatResponse`) | Ollama exposes it on `GenerateResponse` but not `ChatResponse` — needs upstream PR.        |
| `output_tokens`           | `candidates_token_count` ✅          | `eval_count` ⚠️ (not yet on `ChatResponse`)        | Same upstream gap as `input_tokens`.                                                        |
| `cached_input_tokens`     | ❌ not exposed (`cached_content_token_count` exists in API) | ❌ not in Ollama API | Geologia needs upstream PR ([geologia#11](https://github.com/andreban/geologia/issues/11)); Ollama API itself has no equivalent. |
| `thinking_tokens`         | ❌ not exposed (`thoughts_token_count` exists in API)       | ❌ not in Ollama API | Geologia needs upstream PR ([geologia#11](https://github.com/andreban/geologia/issues/11)); Ollama API itself has no equivalent. |
| `tool_use_prompt_tokens`  | ❌ not exposed (`tool_use_prompt_token_count` exists in API)| ❌ not in Ollama API | Geologia upstream gap — same PR as above can add it. Ollama API itself has no equivalent.   |
| *(derived)* total         | `total_token_count` available but unused | Not exposed                          | Intentionally not stored on `TokenUsage` — consumers compute `input + output` when needed. |

Gemini's `*_tokens_details` modality breakdowns and `serviceTier` are intentionally omitted — see §7 ("Out of Scope").

Legend: ✅ wired up today · ⚠️ exists in provider API but not in our Rust client · ❌ not exposed by the provider

### 5.1 Gemini (`src/models/gemini.rs`)

`geologia`'s `UsageMetadata` at commit `08ad998` (currently locked in `Cargo.lock`) exposes:

```rust
pub struct UsageMetadata {
    pub candidates_token_count: Option<u32>,
    pub prompt_token_count: Option<u32>,
    pub total_token_count: Option<u32>,
}
```

Mapping:
- `prompt_token_count` → `input_tokens`
- `candidates_token_count` → `output_tokens`
- `cached_input_tokens` → `None` (geologia does not expose `cached_content_token_count`)
- `thinking_tokens` → `None` (geologia does not expose `thoughts_token_count`)
- `tool_use_prompt_tokens` → `None` (geologia does not expose `tool_use_prompt_token_count`)

**Upstream follow-up:** [geologia#11](https://github.com/andreban/geologia/issues/11) is open to add `cached_content_token_count` and `thoughts_token_count`. The same PR should also add `tool_use_prompt_token_count` — update the issue (or comment) before it lands. Once merged, bump the dependency and wire all three through. Tracked as non-blocking — the main change ships with the two fields available today.

### 5.2 Ollama (`src/models/ollama.rs`)

The Ollama HTTP API returns `prompt_eval_count` and `eval_count` on the final chunk (`done: true`) of `/api/chat`. **`ollama-rs` at commit `08c2dcc` does not expose these on `ChatResponse`** — they're parsed for `/api/generate` but not for `/api/chat`.

This is a blocker for the Ollama side. Two options:

1. **Upstream PR to `ollama-rs`** adding `prompt_eval_count: Option<u64>` and `eval_count: Option<u64>` to `ChatResponse`. Bump the dep, then map `prompt_eval_count` → `input_tokens` (cast `u64` → `u32`, saturating) and `eval_count` → `output_tokens`. Cache/thinking left `None`.
2. **Ship Gemini-only first**, leave `OllamaModel::generate` returning `usage: None`, file a follow-up issue for the upstream `ollama-rs` change.

Recommended: **(1)**, because the `ollama-rs` change is small (two `Option<u64>` fields with serde) and shipping `OllamaModel` with `usage: None` would lock in the missing feature.

If the upstream PR slips, fall back to **(2)** so this issue isn't blocked.

---

## 6. Implementation Steps

1. **`TokenUsage` and `ModelResponse.token_usage`** (`src/model.rs`)
   - Add the struct with `Serialize`/`Deserialize`/`Default`/`PartialEq`/`Eq` derives.
   - Add `pub token_usage: Option<TokenUsage>` to `ModelResponse`.
   - Update every `ModelResponse { ... }` literal in the crate (Gemini adapter, Ollama adapter, runner tests, AgentTool tests, doctests) to set `token_usage: None` by default.

2. **`ModelStreamChunk::Usage`** (`src/model.rs`)
   - Add the variant.
   - Update the default `generate_stream` impl to yield `Usage` after the existing chunks if `response.token_usage.is_some()`.

3. **Gemini adapter** (`src/models/gemini.rs`)
   - Read `response.usage_metadata` after `generate_content` returns.
   - Build a `TokenUsage` and assign it to `ModelResponse.token_usage`.
   - Unit test: a candidate with a known `UsageMetadata` produces the expected `TokenUsage`.

4. **Ollama adapter** (`src/models/ollama.rs`)
   - Once `ollama-rs` exposes the fields: capture them on the final chunk (`done == true`) in both `generate` and `generate_stream`, set `ModelResponse.token_usage` / yield `ModelStreamChunk::Usage`. Cast `u64` to `u32` saturating.
   - Until then: emit `token_usage: None`. Add a `// TODO(#28)` comment pointing at the upstream blocker.

5. **`AgentEvent::Usage`** (`src/runner/events.rs`)
   - Add the variant with a rustdoc note that it fires once per model call.

6. **Runner forwarding** (`src/runner/mod.rs`)
   - In `main_loop`, match `ModelStreamChunk::Usage(u)` and forward as `AgentEvent::Usage(u)`. No aggregation.

7. **Tests**
   - Extend `src/runner/tests.rs` with a scripted `LlmModel` that emits a `TokenUsage` and assert the runner yields `AgentEvent::Usage` with the same values.
   - Multi-turn tool-calling test: assert one `AgentEvent::Usage` per model call, with the right ordering relative to tool events.
   - Backward-compat smoke test: a model that emits `usage: None` produces no `Usage` events and the rest of the stream is unchanged.

8. **Example**
   - `examples/token_usage.rs`: a Gemini run with one tool, printing each `AgentEvent::Usage` with its position in the loop. Add to `Cargo.toml` examples.

9. **Docs**
   - Update `docs/SPEC.md` (Core Types section, ModelResponse, ModelStreamChunk, AgentEvent listings, Gemini/Ollama provider notes).
   - Add `examples/token_usage.rs` to the README example list.
   - Update `skills/agent-rig.md` with a one-line note on consuming `AgentEvent::Usage`.

10. **Follow-up issues**
    - `geologia`: expose `cached_content_token_count` and `thoughts_token_count`.
    - `ollama-rs`: expose `prompt_eval_count` / `eval_count` on `ChatResponse` (if not already in flight).

---

## 7. Out of Scope

- Cost computation. Pricing lives in `arnes`.
- Cross-run aggregation. Consumers sum across `AgentEvent::Usage` themselves.
- A `TurnEnd` event. See §4.4 for the rationale.
- Adding a model identifier to `TokenUsage`. See §4.1 for the rationale.
- A distinct `cache_creation_input_tokens` field for Anthropic-style cache-write
  pricing. See §9 for the forward-looking note.

---

## 8. Resolved Decisions

- **Name:** `TokenUsage`. `TokenCounts` (matching arnes) considered but rejected — `TokenUsage` is the more common SDK idiom and arnes can adapt at the boundary.
- **Cast strategy for Ollama `u64` → `u32`:** saturating. Tokens-per-call exceeding `u32::MAX` (~4.2B) is implausible, so saturating is effectively lossless but defensive.
- **Cache semantics:** **subset.** `cached_input_tokens ⊆ input_tokens`. Documented on the field's rustdoc (§4.1). This matches Gemini (`promptTokenCount` includes cached) and OpenAI (`prompt_tokens_details.cached_tokens` ⊂ `prompt_tokens`). Pricing engines like `arnes` compute cost as `(input - cached) * input_rate + cached * cached_rate + output * output_rate`, with one stable formula across providers.

---

## 9. Future Considerations

- **Anthropic adapter and additive cache semantics.** Anthropic reports `cache_read_input_tokens` and `cache_creation_input_tokens` as separate counters from `input_tokens` (additive, not subset). When an Anthropic adapter lands, the provider adapter normalises by setting `input_tokens = api.input_tokens + cache_read + cache_creation` so the subset invariant in §4.1 holds. The cache-write tier (more expensive than fresh input) cannot be expressed by the current four input-tier model — adding `cache_creation_input_tokens: Option<u32>` to `TokenUsage` should be revisited at that time.
