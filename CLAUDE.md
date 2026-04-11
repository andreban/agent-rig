# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`rust-agent-kit` is a Rust library (edition 2024) for building provider-agnostic AI agents. Supported providers: Google Gemini and Ollama. See `docs/PRD.md` for requirements and `docs/SPEC.md` for the technical specification.

## Working with this project

Before planning any non-trivial task, read `docs/PRD.md` and `docs/SPEC.md`. After implementing changes that affect requirements or design, update the relevant document to reflect the new state.

## Commands

```bash
# Build
cargo build

# Run
cargo run

# Test
cargo test

# Run a single test
cargo test <test_name>

# Release build
cargo build --release
```

## Dependencies

- **google-genai**: Rust client for the Google Generative AI API. Sourced from a private git repo (`https://git.bandarra.me/andreban/google-genai.git`) — not on crates.io.
- Requires `GEMINI_API_KEY` environment variable (use a `.env` file with `dotenvy`).

## google-genai API

Import from `google_genai::prelude::*`. Key types:

- `GeminiClient::new(api_key: String)` — main client
- `Content::builder().role(Role::User).add_text_part("...").build()` — message builder
- `GenerateContentRequest::builder().contents(vec![...]).build()` — request builder
- `client.generate_content(&request, "gemini-2.5-pro-preview").await?` — async call
- `response.candidates[0].get_text()` — extract response text

**Function calling** (for agentic tools):
- `FunctionDeclaration` — define a tool with `name`, `description`, `parameters_json_schema`, `response_json_schema`
- `Tools { function_declarations: Some(vec![...]), ..Default::default() }` — wrap declarations
- Pass `tools(vec![tools])` on the request builder
- Response parts use `PartData::FunctionCall { id, name, args }` — match on this to detect tool calls
- Reply with `PartData::FunctionResponse(FunctionResponse { id, name, response, .. })` — return tool results
- Loop until no more function calls in the response (agentic loop pattern)

**Other capabilities**: streaming (`generate_content_stream`), image generation, token counting, text embeddings.

## Code Standards

- All public items must have rustdoc comments (`///`). Include examples in doc comments where useful.
- Code should be designed for testability. Both unit tests (in `#[cfg(test)]` modules) and integration tests (in `tests/`) are expected.

## Architecture

Library crate. Core abstractions in `src/`: `LlmModel` trait (`model.rs`), `Agent` + `AgentBuilder` (`agent.rs`), `AgentRunner` (`runner.rs`), `Error` (`error.rs`). Provider adapters in `src/models/`: `GeminiModel` and `OllamaModel`. See `docs/SPEC.md` for full details.
