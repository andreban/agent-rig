# Product Requirements Document — rust-agent-kit

## Overview

`rust-agent-kit` is a Rust library for building AI agents that is **provider-agnostic**: the same agent definition runs against any supported LLM backend (Google Gemini, Ollama, and more) without code changes.

## Problem

Building agentic applications in Rust today requires tight coupling to a specific provider's SDK. Switching providers, running locally vs. in the cloud, or comparing model quality means rewriting the integration layer. There is no de-facto standard abstraction in the Rust ecosystem comparable to Python's LangChain or the OpenAI Agents SDK.

## Goals

- **Unified agent API.** Define an agent once (name, instructions, model) and run it against any supported provider.
- **Easy provider swap.** Changing the LLM requires swapping one `Box<dyn LlmModel>` — nothing else changes.
- **Agentic loop support.** The runner handles the request/response loop, including future function-calling (tool use) cycles.
- **Low boilerplate.** Builder patterns for all major types keep call-site code concise and readable.
- **Testability.** The `LlmModel` trait can be implemented by test doubles, so agent logic can be unit-tested without network calls.

## Non-Goals

- This library does not provide a runtime, scheduler, or orchestration layer.
- It does not manage API keys or secrets.
- It is not a general-purpose HTTP client or SDK wrapper — it only exposes the abstractions needed for agentic use.

## Target Users

Rust developers building:
- Chatbots and assistants backed by cloud or local LLMs.
- Automated pipelines that use LLMs for reasoning or classification.
- Tools that need to run the same agent against multiple providers for evaluation or fallback.

## Success Criteria

- A developer can build and run a single-turn agent against Gemini or Ollama with fewer than 20 lines of application code.
- Switching between providers requires changing only the model constructor, not the agent or runner.
- All public types have documentation and tests.
