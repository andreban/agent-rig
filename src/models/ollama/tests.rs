// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

use super::*;
use std::sync::Arc;

use crate::model::{Media, Message, MessageList};
use ollama_rs::types::chat::{Message as OllamaMessage, Role as OllamaRole};

fn chat_response(prompt: Option<u64>, eval: Option<u64>) -> ChatResponse {
    ChatResponse {
        model: "test-model".to_string(),
        created_at: "2026-06-02T00:00:00Z".to_string(),
        message: OllamaMessage {
            content: String::new(),
            role: OllamaRole::Assistant,
            thinking: None,
            tool_calls: vec![],
        },
        done: true,
        done_reason: None,
        total_duration: None,
        load_duration: None,
        prompt_eval_count: prompt,
        prompt_eval_duration: None,
        eval_count: eval,
        eval_duration: None,
    }
}

#[test]
fn to_token_usage_maps_prompt_and_eval_counts() {
    let response = chat_response(Some(123), Some(45));
    let usage = to_token_usage(&response).expect("usage present");
    assert_eq!(usage.input_tokens, Some(123));
    assert_eq!(usage.output_tokens, Some(45));
    assert_eq!(usage.cached_input_tokens, None);
    assert_eq!(usage.thinking_tokens, None);
    assert_eq!(usage.tool_use_prompt_tokens, None);
}

#[test]
fn to_token_usage_returns_none_when_both_counts_absent() {
    let response = chat_response(None, None);
    assert!(to_token_usage(&response).is_none());
}

#[test]
fn to_token_usage_returns_some_when_only_one_count_present() {
    let response = chat_response(Some(10), None);
    let usage = to_token_usage(&response).expect("usage present");
    assert_eq!(usage.input_tokens, Some(10));
    assert_eq!(usage.output_tokens, None);
}

#[test]
fn saturating_u64_to_u32_caps_at_max() {
    assert_eq!(saturating_u64_to_u32(0), 0);
    assert_eq!(saturating_u64_to_u32(42), 42);
    assert_eq!(saturating_u64_to_u32(u32::MAX as u64), u32::MAX);
    assert_eq!(saturating_u64_to_u32(u32::MAX as u64 + 1), u32::MAX);
    assert_eq!(saturating_u64_to_u32(u64::MAX), u32::MAX);
}

fn empty_request() -> ModelRequest {
    ModelRequest {
        messages: Default::default(),
        system: None,
        output_schema: None,
        tools: vec![],
    }
}

#[test]
fn build_chat_request_propagates_think_config() {
    let req = build_chat_request(
        "test-model",
        None,
        Some(Think::Level(ThinkLevel::High)),
        empty_request(),
    )
    .unwrap();
    assert!(matches!(req.think, Some(Think::Level(ThinkLevel::High))));
}

#[test]
fn build_chat_request_leaves_think_none_when_unset() {
    let req = build_chat_request("test-model", None, None, empty_request()).unwrap();
    assert!(req.think.is_none());
}

#[test]
fn test_ollama_model_accessors_and_reexport() {
    let model = OllamaModel::new("http://localhost:11434", "llama3.2");
    assert_eq!(model.model(), "llama3.2");
    let _client: &ollama_rs::OllamaClient = model.client();

    // Verify the re-exported ollama_rs crate is directly usable
    let _options = ollama_rs::types::common::Options::builder()
        .temperature(0.7)
        .build();
}

/// A request whose only message is a tool result carrying one image.
fn request_with_tool_result_media() -> ModelRequest {
    let call = Arc::new(ToolCall::new(
        "call-1".into(),
        "take_screenshot".into(),
        serde_json::json!({}),
    ));
    let image = Media {
        mime_type: "image/png".into(),
        data: "aGVsbG8=".into(),
    };
    ModelRequest {
        messages: MessageList::from(vec![Message::tool_result_with_media(
            call,
            serde_json::json!({"ok": true}),
            vec![image],
        )]),
        system: None,
        output_schema: None,
        tools: vec![],
    }
}

#[tokio::test]
async fn generate_rejects_tool_result_media() {
    let model = OllamaModel::new("http://localhost:11434", "llama3.2");

    let result = model.generate(request_with_tool_result_media()).await;

    assert!(
        matches!(&result, Err(Error::Provider(msg)) if msg == TOOL_RESULT_MEDIA_UNSUPPORTED),
        "{result:?}"
    );
}

#[tokio::test]
async fn generate_stream_rejects_tool_result_media() {
    let model = OllamaModel::new("http://localhost:11434", "llama3.2");

    let first = model
        .generate_stream(request_with_tool_result_media())
        .next()
        .await;

    assert!(
        matches!(&first, Some(Err(Error::Provider(msg))) if msg == TOOL_RESULT_MEDIA_UNSUPPORTED),
        "{first:?}"
    );
}
