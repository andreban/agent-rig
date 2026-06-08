use super::*;
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
        messages: vec![],
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
