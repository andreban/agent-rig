use super::*;
use geologia::prelude::{Candidate, Content, Part, PartData, Role};
use serde_json::json;

fn make_candidate(parts: Vec<Part>) -> Candidate {
    Candidate {
        content: Some(Content {
            role: Some(Role::Model),
            parts: Some(parts),
        }),
        finish_reason: None,
        citation_metadata: None,
        safety_ratings: None,
        index: 0,
    }
}

fn text_part(text: &str) -> Part {
    Part {
        data: PartData::Text(text.to_string()),
        thought: None,
        thought_signature: None,
        part_metadata: None,
        media_resolution: None,
    }
}

fn thought_part(text: &str) -> Part {
    Part {
        data: PartData::Text(text.to_string()),
        thought: Some(true),
        thought_signature: None,
        part_metadata: None,
        media_resolution: None,
    }
}

fn function_call_part(id: &str, name: &str, args: serde_json::Value) -> Part {
    Part {
        data: PartData::FunctionCall {
            id: Some(id.to_string()),
            name: name.to_string(),
            args: Some(args),
        },
        thought: None,
        thought_signature: None,
        part_metadata: None,
        media_resolution: None,
    }
}

#[test]
fn extract_separates_thought_and_text_parts() {
    let candidate = make_candidate(vec![thought_part("hmm..."), text_part("The answer is 42.")]);
    let (thinking, text) = extract_text_and_thinking(&candidate);
    assert_eq!(thinking.as_deref(), Some("hmm..."));
    assert_eq!(text.as_deref(), Some("The answer is 42."));
}

#[test]
fn extract_concatenates_multiple_parts() {
    let candidate = make_candidate(vec![
        thought_part("step 1 "),
        thought_part("step 2"),
        text_part("part a "),
        text_part("part b"),
    ]);
    let (thinking, text) = extract_text_and_thinking(&candidate);
    assert_eq!(thinking.as_deref(), Some("step 1 step 2"));
    assert_eq!(text.as_deref(), Some("part a part b"));
}

#[test]
fn extract_returns_none_when_no_thinking() {
    let candidate = make_candidate(vec![text_part("hello")]);
    let (thinking, text) = extract_text_and_thinking(&candidate);
    assert!(thinking.is_none());
    assert_eq!(text.as_deref(), Some("hello"));
}

#[test]
fn token_usage_from_usage_metadata_maps_all_fields() {
    let meta = UsageMetadata {
        prompt_token_count: Some(123),
        candidates_token_count: Some(45),
        cached_content_token_count: Some(60),
        thoughts_token_count: Some(15),
        tool_use_prompt_token_count: Some(7),
        total_token_count: Some(190),
        ..Default::default()
    };
    let usage = TokenUsage::from(&meta);
    assert_eq!(usage.input_tokens, Some(123));
    assert_eq!(usage.output_tokens, Some(45));
    assert_eq!(usage.cached_input_tokens, Some(60));
    assert_eq!(usage.thinking_tokens, Some(15));
    assert_eq!(usage.tool_use_prompt_tokens, Some(7));
}

#[test]
fn token_usage_from_usage_metadata_passes_through_none() {
    let meta = UsageMetadata::default();
    let usage = TokenUsage::from(&meta);
    assert_eq!(usage.input_tokens, None);
    assert_eq!(usage.output_tokens, None);
    assert_eq!(usage.cached_input_tokens, None);
    assert_eq!(usage.thinking_tokens, None);
    assert_eq!(usage.tool_use_prompt_tokens, None);
}

#[test]
fn extract_returns_none_for_empty_content() {
    let candidate = Candidate {
        content: None,
        finish_reason: None,
        citation_metadata: None,
        safety_ratings: None,
        index: 0,
    };
    let (thinking, text) = extract_text_and_thinking(&candidate);
    assert!(thinking.is_none());
    assert!(text.is_none());
}

#[test]
fn stream_chunks_emits_thinking_then_text_in_part_order() {
    let candidate = make_candidate(vec![
        thought_part("reasoning..."),
        text_part("answer"),
    ]);
    let chunks = stream_chunks_from_candidate(&candidate);
    assert_eq!(chunks.len(), 2);
    assert!(matches!(&chunks[0], ModelStreamChunk::Thinking(t) if t == "reasoning..."));
    assert!(matches!(&chunks[1], ModelStreamChunk::TextDelta(t) if t == "answer"));
}

#[test]
fn stream_chunks_emits_one_chunk_per_part() {
    let candidate = make_candidate(vec![
        text_part("hello "),
        text_part("world"),
    ]);
    let chunks = stream_chunks_from_candidate(&candidate);
    assert_eq!(chunks.len(), 2);
    assert!(matches!(&chunks[0], ModelStreamChunk::TextDelta(t) if t == "hello "));
    assert!(matches!(&chunks[1], ModelStreamChunk::TextDelta(t) if t == "world"));
}

#[test]
fn stream_chunks_skips_empty_text_parts() {
    let candidate = make_candidate(vec![text_part(""), text_part("hi")]);
    let chunks = stream_chunks_from_candidate(&candidate);
    assert_eq!(chunks.len(), 1);
    assert!(matches!(&chunks[0], ModelStreamChunk::TextDelta(t) if t == "hi"));
}

#[test]
fn stream_chunks_emits_function_calls() {
    let candidate = make_candidate(vec![function_call_part(
        "call-1",
        "lookup",
        json!({"q": "rust"}),
    )]);
    let chunks = stream_chunks_from_candidate(&candidate);
    assert_eq!(chunks.len(), 1);
    match &chunks[0] {
        ModelStreamChunk::ToolCall(tc) => {
            assert_eq!(tc.id, "call-1");
            assert_eq!(tc.name, "lookup");
            assert_eq!(tc.args, json!({"q": "rust"}));
            assert!(tc.provider_metadata.is_none());
        }
        other => panic!("expected ToolCall, got {other:?}"),
    }
}

#[test]
fn stream_chunks_preserves_thought_signature_on_tool_calls() {
    let mut part = function_call_part("call-1", "lookup", json!({}));
    part.thought_signature = Some("sig-abc".to_string());
    let candidate = make_candidate(vec![part]);
    let chunks = stream_chunks_from_candidate(&candidate);
    match &chunks[0] {
        ModelStreamChunk::ToolCall(tc) => {
            let meta = tc.provider_metadata.as_ref().expect("provider_metadata");
            assert_eq!(meta["thought_signature"], "sig-abc");
        }
        other => panic!("expected ToolCall, got {other:?}"),
    }
}

#[test]
fn stream_chunks_returns_empty_for_candidate_without_content() {
    let candidate = Candidate {
        content: None,
        finish_reason: None,
        citation_metadata: None,
        safety_ratings: None,
        index: 0,
    };
    assert!(stream_chunks_from_candidate(&candidate).is_empty());
}

#[test]
fn stream_chunks_interleaves_thinking_text_and_tool_calls() {
    let candidate = make_candidate(vec![
        thought_part("planning"),
        text_part("Calling tool now"),
        function_call_part("c1", "fetch", json!({"url": "x"})),
    ]);
    let chunks = stream_chunks_from_candidate(&candidate);
    assert_eq!(chunks.len(), 3);
    assert!(matches!(&chunks[0], ModelStreamChunk::Thinking(_)));
    assert!(matches!(&chunks[1], ModelStreamChunk::TextDelta(_)));
    assert!(matches!(&chunks[2], ModelStreamChunk::ToolCall(_)));
}
