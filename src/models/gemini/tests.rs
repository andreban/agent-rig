use super::*;
use geologia::prelude::{Candidate, Content, Part, PartData, Role};

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
