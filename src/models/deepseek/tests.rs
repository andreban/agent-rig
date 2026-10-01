// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use super::*;
use crate::model::{Message, MessageList};
use cetologia::types::response::PromptTokensDetails;
use schemars::JsonSchema;
use serde_json::json;

fn empty_request() -> ModelRequest {
    ModelRequest {
        messages: MessageList::new(),
        system: None,
        output_schema: None,
        tools: vec![],
    }
}

#[derive(JsonSchema)]
#[allow(dead_code)]
struct TestSchema {
    answer: String,
}

#[test]
fn test_build_chat_request_empty() {
    let req = build_chat_request("deepseek-chat", None, None, None, None, empty_request());
    assert_eq!(req.model, "deepseek-chat");
    assert!(req.messages.is_empty());
}

#[test]
fn test_to_token_usage_with_prompt_cache_hit() {
    let usage = CetologiaUsage {
        prompt_tokens: 100,
        completion_tokens: 50,
        total_tokens: 150,
        prompt_cache_hit_tokens: Some(40),
        prompt_cache_miss_tokens: Some(60),
        prompt_tokens_details: None,
    };

    let token_usage = to_token_usage(&usage).expect("usage should be parsed");
    assert_eq!(token_usage.input_tokens, Some(100));
    assert_eq!(token_usage.output_tokens, Some(50));
    assert_eq!(token_usage.cached_input_tokens, Some(40));
}

#[test]
fn test_to_token_usage_with_prompt_tokens_details() {
    let usage = CetologiaUsage {
        prompt_tokens: 100,
        completion_tokens: 50,
        total_tokens: 150,
        prompt_cache_hit_tokens: None,
        prompt_cache_miss_tokens: None,
        prompt_tokens_details: Some(PromptTokensDetails {
            cached_tokens: Some(75),
        }),
    };

    let token_usage = to_token_usage(&usage).expect("usage should be parsed");
    assert_eq!(token_usage.cached_input_tokens, Some(75));
}

#[test]
fn test_build_chat_request_with_system_and_messages() {
    let req = ModelRequest {
        messages: MessageList::from(vec![
            Message::user("Hello DeepSeek"),
            Message::assistant("Hello human"),
        ]),
        system: Some("Act as a pirate".into()),
        output_schema: None,
        tools: vec![],
    };

    let chat_req = build_chat_request("deepseek-chat", Some(0.7), Some(100), Some(0.9), None, req);

    assert_eq!(chat_req.model, "deepseek-chat");
    assert_eq!(chat_req.temperature, Some(0.7));
    assert_eq!(chat_req.max_tokens, Some(100));
    assert_eq!(chat_req.top_p, Some(0.9));
    assert_eq!(chat_req.messages.len(), 3);
    assert_eq!(chat_req.messages[0].role, CetologiaRole::System);
    assert_eq!(
        chat_req.messages[0].content.as_deref(),
        Some("Act as a pirate")
    );
    assert_eq!(chat_req.messages[1].role, CetologiaRole::User);
    assert_eq!(chat_req.messages[2].role, CetologiaRole::Assistant);
}

#[test]
fn test_build_chat_request_with_tools_and_schema() {
    let schema = schemars::schema_for!(TestSchema);

    let tool_def = ToolDefinition {
        name: "get_weather".into(),
        description: "Fetch weather".into(),
        parameters: schemars::schema_for!(TestSchema),
    };

    let req = ModelRequest {
        messages: MessageList::from(vec![Message::user("What's the weather?")]),
        system: None,
        output_schema: Some(schema),
        tools: vec![tool_def],
    };

    let chat_req = build_chat_request("deepseek-chat", None, None, None, None, req);

    assert_eq!(chat_req.response_format, Some(ResponseFormat::JsonObject));
    assert_eq!(chat_req.tools.as_ref().unwrap().len(), 1);
    assert_eq!(
        chat_req.tools.as_ref().unwrap()[0].function.name,
        "get_weather"
    );
}

#[test]
fn test_build_chat_request_with_tool_calls_and_results() {
    let call = Arc::new(ToolCall {
        id: "call_123".into(),
        name: "test_func".into(),
        args: json!({ "x": 1 }),
        provider_metadata: None,
    });

    let req = ModelRequest {
        messages: MessageList::from(vec![
            Message::tool_calls(vec![Arc::clone(&call)]),
            Message {
                role: Role::User,
                content: MessageContent::ToolResult {
                    tool_call: Arc::clone(&call),
                    result: json!({ "success": true }),
                },
                provider_metadata: None,
            },
        ]),
        system: None,
        output_schema: None,
        tools: vec![],
    };

    let chat_req = build_chat_request("deepseek-chat", None, None, None, None, req);

    assert_eq!(chat_req.messages.len(), 2);
    assert_eq!(chat_req.messages[0].role, CetologiaRole::Assistant);
    assert_eq!(chat_req.messages[0].tool_calls.as_ref().unwrap().len(), 1);
    assert_eq!(
        chat_req.messages[0].tool_calls.as_ref().unwrap()[0].id,
        "call_123"
    );

    assert_eq!(chat_req.messages[1].role, CetologiaRole::Tool);
    assert_eq!(
        chat_req.messages[1].tool_call_id.as_deref(),
        Some("call_123")
    );
}

#[test]
fn test_deepseek_model_accessors_and_reexport() {
    let model = DeepSeekModel::new("test-key", "deepseek-chat");
    assert_eq!(model.model(), "deepseek-chat");
    let _client: &cetologia::prelude::CetologiaClient = model.client();

    // Verify the re-exported cetologia crate is directly usable
    let _req = cetologia::prelude::ChatCompletionRequest::builder("deepseek-chat").build();
}

#[test]
fn test_build_chat_request_replays_reasoning_on_assistant_turns() {
    let call = Arc::new(ToolCall::new("call_1".into(), "f".into(), json!({})));

    let req = ModelRequest {
        messages: MessageList::from(vec![
            Message::user("hi"),
            Message::tool_calls(vec![Arc::clone(&call)])
                .with_provider_metadata(reasoning_metadata("tool reasoning").unwrap()),
            Message::tool_result(Arc::clone(&call), json!({ "ok": true })),
            Message::assistant("done")
                .with_provider_metadata(reasoning_metadata("text reasoning").unwrap()),
            Message::assistant("no metadata"),
            // Another provider's state is ignored, not misreplayed.
            Message::assistant("foreign")
                .with_provider_metadata(json!({ "other": { "reasoning_content": "x" } })),
        ]),
        system: None,
        output_schema: None,
        tools: vec![],
    };

    let chat_req = build_chat_request("deepseek-reasoner", None, None, None, None, req);

    assert_eq!(chat_req.messages.len(), 6);
    assert_eq!(chat_req.messages[0].reasoning_content, None);
    assert_eq!(
        chat_req.messages[1].reasoning_content.as_deref(),
        Some("tool reasoning")
    );
    assert_eq!(chat_req.messages[2].reasoning_content, None);
    assert_eq!(chat_req.messages[3].content.as_deref(), Some("done"));
    assert_eq!(
        chat_req.messages[3].reasoning_content.as_deref(),
        Some("text reasoning")
    );
    assert_eq!(chat_req.messages[4].reasoning_content, None);
    assert_eq!(chat_req.messages[5].reasoning_content, None);
}

#[test]
fn test_reasoning_metadata_shape() {
    assert_eq!(
        reasoning_metadata("why"),
        Some(json!({ "deepseek": { "reasoning_content": "why" } }))
    );
    assert_eq!(reasoning_metadata(""), None);
}

#[test]
fn test_build_chat_request_reasoning_effort() {
    let req = build_chat_request(
        "deepseek-reasoner",
        None,
        None,
        None,
        Some(ReasoningEffort::None),
        empty_request(),
    );
    assert_eq!(req.reasoning_effort, Some(ReasoningEffort::None));
    let body = serde_json::to_value(&req).unwrap();
    assert_eq!(body["reasoning_effort"], "none");

    let req = build_chat_request("deepseek-reasoner", None, None, None, None, empty_request());
    let body = serde_json::to_value(&req).unwrap();
    assert!(body.get("reasoning_effort").is_none());
}
