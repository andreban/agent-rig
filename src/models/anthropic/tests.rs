// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use super::*;
use crate::model::Media;
use crate::model::{Message, MessageList};
use futures_util::StreamExt;
use schemars::JsonSchema;
use serde_json::json;
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[derive(JsonSchema)]
#[allow(dead_code)]
struct TestSchema {
    answer: String,
}

fn request(messages: Vec<Message>) -> ModelRequest {
    ModelRequest {
        messages: MessageList::from(messages),
        system: None,
        output_schema: None,
        tools: vec![],
    }
}

fn weather_call() -> Arc<ToolCall> {
    Arc::new(ToolCall::new(
        "toolu_1".into(),
        "get_weather".into(),
        json!({ "city": "Paris" }),
    ))
}

#[test]
fn test_build_request_with_system_tools_schema_and_settings() {
    let req = ModelRequest {
        messages: MessageList::from(vec![Message::user("Weather?")]),
        system: Some("Be brief.".into()),
        output_schema: Some(schemars::schema_for!(TestSchema)),
        tools: vec![ToolDefinition {
            name: "get_weather".into(),
            description: "Fetch weather".into(),
            parameters: schemars::schema_for!(TestSchema),
        }],
    };
    let settings = Settings {
        max_tokens: Some(1000),
        effort: Some(Effort::Low),
        ..Default::default()
    };

    let body =
        serde_json::to_value(build_messages_request("claude-opus-5-5", &settings, req)).unwrap();

    assert_eq!(body["model"], "claude-opus-5-5");
    assert_eq!(body["max_tokens"], 1000);
    assert_eq!(body["system"], "Be brief.");
    assert_eq!(body["tools"][0]["name"], "get_weather");
    assert_eq!(body["tools"][0]["input_schema"]["type"], "object");
    assert_eq!(body["output_config"]["effort"], "low");
    let schema = &body["output_config"]["format"]["schema"];
    assert_eq!(schema["additionalProperties"], false);
    assert!(schema.get("$schema").is_none());
}

#[test]
fn test_tool_results_merge_into_one_user_turn_and_flag_errors() {
    let second = Arc::new(ToolCall::new(
        "toolu_2".into(),
        "get_weather".into(),
        json!({}),
    ));
    let req = request(vec![
        Message::user("Weather in Paris and Rome?"),
        Message::tool_calls(vec![weather_call(), Arc::clone(&second)]),
        Message::tool_result(weather_call(), json!({ "success": "20C" })),
        Message::tool_result(second, json!({ "error": "unknown city" })),
    ]);

    let claude = build_messages_request("claude-opus-5-5", &Settings::default(), req);

    assert_eq!(claude.messages.len(), 3);
    assert_eq!(claude.messages[1].role, ClaudeRole::Assistant);
    assert_eq!(claude.messages[1].content.len(), 2);
    assert_eq!(claude.messages[2].role, ClaudeRole::User);
    let results = serde_json::to_value(&claude.messages[2].content).unwrap();
    assert_eq!(
        results,
        json!([
            { "type": "tool_result", "tool_use_id": "toolu_1", "content": "{\"success\":\"20C\"}" },
            { "type": "tool_result", "tool_use_id": "toolu_2", "content": "{\"error\":\"unknown city\"}", "is_error": true }
        ])
    );
}

#[test]
fn test_assistant_turns_replay_stored_content_blocks() {
    let stored = vec![
        ContentBlock::Thinking {
            thinking: String::new(),
            signature: "sig".into(),
        },
        ContentBlock::tool_use("toolu_1", "get_weather", json!({ "city": "Paris" })),
    ];
    let turn = Message::tool_calls(vec![weather_call()])
        .with_provider_metadata(content_metadata(&stored).unwrap());

    let claude = build_messages_request(
        "claude-opus-5-5",
        &Settings::default(),
        request(vec![Message::user("Weather?"), turn]),
    );

    assert_eq!(claude.messages[1].content, stored);
}

#[test]
fn test_other_providers_metadata_is_ignored() {
    let turn = Message::assistant("Paris.")
        .with_provider_metadata(json!({ "deepseek": { "reasoning_content": "..." } }));

    let claude =
        build_messages_request("claude-opus-5-5", &Settings::default(), request(vec![turn]));

    assert_eq!(
        claude.messages[0].content,
        vec![ContentBlock::text("Paris.")]
    );
}

fn response_json(content: Value, stop_reason: &str) -> Value {
    json!({
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
        "content": content, "stop_reason": stop_reason, "stop_sequence": null,
        "usage": { "input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 90, "cache_creation_input_tokens": 0 }
    })
}

#[test]
fn test_model_response_with_tool_calls_keeps_blocks_and_normalizes_usage() {
    let response: MessagesResponse = serde_json::from_value(response_json(
        json!([
            { "type": "thinking", "thinking": "Look it up.", "signature": "sig" },
            { "type": "text", "text": "Checking." },
            { "type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": { "city": "Paris" } }
        ]),
        "tool_use",
    ))
    .unwrap();

    let model_response = model_response(response).unwrap();

    assert_eq!(model_response.text, None);
    assert_eq!(model_response.tool_calls.len(), 1);
    assert_eq!(
        model_response.tool_calls[0].args,
        json!({ "city": "Paris" })
    );
    assert_eq!(model_response.thinking.as_deref(), Some("Look it up."));
    let metadata = model_response.provider_metadata.unwrap();
    assert_eq!(metadata["anthropic"]["content"][0]["signature"], "sig");
    let usage = model_response.token_usage.unwrap();
    assert_eq!(usage.input_tokens, Some(100));
    assert_eq!(usage.cached_input_tokens, Some(90));
    assert_eq!(usage.output_tokens, Some(5));
}

#[test]
fn test_refusal_is_an_error() {
    let response: MessagesResponse =
        serde_json::from_value(response_json(json!([]), "refusal")).unwrap();

    assert!(matches!(model_response(response), Err(Error::Provider(_))));
}

#[tokio::test]
async fn test_generate_calls_messages_endpoint() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(body_partial_json(json!({ "model": "claude-opus-5-5" })))
        .respond_with(ResponseTemplate::new(200).set_body_json(response_json(
            json!([{ "type": "text", "text": "Paris." }]),
            "end_turn",
        )))
        .expect(1)
        .mount(&server)
        .await;
    let model = AnthropicModel::builder("key", "claude-opus-5-5")
        .base_url(server.uri())
        .build();

    let response = model
        .generate(request(vec![Message::user("Capital of France?")]))
        .await
        .unwrap();

    assert_eq!(response.text.as_deref(), Some("Paris."));
}

#[tokio::test]
async fn test_generate_stream_emits_text_tool_calls_metadata_and_usage() {
    let events = [
        json!({ "type": "message_start", "message": {
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": [], "stop_reason": null, "stop_sequence": null,
            "usage": { "input_tokens": 10, "output_tokens": 1 } } }),
        json!({ "type": "content_block_start", "index": 0, "content_block": { "type": "text", "text": "" } }),
        json!({ "type": "content_block_delta", "index": 0, "delta": { "type": "text_delta", "text": "Checking." } }),
        json!({ "type": "content_block_stop", "index": 0 }),
        json!({ "type": "content_block_start", "index": 1, "content_block": { "type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {} } }),
        json!({ "type": "content_block_delta", "index": 1, "delta": { "type": "input_json_delta", "partial_json": "{\"city\":\"Paris\"}" } }),
        json!({ "type": "content_block_stop", "index": 1 }),
        json!({ "type": "message_delta", "delta": { "stop_reason": "tool_use", "stop_sequence": null }, "usage": { "output_tokens": 20 } }),
        json!({ "type": "message_stop" }),
    ];
    let body: String = events
        .iter()
        .map(|e| format!("event: {}\ndata: {e}\n\n", e["type"].as_str().unwrap()))
        .collect();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(body_partial_json(json!({ "stream": true })))
        .respond_with(ResponseTemplate::new(200).set_body_raw(body, "text/event-stream"))
        .mount(&server)
        .await;
    let model = AnthropicModel::builder("key", "claude-opus-5-5")
        .base_url(server.uri())
        .build();

    let chunks: Vec<ModelStreamChunk> = model
        .generate_stream(request(vec![Message::user("Weather in Paris?")]))
        .map(|chunk| chunk.unwrap())
        .collect()
        .await;

    assert!(matches!(&chunks[0], ModelStreamChunk::TextDelta(t) if t == "Checking."));
    assert!(
        matches!(&chunks[1], ModelStreamChunk::ToolCall(c) if c.id == "toolu_1" && c.args == json!({ "city": "Paris" }))
    );
    assert!(
        matches!(&chunks[2], ModelStreamChunk::ProviderMetadata(m) if m["anthropic"]["content"][1]["type"] == "tool_use")
    );
    assert!(
        matches!(&chunks[3], ModelStreamChunk::Usage(u) if u.input_tokens == Some(10) && u.output_tokens == Some(20))
    );
    assert_eq!(chunks.len(), 4);
}

#[test]
fn test_anthropic_model_accessors_and_reexport() {
    let model = AnthropicModel::new("test-key", "claude-opus-5-5");
    assert_eq!(model.model(), "claude-opus-5-5");
    let _client: &antropologia::prelude::AntropologiaClient = model.client();

    // The re-exported antropologia crate is directly usable.
    let _req = antropologia::prelude::MessagesRequest::builder("claude-opus-5-5").build();
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
    let model = AnthropicModel::builder("test-key", "claude-opus-5-5").build();

    let result = model.generate(request_with_tool_result_media()).await;

    assert!(
        matches!(&result, Err(Error::Provider(msg)) if msg == TOOL_RESULT_MEDIA_UNSUPPORTED),
        "{result:?}"
    );
}

#[tokio::test]
async fn generate_stream_rejects_tool_result_media() {
    let model = AnthropicModel::builder("test-key", "claude-opus-5-5").build();

    let first = model
        .generate_stream(request_with_tool_result_media())
        .next()
        .await;

    assert!(
        matches!(&first, Some(Err(Error::Provider(msg))) if msg == TOOL_RESULT_MEDIA_UNSUPPORTED),
        "{first:?}"
    );
}
