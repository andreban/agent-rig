// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use super::*;
use crate::model::Media;
use crate::model::{Message, MessageList};
use cetologia::types::response::PromptTokensDetails;
use schemars::JsonSchema;
use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

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
    let req = build_chat_request("deepseek-chat", None, None, None, None, empty_request()).unwrap();
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

    let chat_req =
        build_chat_request("deepseek-chat", Some(0.7), Some(100), Some(0.9), None, req).unwrap();

    assert_eq!(chat_req.model, "deepseek-chat");
    assert_eq!(chat_req.temperature, Some(0.7));
    assert_eq!(chat_req.max_tokens, Some(100));
    assert_eq!(chat_req.top_p, Some(0.9));
    assert_eq!(chat_req.messages.len(), 3);
    assert_eq!(chat_req.messages[0].role, CetologiaRole::System);
    assert_eq!(chat_req.messages[0].text(), Some("Act as a pirate"));
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

    let chat_req = build_chat_request("deepseek-chat", None, None, None, None, req).unwrap();

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
            Message::tool_result(Arc::clone(&call), json!({ "success": true })),
        ]),
        system: None,
        output_schema: None,
        tools: vec![],
    };

    let chat_req = build_chat_request("deepseek-chat", None, None, None, None, req).unwrap();

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
    assert_eq!(chat_req.messages[1].text(), Some(r#"{"success":true}"#));
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

    let chat_req = build_chat_request("deepseek-reasoner", None, None, None, None, req).unwrap();

    assert_eq!(chat_req.messages.len(), 6);
    assert_eq!(chat_req.messages[0].reasoning_content, None);
    assert_eq!(
        chat_req.messages[1].reasoning_content.as_deref(),
        Some("tool reasoning")
    );
    assert_eq!(chat_req.messages[2].reasoning_content, None);
    assert_eq!(chat_req.messages[3].text(), Some("done"));
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
    )
    .unwrap();
    assert_eq!(req.reasoning_effort, Some(ReasoningEffort::None));
    let body = serde_json::to_value(&req).unwrap();
    assert_eq!(body["reasoning_effort"], "none");

    let req =
        build_chat_request("deepseek-reasoner", None, None, None, None, empty_request()).unwrap();
    let body = serde_json::to_value(&req).unwrap();
    assert!(body.get("reasoning_effort").is_none());
}

/// A request whose only message is a tool result carrying one image.
fn request_with_tool_result_media() -> ModelRequest {
    let call = Arc::new(ToolCall::new(
        "call-1".into(),
        "take_screenshot".into(),
        serde_json::json!({}),
    ));
    let image = Media::inline("image/png", "aGVsbG8=");
    ModelRequest {
        messages: MessageList::from(vec![Message::tool_result(call, image)]),
        system: None,
        output_schema: None,
        tools: vec![],
    }
}

#[test]
fn test_build_chat_request_tool_result_with_media() {
    for model in [
        "deepseek-flash",
        "deepseek-flash-2026-10",
        "deepseek-chat",
        "deepseek-reasoner",
    ] {
        let req = build_chat_request(
            model,
            None,
            None,
            None,
            None,
            request_with_tool_result_media(),
        )
        .unwrap();
        let body = serde_json::to_value(&req).unwrap();

        assert_eq!(
            body["messages"],
            json!([
                {
                    "role": "tool",
                    "tool_call_id": "call-1",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,aGVsbG8="
                            }
                        }
                    ]
                }
            ])
        );
    }
}

#[test]
fn test_build_chat_request_tool_result_with_json_and_media() {
    let call = Arc::new(ToolCall::new(
        "call-1".into(),
        "take_screenshot".into(),
        json!({}),
    ));
    let req = ModelRequest {
        messages: MessageList::from(vec![Message::tool_result(
            call,
            vec![
                ToolResultContent::from(json!({"caption": "login page"})),
                ToolResultContent::from(Media::inline("image/jpeg", "ff00")),
            ],
        )]),
        system: None,
        output_schema: None,
        tools: vec![],
    };

    let chat_req = build_chat_request("deepseek-chat", None, None, None, None, req).unwrap();
    let body = serde_json::to_value(&chat_req).unwrap();

    assert_eq!(
        body["messages"],
        json!([
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": [
                    {
                        "type": "text",
                        "text": r#"{"caption":"login page"}"#
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64,ff00"
                        }
                    }
                ]
            }
        ])
    );
}

#[test]
fn test_build_chat_request_rejects_unsupported_media_type() {
    let call = Arc::new(ToolCall::new("call-1".into(), "f".into(), json!({})));
    let req = ModelRequest {
        messages: MessageList::from(vec![Message::tool_result(
            call,
            Media::inline("image/bmp", "aGVsbG8="),
        )]),
        system: None,
        output_schema: None,
        tools: vec![],
    };

    let result = build_chat_request("deepseek-chat", None, None, None, None, req);

    assert!(
        matches!(
            &result,
            Err(Error::Provider(msg)) if msg.contains("image/bmp")
        ),
        "{result:?}"
    );
}

#[tokio::test]
async fn generate_rejects_tool_result_media_for_text_only_model() {
    let model = DeepSeekModel::new("test-key", "deepseek-v4-pro");

    let result = model.generate(request_with_tool_result_media()).await;

    assert!(
        matches!(
            &result,
            Err(Error::Provider(msg)) if msg.contains("deepseek-v4-pro")
        ),
        "{result:?}"
    );
}

#[tokio::test]
async fn generate_stream_rejects_tool_result_media_for_text_only_model() {
    let model = DeepSeekModel::new("test-key", "deepseek-v4-pro");

    let first = model
        .generate_stream(request_with_tool_result_media())
        .next()
        .await;

    assert!(
        matches!(
            &first,
            Some(Err(Error::Provider(msg))) if msg.contains("deepseek-v4-pro")
        ),
        "{first:?}"
    );
}

/// A model whose API is a mock server answering every chat request with
/// `status` and a DeepSeek error body.
async fn model_answering_with(status: u16) -> (MockServer, DeepSeekModel) {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(status).set_body_json(json!({
            "error": { "message": "error", "type": "error", "code": null }
        })))
        .mount(&server)
        .await;
    let model = DeepSeekModel::builder("test-key", "deepseek-chat")
        .base_url(server.uri())
        .build();
    (server, model)
}

#[tokio::test]
async fn rate_limits_and_server_errors_are_transient() {
    for status in [429, 500, 503] {
        let (_server, model) = model_answering_with(status).await;

        let generated = model.generate(empty_request()).await.unwrap_err();
        let streamed = model
            .generate_stream(empty_request())
            .next()
            .await
            .unwrap()
            .unwrap_err();

        assert!(
            matches!(
                &generated,
                Error::ProviderTransient {
                    retry_after: None,
                    ..
                }
            ),
            "{status}: {generated:?}"
        );
        assert_eq!(generated.retry_after(), None);
        assert!(
            matches!(
                &streamed,
                Error::ProviderTransient {
                    retry_after: None,
                    ..
                }
            ),
            "{status}: {streamed:?}"
        );
        assert_eq!(streamed.retry_after(), None);
    }
}

#[tokio::test]
async fn client_errors_are_not_transient() {
    for status in [400, 401, 422] {
        let (_server, model) = model_answering_with(status).await;

        let generated = model.generate(empty_request()).await.unwrap_err();
        let streamed = model
            .generate_stream(empty_request())
            .next()
            .await
            .unwrap()
            .unwrap_err();

        assert!(
            matches!(&generated, Error::Provider(_)),
            "{status}: {generated:?}"
        );
        assert_eq!(generated.retry_after(), None);
        assert!(
            matches!(&streamed, Error::Provider(_)),
            "{status}: {streamed:?}"
        );
        assert_eq!(streamed.retry_after(), None);
    }
}

#[tokio::test]
async fn connection_failures_are_transient() {
    // Bind and drop a listener so nothing is listening on the port.
    let address = std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap();
    let model = DeepSeekModel::builder("test-key", "deepseek-chat")
        .base_url(format!("http://{address}"))
        .build();

    let generated = model.generate(empty_request()).await.unwrap_err();
    let streamed = model
        .generate_stream(empty_request())
        .next()
        .await
        .unwrap()
        .unwrap_err();

    assert!(
        matches!(
            &generated,
            Error::ProviderTransient {
                retry_after: None,
                ..
            }
        ),
        "{generated:?}"
    );
    assert_eq!(generated.retry_after(), None);
    assert!(
        matches!(
            &streamed,
            Error::ProviderTransient {
                retry_after: None,
                ..
            }
        ),
        "{streamed:?}"
    );
    assert_eq!(streamed.retry_after(), None);
}
