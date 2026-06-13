// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

//! Google Gemini provider adapter.
//!
//! Implements [`LlmModel`] on top of the
//! [`geologia`](https://github.com/andreban/geologia) client. Requires the
//! `gemini` Cargo feature.

use std::pin::Pin;

use async_trait::async_trait;
use futures_util::{Stream, StreamExt};
use geologia::prelude::{
    Candidate, Content, FunctionDeclaration, FunctionResponse, GeminiClient,
    GenerateContentRequest, GenerationConfig, Part, PartData, Role, ThinkingConfig, Tools,
    UsageMetadata,
};
use serde_json::Value;

use crate::{
    error::Error,
    model::{
        LlmModel, MessageContent, ModelRequest, ModelResponse, ModelStreamChunk, Role as AgentRole,
        TokenUsage, ToolCall,
    },
    tools::ToolDefinition,
};

/// LLM provider backed by Google Gemini.
///
/// Use [`GeminiModel::new`] for the simple case, or [`GeminiModel::builder`]
/// to configure generation settings such as temperature.
///
/// # Examples
///
/// ```no_run
/// use agent_rig::models::gemini::GeminiModel;
///
/// // Simple
/// let model = GeminiModel::new("API_KEY", "gemini-2.5-pro-preview-03-25");
///
/// // With settings
/// let model = GeminiModel::builder("API_KEY", "gemini-2.5-pro-preview-03-25")
///     .temperature(0.7)
///     .max_output_tokens(1024)
///     .build();
/// ```
pub struct GeminiModel {
    client: GeminiClient,
    model: String,
    generation_config: Option<GenerationConfig>,
}

impl GeminiModel {
    /// Creates a new `GeminiModel` with default generation settings.
    ///
    /// - `api_key` — your Gemini API key.
    /// - `model` — the Gemini model identifier (e.g. `"gemini-2.5-pro-preview-03-25"`).
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: GeminiClient::new(api_key.into()),
            model: model.into(),
            generation_config: None,
        }
    }

    /// Returns a [`GeminiModelBuilder`] for constructing a `GeminiModel` with custom settings.
    ///
    /// - `api_key` — your Gemini API key.
    /// - `model` — the Gemini model identifier (e.g. `"gemini-2.5-pro-preview-03-25"`).
    pub fn builder(api_key: impl Into<String>, model: impl Into<String>) -> GeminiModelBuilder {
        GeminiModelBuilder {
            api_key: api_key.into(),
            model: model.into(),
            generation_config: GenerationConfig::builder(),
        }
    }
}

/// Builder for [`GeminiModel`] with generation settings.
pub struct GeminiModelBuilder {
    api_key: String,
    model: String,
    generation_config: geologia::prelude::GenerationConfigBuilder,
}

impl GeminiModelBuilder {
    /// Sets the sampling temperature. Higher values produce more random output.
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.generation_config = self.generation_config.temperature(temperature);
        self
    }

    /// Sets the maximum number of tokens to generate.
    pub fn max_output_tokens(mut self, tokens: i32) -> Self {
        self.generation_config = self.generation_config.max_output_tokens(tokens);
        self
    }

    /// Sets the nucleus sampling probability threshold.
    pub fn top_p(mut self, top_p: f32) -> Self {
        self.generation_config = self.generation_config.top_p(top_p);
        self
    }

    /// Limits the next-token selection to the K most likely tokens.
    pub fn top_k(mut self, top_k: i32) -> Self {
        self.generation_config = self.generation_config.top_k(top_k);
        self
    }

    /// Sets stop sequences that halt generation when produced.
    pub fn stop_sequences(mut self, stop_sequences: Vec<String>) -> Self {
        self.generation_config = self.generation_config.stop_sequences(stop_sequences);
        self
    }

    /// Configures the model's thinking (chain-of-thought) behaviour.
    ///
    /// Set `include_thoughts: true` to receive thinking tokens in the response.
    /// Use [`ThinkingLevel`](geologia::prelude::ThinkingLevel) or a token
    /// budget to control how much reasoning the model performs.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use agent_rig::models::gemini::GeminiModel;
    /// use geologia::prelude::{ThinkingConfig, ThinkingLevel};
    ///
    /// let model = GeminiModel::builder("API_KEY", "gemini-2.5-flash-preview-04-17")
    ///     .thinking_config(ThinkingConfig {
    ///         include_thoughts: true,
    ///         thinking_level: Some(ThinkingLevel::High),
    ///         ..Default::default()
    ///     })
    ///     .build();
    /// ```
    pub fn thinking_config(mut self, thinking_config: ThinkingConfig) -> Self {
        self.generation_config = self.generation_config.thinking_config(thinking_config);
        self
    }

    /// Builds the [`GeminiModel`].
    pub fn build(self) -> GeminiModel {
        GeminiModel {
            client: GeminiClient::new(self.api_key),
            model: self.model,
            generation_config: Some(self.generation_config.build()),
        }
    }
}

/// Normalises a schemars-generated JSON Schema into a Gemini-compatible schema.
///
/// The Gemini API accepts a subset of JSON Schema and rejects meta-fields like
/// `$schema`, `title`, and `definitions`. This function strips those fields and
/// inlines every `$ref` so the result is self-contained.
fn normalise_for_gemini(mut root: Value) -> Value {
    // schemars may use either `definitions` (draft-07) or `$defs` (2019-09+).
    let definitions = root
        .as_object_mut()
        .and_then(|o| o.remove("definitions").or_else(|| o.remove("$defs")))
        .unwrap_or(Value::Null);

    resolve_refs(&mut root, &definitions);
    root
}

/// Ensures a tool result is a JSON object, as required by Gemini's
/// `functionResponse.response` field (a protobuf `Struct`).
///
/// Tool results that already serialize to an object pass through unchanged.
/// Scalar, array, or null results — which the Gemini API rejects with a
/// `400 Bad Request` — are wrapped in `{ "output": <value> }`. This covers
/// the synthesized string results for denied / errored / unknown tool calls
/// as well as any tool whose `Ok` value is not an object.
fn ensure_object_response(result: Value) -> Value {
    match result {
        Value::Object(_) => result,
        other => serde_json::json!({ "output": other }),
    }
}

fn resolve_refs(value: &mut Value, definitions: &Value) {
    match value {
        Value::Object(obj) => {
            // Inline $ref before doing anything else with this node.
            if let Some(ref_val) = obj.get("$ref").cloned()
                && let Some(ref_str) = ref_val.as_str()
            {
                let def_name = ref_str
                    .strip_prefix("#/definitions/")
                    .or_else(|| ref_str.strip_prefix("#/$defs/"));
                if let Some(def_name) = def_name
                    && let Some(def) = definitions.get(def_name)
                {
                    let mut resolved = def.clone();
                    resolve_refs(&mut resolved, definitions);
                    *value = resolved;
                    return;
                }
            }
            // Strip meta-fields the Gemini API does not recognise.
            obj.remove("$schema");
            for v in obj.values_mut() {
                resolve_refs(v, definitions);
            }
        }
        Value::Array(arr) => {
            for v in arr.iter_mut() {
                resolve_refs(v, definitions);
            }
        }
        _ => {}
    }
}

impl GeminiModel {
    /// Translates a [`ModelRequest`] into a Gemini [`GenerateContentRequest`].
    ///
    /// Shared by [`LlmModel::generate`] and [`LlmModel::generate_stream`] so the
    /// streaming and non-streaming endpoints see identical request bodies.
    fn build_gemini_request(&self, request: ModelRequest) -> GenerateContentRequest {
        let contents: Vec<Content> = request
            .messages
            .iter()
            .map(|msg| {
                let role = match msg.role {
                    AgentRole::User => Role::User,
                    AgentRole::Assistant => Role::Model,
                };
                match &msg.content {
                    MessageContent::Text(text) => Content::builder()
                        .role(role)
                        .add_text_part(text.clone())
                        .build(),
                    MessageContent::ToolCalls(calls) => Content {
                        role: Some(Role::Model),
                        parts: Some(
                            calls
                                .iter()
                                .map(|call| {
                                    let thought_signature = call
                                        .provider_metadata
                                        .as_ref()
                                        .and_then(|m| m["thought_signature"].as_str())
                                        .map(|s| s.to_string());
                                    Part {
                                        data: PartData::FunctionCall {
                                            id: Some(call.id.clone()),
                                            name: call.name.clone(),
                                            args: Some(call.args.clone()),
                                        },
                                        thought: None,
                                        thought_signature,
                                        part_metadata: None,
                                        media_resolution: None,
                                    }
                                })
                                .collect(),
                        ),
                    },
                    MessageContent::ToolResult {
                        id,
                        name,
                        result,
                        provider_metadata,
                    } => {
                        let thought_signature = provider_metadata
                            .as_ref()
                            .and_then(|m| m["thought_signature"].as_str())
                            .map(|s| s.to_string());
                        Content {
                            role: Some(Role::User),
                            parts: Some(vec![Part {
                                data: PartData::FunctionResponse(FunctionResponse {
                                    id: Some(id.clone()),
                                    name: name.clone(),
                                    response: ensure_object_response(result.clone()),
                                    parts: None,
                                    will_continue: None,
                                    scheduling: None,
                                }),
                                thought: None,
                                thought_signature,
                                part_metadata: None,
                                media_resolution: None,
                            }]),
                        }
                    }
                }
            })
            .collect();

        let mut builder = GenerateContentRequest::builder().contents(contents);

        if let Some(system) = &request.system {
            let system_content = Content::builder().add_text_part(system.clone()).build();
            builder = builder.system_instruction(system_content);
        }

        // Attach tool declarations when the request includes tools.
        if !request.tools.is_empty() {
            let declarations: Vec<FunctionDeclaration> = request
                .tools
                .iter()
                .map(FunctionDeclaration::from)
                .collect();
            let tools = Tools {
                function_declarations: Some(declarations),
                ..Default::default()
            };
            builder = builder.tools(vec![tools]);
        }

        // Request-level schema takes precedence over any model-level config.
        // When a schema is present we build a fresh config with the schema
        // fields, but carry over thinking_config from the model-level config so
        // that models with extended thinking still emit reasoning tokens in
        // structured-output mode.
        let effective_config = if let Some(schema) = request.output_schema {
            let normalised = normalise_for_gemini(schema.to_value());
            let mut builder = GenerationConfig::builder()
                .response_mime_type("application/json")
                .response_schema(normalised);
            if let Some(tc) = self
                .generation_config
                .as_ref()
                .and_then(|c| c.thinking_config.clone())
            {
                builder = builder.thinking_config(tc);
            }
            Some(builder.build())
        } else {
            self.generation_config.clone()
        };

        if let Some(config) = effective_config {
            builder = builder.generation_config(config);
        }

        builder.build()
    }
}

#[async_trait]
impl LlmModel for GeminiModel {
    async fn generate(&self, request: ModelRequest) -> Result<ModelResponse, Error> {
        let gemini_request = self.build_gemini_request(request);

        let response = self
            .client
            .generate_content(&gemini_request, &self.model)
            .await
            .map_err(|e| Error::Provider(e.to_string()))?;

        let candidate = response
            .candidates
            .first()
            .ok_or_else(|| Error::Provider("empty candidates in Gemini response".to_string()))?;

        // Collect any function call parts from the response.
        let tool_calls: Vec<ToolCall> = candidate
            .content
            .as_ref()
            .and_then(|c| c.parts.as_ref())
            .map(|parts| {
                parts
                    .iter()
                    .filter_map(|part| {
                        if let PartData::FunctionCall { id, name, args } = &part.data {
                            // Stash the thought_signature so it can be echoed
                            // back on both the replayed FunctionCall and the
                            // FunctionResponse parts in subsequent turns.
                            let provider_metadata = part
                                .thought_signature
                                .as_ref()
                                .map(|ts| serde_json::json!({ "thought_signature": ts }));
                            Some(ToolCall {
                                id: id.clone().unwrap_or_default(),
                                name: name.clone(),
                                args: args.clone().unwrap_or(serde_json::Value::Null),
                                provider_metadata,
                            })
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();

        // Separate thought parts (thought == Some(true)) from regular text parts.
        // Thinking can appear on any turn, including tool-calling turns.
        let (thinking, text) = extract_text_and_thinking(candidate);

        let token_usage = response.usage_metadata.as_ref().map(TokenUsage::from);

        if !tool_calls.is_empty() {
            return Ok(ModelResponse {
                text: None,
                tool_calls,
                thinking,
                token_usage,
            });
        }

        Ok(ModelResponse {
            text,
            tool_calls: vec![],
            thinking,
            token_usage,
        })
    }

    /// Streams the Gemini response as [`ModelStreamChunk`] values.
    ///
    /// Consumes the SSE stream returned by `stream_generate_content` and emits
    /// one chunk per part: [`ModelStreamChunk::Thinking`] for thought parts,
    /// [`ModelStreamChunk::TextDelta`] for regular text parts, and
    /// [`ModelStreamChunk::ToolCall`] for function calls. A trailing
    /// [`ModelStreamChunk::Usage`] chunk carries the final [`UsageMetadata`]
    /// reported by the provider, if any.
    fn generate_stream(
        &self,
        request: ModelRequest,
    ) -> Pin<Box<dyn Stream<Item = Result<ModelStreamChunk, Error>> + Send + '_>> {
        Box::pin(async_stream::stream! {
            let gemini_request = self.build_gemini_request(request);
            let stream = match self
                .client
                .stream_generate_content(&gemini_request, &self.model)
                .await {
                    Ok(stream) => stream,
                    Err(e) => {
                        yield Err(Error::Provider(e.to_string()));
                        return;
                    }
                };

            let mut stream = Box::pin(stream);

            // Gemini reports usage cumulatively on each chunk; keep the latest
            // and emit it once after the stream ends.
            let mut latest_usage: Option<UsageMetadata> = None;

            while let Some(chunk) = stream.next().await {
                let response = match chunk {
                    Ok(response) => response,
                    Err(e) => {
                        yield Err(Error::Provider(e.to_string()));
                        return;
                    }
                };

                if let Some(usage) = &response.usage_metadata {
                    latest_usage = Some(usage.clone());
                }

                if let Some(candidate) = response.candidates.first() {
                    // Yield text_chunks first so message and thoughts related to the tool
                    // call are made available before the tool call happens, giving more
                    // context to the too usage/permission.
                    let (text_chunks, tool_calls) = stream_chunks_from_candidate(candidate);
                    for chunk in text_chunks {
                        yield Ok(chunk);
                    }
                    for chunk in tool_calls {
                        yield Ok(chunk);
                    }
                }
            }

            if let Some(usage) = latest_usage {
                yield Ok(ModelStreamChunk::Usage(TokenUsage::from(&usage)));
            }
        })
    }
}

/// Converts a single streamed [`Candidate`] into per-part [`ModelStreamChunk`]s.
///
/// Iterates the candidate's parts in order, emitting [`ModelStreamChunk::Thinking`]
/// for thought text parts, [`ModelStreamChunk::TextDelta`] for regular text parts,
/// and [`ModelStreamChunk::ToolCall`] for function calls (with `thought_signature`
/// preserved in `provider_metadata`). Empty text parts and any other part kinds
/// are skipped.
///
/// Returns `(text_chunks, tool_calls)`, keeping text/thinking chunks separate
/// from tool-call chunks so callers can emit text before dispatching tools.
/// Both vectors are empty when the candidate has no content or parts.
fn stream_chunks_from_candidate(
    candidate: &Candidate,
) -> (Vec<ModelStreamChunk>, Vec<ModelStreamChunk>) {
    let Some(parts) = candidate.content.as_ref().and_then(|c| c.parts.as_ref()) else {
        return (vec![], vec![]);
    };

    let mut text_chunks = vec![];
    let mut tool_calls = vec![];
    for part in parts {
        match &part.data {
            PartData::Text(text) if !text.is_empty() => {
                if part.thought == Some(true) {
                    text_chunks.push(ModelStreamChunk::Thinking(text.clone()));
                } else {
                    text_chunks.push(ModelStreamChunk::TextDelta(text.clone()));
                }
            }
            PartData::FunctionCall { id, name, args } => {
                let provider_metadata = part
                    .thought_signature
                    .as_ref()
                    .map(|ts| serde_json::json!({ "thought_signature": ts }));
                tool_calls.push(ModelStreamChunk::ToolCall(ToolCall {
                    id: id.clone().unwrap_or_default(),
                    name: name.clone(),
                    args: args.clone().unwrap_or(serde_json::Value::Null),
                    provider_metadata,
                }));
            }
            _ => {}
        }
    }
    (text_chunks, tool_calls)
}

/// Maps Gemini's [`UsageMetadata`] into [`TokenUsage`].
///
/// Per-modality breakdowns (`*_tokens_details`) and `service_tier` are
/// intentionally not propagated — `TokenUsage` only carries totals.
impl From<&UsageMetadata> for TokenUsage {
    fn from(meta: &UsageMetadata) -> Self {
        TokenUsage {
            input_tokens: meta.prompt_token_count,
            output_tokens: meta.candidates_token_count,
            cached_input_tokens: meta.cached_content_token_count,
            thinking_tokens: meta.thoughts_token_count,
            tool_use_prompt_tokens: meta.tool_use_prompt_token_count,
        }
    }
}

/// Translates a [`ToolDefinition`] into a Gemini [`FunctionDeclaration`].
impl From<&ToolDefinition> for FunctionDeclaration {
    fn from(def: &ToolDefinition) -> FunctionDeclaration {
        FunctionDeclaration {
            name: def.name.clone(),
            description: def.description.clone(),
            parameters: None,
            parameters_json_schema: Some(def.parameters.clone().into()),
            response: None,
            response_json_schema: None,
        }
    }
}

/// Splits a candidate's parts into thinking text and regular text.
///
/// Parts where `thought == Some(true)` are concatenated into the thinking
/// string; all other `Text` parts form the regular response text.
fn extract_text_and_thinking(candidate: &Candidate) -> (Option<String>, Option<String>) {
    let parts = match candidate.content.as_ref().and_then(|c| c.parts.as_ref()) {
        Some(p) => p,
        None => return (None, None),
    };

    let mut thinking_buf = String::new();
    let mut text_buf = String::new();

    for part in parts {
        if let PartData::Text(t) = &part.data {
            if part.thought == Some(true) {
                thinking_buf.push_str(t);
            } else {
                text_buf.push_str(t);
            }
        }
    }

    let thinking = if thinking_buf.is_empty() {
        None
    } else {
        Some(thinking_buf)
    };
    let text = if text_buf.is_empty() {
        None
    } else {
        Some(text_buf)
    };
    (thinking, text)
}

#[cfg(test)]
mod tests;
