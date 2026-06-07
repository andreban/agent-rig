// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

//! Google Gemini provider adapter.
//!
//! Implements [`LlmModel`] on top of the
//! [`geologia`](https://github.com/andreban/geologia) client. Requires the
//! `gemini` Cargo feature.

use async_trait::async_trait;
use geologia::prelude::{
    Candidate, Content, FunctionDeclaration, FunctionResponse, GeminiClient,
    GenerateContentRequest, GenerationConfig, Part, PartData, Role, ThinkingConfig, Tools,
    UsageMetadata,
};
use serde_json::Value;

use crate::{
    error::Error,
    model::{
        LlmModel, MessageContent, ModelRequest, ModelResponse, Role as AgentRole, TokenUsage,
        ToolCall,
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

/// Translates a [`ToolDefinition`] into a Gemini [`FunctionDeclaration`].
fn to_function_declaration(def: &ToolDefinition) -> FunctionDeclaration {
    FunctionDeclaration {
        name: def.name.clone(),
        description: def.description.clone(),
        parameters: None,
        parameters_json_schema: Some(def.parameters.clone().into()),
        response: None,
        response_json_schema: None,
    }
}

#[async_trait]
impl LlmModel for GeminiModel {
    async fn generate(&self, request: ModelRequest) -> Result<ModelResponse, Error> {
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
                                    response: result.clone(),
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
            let declarations: Vec<FunctionDeclaration> =
                request.tools.iter().map(to_function_declaration).collect();
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
            let normalised = normalise_for_gemini(schema);
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

        let gemini_request = builder.build();

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
mod tests {
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
        let candidate =
            make_candidate(vec![thought_part("hmm..."), text_part("The answer is 42.")]);
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
}
