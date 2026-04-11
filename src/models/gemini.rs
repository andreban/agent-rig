use async_trait::async_trait;
use google_genai::prelude::{
    Content, GeminiClient, GenerateContentRequest, GenerationConfig, Role,
};

use crate::{
    error::Error,
    model::{LlmModel, ModelRequest, ModelResponse, Role as AgentRole},
};

/// LLM provider backed by Google Gemini.
///
/// Use [`GeminiModel::new`] for the simple case, or [`GeminiModel::builder`]
/// to configure generation settings such as temperature.
///
/// # Examples
///
/// ```no_run
/// use rust_agent_kit::models::gemini::GeminiModel;
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
    pub fn builder(
        api_key: impl Into<String>,
        model: impl Into<String>,
    ) -> GeminiModelBuilder {
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
    generation_config: google_genai::prelude::GenerationConfigBuilder,
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

    /// Sets the MIME type of the response (e.g. `"application/json"`).
    ///
    /// Use together with [`response_schema`](Self::response_schema) to request
    /// structured JSON output from the model.
    pub fn response_mime_type(mut self, mime_type: impl Into<String>) -> Self {
        self.generation_config = self.generation_config.response_mime_type(mime_type.into());
        self
    }

    /// Sets a JSON Schema that the model's response must conform to.
    ///
    /// Pass the schema as a [`serde_json::Value`]. When combined with
    /// [`response_mime_type("application/json")`](Self::response_mime_type),
    /// the model returns a JSON object that matches the provided schema.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rust_agent_kit::models::gemini::GeminiModel;
    /// use schemars::JsonSchema;
    /// use serde::Deserialize;
    ///
    /// #[derive(Deserialize, JsonSchema)]
    /// struct Answer {
    ///     value: String,
    /// }
    ///
    /// let schema = serde_json::to_value(schemars::schema_for!(Answer)).unwrap();
    /// let model = GeminiModel::builder("API_KEY", "gemini-2.5-pro-preview-03-25")
    ///     .response_mime_type("application/json")
    ///     .response_schema(schema)
    ///     .build();
    /// ```
    pub fn response_schema(mut self, schema: serde_json::Value) -> Self {
        self.generation_config = self.generation_config.response_schema(schema);
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
                Content::builder()
                    .role(role)
                    .add_text_part(msg.content.clone())
                    .build()
            })
            .collect();

        let mut builder = GenerateContentRequest::builder().contents(contents);

        if let Some(system) = &request.system {
            let system_content = Content::builder().add_text_part(system.clone()).build();
            builder = builder.system_instruction(system_content);
        }

        if let Some(config) = &self.generation_config {
            builder = builder.generation_config(config.clone());
        }

        let gemini_request = builder.build();

        let response = self
            .client
            .generate_content(&gemini_request, &self.model)
            .await
            .map_err(|e| Error::Provider(e.to_string()))?;

        let text = response
            .candidates
            .first()
            .and_then(|c| c.get_text())
            .unwrap_or_default();

        Ok(ModelResponse { text })
    }
}
