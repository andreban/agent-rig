// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

//! Provider-agnostic conversation primitives.
//!
//! This module defines the data types every provider adapter speaks: [`Message`]
//! and its [`MessageContent`] variants, the [`ToolCall`] issued by the model,
//! the [`ModelRequest`] / [`ModelResponse`] envelope, and the [`LlmModel`]
//! trait that every provider implements. The runner in [`crate::runner`]
//! drives [`LlmModel`] in a loop until the model produces no more tool calls.

use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures_util::stream::Stream;

use schemars::Schema;
use serde::{Deserialize, Serialize};

use crate::error::Error;
use crate::tools::ToolDefinition;

/// The role of a participant in a conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// A message from the user.
    User,
    /// A message from the assistant/model.
    Assistant,
}

/// The content carried by a [`Message`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", content = "content", rename_all = "snake_case")]
pub enum MessageContent {
    /// Plain text.
    Text(String),
    /// All tool calls issued by the model in one assistant turn. Grouped
    /// together so provider adapters can reconstruct a single message with
    /// multiple call parts (Gemini) or a `tool_calls` array (Ollama).
    ToolCalls(Vec<Arc<ToolCall>>),
    /// The result of executing one tool (one message per result).
    ToolResult {
        tool_call: Arc<ToolCall>,
        result: serde_json::Value,
    },
}

/// A single message in a conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    /// The role of the message sender.
    pub role: Role,
    /// The content of the message.
    pub content: MessageContent,
    /// Opaque provider state attached to an assistant turn, round-tripped back
    /// to the adapter that produced it on later requests.
    ///
    /// This is the message-level counterpart of [`ToolCall::provider_metadata`].
    /// Adapters emit it via [`ModelStreamChunk::ProviderMetadata`] (or
    /// [`ModelResponse::provider_metadata`]) when their provider needs state
    /// replayed — e.g. [`DeepSeekModel`] stores the turn's `reasoning_content`
    /// under a `"deepseek"` key. The runner attaches whatever it received to the
    /// assistant message it appends and never looks inside. Adapters read only
    /// their own key, so a thread moved across providers is ignored rather than
    /// misreplayed.
    ///
    /// Omitted from serialized output when `None`, and defaults to `None` when
    /// absent, so previously persisted conversations still deserialize.
    ///
    /// [`DeepSeekModel`]: crate::models::deepseek::DeepSeekModel
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<serde_json::Value>,
}

impl Message {
    /// Creates a new user text message.
    ///
    /// # Examples
    ///
    /// ```
    /// use agent_rig::model::Message;
    ///
    /// let msg = Message::user("What is the capital of France?");
    /// ```
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: MessageContent::Text(content.into()),
            provider_metadata: None,
        }
    }

    /// Creates a new assistant text message.
    ///
    /// # Examples
    ///
    /// ```
    /// use agent_rig::model::Message;
    ///
    /// let msg = Message::assistant("The capital of France is Paris.");
    /// ```
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: MessageContent::Text(content.into()),
            provider_metadata: None,
        }
    }

    /// Creates an assistant message representing all tool calls from one model turn.
    pub fn tool_calls(calls: Vec<Arc<ToolCall>>) -> Self {
        Self {
            role: Role::Assistant,
            content: MessageContent::ToolCalls(calls),
            provider_metadata: None,
        }
    }

    /// Creates a message carrying the result of one tool execution.
    pub fn tool_result(tool_call: Arc<ToolCall>, result: serde_json::Value) -> Self {
        Self {
            role: Role::User,
            content: MessageContent::ToolResult { tool_call, result },
            provider_metadata: None,
        }
    }

    /// Attaches opaque provider state to this message; see
    /// [`Message::provider_metadata`].
    ///
    /// # Examples
    ///
    /// ```
    /// use agent_rig::model::Message;
    /// use serde_json::json;
    ///
    /// let msg = Message::assistant("Paris.")
    ///     .with_provider_metadata(json!({ "my_provider": { "state": "..." } }));
    /// assert!(msg.provider_metadata.is_some());
    /// ```
    pub fn with_provider_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.provider_metadata = Some(metadata);
        self
    }
}

/// A list of messages representing a conversation thread.
///
/// Under the hood, this wraps a `Vec<Arc<Message>>` to ensure that messages
/// are cheap to clone and share, while providing a clean and ergonomic API.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(transparent)]
pub struct MessageList {
    inner: Vec<Arc<Message>>,
}

impl MessageList {
    /// Creates a new, empty `MessageList`.
    pub fn new() -> Self {
        Self { inner: Vec::new() }
    }

    /// Appends a message to the end of the list, wrapping it in an `Arc`.
    pub fn push(&mut self, message: Message) {
        self.inner.push(Arc::new(message));
    }

    /// Appends a pre-wrapped message (`Arc<Message>`) to the end of the list.
    pub fn push_arc(&mut self, message: Arc<Message>) {
        self.inner.push(message);
    }

    /// Removes the last element from the list and returns it, or `None` if empty.
    pub fn pop(&mut self) -> Option<Arc<Message>> {
        self.inner.pop()
    }

    /// Clears the list, removing all values.
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// Shortens the list, keeping the first `len` elements and dropping the rest.
    pub fn truncate(&mut self, len: usize) {
        self.inner.truncate(len);
    }
}

impl std::ops::Deref for MessageList {
    type Target = [Arc<Message>];

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl From<Vec<Arc<Message>>> for MessageList {
    fn from(inner: Vec<Arc<Message>>) -> Self {
        Self { inner }
    }
}

impl From<MessageList> for Vec<Arc<Message>> {
    fn from(list: MessageList) -> Self {
        list.inner
    }
}

impl From<Vec<Message>> for MessageList {
    fn from(msgs: Vec<Message>) -> Self {
        Self {
            inner: msgs.into_iter().map(Arc::new).collect(),
        }
    }
}

impl FromIterator<Arc<Message>> for MessageList {
    fn from_iter<T: IntoIterator<Item = Arc<Message>>>(iter: T) -> Self {
        Self {
            inner: iter.into_iter().collect(),
        }
    }
}

impl FromIterator<Message> for MessageList {
    fn from_iter<T: IntoIterator<Item = Message>>(iter: T) -> Self {
        Self {
            inner: iter.into_iter().map(Arc::new).collect(),
        }
    }
}

impl IntoIterator for MessageList {
    type Item = Arc<Message>;
    type IntoIter = std::vec::IntoIter<Arc<Message>>;

    fn into_iter(self) -> Self::IntoIter {
        self.inner.into_iter()
    }
}

impl<'a> IntoIterator for &'a MessageList {
    type Item = &'a Arc<Message>;
    type IntoIter = std::slice::Iter<'a, Arc<Message>>;

    fn into_iter(self) -> Self::IntoIter {
        self.inner.iter()
    }
}

impl Extend<Message> for MessageList {
    fn extend<T: IntoIterator<Item = Message>>(&mut self, iter: T) {
        self.inner.extend(iter.into_iter().map(Arc::new));
    }
}

impl Extend<Arc<Message>> for MessageList {
    fn extend<T: IntoIterator<Item = Arc<Message>>>(&mut self, iter: T) {
        self.inner.extend(iter);
    }
}

/// A request to an LLM model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRequest {
    /// The conversation history, in chronological order.
    pub messages: MessageList,
    /// Optional system-level instructions that guide the model's behaviour.
    pub system: Option<String>,
    /// Optional JSON Schema the model's response must conform to.
    ///
    /// When set, the provider adapter applies structured-output constraints
    /// using provider-specific mechanisms. Providers that do not support
    /// structured output ignore this field.
    pub output_schema: Option<Schema>,
    /// Tool definitions available to the model on this request.
    ///
    /// An empty `Vec` means no tools are available.
    pub tools: Vec<ToolDefinition>,
}

/// A tool call issued by the model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    /// Provider-assigned call identifier. Must be echoed in the tool response
    /// for providers that require it (e.g. Gemini).
    pub id: String,
    /// The name of the tool to invoke.
    pub name: String,
    /// The arguments the model wants to pass, as a JSON object.
    pub args: serde_json::Value,
    /// Opaque provider metadata that must be round-tripped back with the tool
    /// response. Used by Gemini to carry the `thought_signature`; other
    /// providers leave this as `None`.
    ///
    /// External [`LlmModel`] implementations populate this when constructing a
    /// [`ToolCall`] from a provider response, and read it when echoing the call
    /// back on the next turn.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<serde_json::Value>,
}

impl ToolCall {
    /// Creates a new `ToolCall`.
    pub fn new(id: String, name: String, args: serde_json::Value) -> Self {
        Self {
            id,
            name,
            args,
            provider_metadata: None,
        }
    }
}

/// Token counts reported by a provider for one model call.
///
/// Every field is `Option<u32>` so providers that do not report a given
/// dimension leave it `None` — distinct from `Some(0)`, which means
/// the provider reported zero tokens in this dimension.
///
/// Consumers concatenate per-call counts themselves to derive run totals.
/// Total tokens are intentionally not stored — derive as
/// `input_tokens + output_tokens` (plus the optional dimensions) when
/// needed, to avoid drift against provider-reported totals that may round
/// each dimension independently.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenUsage {
    /// Prompt / input tokens billed for this call.
    pub input_tokens: Option<u32>,
    /// Generated / output tokens billed for this call.
    pub output_tokens: Option<u32>,
    /// Tokens served from the provider's prompt cache.
    ///
    /// **Contract:** subset of `input_tokens`. That is, `input_tokens`
    /// includes cached tokens; `cached_input_tokens` is the portion of
    /// that count served from the cache. The non-cached portion is
    /// `input_tokens - cached_input_tokens`.
    ///
    /// This matches Gemini's `promptTokenCount` / `cachedContentTokenCount`
    /// and OpenAI's `prompt_tokens` / `prompt_tokens_details.cached_tokens`.
    /// Providers that report cache tokens additively (e.g. Anthropic's
    /// `cache_read_input_tokens`) must normalise inside the adapter so
    /// this invariant holds.
    pub cached_input_tokens: Option<u32>,
    /// Reasoning / thinking tokens, when the provider bills them
    /// separately from `output_tokens`.
    pub thinking_tokens: Option<u32>,
    /// Tokens consumed by tool-use prompt parts, when the provider bills
    /// them separately from `input_tokens` (Gemini's
    /// `toolUsePromptTokenCount`).
    pub tool_use_prompt_tokens: Option<u32>,
}

/// A response from an LLM model.
///
/// Exactly one of `text` or `tool_calls` will be non-empty per turn:
/// a final response carries text; an intermediate turn carries tool calls.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelResponse {
    /// The generated text output, present only when the model produced a final
    /// text response (i.e. `tool_calls` is empty).
    pub text: Option<String>,
    /// Tool calls the model wants the runner to execute. Empty on a text turn.
    pub tool_calls: Vec<ToolCall>,
    /// Reasoning/thinking text produced by the model before its final answer.
    ///
    /// Only populated by provider adapters that support extended thinking
    /// (e.g. [`GeminiModel`] when `include_thoughts` is enabled via
    /// [`ThinkingConfig`], or [`DeepSeekModel`] in thinking mode). Other
    /// adapters leave this as `None`.
    ///
    /// [`GeminiModel`]: crate::models::gemini::GeminiModel
    /// [`DeepSeekModel`]: crate::models::deepseek::DeepSeekModel
    /// [`ThinkingConfig`]: geologia::prelude::ThinkingConfig
    pub thinking: Option<String>,
    /// Opaque provider state to store on the assistant message for this turn
    /// and replay on later requests; see [`Message::provider_metadata`].
    ///
    /// The default [`LlmModel::generate_stream`] forwards it as a
    /// [`ModelStreamChunk::ProviderMetadata`] chunk.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<serde_json::Value>,
    /// Token counts reported by the provider for this call.
    ///
    /// `None` when the provider did not report usage on this response.
    /// Adapters that support usage reporting populate this; the default
    /// [`LlmModel::generate_stream`] forwards it as a trailing
    /// [`ModelStreamChunk::Usage`] chunk.
    pub token_usage: Option<TokenUsage>,
}

/// A chunk yielded by [`LlmModel::generate_stream`] during a single model turn.
///
/// Provider adapters emit these values; the runner wraps them into [`AgentEvent`]
/// and adds tool-call lifecycle events on top.
///
/// [`AgentEvent`]: crate::runner::AgentEvent
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "content", rename_all = "snake_case")]
pub enum ModelStreamChunk {
    /// A reasoning/thinking token from a model that supports extended thinking
    /// (e.g. Gemini 2.5 with extended thinking enabled).
    Thinking(String),
    /// An incremental chunk of the model's text output.
    TextDelta(String),
    /// A complete tool call. Tool calls are not streamed mid-call; the full
    /// call is emitted as a single chunk once the model has finished specifying it.
    ToolCall(ToolCall),
    /// Token counts reported by the provider for this model call.
    ///
    /// Emitted at most once per [`LlmModel::generate_stream`] invocation,
    /// typically as the final chunk before the stream ends. Provider
    /// adapters that do not report usage simply never yield this variant.
    Usage(TokenUsage),
    /// Opaque provider state for this turn, to be stored on the assistant
    /// message the runner appends (see [`Message::provider_metadata`]).
    ///
    /// Emitted at most once per [`LlmModel::generate_stream`] invocation,
    /// after the turn's content. Adapters that need no replay state never
    /// yield this variant.
    ProviderMetadata(serde_json::Value),
}

/// Trait implemented by all LLM provider backends.
///
/// Implement this trait to add support for a new LLM provider. The runner
/// holds a `Box<dyn LlmModel>` and calls [`generate`](LlmModel::generate)
/// on each turn of the agent loop.
///
/// # Examples
///
/// ```no_run
/// use async_trait::async_trait;
/// use agent_rig::error::Error;
/// use agent_rig::model::{LlmModel, MessageContent, ModelRequest, ModelResponse};
///
/// struct EchoModel;
///
/// #[async_trait]
/// impl LlmModel for EchoModel {
///     async fn generate(&self, request: ModelRequest) -> Result<ModelResponse, Error> {
///         let echo = request.messages.last().and_then(|m| {
///             if let MessageContent::Text(t) = &m.content { Some(t.clone()) } else { None }
///         });
///         Ok(ModelResponse {
///             text: echo,
///             tool_calls: vec![],
///             thinking: None,
///             provider_metadata: None,
///             token_usage: None,
///         })
///     }
/// }
/// ```
#[async_trait]
pub trait LlmModel: Send + Sync {
    /// Generate a response for the given [`ModelRequest`].
    async fn generate(&self, request: ModelRequest) -> Result<ModelResponse, Error>;

    /// Stream a response for the given [`ModelRequest`] as a sequence of
    /// [`ModelStreamChunk`] values.
    ///
    /// The default implementation calls [`generate`] and emits the result as
    /// one or more chunks, so existing adapters work without modification.
    /// Override this method to provide true token-by-token streaming.
    ///
    /// Tool calls are never streamed mid-call; each complete [`ToolCall`] is
    /// emitted as a single [`ModelStreamChunk::ToolCall`] chunk. [`Thinking`]
    /// and [`TextDelta`] chunks may be emitted across many events.
    ///
    /// [`generate`]: LlmModel::generate
    /// [`Thinking`]: ModelStreamChunk::Thinking
    /// [`TextDelta`]: ModelStreamChunk::TextDelta
    fn generate_stream(
        &self,
        request: ModelRequest,
    ) -> Pin<Box<dyn Stream<Item = Result<ModelStreamChunk, Error>> + Send + '_>> {
        Box::pin(async_stream::stream! {
            let response = self.generate(request).await?;
            if let Some(thinking) = response.thinking {
                yield Ok(ModelStreamChunk::Thinking(thinking));
            }
            for call in response.tool_calls {
                yield Ok(ModelStreamChunk::ToolCall(call));
            }
            if let Some(text) = response.text {
                yield Ok(ModelStreamChunk::TextDelta(text));
            }
            if let Some(metadata) = response.provider_metadata {
                yield Ok(ModelStreamChunk::ProviderMetadata(metadata));
            }
            if let Some(token_usage) = response.token_usage {
                yield Ok(ModelStreamChunk::Usage(token_usage));
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_user_sets_correct_role() {
        let msg = Message::user("hello");
        assert_eq!(msg.role, Role::User);
        assert!(matches!(msg.content, MessageContent::Text(t) if t == "hello"));
    }

    #[test]
    fn message_assistant_sets_correct_role() {
        let msg = Message::assistant("hi");
        assert_eq!(msg.role, Role::Assistant);
        assert!(matches!(msg.content, MessageContent::Text(t) if t == "hi"));
    }

    #[test]
    fn message_provider_metadata_round_trips_through_serde() {
        let msg = Message::assistant("hi").with_provider_metadata(serde_json::json!({ "p": 1 }));
        let json = serde_json::to_value(&msg).unwrap();
        assert_eq!(json["provider_metadata"], serde_json::json!({ "p": 1 }));
        let back: Message = serde_json::from_value(json).unwrap();
        assert_eq!(back, msg);
    }

    #[test]
    fn message_without_provider_metadata_omits_field_and_loads_legacy_json() {
        let msg = Message::assistant("hi");
        let json = serde_json::to_value(&msg).unwrap();
        assert!(json.get("provider_metadata").is_none());

        // Conversations persisted before the field existed still load.
        let legacy = serde_json::json!({
            "role": "assistant",
            "content": { "type": "text", "content": "hi" }
        });
        let back: Message = serde_json::from_value(legacy).unwrap();
        assert_eq!(back, msg);
    }

    #[test]
    fn message_list_ergonomics_and_behavior() {
        let mut list = MessageList::new();
        assert!(list.is_empty());

        // Push wraps in Arc automatically
        list.push(Message::user("hello"));
        list.push(Message::assistant("hi"));
        assert_eq!(list.len(), 2);

        // Deref interface
        assert_eq!(list[0].role, Role::User);
        assert_eq!(list[1].role, Role::Assistant);

        // Conversions
        let single_list: MessageList = vec![Message::user("one")].into();
        assert_eq!(single_list.len(), 1);

        let vec_list: MessageList = vec![Message::user("a"), Message::user("b")].into();
        assert_eq!(vec_list.len(), 2);

        // Transparent Serialization
        let serialized = serde_json::to_string(&list).unwrap();
        let deserialized: MessageList = serde_json::from_str(&serialized).unwrap();
        assert_eq!(list, deserialized);

        // Ensure it serializes as a plain JSON array, not a wrapped object
        let val: serde_json::Value = serde_json::from_str(&serialized).unwrap();
        assert!(val.is_array());
        assert_eq!(val.as_array().unwrap().len(), 2);
    }
}
