use std::pin::Pin;

use async_trait::async_trait;
use futures_util::stream::Stream;

use crate::error::Error;
use crate::tool::ToolDefinition;

/// The role of a participant in a conversation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Role {
    /// A message from the user.
    User,
    /// A message from the assistant/model.
    Assistant,
}

/// The content carried by a [`Message`].
#[derive(Debug, Clone)]
pub enum MessageContent {
    /// Plain text.
    Text(String),
    /// All tool calls issued by the model in one assistant turn. Grouped
    /// together so provider adapters can reconstruct a single message with
    /// multiple call parts (Gemini) or a `tool_calls` array (Ollama).
    ToolCalls(Vec<ToolCall>),
    /// The result of executing one tool (one message per result).
    ToolResult {
        /// The ID from the originating [`ToolCall`].
        id: String,
        /// Tool name.
        name: String,
        /// Return value as a JSON value.
        result: serde_json::Value,
        /// Opaque provider metadata copied from the originating [`ToolCall`].
        /// Used by Gemini to echo `thought_signature`; other providers ignore it.
        provider_metadata: Option<serde_json::Value>,
    },
}

/// A single message in a conversation.
#[derive(Debug, Clone)]
pub struct Message {
    /// The role of the message sender.
    pub role: Role,
    /// The content of the message.
    pub content: MessageContent,
}

impl Message {
    /// Creates a new user text message.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_agent_kit::model::Message;
    ///
    /// let msg = Message::user("What is the capital of France?");
    /// ```
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: MessageContent::Text(content.into()),
        }
    }

    /// Creates a new assistant text message.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_agent_kit::model::Message;
    ///
    /// let msg = Message::assistant("The capital of France is Paris.");
    /// ```
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: MessageContent::Text(content.into()),
        }
    }

    /// Creates an assistant message representing all tool calls from one model turn.
    pub(crate) fn tool_calls(calls: Vec<ToolCall>) -> Self {
        Self {
            role: Role::Assistant,
            content: MessageContent::ToolCalls(calls),
        }
    }

    /// Creates a message carrying the result of one tool execution.
    pub(crate) fn tool_result(
        id: String,
        name: String,
        result: serde_json::Value,
        provider_metadata: Option<serde_json::Value>,
    ) -> Self {
        Self {
            role: Role::User,
            content: MessageContent::ToolResult { id, name, result, provider_metadata },
        }
    }
}

/// A request to an LLM model.
#[derive(Debug, Clone)]
pub struct ModelRequest {
    /// The conversation history, in chronological order.
    pub messages: Vec<Message>,
    /// Optional system-level instructions that guide the model's behaviour.
    pub system: Option<String>,
    /// Optional JSON Schema the model's response must conform to.
    ///
    /// When set, the provider adapter applies structured-output constraints
    /// using provider-specific mechanisms. Providers that do not support
    /// structured output ignore this field.
    pub output_schema: Option<serde_json::Value>,
    /// Tool definitions available to the model on this request.
    ///
    /// An empty `Vec` means no tools are available.
    pub tools: Vec<ToolDefinition>,
}

/// A tool call issued by the model.
#[derive(Debug, Clone)]
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
    pub(crate) provider_metadata: Option<serde_json::Value>,
}

/// A response from an LLM model.
///
/// Exactly one of `text` or `tool_calls` will be non-empty per turn:
/// a final response carries text; an intermediate turn carries tool calls.
#[derive(Debug, Clone)]
pub struct ModelResponse {
    /// The generated text output, present only when the model produced a final
    /// text response (i.e. `tool_calls` is empty).
    pub text: Option<String>,
    /// Tool calls the model wants the runner to execute. Empty on a text turn.
    pub tool_calls: Vec<ToolCall>,
    /// Reasoning/thinking text produced by the model before its final answer.
    ///
    /// Only populated by provider adapters that support extended thinking
    /// (currently [`GeminiModel`] when `include_thoughts` is enabled via
    /// [`ThinkingConfig`]). All other adapters leave this as `None`.
    ///
    /// [`GeminiModel`]: crate::models::gemini::GeminiModel
    /// [`ThinkingConfig`]: google_genai::prelude::ThinkingConfig
    pub thinking: Option<String>,
}

/// A chunk yielded by [`LlmModel::generate_stream`] during a single model turn.
///
/// Provider adapters emit these values; the runner wraps them into [`AgentEvent`]
/// and adds tool-call lifecycle events on top.
///
/// [`AgentEvent`]: crate::AgentEvent
#[derive(Debug, Clone)]
pub enum ModelStreamChunk {
    /// A reasoning/thinking token from a model that supports extended thinking
    /// (e.g. Gemini 2.5 with extended thinking enabled).
    Thinking(String),
    /// An incremental chunk of the model's text output.
    TextDelta(String),
    /// A complete tool call. Tool calls are not streamed mid-call; the full
    /// call is emitted as a single chunk once the model has finished specifying it.
    ToolCall(ToolCall),
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
/// use rust_agent_kit::error::Error;
/// use rust_agent_kit::model::{LlmModel, MessageContent, ModelRequest, ModelResponse};
///
/// struct EchoModel;
///
/// #[async_trait]
/// impl LlmModel for EchoModel {
///     async fn generate(&self, request: ModelRequest) -> Result<ModelResponse, Error> {
///         let echo = request.messages.last().and_then(|m| {
///             if let MessageContent::Text(t) = &m.content { Some(t.clone()) } else { None }
///         });
///         Ok(ModelResponse { text: echo, tool_calls: vec![], thinking: None })
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
}
