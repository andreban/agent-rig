use async_trait::async_trait;

use crate::error::Error;

/// The role of a participant in a conversation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Role {
    /// A message from the user.
    User,
    /// A message from the assistant/model.
    Assistant,
}

/// A single message in a conversation.
#[derive(Debug, Clone)]
pub struct Message {
    /// The role of the message sender.
    pub role: Role,
    /// The text content of the message.
    pub content: String,
}

impl Message {
    /// Creates a new user message.
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
            content: content.into(),
        }
    }

    /// Creates a new assistant message.
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
            content: content.into(),
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
}

/// A response from an LLM model.
#[derive(Debug, Clone)]
pub struct ModelResponse {
    /// The generated text output.
    pub text: String,
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
/// use rust_agent_kit::model::{LlmModel, ModelRequest, ModelResponse};
///
/// struct EchoModel;
///
/// #[async_trait]
/// impl LlmModel for EchoModel {
///     async fn generate(&self, request: ModelRequest) -> Result<ModelResponse, Error> {
///         let echo = request.messages.last().map(|m| m.content.clone()).unwrap_or_default();
///         Ok(ModelResponse { text: echo })
///     }
/// }
/// ```
#[async_trait]
pub trait LlmModel: Send + Sync {
    /// Generate a response for the given [`ModelRequest`].
    async fn generate(&self, request: ModelRequest) -> Result<ModelResponse, Error>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_user_sets_correct_role() {
        let msg = Message::user("hello");
        assert_eq!(msg.role, Role::User);
        assert_eq!(msg.content, "hello");
    }

    #[test]
    fn message_assistant_sets_correct_role() {
        let msg = Message::assistant("hi");
        assert_eq!(msg.role, Role::Assistant);
        assert_eq!(msg.content, "hi");
    }
}
