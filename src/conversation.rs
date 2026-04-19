// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

use std::{
    pin::Pin,
    task::{Context, Poll},
};

use futures_util::{Stream, StreamExt};

use crate::{
    agent::Agent,
    error::Error,
    model::Message,
    runner::{AgentEvent, AgentResult, AgentRunner},
};

/// A stateful multi-turn conversation between a user and an [`Agent`].
///
/// `Conversation` wraps an [`AgentRunner`] and an [`Agent`], automatically
/// maintaining the message history across turns. After each completed turn the
/// user message and assistant reply are appended to the internal history, so
/// the next call sees the full context without any extra bookkeeping by the
/// caller.
///
/// The history can be inspected or modified at any time via [`history`] and
/// [`history_mut`], which allows callers to implement compression, trimming, or
/// any other history management strategy.
///
/// # Examples
///
/// ```no_run,ignore
/// use agent_rig::{Agent, AgentRunner};
/// use agent_rig::models::gemini::GeminiModel;
///
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let model = GeminiModel::builder("API_KEY", "gemini-2.5-flash").build();
/// let agent = Agent::builder()
///     .name("Assistant")
///     .instructions("You are a helpful assistant.")
///     .build();
/// let runner = AgentRunner::new(Box::new(model));
///
/// let mut conv = runner.conversation(&agent);
/// conv.run("My name is Alice.").await?;
/// let result = conv.run("What is my name?").await?;
/// println!("{}", result.output); // "Alice"
/// # Ok(())
/// # }
/// ```
///
/// [`history`]: Conversation::history
/// [`history_mut`]: Conversation::history_mut
pub struct Conversation<'a> {
    runner: &'a AgentRunner,
    agent: &'a Agent,
    history: Vec<Message>,
}

impl<'a> Conversation<'a> {
    pub(crate) fn new(runner: &'a AgentRunner, agent: &'a Agent) -> Self {
        Conversation {
            runner,
            agent,
            history: Vec::new(),
        }
    }

    /// Returns a shared reference to the conversation history.
    pub fn history(&self) -> &[Message] {
        &self.history
    }

    /// Returns a mutable reference to the conversation history.
    ///
    /// Use this to implement history compression, trimming, or injection of
    /// synthetic turns before the next call.
    pub fn history_mut(&mut self) -> &mut Vec<Message> {
        &mut self.history
    }

    /// Runs one turn of the conversation and returns the agent's reply.
    ///
    /// The user message and assistant reply are automatically appended to the
    /// internal history so that subsequent calls have full context.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Agent`] if the model loop ends without a text response,
    /// or if a declared tool has no registered implementation.
    pub async fn run(&mut self, input: &str) -> Result<AgentResult, Error> {
        let result = self
            .runner
            .run_builder(self.agent)
            .history(self.history.clone())
            .run(input)
            .await?;

        self.history.push(Message::user(input));
        self.history.push(Message::assistant(&result.output));
        Ok(result)
    }

    /// Streams events for one turn of the conversation.
    ///
    /// Returns a [`ConversationStream`] that yields [`AgentEvent`] values
    /// exactly like [`AgentRunner::run_stream`]. When the stream is fully
    /// consumed (returns `None`), the user message and assistant reply are
    /// automatically appended to the internal history.
    ///
    /// The stream must be fully consumed for history to be updated. If it is
    /// dropped early the history is not modified.
    ///
    /// # Errors
    ///
    /// Yields [`Error::Agent`] if the model loop ends without a text response,
    /// or if a declared tool has no registered implementation.
    pub fn run_stream<'b>(&'b mut self, input: &'b str) -> ConversationStream<'b> {
        let inner = self
            .runner
            .run_builder(self.agent)
            .history(self.history.clone())
            .run_stream(input);

        ConversationStream {
            inner: Box::pin(inner),
            history: &mut self.history,
            input: input.to_string(),
            reply: String::new(),
            done: false,
        }
    }
}

/// A stream produced by [`Conversation::run_stream`].
///
/// Wraps the inner event stream and automatically updates the conversation
/// history when the stream is exhausted. Drop the stream early to skip the
/// history update.
pub struct ConversationStream<'a> {
    inner: Pin<Box<dyn Stream<Item = Result<AgentEvent, Error>> + Send + 'a>>,
    history: &'a mut Vec<Message>,
    input: String,
    reply: String,
    done: bool,
}

impl Stream for ConversationStream<'_> {
    type Item = Result<AgentEvent, Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.done {
            return Poll::Ready(None);
        }

        match self.inner.poll_next_unpin(cx) {
            Poll::Ready(Some(Ok(event))) => {
                if let AgentEvent::TextDelta(ref chunk) = event {
                    self.reply.push_str(chunk);
                }
                Poll::Ready(Some(Ok(event)))
            }
            Poll::Ready(Some(Err(e))) => {
                self.done = true;
                Poll::Ready(Some(Err(e)))
            }
            Poll::Ready(None) => {
                self.done = true;
                let input = std::mem::take(&mut self.input);
                let reply = std::mem::take(&mut self.reply);
                self.history.push(Message::user(input));
                self.history.push(Message::assistant(reply));
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Agent,
        model::{LlmModel, ModelRequest, ModelResponse},
        runner::AgentRunner,
    };
    use async_trait::async_trait;

    struct MessageCountModel;

    #[async_trait]
    impl LlmModel for MessageCountModel {
        async fn generate(&self, request: ModelRequest) -> Result<ModelResponse, Error> {
            Ok(ModelResponse {
                text: Some(request.messages.len().to_string()),
                tool_calls: vec![],
                thinking: None,
            })
        }
    }

    fn agent() -> Agent {
        Agent::builder().name("T").instructions("i").build()
    }

    #[tokio::test]
    async fn run_accumulates_history() {
        let runner = AgentRunner::new(Box::new(MessageCountModel));
        let agent = agent();
        let mut conv = runner.conversation(&agent);

        // Turn 1: 1 message sent (the user input), reply = "1"
        let r1 = conv.run("turn1").await.unwrap();
        assert_eq!(r1.output, "1");

        // Turn 2: 2 history messages + 1 new = 3, reply = "3"
        let r2 = conv.run("turn2").await.unwrap();
        assert_eq!(r2.output, "3");
    }

    #[tokio::test]
    async fn history_mut_allows_trimming() {
        let runner = AgentRunner::new(Box::new(MessageCountModel));
        let agent = agent();
        let mut conv = runner.conversation(&agent);

        conv.run("turn1").await.unwrap(); // history: [user, assistant]
        conv.history_mut().clear(); // wipe it

        // Turn 2 sees only the new user message → "1"
        let r2 = conv.run("turn2").await.unwrap();
        assert_eq!(r2.output, "1");
    }

    #[tokio::test]
    async fn run_stream_accumulates_history() {
        let runner = AgentRunner::new(Box::new(MessageCountModel));
        let agent = agent();
        let mut conv = runner.conversation(&agent);

        // Consume the stream for turn 1.
        {
            let stream = conv.run_stream("turn1");
            futures_util::pin_mut!(stream);
            while stream.next().await.is_some() {}
        }

        // Turn 2 should see 2 history messages + new input = 3.
        let r2 = conv.run("turn2").await.unwrap();
        assert_eq!(r2.output, "3");
    }

    #[tokio::test]
    async fn run_stream_dropped_early_does_not_update_history() {
        let runner = AgentRunner::new(Box::new(MessageCountModel));
        let agent = agent();
        let mut conv = runner.conversation(&agent);

        // Drop the stream before consuming it.
        drop(conv.run_stream("turn1"));

        // Turn 2 should see only 1 message (no history from turn 1).
        let r2 = conv.run("turn2").await.unwrap();
        assert_eq!(r2.output, "1");
    }
}
