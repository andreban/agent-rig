use std::{pin::Pin, sync::Arc};

use futures_util::{Stream, StreamExt, future::join_all};
use serde_json::Value;
use tokio::sync::mpsc::{self, Sender};
use tracing::debug;

use crate::{
    Agent,
    model::{LlmModel, Message, ModelRequest, ModelStreamChunk, ToolCall},
    tool::{ToolDefinition, ToolRegistry},
};

#[derive(Debug)]
pub enum AgentEvent {
    ToolCallStarted {
        name: String,
        args: serde_json::Value,
    },
    ToolCallFinished {
        name: String,
        result: serde_json::Value,
    },
    ToolCallError {
        name: String,
        error: crate::error::Error,
    },
    ThinkingDelta(String),
    TextDelta(String),
    Error(crate::error::Error),
}

pub struct RunnerEvent {
    pub thread_id: usize,
    pub depth: usize,
    pub agent_event: AgentEvent,
}

#[derive(Clone)]
pub struct MpscRunner {
    model: Arc<dyn LlmModel>,
    registry: Arc<ToolRegistry>,
}

impl MpscRunner {
    pub fn new(model: Arc<dyn LlmModel>) -> Self {
        MpscRunner {
            model,
            registry: Arc::new(ToolRegistry::new()),
        }
    }

    pub fn with_registry(model: Arc<dyn LlmModel>, registry: Arc<ToolRegistry>) -> Self {
        MpscRunner { model, registry }
    }

    pub fn run(
        &self,
        agent: Agent,
        thread: Vec<Message>,
    ) -> Pin<Box<dyn Stream<Item = AgentEvent>>> {
        debug!(agent = agent.name(), "starting run");
        // Clone `self` outside the `stream!` macro block to prevent the generator from
        // capturing the non-'static `&self` reference, satisfying `'static` for the trait object.
        let cloned = self.clone();

        let stream = async_stream::stream! {
          let (tx, mut rx) = mpsc::channel::<AgentEvent>(100);
          tokio::spawn(cloned.main_loop(tx, agent, thread));

          while let Some(message) = rx.recv().await {
            yield message;
          }
        };
        Box::pin(stream)
    }

    async fn main_loop(self, tx: Sender<AgentEvent>, agent: Agent, mut thread: Vec<Message>) {
        let tools: Vec<ToolDefinition> = self.registry.definitions();

        loop {
            let request = ModelRequest {
                messages: thread.clone(),
                system: Some(agent.instructions().to_string()),
                output_schema: agent.output_schema().cloned(),
                tools: tools.clone(),
            };

            let mut model_stream = self.model.generate_stream(request);
            let mut tool_calls: Vec<ToolCall> = Vec::new();
            while let Some(chunk) = model_stream.next().await {
                match chunk {
                    Ok(ModelStreamChunk::Thinking(t)) => {
                        let _ = tx.send(AgentEvent::ThinkingDelta(t)).await;
                    }
                    Ok(ModelStreamChunk::TextDelta(t)) => {
                        let _ = tx.send(AgentEvent::TextDelta(t)).await;
                    }
                    Ok(ModelStreamChunk::ToolCall(call)) => {
                        tool_calls.push(call);
                    }
                    Err(error) => {
                        let _ = tx.send(AgentEvent::Error(error)).await;
                        return;
                    }
                }
            }

            if tool_calls.is_empty() {
                break;
            }

            self.handle_tool_calls(&tx, tool_calls, &mut thread).await;
        }
    }

    async fn handle_tool_calls(
        &self,
        tx: &Sender<AgentEvent>,
        tool_calls: Vec<ToolCall>,
        thread: &mut Vec<Message>,
    ) {
        // Append the tool calls as a single assistant turn.
        thread.push(Message::tool_calls(tool_calls.clone()));

        // Each future runs the full lifecycle for one call: emit Started (if
        // the tool is registered), execute, emit Finished/Error. Hallucinated
        // calls skip Started but still produce a synthetic error result so the
        // assistant turn and tool-result messages remain paired.
        let tool_futures = tool_calls.into_iter().map(|call| async move {
            let result = match self.registry.get(&call.name) {
                Some(tool) => {
                    let _ = tx
                        .send(AgentEvent::ToolCallStarted {
                            name: call.name.clone(),
                            args: call.args.clone(),
                        })
                        .await;
                    tool.call(call.args.clone()).await
                }
                None => Ok(Value::from(format!("unknown tool: {}", call.name))),
            };

            let (event, result_value) = match result {
                Ok(value) => (
                    AgentEvent::ToolCallFinished {
                        name: call.name.clone(),
                        result: value.clone(),
                    },
                    value,
                ),
                Err(error) => {
                    let value = Value::from(format!("Error: {}", error));
                    (
                        AgentEvent::ToolCallError {
                            name: call.name.clone(),
                            error,
                        },
                        value,
                    )
                }
            };

            debug!(tool = call.name, "tool call complete");
            let _ = tx.send(event).await;
            (call, result_value)
        });

        // Run all calls concurrently. `join_all` preserves input order in the
        // returned Vec, so tool-result messages are appended in the same order
        // the model requested them — even though events may interleave.
        let results = join_all(tool_futures).await;
        for (call, result) in results {
            thread.push(Message::tool_result(
                call.id,
                call.name,
                result,
                call.provider_metadata,
            ));
        }
    }
}
