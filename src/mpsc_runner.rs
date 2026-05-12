use std::{pin::Pin, sync::Arc};

use futures_util::Stream;
use tokio::sync::mpsc::{self, Sender};

use crate::{model::LlmModel, tool::ToolRegistry};

pub enum AgentEvent {
    ToolCallStarted {
        name: String,
        args: serde_json::Value,
    },
    ToolCallFinished {
        name: String,
        result: serde_json::Value,
    },
    ThinkingDelta(String),
    TextDelta(String),
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

    pub fn run(&self, input: String) -> Pin<Box<dyn Stream<Item = String>>> {
        // Clone `self` outside the `stream!` macro block to prevent the generator from
        // capturing the non-'static `&self` reference, satisfying `'static` for the trait object.
        let cloned = self.clone();

        let stream = async_stream::stream! {
          let (tx, mut rx) = mpsc::channel::<String>(100);
          tokio::spawn(cloned.main_loop(tx, input));

          while let Some(message) = rx.recv().await {
            yield message;
          }
        };
        Box::pin(stream)
    }

    async fn main_loop(self, tx: Sender<String>, input: String) {
        let _ = tx.send(input).await;

        let cloned = tx.clone();
        tokio::spawn(async move {
            let _ = cloned.send("cloned".to_string()).await;
        });
    }
}
