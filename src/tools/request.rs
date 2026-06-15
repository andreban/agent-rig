use serde_json::Value;
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

#[derive(Debug)]
pub struct ToolCallRequest {
    pub tool_call_id: String,
    pub tool_name: String,
    pub args: Value,
    pub cancellation_token: CancellationToken,
    resolver: oneshot::Sender<Value>,
}

impl ToolCallRequest {
    pub fn new(
        tool_call_id: String,
        tool_name: String,
        args: Value,
        cancellation_token: CancellationToken,
        resolver: oneshot::Sender<Value>,
    ) -> Self {
        Self {
            tool_call_id,
            tool_name,
            args,
            cancellation_token,
            resolver,
        }
    }
    pub fn resolve(self, result: impl Into<Value>) {
        let _ = self.resolver.send(result.into());
    }
}
