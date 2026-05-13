// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;

use serde::{Deserialize, Serialize};

use crate::error::Error;

/// Describes a tool to the model: its name, purpose, and parameter schema.
///
/// `ToolDefinition` is the contract between the agent and the LLM. It is
/// returned by [`Tool::definition`] and forwarded to the model on every run.
/// It is never stored in [`Agent`] — definitions live in the [`ToolRegistry`]
/// alongside their implementations.
///
/// [`Agent`]: crate::Agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// The tool name the model uses to invoke it. Must match the key in the
    /// [`ToolRegistry`].
    pub name: String,
    /// A human-readable description that helps the model decide when to call
    /// this tool.
    pub description: String,
    /// JSON Schema object describing the arguments the model must pass.
    pub parameters: serde_json::Value,
}

/// A callable tool that an agent can invoke during inference.
///
/// Implement this trait to expose executable logic to the agentic loop.
/// The [`definition`](Tool::definition) method tells the model what the tool
/// does; [`call`](Tool::call) executes it when the model requests it.
///
/// # Examples
///
/// ```no_run
/// use async_trait::async_trait;
/// use agent_rig::error::Error;
/// use agent_rig::tool::{Tool, ToolDefinition};
/// use serde_json::{Value, json};
///
/// struct AddTool;
///
/// #[async_trait]
/// impl Tool for AddTool {
///     fn definition(&self) -> ToolDefinition {
///         ToolDefinition {
///             name: "add".to_string(),
///             description: "Adds two integers and returns the sum.".to_string(),
///             parameters: json!({
///                 "type": "object",
///                 "properties": {
///                     "a": { "type": "integer" },
///                     "b": { "type": "integer" }
///                 },
///                 "required": ["a", "b"]
///             }),
///         }
///     }
///
///     async fn call(&self, args: Value) -> Result<Value, Error> {
///         let a = args["a"].as_i64().unwrap_or(0);
///         let b = args["b"].as_i64().unwrap_or(0);
///         Ok(json!({ "result": a + b }))
///     }
/// }
/// ```
#[async_trait]
pub trait Tool: Send + Sync {
    /// Returns the definition that describes this tool to the model.
    fn definition(&self) -> ToolDefinition;

    /// Executes the tool with the JSON arguments the model provided.
    ///
    /// `args` is the raw JSON object from the model's tool call. Returns a
    /// JSON value that is sent back to the model as the tool result.
    async fn call(&self, args: serde_json::Value) -> Result<serde_json::Value, Error>;
}

/// A collection of [`Tool`] implementations keyed by name.
///
/// `ToolRegistry` is independent of any [`AgentRunner`] so a single registry
/// can be shared across multiple runners via [`Arc`].
///
/// # Examples
///
/// ```no_run
/// use std::sync::Arc;
/// use agent_rig::tool::ToolRegistry;
///
/// # struct MyTool;
/// # use async_trait::async_trait;
/// # use agent_rig::tool::{Tool, ToolDefinition};
/// # use agent_rig::error::Error;
/// # #[async_trait]
/// # impl Tool for MyTool {
/// #     fn definition(&self) -> ToolDefinition { unimplemented!() }
/// #     async fn call(&self, _: serde_json::Value) -> Result<serde_json::Value, Error> { unimplemented!() }
/// # }
/// let registry = Arc::new(
///     ToolRegistry::new()
///         .register(Box::new(MyTool))
/// );
/// ```
///
/// [`AgentRunner`]: crate::AgentRunner
pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>,
}

impl ToolRegistry {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Registers a tool, keyed by its [`ToolDefinition::name`].
    ///
    /// Consumes and returns `self` for builder-style chaining.
    pub fn register(mut self, tool: Box<dyn Tool>) -> Self {
        let name = tool.definition().name.clone();
        self.tools.insert(name, tool);
        self
    }

    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools.values().map(|t| t.definition()).collect()
    }

    /// Returns the tool registered under `name`, or `None` if not found.
    pub(crate) fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.get(name).map(|t| t.as_ref())
    }

    /// Returns `true` if a tool with the given name is registered.
    pub(crate) fn contains(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// Returns an empty registry wrapped in `Arc`.
    pub(crate) fn empty() -> Arc<Self> {
        Arc::new(Self::new())
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}
