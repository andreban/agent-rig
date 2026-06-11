use std::collections::HashMap;

use serde::{Serialize, de::DeserializeOwned};
use serde_json::Value;

use crate::tools::{
    agent_tool::AgentTool,
    tool::{ErasedTool, Tool, ToolBridge, ToolDefinition},
};
/// One entry stored in a [`ToolRegistry`].
///
/// The registry holds two kinds of callables — plain [`Tool`] implementations
/// and sub-agents wrapped in [`AgentTool`]. The runner dispatches each
/// variant differently: plain tools resolve to a single JSON value, while
/// agents produce a stream of events that the parent forwards.
pub enum ToolRegistryEntry {
    /// A plain tool implementation. Stored behind an object-safe wrapper so
    /// the registry can hold tools with different typed `I`/`O` parameters
    /// in the same map.
    Tool(Box<dyn ErasedTool>),
    /// A sub-agent registered as a tool. Boxed to keep the enum compact —
    /// `AgentTool` is much larger than `Box<dyn ErasedTool>`.
    Agent(Box<AgentTool>),
}

impl ToolRegistryEntry {
    /// Returns the public [`ToolDefinition`] for this entry regardless of
    /// variant.
    pub fn definition(&self) -> &ToolDefinition {
        match self {
            ToolRegistryEntry::Tool(t) => t.definition(),
            ToolRegistryEntry::Agent(a) => a.definition(),
        }
    }

    pub fn title(&self, args: &Value) -> String {
        match self {
            ToolRegistryEntry::Tool(t) => t.title(args).to_string(),
            ToolRegistryEntry::Agent(a) => a.name().to_string(),
        }
    }
}

/// A collection of [`Tool`]s and [`AgentTool`]s keyed by name.
///
/// `ToolRegistry` is independent of any [`AgentRunner`](crate::runner::AgentRunner)
/// so a single registry can be shared across multiple runners via [`Arc`](std::sync::Arc).
/// Build one with the chained [`register`](Self::register) /
/// [`register_agent`](Self::register_agent) methods.
///
/// # Examples
///
/// ```no_run
/// use std::sync::Arc;
/// use agent_rig::tools::ToolRegistry;
///
/// # use async_trait::async_trait;
/// # use agent_rig::tools::{Tool, ToolDefinition};
/// # use agent_rig::error::Error;
/// # use schemars::json_schema;
/// # struct MyTool {
/// #     definition: ToolDefinition,
/// # }
/// # impl Default for MyTool {
/// #     fn default() -> Self {
/// #         Self {
/// #             definition: ToolDefinition {
/// #                 name: "noop".into(),
/// #                 description: "noop".into(),
/// #                 parameters: json_schema!({"type": "object"}),
/// #             },
/// #         }
/// #     }
/// # }
/// # #[async_trait]
/// # impl Tool<serde_json::Value, serde_json::Value> for MyTool {
/// #     fn definition(&self) -> &ToolDefinition {
/// #         &self.definition
/// #     }
/// #     async fn call(&self, _: serde_json::Value, _: tokio_util::sync::CancellationToken)
/// #         -> Result<serde_json::Value, Error> { Ok(serde_json::json!({})) }
/// # }
/// let registry = Arc::new(
///     ToolRegistry::new()
///         .register(MyTool::default())
/// );
/// ```
pub struct ToolRegistry {
    tools: HashMap<String, ToolRegistryEntry>,
}

impl ToolRegistry {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Registers a [`Tool`], keyed by its [`ToolDefinition::name`].
    ///
    /// Takes the tool by value and stores it behind an internal object-safe
    /// wrapper, which is what lets a single registry hold tools with
    /// different `I`/`O` types. Consumes and returns `self` for builder-style
    /// chaining. If a tool with the same name is already registered, it is
    /// overwritten.
    pub fn register<T, I, O>(mut self, tool: T) -> Self
    where
        T: Tool<I, O> + 'static,
        I: DeserializeOwned + Send + 'static,
        O: Serialize + Send + 'static,
    {
        let entry: Box<dyn ErasedTool> = Box::new(ToolBridge::new(tool));
        let name = entry.definition().name.clone();
        self.tools.insert(name, ToolRegistryEntry::Tool(entry));
        self
    }

    /// Registers a sub-agent as a tool, keyed by its [`ToolDefinition::name`].
    ///
    /// Consumes and returns `self` for builder-style chaining. If a tool with
    /// the same name is already registered, it is overwritten.
    pub fn register_agent(mut self, agent: AgentTool) -> Self {
        let name = agent.definition().name.clone();
        self.tools
            .insert(name, ToolRegistryEntry::Agent(Box::new(agent)));
        self
    }

    /// Returns every registered tool's [`ToolDefinition`].
    ///
    /// The order is unspecified.
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools
            .values()
            .map(|t| t.definition().clone())
            .collect()
    }

    /// Returns the tool registered under `name`, or `None` if not found.
    pub(crate) fn get(&self, name: &str) -> Option<&ToolRegistryEntry> {
        self.tools.get(name)
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;
    use async_trait::async_trait;
    use schemars::json_schema;
    use serde_json::{Value, json};

    struct StubTool {
        definition: ToolDefinition,
    }

    #[async_trait]
    impl Tool<Value, Value> for StubTool {
        fn definition(&self) -> &ToolDefinition {
            &self.definition
        }

        async fn call(
            &self,
            _args: Value,
            _cancel: tokio_util::sync::CancellationToken,
        ) -> Result<Value, Error> {
            Ok(json!({}))
        }
    }

    #[test]
    fn new_registry_is_empty() {
        let reg = ToolRegistry::new();
        assert!(reg.definitions().is_empty());
        assert!(reg.get("anything").is_none());
    }

    #[test]
    fn register_keys_by_definition_name() {
        let reg = ToolRegistry::new().register(StubTool {
            definition: ToolDefinition {
                name: "alpha".to_string(),
                description: "stub".to_string(),
                parameters: json_schema!({"type": "object"}),
            },
        });
        assert!(matches!(reg.get("alpha"), Some(ToolRegistryEntry::Tool(_))));
    }

    #[test]
    fn definitions_lists_every_registered_tool() {
        let reg = ToolRegistry::new()
            .register(StubTool {
                definition: ToolDefinition {
                    name: "alpha".to_string(),
                    description: "stub".to_string(),
                    parameters: json_schema!({"type": "object"}),
                },
            })
            .register(StubTool {
                definition: ToolDefinition {
                    name: "beta".to_string(),
                    description: "stub".to_string(),
                    parameters: json_schema!({"type": "object"}),
                },
            });
        let mut names: Vec<String> = reg.definitions().into_iter().map(|d| d.name).collect();
        names.sort();
        assert_eq!(names, vec!["alpha".to_string(), "beta".to_string()]);
    }

    #[test]
    fn last_registration_wins_on_name_collision() {
        let reg = ToolRegistry::new()
            .register(StubTool {
                definition: ToolDefinition {
                    name: "alpha".to_string(),
                    description: "stub".to_string(),
                    parameters: json_schema!({"type": "object"}),
                },
            })
            .register(StubTool {
                definition: ToolDefinition {
                    name: "alpha".to_string(),
                    description: "stub".to_string(),
                    parameters: json_schema!({"type": "object"}),
                },
            });
        // The registry is keyed by name; a second `register` with the same
        // name overwrites the first. Verifying via count so a future change
        // (e.g. rejecting duplicates) surfaces here.
        assert_eq!(reg.definitions().len(), 1);
    }

    #[test]
    fn register_agent_stores_an_agent_entry() {
        use crate::Agent;
        use crate::runner::AgentRunner;
        use crate::tools::agent_tool::AgentTool;
        use std::sync::Arc;

        // Build a runner with no model interactions — the test never invokes
        // the AgentTool, only verifies the registry classifies it as an Agent
        // entry.
        struct DummyModel;
        #[async_trait]
        impl crate::model::LlmModel for DummyModel {
            async fn generate(
                &self,
                _request: crate::model::ModelRequest,
            ) -> Result<crate::model::ModelResponse, Error> {
                unreachable!("not called in this test")
            }
        }

        let agent = Agent::builder().name("Child").instructions("noop").build();
        let runner = AgentRunner::new(Arc::new(DummyModel));
        let tool = AgentTool::new(
            ToolDefinition {
                name: "delegate".to_string(),
                description: "delegate to child".to_string(),
                parameters: json_schema!({"type": "object"}),
            },
            agent,
            runner,
        );

        let reg = ToolRegistry::new().register_agent(tool);
        assert!(matches!(
            reg.get("delegate"),
            Some(ToolRegistryEntry::Agent(_))
        ));
        assert_eq!(reg.definitions().len(), 1);
    }
}
