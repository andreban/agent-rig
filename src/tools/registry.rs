// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use crate::tools::tool::{Tool, ToolDefinition};

/// A collection of [`Tool`]s (including [`AgentTool`](crate::tools::AgentTool)s)
/// keyed by name.
///
/// `ToolRegistry` is independent of any [`AgentRunner`](crate::runner::AgentRunner)
/// so a single registry can be shared across multiple runners via [`Arc`](std::sync::Arc).
/// Build one with the chained [`register`](Self::register) method.
///
/// # Examples
///
/// ```no_run
/// use std::sync::Arc;
/// use agent_rig::tools::ToolRegistry;
///
/// # use async_trait::async_trait;
/// # use agent_rig::tools::{ProgressReporter, Tool, ToolDefinition};
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
/// # impl Tool for MyTool {
/// #     fn definition(&self) -> &ToolDefinition {
/// #         &self.definition
/// #     }
/// #     async fn call(&self, _: serde_json::Value, _: &dyn ProgressReporter, _: tokio_util::sync::CancellationToken)
/// #         -> Result<serde_json::Value, Error> { Ok(serde_json::json!({})) }
/// # }
/// let registry = Arc::new(
///     ToolRegistry::new()
///         .register(MyTool::default())
/// );
/// ```
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

    /// Registers a [`Tool`], keyed by its [`ToolDefinition::name`].
    ///
    /// Accepts anything that implements [`Tool`] — including typed
    /// [`SimpleTool`](crate::tools::SimpleTool)s via their blanket impl — and
    /// stores it behind `Box<dyn Tool>`, which is what lets a single registry
    /// hold tools of different shapes. Consumes and returns `self` for
    /// builder-style chaining. If a tool with the same name is already
    /// registered, it is overwritten.
    pub fn register<T>(mut self, tool: T) -> Self
    where
        T: Tool + 'static,
    {
        let entry: Box<dyn Tool> = Box::new(tool);
        let name = entry.definition().name.clone();
        self.tools.insert(name, entry);
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
    pub(crate) fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.get(name).map(|t| t.as_ref())
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
    use crate::tools::ProgressReporter;
    use async_trait::async_trait;
    use schemars::json_schema;
    use serde_json::{Value, json};

    struct StubTool {
        definition: ToolDefinition,
    }

    #[async_trait]
    impl Tool for StubTool {
        fn definition(&self) -> &ToolDefinition {
            &self.definition
        }

        async fn call(
            &self,
            _args: Value,
            _progress: &dyn ProgressReporter,
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
        assert!(reg.get("alpha").is_some());
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
    fn register_accepts_an_agent_tool() {
        use crate::Agent;
        use crate::runner::AgentRunner;
        use crate::tools::agent_tool::AgentTool;
        use std::sync::Arc;

        // Build a runner with no model interactions — the test never invokes
        // the AgentTool, only verifies an AgentTool registers like any other
        // tool and is retrievable by name.
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

        let reg = ToolRegistry::new().register(tool);
        assert!(reg.get("delegate").is_some());
        assert_eq!(reg.definitions().len(), 1);
    }
}
