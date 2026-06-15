// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

//! Tool and registry types used by [`AgentRunner`](crate::runner::AgentRunner).
//!
//! - [`Tool`] is the trait an application implements to expose a callable
//!   function to the model.
//! - [`AgentTool`] wraps a sub-agent so it can be invoked through the same
//!   tool-call mechanism.
//! - [`ToolRegistry`] stores both kinds keyed by name; the runner looks up
//!   each tool call against it.

mod agent_tool;
mod approval;
mod registry;
mod tool;

pub use agent_tool::AgentTool;
pub use approval::ApprovalRequest;
pub use registry::ToolRegistry;
pub use tool::{ProgressDetails, ProgressReporter, Tool, ToolDefinition};
