//! Structured agent-to-agent messaging protocol.
//!
//! Defines typed message envelopes for task submission, result delivery,
//! and capacity advertisement. Agents are identified by Harmony address
//! hash and communicate via Zenoh query-reply.

pub mod types;
// pub mod wire;      // Task 2
// pub mod task_id;   // Task 3

pub use types::{
    AgentCapacity, AgentResult, AgentStatus, AgentTask, TaskContext, TaskStatus,
};
