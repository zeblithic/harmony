//! Structured agent-to-agent messaging protocol.
//!
//! Defines typed message envelopes for task submission, result delivery,
//! and capacity advertisement. Agents are identified by Harmony address
//! hash and communicate via Zenoh query-reply.

pub mod task_id;
pub mod types;
pub mod wire;

pub use task_id::generate_task_id;
pub use types::{
    AgentCapacity, AgentResult, AgentStatus, AgentTask, StreamChunk, TaskContext, TaskStatus,
};
pub use wire::{
    decode_capacity, decode_result, decode_task, encode_capacity, encode_result, encode_task,
    AgentError,
};
