//! Tier 3 compute tier: sans-I/O state machine wrapping WasmiRuntime with a task queue.
//!
//! Accepts [`ComputeTierEvent`]s, enqueues compute tasks, and returns
//! [`ComputeTierAction`]s for the caller to execute (send replies, fetch modules).

// Scaffolding: types and methods are consumed only by tests until wired into NodeRuntime.
#![allow(dead_code)]

use std::collections::VecDeque;

use harmony_compute::{InstructionBudget, WasmiRuntime};

// ── Events (inbound) ──────────────────────────────────────────────────

/// Inbound events for the compute tier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComputeTierEvent {
    /// Execute an inline WASM module with the given input.
    ExecuteInline {
        query_id: u64,
        module: Vec<u8>,
        input: Vec<u8>,
    },
    /// Execute a WASM module identified by its content hash.
    ExecuteByCid {
        query_id: u64,
        module_cid: [u8; 32],
        input: Vec<u8>,
    },
    /// A previously requested module has been fetched successfully.
    ModuleFetched { cid: [u8; 32], module: Vec<u8> },
    /// A previously requested module fetch has failed.
    ModuleFetchFailed { cid: [u8; 32] },
}

// ── Actions (outbound) ────────────────────────────────────────────────

/// Outbound actions returned by the compute tier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComputeTierAction {
    /// Send a successful computation result back to the querier.
    SendReply { query_id: u64, payload: Vec<u8> },
    /// Request the storage layer to fetch a WASM module by CID.
    FetchModule { cid: [u8; 32] },
    /// Send an error response back to the querier.
    SendError { query_id: u64, error: String },
}

// ── Internal types (private) ──────────────────────────────────────────

/// A compute task waiting in the queue.
enum ComputeTask {
    /// Module bytes are available; ready to execute.
    Ready {
        query_id: u64,
        module: Vec<u8>,
        input: Vec<u8>,
    },
    /// Waiting for the module to be fetched from storage.
    WaitingForModule {
        query_id: u64,
        cid: [u8; 32],
        input: Vec<u8>,
    },
}

/// Tracks the currently executing computation.
struct ActiveExecution {
    query_id: u64,
}

// ── ComputeTier ───────────────────────────────────────────────────────

/// Sans-I/O state machine for Tier 3 (compute).
///
/// Wraps a [`WasmiRuntime`] with a task queue. The caller feeds events
/// via [`handle`](Self::handle) and drives execution via [`tick`](Self::tick).
pub struct ComputeTier {
    runtime: WasmiRuntime,
    queue: VecDeque<ComputeTask>,
    active: Option<ActiveExecution>,
    budget: InstructionBudget,
}

impl ComputeTier {
    /// Create a new compute tier with the given per-slice instruction budget.
    pub fn new(budget: InstructionBudget) -> Self {
        Self {
            runtime: WasmiRuntime::new(),
            queue: VecDeque::new(),
            active: None,
            budget,
        }
    }

    /// Process an inbound event, returning any immediate actions.
    pub fn handle(&mut self, _event: ComputeTierEvent) -> Vec<ComputeTierAction> {
        Vec::new()
    }

    /// Run one iteration of the compute loop, returning any resulting actions.
    pub fn tick(&mut self) -> Vec<ComputeTierAction> {
        Vec::new()
    }

    /// Number of tasks waiting in the queue.
    pub fn queue_len(&self) -> usize {
        self.queue.len()
    }

    /// Whether there is an active (in-progress) execution.
    pub fn has_active(&self) -> bool {
        self.active.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tier() -> ComputeTier {
        ComputeTier::new(InstructionBudget { fuel: 100_000 })
    }

    #[test]
    fn tick_with_no_tasks_returns_nothing() {
        let mut tier = make_tier();
        let actions = tier.tick();
        assert!(actions.is_empty());
    }

    #[test]
    fn queue_starts_empty() {
        let tier = make_tier();
        assert_eq!(tier.queue_len(), 0);
        assert!(!tier.has_active());
    }
}
