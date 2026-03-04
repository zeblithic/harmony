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
    pub fn handle(&mut self, event: ComputeTierEvent) -> Vec<ComputeTierAction> {
        match event {
            ComputeTierEvent::ExecuteInline {
                query_id,
                module,
                input,
            } => {
                self.queue.push_back(ComputeTask::Ready {
                    query_id,
                    module,
                    input,
                });
                Vec::new()
            }
            ComputeTierEvent::ExecuteByCid {
                query_id,
                module_cid,
                input,
            } => {
                self.queue.push_back(ComputeTask::WaitingForModule {
                    query_id,
                    cid: module_cid,
                    input,
                });
                vec![ComputeTierAction::FetchModule { cid: module_cid }]
            }
            ComputeTierEvent::ModuleFetched { cid, module } => {
                if let Some(pos) = self.queue.iter().position(|t| {
                    matches!(t, ComputeTask::WaitingForModule { cid: c, .. } if *c == cid)
                }) {
                    if let Some(ComputeTask::WaitingForModule {
                        query_id, input, ..
                    }) = self.queue.remove(pos)
                    {
                        self.queue.push_back(ComputeTask::Ready {
                            query_id,
                            module,
                            input,
                        });
                    }
                }
                Vec::new()
            }
            ComputeTierEvent::ModuleFetchFailed { cid } => {
                let mut actions = Vec::new();
                if let Some(pos) = self.queue.iter().position(|t| {
                    matches!(t, ComputeTask::WaitingForModule { cid: c, .. } if *c == cid)
                }) {
                    if let Some(ComputeTask::WaitingForModule { query_id, .. }) =
                        self.queue.remove(pos)
                    {
                        actions.push(ComputeTierAction::SendError {
                            query_id,
                            error: format!("module not found: {}", hex::encode(cid)),
                        });
                    }
                }
                actions
            }
        }
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

    /// WAT module: reads two i32s from input, writes their sum as output.
    pub(crate) const ADD_WAT: &str = r#"
    (module
      (memory (export "memory") 1)
      (func (export "compute") (param $input_ptr i32) (param $input_len i32) (result i32)
        (i32.store
          (i32.add (local.get $input_ptr) (local.get $input_len))
          (i32.add
            (i32.load (local.get $input_ptr))
            (i32.load (i32.add (local.get $input_ptr) (i32.const 4)))))
        (i32.const 4)))
"#;

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

    #[test]
    fn handle_execute_inline_queues_ready_task() {
        let mut tier = make_tier();
        let actions = tier.handle(ComputeTierEvent::ExecuteInline {
            query_id: 1,
            module: ADD_WAT.as_bytes().to_vec(),
            input: vec![1, 2, 3, 4],
        });
        assert!(actions.is_empty());
        assert_eq!(tier.queue_len(), 1);
    }

    #[test]
    fn handle_execute_by_cid_emits_fetch() {
        let mut tier = make_tier();
        let cid = [0xAB; 32];
        let actions = tier.handle(ComputeTierEvent::ExecuteByCid {
            query_id: 2,
            module_cid: cid,
            input: vec![5, 6],
        });
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0], ComputeTierAction::FetchModule { cid });
        assert_eq!(tier.queue_len(), 1);
    }

    #[test]
    fn handle_module_fetched_promotes_to_ready() {
        let mut tier = make_tier();
        let cid = [0xCD; 32];
        // First, enqueue a task waiting for a module.
        tier.handle(ComputeTierEvent::ExecuteByCid {
            query_id: 3,
            module_cid: cid,
            input: vec![7, 8],
        });
        assert_eq!(tier.queue_len(), 1);

        // Now deliver the module.
        let actions = tier.handle(ComputeTierEvent::ModuleFetched {
            cid,
            module: ADD_WAT.as_bytes().to_vec(),
        });
        assert!(actions.is_empty());
        assert_eq!(tier.queue_len(), 1); // still 1, but promoted to Ready
    }

    #[test]
    fn handle_module_fetch_failed_returns_error() {
        let mut tier = make_tier();
        let cid = [0xEF; 32];
        // Enqueue a task waiting for a module.
        tier.handle(ComputeTierEvent::ExecuteByCid {
            query_id: 4,
            module_cid: cid,
            input: vec![9, 10],
        });
        assert_eq!(tier.queue_len(), 1);

        // Signal fetch failure.
        let actions = tier.handle(ComputeTierEvent::ModuleFetchFailed { cid });
        assert_eq!(actions.len(), 1);
        assert_eq!(
            actions[0],
            ComputeTierAction::SendError {
                query_id: 4,
                error: format!("module not found: {}", hex::encode(cid)),
            }
        );
        assert_eq!(tier.queue_len(), 0);
    }
}
