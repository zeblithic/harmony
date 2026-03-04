//! Tier 3 compute tier: sans-I/O state machine wrapping WasmiRuntime with a task queue.
//!
//! Accepts [`ComputeTierEvent`]s, enqueues compute tasks, and returns
//! [`ComputeTierAction`]s for the caller to execute (send replies, fetch modules).

// Scaffolding: types and methods are consumed only by tests until wired into NodeRuntime.
#![allow(dead_code)]

use std::collections::VecDeque;

use harmony_compute::{ComputeRuntime, InstructionBudget, WasmiRuntime};

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
                if let Some(pos) = self.queue.iter().position(
                    |t| matches!(t, ComputeTask::WaitingForModule { cid: c, .. } if *c == cid),
                ) {
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
                if let Some(pos) = self.queue.iter().position(
                    |t| matches!(t, ComputeTask::WaitingForModule { cid: c, .. } if *c == cid),
                ) {
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
    ///
    /// If there is an active (yielded) execution, resumes it with the configured
    /// fuel budget. Otherwise, dequeues the next [`ComputeTask::Ready`] task and
    /// begins execution.
    pub fn tick(&mut self) -> Vec<ComputeTierAction> {
        // If there's an active execution, resume it.
        if self.active.is_some() {
            let result = self.runtime.resume(self.budget);
            return self.handle_compute_result(result);
        }

        // Otherwise, try to dequeue the next Ready task.
        let ready_pos = self
            .queue
            .iter()
            .position(|t| matches!(t, ComputeTask::Ready { .. }));
        let Some(pos) = ready_pos else {
            return Vec::new();
        };
        let Some(ComputeTask::Ready {
            query_id,
            module,
            input,
        }) = self.queue.remove(pos)
        else {
            return Vec::new();
        };

        self.active = Some(ActiveExecution { query_id });
        let result = self.runtime.execute(&module, &input, self.budget);
        self.handle_compute_result(result)
    }

    /// Process the result of a WASM execution or resumption.
    ///
    /// Maps [`harmony_compute::ComputeResult`] variants to [`ComputeTierAction`]s:
    /// - `Complete` -> `SendReply` with `[0x00, output...]` payload
    /// - `Yielded` -> no action (auto-resume on next tick)
    /// - `Failed` -> `SendError`
    /// - `NeedsIO` -> `SendError` (not yet supported)
    fn handle_compute_result(
        &mut self,
        result: harmony_compute::ComputeResult,
    ) -> Vec<ComputeTierAction> {
        use harmony_compute::ComputeResult;

        let query_id = self
            .active
            .as_ref()
            .expect("must have active execution")
            .query_id;

        match result {
            ComputeResult::Complete { output } => {
                self.active = None;
                let mut payload = vec![0x00];
                payload.extend_from_slice(&output);
                vec![ComputeTierAction::SendReply { query_id, payload }]
            }
            ComputeResult::Yielded { .. } => {
                // Auto-resume on next tick — no action needed.
                Vec::new()
            }
            ComputeResult::Failed { error } => {
                self.active = None;
                vec![ComputeTierAction::SendError {
                    query_id,
                    error: format!("{error}"),
                }]
            }
            ComputeResult::NeedsIO { .. } => {
                self.active = None;
                vec![ComputeTierAction::SendError {
                    query_id,
                    error: "NeedsIO not yet supported".into(),
                }]
            }
        }
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
pub(crate) mod tests {
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

    /// WAT module: loop `count` times, write count as output.
    pub(crate) const LOOP_WAT: &str = r#"
    (module
      (memory (export "memory") 1)
      (func (export "compute") (param $input_ptr i32) (param $input_len i32) (result i32)
        (local $count i32)
        (local $i i32)
        (local.set $count (i32.load (local.get $input_ptr)))
        (local.set $i (i32.const 0))
        (block $break
          (loop $loop
            (br_if $break (i32.ge_u (local.get $i) (local.get $count)))
            (local.set $i (i32.add (local.get $i) (i32.const 1)))
            (br $loop)))
        (i32.store
          (i32.add (local.get $input_ptr) (local.get $input_len))
          (local.get $i))
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

    // ── tick() / handle_compute_result() tests ──────────────────────────

    #[test]
    fn execute_inline_completes() {
        let mut tier = make_tier();
        // Input: two i32s in little-endian: 3 and 7
        let mut input = Vec::new();
        input.extend_from_slice(&3i32.to_le_bytes());
        input.extend_from_slice(&7i32.to_le_bytes());

        tier.handle(ComputeTierEvent::ExecuteInline {
            query_id: 10,
            module: ADD_WAT.as_bytes().to_vec(),
            input,
        });
        assert_eq!(tier.queue_len(), 1);

        let actions = tier.tick();
        assert_eq!(actions.len(), 1);

        // Expect SendReply with 0x00 prefix + sum (10) as LE i32 bytes.
        let mut expected_payload = vec![0x00];
        expected_payload.extend_from_slice(&10i32.to_le_bytes());
        assert_eq!(
            actions[0],
            ComputeTierAction::SendReply {
                query_id: 10,
                payload: expected_payload,
            }
        );
        assert!(!tier.has_active());
    }

    #[test]
    fn execute_invalid_module_returns_error() {
        let mut tier = make_tier();
        tier.handle(ComputeTierEvent::ExecuteInline {
            query_id: 20,
            module: b"not wasm".to_vec(),
            input: vec![],
        });

        let actions = tier.tick();
        assert_eq!(actions.len(), 1);
        assert!(
            matches!(
                &actions[0],
                ComputeTierAction::SendError { query_id: 20, .. }
            ),
            "expected SendError for query_id 20, got: {actions:?}"
        );
        assert!(!tier.has_active());
    }

    #[test]
    fn yielded_task_auto_resumes() {
        // Use a small fuel budget so the loop yields multiple times.
        let mut tier = ComputeTier::new(InstructionBudget { fuel: 10_000 });

        // Input: count = 50_000
        let input = 50_000_i32.to_le_bytes().to_vec();
        tier.handle(ComputeTierEvent::ExecuteInline {
            query_id: 30,
            module: LOOP_WAT.as_bytes().to_vec(),
            input,
        });

        // First tick should yield (empty actions, but active).
        let actions = tier.tick();
        assert!(
            actions.is_empty(),
            "first tick should yield, got: {actions:?}"
        );
        assert!(
            tier.has_active(),
            "should have active execution after yield"
        );

        // Loop ticks until SendReply appears.
        let mut tick_count = 1u32;
        let reply = loop {
            let actions = tier.tick();
            tick_count += 1;
            if !actions.is_empty() {
                break actions;
            }
            assert!(tick_count < 10_000, "too many ticks without completion");
        };

        // Should have taken multiple ticks.
        assert!(
            tick_count > 1,
            "should have yielded at least once, but completed in {tick_count} ticks"
        );

        // Verify the reply payload: 0x00 prefix + count (50_000) as LE i32 bytes.
        assert_eq!(reply.len(), 1);
        let mut expected_payload = vec![0x00];
        expected_payload.extend_from_slice(&50_000_i32.to_le_bytes());
        assert_eq!(
            reply[0],
            ComputeTierAction::SendReply {
                query_id: 30,
                payload: expected_payload,
            }
        );
        assert!(!tier.has_active());
    }

    #[test]
    fn multiple_tasks_queued_fifo() {
        let mut tier = make_tier();

        // Task 1: 3 + 7 = 10
        let mut input1 = Vec::new();
        input1.extend_from_slice(&3i32.to_le_bytes());
        input1.extend_from_slice(&7i32.to_le_bytes());
        tier.handle(ComputeTierEvent::ExecuteInline {
            query_id: 40,
            module: ADD_WAT.as_bytes().to_vec(),
            input: input1,
        });

        // Task 2: 100 + 200 = 300
        let mut input2 = Vec::new();
        input2.extend_from_slice(&100i32.to_le_bytes());
        input2.extend_from_slice(&200i32.to_le_bytes());
        tier.handle(ComputeTierEvent::ExecuteInline {
            query_id: 41,
            module: ADD_WAT.as_bytes().to_vec(),
            input: input2,
        });

        assert_eq!(tier.queue_len(), 2);

        // First tick: completes task 1.
        let actions = tier.tick();
        assert_eq!(actions.len(), 1);
        let mut expected1 = vec![0x00];
        expected1.extend_from_slice(&10i32.to_le_bytes());
        assert_eq!(
            actions[0],
            ComputeTierAction::SendReply {
                query_id: 40,
                payload: expected1,
            }
        );

        // Second tick: completes task 2.
        let actions = tier.tick();
        assert_eq!(actions.len(), 1);
        let mut expected2 = vec![0x00];
        expected2.extend_from_slice(&300i32.to_le_bytes());
        assert_eq!(
            actions[0],
            ComputeTierAction::SendReply {
                query_id: 41,
                payload: expected2,
            }
        );

        // Third tick: nothing left.
        let actions = tier.tick();
        assert!(actions.is_empty());
    }
}
