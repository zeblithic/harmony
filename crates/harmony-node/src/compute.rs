//! Tier 3 compute tier: sans-I/O state machine wrapping WasmiRuntime with a task queue.
//!
//! Accepts [`ComputeTierEvent`]s, enqueues compute tasks, and returns
//! [`ComputeTierAction`]s for the caller to execute (send replies, fetch modules).

use std::collections::VecDeque;
use std::sync::Arc;

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
    /// Content data has been fetched for a suspended compute task.
    ContentFetched { cid: [u8; 32], data: Vec<u8> },
    /// Content fetch failed for a suspended compute task.
    ContentFetchFailed { cid: [u8; 32] },
}

// ── Actions (outbound) ────────────────────────────────────────────────

/// Outbound actions returned by the compute tier.
///
/// Wire format encoding (`[0x00]` success / `[0x01]` error tags) is fully
/// handled inside [`ComputeTier`] — `SendReply` payloads are opaque to callers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComputeTierAction {
    /// Send a computation result (success or error) back to the querier.
    /// Payload is already wire-encoded with the appropriate tag byte.
    SendReply { query_id: u64, payload: Vec<u8> },
    /// Request the storage layer to fetch a WASM module by CID.
    FetchModule { cid: [u8; 32] },
    /// Request the storage/network layer to fetch content by CID for a suspended task.
    FetchContent { query_id: u64, cid: [u8; 32] },
}

// ── Internal types (private) ──────────────────────────────────────────

/// A compute task waiting in the queue.
enum ComputeTask {
    /// Module bytes are available; ready to execute.
    /// Module uses `Arc` so that multiple tasks waiting for the same CID
    /// can share the bytes without cloning up to 10 MB per task.
    Ready {
        query_id: u64,
        module: Arc<Vec<u8>>,
        input: Vec<u8>,
    },
    /// Waiting for the module to be fetched from storage.
    WaitingForModule {
        query_id: u64,
        cid: [u8; 32],
        input: Vec<u8>,
    },
    /// Execution suspended waiting for content to be fetched.
    /// The execution state lives in WasmiRuntime.session (HostTrap pending).
    WaitingForContent { query_id: u64, cid: [u8; 32] },
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
                    module: Arc::new(module),
                    input,
                });
                Vec::new()
            }
            ComputeTierEvent::ExecuteByCid {
                query_id,
                module_cid,
                input,
            } => {
                // Only emit FetchModule if no task is already waiting for this CID.
                let already_waiting = self.queue.iter().any(|t| {
                    matches!(t, ComputeTask::WaitingForModule { cid, .. } if *cid == module_cid)
                });
                self.queue.push_back(ComputeTask::WaitingForModule {
                    query_id,
                    cid: module_cid,
                    input,
                });
                if already_waiting {
                    Vec::new()
                } else {
                    vec![ComputeTierAction::FetchModule { cid: module_cid }]
                }
            }
            ComputeTierEvent::ModuleFetched { cid, module } => {
                // Promote ALL tasks waiting for this CID, preserving FIFO order.
                // Wrap module in Arc so multiple promoted tasks share the bytes.
                let module = Arc::new(module);
                let old_queue = std::mem::take(&mut self.queue);
                for task in old_queue {
                    match task {
                        ComputeTask::WaitingForModule {
                            cid: c,
                            query_id,
                            input,
                        } if c == cid => {
                            self.queue.push_back(ComputeTask::Ready {
                                query_id,
                                module: Arc::clone(&module),
                                input,
                            });
                        }
                        other => self.queue.push_back(other),
                    }
                }
                Vec::new()
            }
            ComputeTierEvent::ModuleFetchFailed { cid } => {
                // Fail ALL tasks waiting for this CID, preserving error order.
                let mut actions = Vec::new();
                let old_queue = std::mem::take(&mut self.queue);
                for task in old_queue {
                    match task {
                        ComputeTask::WaitingForModule {
                            cid: c, query_id, ..
                        } if c == cid => {
                            let error_msg =
                                format!("module not found: {}", hex::encode(cid));
                            let mut payload = vec![0x01];
                            payload.extend_from_slice(error_msg.as_bytes());
                            actions.push(ComputeTierAction::SendReply {
                                query_id,
                                payload,
                            });
                        }
                        other => self.queue.push_back(other),
                    }
                }
                actions
            }
            ComputeTierEvent::ContentFetched { cid, data } => {
                // Find the WaitingForContent task matching this CID.
                let pos = self.queue.iter().position(|t| {
                    matches!(t, ComputeTask::WaitingForContent { cid: c, .. } if *c == cid)
                });
                let Some(pos) = pos else {
                    return Vec::new(); // No matching task — drop silently.
                };
                let Some(ComputeTask::WaitingForContent { query_id, .. }) =
                    self.queue.remove(pos)
                else {
                    return Vec::new();
                };

                // Resume the runtime with the fetched content.
                self.active = Some(ActiveExecution { query_id });
                let result = self.runtime.resume_with_io(
                    harmony_compute::IOResponse::ContentReady { data },
                    self.budget,
                );
                self.handle_compute_result(result)
            }
            ComputeTierEvent::ContentFetchFailed { cid } => {
                let pos = self.queue.iter().position(|t| {
                    matches!(t, ComputeTask::WaitingForContent { cid: c, .. } if *c == cid)
                });
                let Some(pos) = pos else {
                    return Vec::new();
                };
                let Some(ComputeTask::WaitingForContent { query_id, .. }) =
                    self.queue.remove(pos)
                else {
                    return Vec::new();
                };

                // Resume with ContentNotFound — the module will see -1.
                self.active = Some(ActiveExecution { query_id });
                let result = self.runtime.resume_with_io(
                    harmony_compute::IOResponse::ContentNotFound,
                    self.budget,
                );
                self.handle_compute_result(result)
            }
        }
    }

    /// Run one iteration of the compute loop, returning any resulting actions.
    ///
    /// If there is an active (yielded) execution, resumes it with the configured
    /// fuel budget. Otherwise, dequeues the next [`ComputeTask::Ready`] task and
    /// begins execution. `WaitingForModule` tasks are skipped — they cannot
    /// execute until their module arrives, and blocking behind them would cause
    /// head-of-line blocking for ready inline tasks.
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
    /// - `Failed` -> `SendReply` with `[0x01, error_utf8...]` payload
    /// - `NeedsIO` -> `WaitingForContent` in queue + `FetchContent` action
    ///
    /// All wire format encoding (`[0x00]` success / `[0x01]` error tags) is
    /// consolidated here — callers treat payloads as opaque bytes.
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
                let mut payload = vec![0x01];
                payload.extend_from_slice(format!("{error}").as_bytes());
                vec![ComputeTierAction::SendReply { query_id, payload }]
            }
            ComputeResult::NeedsIO { request } => {
                match request {
                    harmony_compute::IORequest::FetchContent { cid } => {
                        self.queue.push_back(ComputeTask::WaitingForContent {
                            query_id,
                            cid,
                        });
                        self.active = None;
                        vec![ComputeTierAction::FetchContent { query_id, cid }]
                    }
                }
            }
        }
    }

    /// Number of tasks waiting in the queue.
    pub fn queue_len(&self) -> usize {
        self.queue.len()
    }

    /// Whether there is an active (in-progress) execution.
    #[cfg(test)]
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
        if let ComputeTierAction::SendReply { query_id, payload } = &actions[0] {
            assert_eq!(*query_id, 4);
            assert_eq!(payload[0], 0x01);
            let error_msg = std::str::from_utf8(&payload[1..]).unwrap();
            assert!(error_msg.contains("module not found"));
        } else {
            panic!("expected SendReply with error tag, got: {:?}", actions[0]);
        }
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
                ComputeTierAction::SendReply { query_id: 20, payload } if payload[0] == 0x01
            ),
            "expected SendReply with error tag for query_id 20, got: {actions:?}"
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

    #[test]
    fn duplicate_cid_requests_emit_single_fetch() {
        let mut tier = make_tier();
        let cid = [0xAA; 32];

        // First request for this CID: should emit FetchModule.
        let actions = tier.handle(ComputeTierEvent::ExecuteByCid {
            query_id: 50,
            module_cid: cid,
            input: vec![1],
        });
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0], ComputeTierAction::FetchModule { cid });

        // Second request for same CID: should NOT emit FetchModule.
        let actions = tier.handle(ComputeTierEvent::ExecuteByCid {
            query_id: 51,
            module_cid: cid,
            input: vec![2],
        });
        assert!(actions.is_empty(), "duplicate CID should not re-fetch");
        assert_eq!(tier.queue_len(), 2);
    }

    #[test]
    fn module_fetched_promotes_all_waiting_tasks() {
        let mut tier = make_tier();
        let cid = [0xBB; 32];

        // Two queries for the same CID.
        tier.handle(ComputeTierEvent::ExecuteByCid {
            query_id: 60,
            module_cid: cid,
            input: {
                let mut v = Vec::new();
                v.extend_from_slice(&5i32.to_le_bytes());
                v.extend_from_slice(&3i32.to_le_bytes());
                v
            },
        });
        tier.handle(ComputeTierEvent::ExecuteByCid {
            query_id: 61,
            module_cid: cid,
            input: {
                let mut v = Vec::new();
                v.extend_from_slice(&10i32.to_le_bytes());
                v.extend_from_slice(&20i32.to_le_bytes());
                v
            },
        });
        assert_eq!(tier.queue_len(), 2);

        // Deliver the module — both should be promoted.
        tier.handle(ComputeTierEvent::ModuleFetched {
            cid,
            module: ADD_WAT.as_bytes().to_vec(),
        });
        assert_eq!(tier.queue_len(), 2); // both promoted to Ready

        // Tick twice — both should complete.
        let a1 = tier.tick();
        assert_eq!(a1.len(), 1);
        assert!(matches!(&a1[0], ComputeTierAction::SendReply { query_id: 60, .. }));

        let a2 = tier.tick();
        assert_eq!(a2.len(), 1);
        assert!(matches!(&a2[0], ComputeTierAction::SendReply { query_id: 61, .. }));
    }

    #[test]
    fn module_fetch_failed_errors_all_waiting_tasks() {
        let mut tier = make_tier();
        let cid = [0xCC; 32];

        // Two queries for the same CID.
        tier.handle(ComputeTierEvent::ExecuteByCid {
            query_id: 70,
            module_cid: cid,
            input: vec![1],
        });
        tier.handle(ComputeTierEvent::ExecuteByCid {
            query_id: 71,
            module_cid: cid,
            input: vec![2],
        });
        assert_eq!(tier.queue_len(), 2);

        // Signal fetch failure — both should get errors.
        let actions = tier.handle(ComputeTierEvent::ModuleFetchFailed { cid });
        assert_eq!(actions.len(), 2);
        assert!(matches!(&actions[0], ComputeTierAction::SendReply { query_id: 70, payload } if payload[0] == 0x01));
        assert!(matches!(&actions[1], ComputeTierAction::SendReply { query_id: 71, payload } if payload[0] == 0x01));
        assert_eq!(tier.queue_len(), 0);
    }

    // ── Content I/O tests ───────────────────────────────────────────────

    /// WAT module that calls harmony.fetch_content.
    /// Input: [cid: 32 bytes]
    /// Output: [result_code: i32 LE] [fetched_data if result > 0]
    pub(crate) const FETCH_WAT: &str = r#"
    (module
      (import "harmony" "fetch_content" (func $fetch (param i32 i32 i32) (result i32)))
      (memory (export "memory") 1)
      (func (export "compute") (param $input_ptr i32) (param $input_len i32) (result i32)
        (local $result i32)
        (local $out_ptr i32)
        (local.set $out_ptr (i32.add (local.get $input_ptr) (local.get $input_len)))
        (local.set $result
          (call $fetch
            (local.get $input_ptr)
            (i32.add (local.get $out_ptr) (i32.const 4))
            (i32.const 1024)))
        (i32.store (local.get $out_ptr) (local.get $result))
        (if (result i32) (i32.gt_s (local.get $result) (i32.const 0))
          (then (i32.add (i32.const 4) (local.get $result)))
          (else (i32.const 4)))))
"#;

    #[test]
    fn needs_io_emits_fetch_content() {
        let mut tier = make_tier();
        let cid = [0xAB; 32];

        tier.handle(ComputeTierEvent::ExecuteInline {
            query_id: 100,
            module: FETCH_WAT.as_bytes().to_vec(),
            input: cid.to_vec(),
        });

        let actions = tier.tick();
        assert_eq!(actions.len(), 1);
        assert!(
            matches!(&actions[0], ComputeTierAction::FetchContent { query_id: 100, cid: c } if *c == cid),
            "expected FetchContent for the CID, got: {actions:?}"
        );

        assert!(!tier.has_active());
        assert_eq!(tier.queue_len(), 1); // WaitingForContent in queue
    }

    #[test]
    fn content_fetched_resumes_task() {
        let mut tier = make_tier();
        let cid = [0xCD; 32];
        let content = b"fetched data";

        tier.handle(ComputeTierEvent::ExecuteInline {
            query_id: 101,
            module: FETCH_WAT.as_bytes().to_vec(),
            input: cid.to_vec(),
        });

        let actions = tier.tick();
        assert_eq!(actions.len(), 1);
        assert!(matches!(&actions[0], ComputeTierAction::FetchContent { .. }));

        let actions = tier.handle(ComputeTierEvent::ContentFetched {
            cid,
            data: content.to_vec(),
        });

        assert_eq!(actions.len(), 1);
        if let ComputeTierAction::SendReply { query_id, payload } = &actions[0] {
            assert_eq!(*query_id, 101);
            assert_eq!(payload[0], 0x00, "success tag");
            let result_code = i32::from_le_bytes(payload[1..5].try_into().unwrap());
            assert_eq!(result_code, content.len() as i32);
            assert_eq!(&payload[5..], content);
        } else {
            panic!("expected SendReply, got: {actions:?}");
        }
    }

    #[test]
    fn content_fetch_failed_returns_error_to_module() {
        let mut tier = make_tier();
        let cid = [0xEF; 32];

        tier.handle(ComputeTierEvent::ExecuteInline {
            query_id: 102,
            module: FETCH_WAT.as_bytes().to_vec(),
            input: cid.to_vec(),
        });

        tier.tick();

        let actions = tier.handle(ComputeTierEvent::ContentFetchFailed { cid });

        assert_eq!(actions.len(), 1);
        if let ComputeTierAction::SendReply { query_id, payload } = &actions[0] {
            assert_eq!(*query_id, 102);
            assert_eq!(payload[0], 0x00, "success tag — module completed normally");
            let result_code = i32::from_le_bytes(payload[1..5].try_into().unwrap());
            assert_eq!(result_code, -1, "not-found result code");
        } else {
            panic!("expected SendReply, got: {actions:?}");
        }
    }

    #[test]
    fn waiting_for_content_does_not_block_queue() {
        let mut tier = make_tier();
        let cid = [0x11; 32];

        // First: module that calls fetch_content (will block on IO).
        tier.handle(ComputeTierEvent::ExecuteInline {
            query_id: 110,
            module: FETCH_WAT.as_bytes().to_vec(),
            input: cid.to_vec(),
        });

        // Second: simple add module (should not be blocked).
        let mut input = Vec::new();
        input.extend_from_slice(&4i32.to_le_bytes());
        input.extend_from_slice(&6i32.to_le_bytes());
        tier.handle(ComputeTierEvent::ExecuteInline {
            query_id: 111,
            module: ADD_WAT.as_bytes().to_vec(),
            input,
        });

        // First tick: executes fetch module → NeedsIO → WaitingForContent.
        let actions = tier.tick();
        assert!(matches!(&actions[0], ComputeTierAction::FetchContent { .. }));
        assert!(!tier.has_active());

        // Second tick: should skip WaitingForContent and execute the add module.
        let actions = tier.tick();
        assert_eq!(actions.len(), 1);
        if let ComputeTierAction::SendReply { query_id, payload } = &actions[0] {
            assert_eq!(*query_id, 111);
            assert_eq!(payload[0], 0x00);
            let sum = i32::from_le_bytes(payload[1..5].try_into().unwrap());
            assert_eq!(sum, 10);
        } else {
            panic!("expected add result, got: {actions:?}");
        }
    }
}
