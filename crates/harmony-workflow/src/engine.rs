use std::collections::{HashMap, VecDeque};

use harmony_compute::{ComputeResult, ComputeRuntime, IORequest, IOResponse, InstructionBudget};

use crate::offload::ComputeHint;
use crate::types::{
    HistoryEvent, WorkflowAction, WorkflowEvent, WorkflowHistory, WorkflowId, WorkflowStatus,
};

/// Internal state for a single tracked workflow.
struct WorkflowState {
    status: WorkflowStatus,
    history: WorkflowHistory,
    #[allow(dead_code)]
    hint: ComputeHint,
    /// Module bytes for execution. Set on Submit, consumed on first tick.
    module_bytes: Option<Vec<u8>>,
    /// Saved runtime session for a workflow suspended on IO.
    saved_session: Option<Box<dyn std::any::Any>>,
    /// Cached IO responses from a prior execution (used during recovery replay).
    /// Populated from WorkflowHistory.events on recover(). Consumed during replay.
    replay_cache: HashMap<[u8; 32], Option<Vec<u8>>>,
    /// Cached output for completed workflows. Allows re-emitting WorkflowComplete
    /// when a duplicate submit arrives after the workflow has already finished.
    completed_output: Option<Vec<u8>>,
}

/// Sans-I/O workflow engine that orchestrates WASM workflow execution.
///
/// Owns a `ComputeRuntime` and manages workflow lifecycle: submit, execute,
/// yield/resume, complete. The caller drives execution by calling `tick()`
/// repeatedly and handling the returned `WorkflowAction`s.
///
/// Currently handles the happy path: submit inline workflow, tick until
/// complete, emit `WorkflowComplete`. IO handling is deferred to Task 4.
pub struct WorkflowEngine {
    runtime: Box<dyn ComputeRuntime>,
    budget: InstructionBudget,
    workflows: HashMap<WorkflowId, WorkflowState>,
    active: Option<WorkflowId>,
    /// FIFO queue of pending workflow IDs for deterministic scheduling.
    /// Using a VecDeque instead of relying on HashMap iteration order.
    pending_queue: VecDeque<WorkflowId>,
}

impl WorkflowEngine {
    /// Create a new `WorkflowEngine` with the given compute runtime and
    /// per-tick instruction budget.
    pub fn new(runtime: Box<dyn ComputeRuntime>, budget: InstructionBudget) -> Self {
        Self {
            runtime,
            budget,
            workflows: HashMap::new(),
            active: None,
            pending_queue: VecDeque::new(),
        }
    }

    /// Dispatch an inbound event and return any resulting actions.
    pub fn handle(&mut self, event: WorkflowEvent) -> Vec<WorkflowAction> {
        match event {
            WorkflowEvent::Submit {
                module,
                input,
                hint,
            } => self.handle_submit(module, input, hint),
            WorkflowEvent::ContentFetched { cid, data } => self.handle_content_fetched(cid, data),
            WorkflowEvent::ContentFetchFailed { cid } => self.handle_content_fetch_failed(cid),
            // Stubs for future tasks.
            WorkflowEvent::SubmitByCid { .. }
            | WorkflowEvent::ModuleFetched { .. }
            | WorkflowEvent::ModuleFetchFailed { .. } => Vec::new(),
        }
    }

    /// Advance execution: resume an active yielded workflow, or start the
    /// next pending workflow. Returns actions produced by the execution step.
    pub fn tick(&mut self) -> Vec<WorkflowAction> {
        // If there is an active workflow with a pending resumable, resume it.
        if let Some(wf_id) = self.active {
            if self.runtime.has_pending() {
                let result = self.runtime.resume(self.budget);
                return self.handle_compute_result(wf_id, result);
            }
        }

        // Otherwise, pop the next pending workflow from the FIFO queue.
        let next = loop {
            match self.pending_queue.pop_front() {
                Some(id) => {
                    // Verify it's still pending (could have been removed or failed).
                    if self
                        .workflows
                        .get(&id)
                        .is_some_and(|s| s.status == WorkflowStatus::Pending)
                    {
                        break Some(id);
                    }
                }
                None => break None,
            }
        };

        if let Some(wf_id) = next {
            let state = self.workflows.get_mut(&wf_id).expect("workflow must exist");
            let module_bytes = match state.module_bytes.take() {
                Some(bytes) => bytes,
                None => {
                    // No module bytes available (shouldn't happen for inline submit).
                    state.status = WorkflowStatus::Failed;
                    self.active = None;
                    return vec![WorkflowAction::WorkflowFailed {
                        workflow_id: wf_id,
                        error: "no module bytes available".into(),
                    }];
                }
            };
            let input = state.history.input.clone();

            state.status = WorkflowStatus::Executing;
            self.active = Some(wf_id);

            let result = self.runtime.execute(&module_bytes, &input, self.budget);
            self.handle_compute_result(wf_id, result)
        } else {
            Vec::new()
        }
    }

    /// Query the current status of a workflow.
    pub fn workflow_status(&self, id: &WorkflowId) -> Option<&WorkflowStatus> {
        self.workflows.get(id).map(|s| &s.status)
    }

    /// Clone the workflow's history for persistence or inspection.
    pub fn take_history(&self, id: &WorkflowId) -> Option<WorkflowHistory> {
        self.workflows.get(id).map(|s| s.history.clone())
    }

    /// Number of tracked workflows (all states).
    pub fn workflow_count(&self) -> usize {
        self.workflows.len()
    }

    /// Recover a workflow from a persisted history.
    ///
    /// Re-submits the module for execution. During replay, if `fetch_content`
    /// is called for a CID present in the history as `IoResolved`, the cached
    /// data is fed immediately (no external fetch). If execution reaches
    /// beyond the recorded history, new IO requests are emitted normally.
    pub fn recover(
        &mut self,
        history: WorkflowHistory,
        module_bytes: Vec<u8>,
    ) -> Vec<WorkflowAction> {
        let wf_id = history.workflow_id;

        // Don't overwrite an existing workflow (could be currently executing).
        if self.workflows.contains_key(&wf_id) {
            return vec![WorkflowAction::WorkflowFailed {
                workflow_id: wf_id,
                error: "workflow already exists, cannot recover over it".into(),
            }];
        }

        // Validate module bytes match the recorded hash.
        let actual_hash = harmony_crypto::hash::blake3_hash(&module_bytes);
        if actual_hash != history.module_hash {
            return vec![WorkflowAction::WorkflowFailed {
                workflow_id: wf_id,
                error: "module hash mismatch during recovery".into(),
            }];
        }

        // Build replay cache from history's IoResolved events.
        let mut replay_cache = HashMap::new();
        for event in &history.events {
            if let HistoryEvent::IoResolved { cid, data } = event {
                replay_cache.insert(*cid, data.clone());
            }
        }

        // Create workflow state with replay cache populated.
        let state = WorkflowState {
            status: WorkflowStatus::Pending,
            history: WorkflowHistory {
                workflow_id: wf_id,
                module_hash: history.module_hash,
                input: history.input.clone(),
                events: Vec::new(), // Fresh event log for this execution
                total_fuel_consumed: 0,
            },
            hint: ComputeHint::PreferLocal,
            module_bytes: Some(module_bytes),
            saved_session: None,
            replay_cache,
            completed_output: None,
        };

        self.workflows.insert(wf_id, state);
        self.pending_queue.push_back(wf_id);
        Vec::new() // Will execute on next tick()
    }

    // --- Private helpers ---

    fn handle_submit(
        &mut self,
        module: Vec<u8>,
        input: Vec<u8>,
        hint: ComputeHint,
    ) -> Vec<WorkflowAction> {
        let module_hash = harmony_crypto::hash::blake3_hash(&module);
        let wf_id = WorkflowId::new(&module_hash, &input);

        // Dedup: if this workflow already exists, re-emit result for completed
        // workflows so callers always get a reply.
        if let Some(existing) = self.workflows.get(&wf_id) {
            return match &existing.status {
                WorkflowStatus::Complete => {
                    if let Some(output) = &existing.completed_output {
                        vec![WorkflowAction::WorkflowComplete {
                            workflow_id: wf_id,
                            output: output.clone(),
                        }]
                    } else {
                        Vec::new()
                    }
                }
                WorkflowStatus::Failed => {
                    vec![WorkflowAction::WorkflowFailed {
                        workflow_id: wf_id,
                        error: "workflow previously failed".into(),
                    }]
                }
                // Still running — caller will get a reply when it completes.
                _ => Vec::new(),
            };
        }

        let state = WorkflowState {
            status: WorkflowStatus::Pending,
            history: WorkflowHistory {
                workflow_id: wf_id,
                module_hash,
                input,
                events: Vec::new(),
                total_fuel_consumed: 0,
            },
            hint,
            module_bytes: Some(module),
            saved_session: None,
            replay_cache: HashMap::new(),
            completed_output: None,
        };

        self.workflows.insert(wf_id, state);
        self.pending_queue.push_back(wf_id);
        Vec::new()
    }

    /// Find the workflow waiting for IO on a specific CID.
    fn find_waiting_for_io(&self, cid: &[u8; 32]) -> Option<WorkflowId> {
        self.workflows
            .iter()
            .find(|(_, state)| state.status == WorkflowStatus::WaitingForIo { cid: *cid })
            .map(|(id, _)| *id)
    }

    fn handle_content_fetched(&mut self, cid: [u8; 32], data: Vec<u8>) -> Vec<WorkflowAction> {
        let wf_id = match self.find_waiting_for_io(&cid) {
            Some(id) => id,
            None => return Vec::new(),
        };

        let state = self.workflows.get_mut(&wf_id).expect("workflow must exist");

        state.history.events.push(HistoryEvent::IoResolved {
            cid,
            data: Some(data.clone()),
        });

        let saved_session = state
            .saved_session
            .take()
            .expect("WaitingForIo implies saved session");
        self.runtime.restore_session(saved_session);
        self.active = Some(wf_id);

        let result = self
            .runtime
            .resume_with_io(IOResponse::ContentReady { data }, self.budget);
        self.handle_compute_result(wf_id, result)
    }

    fn handle_content_fetch_failed(&mut self, cid: [u8; 32]) -> Vec<WorkflowAction> {
        let wf_id = match self.find_waiting_for_io(&cid) {
            Some(id) => id,
            None => return Vec::new(),
        };

        let state = self.workflows.get_mut(&wf_id).expect("workflow must exist");

        state
            .history
            .events
            .push(HistoryEvent::IoResolved { cid, data: None });

        let saved_session = state
            .saved_session
            .take()
            .expect("WaitingForIo implies saved session");
        self.runtime.restore_session(saved_session);
        self.active = Some(wf_id);

        let result = self
            .runtime
            .resume_with_io(IOResponse::ContentNotFound, self.budget);
        self.handle_compute_result(wf_id, result)
    }

    fn handle_compute_result(
        &mut self,
        wf_id: WorkflowId,
        result: ComputeResult,
    ) -> Vec<WorkflowAction> {
        match result {
            ComputeResult::Complete { output } => {
                if let Some(state) = self.workflows.get_mut(&wf_id) {
                    state.status = WorkflowStatus::Complete;
                    state.completed_output = Some(output.clone());
                }
                self.active = None;
                vec![
                    WorkflowAction::WorkflowComplete {
                        workflow_id: wf_id,
                        output,
                    },
                    WorkflowAction::PersistHistory { workflow_id: wf_id },
                ]
            }
            ComputeResult::Yielded { fuel_consumed } => {
                // Accumulate fuel consumed across yield points.
                if let Some(state) = self.workflows.get_mut(&wf_id) {
                    state.history.total_fuel_consumed += fuel_consumed;
                }
                // Keep active set; status stays Executing. Resume on next tick.
                Vec::new()
            }
            ComputeResult::Failed { error } => {
                if let Some(state) = self.workflows.get_mut(&wf_id) {
                    state.status = WorkflowStatus::Failed;
                }
                self.active = None;
                vec![WorkflowAction::WorkflowFailed {
                    workflow_id: wf_id,
                    error: error.to_string(),
                }]
            }
            ComputeResult::NeedsIO { request } => {
                let IORequest::FetchContent { cid } = request;

                if let Some(state) = self.workflows.get_mut(&wf_id) {
                    state.history.events.push(HistoryEvent::IoRequested { cid });

                    // Check replay cache BEFORE extracting session. Use get() (not
                    // remove) so a workflow that fetches the same CID twice can replay
                    // both requests from the cache.
                    if let Some(cached_data) = state.replay_cache.get(&cid).cloned() {
                        // Cache hit: feed cached data immediately, no external fetch.
                        state.history.events.push(HistoryEvent::IoResolved {
                            cid,
                            data: cached_data.clone(),
                        });
                        // Resume execution with cached data.
                        let result = match cached_data {
                            Some(data) => self
                                .runtime
                                .resume_with_io(IOResponse::ContentReady { data }, self.budget),
                            None => self
                                .runtime
                                .resume_with_io(IOResponse::ContentNotFound, self.budget),
                        };
                        return self.handle_compute_result(wf_id, result);
                    }

                    // Cache miss: normal path — extract session, emit FetchContent.
                    let saved_session = self
                        .runtime
                        .take_session()
                        .expect("NeedsIO implies session");
                    state.saved_session = Some(saved_session);
                    state.status = WorkflowStatus::WaitingForIo { cid };
                }

                self.active = None;

                vec![
                    WorkflowAction::FetchContent {
                        workflow_id: wf_id,
                        cid,
                    },
                    WorkflowAction::PersistHistory { workflow_id: wf_id },
                ]
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use harmony_compute::WasmiRuntime;

    /// WAT module: reads two i32s from input, writes their sum as output.
    const ADD_WAT: &str = r#"
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

    fn make_engine() -> WorkflowEngine {
        WorkflowEngine::new(
            Box::new(WasmiRuntime::new()),
            InstructionBudget { fuel: 100_000 },
        )
    }

    fn add_input(a: i32, b: i32) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&a.to_le_bytes());
        v.extend_from_slice(&b.to_le_bytes());
        v
    }

    #[test]
    fn submit_creates_workflow_with_deterministic_id() {
        let mut engine = make_engine();

        let module = ADD_WAT.as_bytes().to_vec();
        let input = add_input(3, 7);

        let actions = engine.handle(WorkflowEvent::Submit {
            module: module.clone(),
            input: input.clone(),
            hint: ComputeHint::PreferLocal,
        });

        // Submit returns no immediate actions.
        assert!(actions.is_empty());
        assert_eq!(engine.workflow_count(), 1);

        // Verify status is Pending.
        let module_hash = harmony_crypto::hash::blake3_hash(&module);
        let expected_id = WorkflowId::new(&module_hash, &input);
        assert_eq!(
            engine.workflow_status(&expected_id),
            Some(&WorkflowStatus::Pending)
        );
    }

    #[test]
    fn tick_completes_simple_workflow() {
        let mut engine = make_engine();

        let module = ADD_WAT.as_bytes().to_vec();
        let input = add_input(3, 7);

        engine.handle(WorkflowEvent::Submit {
            module,
            input,
            hint: ComputeHint::PreferLocal,
        });

        // Tick until WorkflowComplete appears.
        let mut all_actions = Vec::new();
        for _ in 0..10 {
            let actions = engine.tick();
            let done = actions
                .iter()
                .any(|a| matches!(a, WorkflowAction::WorkflowComplete { .. }));
            all_actions.extend(actions);
            if done {
                break;
            }
        }

        // Find the WorkflowComplete action and verify output.
        let complete = all_actions
            .iter()
            .find(|a| matches!(a, WorkflowAction::WorkflowComplete { .. }))
            .expect("should have WorkflowComplete action");

        match complete {
            WorkflowAction::WorkflowComplete { output, .. } => {
                let result = i32::from_le_bytes(output[..4].try_into().unwrap());
                assert_eq!(result, 10);
            }
            _ => unreachable!(),
        }

        // Should also have a PersistHistory action.
        assert!(all_actions
            .iter()
            .any(|a| matches!(a, WorkflowAction::PersistHistory { .. })));
    }

    #[test]
    fn duplicate_submit_returns_existing_id() {
        let mut engine = make_engine();

        let module = ADD_WAT.as_bytes().to_vec();
        let input = add_input(3, 7);

        engine.handle(WorkflowEvent::Submit {
            module: module.clone(),
            input: input.clone(),
            hint: ComputeHint::PreferLocal,
        });

        // Submit the exact same module + input again.
        let actions = engine.handle(WorkflowEvent::Submit {
            module,
            input,
            hint: ComputeHint::PreferLocal,
        });

        // Dedup: no actions, count still 1.
        assert!(actions.is_empty());
        assert_eq!(engine.workflow_count(), 1);
    }

    #[test]
    fn duplicate_submit_after_completion_re_emits_result() {
        let mut engine = make_engine();
        let module = ADD_WAT.as_bytes().to_vec();
        let input = add_input(3, 7);

        // Submit and run to completion.
        engine.handle(WorkflowEvent::Submit {
            module: module.clone(),
            input: input.clone(),
            hint: ComputeHint::PreferLocal,
        });
        for _ in 0..10 {
            let actions = engine.tick();
            if actions
                .iter()
                .any(|a| matches!(a, WorkflowAction::WorkflowComplete { .. }))
            {
                break;
            }
        }

        // Submit the same module+input again AFTER completion.
        let actions = engine.handle(WorkflowEvent::Submit {
            module,
            input,
            hint: ComputeHint::PreferLocal,
        });

        // Should re-emit WorkflowComplete with the cached output.
        let complete = actions
            .iter()
            .find(|a| matches!(a, WorkflowAction::WorkflowComplete { .. }))
            .expect("should re-emit WorkflowComplete for completed workflow");

        match complete {
            WorkflowAction::WorkflowComplete { output, .. } => {
                let result = i32::from_le_bytes(output[..4].try_into().unwrap());
                assert_eq!(result, 10, "should return same result as original");
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn workflow_status_transitions() {
        let mut engine = make_engine();

        let module = ADD_WAT.as_bytes().to_vec();
        let input = add_input(5, 5);

        let module_hash = harmony_crypto::hash::blake3_hash(&module);
        let wf_id = WorkflowId::new(&module_hash, &input);

        engine.handle(WorkflowEvent::Submit {
            module,
            input,
            hint: ComputeHint::PreferLocal,
        });

        // Status should be Pending before any tick.
        assert_eq!(
            engine.workflow_status(&wf_id),
            Some(&WorkflowStatus::Pending)
        );

        // Tick to completion.
        for _ in 0..10 {
            let actions = engine.tick();
            if actions
                .iter()
                .any(|a| matches!(a, WorkflowAction::WorkflowComplete { .. }))
            {
                break;
            }
        }

        // Status should be Complete.
        assert_eq!(
            engine.workflow_status(&wf_id),
            Some(&WorkflowStatus::Complete)
        );
    }

    #[test]
    fn take_history_returns_workflow_history() {
        let mut engine = make_engine();

        let module = ADD_WAT.as_bytes().to_vec();
        let input = add_input(1, 2);

        let module_hash = harmony_crypto::hash::blake3_hash(&module);
        let wf_id = WorkflowId::new(&module_hash, &input);

        engine.handle(WorkflowEvent::Submit {
            module,
            input: input.clone(),
            hint: ComputeHint::PreferLocal,
        });

        let history = engine
            .take_history(&wf_id)
            .expect("history should exist after submit");

        assert_eq!(history.workflow_id, wf_id);
        assert_eq!(history.module_hash, module_hash);
        assert_eq!(history.input, input);
        assert!(history.events.is_empty());
        assert_eq!(history.total_fuel_consumed, 0);
    }

    // --- IO handling tests (Task 4) ---

    /// WAT module that imports harmony.fetch_content and calls it.
    /// Input: [cid: 32 bytes]
    /// Output: [result_code: i32 LE] [fetched_data if result > 0]
    const FETCH_WAT: &str = r#"
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

    fn make_engine_high_fuel() -> WorkflowEngine {
        WorkflowEngine::new(
            Box::new(WasmiRuntime::new()),
            InstructionBudget { fuel: 1_000_000 },
        )
    }

    #[test]
    fn io_request_recorded_in_history() {
        let mut engine = make_engine_high_fuel();
        let cid = [0xAB; 32];
        let module = FETCH_WAT.as_bytes().to_vec();

        let module_hash = harmony_crypto::hash::blake3_hash(&module);
        let wf_id = WorkflowId::new(&module_hash, &cid);

        engine.handle(WorkflowEvent::Submit {
            module,
            input: cid.to_vec(),
            hint: ComputeHint::PreferLocal,
        });

        let actions = engine.tick();

        // Should have FetchContent action.
        let fetch_action = actions
            .iter()
            .find(|a| matches!(a, WorkflowAction::FetchContent { .. }))
            .expect("should have FetchContent action");
        match fetch_action {
            WorkflowAction::FetchContent {
                workflow_id,
                cid: action_cid,
            } => {
                assert_eq!(*workflow_id, wf_id);
                assert_eq!(*action_cid, cid);
            }
            _ => unreachable!(),
        }

        // History should have IoRequested.
        let history = engine.take_history(&wf_id).expect("history should exist");
        assert_eq!(history.events.len(), 1);
        assert!(matches!(&history.events[0], HistoryEvent::IoRequested { cid: c } if *c == cid));
    }

    #[test]
    fn content_fetched_resumes_and_completes() {
        let mut engine = make_engine_high_fuel();
        let cid = [0xAB; 32];
        let content = b"hello world";

        engine.handle(WorkflowEvent::Submit {
            module: FETCH_WAT.as_bytes().to_vec(),
            input: cid.to_vec(),
            hint: ComputeHint::PreferLocal,
        });

        // Tick — should get NeedsIO → FetchContent action.
        let actions = engine.tick();
        assert!(actions
            .iter()
            .any(|a| matches!(a, WorkflowAction::FetchContent { .. })));

        // Deliver the content.
        let actions = engine.handle(WorkflowEvent::ContentFetched {
            cid,
            data: content.to_vec(),
        });

        // Should have WorkflowComplete with correct output.
        let complete = actions
            .iter()
            .find(|a| matches!(a, WorkflowAction::WorkflowComplete { .. }))
            .expect("should have WorkflowComplete action");

        match complete {
            WorkflowAction::WorkflowComplete { output, .. } => {
                // Output format: [result_code: i32 LE] [data]
                let result_code = i32::from_le_bytes(output[0..4].try_into().unwrap());
                assert_eq!(result_code, content.len() as i32);
                assert_eq!(&output[4..], content);
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn io_response_recorded_in_history() {
        let mut engine = make_engine_high_fuel();
        let cid = [0xCD; 32];
        let content = vec![10, 20, 30];
        let module = FETCH_WAT.as_bytes().to_vec();

        let module_hash = harmony_crypto::hash::blake3_hash(&module);
        let wf_id = WorkflowId::new(&module_hash, &cid);

        engine.handle(WorkflowEvent::Submit {
            module,
            input: cid.to_vec(),
            hint: ComputeHint::PreferLocal,
        });

        // Tick — NeedsIO.
        engine.tick();

        // Deliver content.
        engine.handle(WorkflowEvent::ContentFetched {
            cid,
            data: content.clone(),
        });

        // History should have both IoRequested and IoResolved.
        let history = engine.take_history(&wf_id).expect("history should exist");
        assert_eq!(history.events.len(), 2);

        assert!(matches!(&history.events[0], HistoryEvent::IoRequested { cid: c } if *c == cid));
        assert!(
            matches!(&history.events[1], HistoryEvent::IoResolved { cid: c, data: Some(d) } if *c == cid && *d == content)
        );
    }

    #[test]
    fn content_fetch_failed_resolves_with_not_found() {
        let mut engine = make_engine_high_fuel();
        let cid = [0xEF; 32];

        engine.handle(WorkflowEvent::Submit {
            module: FETCH_WAT.as_bytes().to_vec(),
            input: cid.to_vec(),
            hint: ComputeHint::PreferLocal,
        });

        // Tick — NeedsIO.
        engine.tick();

        // Deliver failure.
        let actions = engine.handle(WorkflowEvent::ContentFetchFailed { cid });

        // Module should still complete (with -1 result code).
        let complete = actions
            .iter()
            .find(|a| matches!(a, WorkflowAction::WorkflowComplete { .. }))
            .expect("should have WorkflowComplete action");

        match complete {
            WorkflowAction::WorkflowComplete { output, .. } => {
                let result_code = i32::from_le_bytes(output[0..4].try_into().unwrap());
                assert_eq!(result_code, -1, "content not found should return -1");
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn waiting_for_io_status() {
        let mut engine = make_engine_high_fuel();
        let cid = [0x42; 32];
        let module = FETCH_WAT.as_bytes().to_vec();

        let module_hash = harmony_crypto::hash::blake3_hash(&module);
        let wf_id = WorkflowId::new(&module_hash, &cid);

        engine.handle(WorkflowEvent::Submit {
            module,
            input: cid.to_vec(),
            hint: ComputeHint::PreferLocal,
        });

        // Tick — NeedsIO.
        engine.tick();

        // Status should be WaitingForIo with the correct CID.
        assert_eq!(
            engine.workflow_status(&wf_id),
            Some(&WorkflowStatus::WaitingForIo { cid })
        );
    }

    // --- Recovery tests (Task 5) ---

    #[test]
    fn recover_replays_from_history() {
        // First: run a workflow to completion with IO.
        let mut engine = make_engine_high_fuel();
        let cid = [0x33; 32];
        let content = b"recovery test data";

        engine.handle(WorkflowEvent::Submit {
            module: FETCH_WAT.as_bytes().to_vec(),
            input: cid.to_vec(),
            hint: ComputeHint::PreferLocal,
        });
        engine.tick(); // -> NeedsIO
        engine.handle(WorkflowEvent::ContentFetched {
            cid,
            data: content.to_vec(),
        });

        let module_hash = harmony_crypto::hash::blake3_hash(FETCH_WAT.as_bytes());
        let wf_id = WorkflowId::new(&module_hash, &cid);
        let history = engine.take_history(&wf_id).unwrap();

        // Now: create a FRESH engine and recover from the history.
        let mut engine2 = make_engine_high_fuel();
        let _actions = engine2.recover(history, FETCH_WAT.as_bytes().to_vec());

        // Tick to completion. Since all IO is cached, should complete
        // without emitting any FetchContent actions.
        let mut all_actions = Vec::new();
        for _ in 0..100 {
            let actions = engine2.tick();
            let done = actions
                .iter()
                .any(|a| matches!(a, WorkflowAction::WorkflowComplete { .. }));
            all_actions.extend(actions);
            if done {
                break;
            }
        }

        // Verify no FetchContent was emitted (all IO was cached).
        assert!(
            !all_actions
                .iter()
                .any(|a| matches!(a, WorkflowAction::FetchContent { .. })),
            "recovery should not emit FetchContent for cached IO"
        );

        let complete = all_actions
            .iter()
            .find(|a| matches!(a, WorkflowAction::WorkflowComplete { .. }));
        assert!(
            complete.is_some(),
            "recovery should complete, got: {all_actions:?}"
        );

        if let Some(WorkflowAction::WorkflowComplete { output, .. }) = complete {
            let result_code = i32::from_le_bytes(output[..4].try_into().unwrap());
            assert_eq!(result_code, content.len() as i32);
            assert_eq!(&output[4..], content);
        }
    }

    #[test]
    fn recover_with_new_io_beyond_history() {
        // Create a history with NO recorded IO (as if module crashed at NeedsIO).
        let module_hash = harmony_crypto::hash::blake3_hash(FETCH_WAT.as_bytes());
        let cid = [0x44; 32];
        let wf_id = WorkflowId::new(&module_hash, &cid);

        let history = WorkflowHistory {
            workflow_id: wf_id,
            module_hash,
            input: cid.to_vec(),
            events: vec![], // No recorded IO
            total_fuel_consumed: 0,
        };

        let mut engine = make_engine_high_fuel();
        engine.recover(history, FETCH_WAT.as_bytes().to_vec());

        // Tick: module calls fetch_content -> NeedsIO -> FetchContent (not cached).
        let mut all_actions = Vec::new();
        for _ in 0..100 {
            let actions = engine.tick();
            all_actions.extend(actions.clone());
            if actions
                .iter()
                .any(|a| matches!(a, WorkflowAction::FetchContent { .. }))
            {
                break;
            }
        }

        assert!(
            all_actions
                .iter()
                .any(|a| matches!(a, WorkflowAction::FetchContent { .. })),
            "should emit FetchContent for IO beyond recorded history"
        );
    }

    #[test]
    fn recover_rejects_module_hash_mismatch() {
        let module_hash = harmony_crypto::hash::blake3_hash(FETCH_WAT.as_bytes());
        let cid = [0x55; 32];
        let wf_id = WorkflowId::new(&module_hash, &cid);

        let history = WorkflowHistory {
            workflow_id: wf_id,
            module_hash,
            input: cid.to_vec(),
            events: vec![],
            total_fuel_consumed: 0,
        };

        let mut engine = make_engine_high_fuel();
        // Pass wrong module bytes — should fail.
        let actions = engine.recover(history, b"wrong module bytes".to_vec());

        assert_eq!(actions.len(), 1);
        assert!(
            matches!(&actions[0], WorkflowAction::WorkflowFailed { error, .. } if error.contains("hash mismatch")),
            "should fail with hash mismatch, got: {actions:?}"
        );
        assert_eq!(engine.workflow_count(), 0, "should not create workflow on mismatch");
    }

    #[test]
    fn recover_rejects_if_workflow_already_exists() {
        let mut engine = make_engine_high_fuel();
        let module = FETCH_WAT.as_bytes().to_vec();
        let cid = [0x66; 32];

        // Submit and start executing the workflow.
        engine.handle(WorkflowEvent::Submit {
            module: module.clone(),
            input: cid.to_vec(),
            hint: ComputeHint::PreferLocal,
        });
        engine.tick(); // Starts executing

        // Now try to recover the same workflow — should fail.
        let module_hash = harmony_crypto::hash::blake3_hash(&module);
        let wf_id = WorkflowId::new(&module_hash, &cid);
        let history = WorkflowHistory {
            workflow_id: wf_id,
            module_hash,
            input: cid.to_vec(),
            events: vec![],
            total_fuel_consumed: 0,
        };

        let actions = engine.recover(history, module);

        assert_eq!(actions.len(), 1);
        assert!(
            matches!(&actions[0], WorkflowAction::WorkflowFailed { error, .. } if error.contains("already exists")),
            "should reject recover over existing workflow, got: {actions:?}"
        );
        assert_eq!(engine.workflow_count(), 1, "original workflow should be untouched");
    }
}
