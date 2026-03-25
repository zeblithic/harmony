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
    /// Populated from WorkflowHistory.events on recover().
    ///
    /// Keyed by CID. In a content-addressed system, the same CID always maps to
    /// the same data, so a flat HashMap is correct — a CID cannot return different
    /// results across fetches.
    replay_cache: HashMap<[u8; 32], Option<Vec<u8>>>,
    /// Cached output for completed workflows. Allows re-emitting WorkflowComplete
    /// when a duplicate submit arrives after the workflow has already finished.
    completed_output: Option<Vec<u8>>,
    /// IO response waiting to be delivered. Set when multiple workflows are
    /// waiting for the same CID and this workflow wasn't the first to be resumed.
    deferred_io: Option<IOResponse>,
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
    /// Workflows with deferred IO responses, ready to be resumed on next tick.
    deferred_resume_queue: VecDeque<WorkflowId>,
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
            deferred_resume_queue: VecDeque::new(),
        }
    }

    /// Read-only access to the per-slice instruction budget.
    pub fn budget(&self) -> InstructionBudget {
        self.budget
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
            WorkflowEvent::ModelLoaded {
                gguf_cid,
                tokenizer_cid,
                gguf_data,
                tokenizer_data,
            } => self.handle_model_loaded(gguf_cid, tokenizer_cid, gguf_data, tokenizer_data),
            WorkflowEvent::ModelLoadFailed {
                gguf_cid,
                tokenizer_cid,
            } => self.handle_model_load_failed(gguf_cid, tokenizer_cid),
            // Stubs for future tasks. Currently, NodeRuntime handles module
            // fetching externally via cid_to_query and converts results into
            // inline Submit events. When these stubs are implemented, remove
            // the manual handling in NodeRuntime::push_event(ModuleFetchResponse)
            // to avoid double-submission. See runtime.rs ModuleFetchResponse arm.
            WorkflowEvent::SubmitByCid { .. }
            | WorkflowEvent::ModuleFetched { .. }
            | WorkflowEvent::ModuleFetchFailed { .. } => Vec::new(),
        }
    }

    /// Advance execution: resume an active yielded workflow, resume a
    /// workflow with deferred IO, or start the next pending workflow.
    /// Returns actions produced by the execution step.
    pub fn tick(&mut self) -> Vec<WorkflowAction> {
        // Priority 1: resume an active workflow that yielded mid-execution.
        if let Some(wf_id) = self.active {
            if self.runtime.has_pending() {
                let result = self.runtime.resume(self.budget);
                return self.handle_compute_result(wf_id, result);
            }
            // Runtime lost the session — fail the orphaned workflow to prevent
            // it from staying in Executing state forever.
            if let Some(state) = self.workflows.get_mut(&wf_id) {
                state.status = WorkflowStatus::Failed;
            }
            self.active = None;
            return vec![
                WorkflowAction::WorkflowFailed {
                    workflow_id: wf_id,
                    error: "runtime session lost while workflow was active".into(),
                },
                WorkflowAction::PersistHistory { workflow_id: wf_id },
            ];
        }

        // Priority 2: resume a workflow with deferred IO (multiple workflows
        // were waiting for the same CID; only the first was resumed immediately).
        if let Some(wf_id) = self.deferred_resume_queue.pop_front() {
            if let Some(state) = self.workflows.get_mut(&wf_id) {
                if let Some(io_response) = state.deferred_io.take() {
                    let saved_session = state
                        .saved_session
                        .take()
                        .expect("deferred IO implies saved session");
                    self.runtime.restore_session(saved_session);
                    self.active = Some(wf_id);

                    let result = self.runtime.resume_with_io(io_response, self.budget);
                    return self.handle_compute_result(wf_id, result);
                }
            }
        }

        // Priority 3: pop the next pending workflow from the FIFO queue.
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

    /// Run one compute slice with a specific fuel budget (for adaptive scheduling).
    ///
    /// Uses a drop guard to restore the original budget even if `tick()` panics
    /// (e.g. from WASM execution), preventing permanent budget corruption.
    pub fn tick_with_budget(&mut self, budget: InstructionBudget) -> Vec<WorkflowAction> {
        struct BudgetGuard<'a> {
            engine: &'a mut WorkflowEngine,
            saved: InstructionBudget,
        }
        impl<'a> Drop for BudgetGuard<'a> {
            fn drop(&mut self) {
                self.engine.budget = self.saved;
            }
        }

        let saved = self.budget;
        self.budget = budget;
        // Guard restores self.budget on drop (normal return or panic).
        let guard = BudgetGuard {
            engine: self,
            saved,
        };
        let result = guard.engine.tick();
        // Explicit drop triggers restore before we return.
        drop(guard);
        result
    }

    /// Query the current status of a workflow.
    pub fn workflow_status(&self, id: &WorkflowId) -> Option<&WorkflowStatus> {
        self.workflows.get(id).map(|s| &s.status)
    }

    /// Clone the workflow's history for persistence or inspection.
    pub fn clone_history(&self, id: &WorkflowId) -> Option<WorkflowHistory> {
        self.workflows.get(id).map(|s| s.history.clone())
    }

    /// Number of tracked workflows (all states).
    pub fn workflow_count(&self) -> usize {
        self.workflows.len()
    }

    /// Remove a completed or failed workflow from the engine.
    ///
    /// Strip runtime-heavy data from a terminal workflow while preserving
    /// its entry for dedup and its history for persistence/recovery.
    ///
    /// Only compacts workflows in terminal states (`Complete` or `Failed`).
    /// Clears module bytes, sessions, and replay cache but retains
    /// `history.events` and `history.input` — those are needed for crash
    /// recovery via deterministic replay and must survive until the caller
    /// has persisted the history externally.
    pub fn compact_workflow(&mut self, id: &WorkflowId) -> bool {
        match self.workflows.get_mut(id) {
            Some(state)
                if state.status == WorkflowStatus::Complete
                    || state.status == WorkflowStatus::Failed =>
            {
                state.module_bytes = None;
                state.saved_session = None;
                state.replay_cache.clear();
                state.deferred_io = None;
                // NOTE: history.events and history.input are intentionally
                // preserved — they are the durable checkpoint for crash
                // recovery. Clear them only after external persistence
                // confirms the data is safely stored.
                //
                // TODO: completed_output is also preserved so duplicate
                // submits can re-emit the result. For workflows producing
                // large outputs, consider bounding by size or TTL to
                // prevent unbounded memory growth.
                true
            }
            _ => false,
        }
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
        // Validate workflow ID is correctly derived from hash+input.
        let actual_hash = harmony_crypto::hash::blake3_hash(&module_bytes);
        let expected_id = WorkflowId::new(&actual_hash, &history.input);
        if history.workflow_id != expected_id {
            return vec![WorkflowAction::WorkflowFailed {
                workflow_id: history.workflow_id,
                error: "workflow ID does not match module hash + input".into(),
            }];
        }
        let wf_id = expected_id;

        // Don't overwrite an existing workflow (could be currently executing).
        if self.workflows.contains_key(&wf_id) {
            return vec![WorkflowAction::WorkflowFailed {
                workflow_id: wf_id,
                error: "workflow already exists, cannot recover over it".into(),
            }];
        }

        // NOTE: No separate module_hash check needed here — the WorkflowId
        // validation above already guarantees actual_hash == history.module_hash
        // via BLAKE3 collision resistance (ID = BLAKE3(hash || input)).

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
                events: Vec::new(),     // Fresh event log for this execution
                total_fuel_consumed: 0, // Reset: recovery re-executes from scratch
            },
            hint: ComputeHint::PreferLocal,
            module_bytes: Some(module_bytes),
            saved_session: None,
            replay_cache,
            completed_output: None,
            deferred_io: None,
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
            deferred_io: None,
        };

        self.workflows.insert(wf_id, state);
        self.pending_queue.push_back(wf_id);
        Vec::new()
    }

    /// Find all workflows waiting for IO on a specific CID.
    /// Returns results sorted by WorkflowId bytes for deterministic scheduling.
    fn find_all_waiting_for_io(&self, cid: &[u8; 32]) -> Vec<WorkflowId> {
        let mut waiting: Vec<WorkflowId> = self
            .workflows
            .iter()
            .filter(|(_, state)| state.status == WorkflowStatus::WaitingForIo { cid: *cid })
            .map(|(id, _)| *id)
            .collect();
        waiting.sort_by_key(|id| *id.as_bytes());
        waiting
    }

    fn handle_content_fetched(&mut self, cid: [u8; 32], data: Vec<u8>) -> Vec<WorkflowAction> {
        self.handle_io_resolution(cid, Some(data.clone()), IOResponse::ContentReady { data })
    }

    fn handle_content_fetch_failed(&mut self, cid: [u8; 32]) -> Vec<WorkflowAction> {
        self.handle_io_resolution(cid, None, IOResponse::ContentNotFound)
    }

    fn handle_model_loaded(
        &mut self,
        gguf_cid: [u8; 32],
        tokenizer_cid: [u8; 32],
        gguf_data: Vec<u8>,
        tokenizer_data: Vec<u8>,
    ) -> Vec<WorkflowAction> {
        self.handle_model_resolution(
            gguf_cid,
            tokenizer_cid,
            true,
            IOResponse::ModelReady {
                gguf_data,
                tokenizer_data,
            },
        )
    }

    fn handle_model_load_failed(
        &mut self,
        gguf_cid: [u8; 32],
        tokenizer_cid: [u8; 32],
    ) -> Vec<WorkflowAction> {
        self.handle_model_resolution(
            gguf_cid,
            tokenizer_cid,
            false,
            IOResponse::ModelGgufNotFound,
        )
    }

    /// Find all workflows waiting for a specific model (by GGUF + tokenizer CIDs).
    /// Returns results sorted by WorkflowId bytes for deterministic scheduling.
    fn find_all_waiting_for_model(
        &self,
        gguf_cid: &[u8; 32],
        tokenizer_cid: &[u8; 32],
    ) -> Vec<WorkflowId> {
        let mut waiting: Vec<WorkflowId> = self
            .workflows
            .iter()
            .filter(|(_, state)| {
                state.status
                    == WorkflowStatus::WaitingForModel {
                        gguf_cid: *gguf_cid,
                        tokenizer_cid: *tokenizer_cid,
                    }
            })
            .map(|(id, _)| *id)
            .collect();
        waiting.sort_by_key(|id| *id.as_bytes());
        waiting
    }

    /// Common handler for model resolution (loaded or load failed).
    /// Mirrors handle_io_resolution but for model requests.
    fn handle_model_resolution(
        &mut self,
        gguf_cid: [u8; 32],
        tokenizer_cid: [u8; 32],
        success: bool,
        io_response: IOResponse,
    ) -> Vec<WorkflowAction> {
        let waiting = self.find_all_waiting_for_model(&gguf_cid, &tokenizer_cid);
        if waiting.is_empty() {
            return Vec::new();
        }

        // If a workflow is currently active (yielded mid-execution), we cannot
        // restore a different session without destroying the active one. Defer
        // ALL waiting workflows so they resume after the active one completes.
        let immediate_resume = self.active.is_none();

        if immediate_resume {
            let first = waiting[0];
            let state = self.workflows.get_mut(&first).expect("workflow must exist");
            state.status = WorkflowStatus::Executing;
            state.history.events.push(HistoryEvent::ModelResolved {
                gguf_cid,
                tokenizer_cid,
                success,
            });
            let saved_session = state
                .saved_session
                .take()
                .expect("WaitingForModel implies saved session");
            self.runtime.restore_session(saved_session);
            self.active = Some(first);
        }

        // Defer remaining workflows (or all, if an active workflow is running).
        let defer_start = if immediate_resume { 1 } else { 0 };
        for &wf_id in &waiting[defer_start..] {
            if let Some(state) = self.workflows.get_mut(&wf_id) {
                state.history.events.push(HistoryEvent::ModelResolved {
                    gguf_cid,
                    tokenizer_cid,
                    success,
                });
                state.status = WorkflowStatus::Executing;
                state.deferred_io = Some(io_response.clone());
                self.deferred_resume_queue.push_back(wf_id);
            }
        }

        if immediate_resume {
            let result = self.runtime.resume_with_io(io_response, self.budget);
            self.handle_compute_result(waiting[0], result)
        } else {
            Vec::new()
        }
    }

    /// Common handler for IO resolution (content fetched or fetch failed).
    fn handle_io_resolution(
        &mut self,
        cid: [u8; 32],
        data: Option<Vec<u8>>,
        io_response: IOResponse,
    ) -> Vec<WorkflowAction> {
        let waiting = self.find_all_waiting_for_io(&cid);
        if waiting.is_empty() {
            return Vec::new();
        }

        // If a workflow is currently active (yielded mid-execution), we cannot
        // restore a different session without destroying the active one. Defer
        // ALL waiting workflows so they resume after the active one completes.
        let immediate_resume = self.active.is_none();

        if immediate_resume {
            // Resume the first workflow immediately.
            // Set status to Executing BEFORE resume so find_all_waiting_for_io
            // won't rediscover it if resume_with_io returns Yielded.
            let first = waiting[0];
            let state = self.workflows.get_mut(&first).expect("workflow must exist");
            state.status = WorkflowStatus::Executing;
            state.history.events.push(HistoryEvent::IoResolved {
                cid,
                data: data.clone(),
            });
            let saved_session = state
                .saved_session
                .take()
                .expect("WaitingForIo implies saved session");
            self.runtime.restore_session(saved_session);
            self.active = Some(first);
        }

        // Defer remaining workflows (or all, if an active workflow is running).
        let defer_start = if immediate_resume { 1 } else { 0 };
        for &wf_id in &waiting[defer_start..] {
            if let Some(state) = self.workflows.get_mut(&wf_id) {
                state.history.events.push(HistoryEvent::IoResolved {
                    cid,
                    data: data.clone(),
                });
                // NOTE: Status is set to Executing here even though the workflow
                // won't actually resume until a future tick() picks it up from
                // deferred_resume_queue. Callers of workflow_status() will briefly
                // see Executing for a still-suspended workflow. Acceptable for now;
                // a DeferredResume status could improve observability later.
                state.status = WorkflowStatus::Executing;
                state.deferred_io = Some(io_response.clone());
                self.deferred_resume_queue.push_back(wf_id);
            }
        }

        if immediate_resume {
            let result = self.runtime.resume_with_io(io_response, self.budget);
            self.handle_compute_result(waiting[0], result)
        } else {
            Vec::new()
        }
    }

    /// Process a compute result, iteratively handling replay-cache hits in the
    /// NeedsIO path to avoid unbounded recursion for workflows with many cached
    /// IO entries (e.g., large Merkle DAG traversals).
    fn handle_compute_result(
        &mut self,
        wf_id: WorkflowId,
        result: ComputeResult,
    ) -> Vec<WorkflowAction> {
        let mut current_result = result;

        loop {
            match current_result {
                ComputeResult::Complete { output } => {
                    if let Some(state) = self.workflows.get_mut(&wf_id) {
                        state.status = WorkflowStatus::Complete;
                        state.completed_output = Some(output.clone());
                    }
                    self.active = None;
                    return vec![
                        WorkflowAction::WorkflowComplete {
                            workflow_id: wf_id,
                            output,
                        },
                        WorkflowAction::PersistHistory { workflow_id: wf_id },
                    ];
                }
                ComputeResult::Yielded { fuel_consumed } => {
                    if let Some(state) = self.workflows.get_mut(&wf_id) {
                        state.history.total_fuel_consumed += fuel_consumed;
                    }
                    return Vec::new();
                }
                ComputeResult::Failed { error } => {
                    if let Some(state) = self.workflows.get_mut(&wf_id) {
                        state.status = WorkflowStatus::Failed;
                    }
                    self.active = None;
                    return vec![
                        WorkflowAction::WorkflowFailed {
                            workflow_id: wf_id,
                            error: error.to_string(),
                        },
                        WorkflowAction::PersistHistory { workflow_id: wf_id },
                    ];
                }
                ComputeResult::NeedsIO { request } => match request {
                    IORequest::FetchContent { cid } => {
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
                                // Resume and loop (iterative, not recursive).
                                current_result = match cached_data {
                                    Some(data) => self.runtime.resume_with_io(
                                        IOResponse::ContentReady { data },
                                        self.budget,
                                    ),
                                    None => self.runtime.resume_with_io(
                                        IOResponse::ContentNotFound,
                                        self.budget,
                                    ),
                                };
                                continue;
                            }

                            // Cache miss: normal path — extract session, emit FetchContent.
                            let saved_session = self
                                .runtime
                                .take_session()
                                .expect("NeedsIO implies session");
                            state.saved_session = Some(saved_session);
                            state.status = WorkflowStatus::WaitingForIo { cid };
                            self.active = None;

                            // NOTE: Multiple workflows requesting the same CID will each
                            // emit a FetchContent action. Deduplication is left to the
                            // caller (NodeRuntime) or the storage layer (CAS).
                            return vec![
                                WorkflowAction::FetchContent {
                                    workflow_id: wf_id,
                                    cid,
                                },
                                WorkflowAction::PersistHistory { workflow_id: wf_id },
                            ];
                        }

                        // Workflow not found — still clear active, but emit nothing.
                        self.active = None;
                        return Vec::new();
                    }
                    IORequest::LoadModel {
                        gguf_cid,
                        tokenizer_cid,
                    } => {
                        if let Some(state) = self.workflows.get_mut(&wf_id) {
                            state.history.events.push(HistoryEvent::ModelRequested {
                                gguf_cid,
                                tokenizer_cid,
                            });

                            // Extract session, suspend workflow, emit LoadModel action.
                            let saved_session = self
                                .runtime
                                .take_session()
                                .expect("NeedsIO implies session");
                            state.saved_session = Some(saved_session);
                            state.status = WorkflowStatus::WaitingForModel {
                                gguf_cid,
                                tokenizer_cid,
                            };
                            self.active = None;

                            return vec![
                                WorkflowAction::LoadModel {
                                    workflow_id: wf_id,
                                    gguf_cid,
                                    tokenizer_cid,
                                },
                                WorkflowAction::PersistHistory { workflow_id: wf_id },
                            ];
                        }

                        // Workflow not found — still clear active, but emit nothing.
                        self.active = None;
                        return Vec::new();
                    }
                },
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
    fn clone_history_returns_workflow_history() {
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
            .clone_history(&wf_id)
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
        let history = engine.clone_history(&wf_id).expect("history should exist");
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
        let history = engine.clone_history(&wf_id).expect("history should exist");
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
        let history = engine.clone_history(&wf_id).unwrap();

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
        // Pass wrong module bytes — ID derivation check catches this before
        // the hash mismatch check since the wrong bytes produce a different ID.
        let actions = engine.recover(history, b"wrong module bytes".to_vec());

        assert_eq!(actions.len(), 1);
        assert!(
            matches!(&actions[0], WorkflowAction::WorkflowFailed { error, .. }
                if error.contains("does not match") || error.contains("hash mismatch")),
            "should fail validation, got: {actions:?}"
        );
        assert_eq!(
            engine.workflow_count(),
            0,
            "should not create workflow on mismatch"
        );
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
        assert_eq!(
            engine.workflow_count(),
            1,
            "original workflow should be untouched"
        );
    }

    #[test]
    fn multiple_workflows_waiting_for_same_cid_all_resume() {
        let mut engine = make_engine_high_fuel();
        let cid = [0x77; 32];
        let content = b"shared content";

        // Submit two different workflows that both fetch the same CID.
        // Use different inputs so they get different WorkflowIds.
        let module = FETCH_WAT.as_bytes().to_vec();
        let module_hash = harmony_crypto::hash::blake3_hash(&module);

        // Workflow A: input = cid (fetches cid[0..32] from input)
        engine.handle(WorkflowEvent::Submit {
            module: module.clone(),
            input: cid.to_vec(),
            hint: ComputeHint::PreferLocal,
        });
        let wf_a = WorkflowId::new(&module_hash, &cid);

        // Workflow B: same CID but with extra byte to get a different WorkflowId.
        let mut input_b = cid.to_vec();
        input_b.push(0xFF); // extra byte — WASM only reads first 32 bytes as CID
        engine.handle(WorkflowEvent::Submit {
            module: module.clone(),
            input: input_b.clone(),
            hint: ComputeHint::PreferLocal,
        });
        let wf_b = WorkflowId::new(&module_hash, &input_b);

        assert_ne!(wf_a, wf_b, "workflows should have different IDs");

        // Tick A to NeedsIO → WaitingForIo.
        engine.tick();
        assert_eq!(
            engine.workflow_status(&wf_a),
            Some(&WorkflowStatus::WaitingForIo { cid })
        );

        // Tick B to NeedsIO → WaitingForIo.
        engine.tick();
        assert_eq!(
            engine.workflow_status(&wf_b),
            Some(&WorkflowStatus::WaitingForIo { cid })
        );

        // Deliver content once — should resume one immediately and defer the other.
        let mut all_actions = engine.handle(WorkflowEvent::ContentFetched {
            cid,
            data: content.to_vec(),
        });

        // One workflow should complete immediately.
        let first_complete: Vec<_> = all_actions
            .iter()
            .filter_map(|a| match a {
                WorkflowAction::WorkflowComplete { workflow_id, .. } => Some(*workflow_id),
                _ => None,
            })
            .collect();
        assert_eq!(
            first_complete.len(),
            1,
            "exactly one should complete immediately"
        );

        // Tick to resume the deferred workflow.
        all_actions.extend(engine.tick());

        // Both workflows should now be complete.
        let completed: Vec<_> = all_actions
            .iter()
            .filter_map(|a| match a {
                WorkflowAction::WorkflowComplete { workflow_id, .. } => Some(*workflow_id),
                _ => None,
            })
            .collect();
        assert_eq!(completed.len(), 2, "both workflows should complete");
        assert!(completed.contains(&wf_a), "workflow A should complete");
        assert!(completed.contains(&wf_b), "workflow B should complete");
    }

    /// Scriptable mock runtime that returns pre-programmed results.
    /// Used to test engine logic without depending on wasmi's fuel semantics.
    struct ScriptedRuntime {
        results: std::cell::RefCell<VecDeque<ComputeResult>>,
        pending: bool,
        session: Option<Box<dyn std::any::Any>>,
    }

    impl ScriptedRuntime {
        fn new(results: Vec<ComputeResult>) -> Self {
            Self {
                results: std::cell::RefCell::new(VecDeque::from(results)),
                pending: false,
                session: None,
            }
        }

        fn process_result(&mut self) -> ComputeResult {
            let result = self
                .results
                .borrow_mut()
                .pop_front()
                .expect("ScriptedRuntime: no more scripted results");
            match &result {
                ComputeResult::Yielded { .. } | ComputeResult::NeedsIO { .. } => {
                    self.pending = true;
                    if self.session.is_none() {
                        self.session = Some(Box::new(()));
                    }
                }
                _ => {
                    self.pending = false;
                }
            }
            result
        }
    }

    impl ComputeRuntime for ScriptedRuntime {
        fn execute(
            &mut self,
            _module: &[u8],
            _input: &[u8],
            _budget: InstructionBudget,
        ) -> ComputeResult {
            // New execution creates a fresh session.
            self.session = Some(Box::new(()));
            self.process_result()
        }

        fn resume(&mut self, _budget: InstructionBudget) -> ComputeResult {
            self.process_result()
        }

        fn resume_with_io(
            &mut self,
            _response: IOResponse,
            _budget: InstructionBudget,
        ) -> ComputeResult {
            self.process_result()
        }

        fn has_pending(&self) -> bool {
            self.pending
        }

        fn take_session(&mut self) -> Option<Box<dyn std::any::Any>> {
            self.pending = false;
            self.session.take()
        }

        fn restore_session(&mut self, session: Box<dyn std::any::Any>) {
            self.session = Some(session);
            self.pending = true;
        }

        fn snapshot(&self) -> Result<harmony_compute::Checkpoint, harmony_compute::ComputeError> {
            Err(harmony_compute::ComputeError::NoPendingExecution)
        }
    }

    #[test]
    fn content_fetched_while_active_defers_io_workflow() {
        let cid = [0x88; 32];

        // Script: B executes→NeedsIO, A executes→Yielded, A resumes→Complete, B resumes→Complete
        let runtime = ScriptedRuntime::new(vec![
            // tick 1: B executes → NeedsIO (requests cid)
            ComputeResult::NeedsIO {
                request: IORequest::FetchContent { cid },
            },
            // tick 2: A executes → Yielded (active, mid-execution)
            ComputeResult::Yielded { fuel_consumed: 10 },
            // tick 3: A resumes → Complete
            ComputeResult::Complete {
                output: vec![1, 2, 3, 4],
            },
            // tick 4: B resumes with IO → Complete
            ComputeResult::Complete {
                output: vec![5, 6, 7, 8],
            },
        ]);

        let mut engine =
            WorkflowEngine::new(Box::new(runtime), InstructionBudget { fuel: 100_000 });

        let module_b = b"module_b".to_vec();
        let module_a = b"module_a".to_vec();
        let module_b_hash = harmony_crypto::hash::blake3_hash(&module_b);
        let module_a_hash = harmony_crypto::hash::blake3_hash(&module_a);
        let wf_b = WorkflowId::new(&module_b_hash, &cid);
        let input_a = b"input_a".to_vec();
        let wf_a = WorkflowId::new(&module_a_hash, &input_a);

        // Submit B (IO-requesting workflow).
        engine.handle(WorkflowEvent::Submit {
            module: module_b,
            input: cid.to_vec(),
            hint: ComputeHint::PreferLocal,
        });

        // Tick 1: B executes → NeedsIO → WaitingForIo, active cleared.
        let actions = engine.tick();
        assert!(actions
            .iter()
            .any(|a| matches!(a, WorkflowAction::FetchContent { .. })));
        assert_eq!(
            engine.workflow_status(&wf_b),
            Some(&WorkflowStatus::WaitingForIo { cid })
        );

        // Submit A (simple computation).
        engine.handle(WorkflowEvent::Submit {
            module: module_a,
            input: input_a,
            hint: ComputeHint::PreferLocal,
        });

        // Tick 2: A executes → Yielded → active = Some(A).
        let actions = engine.tick();
        assert!(actions.is_empty(), "Yielded should produce no actions");
        assert_eq!(
            engine.workflow_status(&wf_a),
            Some(&WorkflowStatus::Executing)
        );

        // Deliver content for B while A is active (yielded).
        // Before the fix, this would destroy A's runtime session.
        let actions = engine.handle(WorkflowEvent::ContentFetched {
            cid,
            data: b"content".to_vec(),
        });

        // B should be DEFERRED — not immediately resumed.
        assert!(
            actions.is_empty(),
            "B should be deferred when A is active, not immediately resumed"
        );

        // Tick 3: Priority 1 resumes A → Complete.
        let actions = engine.tick();
        let a_complete = actions
            .iter()
            .any(|a| matches!(a, WorkflowAction::WorkflowComplete { workflow_id, .. } if *workflow_id == wf_a));
        assert!(a_complete, "A should complete on this tick");

        // Tick 4: Priority 2 resumes B (deferred) → Complete.
        let actions = engine.tick();
        let b_complete = actions
            .iter()
            .any(|a| matches!(a, WorkflowAction::WorkflowComplete { workflow_id, .. } if *workflow_id == wf_b));
        assert!(b_complete, "B should complete on this tick");
    }
}
