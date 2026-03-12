// SPDX-License-Identifier: Apache-2.0 OR MIT
//! KitriEngine — the sans-I/O state machine for Kitri workflow orchestration.
//!
//! Follows the same Event -> handle() -> Vec<Action> pattern as
//! WorkflowEngine, PubSubRouter, Session, and all other Harmony state machines.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use crate::compensation::CompensationLog;
use crate::event::KitriEventLog;
use crate::io::{KitriIoOp, KitriIoResult};
use crate::program::{kitri_workflow_id, KitriProgram};
use crate::staging::{StagedWrite, StagingBuffer};

/// Inbound events to the Kitri engine.
#[derive(Debug, Clone)]
pub enum KitriEngineEvent {
    /// Submit a new workflow for execution.
    Submit {
        program: KitriProgram,
        input: Vec<u8>,
    },
    /// A workflow requested a durable I/O operation.
    IoRequested {
        workflow_id: [u8; 32],
        op: KitriIoOp,
    },
    /// An I/O operation completed.
    IoResolved {
        workflow_id: [u8; 32],
        seq: u64,
        result: KitriIoResult,
    },
    /// A workflow completed successfully.
    WorkflowComplete {
        workflow_id: [u8; 32],
        output: Vec<u8>,
    },
    /// A workflow failed.
    WorkflowFailed {
        workflow_id: [u8; 32],
        error: String,
    },
    /// An explicit checkpoint was requested.
    CheckpointRequested {
        workflow_id: [u8; 32],
        state: Vec<u8>,
        state_cid: [u8; 32],
    },
    /// A compensator was registered.
    CompensatorRegistered {
        workflow_id: [u8; 32],
        step_id: u64,
        rollback_op: KitriIoOp,
    },
    /// All compensators finished executing (saga rollback complete).
    CompensationComplete { workflow_id: [u8; 32] },
}

/// Outbound actions emitted by the Kitri engine.
#[derive(Debug, Clone)]
pub enum KitriAction {
    /// A new workflow was accepted and assigned its deterministic ID.
    WorkflowAccepted { workflow_id: [u8; 32] },
    /// A duplicate submission was detected; existing workflow ID returned.
    Deduplicated { workflow_id: [u8; 32] },
    /// An I/O operation should be executed by the runtime.
    ExecuteIo {
        workflow_id: [u8; 32],
        seq: u64,
        op: KitriIoOp,
    },
    /// A staged write should be committed (workflow succeeded).
    CommitStagedWrite {
        workflow_id: [u8; 32],
        write: StagedWrite,
    },
    /// The event log should be persisted to durable storage.
    PersistEventLog { workflow_id: [u8; 32] },
    /// A workflow completed successfully.
    Complete {
        workflow_id: [u8; 32],
        output: Vec<u8>,
    },
    /// A workflow failed.
    Failed {
        workflow_id: [u8; 32],
        error: String,
    },
    /// Checkpoint state bytes should be persisted to content-addressed storage.
    PersistCheckpoint {
        workflow_id: [u8; 32],
        state: Vec<u8>,
        state_cid: [u8; 32],
    },
    /// A compensator should be executed (saga rollback).
    ExecuteCompensator {
        workflow_id: [u8; 32],
        step_id: u64,
        rollback_op: KitriIoOp,
    },
    /// The runtime should delay this many milliseconds before starting the
    /// next attempt. Computed from the workflow's `BackoffStrategy`.
    RetryAfter {
        workflow_id: [u8; 32],
        delay_ms: u64,
    },
}

/// Status of a Kitri workflow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KitriWorkflowStatus {
    /// Submitted but not yet executing.
    Pending,
    /// Actively executing workflow logic.
    Executing,
    /// Blocked waiting for an I/O result.
    WaitingForIo,
    /// Completed successfully.
    Complete,
    /// Failed (possibly after compensation).
    Failed,
    /// Running compensators (saga rollback in progress).
    Compensating,
}

/// Internal state for a single workflow.
struct WorkflowState {
    program: KitriProgram,
    status: KitriWorkflowStatus,
    event_log: KitriEventLog,
    staging: StagingBuffer,
    compensation: CompensationLog,
    next_seq: u64,
    output: Option<Vec<u8>>,
    /// Error message deferred until compensation completes.
    pending_error: Option<String>,
    /// Current attempt number (0 = first run, incremented on each retry).
    attempt: u32,
}

/// The Kitri engine — manages workflow lifecycle with durable execution.
///
/// This is a sans-I/O state machine: the caller feeds it `KitriEngineEvent`s
/// and receives `KitriAction`s to execute. No I/O is performed internally.
pub struct KitriEngine {
    workflows: BTreeMap<[u8; 32], WorkflowState>,
}

impl KitriEngine {
    /// Create a new, empty Kitri engine.
    pub fn new() -> Self {
        Self {
            workflows: BTreeMap::new(),
        }
    }

    /// Query the current status of a workflow by ID.
    pub fn status(&self, workflow_id: &[u8; 32]) -> Option<KitriWorkflowStatus> {
        self.workflows.get(workflow_id).map(|w| w.status)
    }

    /// Retrieve the output of a completed workflow.
    ///
    /// Returns `Some` only if the workflow has reached `Complete` status.
    pub fn output(&self, workflow_id: &[u8; 32]) -> Option<&[u8]> {
        self.workflows
            .get(workflow_id)
            .and_then(|w| w.output.as_deref())
    }

    /// Returns the event log for a workflow, if it exists.
    ///
    /// Called by the runtime to fulfill `PersistEventLog` actions.
    pub fn event_log(&self, workflow_id: &[u8; 32]) -> Option<&KitriEventLog> {
        self.workflows.get(workflow_id).map(|w| &w.event_log)
    }

    /// Remove a terminal workflow from the engine, freeing its memory.
    ///
    /// Returns `true` if the workflow was evicted. Returns `false` if the
    /// workflow doesn't exist or is not in a terminal state (`Complete` or
    /// `Failed`), preventing premature removal of in-progress workflows.
    pub fn evict(&mut self, workflow_id: &[u8; 32]) -> bool {
        if let Some(state) = self.workflows.get(workflow_id) {
            if matches!(
                state.status,
                KitriWorkflowStatus::Complete | KitriWorkflowStatus::Failed
            ) {
                self.workflows.remove(workflow_id);
                return true;
            }
        }
        false
    }

    /// Returns the number of tracked workflows.
    pub fn workflow_count(&self) -> usize {
        self.workflows.len()
    }

    /// Process an inbound event, returning zero or more outbound actions.
    pub fn handle(&mut self, event: KitriEngineEvent) -> Vec<KitriAction> {
        match event {
            KitriEngineEvent::Submit { program, input } => self.handle_submit(program, input),
            KitriEngineEvent::IoRequested { workflow_id, op } => {
                self.handle_io_requested(workflow_id, op)
            }
            KitriEngineEvent::IoResolved {
                workflow_id,
                seq,
                result,
            } => self.handle_io_resolved(workflow_id, seq, result),
            KitriEngineEvent::WorkflowComplete {
                workflow_id,
                output,
            } => self.handle_complete(workflow_id, output),
            KitriEngineEvent::WorkflowFailed { workflow_id, error } => {
                self.handle_failed(workflow_id, error)
            }
            KitriEngineEvent::CheckpointRequested {
                workflow_id,
                state,
                state_cid,
            } => self.handle_checkpoint(workflow_id, state, state_cid),
            KitriEngineEvent::CompensatorRegistered {
                workflow_id,
                step_id,
                rollback_op,
            } => self.handle_compensator(workflow_id, step_id, rollback_op),
            KitriEngineEvent::CompensationComplete { workflow_id } => {
                self.handle_compensation_complete(workflow_id)
            }
        }
    }

    fn handle_submit(&mut self, program: KitriProgram, input: Vec<u8>) -> Vec<KitriAction> {
        let wf_id = kitri_workflow_id(&program.cid, &input);

        if let Some(existing) = self.workflows.get_mut(&wf_id) {
            if existing.status == KitriWorkflowStatus::Failed {
                // Enforce retry policy: reject re-submission if max_retries exhausted.
                let max_retries = existing.program.retry_policy.max_retries;
                if existing.attempt >= max_retries {
                    return vec![KitriAction::Deduplicated { workflow_id: wf_id }];
                }

                // Allow retry of failed workflows. The event log is preserved
                // as an audit trail (not for cached-result dedup — new I/O gets
                // new seq numbers). Append a WorkflowRestarted boundary event
                // so recovery algorithms skip the stale WorkflowFailed terminal
                // event and understand this as a distinct execution attempt.
                let delay_ms = existing
                    .program
                    .retry_policy
                    .backoff
                    .delay_ms(existing.attempt);
                existing.attempt += 1;
                let seq = existing.next_seq;
                existing.next_seq += 1;
                existing
                    .event_log
                    .append(crate::event::KitriEvent::WorkflowRestarted {
                        seq,
                        attempt: existing.attempt,
                    });
                existing.status = KitriWorkflowStatus::Pending;
                existing.pending_error = None;
                return vec![
                    KitriAction::PersistEventLog { workflow_id: wf_id },
                    KitriAction::RetryAfter {
                        workflow_id: wf_id,
                        delay_ms,
                    },
                    KitriAction::WorkflowAccepted { workflow_id: wf_id },
                ];
            }
            return vec![KitriAction::Deduplicated { workflow_id: wf_id }];
        }

        self.workflows.insert(
            wf_id,
            WorkflowState {
                program,
                status: KitriWorkflowStatus::Pending,
                event_log: KitriEventLog::new(wf_id),
                staging: StagingBuffer::new(),
                compensation: CompensationLog::new(),
                next_seq: 0,
                output: None,
                pending_error: None,
                attempt: 0,
            },
        );

        vec![KitriAction::WorkflowAccepted { workflow_id: wf_id }]
    }

    fn handle_io_requested(&mut self, workflow_id: [u8; 32], op: KitriIoOp) -> Vec<KitriAction> {
        let Some(state) = self.workflows.get_mut(&workflow_id) else {
            return vec![];
        };

        // Only accept I/O requests from active, non-waiting states.
        if !matches!(
            state.status,
            KitriWorkflowStatus::Pending | KitriWorkflowStatus::Executing
        ) {
            return vec![];
        }

        let seq = state.next_seq;
        state.next_seq += 1;
        state.status = KitriWorkflowStatus::WaitingForIo;

        // Log the request in the event log.
        state
            .event_log
            .append(crate::event::KitriEvent::IoRequested {
                seq,
                op: op.clone(),
            });

        // Stage side effects (Publish, Store, Seal, Spawn).
        if op.is_side_effect() {
            state.staging.stage(StagedWrite {
                seq,
                op: op.clone(),
            });
        }

        vec![
            KitriAction::ExecuteIo {
                workflow_id,
                seq,
                op,
            },
            KitriAction::PersistEventLog { workflow_id },
        ]
    }

    fn handle_io_resolved(
        &mut self,
        workflow_id: [u8; 32],
        seq: u64,
        result: KitriIoResult,
    ) -> Vec<KitriAction> {
        let Some(state) = self.workflows.get_mut(&workflow_id) else {
            return vec![];
        };

        // Only accept resolutions when actually waiting for I/O.
        if state.status != KitriWorkflowStatus::WaitingForIo {
            return vec![];
        }

        // Validate the seq matches the outstanding I/O request. Since we only
        // allow one outstanding I/O at a time (single WaitingForIo state), the
        // expected seq is always next_seq - 1 (the seq assigned in handle_io_requested).
        // Rejecting stale/corrupted seq values prevents event log corruption
        // that would break cached_result() lookups during crash-recovery replay.
        let expected_seq = state.next_seq - 1;
        if seq != expected_seq {
            return vec![];
        }

        state.status = KitriWorkflowStatus::Executing;

        // If the I/O failed, remove the corresponding staged write.
        // The side effect never executed, so it must not be committed
        // on workflow completion.
        if matches!(result, KitriIoResult::Failed { .. }) {
            state.staging.unstage(seq);
        }

        state
            .event_log
            .append(crate::event::KitriEvent::IoResolved { seq, result });

        vec![KitriAction::PersistEventLog { workflow_id }]
    }

    fn handle_complete(&mut self, workflow_id: [u8; 32], output: Vec<u8>) -> Vec<KitriAction> {
        let Some(state) = self.workflows.get_mut(&workflow_id) else {
            return vec![];
        };

        // Only accept completion from Pending (zero-I/O) or Executing (all I/O resolved).
        // WaitingForIo is excluded: completing with unresolved I/O would commit
        // staged writes whose side effects haven't been confirmed.
        if !matches!(
            state.status,
            KitriWorkflowStatus::Pending | KitriWorkflowStatus::Executing
        ) {
            return vec![];
        }

        state.status = KitriWorkflowStatus::Complete;
        state.output = Some(output.clone());
        state.compensation.discard_all();

        let seq = state.next_seq;
        state.next_seq += 1;
        state
            .event_log
            .append(crate::event::KitriEvent::WorkflowCompleted { seq });

        // Crash-safety invariant: persist the event log (WAL) BEFORE committing
        // staged writes. If the runtime crashes after PersistEventLog but before
        // CommitStagedWrite, recovery sees WorkflowCompleted and replays the
        // staged writes. If it crashes before PersistEventLog, the workflow is
        // still in-progress and can be re-executed safely.
        let mut actions = vec![KitriAction::PersistEventLog { workflow_id }];

        actions.extend(state.staging.commit_all().into_iter().map(|w| {
            KitriAction::CommitStagedWrite {
                workflow_id,
                write: w,
            }
        }));

        actions.push(KitriAction::Complete {
            workflow_id,
            output,
        });

        actions
    }

    /// Handle a workflow failure.
    ///
    /// Accepts failure from `Pending`, `Executing`, or `WaitingForIo`. When
    /// failing from `WaitingForIo`, the engine does NOT emit a cancellation
    /// action for the in-flight I/O — the runtime must drop any pending
    /// `ExecuteIo` result for this workflow. The `handle_io_resolved` status
    /// guard will reject stale resolutions since the workflow has moved to
    /// `Compensating` or `Failed`.
    fn handle_failed(&mut self, workflow_id: [u8; 32], error: String) -> Vec<KitriAction> {
        let Some(state) = self.workflows.get_mut(&workflow_id) else {
            return vec![];
        };

        // Reject failure for terminal or already-compensating states.
        if matches!(
            state.status,
            KitriWorkflowStatus::Complete
                | KitriWorkflowStatus::Failed
                | KitriWorkflowStatus::Compensating
        ) {
            return vec![];
        }

        state.staging.discard_all();

        let compensators = state.compensation.drain_reverse();
        let mut actions: Vec<KitriAction> = compensators
            .into_iter()
            .map(|c| KitriAction::ExecuteCompensator {
                workflow_id,
                step_id: c.step_id,
                rollback_op: c.rollback_op,
            })
            .collect();

        if actions.is_empty() {
            // No compensators — fail immediately.
            state.status = KitriWorkflowStatus::Failed;
            let seq = state.next_seq;
            state.next_seq += 1;
            state
                .event_log
                .append(crate::event::KitriEvent::WorkflowFailed {
                    seq,
                    error: error.clone(),
                });
            actions.push(KitriAction::PersistEventLog { workflow_id });
            actions.push(KitriAction::Failed { workflow_id, error });
        } else {
            // Compensators need to run first. Defer Failed until
            // CompensationComplete arrives.
            state.status = KitriWorkflowStatus::Compensating;
            let seq = state.next_seq;
            state.next_seq += 1;
            state
                .event_log
                .append(crate::event::KitriEvent::CompensationStarted {
                    seq,
                    error: error.clone(),
                });
            state.pending_error = Some(error);
            // WAL invariant: persist CompensationStarted BEFORE dispatching
            // any ExecuteCompensator action. If the process crashes after a
            // compensator runs, recovery must see Compensating state in the
            // log so it knows not to re-execute the workflow from scratch.
            actions.insert(0, KitriAction::PersistEventLog { workflow_id });
        }

        actions
    }

    fn handle_compensation_complete(&mut self, workflow_id: [u8; 32]) -> Vec<KitriAction> {
        let Some(state) = self.workflows.get_mut(&workflow_id) else {
            return vec![];
        };

        if state.status != KitriWorkflowStatus::Compensating {
            return vec![];
        }

        state.status = KitriWorkflowStatus::Failed;
        let error = state.pending_error.take().unwrap_or_default();
        let seq = state.next_seq;
        state.next_seq += 1;
        state
            .event_log
            .append(crate::event::KitriEvent::WorkflowFailed {
                seq,
                error: error.clone(),
            });
        vec![
            KitriAction::PersistEventLog { workflow_id },
            KitriAction::Failed { workflow_id, error },
        ]
    }

    fn handle_checkpoint(
        &mut self,
        workflow_id: [u8; 32],
        checkpoint_state: Vec<u8>,
        state_cid: [u8; 32],
    ) -> Vec<KitriAction> {
        let Some(state) = self.workflows.get_mut(&workflow_id) else {
            return vec![];
        };

        // Only checkpoint alive workflows.
        if !matches!(
            state.status,
            KitriWorkflowStatus::Pending
                | KitriWorkflowStatus::Executing
                | KitriWorkflowStatus::WaitingForIo
        ) {
            return vec![];
        }

        let seq = state.next_seq;
        state.next_seq += 1;
        state
            .event_log
            .append(crate::event::KitriEvent::CheckpointSaved { seq, state_cid });

        // WAL invariant: persist the event log entry (CheckpointSaved with CID)
        // BEFORE writing checkpoint bytes to content-addressed storage.
        // Crash scenario A: crash before PersistEventLog — log has no checkpoint,
        //   recovery replays from the prior checkpoint or from scratch (safe).
        // Crash scenario B: crash after PersistEventLog but before PersistCheckpoint —
        //   log records the CID but the bytes were never stored. Recovery must treat
        //   this as a missing checkpoint and fall back to the prior checkpoint or
        //   replay from scratch. The runtime must handle a CID-not-found gracefully.
        vec![
            KitriAction::PersistEventLog { workflow_id },
            KitriAction::PersistCheckpoint {
                workflow_id,
                state: checkpoint_state,
                state_cid,
            },
        ]
    }

    fn handle_compensator(
        &mut self,
        workflow_id: [u8; 32],
        step_id: u64,
        rollback_op: KitriIoOp,
    ) -> Vec<KitriAction> {
        let Some(state) = self.workflows.get_mut(&workflow_id) else {
            return vec![];
        };

        // Only register compensators during active execution.
        if !matches!(
            state.status,
            KitriWorkflowStatus::Pending | KitriWorkflowStatus::Executing
        ) {
            return vec![];
        }

        let seq = state.next_seq;
        state.next_seq += 1;
        state
            .event_log
            .append(crate::event::KitriEvent::CompensatorRegistered {
                seq,
                step_id,
                rollback_op: rollback_op.clone(),
            });

        state
            .compensation
            .register(crate::compensation::Compensator {
                step_id,
                rollback_op,
            });

        vec![KitriAction::PersistEventLog { workflow_id }]
    }
}

impl Default for KitriEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use crate::io::{KitriIoOp, KitriIoResult};
    use crate::program::{kitri_workflow_id, KitriProgram};
    use crate::retry::{BackoffStrategy, RetryPolicy};
    use crate::trust::{CapabilitySet, TrustTier};

    use super::*;

    fn test_program() -> KitriProgram {
        KitriProgram {
            cid: [0xAA; 32],
            name: "test-workflow".into(),
            version: "0.1.0".into(),
            trust_tier: TrustTier::Owner,
            capabilities: CapabilitySet::new(),
            retry_policy: RetryPolicy::default(),
            prefer_native: true,
        }
    }

    #[test]
    fn engine_submit_returns_workflow_id() {
        let mut engine = KitriEngine::new();
        let program = test_program();
        let input = vec![1, 2, 3];

        let actions = engine.handle(KitriEngineEvent::Submit {
            program: program.clone(),
            input: input.clone(),
        });

        let expected_id = kitri_workflow_id(&program.cid, &input);
        assert!(actions.iter().any(|a| matches!(
            a,
            KitriAction::WorkflowAccepted { workflow_id } if *workflow_id == expected_id
        )));
    }

    #[test]
    fn engine_duplicate_submit_deduplicates() {
        let mut engine = KitriEngine::new();
        let program = test_program();
        let input = vec![1, 2, 3];

        engine.handle(KitriEngineEvent::Submit {
            program: program.clone(),
            input: input.clone(),
        });
        let actions = engine.handle(KitriEngineEvent::Submit { program, input });

        assert!(actions
            .iter()
            .any(|a| matches!(a, KitriAction::Deduplicated { .. })));
    }

    #[test]
    fn engine_io_requested_stages_side_effects() {
        let mut engine = KitriEngine::new();
        let program = test_program();
        let input = vec![1, 2, 3];
        let wf_id = kitri_workflow_id(&program.cid, &input);

        engine.handle(KitriEngineEvent::Submit { program, input });

        let actions = engine.handle(KitriEngineEvent::IoRequested {
            workflow_id: wf_id,
            op: KitriIoOp::Publish {
                topic: "test/out".into(),
                payload: vec![42],
            },
        });

        assert!(actions
            .iter()
            .any(|a| matches!(a, KitriAction::ExecuteIo { .. })));
    }

    #[test]
    fn engine_workflow_complete_commits_staged_writes() {
        let mut engine = KitriEngine::new();
        let program = test_program();
        let input = vec![1, 2, 3];
        let wf_id = kitri_workflow_id(&program.cid, &input);

        engine.handle(KitriEngineEvent::Submit { program, input });

        engine.handle(KitriEngineEvent::IoRequested {
            workflow_id: wf_id,
            op: KitriIoOp::Publish {
                topic: "test/out".into(),
                payload: vec![42],
            },
        });
        engine.handle(KitriEngineEvent::IoResolved {
            workflow_id: wf_id,
            seq: 0,
            result: KitriIoResult::Published,
        });

        let actions = engine.handle(KitriEngineEvent::WorkflowComplete {
            workflow_id: wf_id,
            output: vec![99],
        });

        assert!(actions
            .iter()
            .any(|a| matches!(a, KitriAction::CommitStagedWrite { .. })));
        assert!(actions
            .iter()
            .any(|a| matches!(a, KitriAction::Complete { .. })));
    }

    #[test]
    fn engine_workflow_failed_discards_staged_writes() {
        let mut engine = KitriEngine::new();
        let program = test_program();
        let input = vec![1, 2, 3];
        let wf_id = kitri_workflow_id(&program.cid, &input);

        engine.handle(KitriEngineEvent::Submit { program, input });

        engine.handle(KitriEngineEvent::IoRequested {
            workflow_id: wf_id,
            op: KitriIoOp::Store { data: vec![1] },
        });

        let actions = engine.handle(KitriEngineEvent::WorkflowFailed {
            workflow_id: wf_id,
            error: "boom".into(),
        });

        assert!(!actions
            .iter()
            .any(|a| matches!(a, KitriAction::CommitStagedWrite { .. })));
        assert!(actions
            .iter()
            .any(|a| matches!(a, KitriAction::Failed { .. })));
    }

    #[test]
    fn engine_failed_with_compensators_defers_failed_action() {
        let mut engine = KitriEngine::new();
        let program = test_program();
        let input = vec![1, 2, 3];
        let wf_id = kitri_workflow_id(&program.cid, &input);

        engine.handle(KitriEngineEvent::Submit { program, input });
        engine.handle(KitriEngineEvent::CompensatorRegistered {
            workflow_id: wf_id,
            step_id: 0,
            rollback_op: KitriIoOp::Query {
                topic: "undo/step0".into(),
                payload: vec![],
            },
        });
        engine.handle(KitriEngineEvent::CompensatorRegistered {
            workflow_id: wf_id,
            step_id: 1,
            rollback_op: KitriIoOp::Query {
                topic: "undo/step1".into(),
                payload: vec![],
            },
        });

        let actions = engine.handle(KitriEngineEvent::WorkflowFailed {
            workflow_id: wf_id,
            error: "boom".into(),
        });

        // Should be Compensating, not Failed yet.
        assert_eq!(
            engine.status(&wf_id),
            Some(KitriWorkflowStatus::Compensating)
        );

        // Should emit ExecuteCompensator actions in LIFO order, but NOT Failed.
        let comp_ids: Vec<_> = actions
            .iter()
            .filter_map(|a| match a {
                KitriAction::ExecuteCompensator { step_id, .. } => Some(*step_id),
                _ => None,
            })
            .collect();
        assert_eq!(comp_ids, vec![1, 0]); // LIFO
        assert!(!actions
            .iter()
            .any(|a| matches!(a, KitriAction::Failed { .. })));

        // CompensationComplete transitions to Failed.
        let actions = engine.handle(KitriEngineEvent::CompensationComplete { workflow_id: wf_id });
        assert_eq!(engine.status(&wf_id), Some(KitriWorkflowStatus::Failed));
        assert!(actions
            .iter()
            .any(|a| matches!(a, KitriAction::Failed { .. })));
    }

    #[test]
    fn engine_io_resolved_rejected_after_completion() {
        let mut engine = KitriEngine::new();
        let program = test_program();
        let input = vec![1, 2, 3];
        let wf_id = kitri_workflow_id(&program.cid, &input);

        engine.handle(KitriEngineEvent::Submit { program, input });
        // Go through a valid I/O cycle to reach Executing, then Complete.
        engine.handle(KitriEngineEvent::IoRequested {
            workflow_id: wf_id,
            op: KitriIoOp::Fetch { cid: [0; 32] },
        });
        engine.handle(KitriEngineEvent::IoResolved {
            workflow_id: wf_id,
            seq: 0,
            result: KitriIoResult::Fetched { data: vec![1] },
        });
        engine.handle(KitriEngineEvent::WorkflowComplete {
            workflow_id: wf_id,
            output: vec![],
        });

        // Stale IoResolved should be rejected — status stays Complete.
        let actions = engine.handle(KitriEngineEvent::IoResolved {
            workflow_id: wf_id,
            seq: 0,
            result: KitriIoResult::Fetched { data: vec![1] },
        });
        assert!(actions.is_empty());
        assert_eq!(engine.status(&wf_id), Some(KitriWorkflowStatus::Complete));
    }

    #[test]
    fn engine_checkpoint_increments_seq() {
        let mut engine = KitriEngine::new();
        let program = test_program();
        let input = vec![1, 2, 3];
        let wf_id = kitri_workflow_id(&program.cid, &input);

        engine.handle(KitriEngineEvent::Submit { program, input });

        // I/O request gets seq 0.
        let actions = engine.handle(KitriEngineEvent::IoRequested {
            workflow_id: wf_id,
            op: KitriIoOp::Fetch { cid: [0; 32] },
        });
        assert!(actions
            .iter()
            .any(|a| matches!(a, KitriAction::ExecuteIo { seq: 0, .. })));

        engine.handle(KitriEngineEvent::IoResolved {
            workflow_id: wf_id,
            seq: 0,
            result: KitriIoResult::Fetched { data: vec![1] },
        });

        // Checkpoint should consume seq 1.
        engine.handle(KitriEngineEvent::CheckpointRequested {
            workflow_id: wf_id,
            state: vec![],
            state_cid: [0xAA; 32],
        });

        // Next I/O request should get seq 2 (not 1).
        let actions = engine.handle(KitriEngineEvent::IoRequested {
            workflow_id: wf_id,
            op: KitriIoOp::Fetch { cid: [1; 32] },
        });
        assert!(actions
            .iter()
            .any(|a| matches!(a, KitriAction::ExecuteIo { seq: 2, .. })));
    }

    #[test]
    fn engine_status_tracking() {
        let mut engine = KitriEngine::new();
        let program = test_program();
        let input = vec![1, 2, 3];
        let wf_id = kitri_workflow_id(&program.cid, &input);

        assert_eq!(engine.status(&wf_id), None);

        engine.handle(KitriEngineEvent::Submit { program, input });
        assert_eq!(engine.status(&wf_id), Some(KitriWorkflowStatus::Pending));

        // Transition through Executing via I/O cycle.
        engine.handle(KitriEngineEvent::IoRequested {
            workflow_id: wf_id,
            op: KitriIoOp::Fetch { cid: [0; 32] },
        });
        assert_eq!(
            engine.status(&wf_id),
            Some(KitriWorkflowStatus::WaitingForIo)
        );

        engine.handle(KitriEngineEvent::IoResolved {
            workflow_id: wf_id,
            seq: 0,
            result: KitriIoResult::Fetched { data: vec![1] },
        });
        assert_eq!(engine.status(&wf_id), Some(KitriWorkflowStatus::Executing));

        engine.handle(KitriEngineEvent::WorkflowComplete {
            workflow_id: wf_id,
            output: vec![],
        });
        assert_eq!(engine.status(&wf_id), Some(KitriWorkflowStatus::Complete));
    }

    #[test]
    fn engine_complete_rejected_during_compensating() {
        let mut engine = KitriEngine::new();
        let program = test_program();
        let input = vec![1, 2, 3];
        let wf_id = kitri_workflow_id(&program.cid, &input);

        engine.handle(KitriEngineEvent::Submit { program, input });

        // Get to Executing state via I/O.
        engine.handle(KitriEngineEvent::IoRequested {
            workflow_id: wf_id,
            op: KitriIoOp::Fetch { cid: [0; 32] },
        });
        engine.handle(KitriEngineEvent::IoResolved {
            workflow_id: wf_id,
            seq: 0,
            result: KitriIoResult::Fetched { data: vec![1] },
        });

        engine.handle(KitriEngineEvent::CompensatorRegistered {
            workflow_id: wf_id,
            step_id: 0,
            rollback_op: KitriIoOp::Query {
                topic: "undo".into(),
                payload: vec![],
            },
        });

        // Fail -> Compensating.
        engine.handle(KitriEngineEvent::WorkflowFailed {
            workflow_id: wf_id,
            error: "boom".into(),
        });
        assert_eq!(
            engine.status(&wf_id),
            Some(KitriWorkflowStatus::Compensating)
        );

        // Complete should be rejected during Compensating.
        let actions = engine.handle(KitriEngineEvent::WorkflowComplete {
            workflow_id: wf_id,
            output: vec![99],
        });
        assert!(actions.is_empty());
        assert_eq!(
            engine.status(&wf_id),
            Some(KitriWorkflowStatus::Compensating)
        );
    }

    #[test]
    fn engine_failed_rejected_for_complete_workflow() {
        let mut engine = KitriEngine::new();
        let program = test_program();
        let input = vec![1, 2, 3];
        let wf_id = kitri_workflow_id(&program.cid, &input);

        engine.handle(KitriEngineEvent::Submit { program, input });
        engine.handle(KitriEngineEvent::IoRequested {
            workflow_id: wf_id,
            op: KitriIoOp::Fetch { cid: [0; 32] },
        });
        engine.handle(KitriEngineEvent::IoResolved {
            workflow_id: wf_id,
            seq: 0,
            result: KitriIoResult::Fetched { data: vec![1] },
        });
        engine.handle(KitriEngineEvent::WorkflowComplete {
            workflow_id: wf_id,
            output: vec![],
        });

        // Failed should be rejected for a Complete workflow.
        let actions = engine.handle(KitriEngineEvent::WorkflowFailed {
            workflow_id: wf_id,
            error: "stale".into(),
        });
        assert!(actions.is_empty());
        assert_eq!(engine.status(&wf_id), Some(KitriWorkflowStatus::Complete));
    }

    #[test]
    fn engine_compensator_registration_emits_persist() {
        let mut engine = KitriEngine::new();
        let program = test_program();
        let input = vec![1, 2, 3];
        let wf_id = kitri_workflow_id(&program.cid, &input);

        engine.handle(KitriEngineEvent::Submit { program, input });

        let actions = engine.handle(KitriEngineEvent::CompensatorRegistered {
            workflow_id: wf_id,
            step_id: 0,
            rollback_op: KitriIoOp::Query {
                topic: "undo".into(),
                payload: vec![],
            },
        });

        assert!(actions
            .iter()
            .any(|a| matches!(a, KitriAction::PersistEventLog { .. })));
    }

    #[test]
    fn engine_io_requested_rejected_for_terminal_workflow() {
        let mut engine = KitriEngine::new();
        let program = test_program();
        let input = vec![1, 2, 3];
        let wf_id = kitri_workflow_id(&program.cid, &input);

        engine.handle(KitriEngineEvent::Submit { program, input });
        engine.handle(KitriEngineEvent::IoRequested {
            workflow_id: wf_id,
            op: KitriIoOp::Fetch { cid: [0; 32] },
        });
        engine.handle(KitriEngineEvent::IoResolved {
            workflow_id: wf_id,
            seq: 0,
            result: KitriIoResult::Fetched { data: vec![1] },
        });
        engine.handle(KitriEngineEvent::WorkflowComplete {
            workflow_id: wf_id,
            output: vec![],
        });

        // Stale IoRequested should be rejected for a Complete workflow.
        let actions = engine.handle(KitriEngineEvent::IoRequested {
            workflow_id: wf_id,
            op: KitriIoOp::Fetch { cid: [1; 32] },
        });
        assert!(actions.is_empty());
        assert_eq!(engine.status(&wf_id), Some(KitriWorkflowStatus::Complete));
    }

    #[test]
    fn engine_io_requested_rejected_while_waiting() {
        let mut engine = KitriEngine::new();
        let program = test_program();
        let input = vec![1, 2, 3];
        let wf_id = kitri_workflow_id(&program.cid, &input);

        engine.handle(KitriEngineEvent::Submit { program, input });
        engine.handle(KitriEngineEvent::IoRequested {
            workflow_id: wf_id,
            op: KitriIoOp::Fetch { cid: [0; 32] },
        });
        assert_eq!(
            engine.status(&wf_id),
            Some(KitriWorkflowStatus::WaitingForIo)
        );

        // Second IoRequested while already waiting should be rejected.
        let actions = engine.handle(KitriEngineEvent::IoRequested {
            workflow_id: wf_id,
            op: KitriIoOp::Fetch { cid: [1; 32] },
        });
        assert!(actions.is_empty());
        assert_eq!(
            engine.status(&wf_id),
            Some(KitriWorkflowStatus::WaitingForIo)
        );
    }

    #[test]
    fn engine_checkpoint_rejected_for_terminal_workflow() {
        let mut engine = KitriEngine::new();
        let program = test_program();
        let input = vec![1, 2, 3];
        let wf_id = kitri_workflow_id(&program.cid, &input);

        engine.handle(KitriEngineEvent::Submit { program, input });
        engine.handle(KitriEngineEvent::IoRequested {
            workflow_id: wf_id,
            op: KitriIoOp::Fetch { cid: [0; 32] },
        });
        engine.handle(KitriEngineEvent::IoResolved {
            workflow_id: wf_id,
            seq: 0,
            result: KitriIoResult::Fetched { data: vec![1] },
        });
        engine.handle(KitriEngineEvent::WorkflowFailed {
            workflow_id: wf_id,
            error: "boom".into(),
        });

        // Checkpoint should be rejected for a Failed workflow.
        let actions = engine.handle(KitriEngineEvent::CheckpointRequested {
            workflow_id: wf_id,
            state: vec![],
            state_cid: [0xCC; 32],
        });
        assert!(actions.is_empty());
        assert_eq!(engine.status(&wf_id), Some(KitriWorkflowStatus::Failed));
    }

    #[test]
    fn engine_compensator_rejected_for_terminal_workflow() {
        let mut engine = KitriEngine::new();
        let program = test_program();
        let input = vec![1, 2, 3];
        let wf_id = kitri_workflow_id(&program.cid, &input);

        engine.handle(KitriEngineEvent::Submit { program, input });
        engine.handle(KitriEngineEvent::IoRequested {
            workflow_id: wf_id,
            op: KitriIoOp::Fetch { cid: [0; 32] },
        });
        engine.handle(KitriEngineEvent::IoResolved {
            workflow_id: wf_id,
            seq: 0,
            result: KitriIoResult::Fetched { data: vec![1] },
        });
        engine.handle(KitriEngineEvent::WorkflowComplete {
            workflow_id: wf_id,
            output: vec![],
        });

        // Compensator registration should be rejected for a Complete workflow.
        let actions = engine.handle(KitriEngineEvent::CompensatorRegistered {
            workflow_id: wf_id,
            step_id: 99,
            rollback_op: KitriIoOp::Query {
                topic: "undo".into(),
                payload: vec![],
            },
        });
        assert!(actions.is_empty());
        assert_eq!(engine.status(&wf_id), Some(KitriWorkflowStatus::Complete));
    }

    #[test]
    fn engine_zero_io_workflow_can_complete() {
        let mut engine = KitriEngine::new();
        let program = test_program();
        let input = vec![1, 2, 3];
        let wf_id = kitri_workflow_id(&program.cid, &input);

        engine.handle(KitriEngineEvent::Submit { program, input });
        assert_eq!(engine.status(&wf_id), Some(KitriWorkflowStatus::Pending));

        // A zero-I/O workflow completes directly from Pending.
        let actions = engine.handle(KitriEngineEvent::WorkflowComplete {
            workflow_id: wf_id,
            output: vec![42],
        });

        assert!(actions
            .iter()
            .any(|a| matches!(a, KitriAction::Complete { .. })));
        assert_eq!(engine.status(&wf_id), Some(KitriWorkflowStatus::Complete));
    }

    #[test]
    fn engine_failed_emits_persist_event_log() {
        let mut engine = KitriEngine::new();
        let program = test_program();
        let input = vec![1, 2, 3];
        let wf_id = kitri_workflow_id(&program.cid, &input);

        engine.handle(KitriEngineEvent::Submit { program, input });

        let actions = engine.handle(KitriEngineEvent::WorkflowFailed {
            workflow_id: wf_id,
            error: "boom".into(),
        });

        // PersistEventLog must appear before Failed for crash safety.
        let persist_pos = actions
            .iter()
            .position(|a| matches!(a, KitriAction::PersistEventLog { .. }));
        let failed_pos = actions
            .iter()
            .position(|a| matches!(a, KitriAction::Failed { .. }));
        assert!(persist_pos.is_some());
        assert!(failed_pos.is_some());
        assert!(persist_pos.unwrap() < failed_pos.unwrap());
    }

    #[test]
    fn engine_compensation_complete_emits_persist_event_log() {
        let mut engine = KitriEngine::new();
        let program = test_program();
        let input = vec![1, 2, 3];
        let wf_id = kitri_workflow_id(&program.cid, &input);

        engine.handle(KitriEngineEvent::Submit { program, input });
        engine.handle(KitriEngineEvent::CompensatorRegistered {
            workflow_id: wf_id,
            step_id: 0,
            rollback_op: KitriIoOp::Query {
                topic: "undo".into(),
                payload: vec![],
            },
        });
        engine.handle(KitriEngineEvent::WorkflowFailed {
            workflow_id: wf_id,
            error: "boom".into(),
        });

        let actions = engine.handle(KitriEngineEvent::CompensationComplete { workflow_id: wf_id });

        // PersistEventLog must appear before Failed for crash safety.
        let persist_pos = actions
            .iter()
            .position(|a| matches!(a, KitriAction::PersistEventLog { .. }));
        let failed_pos = actions
            .iter()
            .position(|a| matches!(a, KitriAction::Failed { .. }));
        assert!(persist_pos.is_some());
        assert!(failed_pos.is_some());
        assert!(persist_pos.unwrap() < failed_pos.unwrap());
    }

    #[test]
    fn engine_failed_workflow_can_be_resubmitted() {
        let mut engine = KitriEngine::new();
        let program = test_program();
        let input = vec![1, 2, 3];
        let wf_id = kitri_workflow_id(&program.cid, &input);

        engine.handle(KitriEngineEvent::Submit {
            program: program.clone(),
            input: input.clone(),
        });
        engine.handle(KitriEngineEvent::WorkflowFailed {
            workflow_id: wf_id,
            error: "transient".into(),
        });
        assert_eq!(engine.status(&wf_id), Some(KitriWorkflowStatus::Failed));

        // Re-submit the same (program, input) — should be accepted, not deduplicated.
        let actions = engine.handle(KitriEngineEvent::Submit { program, input });
        assert!(actions
            .iter()
            .any(|a| matches!(a, KitriAction::WorkflowAccepted { .. })));
        assert_eq!(engine.status(&wf_id), Some(KitriWorkflowStatus::Pending));
    }

    #[test]
    fn engine_complete_workflow_stays_deduplicated() {
        let mut engine = KitriEngine::new();
        let program = test_program();
        let input = vec![1, 2, 3];
        let wf_id = kitri_workflow_id(&program.cid, &input);

        engine.handle(KitriEngineEvent::Submit {
            program: program.clone(),
            input: input.clone(),
        });
        // Complete directly (zero-I/O).
        engine.handle(KitriEngineEvent::WorkflowComplete {
            workflow_id: wf_id,
            output: vec![99],
        });
        assert_eq!(engine.status(&wf_id), Some(KitriWorkflowStatus::Complete));

        // Re-submit should be deduplicated — completed workflows are final.
        let actions = engine.handle(KitriEngineEvent::Submit { program, input });
        assert!(actions
            .iter()
            .any(|a| matches!(a, KitriAction::Deduplicated { .. })));
        assert_eq!(engine.status(&wf_id), Some(KitriWorkflowStatus::Complete));
    }

    #[test]
    fn engine_complete_rejected_while_waiting_for_io() {
        let mut engine = KitriEngine::new();
        let program = test_program();
        let input = vec![1, 2, 3];
        let wf_id = kitri_workflow_id(&program.cid, &input);

        engine.handle(KitriEngineEvent::Submit { program, input });
        engine.handle(KitriEngineEvent::IoRequested {
            workflow_id: wf_id,
            op: KitriIoOp::Store { data: vec![1] },
        });
        assert_eq!(
            engine.status(&wf_id),
            Some(KitriWorkflowStatus::WaitingForIo)
        );

        // Complete while waiting for I/O should be rejected — unresolved
        // staged writes would be committed without confirmation.
        let actions = engine.handle(KitriEngineEvent::WorkflowComplete {
            workflow_id: wf_id,
            output: vec![99],
        });
        assert!(actions.is_empty());
        assert_eq!(
            engine.status(&wf_id),
            Some(KitriWorkflowStatus::WaitingForIo)
        );
    }

    #[test]
    fn engine_compensation_path_emits_persist_event_log() {
        let mut engine = KitriEngine::new();
        let program = test_program();
        let input = vec![1, 2, 3];
        let wf_id = kitri_workflow_id(&program.cid, &input);

        engine.handle(KitriEngineEvent::Submit { program, input });
        engine.handle(KitriEngineEvent::CompensatorRegistered {
            workflow_id: wf_id,
            step_id: 0,
            rollback_op: KitriIoOp::Query {
                topic: "undo".into(),
                payload: vec![],
            },
        });

        let actions = engine.handle(KitriEngineEvent::WorkflowFailed {
            workflow_id: wf_id,
            error: "boom".into(),
        });

        // WAL invariant: PersistEventLog must come before ExecuteCompensator.
        let persist_pos = actions
            .iter()
            .position(|a| matches!(a, KitriAction::PersistEventLog { .. }));
        let compensator_pos = actions
            .iter()
            .position(|a| matches!(a, KitriAction::ExecuteCompensator { .. }));
        assert!(persist_pos.is_some());
        assert!(compensator_pos.is_some());
        assert!(persist_pos.unwrap() < compensator_pos.unwrap());
    }

    #[test]
    fn engine_checkpoint_emits_persist_checkpoint() {
        let mut engine = KitriEngine::new();
        let program = test_program();
        let input = vec![1, 2, 3];
        let wf_id = kitri_workflow_id(&program.cid, &input);

        engine.handle(KitriEngineEvent::Submit { program, input });

        let state_bytes = vec![10, 20, 30];
        let state_cid = [0xAA; 32];
        let actions = engine.handle(KitriEngineEvent::CheckpointRequested {
            workflow_id: wf_id,
            state: state_bytes,
            state_cid,
        });

        // WAL invariant: PersistEventLog must come before PersistCheckpoint.
        let persist_log_pos = actions
            .iter()
            .position(|a| matches!(a, KitriAction::PersistEventLog { .. }));
        let persist_ckpt_pos = actions.iter().position(
            |a| matches!(a, KitriAction::PersistCheckpoint { state_cid: cid, .. } if *cid == [0xAA; 32]),
        );
        assert!(persist_log_pos.is_some());
        assert!(persist_ckpt_pos.is_some());
        assert!(persist_log_pos.unwrap() < persist_ckpt_pos.unwrap());
    }

    #[test]
    fn engine_capability_set_deduplicates() {
        use crate::trust::{CapabilityDecl, CapabilitySet};

        let mut caps = CapabilitySet::new();
        caps.add(CapabilityDecl::Infer);
        caps.add(CapabilityDecl::Infer); // duplicate
        caps.add(CapabilityDecl::Subscribe {
            topic: "foo".into(),
        });
        caps.add(CapabilityDecl::Subscribe {
            topic: "foo".into(),
        }); // duplicate

        assert_eq!(caps.declarations().len(), 2);
    }

    #[test]
    fn engine_complete_persists_event_log_before_staged_writes() {
        let mut engine = KitriEngine::new();
        let program = test_program();
        let input = vec![1, 2, 3];
        let wf_id = kitri_workflow_id(&program.cid, &input);

        engine.handle(KitriEngineEvent::Submit { program, input });

        // Stage a side-effect I/O.
        engine.handle(KitriEngineEvent::IoRequested {
            workflow_id: wf_id,
            op: KitriIoOp::Publish {
                topic: "test/out".into(),
                payload: vec![42],
            },
        });
        engine.handle(KitriEngineEvent::IoResolved {
            workflow_id: wf_id,
            seq: 0,
            result: KitriIoResult::Published,
        });

        let actions = engine.handle(KitriEngineEvent::WorkflowComplete {
            workflow_id: wf_id,
            output: vec![99],
        });

        // Crash-safety invariant: PersistEventLog must come before CommitStagedWrite.
        // This ensures the WAL records completion before side effects are committed.
        let persist_pos = actions
            .iter()
            .position(|a| matches!(a, KitriAction::PersistEventLog { .. }));
        let commit_pos = actions
            .iter()
            .position(|a| matches!(a, KitriAction::CommitStagedWrite { .. }));
        let complete_pos = actions
            .iter()
            .position(|a| matches!(a, KitriAction::Complete { .. }));
        assert!(persist_pos.is_some());
        assert!(commit_pos.is_some());
        assert!(complete_pos.is_some());
        assert!(persist_pos.unwrap() < commit_pos.unwrap());
        assert!(commit_pos.unwrap() < complete_pos.unwrap());
    }

    #[test]
    fn engine_io_resolved_rejects_wrong_seq() {
        let mut engine = KitriEngine::new();
        let program = test_program();
        let input = vec![1, 2, 3];
        let wf_id = kitri_workflow_id(&program.cid, &input);

        engine.handle(KitriEngineEvent::Submit { program, input });

        // I/O request gets seq 0.
        engine.handle(KitriEngineEvent::IoRequested {
            workflow_id: wf_id,
            op: KitriIoOp::Fetch { cid: [0; 32] },
        });
        assert_eq!(
            engine.status(&wf_id),
            Some(KitriWorkflowStatus::WaitingForIo)
        );

        // Stale seq (99) should be rejected — status stays WaitingForIo.
        let actions = engine.handle(KitriEngineEvent::IoResolved {
            workflow_id: wf_id,
            seq: 99,
            result: KitriIoResult::Fetched { data: vec![1] },
        });
        assert!(actions.is_empty());
        assert_eq!(
            engine.status(&wf_id),
            Some(KitriWorkflowStatus::WaitingForIo)
        );

        // Correct seq (0) should be accepted.
        let actions = engine.handle(KitriEngineEvent::IoResolved {
            workflow_id: wf_id,
            seq: 0,
            result: KitriIoResult::Fetched { data: vec![1] },
        });
        assert!(!actions.is_empty());
        assert_eq!(engine.status(&wf_id), Some(KitriWorkflowStatus::Executing));
    }

    #[test]
    fn engine_retry_appends_restart_boundary_event() {
        let mut engine = KitriEngine::new();
        let program = test_program();
        let input = vec![1, 2, 3];
        let wf_id = kitri_workflow_id(&program.cid, &input);

        engine.handle(KitriEngineEvent::Submit {
            program: program.clone(),
            input: input.clone(),
        });
        engine.handle(KitriEngineEvent::WorkflowFailed {
            workflow_id: wf_id,
            error: "transient".into(),
        });
        assert_eq!(engine.status(&wf_id), Some(KitriWorkflowStatus::Failed));

        // Re-submit — should append WorkflowRestarted boundary event.
        let actions = engine.handle(KitriEngineEvent::Submit {
            program: program.clone(),
            input: input.clone(),
        });
        assert!(actions
            .iter()
            .any(|a| matches!(a, KitriAction::WorkflowAccepted { .. })));
        assert_eq!(engine.status(&wf_id), Some(KitriWorkflowStatus::Pending));

        // Verify the event log contains the restart boundary.
        let state = engine.workflows.get(&wf_id).unwrap();
        let events = state.event_log.events();
        assert!(events.iter().any(|e| matches!(
            e,
            crate::event::KitriEvent::WorkflowRestarted { attempt: 1, .. }
        )));

        // Fail and retry again — attempt should increment.
        engine.handle(KitriEngineEvent::WorkflowFailed {
            workflow_id: wf_id,
            error: "transient again".into(),
        });
        engine.handle(KitriEngineEvent::Submit { program, input });

        let state = engine.workflows.get(&wf_id).unwrap();
        let events = state.event_log.events();
        assert!(events.iter().any(|e| matches!(
            e,
            crate::event::KitriEvent::WorkflowRestarted { attempt: 2, .. }
        )));
    }

    #[test]
    fn engine_retry_emits_persist_event_log() {
        let mut engine = KitriEngine::new();
        let program = test_program();
        let input = vec![1, 2, 3];
        let wf_id = kitri_workflow_id(&program.cid, &input);

        engine.handle(KitriEngineEvent::Submit {
            program: program.clone(),
            input: input.clone(),
        });
        engine.handle(KitriEngineEvent::WorkflowFailed {
            workflow_id: wf_id,
            error: "transient".into(),
        });

        // Re-submit — must emit PersistEventLog, RetryAfter, WorkflowAccepted
        // in that order. The backoff delay must come before the runtime starts
        // the next attempt.
        let actions = engine.handle(KitriEngineEvent::Submit { program, input });

        let persist_pos = actions
            .iter()
            .position(|a| matches!(a, KitriAction::PersistEventLog { .. }));
        let retry_pos = actions
            .iter()
            .position(|a| matches!(a, KitriAction::RetryAfter { .. }));
        let accepted_pos = actions
            .iter()
            .position(|a| matches!(a, KitriAction::WorkflowAccepted { .. }));
        assert!(persist_pos.is_some());
        assert!(retry_pos.is_some());
        assert!(accepted_pos.is_some());
        assert!(persist_pos.unwrap() < retry_pos.unwrap());
        assert!(retry_pos.unwrap() < accepted_pos.unwrap());
    }

    #[test]
    fn engine_last_checkpoint_ignores_prior_attempts() {
        use crate::event::KitriEventLog;

        let mut log = KitriEventLog::new([0; 32]);

        // Attempt 1: checkpoint at seq 3, then failure.
        log.append(crate::event::KitriEvent::CheckpointSaved {
            seq: 3,
            state_cid: [0xAA; 32],
        });
        log.append(crate::event::KitriEvent::WorkflowFailed {
            seq: 5,
            error: "boom".into(),
        });

        // Before restart, checkpoint from attempt 1 is visible.
        assert_eq!(log.last_checkpoint(), Some((3, [0xAA; 32])));

        // Restart boundary.
        log.append(crate::event::KitriEvent::WorkflowRestarted { seq: 6, attempt: 1 });

        // After restart, attempt 1's checkpoint is stale — not returned.
        assert_eq!(log.last_checkpoint(), None);

        // Attempt 2 saves its own checkpoint.
        log.append(crate::event::KitriEvent::CheckpointSaved {
            seq: 8,
            state_cid: [0xBB; 32],
        });

        // Only attempt 2's checkpoint is returned.
        assert_eq!(log.last_checkpoint(), Some((8, [0xBB; 32])));
    }

    #[test]
    fn engine_failed_io_unstages_side_effect() {
        let mut engine = KitriEngine::new();
        let program = test_program();
        let input = vec![1, 2, 3];
        let wf_id = kitri_workflow_id(&program.cid, &input);

        engine.handle(KitriEngineEvent::Submit { program, input });

        // Request a side-effect I/O (Publish → staged).
        engine.handle(KitriEngineEvent::IoRequested {
            workflow_id: wf_id,
            op: KitriIoOp::Publish {
                topic: "test/out".into(),
                payload: vec![42],
            },
        });

        // I/O fails — the staged write should be removed.
        engine.handle(KitriEngineEvent::IoResolved {
            workflow_id: wf_id,
            seq: 0,
            result: KitriIoResult::Failed {
                error: "timeout".into(),
            },
        });

        // Workflow completes — should NOT emit CommitStagedWrite for the failed I/O.
        let actions = engine.handle(KitriEngineEvent::WorkflowComplete {
            workflow_id: wf_id,
            output: vec![99],
        });

        assert!(!actions
            .iter()
            .any(|a| matches!(a, KitriAction::CommitStagedWrite { .. })));
        assert!(actions
            .iter()
            .any(|a| matches!(a, KitriAction::Complete { .. })));
    }

    #[test]
    fn engine_successful_io_keeps_staged_write() {
        let mut engine = KitriEngine::new();
        let program = test_program();
        let input = vec![1, 2, 3];
        let wf_id = kitri_workflow_id(&program.cid, &input);

        engine.handle(KitriEngineEvent::Submit { program, input });

        // Request a side-effect I/O (Publish → staged).
        engine.handle(KitriEngineEvent::IoRequested {
            workflow_id: wf_id,
            op: KitriIoOp::Publish {
                topic: "test/out".into(),
                payload: vec![42],
            },
        });

        // I/O succeeds — staged write remains.
        engine.handle(KitriEngineEvent::IoResolved {
            workflow_id: wf_id,
            seq: 0,
            result: KitriIoResult::Published,
        });

        // Workflow completes — should emit CommitStagedWrite.
        let actions = engine.handle(KitriEngineEvent::WorkflowComplete {
            workflow_id: wf_id,
            output: vec![99],
        });

        assert!(actions
            .iter()
            .any(|a| matches!(a, KitriAction::CommitStagedWrite { .. })));
    }

    #[test]
    fn engine_retry_policy_enforced() {
        let mut engine = KitriEngine::new();
        let mut program = test_program();
        program.retry_policy = RetryPolicy {
            max_retries: 2,
            backoff: BackoffStrategy::Fixed { interval_ms: 100 },
            timeout_ms: 0,
        };
        let input = vec![1, 2, 3];
        let wf_id = kitri_workflow_id(&program.cid, &input);

        // First run → fail.
        engine.handle(KitriEngineEvent::Submit {
            program: program.clone(),
            input: input.clone(),
        });
        engine.handle(KitriEngineEvent::WorkflowFailed {
            workflow_id: wf_id,
            error: "transient".into(),
        });
        assert_eq!(engine.status(&wf_id), Some(KitriWorkflowStatus::Failed));

        // Retry 1 (attempt=1) → accepted.
        let actions = engine.handle(KitriEngineEvent::Submit {
            program: program.clone(),
            input: input.clone(),
        });
        assert!(actions
            .iter()
            .any(|a| matches!(a, KitriAction::WorkflowAccepted { .. })));

        // Fail again.
        engine.handle(KitriEngineEvent::WorkflowFailed {
            workflow_id: wf_id,
            error: "transient again".into(),
        });

        // Retry 2 (attempt=2) → accepted (max_retries=2).
        let actions = engine.handle(KitriEngineEvent::Submit {
            program: program.clone(),
            input: input.clone(),
        });
        assert!(actions
            .iter()
            .any(|a| matches!(a, KitriAction::WorkflowAccepted { .. })));

        // Fail again.
        engine.handle(KitriEngineEvent::WorkflowFailed {
            workflow_id: wf_id,
            error: "transient yet again".into(),
        });

        // Retry 3 (attempt=3 >= max_retries=2) → rejected as deduplicated.
        let actions = engine.handle(KitriEngineEvent::Submit { program, input });
        assert!(actions
            .iter()
            .any(|a| matches!(a, KitriAction::Deduplicated { .. })));
    }

    #[test]
    fn engine_retry_policy_none_rejects_immediately() {
        let mut engine = KitriEngine::new();
        let mut program = test_program();
        program.retry_policy = RetryPolicy::none();
        let input = vec![1, 2, 3];
        let wf_id = kitri_workflow_id(&program.cid, &input);

        engine.handle(KitriEngineEvent::Submit {
            program: program.clone(),
            input: input.clone(),
        });
        engine.handle(KitriEngineEvent::WorkflowFailed {
            workflow_id: wf_id,
            error: "fatal".into(),
        });

        // max_retries=0 → first retry attempt rejected.
        let actions = engine.handle(KitriEngineEvent::Submit { program, input });
        assert!(actions
            .iter()
            .any(|a| matches!(a, KitriAction::Deduplicated { .. })));
    }

    #[test]
    fn engine_output_query() {
        let mut engine = KitriEngine::new();
        let program = test_program();
        let input = vec![1, 2, 3];
        let wf_id = kitri_workflow_id(&program.cid, &input);

        engine.handle(KitriEngineEvent::Submit { program, input });

        // Output is None before completion.
        assert_eq!(engine.output(&wf_id), None);

        engine.handle(KitriEngineEvent::WorkflowComplete {
            workflow_id: wf_id,
            output: vec![42, 43],
        });

        // Output is available after completion.
        assert_eq!(engine.output(&wf_id), Some(&[42, 43][..]));
    }

    #[test]
    fn engine_evict_completed_workflow() {
        let mut engine = KitriEngine::new();
        let program = test_program();
        let input = vec![1, 2, 3];
        let wf_id = kitri_workflow_id(&program.cid, &input);

        engine.handle(KitriEngineEvent::Submit { program, input });
        engine.handle(KitriEngineEvent::WorkflowComplete {
            workflow_id: wf_id,
            output: vec![42],
        });
        assert_eq!(engine.workflow_count(), 1);

        assert!(engine.evict(&wf_id));
        assert_eq!(engine.workflow_count(), 0);
        assert_eq!(engine.status(&wf_id), None);
    }

    #[test]
    fn engine_evict_failed_workflow() {
        let mut engine = KitriEngine::new();
        let program = test_program();
        let input = vec![1, 2, 3];
        let wf_id = kitri_workflow_id(&program.cid, &input);

        engine.handle(KitriEngineEvent::Submit { program, input });
        engine.handle(KitriEngineEvent::WorkflowFailed {
            workflow_id: wf_id,
            error: "boom".into(),
        });

        assert!(engine.evict(&wf_id));
        assert_eq!(engine.workflow_count(), 0);
    }

    #[test]
    fn engine_evict_rejects_active_workflow() {
        let mut engine = KitriEngine::new();
        let program = test_program();
        let input = vec![1, 2, 3];
        let wf_id = kitri_workflow_id(&program.cid, &input);

        engine.handle(KitriEngineEvent::Submit { program, input });
        assert_eq!(engine.status(&wf_id), Some(KitriWorkflowStatus::Pending));

        // Cannot evict a Pending workflow.
        assert!(!engine.evict(&wf_id));
        assert_eq!(engine.workflow_count(), 1);
    }

    #[test]
    fn engine_evict_nonexistent_returns_false() {
        let mut engine = KitriEngine::new();
        assert!(!engine.evict(&[0xFF; 32]));
    }

    #[test]
    fn engine_event_log_accessor() {
        let mut engine = KitriEngine::new();
        let program = test_program();
        let input = vec![1, 2, 3];
        let wf_id = kitri_workflow_id(&program.cid, &input);

        assert!(engine.event_log(&wf_id).is_none());

        engine.handle(KitriEngineEvent::Submit { program, input });

        let log = engine.event_log(&wf_id).unwrap();
        assert!(log.is_empty());

        engine.handle(KitriEngineEvent::IoRequested {
            workflow_id: wf_id,
            op: KitriIoOp::Fetch { cid: [0; 32] },
        });

        let log = engine.event_log(&wf_id).unwrap();
        assert_eq!(log.len(), 1);
    }

    #[test]
    fn engine_retry_emits_backoff_delay() {
        let mut engine = KitriEngine::new();
        let mut program = test_program();
        program.retry_policy = RetryPolicy {
            max_retries: 3,
            backoff: BackoffStrategy::Exponential {
                initial_ms: 100,
                max_ms: 10_000,
            },
            timeout_ms: 0,
        };
        let input = vec![1, 2, 3];
        let wf_id = kitri_workflow_id(&program.cid, &input);

        engine.handle(KitriEngineEvent::Submit {
            program: program.clone(),
            input: input.clone(),
        });
        engine.handle(KitriEngineEvent::WorkflowFailed {
            workflow_id: wf_id,
            error: "transient".into(),
        });

        // First retry (attempt 0 at time of delay_ms call) → 100ms.
        let actions = engine.handle(KitriEngineEvent::Submit {
            program: program.clone(),
            input: input.clone(),
        });
        let delay = actions.iter().find_map(|a| match a {
            KitriAction::RetryAfter { delay_ms, .. } => Some(*delay_ms),
            _ => None,
        });
        assert_eq!(delay, Some(100));

        // Fail again.
        engine.handle(KitriEngineEvent::WorkflowFailed {
            workflow_id: wf_id,
            error: "transient".into(),
        });

        // Second retry (attempt 1 at time of delay_ms call) → 200ms.
        let actions = engine.handle(KitriEngineEvent::Submit { program, input });
        let delay = actions.iter().find_map(|a| match a {
            KitriAction::RetryAfter { delay_ms, .. } => Some(*delay_ms),
            _ => None,
        });
        assert_eq!(delay, Some(200));
    }
}
