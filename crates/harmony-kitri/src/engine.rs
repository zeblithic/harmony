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
    /// A compensator should be executed (saga rollback).
    ExecuteCompensator {
        workflow_id: [u8; 32],
        step_id: u64,
        rollback_op: KitriIoOp,
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
    #[allow(dead_code)]
    program: KitriProgram,
    status: KitriWorkflowStatus,
    event_log: KitriEventLog,
    staging: StagingBuffer,
    compensation: CompensationLog,
    next_seq: u64,
    #[allow(dead_code)]
    output: Option<Vec<u8>>,
    /// Error message deferred until compensation completes.
    pending_error: Option<String>,
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
                state: _,
                state_cid,
            } => self.handle_checkpoint(workflow_id, state_cid),
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
                // Allow retry of failed workflows. Keep the event log so replay
                // can skip I/O operations that already completed, preventing
                // duplicate side effects.
                existing.status = KitriWorkflowStatus::Pending;
                existing.pending_error = None;
                return vec![KitriAction::WorkflowAccepted { workflow_id: wf_id }];
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

        state.status = KitriWorkflowStatus::Executing;

        state
            .event_log
            .append(crate::event::KitriEvent::IoResolved { seq, result });

        vec![KitriAction::PersistEventLog { workflow_id }]
    }

    fn handle_complete(&mut self, workflow_id: [u8; 32], output: Vec<u8>) -> Vec<KitriAction> {
        let Some(state) = self.workflows.get_mut(&workflow_id) else {
            return vec![];
        };

        // Only accept completion from active states (including Pending for zero-I/O workflows).
        if !matches!(
            state.status,
            KitriWorkflowStatus::Pending
                | KitriWorkflowStatus::Executing
                | KitriWorkflowStatus::WaitingForIo
        ) {
            return vec![];
        }

        state.status = KitriWorkflowStatus::Complete;
        state.output = Some(output.clone());
        state.compensation.discard_all();

        let mut actions: Vec<KitriAction> = state
            .staging
            .commit_all()
            .into_iter()
            .map(|w| KitriAction::CommitStagedWrite {
                workflow_id,
                write: w,
            })
            .collect();

        actions.push(KitriAction::PersistEventLog { workflow_id });
        actions.push(KitriAction::Complete {
            workflow_id,
            output,
        });

        actions
    }

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
            actions.push(KitriAction::PersistEventLog { workflow_id });
            actions.push(KitriAction::Failed { workflow_id, error });
        } else {
            // Compensators need to run first. Defer Failed until
            // CompensationComplete arrives.
            state.status = KitriWorkflowStatus::Compensating;
            state.pending_error = Some(error);
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
        vec![
            KitriAction::PersistEventLog { workflow_id },
            KitriAction::Failed { workflow_id, error },
        ]
    }

    fn handle_checkpoint(
        &mut self,
        workflow_id: [u8; 32],
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

        vec![KitriAction::PersistEventLog { workflow_id }]
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

        state
            .compensation
            .register(crate::compensation::Compensator {
                step_id,
                rollback_op,
            });

        let seq = state.next_seq;
        state.next_seq += 1;
        state
            .event_log
            .append(crate::event::KitriEvent::CompensatorRegistered { seq, step_id });

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
    use crate::retry::RetryPolicy;
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
}
