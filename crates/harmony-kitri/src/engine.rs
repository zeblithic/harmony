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
                result,
            } => self.handle_io_resolved(workflow_id, result),
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
        }
    }

    fn handle_submit(&mut self, program: KitriProgram, input: Vec<u8>) -> Vec<KitriAction> {
        let wf_id = kitri_workflow_id(&program.cid, &input);

        if self.workflows.contains_key(&wf_id) {
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
            },
        );

        vec![KitriAction::WorkflowAccepted { workflow_id: wf_id }]
    }

    fn handle_io_requested(&mut self, workflow_id: [u8; 32], op: KitriIoOp) -> Vec<KitriAction> {
        let Some(state) = self.workflows.get_mut(&workflow_id) else {
            return vec![];
        };

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
        result: KitriIoResult,
    ) -> Vec<KitriAction> {
        let Some(state) = self.workflows.get_mut(&workflow_id) else {
            return vec![];
        };

        let seq = if state.next_seq > 0 {
            state.next_seq - 1
        } else {
            0
        };
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

        state.status = if actions.is_empty() {
            KitriWorkflowStatus::Failed
        } else {
            KitriWorkflowStatus::Compensating
        };

        actions.push(KitriAction::Failed { workflow_id, error });
        actions
    }

    fn handle_checkpoint(
        &mut self,
        workflow_id: [u8; 32],
        state_cid: [u8; 32],
    ) -> Vec<KitriAction> {
        let Some(state) = self.workflows.get_mut(&workflow_id) else {
            return vec![];
        };

        let seq = state.next_seq;
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

        state
            .compensation
            .register(crate::compensation::Compensator {
                step_id,
                rollback_op,
            });

        let seq = state.next_seq;
        state
            .event_log
            .append(crate::event::KitriEvent::CompensatorRegistered { seq, step_id });

        vec![]
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
    fn engine_status_tracking() {
        let mut engine = KitriEngine::new();
        let program = test_program();
        let input = vec![1, 2, 3];
        let wf_id = kitri_workflow_id(&program.cid, &input);

        assert_eq!(engine.status(&wf_id), None);

        engine.handle(KitriEngineEvent::Submit { program, input });
        assert_eq!(engine.status(&wf_id), Some(KitriWorkflowStatus::Pending));

        engine.handle(KitriEngineEvent::WorkflowComplete {
            workflow_id: wf_id,
            output: vec![],
        });
        assert_eq!(engine.status(&wf_id), Some(KitriWorkflowStatus::Complete));
    }
}
