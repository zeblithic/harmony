// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Saga compensation — opt-in rollback for multi-step workflows.

use alloc::vec::Vec;

use crate::io::KitriIoOp;

/// A registered compensator that runs if the workflow fails after this step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Compensator {
    pub step_id: u64,
    pub rollback_op: KitriIoOp,
}

/// Log of registered compensators for a workflow.
///
/// On success, all compensators are discarded.
/// On failure, compensators execute in reverse order (LIFO)
/// to undo completed steps.
#[derive(Debug, Clone, Default)]
pub struct CompensationLog {
    compensators: Vec<Compensator>,
}

impl CompensationLog {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, compensator: Compensator) {
        self.compensators.push(compensator);
    }

    pub fn len(&self) -> usize {
        self.compensators.len()
    }

    pub fn is_empty(&self) -> bool {
        self.compensators.is_empty()
    }

    /// Drain all compensators in reverse order (LIFO) for saga rollback.
    pub fn drain_reverse(&mut self) -> Vec<Compensator> {
        let mut result = core::mem::take(&mut self.compensators);
        result.reverse();
        result
    }

    /// Discard all compensators (workflow completed successfully).
    pub fn discard_all(&mut self) {
        self.compensators.clear();
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::io::KitriIoOp;

    #[test]
    fn compensator_registration() {
        let comp = Compensator {
            step_id: 0,
            rollback_op: KitriIoOp::Query {
                topic: "accounts/credit".into(),
                payload: vec![1, 2, 3],
            },
        };
        assert_eq!(comp.step_id, 0);
    }

    #[test]
    fn compensation_log_empty_initially() {
        let log = CompensationLog::new();
        assert!(log.is_empty());
    }

    #[test]
    fn compensation_log_register_and_drain() {
        let mut log = CompensationLog::new();
        log.register(Compensator {
            step_id: 0,
            rollback_op: KitriIoOp::Query {
                topic: "accounts/credit".into(),
                payload: vec![1],
            },
        });
        log.register(Compensator {
            step_id: 1,
            rollback_op: KitriIoOp::Query {
                topic: "inventory/restock".into(),
                payload: vec![2],
            },
        });
        assert_eq!(log.len(), 2);

        // Drain returns compensators in REVERSE order (LIFO for saga rollback).
        let drained = log.drain_reverse();
        assert_eq!(drained.len(), 2);
        assert_eq!(drained[0].step_id, 1); // most recent first
        assert_eq!(drained[1].step_id, 0);
        assert!(log.is_empty());
    }

    #[test]
    fn compensation_log_discard() {
        let mut log = CompensationLog::new();
        log.register(Compensator {
            step_id: 0,
            rollback_op: KitriIoOp::Query {
                topic: "t".into(),
                payload: vec![],
            },
        });
        // On success, discard all compensators.
        log.discard_all();
        assert!(log.is_empty());
    }
}
