// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Event log — the durable record of a Kitri workflow's I/O history.

use alloc::vec::Vec;

use crate::io::{KitriIoOp, KitriIoResult};

/// A single event in a Kitri workflow's history.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KitriEvent {
    /// An I/O operation was requested by the workflow.
    IoRequested { seq: u64, op: KitriIoOp },
    /// An I/O operation completed (success or failure).
    IoResolved { seq: u64, result: KitriIoResult },
    /// An explicit checkpoint was saved.
    CheckpointSaved { seq: u64, state_cid: [u8; 32] },
    /// A compensator was registered for saga rollback.
    CompensatorRegistered { seq: u64, step_id: u64 },
}

/// The full event log for a workflow instance.
///
/// Content-addressable: the log can be stored and replicated via CID.
/// On crash recovery, the runtime replays this log to skip completed I/O.
#[derive(Debug, Clone)]
pub struct KitriEventLog {
    workflow_id: [u8; 32],
    events: Vec<KitriEvent>,
}

impl KitriEventLog {
    pub fn new(workflow_id: [u8; 32]) -> Self {
        Self {
            workflow_id,
            events: Vec::new(),
        }
    }

    pub fn workflow_id(&self) -> &[u8; 32] {
        &self.workflow_id
    }

    pub fn append(&mut self, event: KitriEvent) {
        self.events.push(event);
    }

    pub fn events(&self) -> &[KitriEvent] {
        &self.events
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Look up the cached result for a given sequence number.
    /// Used during replay to skip completed I/O.
    pub fn cached_result(&self, seq: u64) -> Option<&KitriIoResult> {
        self.events.iter().find_map(|e| match e {
            KitriEvent::IoResolved { seq: s, result } if *s == seq => Some(result),
            _ => None,
        })
    }

    /// Find the most recent checkpoint, if any.
    pub fn last_checkpoint(&self) -> Option<(u64, [u8; 32])> {
        self.events.iter().rev().find_map(|e| match e {
            KitriEvent::CheckpointSaved { seq, state_cid } => Some((*seq, *state_cid)),
            _ => None,
        })
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::io::{KitriIoOp, KitriIoResult};

    #[test]
    fn kitri_event_io_roundtrip() {
        let op = KitriIoOp::Fetch { cid: [0xAA; 32] };
        let result = KitriIoResult::Fetched {
            data: vec![1, 2, 3],
        };

        let requested = KitriEvent::IoRequested {
            seq: 0,
            op: op.clone(),
        };
        let resolved = KitriEvent::IoResolved {
            seq: 0,
            result: result.clone(),
        };

        assert!(matches!(requested, KitriEvent::IoRequested { seq: 0, .. }));
        assert!(matches!(resolved, KitriEvent::IoResolved { seq: 0, .. }));
    }

    #[test]
    fn kitri_event_checkpoint() {
        let event = KitriEvent::CheckpointSaved {
            seq: 5,
            state_cid: [0xBB; 32],
        };
        assert!(matches!(event, KitriEvent::CheckpointSaved { seq: 5, .. }));
    }

    #[test]
    fn event_log_append_and_len() {
        let mut log = KitriEventLog::new([0xCC; 32]);
        assert_eq!(log.len(), 0);
        assert!(log.is_empty());

        log.append(KitriEvent::IoRequested {
            seq: 0,
            op: KitriIoOp::Fetch { cid: [0; 32] },
        });
        assert_eq!(log.len(), 1);
        assert!(!log.is_empty());
    }

    #[test]
    fn event_log_workflow_id() {
        let id = [0xDD; 32];
        let log = KitriEventLog::new(id);
        assert_eq!(log.workflow_id(), &id);
    }

    #[test]
    fn event_log_replay_cache() {
        let mut log = KitriEventLog::new([0; 32]);
        let op = KitriIoOp::Fetch { cid: [0xAA; 32] };
        let result = KitriIoResult::Fetched { data: vec![42] };

        log.append(KitriEvent::IoRequested { seq: 0, op });
        log.append(KitriEvent::IoResolved {
            seq: 0,
            result: result.clone(),
        });

        // Replay should find the cached result for seq 0.
        assert_eq!(log.cached_result(0), Some(&result));
        assert_eq!(log.cached_result(1), None);
    }

    #[test]
    fn event_log_last_checkpoint() {
        let mut log = KitriEventLog::new([0; 32]);
        assert_eq!(log.last_checkpoint(), None);

        log.append(KitriEvent::CheckpointSaved {
            seq: 3,
            state_cid: [0xAA; 32],
        });
        log.append(KitriEvent::IoRequested {
            seq: 4,
            op: KitriIoOp::Fetch { cid: [0; 32] },
        });
        log.append(KitriEvent::CheckpointSaved {
            seq: 7,
            state_cid: [0xBB; 32],
        });

        let (seq, cid) = log.last_checkpoint().unwrap();
        assert_eq!(seq, 7);
        assert_eq!(cid, [0xBB; 32]);
    }
}
