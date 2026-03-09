// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Checkpoint types — explicit durability save points for expensive computations.

use alloc::vec::Vec;

/// A serialized checkpoint of workflow state.
///
/// Created by `kitri::checkpoint()`. Stored in content-addressed storage.
/// On crash recovery, the runtime deserializes this and skips forward
/// instead of replaying from scratch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KitriCheckpoint {
    /// Which workflow this checkpoint belongs to.
    pub workflow_id: [u8; 32],
    /// The event log sequence number at which this checkpoint was taken.
    pub seq: u64,
    /// Serialized workflow state (opaque bytes — the workflow chooses format).
    pub state: Vec<u8>,
    /// CID of the serialized state in content-addressed storage.
    pub state_cid: [u8; 32],
    /// Total fuel consumed up to this checkpoint.
    pub fuel_consumed: u64,
}

impl KitriCheckpoint {
    /// Returns true if this checkpoint is more recent than `other`.
    pub fn is_newer_than(&self, other: &Self) -> bool {
        self.seq > other.seq
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_creation() {
        let cp = KitriCheckpoint {
            workflow_id: [0xAA; 32],
            seq: 5,
            state: vec![1, 2, 3],
            state_cid: [0xBB; 32],
            fuel_consumed: 42_000,
        };
        assert_eq!(cp.seq, 5);
        assert_eq!(cp.fuel_consumed, 42_000);
        assert_eq!(cp.state, vec![1, 2, 3]);
    }

    #[test]
    fn checkpoint_is_newer_than() {
        let cp1 = KitriCheckpoint {
            workflow_id: [0; 32],
            seq: 3,
            state: vec![],
            state_cid: [0; 32],
            fuel_consumed: 100,
        };
        let cp2 = KitriCheckpoint {
            workflow_id: [0; 32],
            seq: 7,
            state: vec![],
            state_cid: [0; 32],
            fuel_consumed: 200,
        };
        assert!(cp2.is_newer_than(&cp1));
        assert!(!cp1.is_newer_than(&cp2));
        assert!(!cp1.is_newer_than(&cp1));
    }
}
