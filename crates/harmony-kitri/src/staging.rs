// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Staged writes — buffer externally-visible side effects until audit passes.

use alloc::vec::Vec;

use crate::io::KitriIoOp;

/// A side-effect operation waiting in the staging buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StagedWrite {
    pub seq: u64,
    pub op: KitriIoOp,
}

/// Buffer for externally-visible side effects.
///
/// All Publish, Store, Seal, and Spawn operations are staged here
/// before being committed. On workflow failure, the buffer is
/// discarded atomically — no partial side effects.
#[derive(Debug, Clone, Default)]
pub struct StagingBuffer {
    writes: Vec<StagedWrite>,
}

impl StagingBuffer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn stage(&mut self, write: StagedWrite) {
        self.writes.push(write);
    }

    pub fn len(&self) -> usize {
        self.writes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.writes.is_empty()
    }

    /// Commit all staged writes. Returns the committed operations
    /// for the caller to execute. Clears the buffer.
    pub fn commit_all(&mut self) -> Vec<StagedWrite> {
        core::mem::take(&mut self.writes)
    }

    /// Discard all staged writes (workflow failed or was killed).
    pub fn discard_all(&mut self) {
        self.writes.clear();
    }

    /// View the staged writes without committing.
    pub fn pending(&self) -> &[StagedWrite] {
        &self.writes
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::io::KitriIoOp;

    #[test]
    fn staging_buffer_empty_initially() {
        let buf = StagingBuffer::new();
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
    }

    #[test]
    fn staging_buffer_stage_and_commit() {
        let mut buf = StagingBuffer::new();
        buf.stage(StagedWrite {
            seq: 0,
            op: KitriIoOp::Publish {
                topic: "test/topic".into(),
                payload: vec![1, 2, 3],
            },
        });
        assert_eq!(buf.len(), 1);

        let committed = buf.commit_all();
        assert_eq!(committed.len(), 1);
        assert!(buf.is_empty());
    }

    #[test]
    fn staging_buffer_discard() {
        let mut buf = StagingBuffer::new();
        buf.stage(StagedWrite {
            seq: 0,
            op: KitriIoOp::Store { data: vec![1] },
        });
        buf.stage(StagedWrite {
            seq: 1,
            op: KitriIoOp::Publish {
                topic: "t".into(),
                payload: vec![2],
            },
        });
        assert_eq!(buf.len(), 2);

        buf.discard_all();
        assert!(buf.is_empty());
    }
}
