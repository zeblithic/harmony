use alloc::vec::Vec;
use crate::types::{GroupOp, OpId};

/// Returns the subset of `local_ops` that the remote peer is missing, given the
/// set of op IDs the remote peer already has.
///
/// This is a placeholder stub; full implementation follows in a later task.
pub fn ops_to_send<'a>(local_ops: &'a [GroupOp], remote_has: &[OpId]) -> Vec<&'a GroupOp> {
    local_ops
        .iter()
        .filter(|op| !remote_has.contains(&op.id))
        .collect()
}
