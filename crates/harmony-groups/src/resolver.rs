use crate::error::ResolveError;
use crate::types::{GroupOp, GroupState};

/// Resolve a slice of `GroupOp`s into a `GroupState` by building the DAG and
/// replaying ops in causal order.
///
/// This is a placeholder stub; full implementation follows in a later task.
pub fn resolve(_ops: &[GroupOp]) -> Result<GroupState, ResolveError> {
    Err(ResolveError::EmptyDag)
}
