use alloc::collections::VecDeque;
use alloc::vec::Vec;
use hashbrown::{HashMap, HashSet};

use crate::error::ResolveError;
use crate::types::{GroupAction, GroupOp, OpId};

/// Validated Authenticated Causal DAG of `GroupOp`s.
///
/// Invariants upheld by `Dag::build`:
/// - Exactly one genesis op (a `Create` op with no parents).
/// - Every parent reference resolves to an op within the set.
/// - No cycles (the graph is a DAG).
pub struct Dag {
    /// All ops keyed by their content-addressed ID.
    pub ops: HashMap<OpId, GroupOp>,
    /// Forward edges: op ID → list of child op IDs.
    pub children: HashMap<OpId, Vec<OpId>>,
    /// The single genesis op ID.
    pub genesis: OpId,
}

impl Dag {
    /// Build and validate a `Dag` from a slice of `GroupOp`s.
    ///
    /// # Errors
    /// - [`ResolveError::EmptyDag`] — no ops provided.
    /// - [`ResolveError::NoGenesis`] — no op is a `Create` with empty parents.
    /// - [`ResolveError::MultipleGenesis`] — more than one such op.
    /// - [`ResolveError::InvalidGenesis`] — the single genesis op is malformed.
    /// - [`ResolveError::MissingParent`] — a parent reference is not in the set.
    /// - [`ResolveError::CycleDetected`] — the graph is not acyclic.
    pub fn build(ops: &[GroupOp]) -> Result<Self, ResolveError> {
        if ops.is_empty() {
            return Err(ResolveError::EmptyDag);
        }

        // Deduplicate by ID (last writer wins for identical IDs, which are
        // content-addressed and therefore identical in content too).
        let mut op_map: HashMap<OpId, GroupOp> = HashMap::new();
        for op in ops {
            op_map.entry(op.id).or_insert_with(|| op.clone());
        }

        // Identify genesis ops: Create action with no parents.
        let mut genesis_ids: Vec<OpId> = op_map
            .iter()
            .filter(|(_, op)| op.parents.is_empty())
            .map(|(id, _)| *id)
            .collect();

        match genesis_ids.len() {
            0 => return Err(ResolveError::NoGenesis),
            1 => {}
            _ => return Err(ResolveError::MultipleGenesis),
        }

        let genesis = genesis_ids.pop().unwrap();

        // Validate that the genesis op is actually a Create.
        match &op_map[&genesis].action {
            GroupAction::Create { .. } => {}
            _ => return Err(ResolveError::InvalidGenesis),
        }

        // Verify all parent references exist.
        for op in op_map.values() {
            for &parent in &op.parents {
                if !op_map.contains_key(&parent) {
                    return Err(ResolveError::MissingParent {
                        op: op.id,
                        parent,
                    });
                }
            }
        }

        // Build forward-edge (children) map.
        let mut children: HashMap<OpId, Vec<OpId>> = HashMap::new();
        // Ensure every op has an entry (even if it has no children).
        for &id in op_map.keys() {
            children.entry(id).or_insert_with(Vec::new);
        }
        for op in op_map.values() {
            for &parent in &op.parents {
                children.entry(parent).or_insert_with(Vec::new).push(op.id);
            }
        }

        // Cycle detection via Kahn's topological sort (in-degree counting).
        let mut in_degree: HashMap<OpId, usize> = op_map.keys().map(|&id| (id, 0)).collect();
        for op in op_map.values() {
            for &parent in &op.parents {
                *in_degree.entry(op.id).or_insert(0) += 1;
                let _ = in_degree.entry(parent).or_insert(0);
            }
        }
        // Recompute properly.
        let mut in_degree: HashMap<OpId, usize> = op_map.keys().map(|&id| (id, 0)).collect();
        for op in op_map.values() {
            for _ in &op.parents {
                *in_degree.get_mut(&op.id).unwrap() += 1;
            }
        }

        let mut queue: VecDeque<OpId> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&id, _)| id)
            .collect();

        let mut visited = 0usize;
        while let Some(id) = queue.pop_front() {
            visited += 1;
            for &child in children.get(&id).into_iter().flatten() {
                let deg = in_degree.get_mut(&child).unwrap();
                *deg -= 1;
                if *deg == 0 {
                    queue.push_back(child);
                }
            }
        }

        if visited != op_map.len() {
            return Err(ResolveError::CycleDetected);
        }

        Ok(Dag { ops: op_map, children, genesis })
    }

    /// Returns the set of all transitive ancestors of `op_id` (not including
    /// `op_id` itself).
    pub fn ancestors(&self, op_id: &OpId) -> HashSet<OpId> {
        let mut result = HashSet::new();
        let mut stack: Vec<OpId> = match self.ops.get(op_id) {
            Some(op) => op.parents.clone(),
            None => return result,
        };
        while let Some(id) = stack.pop() {
            if result.insert(id) {
                if let Some(op) = self.ops.get(&id) {
                    stack.extend_from_slice(&op.parents);
                }
            }
        }
        result
    }

    /// Returns the IDs of all "head" ops — ops that have no children
    /// (i.e. the DAG tips). The result is sorted for determinism.
    pub fn head_ops(&self) -> Vec<OpId> {
        let mut heads: Vec<OpId> = self
            .children
            .iter()
            .filter(|(_, children)| children.is_empty())
            .map(|(&id, _)| id)
            .collect();
        heads.sort_unstable();
        heads
    }

    /// Returns the in-degree (number of parents) of each op.
    pub fn in_degrees(&self) -> HashMap<OpId, usize> {
        self.ops.iter().map(|(&id, op)| (id, op.parents.len())).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{GroupAction, GroupMode, GroupOp};
    use alloc::string::String;
    use alloc::vec;

    fn make_genesis(group_id: [u8; 16], author: [u8; 16], ts: u64) -> GroupOp {
        let (op, _) = GroupOp::new_unsigned(
            vec![],
            author,
            ts,
            GroupAction::Create {
                group_id,
                name: String::from("Test"),
                mode: GroupMode::InviteOnly,
            },
        );
        op
    }

    fn make_op(parents: Vec<OpId>, author: [u8; 16], ts: u64) -> GroupOp {
        let (op, _) = GroupOp::new_unsigned(parents, author, ts, GroupAction::Leave);
        op
    }

    // ── basic happy path ────────────────────────────────────────────────────

    #[test]
    fn single_genesis() {
        let g = make_genesis([0x01; 16], [0xAA; 16], 1000);
        let dag = Dag::build(&[g.clone()]).unwrap();
        assert_eq!(dag.genesis, g.id);
        assert_eq!(dag.ops.len(), 1);
    }

    #[test]
    fn linear_chain() {
        let g = make_genesis([0x02; 16], [0xAA; 16], 1);
        let op1 = make_op(vec![g.id], [0xAA; 16], 2);
        let op2 = make_op(vec![op1.id], [0xAA; 16], 3);
        let dag = Dag::build(&[g.clone(), op1.clone(), op2.clone()]).unwrap();
        assert_eq!(dag.ops.len(), 3);
        assert_eq!(dag.genesis, g.id);
        // Only op2 is a head
        assert_eq!(dag.head_ops(), vec![op2.id]);
    }

    #[test]
    fn head_ops_sorted() {
        let g = make_genesis([0x03; 16], [0xAA; 16], 1);
        // Two independent children off the genesis → both are heads
        let op_a = make_op(vec![g.id], [0xAA; 16], 2);
        let op_b = make_op(vec![g.id], [0xBB; 16], 3);
        let dag = Dag::build(&[g.clone(), op_a.clone(), op_b.clone()]).unwrap();
        let heads = dag.head_ops();
        assert_eq!(heads.len(), 2);
        // Must be sorted
        assert!(heads[0] <= heads[1]);
    }

    #[test]
    fn ancestors_transitive() {
        let g = make_genesis([0x04; 16], [0xAA; 16], 1);
        let op1 = make_op(vec![g.id], [0xAA; 16], 2);
        let op2 = make_op(vec![op1.id], [0xAA; 16], 3);
        let dag = Dag::build(&[g.clone(), op1.clone(), op2.clone()]).unwrap();

        let anc = dag.ancestors(&op2.id);
        assert!(anc.contains(&op1.id));
        assert!(anc.contains(&g.id));
        assert_eq!(anc.len(), 2);

        // Genesis has no ancestors
        assert!(dag.ancestors(&g.id).is_empty());
    }

    #[test]
    fn in_degrees() {
        let g = make_genesis([0x05; 16], [0xAA; 16], 1);
        let op1 = make_op(vec![g.id], [0xAA; 16], 2);
        let op2 = make_op(vec![g.id, op1.id], [0xAA; 16], 3);
        let dag = Dag::build(&[g.clone(), op1.clone(), op2.clone()]).unwrap();
        let degs = dag.in_degrees();
        assert_eq!(*degs.get(&g.id).unwrap(), 0);
        assert_eq!(*degs.get(&op1.id).unwrap(), 1);
        assert_eq!(*degs.get(&op2.id).unwrap(), 2);
    }

    #[test]
    fn dedup_identical_ops() {
        let g = make_genesis([0x06; 16], [0xAA; 16], 1);
        // Same op submitted twice — should deduplicate
        let dag = Dag::build(&[g.clone(), g.clone()]).unwrap();
        assert_eq!(dag.ops.len(), 1);
    }

    // ── error cases ────────────────────────────────────────────────────────

    #[test]
    fn empty_dag() {
        assert!(matches!(Dag::build(&[]), Err(ResolveError::EmptyDag)));
    }

    #[test]
    fn no_genesis() {
        // An op with a fake parent → no zero-parent op → NoGenesis
        let fake_parent: OpId = [0xFF; 32];
        let op = make_op(vec![fake_parent], [0xAA; 16], 1);
        assert!(matches!(Dag::build(&[op]), Err(ResolveError::NoGenesis)));

        // Leave op with no parents → zero-parent op exists but InvalidGenesis
        let not_genesis = make_op(vec![], [0xAA; 16], 1);
        assert!(matches!(
            Dag::build(&[not_genesis]),
            Err(ResolveError::InvalidGenesis)
        ));
    }

    #[test]
    fn multiple_genesis() {
        let g1 = make_genesis([0x07; 16], [0xAA; 16], 1);
        let g2 = make_genesis([0x08; 16], [0xBB; 16], 2);
        assert!(matches!(
            Dag::build(&[g1, g2]),
            Err(ResolveError::MultipleGenesis)
        ));
    }

    #[test]
    fn missing_parent() {
        let g = make_genesis([0x09; 16], [0xAA; 16], 1);
        let ghost: OpId = [0xDE; 32];
        let op = make_op(vec![g.id, ghost], [0xAA; 16], 2);
        let result = Dag::build(&[g, op.clone()]);
        assert!(matches!(
            result,
            Err(ResolveError::MissingParent { op: actual_op, parent: actual_parent })
                if actual_op == op.id && actual_parent == ghost
        ));
    }

    #[test]
    fn invalid_genesis_action() {
        // An op with no parents but action != Create
        let (op, _) = GroupOp::new_unsigned(vec![], [0xAA; 16], 1, GroupAction::Dissolve);
        assert!(matches!(
            Dag::build(&[op]),
            Err(ResolveError::InvalidGenesis)
        ));
    }
}
