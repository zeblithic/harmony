use alloc::collections::VecDeque;
use alloc::vec::Vec;
use hashbrown::{HashMap, HashSet};

use crate::types::{GroupOp, OpId};

/// Returns the `OpId`s from `local_ops` that the remote peer is missing.
///
/// Given the remote peer's DAG tip hashes (`remote_tips`), we walk backwards
/// through the parent graph of our local ops to discover every op the remote
/// already has (their tips plus all transitive ancestors of those tips that
/// exist in our local set). Everything in `local_ops` that is **not** reachable
/// from those tips is missing on the remote side.
///
/// # Behaviour
/// - If `remote_tips` is empty, the remote has nothing — all local op IDs are
///   returned.
/// - If a remote tip is unknown locally, its parents cannot be walked, so
///   the remote's knowledge cannot be determined from that tip alone.
///   Any overlap discovered via other known remote tips is still excluded
///   from the result. Only when *all* tips are unknown does this effectively
///   return all local ops.
pub fn ops_to_send(local_ops: &[GroupOp], remote_tips: &[OpId]) -> Vec<OpId> {
    if remote_tips.is_empty() {
        // Remote has nothing — send every local op.
        return local_ops.iter().map(|o| o.id).collect();
    }

    // Build a map from OpId → parents for fast parent lookups.
    let parent_map: hashbrown::HashMap<OpId, &[OpId]> =
        local_ops.iter().map(|o| (o.id, o.parents.as_slice())).collect();

    // BFS/DFS backwards from each remote tip to collect all ops the remote has.
    let mut remote_has: HashSet<OpId> = HashSet::new();
    let mut stack: Vec<OpId> = Vec::new();

    for &tip in remote_tips {
        if remote_has.insert(tip) {
            stack.push(tip);
        }
    }

    while let Some(id) = stack.pop() {
        if let Some(&parents) = parent_map.get(&id) {
            for &parent in parents {
                if remote_has.insert(parent) {
                    stack.push(parent);
                }
            }
        }
        // If the id is not in our local set, we cannot walk its parents but
        // the id itself is already recorded in remote_has.
    }

    // Collect missing ops and topologically sort so parents precede children.
    let missing: Vec<&GroupOp> = local_ops
        .iter()
        .filter(|o| !remote_has.contains(&o.id))
        .collect();

    if missing.is_empty() {
        return Vec::new();
    }

    let missing_set: HashSet<OpId> = missing.iter().map(|o| o.id).collect();
    let mut in_degree: HashMap<OpId, usize> = HashMap::new();
    let mut children: HashMap<OpId, Vec<OpId>> = HashMap::new();
    for &op in &missing {
        let deg = op.parents.iter().filter(|p| missing_set.contains(*p)).count();
        in_degree.insert(op.id, deg);
        for &p in &op.parents {
            if missing_set.contains(&p) {
                children.entry(p).or_default().push(op.id);
            }
        }
    }

    let mut queue: VecDeque<OpId> = in_degree
        .iter()
        .filter(|(_, &d)| d == 0)
        .map(|(&id, _)| id)
        .collect();
    let mut result = Vec::with_capacity(missing.len());
    while let Some(id) = queue.pop_front() {
        result.push(id);
        if let Some(kids) = children.get(&id) {
            for &kid in kids {
                let d = in_degree.get_mut(&kid).unwrap();
                *d -= 1;
                if *d == 0 {
                    queue.push_back(kid);
                }
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{GroupAction, GroupMode, GroupOp};
    use alloc::string::String;
    use alloc::vec;

    const FOUNDER: [u8; 16] = [0x01; 16];
    const ALICE: [u8; 16] = [0x02; 16];
    const GROUP_ID: [u8; 16] = [0xAA; 16];

    fn make_genesis() -> GroupOp {
        let (op, _) = GroupOp::new_unsigned(
            vec![],
            FOUNDER,
            1000,
            GroupAction::Create {
                group_id: GROUP_ID,
                name: String::from("TestGroup"),
                mode: GroupMode::InviteOnly,
            },
        );
        op
    }

    fn make_op(parents: Vec<OpId>, author: [u8; 16], ts: u64, action: GroupAction) -> GroupOp {
        let (op, _) = GroupOp::new_unsigned(parents, author, ts, action);
        op
    }

    // ── empty remote tips → send all ──────────────────────────────────────

    #[test]
    fn empty_remote_tips_sends_all() {
        let g = make_genesis();
        let invite = make_op(
            vec![g.id],
            FOUNDER,
            1001,
            GroupAction::Invite { invitee: ALICE },
        );

        let local = vec![g.clone(), invite.clone()];
        let to_send = ops_to_send(&local, &[]);

        let mut ids: Vec<OpId> = to_send;
        ids.sort_unstable();
        let mut expected = vec![g.id, invite.id];
        expected.sort_unstable();
        assert_eq!(ids, expected);
    }

    // ── remote has same tip → send nothing ───────────────────────────────

    #[test]
    fn same_tip_sends_nothing() {
        let g = make_genesis();
        let invite = make_op(
            vec![g.id],
            FOUNDER,
            1001,
            GroupAction::Invite { invitee: ALICE },
        );

        let local = vec![g.clone(), invite.clone()];
        // Remote tip is the invite (the head). Because it and all its ancestors
        // are reachable, nothing should be sent.
        let to_send = ops_to_send(&local, &[invite.id]);

        assert!(to_send.is_empty(), "expected nothing to send, got: {:?}", to_send);
    }

    // ── remote behind by one op → sends the missing op ───────────────────

    #[test]
    fn remote_behind_sends_missing() {
        let g = make_genesis();
        let invite = make_op(
            vec![g.id],
            FOUNDER,
            1001,
            GroupAction::Invite { invitee: ALICE },
        );
        let accept = make_op(
            vec![invite.id],
            ALICE,
            1002,
            GroupAction::Accept { invite_op: invite.id },
        );

        let local = vec![g.clone(), invite.clone(), accept.clone()];
        // Remote knows up to `invite` — it's missing `accept`.
        let to_send = ops_to_send(&local, &[invite.id]);

        assert_eq!(to_send, vec![accept.id]);
    }

    // ── unknown remote tip → sends all local ops ─────────────────────────

    #[test]
    fn unknown_remote_tip_sends_all() {
        let g = make_genesis();
        let invite = make_op(
            vec![g.id],
            FOUNDER,
            1001,
            GroupAction::Invite { invitee: ALICE },
        );

        let local = vec![g.clone(), invite.clone()];
        // Remote claims a tip that doesn't exist locally.
        let unknown_tip: OpId = [0xFF; 32];
        let to_send = ops_to_send(&local, &[unknown_tip]);

        // The unknown tip is in remote_has but none of our local ops are
        // ancestors of it (we can't walk its parents), so all local ops
        // are returned.
        let mut ids = to_send;
        ids.sort_unstable();
        let mut expected = vec![g.id, invite.id];
        expected.sort_unstable();
        assert_eq!(ids, expected);
    }

    // ── remote behind on one branch of a fork ────────────────────────────

    #[test]
    fn remote_missing_one_branch() {
        let g = make_genesis();
        // Two concurrent ops off genesis (a fork).
        let invite_a = make_op(
            vec![g.id],
            FOUNDER,
            1001,
            GroupAction::Invite { invitee: ALICE },
        );
        let invite_b = make_op(
            vec![g.id],
            FOUNDER,
            1002,
            GroupAction::Invite { invitee: [0x05; 16] },
        );

        let local = vec![g.clone(), invite_a.clone(), invite_b.clone()];
        // Remote knows only invite_a (and transitively, genesis).
        let to_send = ops_to_send(&local, &[invite_a.id]);

        // Only invite_b should be sent.
        assert_eq!(to_send, vec![invite_b.id]);
    }
}
