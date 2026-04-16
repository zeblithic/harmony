use alloc::collections::BinaryHeap;
use alloc::vec::Vec;
use core::cmp::Ordering;
use hashbrown::{HashMap, HashSet};

use crate::dag::Dag;
use crate::error::ResolveError;
use crate::types::{
    GroupAction, GroupMode, GroupOp, GroupState, MemberAddr, MemberEntry, OpId, Role,
};

/// Wrapper for topological sort priority queue.
/// Sorts by: authority level ascending (Founder=0 first), then lexicographic OpId.
#[derive(Eq, PartialEq)]
struct ReadyOp {
    authority: u8,
    id: OpId,
}

impl Ord for ReadyOp {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max-heap, so we reverse:
        // - Lower authority number = higher priority (should come first)
        // - For equal authority, smaller OpId = higher priority
        other
            .authority
            .cmp(&self.authority)
            .then_with(|| other.id.cmp(&self.id))
    }
}

impl PartialOrd for ReadyOp {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Resolve a slice of `GroupOp`s into a `GroupState` by building the DAG and
/// replaying ops in causal (topological) order with authorization checks.
///
/// This is a pure function: ops in, state out. Invalid ops are silently skipped.
///
/// # Security: caller must verify signatures
///
/// This crate is sans-I/O and **does not verify signatures**. Authorization
/// checks assume `op.author` is authentic. Callers **must** verify that each
/// op was signed by the private key matching `op.author` before passing ops
/// to this function. Without signature verification, any peer can forge ops
/// claiming any author identity and bypass all role-based access control.
///
/// The recommended integration pattern:
/// 1. At the network boundary, validate `op.author == authenticated_sender`.
/// 2. If the protocol supports detached signatures, verify `op.signature`
///    against the author's public key before calling `resolve()`.
pub fn resolve(ops: &[GroupOp]) -> Result<GroupState, ResolveError> {
    let dag = Dag::build(ops)?;
    let mut in_deg = dag.in_degrees();
    let mut state = GroupState::default();
    let mut authorized_ops: HashSet<OpId> = HashSet::new();
    let mut visited = 0usize;

    // Seed the ready set with zero-in-degree ops (just genesis initially).
    let mut ready = BinaryHeap::new();
    for (&id, &deg) in &in_deg {
        if deg == 0 {
            let authority = author_authority(&state, &dag.ops[&id]);
            ready.push(ReadyOp { authority, id });
        }
    }

    while !ready.is_empty() {
        // Drain all ops at the highest authority level (lowest number) into a batch.
        let top_authority = ready.peek().unwrap().authority;
        let mut batch: Vec<OpId> = Vec::new();
        while let Some(top) = ready.peek() {
            if top.authority == top_authority {
                batch.push(ready.pop().unwrap().id);
            } else {
                break;
            }
        }

        // For equal-authority concurrent ops: validate ALL against current state
        // before applying any. This enables mutual kicks at equal rank.
        let mut authorized: Vec<OpId> = Vec::new();
        for &op_id in &batch {
            let op = &dag.ops[&op_id];
            if is_authorized(&state, op, &dag, &authorized_ops) {
                authorized.push(op_id);
            }
        }

        // Sort authorized ops by OpId for deterministic application order.
        authorized.sort_unstable();

        // Apply authorized ops and record them.
        for &op_id in &authorized {
            let op = &dag.ops[&op_id];
            apply_op(&mut state, op);
            authorized_ops.insert(op_id);
        }

        // Update in-degrees and enqueue newly ready ops. Authority is
        // intentionally snapshotted at enqueue time so that higher-authority
        // ops (e.g. Founder promotions) take effect before lower-authority
        // concurrent ops are batched and validated.
        for &op_id in &batch {
            visited += 1;
            if let Some(children) = dag.children.get(&op_id) {
                for &child_id in children {
                    let deg = in_deg.get_mut(&child_id).unwrap();
                    *deg -= 1;
                    if *deg == 0 {
                        let authority = author_authority(&state, &dag.ops[&child_id]);
                        ready.push(ReadyOp {
                            authority,
                            id: child_id,
                        });
                    }
                }
            }
        }
    }

    // Cycle detection: all ops must have been visited.
    if visited != dag.ops.len() {
        return Err(ResolveError::CycleDetected);
    }

    Ok(state)
}

/// Returns the authority level of the op's author in the current state.
/// Founder=0, Officer=1, Member=2, non-member=3 (lowest priority).
fn author_authority(state: &GroupState, op: &GroupOp) -> u8 {
    // Genesis Create is always highest priority.
    if op.parents.is_empty() {
        if let GroupAction::Create { .. } = &op.action {
            return 0;
        }
    }
    match state.role_of(&op.author) {
        Some(role) => role.power_level(),
        None => 3, // non-member
    }
}

/// Check if an op is authorized against the current group state.
fn is_authorized(state: &GroupState, op: &GroupOp, dag: &Dag, authorized_ops: &HashSet<OpId>) -> bool {
    // If the group is dissolved, reject everything.
    if state.dissolved {
        return false;
    }

    match &op.action {
        GroupAction::Create { .. } => {
            // Genesis: state must be empty (no members yet).
            state.members.is_empty()
        }

        GroupAction::Invite { invitee } => {
            // Author must be Founder or Officer.
            match state.role_of(&op.author) {
                Some(Role::Founder) | Some(Role::Officer) => {}
                _ => return false,
            }
            // Target must not already be a member.
            !state.is_member(invitee)
        }

        GroupAction::Join => {
            // Group must be Open mode.
            if state.mode != GroupMode::Open {
                return false;
            }
            // Author must not already be a member.
            !state.is_member(&op.author)
        }

        GroupAction::Accept { invite_op } => {
            if state.is_member(&op.author) {
                return false;
            }
            // The invite must have passed authorization during replay.
            if !authorized_ops.contains(invite_op) {
                return false;
            }
            // The invite must be a causal ancestor of this Accept.
            if !dag.ancestors(&op.id).contains(invite_op) {
                return false;
            }
            match dag.ops.get(invite_op) {
                Some(inv) => match &inv.action {
                    GroupAction::Invite { invitee } => *invitee == op.author,
                    _ => false,
                },
                None => false,
            }
        }

        GroupAction::Leave => {
            // Author must be a member.
            state.is_member(&op.author)
        }

        GroupAction::Kick { target } => {
            match state.role_of(&op.author) {
                Some(Role::Founder) => {
                    // Founder can kick anyone except self.
                    *target != op.author && state.is_member(target)
                }
                Some(Role::Officer) => {
                    // Officer can only kick Members.
                    match state.role_of(target) {
                        Some(Role::Member) => true,
                        _ => false,
                    }
                }
                _ => false,
            }
        }

        GroupAction::Promote { target } => {
            // Founder-only.
            if state.role_of(&op.author) != Some(Role::Founder) {
                return false;
            }
            // Target must be a plain Member (not already Officer or Founder).
            matches!(state.role_of(target), Some(Role::Member))
        }

        GroupAction::Demote { target } => {
            // Founder-only.
            if state.role_of(&op.author) != Some(Role::Founder) {
                return false;
            }
            // Target must be a member whose role can be lowered.
            // Can only demote Officer → Member.
            match state.role_of(target) {
                Some(Role::Officer) => true,
                _ => false,
            }
        }

        GroupAction::Dissolve => {
            // Founder-only.
            state.role_of(&op.author) == Some(Role::Founder)
        }

        GroupAction::UpdateInfo { .. } => {
            // Founder-only.
            state.role_of(&op.author) == Some(Role::Founder)
        }
    }
}

/// Apply an authorized op to the state, mutating it.
fn apply_op(state: &mut GroupState, op: &GroupOp) {
    match &op.action {
        GroupAction::Create {
            group_id,
            name,
            mode,
        } => {
            state.group_id = *group_id;
            state.name = name.clone();
            state.mode = *mode;
            state.members.insert(
                op.author,
                MemberEntry {
                    role: Role::Founder,
                    joined_at: op.timestamp,
                },
            );
        }

        GroupAction::Invite { .. } => {
            // Invite itself doesn't change state; it just creates an op
            // that Accept can reference.
        }

        GroupAction::Join => {
            state.members.entry(op.author).or_insert(MemberEntry {
                role: Role::Member,
                joined_at: op.timestamp,
            });
        }

        GroupAction::Accept { .. } => {
            state.members.entry(op.author).or_insert(MemberEntry {
                role: Role::Member,
                joined_at: op.timestamp,
            });
        }

        GroupAction::Leave => {
            let was_founder = state.role_of(&op.author) == Some(Role::Founder);
            state.members.remove(&op.author);
            if was_founder {
                handle_founder_removed(state);
            }
        }

        GroupAction::Kick { target } => {
            // Kick authorization normally prevents a Founder from being
            // kicked, but concurrent Leave+Kick in the same batch (both by
            // the Founder) can promote `target` to Founder via successor
            // selection before this Kick applies. Re-run succession if the
            // removed member was currently the Founder.
            let was_founder = state.role_of(target) == Some(Role::Founder);
            state.members.remove(target);
            if was_founder {
                handle_founder_removed(state);
            }
        }

        GroupAction::Promote { target } => {
            // Pre-batch authorization required target to be a Member, but a
            // concurrent Leave+succession in the same batch may have promoted
            // the target to Founder before this op applies. Only raise a
            // Member — never overwrite a Founder's role.
            if let Some(entry) = state.members.get_mut(target) {
                if entry.role == Role::Member {
                    entry.role = Role::Officer;
                }
            }
        }

        GroupAction::Demote { target } => {
            // Similar guard: pre-batch auth saw target as Officer, but a
            // concurrent Leave+succession may have promoted them to Founder.
            // Only demote an Officer — never a Founder.
            if let Some(entry) = state.members.get_mut(target) {
                if entry.role == Role::Officer {
                    entry.role = Role::Member;
                }
            }
        }

        GroupAction::Dissolve => {
            state.dissolved = true;
        }

        GroupAction::UpdateInfo { name, mode } => {
            if let Some(n) = name {
                state.name = n.clone();
            }
            if let Some(m) = mode {
                state.mode = *m;
            }
        }
    }
}

/// Promote a successor after the current Founder has been removed, or
/// dissolve the group if no members remain. Called by both `Leave` and
/// `Kick` apply paths when the removed member held the Founder role.
fn handle_founder_removed(state: &mut GroupState) {
    if state.members.is_empty() {
        state.dissolved = true;
        return;
    }
    if let Some(addr) = find_successor(&state.members) {
        if let Some(entry) = state.members.get_mut(&addr) {
            entry.role = Role::Founder;
        }
    }
}

/// Find the best successor when the Founder leaves.
/// Priority: longest-tenured Officer, then longest-tenured Member.
/// Ties broken by smallest MemberAddr for determinism.
fn find_successor(members: &HashMap<MemberAddr, MemberEntry>) -> Option<MemberAddr> {
    // Try Officers first.
    let mut best_officer: Option<(MemberAddr, u64)> = None;
    let mut best_member: Option<(MemberAddr, u64)> = None;

    for (&addr, entry) in members {
        match entry.role {
            Role::Officer => match best_officer {
                None => best_officer = Some((addr, entry.joined_at)),
                Some((prev_addr, prev_ts)) => {
                    if entry.joined_at < prev_ts
                        || (entry.joined_at == prev_ts && addr < prev_addr)
                    {
                        best_officer = Some((addr, entry.joined_at));
                    }
                }
            },
            Role::Member => match best_member {
                None => best_member = Some((addr, entry.joined_at)),
                Some((prev_addr, prev_ts)) => {
                    if entry.joined_at < prev_ts
                        || (entry.joined_at == prev_ts && addr < prev_addr)
                    {
                        best_member = Some((addr, entry.joined_at));
                    }
                }
            },
            Role::Founder => {
                // There shouldn't be another Founder, but if somehow there is, skip.
            }
        }
    }

    best_officer
        .or(best_member)
        .map(|(addr, _)| addr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{GroupAction, GroupMode, GroupOp};
    use alloc::string::String;
    use alloc::vec;
    use alloc::vec::Vec;

    // ── helpers ─────────────────────────────────────────────────────────────

    const FOUNDER: MemberAddr = [0x01; 16];
    const ALICE: MemberAddr = [0x02; 16];
    const BOB: MemberAddr = [0x03; 16];
    const CAROL: MemberAddr = [0x04; 16];
    const GROUP_ID: [u8; 16] = [0xAA; 16];

    fn genesis(author: MemberAddr, mode: GroupMode) -> GroupOp {
        let (op, _) = GroupOp::new_unsigned(
            vec![],
            author,
            1000,
            GroupAction::Create {
                group_id: GROUP_ID,
                name: String::from("TestGroup"),
                mode,
            },
        );
        op
    }

    fn op(
        parents: Vec<OpId>,
        author: MemberAddr,
        ts: u64,
        action: GroupAction,
    ) -> GroupOp {
        let (o, _) = GroupOp::new_unsigned(parents, author, ts, action);
        o
    }

    // ── genesis ────────────────────────────────────────────────────────────

    #[test]
    fn genesis_only() {
        let g = genesis(FOUNDER, GroupMode::InviteOnly);
        let state = resolve(&[g]).unwrap();
        assert_eq!(state.group_id, GROUP_ID);
        assert_eq!(state.name, "TestGroup");
        assert_eq!(state.mode, GroupMode::InviteOnly);
        assert!(!state.dissolved);
        assert_eq!(state.members.len(), 1);
        assert_eq!(state.role_of(&FOUNDER), Some(Role::Founder));
    }

    // ── invite → accept flow ───────────────────────────────────────────────

    #[test]
    fn invite_accept_flow() {
        let g = genesis(FOUNDER, GroupMode::InviteOnly);
        let invite = op(
            vec![g.id],
            FOUNDER,
            1001,
            GroupAction::Invite { invitee: ALICE },
        );
        let accept = op(
            vec![invite.id],
            ALICE,
            1002,
            GroupAction::Accept {
                invite_op: invite.id,
            },
        );
        let state = resolve(&[g, invite, accept]).unwrap();
        assert!(state.is_member(&FOUNDER));
        assert!(state.is_member(&ALICE));
        assert_eq!(state.role_of(&ALICE), Some(Role::Member));
        assert_eq!(state.members.len(), 2);
    }

    // ── open group join ────────────────────────────────────────────────────

    #[test]
    fn open_group_join() {
        let g = genesis(FOUNDER, GroupMode::Open);
        let join = op(vec![g.id], ALICE, 1001, GroupAction::Join);
        let state = resolve(&[g, join]).unwrap();
        assert!(state.is_member(&ALICE));
        assert_eq!(state.role_of(&ALICE), Some(Role::Member));
    }

    // ── join rejected on invite-only ───────────────────────────────────────

    #[test]
    fn join_rejected_on_invite_only() {
        let g = genesis(FOUNDER, GroupMode::InviteOnly);
        let join = op(vec![g.id], ALICE, 1001, GroupAction::Join);
        let state = resolve(&[g, join]).unwrap();
        // Alice should NOT be a member.
        assert!(!state.is_member(&ALICE));
        assert_eq!(state.members.len(), 1);
    }

    // ── kick (founder kicks member) ────────────────────────────────────────

    #[test]
    fn founder_kicks_member() {
        let g = genesis(FOUNDER, GroupMode::Open);
        let join = op(vec![g.id], ALICE, 1001, GroupAction::Join);
        let kick = op(
            vec![join.id],
            FOUNDER,
            1002,
            GroupAction::Kick { target: ALICE },
        );
        let state = resolve(&[g, join, kick]).unwrap();
        assert!(!state.is_member(&ALICE));
        assert_eq!(state.members.len(), 1);
    }

    // ── member cannot kick ─────────────────────────────────────────────────

    #[test]
    fn member_cannot_kick() {
        let g = genesis(FOUNDER, GroupMode::Open);
        let join_a = op(vec![g.id], ALICE, 1001, GroupAction::Join);
        let join_b = op(vec![g.id], BOB, 1002, GroupAction::Join);
        // Alice (Member) tries to kick Bob — should be rejected.
        let kick = op(
            vec![join_a.id, join_b.id],
            ALICE,
            1003,
            GroupAction::Kick { target: BOB },
        );
        let state = resolve(&[g, join_a, join_b, kick]).unwrap();
        // Bob should still be a member.
        assert!(state.is_member(&BOB));
        assert!(state.is_member(&ALICE));
    }

    // ── promote and demote ─────────────────────────────────────────────────

    #[test]
    fn promote_and_demote() {
        let g = genesis(FOUNDER, GroupMode::Open);
        let join = op(vec![g.id], ALICE, 1001, GroupAction::Join);
        let promote = op(
            vec![join.id],
            FOUNDER,
            1002,
            GroupAction::Promote { target: ALICE },
        );
        let state_promoted = resolve(&[g.clone(), join.clone(), promote.clone()]).unwrap();
        assert_eq!(state_promoted.role_of(&ALICE), Some(Role::Officer));

        let demote = op(
            vec![promote.id],
            FOUNDER,
            1003,
            GroupAction::Demote { target: ALICE },
        );
        let state_demoted = resolve(&[g, join, promote, demote]).unwrap();
        assert_eq!(state_demoted.role_of(&ALICE), Some(Role::Member));
    }

    // ── officer can invite and kick members but not officers ───────────────

    #[test]
    fn officer_can_invite_and_kick_members_not_officers() {
        let g = genesis(FOUNDER, GroupMode::InviteOnly);
        let invite_alice = op(
            vec![g.id],
            FOUNDER,
            1001,
            GroupAction::Invite { invitee: ALICE },
        );
        let accept_alice = op(
            vec![invite_alice.id],
            ALICE,
            1002,
            GroupAction::Accept {
                invite_op: invite_alice.id,
            },
        );
        let promote_alice = op(
            vec![accept_alice.id],
            FOUNDER,
            1003,
            GroupAction::Promote { target: ALICE },
        );

        // Alice (Officer) invites Bob.
        let invite_bob = op(
            vec![promote_alice.id],
            ALICE,
            1004,
            GroupAction::Invite { invitee: BOB },
        );
        let accept_bob = op(
            vec![invite_bob.id],
            BOB,
            1005,
            GroupAction::Accept {
                invite_op: invite_bob.id,
            },
        );

        // Alice (Officer) also invites Carol.
        let invite_carol = op(
            vec![accept_bob.id],
            ALICE,
            1006,
            GroupAction::Invite { invitee: CAROL },
        );
        let accept_carol = op(
            vec![invite_carol.id],
            CAROL,
            1007,
            GroupAction::Accept {
                invite_op: invite_carol.id,
            },
        );

        // Promote Bob to Officer too.
        let promote_bob = op(
            vec![accept_carol.id],
            FOUNDER,
            1008,
            GroupAction::Promote { target: BOB },
        );

        // Alice (Officer) kicks Carol (Member) — should succeed.
        let kick_carol = op(
            vec![promote_bob.id],
            ALICE,
            1009,
            GroupAction::Kick { target: CAROL },
        );

        // Alice (Officer) tries to kick Bob (Officer) — should fail.
        let kick_bob = op(
            vec![kick_carol.id],
            ALICE,
            1010,
            GroupAction::Kick { target: BOB },
        );

        let state = resolve(&[
            g,
            invite_alice,
            accept_alice,
            promote_alice,
            invite_bob,
            accept_bob,
            invite_carol,
            accept_carol,
            promote_bob,
            kick_carol,
            kick_bob,
        ])
        .unwrap();

        // Carol was kicked by officer Alice.
        assert!(!state.is_member(&CAROL));
        // Bob was NOT kicked (officer can't kick officer).
        assert!(state.is_member(&BOB));
        assert_eq!(state.role_of(&BOB), Some(Role::Officer));
    }

    // ── dissolve ───────────────────────────────────────────────────────────

    #[test]
    fn dissolve() {
        let g = genesis(FOUNDER, GroupMode::InviteOnly);
        let dissolve = op(vec![g.id], FOUNDER, 1001, GroupAction::Dissolve);
        let state = resolve(&[g, dissolve]).unwrap();
        assert!(state.dissolved);
    }

    // ── leave with auto-promote ────────────────────────────────────────────

    #[test]
    fn leave_promotes_officer() {
        let g = genesis(FOUNDER, GroupMode::Open);
        let join_a = op(vec![g.id], ALICE, 1001, GroupAction::Join);
        let join_b = op(vec![g.id], BOB, 1002, GroupAction::Join);
        let promote = op(
            vec![join_a.id, join_b.id],
            FOUNDER,
            1003,
            GroupAction::Promote { target: ALICE },
        );
        let leave = op(vec![promote.id], FOUNDER, 1004, GroupAction::Leave);
        let state = resolve(&[g, join_a, join_b, promote, leave]).unwrap();
        // Alice (Officer) should become Founder.
        assert_eq!(state.role_of(&ALICE), Some(Role::Founder));
        assert!(!state.is_member(&FOUNDER));
    }

    #[test]
    fn leave_promotes_member_when_no_officers() {
        let g = genesis(FOUNDER, GroupMode::Open);
        let join_a = op(vec![g.id], ALICE, 1001, GroupAction::Join);
        let leave = op(vec![join_a.id], FOUNDER, 1002, GroupAction::Leave);
        let state = resolve(&[g, join_a, leave]).unwrap();
        // Alice (Member) should become Founder since there are no Officers.
        assert_eq!(state.role_of(&ALICE), Some(Role::Founder));
        assert!(!state.is_member(&FOUNDER));
        assert!(!state.dissolved);
    }

    #[test]
    fn last_member_leaves_dissolves() {
        let g = genesis(FOUNDER, GroupMode::InviteOnly);
        let leave = op(vec![g.id], FOUNDER, 1001, GroupAction::Leave);
        let state = resolve(&[g, leave]).unwrap();
        assert!(state.dissolved);
        assert!(state.members.is_empty());
    }

    // ── update info ────────────────────────────────────────────────────────

    #[test]
    fn update_info() {
        let g = genesis(FOUNDER, GroupMode::InviteOnly);
        let update = op(
            vec![g.id],
            FOUNDER,
            1001,
            GroupAction::UpdateInfo {
                name: Some(String::from("NewName")),
                mode: Some(GroupMode::Open),
            },
        );
        let state = resolve(&[g, update]).unwrap();
        assert_eq!(state.name, "NewName");
        assert_eq!(state.mode, GroupMode::Open);
    }

    // ── ops after dissolve rejected ────────────────────────────────────────

    #[test]
    fn ops_after_dissolve_rejected() {
        let g = genesis(FOUNDER, GroupMode::Open);
        let dissolve = op(vec![g.id], FOUNDER, 1001, GroupAction::Dissolve);
        let join = op(vec![dissolve.id], ALICE, 1002, GroupAction::Join);
        let state = resolve(&[g, dissolve, join]).unwrap();
        assert!(state.dissolved);
        assert!(!state.is_member(&ALICE));
    }

    // ── non-member promote rejected ────────────────────────────────────────

    #[test]
    fn non_member_promote_rejected() {
        let g = genesis(FOUNDER, GroupMode::InviteOnly);
        // Founder tries to promote Alice who is not a member.
        let promote = op(
            vec![g.id],
            FOUNDER,
            1001,
            GroupAction::Promote { target: ALICE },
        );
        let state = resolve(&[g, promote]).unwrap();
        // Alice should not be a member.
        assert!(!state.is_member(&ALICE));
        assert_eq!(state.members.len(), 1);
    }

    // ── duplicate accept is no-op ──────────────────────────────────────────

    #[test]
    fn duplicate_accept_is_noop() {
        let g = genesis(FOUNDER, GroupMode::InviteOnly);
        let invite = op(
            vec![g.id],
            FOUNDER,
            1001,
            GroupAction::Invite { invitee: ALICE },
        );
        let accept1 = op(
            vec![invite.id],
            ALICE,
            1002,
            GroupAction::Accept {
                invite_op: invite.id,
            },
        );
        // Second accept references the same invite, but Alice is already a member.
        let accept2 = op(
            vec![accept1.id],
            ALICE,
            1003,
            GroupAction::Accept {
                invite_op: invite.id,
            },
        );
        let state = resolve(&[g, invite, accept1, accept2]).unwrap();
        assert!(state.is_member(&ALICE));
        assert_eq!(state.members.len(), 2);
        // joined_at should be from the first accept, not the second.
        assert_eq!(state.members[&ALICE].joined_at, 1002);
    }

    // ── founder cannot kick self ───────────────────────────────────────────

    #[test]
    fn founder_cannot_kick_self() {
        let g = genesis(FOUNDER, GroupMode::InviteOnly);
        let kick_self = op(
            vec![g.id],
            FOUNDER,
            1001,
            GroupAction::Kick { target: FOUNDER },
        );
        let state = resolve(&[g, kick_self]).unwrap();
        assert!(state.is_member(&FOUNDER));
    }

    // ── member cannot promote ──────────────────────────────────────────────

    #[test]
    fn member_cannot_promote() {
        let g = genesis(FOUNDER, GroupMode::Open);
        let join = op(vec![g.id], ALICE, 1001, GroupAction::Join);
        let promote = op(
            vec![join.id],
            ALICE,
            1002,
            GroupAction::Promote { target: BOB },
        );
        let state = resolve(&[g, join, promote]).unwrap();
        assert!(!state.is_member(&BOB));
    }

    // ── officer cannot promote ─────────────────────────────────────────────

    #[test]
    fn officer_cannot_promote() {
        let g = genesis(FOUNDER, GroupMode::Open);
        let join_a = op(vec![g.id], ALICE, 1001, GroupAction::Join);
        let join_b = op(vec![g.id], BOB, 1002, GroupAction::Join);
        let promote_a = op(
            vec![join_a.id, join_b.id],
            FOUNDER,
            1003,
            GroupAction::Promote { target: ALICE },
        );
        // Alice (Officer) tries to promote Bob.
        let promote_b = op(
            vec![promote_a.id],
            ALICE,
            1004,
            GroupAction::Promote { target: BOB },
        );
        let state = resolve(&[g, join_a, join_b, promote_a, promote_b]).unwrap();
        // Bob should still be Member.
        assert_eq!(state.role_of(&BOB), Some(Role::Member));
    }

    // ── non-member cannot leave ────────────────────────────────────────────

    #[test]
    fn non_member_cannot_leave() {
        let g = genesis(FOUNDER, GroupMode::InviteOnly);
        let leave = op(vec![g.id], ALICE, 1001, GroupAction::Leave);
        let state = resolve(&[g, leave]).unwrap();
        assert_eq!(state.members.len(), 1);
        assert!(state.is_member(&FOUNDER));
    }

    // ── partial update info ────────────────────────────────────────────────

    #[test]
    fn update_info_partial() {
        let g = genesis(FOUNDER, GroupMode::InviteOnly);
        let update = op(
            vec![g.id],
            FOUNDER,
            1001,
            GroupAction::UpdateInfo {
                name: None,
                mode: Some(GroupMode::Open),
            },
        );
        let state = resolve(&[g, update]).unwrap();
        assert_eq!(state.name, "TestGroup"); // unchanged
        assert_eq!(state.mode, GroupMode::Open); // changed
    }

    // ── non-founder cannot update info ─────────────────────────────────────

    #[test]
    fn non_founder_cannot_update_info() {
        let g = genesis(FOUNDER, GroupMode::Open);
        let join = op(vec![g.id], ALICE, 1001, GroupAction::Join);
        let update = op(
            vec![join.id],
            ALICE,
            1002,
            GroupAction::UpdateInfo {
                name: Some(String::from("Hacked")),
                mode: None,
            },
        );
        let state = resolve(&[g, join, update]).unwrap();
        assert_eq!(state.name, "TestGroup"); // unchanged
    }

    // ── non-founder cannot dissolve ────────────────────────────────────────

    #[test]
    fn non_founder_cannot_dissolve() {
        let g = genesis(FOUNDER, GroupMode::Open);
        let join = op(vec![g.id], ALICE, 1001, GroupAction::Join);
        let dissolve = op(vec![join.id], ALICE, 1002, GroupAction::Dissolve);
        let state = resolve(&[g, join, dissolve]).unwrap();
        assert!(!state.dissolved);
    }

    // ── demote non-officer is no-op ────────────────────────────────────────

    #[test]
    fn demote_member_rejected() {
        let g = genesis(FOUNDER, GroupMode::Open);
        let join = op(vec![g.id], ALICE, 1001, GroupAction::Join);
        // Try to demote a plain Member — should be rejected.
        let demote = op(
            vec![join.id],
            FOUNDER,
            1002,
            GroupAction::Demote { target: ALICE },
        );
        let state = resolve(&[g, join, demote]).unwrap();
        // Alice should still be Member (not removed, just unchanged).
        assert_eq!(state.role_of(&ALICE), Some(Role::Member));
    }

    // ── leave auto-promote picks longest-tenured officer ───────────────────

    #[test]
    fn leave_promotes_longest_tenured_officer() {
        let g = genesis(FOUNDER, GroupMode::Open);
        let join_a = op(vec![g.id], ALICE, 1001, GroupAction::Join);
        let join_b = op(vec![g.id], BOB, 1002, GroupAction::Join);
        let promote_a = op(
            vec![join_a.id, join_b.id],
            FOUNDER,
            1003,
            GroupAction::Promote { target: ALICE },
        );
        let promote_b = op(
            vec![promote_a.id],
            FOUNDER,
            1004,
            GroupAction::Promote { target: BOB },
        );
        let leave = op(vec![promote_b.id], FOUNDER, 1005, GroupAction::Leave);
        let state = resolve(&[g, join_a, join_b, promote_a, promote_b, leave]).unwrap();
        // Alice was promoted first (ts 1003 apply, joined_at 1001), Bob at 1002.
        // But "longest-tenured" means earliest joined_at — Alice joined at 1001.
        assert_eq!(state.role_of(&ALICE), Some(Role::Founder));
        assert_eq!(state.role_of(&BOB), Some(Role::Officer));
    }

    // ── concurrent ops at same authority: mutual kicks ─────────────────────

    #[test]
    fn concurrent_mutual_kicks_at_equal_rank() {
        // Two officers concurrently kick each other's favorite member.
        let g = genesis(FOUNDER, GroupMode::Open);
        let join_a = op(vec![g.id], ALICE, 1001, GroupAction::Join);
        let join_b = op(vec![g.id], BOB, 1002, GroupAction::Join);
        let join_c = op(vec![g.id], CAROL, 1003, GroupAction::Join);

        let promote_a = op(
            vec![join_a.id, join_b.id, join_c.id],
            FOUNDER,
            1004,
            GroupAction::Promote { target: ALICE },
        );
        let promote_b = op(
            vec![promote_a.id],
            FOUNDER,
            1005,
            GroupAction::Promote { target: BOB },
        );

        // Alice (Officer) kicks Carol, Bob (Officer) kicks Carol too — concurrent ops.
        // Both see Carol as Member at validation time.
        let kick_carol_by_a = op(
            vec![promote_b.id],
            ALICE,
            1006,
            GroupAction::Kick { target: CAROL },
        );
        let kick_carol_by_b = op(
            vec![promote_b.id],
            BOB,
            1006,
            GroupAction::Kick { target: CAROL },
        );

        let state = resolve(&[
            g,
            join_a,
            join_b,
            join_c,
            promote_a,
            promote_b,
            kick_carol_by_a,
            kick_carol_by_b,
        ])
        .unwrap();

        // Carol should be kicked (both kicks were valid at validation time).
        assert!(!state.is_member(&CAROL));
    }

    // ── empty dag error ────────────────────────────────────────────────────

    #[test]
    fn empty_dag_error() {
        let result = resolve(&[]);
        assert!(matches!(result, Err(ResolveError::EmptyDag)));
    }

    // ── already member cannot join again ───────────────────────────────────

    #[test]
    fn already_member_cannot_join() {
        let g = genesis(FOUNDER, GroupMode::Open);
        let join = op(vec![g.id], ALICE, 1001, GroupAction::Join);
        let join2 = op(vec![join.id], ALICE, 1002, GroupAction::Join);
        let state = resolve(&[g, join, join2]).unwrap();
        assert_eq!(state.members.len(), 2); // Founder + Alice
        // joined_at should be from first join.
        assert_eq!(state.members[&ALICE].joined_at, 1001);
    }

    // ── already member cannot be invited ───────────────────────────────────

    #[test]
    fn cannot_invite_existing_member() {
        let g = genesis(FOUNDER, GroupMode::Open);
        let join = op(vec![g.id], ALICE, 1001, GroupAction::Join);
        // Invite Alice who is already a member — should be rejected (no-op).
        let invite = op(
            vec![join.id],
            FOUNDER,
            1002,
            GroupAction::Invite { invitee: ALICE },
        );
        // The invite is rejected, but even if someone tries to accept it, Alice
        // is already a member so Accept would also fail.
        let state = resolve(&[g, join, invite]).unwrap();
        assert_eq!(state.members.len(), 2);
    }

    // ── Strong Removal: Task 5 ─────────────────────────────────────────────

    /// Build a realistic base DAG: genesis → invite Alice → Alice accepts →
    /// Founder promotes Alice to Officer → invite Bob → Bob accepts.
    /// Returns (ops_vec, branch_point_id) where branch_point_id is the id of
    /// the last op before the concurrent branch.
    fn build_base_with_officer_and_member() -> (Vec<GroupOp>, OpId) {
        let g = genesis(FOUNDER, GroupMode::InviteOnly);
        let invite_alice = op(
            vec![g.id],
            FOUNDER,
            1001,
            GroupAction::Invite { invitee: ALICE },
        );
        let accept_alice = op(
            vec![invite_alice.id],
            ALICE,
            1002,
            GroupAction::Accept {
                invite_op: invite_alice.id,
            },
        );
        let promote_alice = op(
            vec![accept_alice.id],
            FOUNDER,
            1003,
            GroupAction::Promote { target: ALICE },
        );
        let invite_bob = op(
            vec![promote_alice.id],
            ALICE,
            1004,
            GroupAction::Invite { invitee: BOB },
        );
        let accept_bob = op(
            vec![invite_bob.id],
            BOB,
            1005,
            GroupAction::Accept {
                invite_op: invite_bob.id,
            },
        );
        let branch_id = accept_bob.id;
        let ops = vec![g, invite_alice, accept_alice, promote_alice, invite_bob, accept_bob];
        (ops, branch_id)
    }

    /// Founder demotes Officer A while Officer A concurrently kicks Member B.
    /// Both ops share the same parent (Strong Removal scenario).
    ///
    /// Expected: Founder's demote (power 0) wins over Officer's kick (power 1).
    /// Alice is demoted to Member. Bob's kick is voided (Alice is no longer an
    /// Officer when her kick is validated), so Bob survives.
    #[test]
    fn concurrent_founder_demotes_officer_voids_officer_kick() {
        let (mut base_ops, branch_id) = build_base_with_officer_and_member();

        // Concurrent branch: Founder demotes Alice, Alice kicks Bob (same parent).
        let demote_alice = op(
            vec![branch_id],
            FOUNDER,
            1006,
            GroupAction::Demote { target: ALICE },
        );
        let kick_bob_by_alice = op(
            vec![branch_id],
            ALICE,
            1006,
            GroupAction::Kick { target: BOB },
        );

        base_ops.push(demote_alice);
        base_ops.push(kick_bob_by_alice);

        let state = resolve(&base_ops).unwrap();

        // Alice is demoted to Member (Founder's op wins at authority 0).
        assert_eq!(state.role_of(&ALICE), Some(Role::Member));
        // Bob survives — Alice's kick is voided because she is no longer
        // an Officer (or Founder) when her kick's authority is checked.
        assert!(state.is_member(&BOB));
        assert_eq!(state.role_of(&BOB), Some(Role::Member));
    }

    /// Founder kicks Officer while Officer concurrently invites a new member.
    /// Officer's invite is voided because the Officer is no longer in the group
    /// when his/her invite's authorization is evaluated.
    #[test]
    fn concurrent_founder_kick_voids_officer_invite() {
        let (mut base_ops, branch_id) = build_base_with_officer_and_member();

        // Introduce Carol as the "new member" Alice would try to invite.
        const CAROL: MemberAddr = [0x04; 16];

        // Concurrent: Founder kicks Alice (Officer), Alice invites Carol.
        let kick_alice = op(
            vec![branch_id],
            FOUNDER,
            1006,
            GroupAction::Kick { target: ALICE },
        );
        let invite_carol_by_alice = op(
            vec![branch_id],
            ALICE,
            1006,
            GroupAction::Invite { invitee: CAROL },
        );

        base_ops.push(kick_alice);
        base_ops.push(invite_carol_by_alice);

        let state = resolve(&base_ops).unwrap();

        // Alice is kicked (Founder wins, power 0).
        assert!(!state.is_member(&ALICE));
        // Carol was never admitted — Alice's invite was voided.
        assert!(!state.is_member(&CAROL));
        // Bob and Founder remain.
        assert!(state.is_member(&BOB));
        assert!(state.is_member(&FOUNDER));
    }

    /// The same set of ops must produce identical state regardless of the order
    /// in which they are passed to `resolve`. Tests forward, reverse, and a
    /// hand-shuffled permutation.
    #[test]
    fn concurrent_ops_deterministic_regardless_of_input_order() {
        let (mut base_ops, branch_id) = build_base_with_officer_and_member();

        let demote_alice = op(
            vec![branch_id],
            FOUNDER,
            1006,
            GroupAction::Demote { target: ALICE },
        );
        let kick_bob_by_alice = op(
            vec![branch_id],
            ALICE,
            1006,
            GroupAction::Kick { target: BOB },
        );

        base_ops.push(demote_alice);
        base_ops.push(kick_bob_by_alice);

        // Reference state using the original order.
        let reference = resolve(&base_ops).unwrap();

        // Reversed order.
        let mut reversed = base_ops.clone();
        reversed.reverse();
        let state_rev = resolve(&reversed).unwrap();
        assert_eq!(
            reference.members.len(),
            state_rev.members.len(),
            "reversed: member count differs"
        );
        assert_eq!(
            reference.role_of(&FOUNDER),
            state_rev.role_of(&FOUNDER),
            "reversed: founder role differs"
        );
        assert_eq!(
            reference.role_of(&ALICE),
            state_rev.role_of(&ALICE),
            "reversed: alice role differs"
        );
        assert_eq!(
            reference.is_member(&BOB),
            state_rev.is_member(&BOB),
            "reversed: bob membership differs"
        );

        // A hand-shuffled permutation (rotate by half).
        let n = base_ops.len();
        let mut shuffled = base_ops.clone();
        shuffled.rotate_right(n / 2);
        let state_shuffled = resolve(&shuffled).unwrap();
        assert_eq!(
            reference.members.len(),
            state_shuffled.members.len(),
            "shuffled: member count differs"
        );
        assert_eq!(
            reference.role_of(&ALICE),
            state_shuffled.role_of(&ALICE),
            "shuffled: alice role differs"
        );
        assert_eq!(
            reference.is_member(&BOB),
            state_shuffled.is_member(&BOB),
            "shuffled: bob membership differs"
        );
    }

    // ── shuffle invariant (Task 6) ─────────────────────────────────────────

    /// Build a complex DAG (~9 ops) and prove that shuffling the input 50 times
    /// with a seeded RNG always produces identical `GroupState`.
    ///
    /// This is the single most important invariant test: it proves the resolver
    /// is fully deterministic regardless of op arrival order.
    #[test]
    fn shuffle_invariant_randomized() {
        use rand::seq::SliceRandom;
        use rand::SeedableRng;

        const CAROL: MemberAddr = [0x04; 16];

        // DAG shape:
        //   genesis → invite_alice → accept_alice → promote_alice (Officer)
        //          → invite_bob
        //               ↓ (two concurrent children)
        //          accept_bob  ←→  kick_bob (both children of invite_bob)
        //               ↓ (merges both branches)
        //          invite_carol → accept_carol
        let g = genesis(FOUNDER, GroupMode::InviteOnly);
        let invite_alice = op(
            vec![g.id],
            FOUNDER,
            1001,
            GroupAction::Invite { invitee: ALICE },
        );
        let accept_alice = op(
            vec![invite_alice.id],
            ALICE,
            1002,
            GroupAction::Accept {
                invite_op: invite_alice.id,
            },
        );
        let promote_alice = op(
            vec![accept_alice.id],
            FOUNDER,
            1003,
            GroupAction::Promote { target: ALICE },
        );
        let invite_bob = op(
            vec![promote_alice.id],
            ALICE,
            1004,
            GroupAction::Invite { invitee: BOB },
        );
        // Two concurrent ops off invite_bob: Bob accepts, Founder kicks Bob.
        let accept_bob = op(
            vec![invite_bob.id],
            BOB,
            1005,
            GroupAction::Accept {
                invite_op: invite_bob.id,
            },
        );
        let kick_bob = op(
            vec![invite_bob.id],
            FOUNDER,
            1005,
            GroupAction::Kick { target: BOB },
        );
        // After the fork merges, Alice invites Carol.
        let invite_carol = op(
            vec![kick_bob.id, accept_bob.id],
            ALICE,
            1006,
            GroupAction::Invite { invitee: CAROL },
        );
        let accept_carol = op(
            vec![invite_carol.id],
            CAROL,
            1007,
            GroupAction::Accept {
                invite_op: invite_carol.id,
            },
        );

        let all_ops = vec![
            g,
            invite_alice,
            accept_alice,
            promote_alice,
            invite_bob,
            accept_bob,
            kick_bob,
            invite_carol,
            accept_carol,
        ];

        // Reference state with canonical order.
        let reference = resolve(&all_ops).unwrap();

        // 50 shuffles with a deterministic seeded RNG — all must match.
        let mut rng = rand::rngs::StdRng::seed_from_u64(12345);
        for i in 0..50 {
            let mut shuffled = all_ops.clone();
            shuffled.shuffle(&mut rng);
            let state = resolve(&shuffled).unwrap();

            assert_eq!(
                reference.members.len(),
                state.members.len(),
                "shuffle {i}: member count differs"
            );
            assert_eq!(
                reference.dissolved,
                state.dissolved,
                "shuffle {i}: dissolved differs"
            );
            assert_eq!(reference.name, state.name, "shuffle {i}: name differs");
            assert_eq!(
                reference.role_of(&FOUNDER),
                state.role_of(&FOUNDER),
                "shuffle {i}: founder role differs"
            );
            assert_eq!(
                reference.role_of(&ALICE),
                state.role_of(&ALICE),
                "shuffle {i}: alice role differs"
            );
            assert_eq!(
                reference.is_member(&BOB),
                state.is_member(&BOB),
                "shuffle {i}: bob membership differs"
            );
            assert_eq!(
                reference.role_of(&CAROL),
                state.role_of(&CAROL),
                "shuffle {i}: carol role differs"
            );
        }
    }

    /// After a member is kicked, any subsequent op authored by that member
    /// must be rejected (they are no longer in the group).
    #[test]
    fn kicked_member_subsequent_ops_rejected() {
        let g = genesis(FOUNDER, GroupMode::InviteOnly);
        let invite_alice = op(
            vec![g.id],
            FOUNDER,
            1001,
            GroupAction::Invite { invitee: ALICE },
        );
        let accept_alice = op(
            vec![invite_alice.id],
            ALICE,
            1002,
            GroupAction::Accept {
                invite_op: invite_alice.id,
            },
        );
        let kick_alice = op(
            vec![accept_alice.id],
            FOUNDER,
            1003,
            GroupAction::Kick { target: ALICE },
        );
        // Alice tries to Leave after being kicked — she is no longer a member.
        let leave_alice = op(
            vec![kick_alice.id],
            ALICE,
            1004,
            GroupAction::Leave,
        );

        let state = resolve(&[g, invite_alice, accept_alice, kick_alice, leave_alice]).unwrap();

        // Alice is not a member (was kicked, Leave was rejected since she's not a member).
        assert!(!state.is_member(&ALICE));
        // Founder is the only member.
        assert!(state.is_member(&FOUNDER));
        assert_eq!(state.members.len(), 1);
    }

    // ── unauthorized invite cannot be accepted ────────────────────────────

    #[test]
    fn unauthorized_invite_cannot_be_accepted() {
        let g = genesis(FOUNDER, GroupMode::InviteOnly);
        let join_alice = op(vec![g.id], ALICE, 1001, GroupAction::Join);
        // Alice is not a member (InviteOnly), but she crafts an invite for Bob.
        // This invite is unauthorized (Alice is not a member/officer/founder).
        let bad_invite = op(
            vec![g.id],
            ALICE,
            1001,
            GroupAction::Invite { invitee: BOB },
        );
        // Bob tries to accept the unauthorized invite.
        let accept = op(
            vec![bad_invite.id],
            BOB,
            1002,
            GroupAction::Accept {
                invite_op: bad_invite.id,
            },
        );
        let state = resolve(&[g, join_alice, bad_invite, accept]).unwrap();
        assert!(!state.is_member(&BOB));
        assert!(!state.is_member(&ALICE));
        assert_eq!(state.members.len(), 1);
    }

    #[test]
    fn accept_requires_causal_ancestry() {
        let g = genesis(FOUNDER, GroupMode::InviteOnly);
        // Founder invites Alice on one branch.
        let invite_alice = op(
            vec![g.id],
            FOUNDER,
            1001,
            GroupAction::Invite { invitee: ALICE },
        );
        // Alice accepts, but her Accept op does NOT have invite as a parent —
        // it branches from genesis directly (not causally linked to the invite).
        let accept_alice = op(
            vec![g.id],
            ALICE,
            1002,
            GroupAction::Accept {
                invite_op: invite_alice.id,
            },
        );
        let state = resolve(&[g, invite_alice, accept_alice]).unwrap();
        // Accept should be rejected: the invite is not in accept's ancestry.
        assert!(!state.is_member(&ALICE));
    }

    // ── concurrent Founder Leave + Kick cannot leave group Founder-less ──

    #[test]
    fn concurrent_founder_leave_and_kick_of_successor_still_has_founder() {
        // Set up: Founder + Alice (Officer) + Bob (Member).
        let g = genesis(FOUNDER, GroupMode::InviteOnly);
        let invite_a = op(vec![g.id], FOUNDER, 1001, GroupAction::Invite { invitee: ALICE });
        let accept_a = op(vec![invite_a.id], ALICE, 1002, GroupAction::Accept { invite_op: invite_a.id });
        let promote_a = op(vec![accept_a.id], FOUNDER, 1003, GroupAction::Promote { target: ALICE });
        let invite_b = op(vec![promote_a.id], FOUNDER, 1004, GroupAction::Invite { invitee: BOB });
        let accept_b = op(vec![invite_b.id], BOB, 1005, GroupAction::Accept { invite_op: invite_b.id });

        // Founder concurrently Leaves and Kicks Alice (the successor).
        // Both ops branch off accept_b — same authority (Founder), same batch.
        let leave = op(vec![accept_b.id], FOUNDER, 1006, GroupAction::Leave);
        let kick_alice = op(vec![accept_b.id], FOUNDER, 1006, GroupAction::Kick { target: ALICE });

        let state = resolve(&[
            g, invite_a, accept_a, promote_a, invite_b, accept_b, leave, kick_alice,
        ])
        .unwrap();

        // Whatever ordering the batch processor picks, the group must have a
        // Founder (or be dissolved) — never Founder-less with active members.
        assert!(!state.is_member(&FOUNDER), "Founder left");
        assert!(!state.is_member(&ALICE), "Alice was kicked");
        if !state.dissolved {
            let founders: Vec<_> = state
                .members
                .iter()
                .filter(|(_, e)| e.role == Role::Founder)
                .collect();
            assert_eq!(
                founders.len(),
                1,
                "group must have exactly one Founder after Leave+Kick race, got {}",
                founders.len()
            );
            assert_eq!(founders[0].0, &BOB, "Bob should have been promoted to Founder");
        }
    }

    // ── concurrent Founder Leave + Demote cannot clobber the new Founder ─

    #[test]
    fn concurrent_founder_leave_and_demote_of_successor_preserves_founder() {
        // Pre-state: FOUNDER + Alice (Officer). If Founder concurrently
        // Leaves and Demotes Alice, Leave's successor logic may promote
        // Alice to Founder before the Demote applies. The Demote must not
        // then clobber Alice's new Founder role back to Member.
        let g = genesis(FOUNDER, GroupMode::InviteOnly);
        let invite_a = op(vec![g.id], FOUNDER, 1001, GroupAction::Invite { invitee: ALICE });
        let accept_a = op(vec![invite_a.id], ALICE, 1002, GroupAction::Accept { invite_op: invite_a.id });
        let promote_a = op(vec![accept_a.id], FOUNDER, 1003, GroupAction::Promote { target: ALICE });

        let leave = op(vec![promote_a.id], FOUNDER, 1004, GroupAction::Leave);
        let demote_alice = op(vec![promote_a.id], FOUNDER, 1004, GroupAction::Demote { target: ALICE });

        let state = resolve(&[g, invite_a, accept_a, promote_a, leave, demote_alice]).unwrap();

        assert!(!state.is_member(&FOUNDER), "Founder left");
        assert!(state.is_member(&ALICE), "Alice still in group");
        assert_eq!(
            state.role_of(&ALICE),
            Some(Role::Founder),
            "Alice must be Founder — succession must not be clobbered by the concurrent Demote"
        );
        assert!(!state.dissolved);
    }

    // ── concurrent Founder Leave + Promote cannot clobber the new Founder ─

    #[test]
    fn concurrent_founder_leave_and_promote_of_successor_preserves_founder() {
        // Pre-state: FOUNDER + Alice (Member, no Officers). If Founder
        // concurrently Leaves and Promotes Alice, Leave's succession will
        // pick Alice (longest-tenured Member) and make her Founder. The
        // Promote must not then clobber her back to Officer.
        //
        // The resolver applies equal-authority ops in OpId order within a
        // batch, and OpIds are BLAKE3 hashes — essentially random relative
        // to op content. To reliably exercise the *buggy* path (Leave
        // applies before Promote, so succession runs before the would-be
        // clobber), we enumerate several plausible timestamp tweaks and
        // require at least one to land in the buggy ordering.
        let g = genesis(FOUNDER, GroupMode::Open);
        let join_a = op(vec![g.id], ALICE, 1001, GroupAction::Join);

        let mut exercised_buggy_ordering = false;
        for ts_tweak in 0u64..10 {
            let leave = op(vec![join_a.id], FOUNDER, 1002 + ts_tweak, GroupAction::Leave);
            let promote_alice = op(
                vec![join_a.id],
                FOUNDER,
                1002 + ts_tweak,
                GroupAction::Promote { target: ALICE },
            );

            // The resolver reverses the compare for max-heap ordering, so
            // the op with the *smaller* OpId is actually popped first and
            // applied first. Confirm this tweak actually exercises the bug
            // path (Leave before Promote).
            if leave.id < promote_alice.id {
                exercised_buggy_ordering = true;
            }

            let state = resolve(&[g.clone(), join_a.clone(), leave, promote_alice]).unwrap();

            assert!(!state.is_member(&FOUNDER), "Founder left");
            assert!(state.is_member(&ALICE));
            assert_eq!(
                state.role_of(&ALICE),
                Some(Role::Founder),
                "Alice must be Founder — succession must not be clobbered by the concurrent Promote (ts_tweak={ts_tweak})"
            );
            assert!(!state.dissolved);
        }
        assert!(
            exercised_buggy_ordering,
            "none of the tried timestamps produced the Leave-before-Promote ordering — test may not be catching the regression"
        );
    }
}
