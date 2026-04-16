use alloc::string::String;
use alloc::vec::Vec;
use hashbrown::HashMap;
use serde::{Deserialize, Serialize};

/// 16-byte unique identifier for a group.
pub type GroupId = [u8; 16];

/// 16-byte address of a group member.
pub type MemberAddr = [u8; 16];

/// 32-byte content-addressed identifier for an op (BLAKE3 hash of canonical payload bytes).
pub type OpId = [u8; 32];

/// Member role within a group, ordered by authority level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[repr(u8)]
pub enum Role {
    Founder = 0,
    Officer = 1,
    Member = 2,
}

impl Role {
    /// Numeric power level — lower numbers mean more authority.
    pub fn power_level(self) -> u8 {
        self as u8
    }

    /// Returns true if `self` outranks `other` (has more authority).
    pub fn outranks(self, other: Role) -> bool {
        self.power_level() < other.power_level()
    }
}

/// Group access control mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GroupMode {
    InviteOnly,
    Open,
}

/// The action carried by a `GroupOp`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GroupAction {
    /// Create the group (genesis op).
    Create {
        group_id: GroupId,
        name: String,
        mode: GroupMode,
    },
    /// Invite a member (officer or founder issues).
    Invite { invitee: MemberAddr },
    /// Join an open group (self-issued).
    Join,
    /// Accept an invitation (invitee confirms).
    Accept { invite_op: OpId },
    /// Leave the group (self-issued).
    Leave,
    /// Kick a member (officer or founder issues).
    Kick { target: MemberAddr },
    /// Promote a member to officer (founder only).
    Promote { target: MemberAddr },
    /// Demote an officer back to member (founder only).
    Demote { target: MemberAddr },
    /// Dissolve (close) the group permanently (founder only).
    Dissolve,
    /// Update group metadata.
    UpdateInfo { name: Option<String>, mode: Option<GroupMode> },
}

/// Canonical op payload used for content addressing (excludes `id` and `signature`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GroupOpPayload {
    /// IDs of causal predecessor ops. Empty only for the genesis op.
    pub parents: Vec<OpId>,
    /// Address of the op author.
    pub author: MemberAddr,
    /// Unix timestamp (seconds) at which the op was created.
    pub timestamp: u64,
    /// The semantic action this op describes.
    pub action: GroupAction,
}

/// A fully-formed group operation, including its content-addressed ID and an
/// optional detached signature over the canonical payload bytes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GroupOp {
    /// Content-addressed identifier (BLAKE3 of canonical postcard bytes).
    pub id: OpId,
    /// IDs of causal predecessor ops.
    pub parents: Vec<OpId>,
    /// Address of the op author.
    pub author: MemberAddr,
    /// Detached signature bytes (empty if unsigned).
    pub signature: Vec<u8>,
    /// Unix timestamp (seconds).
    pub timestamp: u64,
    /// The semantic action.
    pub action: GroupAction,
}

/// Membership record for a single member in a resolved `GroupState`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemberEntry {
    pub role: Role,
    /// Timestamp of the op that admitted this member.
    pub joined_at: u64,
}

/// Resolved group state produced by replaying a validated DAG.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GroupState {
    pub group_id: GroupId,
    pub name: String,
    pub mode: GroupMode,
    pub members: HashMap<MemberAddr, MemberEntry>,
    pub dissolved: bool,
}

impl Default for GroupState {
    fn default() -> Self {
        Self {
            group_id: [0u8; 16],
            name: String::new(),
            mode: GroupMode::InviteOnly,
            members: HashMap::new(),
            dissolved: false,
        }
    }
}

impl GroupState {
    /// Returns the role of `addr` if they are a current member.
    pub fn role_of(&self, addr: &MemberAddr) -> Option<Role> {
        self.members.get(addr).map(|e| e.role)
    }

    /// Returns true if `addr` is a current member.
    pub fn is_member(&self, addr: &MemberAddr) -> bool {
        self.members.contains_key(addr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn role_power_levels() {
        assert_eq!(Role::Founder.power_level(), 0);
        assert_eq!(Role::Officer.power_level(), 1);
        assert_eq!(Role::Member.power_level(), 2);
    }

    #[test]
    fn role_outranks() {
        assert!(Role::Founder.outranks(Role::Officer));
        assert!(Role::Founder.outranks(Role::Member));
        assert!(Role::Officer.outranks(Role::Member));
        assert!(!Role::Member.outranks(Role::Officer));
        assert!(!Role::Member.outranks(Role::Founder));
        assert!(!Role::Officer.outranks(Role::Founder));
        assert!(!Role::Member.outranks(Role::Member));
    }

    #[test]
    fn role_ordering() {
        // Lower ord value = higher authority
        assert!(Role::Founder < Role::Officer);
        assert!(Role::Officer < Role::Member);
        assert!(Role::Founder < Role::Member);
    }

    #[test]
    fn group_state_role_of() {
        let mut state = GroupState::default();
        let addr: MemberAddr = [0x01; 16];
        assert_eq!(state.role_of(&addr), None);
        assert!(!state.is_member(&addr));

        state.members.insert(addr, MemberEntry { role: Role::Founder, joined_at: 1000 });
        assert_eq!(state.role_of(&addr), Some(Role::Founder));
        assert!(state.is_member(&addr));

        let officer: MemberAddr = [0x02; 16];
        state.members.insert(officer, MemberEntry { role: Role::Officer, joined_at: 2000 });
        assert_eq!(state.role_of(&officer), Some(Role::Officer));
    }

    #[test]
    fn serde_round_trip_group_action_create() {
        let action = GroupAction::Create {
            group_id: [0xAB; 16],
            name: alloc::string::String::from("Test Group"),
            mode: GroupMode::InviteOnly,
        };
        let bytes = postcard::to_allocvec(&action).unwrap();
        let decoded: GroupAction = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(action, decoded);
    }

    #[test]
    fn serde_round_trip_group_op_payload() {
        let payload = GroupOpPayload {
            parents: alloc::vec![[0x11; 32], [0x22; 32]],
            author: [0x33; 16],
            timestamp: 1_700_000_000,
            action: GroupAction::Invite { invitee: [0x44; 16] },
        };
        let bytes = postcard::to_allocvec(&payload).unwrap();
        let decoded: GroupOpPayload = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(payload, decoded);
    }

    #[test]
    fn serde_round_trip_member_entry() {
        let entry = MemberEntry { role: Role::Officer, joined_at: 999 };
        let bytes = postcard::to_allocvec(&entry).unwrap();
        let decoded: MemberEntry = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(entry, decoded);
    }

    #[test]
    fn group_state_default() {
        let state = GroupState::default();
        assert_eq!(state.group_id, [0u8; 16]);
        assert!(state.name.is_empty());
        assert_eq!(state.mode, GroupMode::InviteOnly);
        assert!(state.members.is_empty());
        assert!(!state.dissolved);
    }

    #[test]
    fn serde_round_trip_group_state() {
        let mut state = GroupState {
            group_id: [0x7F; 16],
            name: alloc::string::String::from("My Group"),
            mode: GroupMode::Open,
            members: HashMap::new(),
            dissolved: false,
        };
        state.members.insert(
            [0x01; 16],
            MemberEntry { role: Role::Founder, joined_at: 500 },
        );
        let bytes = postcard::to_allocvec(&state).unwrap();
        let decoded: GroupState = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(state, decoded);
    }
}
