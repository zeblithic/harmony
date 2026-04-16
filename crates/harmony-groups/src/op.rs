use alloc::vec::Vec;
use crate::types::{GroupAction, GroupOp, GroupOpPayload, MemberAddr, OpId};

impl GroupOp {
    /// Serialize `payload` with postcard and return the first 32 bytes of its
    /// BLAKE3 hash as the canonical op ID.
    pub fn compute_id(payload: &GroupOpPayload) -> OpId {
        let bytes = postcard::to_allocvec(payload).expect("GroupOpPayload is always serializable");
        let hash = blake3::hash(&bytes);
        let mut id = [0u8; 32];
        id.copy_from_slice(hash.as_bytes());
        id
    }

    /// Build a new unsigned `GroupOp` from its constituent fields.
    ///
    /// Returns `(op, canonical_bytes)` where `canonical_bytes` is the postcard
    /// encoding of the `GroupOpPayload` — the bytes the author should sign.
    pub fn new_unsigned(
        parents: Vec<OpId>,
        author: MemberAddr,
        timestamp: u64,
        action: GroupAction,
    ) -> (Self, Vec<u8>) {
        let payload = GroupOpPayload {
            parents: parents.clone(),
            author,
            timestamp,
            action: action.clone(),
        };
        let canonical =
            postcard::to_allocvec(&payload).expect("GroupOpPayload is always serializable");
        let id = {
            let hash = blake3::hash(&canonical);
            let mut buf = [0u8; 32];
            buf.copy_from_slice(hash.as_bytes());
            buf
        };
        let op = GroupOp {
            id,
            parents,
            author,
            signature: Vec::new(),
            timestamp,
            action,
        };
        (op, canonical)
    }

    /// Recomputes the op's ID from its fields and checks that it matches
    /// the stored `id`.
    pub fn verify_id(&self) -> bool {
        let payload = GroupOpPayload {
            parents: self.parents.clone(),
            author: self.author,
            timestamp: self.timestamp,
            action: self.action.clone(),
        };
        let expected = Self::compute_id(&payload);
        self.id == expected
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{GroupId, GroupMode};
    use alloc::string::String;

    fn make_create_payload(group_id: GroupId) -> GroupOpPayload {
        GroupOpPayload {
            parents: alloc::vec![],
            author: [0xAA; 16],
            timestamp: 1_700_000_000,
            action: GroupAction::Create {
                group_id,
                name: String::from("Test"),
                mode: GroupMode::InviteOnly,
            },
        }
    }

    #[test]
    fn deterministic_ids() {
        let payload = make_create_payload([0x01; 16]);
        let id1 = GroupOp::compute_id(&payload);
        let id2 = GroupOp::compute_id(&payload);
        assert_eq!(id1, id2);
    }

    #[test]
    fn different_payloads_different_ids() {
        let p1 = make_create_payload([0x01; 16]);
        let p2 = make_create_payload([0x02; 16]);
        assert_ne!(GroupOp::compute_id(&p1), GroupOp::compute_id(&p2));
    }

    #[test]
    fn new_unsigned_consistency() {
        let (op, canonical) = GroupOp::new_unsigned(
            alloc::vec![],
            [0xBB; 16],
            42,
            GroupAction::Create {
                group_id: [0x03; 16],
                name: String::from("Alpha"),
                mode: GroupMode::Open,
            },
        );
        // ID must equal hash of the returned canonical bytes
        let hash = blake3::hash(&canonical);
        let mut expected = [0u8; 32];
        expected.copy_from_slice(hash.as_bytes());
        assert_eq!(op.id, expected);
        // Signature must be empty for unsigned ops
        assert!(op.signature.is_empty());
        // verify_id must pass
        assert!(op.verify_id());
    }

    #[test]
    fn tamper_detection() {
        let (mut op, _) = GroupOp::new_unsigned(
            alloc::vec![],
            [0xCC; 16],
            100,
            GroupAction::Dissolve,
        );
        assert!(op.verify_id());
        // Tamper with the timestamp
        op.timestamp = 999;
        assert!(!op.verify_id());
    }

    #[test]
    fn canonical_byte_stability() {
        // The same payload always encodes to the same bytes
        let payload = GroupOpPayload {
            parents: alloc::vec![[0xFF; 32]],
            author: [0x11; 16],
            timestamp: 12345,
            action: GroupAction::Invite { invitee: [0x22; 16] },
        };
        let b1 = postcard::to_allocvec(&payload).unwrap();
        let b2 = postcard::to_allocvec(&payload).unwrap();
        assert_eq!(b1, b2);
        // And the ID is derived deterministically from those bytes
        let id1 = GroupOp::compute_id(&payload);
        let id2 = GroupOp::compute_id(&payload);
        assert_eq!(id1, id2);
    }

    #[test]
    fn verify_id_correct_after_round_trip() {
        let (op, _) = GroupOp::new_unsigned(
            alloc::vec![[0xDE; 32]],
            [0x55; 16],
            500,
            GroupAction::Leave,
        );
        let bytes = postcard::to_allocvec(&op).unwrap();
        let decoded: GroupOp = postcard::from_bytes(&bytes).unwrap();
        assert!(decoded.verify_id());
    }
}
