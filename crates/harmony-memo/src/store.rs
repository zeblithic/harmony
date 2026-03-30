//! In-memory store for signed memos, keyed by input [`ContentId`].
//!
//! `MemoStore` is `no_std` compatible: it uses `hashbrown::HashMap` when
//! the `std` feature is absent and `std::collections::HashMap` otherwise.

#[cfg(not(feature = "std"))]
use hashbrown::HashMap;
#[cfg(feature = "std")]
use std::collections::HashMap;

use alloc::vec::Vec;
use harmony_content::ContentId;
use harmony_identity::IdentityRef;

use crate::Memo;

/// An in-memory collection of signed memos indexed by input [`ContentId`].
///
/// Deduplication key: `(input, output, credential.issuer.hash)`. Inserting
/// a memo whose tuple already exists is a no-op (returns `false`).
pub struct MemoStore {
    by_input: HashMap<ContentId, Vec<Memo>>,
    /// Total memo count across all inputs. Maintained on insert for O(1) len/is_empty.
    total: usize,
}

impl MemoStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self {
            by_input: HashMap::new(),
            total: 0,
        }
    }

    /// Insert a memo into the store.
    ///
    /// Returns `true` if the memo was added, or `false` if a memo with the
    /// same `(input, output, issuer.hash)` triple already exists.
    pub fn insert(&mut self, memo: Memo) -> bool {
        let signer_hash = memo.credential.issuer.hash;
        let entry = self.by_input.entry(memo.input).or_default();

        // Dedup: skip if we already have a memo with the same output + issuer hash.
        let already_present = entry.iter().any(|existing| {
            existing.output == memo.output && existing.credential.issuer.hash == signer_hash
        });

        if already_present {
            return false;
        }

        entry.push(memo);
        self.total += 1;
        true
    }

    /// Return all memos whose input matches `input`.
    ///
    /// Returns an empty slice when no memos are known for this input.
    pub fn get_by_input(&self, input: &ContentId) -> &[Memo] {
        match self.by_input.get(input) {
            Some(v) => v.as_slice(),
            None => &[],
        }
    }

    /// Return all memos from `signer` about `input`.
    /// A signer may have multiple memos for the same input with different outputs
    /// (e.g., if they recomputed and got a different result).
    pub fn get_by_input_and_signer(
        &self,
        input: &ContentId,
        signer: &[u8; 16],
    ) -> Vec<&Memo> {
        match self.by_input.get(input) {
            Some(memos) => memos.iter().filter(|m| &m.credential.issuer.hash == signer).collect(),
            None => Vec::new(),
        }
    }

    /// Group memos for `input` by output CID, returning each distinct output
    /// alongside the list of signers (as [`IdentityRef`]) who attested to it.
    pub fn outputs_for_input(&self, input: &ContentId) -> Vec<(ContentId, Vec<IdentityRef>)> {
        let memos = match self.by_input.get(input) {
            Some(v) => v,
            None => return Vec::new(),
        };

        // Preserve insertion order: build a Vec of (output, signers) pairs.
        let mut groups: Vec<(ContentId, Vec<IdentityRef>)> = Vec::new();

        for memo in memos {
            if let Some(group) = groups.iter_mut().find(|(cid, _)| *cid == memo.output) {
                group.1.push(memo.credential.issuer);
            } else {
                groups.push((memo.output, alloc::vec![memo.credential.issuer]));
            }
        }

        groups
    }

    /// Iterate over the distinct input [`ContentId`]s held in the store.
    ///
    /// Useful for populating a Bloom filter.
    pub fn input_cids(&self) -> impl Iterator<Item = &ContentId> {
        self.by_input.keys()
    }

    /// Total number of memos held in the store. O(1).
    pub fn len(&self) -> usize {
        self.total
    }

    /// Return `true` if the store contains no memos. O(1).
    pub fn is_empty(&self) -> bool {
        self.total == 0
    }
}

impl Default for MemoStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::create::create_memo;
    use harmony_identity::pq_identity::PqPrivateIdentity;

    fn make_cid(byte: u8) -> ContentId {
        ContentId::from_bytes([byte; 32])
    }

    fn make_memo(
        identity: &PqPrivateIdentity,
        input_byte: u8,
        output_byte: u8,
    ) -> Memo {
        create_memo(
            make_cid(input_byte),
            make_cid(output_byte),
            identity,
            &mut rand::rngs::OsRng,
            1000,
            9999,
        )
        .expect("create_memo")
    }

    #[test]
    fn insert_and_query_by_input() {
        let identity = PqPrivateIdentity::generate(&mut rand::rngs::OsRng);
        let mut store = MemoStore::new();

        let memo = make_memo(&identity, 0x01, 0x02);
        let input = memo.input;

        assert!(store.insert(memo));
        let results = store.get_by_input(&input);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].input, make_cid(0x01));
        assert_eq!(results[0].output, make_cid(0x02));
    }

    #[test]
    fn dedup_same_signer_same_input_output() {
        let identity = PqPrivateIdentity::generate(&mut rand::rngs::OsRng);
        let mut store = MemoStore::new();

        let memo1 = make_memo(&identity, 0x01, 0x02);
        let memo2 = make_memo(&identity, 0x01, 0x02);
        let input = memo1.input;

        assert!(store.insert(memo1));
        assert!(!store.insert(memo2), "duplicate memo should be rejected");
        assert_eq!(store.get_by_input(&input).len(), 1);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn different_signers_same_input_output_coexist() {
        let alice = PqPrivateIdentity::generate(&mut rand::rngs::OsRng);
        let bob = PqPrivateIdentity::generate(&mut rand::rngs::OsRng);
        let mut store = MemoStore::new();

        let memo_alice = make_memo(&alice, 0x01, 0x02);
        let memo_bob = make_memo(&bob, 0x01, 0x02);
        let input = memo_alice.input;

        assert!(store.insert(memo_alice));
        assert!(store.insert(memo_bob));
        assert_eq!(store.get_by_input(&input).len(), 2);
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn different_outputs_same_input_coexist() {
        let identity = PqPrivateIdentity::generate(&mut rand::rngs::OsRng);
        let mut store = MemoStore::new();

        let memo_a = make_memo(&identity, 0x01, 0x02);
        let memo_b = make_memo(&identity, 0x01, 0x03);
        let input = memo_a.input;

        assert!(store.insert(memo_a));
        assert!(store.insert(memo_b));
        assert_eq!(store.get_by_input(&input).len(), 2);
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn outputs_for_input_groups_correctly() {
        let alice = PqPrivateIdentity::generate(&mut rand::rngs::OsRng);
        let bob = PqPrivateIdentity::generate(&mut rand::rngs::OsRng);
        let mut store = MemoStore::new();

        // Both alice and bob attest input→output_02
        let memo_alice_02 = make_memo(&alice, 0x01, 0x02);
        let memo_bob_02 = make_memo(&bob, 0x01, 0x02);
        // Alice alone attests input→output_03 (disagreeing output)
        let memo_alice_03 = make_memo(&alice, 0x01, 0x03);

        let input = memo_alice_02.input;

        store.insert(memo_alice_02);
        store.insert(memo_bob_02);
        store.insert(memo_alice_03);

        let groups = store.outputs_for_input(&input);
        assert_eq!(groups.len(), 2, "should have two distinct output CIDs");

        let group_02 = groups.iter().find(|(cid, _)| *cid == make_cid(0x02)).expect("group for 0x02");
        assert_eq!(group_02.1.len(), 2, "two signers for output 0x02");

        let group_03 = groups.iter().find(|(cid, _)| *cid == make_cid(0x03)).expect("group for 0x03");
        assert_eq!(group_03.1.len(), 1, "one signer for output 0x03");
    }

    #[test]
    fn get_by_input_and_signer() {
        let alice = PqPrivateIdentity::generate(&mut rand::rngs::OsRng);
        let bob = PqPrivateIdentity::generate(&mut rand::rngs::OsRng);
        let mut store = MemoStore::new();

        let memo_alice = make_memo(&alice, 0x01, 0x02);
        let input = memo_alice.input;
        let alice_hash = memo_alice.credential.issuer.hash;
        let bob_hash = {
            let m = make_memo(&bob, 0x01, 0x02);
            m.credential.issuer.hash
        };

        store.insert(memo_alice);

        assert!(!store.get_by_input_and_signer(&input, &alice_hash).is_empty(), "alice's memo should be found");
        assert!(store.get_by_input_and_signer(&input, &bob_hash).is_empty(), "bob has no memo");
        assert!(store.get_by_input_and_signer(&make_cid(0xFF), &alice_hash).is_empty(), "unknown input");
    }

    #[test]
    fn input_cids_iteration() {
        let identity = PqPrivateIdentity::generate(&mut rand::rngs::OsRng);
        let mut store = MemoStore::new();

        let memo_a = make_memo(&identity, 0x01, 0x02);
        let memo_b = make_memo(&identity, 0x03, 0x04);
        let input_a = memo_a.input;
        let input_b = memo_b.input;

        store.insert(memo_a);
        store.insert(memo_b);

        let cids: Vec<ContentId> = store.input_cids().copied().collect();
        assert_eq!(cids.len(), 2, "two distinct input CIDs");
        assert!(cids.contains(&input_a), "input_a present");
        assert!(cids.contains(&input_b), "input_b present");
    }
}
