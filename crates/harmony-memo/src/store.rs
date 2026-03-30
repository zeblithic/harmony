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

/// Dedup key for a memo: (input, output, signer_hash).
type MemoKey = (ContentId, ContentId, [u8; 16]);

/// An in-memory collection of signed memos indexed by input [`ContentId`].
///
/// Deduplication key: `(input, output, credential.issuer.hash)`. Inserting
/// a memo whose tuple already exists is a no-op (returns `false`).
pub struct MemoStore {
    by_input: HashMap<ContentId, Vec<Memo>>,
    /// Total memo count across all inputs. Maintained on insert for O(1) len/is_empty.
    total: usize,
    /// LFU access counters keyed by (input, output, signer_hash).
    lfu_counts: HashMap<MemoKey, u32>,
}

impl MemoStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self {
            by_input: HashMap::new(),
            total: 0,
            lfu_counts: HashMap::new(),
        }
    }

    /// Insert a memo into the store.
    ///
    /// Returns `true` if the memo was added, or `false` if a memo with the
    /// same `(input, output, issuer.hash)` triple already exists.
    pub fn insert(&mut self, memo: Memo) -> bool {
        let input = memo.input;
        let output = memo.output;
        let signer_hash = memo.credential.issuer.hash;
        let entry = self.by_input.entry(input).or_insert_with(Vec::new);

        // Dedup: skip if we already have a memo with the same output + issuer hash.
        let already_present = entry.iter().any(|existing| {
            existing.output == output && existing.credential.issuer.hash == signer_hash
        });

        if already_present {
            return false;
        }

        entry.push(memo);
        self.total += 1;
        self.lfu_counts.insert((input, output, signer_hash), 0);
        true
    }

    /// Return all memos whose input matches `input`.
    ///
    /// Returns an empty slice when no memos are known for this input.
    /// Increments LFU counters for each returned memo.
    pub fn get_by_input(&mut self, input: &ContentId) -> &[Memo] {
        // First pass: collect keys to increment
        let keys: Vec<MemoKey> = match self.by_input.get(input) {
            Some(v) => v
                .iter()
                .map(|m| (m.input, m.output, m.credential.issuer.hash))
                .collect(),
            None => return &[],
        };
        // Increment LFU counters
        for key in keys {
            if let Some(count) = self.lfu_counts.get_mut(&key) {
                *count = count.saturating_add(1);
            }
        }
        // Return the slice
        self.by_input
            .get(input)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Return all memos whose input matches `input` without incrementing LFU counters.
    ///
    /// Use this for existence checks and dedup lookups where access shouldn't
    /// influence eviction ordering.
    pub fn peek_by_input(&self, input: &ContentId) -> &[Memo] {
        match self.by_input.get(input) {
            Some(v) => v.as_slice(),
            None => &[],
        }
    }

    /// Return all memos from `signer` about `input`.
    /// A signer may have multiple memos for the same input with different outputs
    /// (e.g., if they recomputed and got a different result).
    /// Increments LFU counters for each matching memo.
    pub fn get_by_input_and_signer(
        &mut self,
        input: &ContentId,
        signer: &[u8; 16],
    ) -> Vec<&Memo> {
        // First pass: collect matching keys to increment
        let keys: Vec<MemoKey> = match self.by_input.get(input) {
            Some(memos) => memos
                .iter()
                .filter(|m| &m.credential.issuer.hash == signer)
                .map(|m| (m.input, m.output, m.credential.issuer.hash))
                .collect(),
            None => return Vec::new(),
        };
        // Increment LFU counters
        for key in keys {
            if let Some(count) = self.lfu_counts.get_mut(&key) {
                *count = count.saturating_add(1);
            }
        }
        // Return matching memos
        match self.by_input.get(input) {
            Some(memos) => memos
                .iter()
                .filter(|m| &m.credential.issuer.hash == signer)
                .collect(),
            None => Vec::new(),
        }
    }

    /// Return the LFU access count for a specific memo identified by its
    /// (input, output, signer_hash) triple.
    pub fn lfu_count(&self, input: &ContentId, output: &ContentId, signer: &[u8; 16]) -> u32 {
        self.lfu_counts
            .get(&(*input, *output, *signer))
            .copied()
            .unwrap_or(0)
    }

    /// Evict the memo with the lowest LFU access count.
    ///
    /// Returns the evicted `Memo`, or `None` if the store is empty.
    pub fn evict_lfu(&mut self) -> Option<Memo> {
        let min_key = self
            .lfu_counts
            .iter()
            .min_by_key(|(_, count)| **count)
            .map(|(key, _)| *key)?;
        let (input, output, signer_hash) = min_key;
        if let Some(memos) = self.by_input.get_mut(&input) {
            if let Some(idx) = memos
                .iter()
                .position(|m| m.output == output && m.credential.issuer.hash == signer_hash)
            {
                let evicted = memos.remove(idx);
                self.total -= 1;
                self.lfu_counts.remove(&min_key);
                if memos.is_empty() {
                    self.by_input.remove(&input);
                }
                return Some(evicted);
            }
        }
        self.lfu_counts.remove(&min_key);
        None
    }

    /// Serialize LFU counters to bytes using postcard.
    pub fn serialize_lfu_counts(&self) -> Result<Vec<u8>, postcard::Error> {
        let entries: Vec<(MemoKey, u32)> =
            self.lfu_counts.iter().map(|(k, v)| (*k, *v)).collect();
        postcard::to_allocvec(&entries)
    }

    /// Load LFU counters from previously serialized data.
    ///
    /// Only restores counts for keys that already exist in the store (i.e.,
    /// the corresponding memos must have been inserted first). Returns the
    /// number of counters successfully loaded.
    pub fn load_lfu_counts(&mut self, data: &[u8]) -> Result<usize, postcard::Error> {
        let entries: Vec<(MemoKey, u32)> = postcard::from_bytes(data)?;
        let mut loaded = 0;
        for (key, count) in entries {
            if self.lfu_counts.contains_key(&key) {
                self.lfu_counts.insert(key, count);
                loaded += 1;
            }
        }
        Ok(loaded)
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

    #[test]
    fn lfu_counts_increment_on_query() {
        let identity = PqPrivateIdentity::generate(&mut rand::rngs::OsRng);
        let mut store = MemoStore::new();

        let memo = make_memo(&identity, 0x01, 0x02);
        let input = memo.input;
        let output = memo.output;
        let signer_hash = memo.credential.issuer.hash;

        store.insert(memo);
        assert_eq!(store.lfu_count(&input, &output, &signer_hash), 0);

        store.get_by_input(&input);
        assert_eq!(store.lfu_count(&input, &output, &signer_hash), 1);

        store.get_by_input(&input);
        assert_eq!(store.lfu_count(&input, &output, &signer_hash), 2);
    }

    #[test]
    fn evict_lfu_returns_lowest() {
        let alice = PqPrivateIdentity::generate(&mut rand::rngs::OsRng);
        let bob = PqPrivateIdentity::generate(&mut rand::rngs::OsRng);
        let carol = PqPrivateIdentity::generate(&mut rand::rngs::OsRng);
        let mut store = MemoStore::new();

        // Three memos with different inputs so we can query them independently.
        let memo_a = make_memo(&alice, 0x01, 0x10);
        let memo_b = make_memo(&bob, 0x02, 0x20);
        let memo_c = make_memo(&carol, 0x03, 0x30);

        let input_a = memo_a.input;
        let input_b = memo_b.input;
        let output_b = memo_b.output;
        let signer_b = memo_b.credential.issuer.hash;

        store.insert(memo_a);
        store.insert(memo_b);
        store.insert(memo_c);

        // Query a 3 times, b 1 time, c 2 times => b has the lowest count
        store.get_by_input(&input_a);
        store.get_by_input(&input_a);
        store.get_by_input(&input_a);
        store.get_by_input(&input_b);
        let input_c = make_cid(0x03);
        store.get_by_input(&input_c);
        store.get_by_input(&input_c);

        let evicted = store.evict_lfu().expect("should evict one memo");
        assert_eq!(evicted.output, output_b, "memo_b has the lowest count (1)");
        assert_eq!(evicted.credential.issuer.hash, signer_b);
        assert_eq!(store.len(), 2, "two memos remain after eviction");
    }

    #[test]
    fn evict_lfu_empty_returns_none() {
        let mut store = MemoStore::new();
        assert!(store.evict_lfu().is_none(), "eviction on empty store should return None");
    }

    #[test]
    fn lfu_counts_serialize_roundtrip() {
        let identity = PqPrivateIdentity::generate(&mut rand::rngs::OsRng);
        let mut store = MemoStore::new();

        let memo = make_memo(&identity, 0x01, 0x02);
        let input = memo.input;
        let output = memo.output;
        let signer_hash = memo.credential.issuer.hash;

        store.insert(memo);

        // Bump the counter to 3
        store.get_by_input(&input);
        store.get_by_input(&input);
        store.get_by_input(&input);
        assert_eq!(store.lfu_count(&input, &output, &signer_hash), 3);

        // Serialize
        let data = store.serialize_lfu_counts().expect("serialize_lfu_counts");

        // Create a fresh store with the same memo (so the key exists)
        let mut store2 = MemoStore::new();
        let memo2 = make_memo(&identity, 0x01, 0x02);
        store2.insert(memo2);
        assert_eq!(store2.lfu_count(&input, &output, &signer_hash), 0);

        // Load the serialized counts
        let loaded = store2.load_lfu_counts(&data).expect("load_lfu_counts");
        assert_eq!(loaded, 1, "one counter should be loaded");
        assert_eq!(store2.lfu_count(&input, &output, &signer_hash), 3, "count should be restored");
    }
}
