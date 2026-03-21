# Credential Delegation Chains Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add delegation chain support to harmony-credential — a `proof` field linking credentials to parent authorities, with recursive chain verification.

**Architecture:** Single optional `proof: Option<[u8; 32]>` field on `Credential` referencing the parent's BLAKE3 content hash. `verify_chain` recursively verifies every ancestor (time bounds, signature, revocation). Loop detection via fixed-size array (no_std-safe). Max depth 8.

**Tech Stack:** Rust, postcard serialization, harmony-crypto (BLAKE3), harmony-identity (Ed25519/ML-DSA-65 verification), no_std + alloc

---

### File Structure

| File | Responsibility | Change |
|------|---------------|--------|
| `crates/harmony-credential/src/credential.rs` | Credential struct + builder | Add `proof` field to `Credential`, `SignablePayload`, `CredentialBuilder` |
| `crates/harmony-credential/src/error.rs` | Error enum | Add `ChainTooDeep`, `ProofNotFound`, `ChainBroken`, `ChainLoop` variants |
| `crates/harmony-credential/src/verify.rs` | Verification logic + traits | Add `CredentialResolver` trait, `MemoryCredentialResolver`, `verify_chain`, `MAX_CHAIN_DEPTH` |
| `crates/harmony-credential/src/lib.rs` | Crate re-exports | Add `verify_chain`, `CredentialResolver`, `MemoryCredentialResolver` |

---

### Task 1: Add `proof` field to data model

**Files:**
- Modify: `crates/harmony-credential/src/credential.rs`

- [ ] **Step 1: Write failing test for proof field on builder**

Add to the `tests` module in `credential.rs`:

```rust
#[test]
fn proof_field_set_on_credential() {
    let parent_hash = [0xDD; 32];
    let mut builder =
        CredentialBuilder::new(test_issuer(), test_subject(), 1000, 2000, [0x01; 16]);
    builder.proof(parent_hash);
    let payload = builder.signable_payload();
    let (cred, _) = builder.build(payload);
    assert_eq!(cred.proof, Some(parent_hash));
}

#[test]
fn proof_defaults_to_none() {
    let builder = CredentialBuilder::new(test_issuer(), test_subject(), 1000, 2000, [0x01; 16]);
    let payload = builder.signable_payload();
    let (cred, _) = builder.build(payload);
    assert!(cred.proof.is_none());
}

#[test]
fn proof_affects_content_hash() {
    let mut builder_a =
        CredentialBuilder::new(test_issuer(), test_subject(), 1000, 2000, [0x01; 16]);
    builder_a.add_claim(1, alloc::vec![0xAA], [0x11; 16]);
    let payload_a = builder_a.signable_payload();
    let (cred_a, _) = builder_a.build(payload_a);

    let mut builder_b =
        CredentialBuilder::new(test_issuer(), test_subject(), 1000, 2000, [0x01; 16]);
    builder_b.add_claim(1, alloc::vec![0xAA], [0x11; 16]);
    builder_b.proof([0xFF; 32]);
    let payload_b = builder_b.signable_payload();
    let (cred_b, _) = builder_b.build(payload_b);

    assert_ne!(cred_a.content_hash(), cred_b.content_hash());
}

#[test]
fn serde_round_trip_with_proof() {
    let mut builder =
        CredentialBuilder::new(test_issuer(), test_subject(), 1000, 2000, [0x01; 16]);
    builder.add_claim(1, alloc::vec![0xAA], [0x11; 16]);
    builder.proof([0xCC; 32]);
    let payload = builder.signable_payload();
    let (cred, _) = builder.build(payload);

    let bytes = cred.serialize().unwrap();
    let restored = Credential::deserialize(&bytes).unwrap();
    assert_eq!(restored.proof, Some([0xCC; 32]));
    assert_eq!(restored.content_hash(), cred.content_hash());
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-credential credential::tests::proof -- 2>&1`
Expected: FAIL — `proof` field doesn't exist

- [ ] **Step 3: Add `proof` field to `SignablePayload`, `Credential`, and `CredentialBuilder`**

In `credential.rs`, add `proof: Option<[u8; 32]>` to:

1. `SignablePayload` struct (after `nonce`):
```rust
    proof: Option<[u8; 32]>,
```

2. `Credential` struct (after `nonce`, before `signature`):
```rust
    pub proof: Option<[u8; 32]>,
```

3. `CredentialBuilder` struct (after `status_list_index`):
```rust
    proof: Option<[u8; 32]>,
```

4. `CredentialBuilder::new()` — add `proof: None` to the initializer.

5. Add setter method on `CredentialBuilder`:
```rust
    /// Set the proof reference to a parent credential's content hash.
    /// Used for delegation chains.
    pub fn proof(&mut self, parent_hash: [u8; 32]) -> &mut Self {
        self.proof = Some(parent_hash);
        self
    }
```

6. Update `signable_bytes()` on `Credential` to include `proof`:
```rust
    proof: self.proof,
```

7. Update `signable_payload()` on `CredentialBuilder` to include `proof`:
```rust
    proof: self.proof,
```

8. Update `build()` on `CredentialBuilder` to include `proof`:
```rust
    proof: self.proof,
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-credential 2>&1`
Expected: ALL tests pass (including existing tests — the new `proof` field is added to all struct initializers so both old and new code paths construct credentials with the field present)

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-credential/src/credential.rs
git commit -m "feat: add proof field to Credential for delegation chains"
```

---

### Task 2: Add delegation error variants

**Files:**
- Modify: `crates/harmony-credential/src/error.rs`

- [ ] **Step 1: Add 4 new error variants**

Add to `CredentialError` enum (after `DeserializeError`):

```rust
    ChainTooDeep,
    ProofNotFound,
    ChainBroken,
    ChainLoop,
```

Add to `Display` impl match (after `DeserializeError` arm):

```rust
    Self::ChainTooDeep => write!(f, "credential chain exceeds maximum depth (8)"),
    Self::ProofNotFound => write!(f, "parent credential not found in resolver"),
    Self::ChainBroken => write!(f, "parent subject does not match child issuer"),
    Self::ChainLoop => write!(f, "credential chain contains a cycle"),
```

- [ ] **Step 2: Run tests to verify compilation**

Run: `cargo test -p harmony-credential 2>&1`
Expected: ALL pass (no tests reference the new variants yet)

- [ ] **Step 3: Commit**

```bash
git add crates/harmony-credential/src/error.rs
git commit -m "feat: add ChainTooDeep, ProofNotFound, ChainBroken, ChainLoop errors"
```

---

### Task 3: Add `CredentialResolver` trait and `MemoryCredentialResolver`

**Files:**
- Modify: `crates/harmony-credential/src/verify.rs`
- Modify: `crates/harmony-credential/src/lib.rs`

- [ ] **Step 1: Add `CredentialResolver` trait and `MemoryCredentialResolver`**

Add to `verify.rs` (after the `CredentialKeyResolver` trait, before `verify_credential`):

```rust
/// Resolve a credential by its content hash for chain verification.
///
/// Used by `verify_chain` to walk delegation chains. Implementations
/// may fetch from a local cache, database, or network.
pub trait CredentialResolver {
    fn resolve(&self, content_hash: &[u8; 32]) -> Option<Credential>;
}
```

Add the test-only implementation (after `MemoryKeyResolver`):

```rust
/// In-memory credential resolver for testing.
#[cfg(any(test, feature = "test-utils"))]
pub struct MemoryCredentialResolver {
    credentials: hashbrown::HashMap<[u8; 32], Credential>,
}

#[cfg(any(test, feature = "test-utils"))]
impl MemoryCredentialResolver {
    pub fn new() -> Self {
        Self {
            credentials: hashbrown::HashMap::new(),
        }
    }

    pub fn insert(&mut self, credential: Credential) {
        let hash = credential.content_hash();
        self.credentials.insert(hash, credential);
    }
}

#[cfg(any(test, feature = "test-utils"))]
impl CredentialResolver for MemoryCredentialResolver {
    fn resolve(&self, content_hash: &[u8; 32]) -> Option<Credential> {
        self.credentials.get(content_hash).cloned()
    }
}
```

- [ ] **Step 2: Update `lib.rs` re-exports**

In `lib.rs`, replace the existing verify re-export line:
```rust
pub use verify::{verify_credential, verify_presentation, CredentialKeyResolver};
```
with:
```rust
pub use verify::{verify_credential, verify_presentation, CredentialKeyResolver, CredentialResolver};
```

Add after the existing `MemoryKeyResolver` export:
```rust
#[cfg(any(test, feature = "test-utils"))]
pub use verify::MemoryCredentialResolver;
```

Note: `verify_chain` is NOT re-exported yet (it doesn't exist until Task 4). Task 4 will add it.

- [ ] **Step 3: Run tests to verify compilation**

Run: `cargo test -p harmony-credential 2>&1`
Expected: ALL pass

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-credential/src/verify.rs crates/harmony-credential/src/lib.rs
git commit -m "feat: add CredentialResolver trait and MemoryCredentialResolver"
```

---

### Task 4: Implement `verify_chain`

**Files:**
- Modify: `crates/harmony-credential/src/verify.rs`
- Modify: `crates/harmony-credential/src/lib.rs`

- [ ] **Step 1: Write failing tests for chain verification**

Add to the `tests` module in `verify.rs`. First, add a helper function for building signed credentials with delegation:

```rust
    /// Build a signed delegation credential: issuer delegates to subject.
    /// Returns (issuer_private, credential, issuer_ref, subject_ref).
    fn build_delegation(
        issuer_priv: &harmony_identity::PrivateIdentity,
        subject_ref: IdentityRef,
        proof: Option<[u8; 32]>,
    ) -> Credential {
        let issuer_ref = IdentityRef::from(issuer_priv.public_identity());
        let mut builder = CredentialBuilder::new(
            issuer_ref, subject_ref, 1000, 5000, [0x30; 16],
        );
        builder.add_claim(1, alloc::vec![0x01], [0xA1; 16]);
        if let Some(hash) = proof {
            builder.proof(hash);
        }
        let payload = builder.signable_payload();
        let signature = issuer_priv.sign(&payload);
        let (cred, _) = builder.build(signature.to_vec());
        cred
    }

    fn setup_chain_resolvers(
        identities: &[(
            &harmony_identity::PrivateIdentity,
            &IdentityRef,
        )],
        credentials: &[&Credential],
    ) -> (MemoryKeyResolver, MemoryStatusListResolver, MemoryCredentialResolver) {
        let mut keys = MemoryKeyResolver::new();
        for (priv_id, id_ref) in identities {
            keys.insert(
                id_ref.hash,
                priv_id.public_identity().verifying_key.to_bytes().to_vec(),
            );
        }
        let status = MemoryStatusListResolver::new();
        let mut creds = MemoryCredentialResolver::new();
        for cred in credentials {
            creds.insert((*cred).clone());
        }
        (keys, status, creds)
    }
```

Then add the test functions:

```rust
    // ---- verify_chain tests ----

    #[test]
    fn chain_root_credential_no_proof() {
        let (private, cred, _, issuer_ref) = build_signed_credential();
        let resolver = setup_resolver(private.public_identity(), &issuer_ref);
        let status = empty_status();
        let creds = MemoryCredentialResolver::new();
        assert!(verify_chain(&cred, 1500, &resolver, &status, &creds).is_ok());
    }

    #[test]
    fn chain_valid_two_level() {
        let gov_priv = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let uni_priv = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let gov_ref = IdentityRef::from(gov_priv.public_identity());
        let uni_ref = IdentityRef::from(uni_priv.public_identity());
        let student_ref = IdentityRef::new([0xCC; 16], CryptoSuite::Ed25519);

        // Government delegates to University (root credential)
        let delegation = build_delegation(&gov_priv, uni_ref, None);
        let delegation_hash = delegation.content_hash();

        // University issues degree to Student (child credential)
        let degree = build_delegation(&uni_priv, student_ref, Some(delegation_hash));

        let (keys, status, creds) = setup_chain_resolvers(
            &[(&gov_priv, &gov_ref), (&uni_priv, &uni_ref)],
            &[&delegation],
        );

        assert!(verify_chain(&degree, 1500, &keys, &status, &creds).is_ok());
    }

    #[test]
    fn chain_valid_three_level() {
        let gov_priv = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let uni_priv = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let dept_priv = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let gov_ref = IdentityRef::from(gov_priv.public_identity());
        let uni_ref = IdentityRef::from(uni_priv.public_identity());
        let dept_ref = IdentityRef::from(dept_priv.public_identity());
        let student_ref = IdentityRef::new([0xCC; 16], CryptoSuite::Ed25519);

        let gov_cred = build_delegation(&gov_priv, uni_ref, None);
        let uni_cred = build_delegation(&uni_priv, dept_ref, Some(gov_cred.content_hash()));
        let dept_cred = build_delegation(&dept_priv, student_ref, Some(uni_cred.content_hash()));

        let (keys, status, creds) = setup_chain_resolvers(
            &[(&gov_priv, &gov_ref), (&uni_priv, &uni_ref), (&dept_priv, &dept_ref)],
            &[&gov_cred, &uni_cred],
        );

        assert!(verify_chain(&dept_cred, 1500, &keys, &status, &creds).is_ok());
    }

    #[test]
    fn chain_revoked_ancestor_fails() {
        let gov_priv = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let uni_priv = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let gov_ref = IdentityRef::from(gov_priv.public_identity());
        let uni_ref = IdentityRef::from(uni_priv.public_identity());
        let student_ref = IdentityRef::new([0xCC; 16], CryptoSuite::Ed25519);

        // Government delegates to University with revocation index
        let mut builder = CredentialBuilder::new(gov_ref, uni_ref, 1000, 5000, [0x30; 16]);
        builder.add_claim(1, alloc::vec![0x01], [0xA1; 16]);
        builder.status_list_index(0);
        let payload = builder.signable_payload();
        let delegation = {
            let sig = gov_priv.sign(&payload);
            let (cred, _) = builder.build(sig.to_vec());
            cred
        };
        let delegation_hash = delegation.content_hash();

        let degree = build_delegation(&uni_priv, student_ref, Some(delegation_hash));

        let mut keys = MemoryKeyResolver::new();
        keys.insert(gov_ref.hash, gov_priv.public_identity().verifying_key.to_bytes().to_vec());
        keys.insert(uni_ref.hash, uni_priv.public_identity().verifying_key.to_bytes().to_vec());

        // Revoke the delegation
        let mut status = MemoryStatusListResolver::new();
        let mut list = StatusList::new(128);
        list.revoke(0).unwrap();
        status.insert(gov_ref.hash, list);

        let mut creds = MemoryCredentialResolver::new();
        creds.insert(delegation);

        assert_eq!(
            verify_chain(&degree, 1500, &keys, &status, &creds).unwrap_err(),
            CredentialError::Revoked
        );
    }

    #[test]
    fn chain_broken_subject_issuer_mismatch() {
        let gov_priv = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let uni_priv = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let gov_ref = IdentityRef::from(gov_priv.public_identity());
        let uni_ref = IdentityRef::from(uni_priv.public_identity());
        let student_ref = IdentityRef::new([0xCC; 16], CryptoSuite::Ed25519);

        // Government delegates to a DIFFERENT identity (not University)
        let wrong_ref = IdentityRef::new([0xEE; 16], CryptoSuite::Ed25519);
        let delegation = build_delegation(&gov_priv, wrong_ref, None);
        let delegation_hash = delegation.content_hash();

        // University tries to use this delegation
        let degree = build_delegation(&uni_priv, student_ref, Some(delegation_hash));

        let (keys, status, creds) = setup_chain_resolvers(
            &[(&gov_priv, &gov_ref), (&uni_priv, &uni_ref)],
            &[&delegation],
        );

        assert_eq!(
            verify_chain(&degree, 1500, &keys, &status, &creds).unwrap_err(),
            CredentialError::ChainBroken
        );
    }

    #[test]
    fn chain_proof_not_found() {
        let uni_priv = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let uni_ref = IdentityRef::from(uni_priv.public_identity());
        let student_ref = IdentityRef::new([0xCC; 16], CryptoSuite::Ed25519);

        let degree = build_delegation(&uni_priv, student_ref, Some([0xFF; 32]));

        let mut keys = MemoryKeyResolver::new();
        keys.insert(uni_ref.hash, uni_priv.public_identity().verifying_key.to_bytes().to_vec());
        let status = empty_status();
        let creds = MemoryCredentialResolver::new(); // empty — parent not found

        assert_eq!(
            verify_chain(&degree, 1500, &keys, &status, &creds).unwrap_err(),
            CredentialError::ProofNotFound
        );
    }

    #[test]
    fn chain_too_deep() {
        // Build a chain of depth 9 (exceeds MAX_CHAIN_DEPTH = 8)
        let mut privates: Vec<harmony_identity::PrivateIdentity> = (0..10)
            .map(|_| harmony_identity::PrivateIdentity::generate(&mut OsRng))
            .collect();
        let refs: Vec<IdentityRef> = privates
            .iter()
            .map(|p| IdentityRef::from(p.public_identity()))
            .collect();

        let mut keys = MemoryKeyResolver::new();
        for (i, p) in privates.iter().enumerate() {
            keys.insert(refs[i].hash, p.public_identity().verifying_key.to_bytes().to_vec());
        }

        let mut creds = MemoryCredentialResolver::new();
        let mut prev_hash: Option<[u8; 32]> = None;

        // Build 9 delegation links (0→1, 1→2, ..., 8→9)
        for i in 0..9 {
            let cred = build_delegation(&privates[i], refs[i + 1], prev_hash);
            prev_hash = Some(cred.content_hash());
            creds.insert(cred);
        }

        // Leaf credential at depth 9
        let leaf_subject = IdentityRef::new([0xCC; 16], CryptoSuite::Ed25519);
        let leaf = build_delegation(&privates[9], leaf_subject, prev_hash);

        let status = empty_status();
        assert_eq!(
            verify_chain(&leaf, 1500, &keys, &status, &creds).unwrap_err(),
            CredentialError::ChainTooDeep
        );
    }

    #[test]
    fn chain_loop_detected() {
        // Content hashes make honest cycles impossible (chicken-and-egg),
        // so we use a custom resolver that deliberately returns a
        // credential whose proof points back to the leaf — simulating
        // what a malicious or buggy resolver might do.
        let alice_priv = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let bob_priv = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let alice_ref = IdentityRef::from(alice_priv.public_identity());
        let bob_ref = IdentityRef::from(bob_priv.public_identity());

        // Build the leaf credential (Alice → Bob, proof = some hash)
        let fake_parent_hash = [0xAA; 32];
        let leaf = build_delegation(&alice_priv, bob_ref, Some(fake_parent_hash));
        let leaf_hash = leaf.content_hash();

        // Build a "parent" whose proof points back to the leaf's hash
        // (creating a cycle: leaf → parent → leaf)
        let parent = build_delegation(&bob_priv, alice_ref, Some(leaf_hash));

        // Custom resolver that maps fake_parent_hash → parent
        // (regardless of actual content hash)
        struct CyclicResolver(Credential);
        impl CredentialResolver for CyclicResolver {
            fn resolve(&self, _content_hash: &[u8; 32]) -> Option<Credential> {
                Some(self.0.clone())
            }
        }

        let mut keys = MemoryKeyResolver::new();
        keys.insert(alice_ref.hash, alice_priv.public_identity().verifying_key.to_bytes().to_vec());
        keys.insert(bob_ref.hash, bob_priv.public_identity().verifying_key.to_bytes().to_vec());
        let status = empty_status();

        let result = verify_chain(&leaf, 1500, &keys, &status, &CyclicResolver(parent));
        assert_eq!(result.unwrap_err(), CredentialError::ChainLoop);
    }

    #[test]
    fn chain_valid_two_level_ml_dsa65() {
        std::thread::Builder::new()
            .stack_size(8 * 1024 * 1024)
            .spawn(|| {
                let (gov_sign_pk, gov_sign_sk) = harmony_crypto::ml_dsa::generate(&mut OsRng);
                let (gov_enc_pk, _) = harmony_crypto::ml_kem::generate(&mut OsRng);
                let gov_pq = harmony_identity::PqIdentity::from_public_keys(gov_enc_pk, gov_sign_pk.clone());
                let gov_ref = IdentityRef::from(&gov_pq);

                let (uni_sign_pk, uni_sign_sk) = harmony_crypto::ml_dsa::generate(&mut OsRng);
                let (uni_enc_pk, _) = harmony_crypto::ml_kem::generate(&mut OsRng);
                let uni_pq = harmony_identity::PqIdentity::from_public_keys(uni_enc_pk, uni_sign_pk.clone());
                let uni_ref = IdentityRef::from(&uni_pq);

                let student_ref = IdentityRef::new([0xCC; 16], CryptoSuite::MlDsa65);

                // Government delegates to University
                let mut builder = CredentialBuilder::new(gov_ref, uni_ref, 1000, 5000, [0x30; 16]);
                builder.add_claim(1, alloc::vec![0x01], [0xA1; 16]);
                let payload = builder.signable_payload();
                let sig = harmony_crypto::ml_dsa::sign(&gov_sign_sk, &payload).unwrap();
                let (delegation, _) = builder.build(sig.as_bytes().to_vec());
                let delegation_hash = delegation.content_hash();

                // University issues degree to Student
                let mut builder2 = CredentialBuilder::new(uni_ref, student_ref, 1000, 5000, [0x31; 16]);
                builder2.add_claim(1, alloc::vec![0x02], [0xA2; 16]);
                builder2.proof(delegation_hash);
                let payload2 = builder2.signable_payload();
                let sig2 = harmony_crypto::ml_dsa::sign(&uni_sign_sk, &payload2).unwrap();
                let (degree, _) = builder2.build(sig2.as_bytes().to_vec());

                let mut keys = MemoryKeyResolver::new();
                keys.insert(gov_ref.hash, gov_sign_pk.as_bytes());
                keys.insert(uni_ref.hash, uni_sign_pk.as_bytes());
                let status = MemoryStatusListResolver::new();
                let mut creds = MemoryCredentialResolver::new();
                creds.insert(delegation);

                assert!(verify_chain(&degree, 1500, &keys, &status, &creds).is_ok());
            })
            .expect("spawn")
            .join()
            .expect("join");
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-credential verify::tests::chain -- 2>&1`
Expected: FAIL — `verify_chain` doesn't exist, `MemoryCredentialResolver` doesn't exist

- [ ] **Step 3: Implement `verify_chain`**

Add to `verify.rs` (after `verify_presentation`, before `verify_signature`):

```rust
/// Maximum delegation chain depth. Chains deeper than this are rejected.
pub const MAX_CHAIN_DEPTH: usize = 8;

/// Verify a credential and its full delegation chain.
///
/// Recursively verifies every ancestor credential — time bounds,
/// signature, and revocation status. If any ancestor is expired,
/// revoked, or invalid, the entire chain fails.
///
/// For root credentials (no `proof` field), this behaves identically
/// to `verify_credential`.
pub fn verify_chain(
    credential: &Credential,
    now: u64,
    keys: &impl CredentialKeyResolver,
    status_lists: &impl StatusListResolver,
    credentials: &impl CredentialResolver,
) -> Result<(), CredentialError> {
    // Seed the seen set with the leaf's content hash for loop detection.
    let mut seen = [[0u8; 32]; MAX_CHAIN_DEPTH + 1];
    seen[0] = credential.content_hash();
    let mut seen_len = 1;

    verify_chain_inner(credential, now, keys, status_lists, credentials, &mut seen, &mut seen_len, 0)
}

fn verify_chain_inner(
    credential: &Credential,
    now: u64,
    keys: &impl CredentialKeyResolver,
    status_lists: &impl StatusListResolver,
    credentials: &impl CredentialResolver,
    seen: &mut [[u8; 32]; MAX_CHAIN_DEPTH + 1],
    seen_len: &mut usize,
    depth: usize,
) -> Result<(), CredentialError> {
    // Verify this credential (time bounds, signature, revocation)
    verify_credential(credential, now, keys, status_lists)?;

    // Walk the chain if there's a proof reference
    if let Some(parent_hash) = credential.proof {
        // Loop detection: check if we've seen this hash before
        if seen[..*seen_len].iter().any(|h| *h == parent_hash) {
            return Err(CredentialError::ChainLoop);
        }

        // Depth check
        if depth + 1 > MAX_CHAIN_DEPTH {
            return Err(CredentialError::ChainTooDeep);
        }

        // Record this hash
        seen[*seen_len] = parent_hash;
        *seen_len += 1;

        // Resolve the parent credential
        let parent = credentials
            .resolve(&parent_hash)
            .ok_or(CredentialError::ProofNotFound)?;

        // Chain link invariant: parent.subject == child.issuer
        if parent.subject != credential.issuer {
            return Err(CredentialError::ChainBroken);
        }

        // Recursively verify the parent
        verify_chain_inner(&parent, now, keys, status_lists, credentials, seen, seen_len, depth + 1)?;
    }

    Ok(())
}
```

- [ ] **Step 4: Update `lib.rs` to re-export `verify_chain`**

Replace the verify re-export line:
```rust
pub use verify::{verify_credential, verify_presentation, CredentialKeyResolver, CredentialResolver};
```
with:
```rust
pub use verify::{verify_chain, verify_credential, verify_presentation, CredentialKeyResolver, CredentialResolver};
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test -p harmony-credential 2>&1`
Expected: ALL tests pass (existing + new chain tests)

- [ ] **Step 6: Run full workspace check**

Run: `cargo test --workspace --exclude harmony-tunnel 2>&1 | grep "^test result:" | head -30`
Expected: All test suites pass

- [ ] **Step 7: Commit**

```bash
git add crates/harmony-credential/src/verify.rs crates/harmony-credential/src/lib.rs
git commit -m "feat: implement verify_chain for credential delegation chains

Recursive chain verification walks proof references, checking time
bounds, signatures, and revocation at every level. Loop detection
via fixed-size array (no_std-safe). Max depth 8.

Closes harmony-2fs"
```
