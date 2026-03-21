# Credential Delegation Chains Design

**Date:** 2026-03-20
**Status:** Approved
**Scope:** Add delegation chain support to harmony-credential
**Bead:** harmony-2fs

## Overview

Credential delegation allows an issuer to authorize another identity to
issue credentials on their behalf. This creates verifiable chains of
authority: a government authorizes a university, the university issues a
degree to a student, and a verifier can walk the chain to confirm the
student's degree traces back to the government's authority.

Delegation is implemented as a single optional field on `Credential`
referencing the parent credential's BLAKE3 content hash. Chain
verification recursively validates every ancestor — time bounds,
signatures, and revocation status — so revoking a delegation
invalidates all downstream credentials.

## Data Model

### New Field

`Credential` and `SignablePayload` gain one optional field:

```rust
pub proof: Option<[u8; 32]>,  // BLAKE3 content_hash of parent credential
```

- `Some(hash)` — this credential was issued under delegated authority.
  The hash references the parent credential that authorized this issuer.
- `None` — root credential, issued by a self-sovereign authority.

The `proof` field is included in `SignablePayload` and covered by the
signature. It cannot be added or removed after issuance.

### Builder Change

`CredentialBuilder` gains a `.proof(hash: [u8; 32])` setter, following
the existing `.status_list_index()` pattern:

```rust
pub fn proof(&mut self, parent_hash: [u8; 32]) -> &mut Self {
    self.proof = Some(parent_hash);
    self
}
```

### Delegation Semantics

A delegation credential is identity-scoped: the parent credential's
subject is authorized to issue credentials. There is no claim-type
restriction — the parent says "identity X is an authorized issuer,"
not "identity X can issue degree claims but not citizenship claims."

This keeps the model simple. Finer-grained authorization (which claim
types an issuer may issue) can be layered via the endorsement system
or a future attenuation field without breaking the chain format.

### Chain Link Invariant

For a valid chain link, the **parent credential's subject** must equal
the **child credential's issuer**. This is the delegation handoff: the
parent credential attests that its subject is authorized, and the child
credential is issued by that same identity.

```
Government (issuer) ──signs──→ Credential A (subject: University)
University (issuer) ──signs──→ Credential B (subject: Student, proof: hash(A))
```

Verifying Credential B checks that A.subject == B.issuer (University).

## Chain Verification

### New Public Function

```rust
pub fn verify_chain(
    credential: &Credential,
    now: u64,
    keys: &impl CredentialKeyResolver,
    status_lists: &impl StatusListResolver,
    credentials: &impl CredentialResolver,
) -> Result<(), CredentialError>
```

### Algorithm

1. Call `verify_credential` on the leaf (full check: time bounds,
   signature, revocation).
2. If `credential.proof` is `Some(parent_hash)`:
   a. Check `parent_hash` against a `seen` set — if present, return
      `ChainLoop`.
   b. Add `parent_hash` to `seen`.
   c. Increment depth counter — if depth > `MAX_CHAIN_DEPTH` (8),
      return `ChainTooDeep`.
   d. Resolve the parent via `credentials.resolve(&parent_hash)` —
      if `None`, return `ProofNotFound`.
   e. Verify parent's subject matches child's issuer — if mismatch,
      return `ChainBroken`.
   f. Recursively verify the parent (full check + chain walk).
3. If `credential.proof` is `None`, this is the root — done.

### Loop Detection

A `HashSet<[u8; 32]>` tracks content hashes seen during the walk. If
a hash appears twice, return `ChainLoop`. This is O(n) in chain depth
with negligible cost since chains are at most 8 deep.

### Max Chain Depth

`MAX_CHAIN_DEPTH = 8` — a hardcoded constant. Realistic delegation
chains are 2-4 levels deep. 8 is generous enough for any legitimate
use case while preventing abuse.

### Backward Compatibility

`verify_credential` is unchanged. Callers who don't use delegation
chains keep working exactly as before. `verify_chain` on a root
credential (no proof) behaves identically to `verify_credential`.

## New Trait

```rust
pub trait CredentialResolver {
    fn resolve(&self, content_hash: &[u8; 32]) -> Option<Credential>;
}
```

Follows the existing trait pattern (`CredentialKeyResolver`,
`StatusListResolver`, `ProfileKeyResolver`). Verifiers provide an
implementation backed by local storage, a cache, or network fetch.

`MemoryCredentialResolver` is provided behind the `test-utils` feature
for testing, following the `MemoryKeyResolver` pattern:

```rust
#[cfg(any(test, feature = "test-utils"))]
pub struct MemoryCredentialResolver {
    credentials: HashMap<[u8; 32], Credential>,
}
```

## New Error Variants

```rust
pub enum CredentialError {
    // ... existing variants ...
    ChainTooDeep,
    ProofNotFound,
    ChainBroken,
    ChainLoop,
}
```

## Testing Strategy

1. **Root credential (no proof)** — `verify_chain` behaves identically
   to `verify_credential`.

2. **Valid 2-level chain** — Government delegates to University,
   University issues degree to Student. Full chain verifies.

3. **Valid 3-level chain** — Government → University → Department →
   Student. Exercises depth > 2.

4. **Revoked ancestor** — Government revokes University's delegation.
   Student's credential fails with `Revoked`.

5. **Broken chain** — Parent subject doesn't match child issuer. Fails
   with `ChainBroken`.

6. **Error cases** — `ProofNotFound` (missing parent), `ChainTooDeep`
   (depth > 8), `ChainLoop` (cycle).

All tests use Ed25519 for speed, with one ML-DSA-65 chain test for
PQ coverage.

## What This Bead Delivers

- `proof: Option<[u8; 32]>` field on `Credential` and `SignablePayload`
- `.proof()` setter on `CredentialBuilder`
- `CredentialResolver` trait + `MemoryCredentialResolver`
- `verify_chain` function
- `ChainTooDeep`, `ProofNotFound`, `ChainBroken`, `ChainLoop` errors
- `MAX_CHAIN_DEPTH` constant (8)
- ~150-200 lines of production code, ~300 lines of tests
- No new crate, no breaking changes
