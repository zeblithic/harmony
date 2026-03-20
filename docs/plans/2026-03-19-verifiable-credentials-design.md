# Verifiable Credentials Design

**Date:** 2026-03-19
**Status:** Approved
**Scope:** New `harmony-credential` crate
**Bead:** harmony-44r

## Overview

Harmony needs a way for identities to make verifiable claims about
themselves or others — government IDs, organization membership, AI agent
licenses, age verification, etc. UCANs handle *capability delegation*
("you can do X"); credentials handle *identity claims* ("you are X").

This design introduces a compact binary credential system inspired by
W3C Verifiable Credentials and SD-JWT, adapted for Harmony's no_std +
postcard ecosystem. Key features:

1. **Schema-free claims** — opaque `(type_id, value)` pairs; the
   credential crate signs and discloses them without interpreting contents
2. **Selective disclosure** — hash-and-reveal with BLAKE3 salted digests;
   holders reveal only the claims they choose
3. **Bitstring revocation** — issuer-managed bitfield for compact,
   efficient credential revocation
4. **Any-identity issuance** — Ed25519, ML-DSA-65, or rotatable (KEL)
   identities can all issue; `CryptoSuite` tag enables consumer policy

W3C JSON-LD interop is explicitly out of scope — filed as a future bead.

## Design Philosophy

### Cryptographic Envelope, Not Claim Semantics

The credential crate provides the cryptographic envelope: issue, sign,
selectively disclose, revoke, verify. It does not interpret what claims
*mean*. Claim semantics are a consumer concern — application code or
`harmony-trust` policy can interpret `type_id` values and enforce rules
like "only accept age claims from issuers with Identity dimension >= 2."

### Sans-I/O

Following the established Harmony pattern, all operations are pure
functions. Signing is the caller's responsibility (builder produces
signable payload, caller signs externally). Verification uses trait-based
resolution for identity lookup and revocation checking. No I/O, no
async, no runtime coupling.

### Strict UCAN Separation

- **UCANs** = capability delegation ("you can do X")
- **Credentials** = identity claims ("you are X" / "you have property X")

These are complementary systems with no overlap. A UCAN grants permission
to act; a credential proves an attribute. They share infrastructure
(`IdentityRef`, `IdentityResolver`, `CryptoSuite`) but have distinct
semantics and verification flows.

## Claim Model

Claims are schema-free binary pairs. The credential crate does not
define or enforce claim type semantics — consumers do.

```rust
/// A single claim: opaque type ID + opaque value.
pub struct Claim {
    pub type_id: u16,       // application-defined claim type
    pub value: Vec<u8>,     // application-defined payload
}
```

### Salted Claims for Selective Disclosure

Each claim is paired with a random salt before hashing. The signed
credential stores only the digests — never plaintext claims.

```rust
/// A claim prepared for selective disclosure.
pub struct SaltedClaim {
    pub claim: Claim,
    pub salt: [u8; 16],     // random per-claim
}

impl SaltedClaim {
    /// BLAKE3(salt || type_id.to_le_bytes() || value)
    pub fn digest(&self) -> [u8; 32] { ... }
}
```

The `type_id` is included in the hash input so that a claim's type
cannot be reinterpreted — revealing `(type_id=1, value=0x12)` only
matches the digest if it was originally issued as type 1.

### Selective Disclosure Flow

1. **Issuance:** Issuer creates `SaltedClaim`s, computes digests, signs
   credential (contains only digests)
2. **Holding:** Subject stores `SaltedClaim`s alongside the credential
3. **Presentation:** Subject selects which claims to reveal, sends
   credential + selected `SaltedClaim`s as a `Presentation`
4. **Verification:** Verifier recomputes digests from revealed claims,
   checks they match the signed digests

The verifier can see the total number of claims (via `claim_digests.len()`)
but not the content of undisclosed claims.

## Credential Structure

```rust
pub struct Credential {
    pub issuer: IdentityRef,            // who signed this (17 bytes)
    pub subject: IdentityRef,           // who this is about (17 bytes)
    pub claim_digests: Vec<[u8; 32]>,   // BLAKE3 hashes of salted claims
    pub status_list_index: Option<u32>, // revocation index (if revocable)
    pub not_before: u64,                // unix timestamp
    pub expires_at: u64,                // unix timestamp
    pub issued_at: u64,                 // for KEL key validity lookups
    pub nonce: [u8; 16],                // replay protection
    pub signature: Vec<u8>,             // Ed25519 (64B) or ML-DSA-65 (3309B)
}
```

### Field Rationale

- **`issuer` / `subject`** — `IdentityRef` carries both address hash and
  `CryptoSuite` tag. The suite tells the verifier which signature
  algorithm to use.

- **`issued_at`** — needed for rotatable issuers (KEL-backed). The
  verifier must determine which signing key was active at issuance time.

- **`status_list_index`** — `Option` because not all credentials need
  revocation. Short-lived credentials may rely solely on expiry.

- **`nonce`** — 16 random bytes for replay protection and correlation
  resistance.

- **`signature`** — variable-length to support Ed25519 (64 bytes) and
  ML-DSA-65 (3309 bytes). The `issuer.suite` disambiguates.

### Signable Payload

Everything except `signature`, serialized with postcard. The issuer signs
this payload; the verifier reconstructs and checks the signature.

### Serialization

Postcard with format version byte prefix (same pattern as `KeyEventLog`
and `TrustStore`).

## Presentation

A credential plus selectively disclosed claims, sent from holder to
verifier.

```rust
pub struct Presentation {
    pub credential: Credential,
    pub disclosed_claims: Vec<SaltedClaim>,
}
```

Verification checks that each disclosed claim's digest appears in
`credential.claim_digests`. No duplicate disclosures are allowed.

## Bitstring Status List

Each issuer manages their own revocation bitfield. A credential
references its index in the issuer's status list.

```rust
pub struct StatusList {
    bits: Vec<u8>,  // packed bitfield; bit 0 = valid, bit 1 = revoked
}

impl StatusList {
    pub fn new(capacity: u32) -> Self { ... }  // default 16384 (2KB)
    pub fn is_revoked(&self, index: u32) -> bool { ... }
    pub fn revoke(&mut self, index: u32) { ... }
    pub fn capacity(&self) -> u32 { ... }
}
```

### Resolution Trait

```rust
pub trait StatusListResolver {
    fn resolve(&self, issuer: &IdentityRef) -> Option<&StatusList>;
}
```

How the status list is distributed (announced, served on request) is a
transport concern outside this crate's scope.

### Default Size

2KB (16,384 credential slots). Configurable at creation time via
`StatusList::new(capacity)`. Sufficient for individual issuers in a
decentralized mesh.

## Verification

### Credential Verification

```rust
pub fn verify_credential(
    credential: &Credential,
    now: u64,
    identities: &impl IdentityResolver,
    status_lists: &impl StatusListResolver,
) -> Result<(), CredentialError>
```

**Steps, in order:**
1. **Time bounds** — `not_before <= now < expires_at`
2. **Issuer resolution** — look up issuer's public key via `IdentityResolver`
3. **Signature verification** — reconstruct signable payload, verify
   against issuer's key using algorithm from `issuer.suite`
4. **Revocation check** — if `status_list_index` is `Some(idx)`, resolve
   issuer's `StatusList` and check `is_revoked(idx)`

### Presentation Verification

```rust
pub fn verify_presentation(
    presentation: &Presentation,
    now: u64,
    identities: &impl IdentityResolver,
    status_lists: &impl StatusListResolver,
) -> Result<(), CredentialError>
```

**Additional steps beyond `verify_credential`:**
5. **Disclosure integrity** — each disclosed claim's digest must appear
   in `credential.claim_digests`
6. **No duplicate disclosures** — each digest matched at most once

### Error Handling

```rust
pub enum CredentialError {
    NotYetValid,
    Expired,
    IssuerNotFound,
    SignatureInvalid,
    Revoked,
    StatusListNotFound,
    DisclosureMismatch,
    DuplicateDisclosure,
    SerializeError(&'static str),
    DeserializeError(&'static str),
}
```

## Issuance API

Sans-I/O builder pattern. The crate produces the signable payload; the
caller (who has the private key) signs externally.

```rust
pub struct CredentialBuilder {
    issuer: IdentityRef,
    subject: IdentityRef,
    issued_at: u64,
    claims: Vec<SaltedClaim>,
    status_list_index: Option<u32>,
    not_before: u64,
    expires_at: u64,
    nonce: [u8; 16],
}

impl CredentialBuilder {
    pub fn new(issuer: IdentityRef, subject: IdentityRef, issued_at: u64) -> Self { ... }
    pub fn add_claim(&mut self, type_id: u16, value: Vec<u8>, salt: [u8; 16]) -> &mut Self { ... }
    pub fn status_list_index(&mut self, index: u32) -> &mut Self { ... }
    pub fn time_bounds(&mut self, not_before: u64, expires_at: u64) -> &mut Self { ... }
    pub fn nonce(&mut self, nonce: [u8; 16]) -> &mut Self { ... }

    /// Produce the signable payload bytes.
    pub fn signable_payload(&self) -> Vec<u8> { ... }

    /// Finalize with a signature to produce Credential + SaltedClaims.
    /// The SaltedClaims should be stored by the holder for later disclosure.
    pub fn build(self, signature: Vec<u8>) -> (Credential, Vec<SaltedClaim>) { ... }
}
```

## Crate Structure

```
harmony-credential/
  Cargo.toml
  src/
    lib.rs              — no_std setup, re-exports
    claim.rs            — Claim, SaltedClaim, digest computation
    credential.rs       — Credential struct, CredentialBuilder, serialization
    disclosure.rs       — Presentation struct
    status_list.rs      — StatusList, StatusListResolver trait
    verify.rs           — verify_credential(), verify_presentation()
    error.rs            — CredentialError enum
```

### Dependencies

```toml
[dependencies]
harmony-identity = { path = "../harmony-identity", default-features = false }
harmony-crypto = { path = "../harmony-crypto", default-features = false, features = ["serde"] }
serde = { workspace = true, default-features = false, features = ["derive", "alloc"] }
postcard = { workspace = true }
hashbrown = { workspace = true }  # for test-utils HashMap

[dev-dependencies]
postcard = { workspace = true }
rand = { workspace = true }

[features]
default = ["std"]
std = ["harmony-identity/std", "harmony-crypto/std", "serde/std", "postcard/use-std"]
test-utils = ["hashbrown/serde"]
```

### Re-exports (lib.rs)

```rust
pub use claim::{Claim, SaltedClaim};
pub use credential::{Credential, CredentialBuilder};
pub use disclosure::Presentation;
pub use error::CredentialError;
pub use status_list::{StatusList, StatusListResolver};
pub use verify::{verify_credential, verify_presentation};

#[cfg(any(test, feature = "test-utils"))]
pub use status_list::MemoryStatusListResolver;
```

## Changes to Existing Crates

None. This is a new additive crate. All dependencies (`IdentityRef`,
`IdentityResolver`, `CryptoSuite`, BLAKE3, ML-DSA-65) already exist.

## Testing Strategy

### Claim tests
- `SaltedClaim::digest()` is deterministic for same inputs
- Different salts produce different digests
- Different type_ids with same value produce different digests

### Credential tests
- Builder produces valid credential with correct digest count
- `signable_payload()` is deterministic (same inputs → same bytes)
- Serde round-trip preserves all fields

### Presentation tests
- Disclosing all claims succeeds
- Disclosing subset succeeds
- Disclosing a tampered claim fails (digest mismatch)
- Disclosing a claim not in the credential fails
- Duplicate disclosures rejected

### Verification tests
- Valid credential passes all checks
- Expired credential rejected
- Not-yet-valid credential rejected
- Unknown issuer rejected
- Tampered signature rejected
- Revoked credential rejected
- Missing status list for revocable credential rejected

### StatusList tests
- New list has all bits unset
- `revoke(idx)` sets the bit
- `is_revoked(idx)` returns correct state
- Out-of-bounds index handling
- Serde round-trip preserves bit state
- Custom capacity works

### Integration tests
- Full flow: build credential → create presentation → verify presentation
- Both Ed25519 and ML-DSA-65 issuer paths
- Revocation after issuance

## Future Beads

- **W3C JSON-LD bridge** — optional `std` feature for JSON-LD
  serialization, DID-based identifiers, external VC wallet interop
- **Credential delegation chains** — optional `proof` field referencing
  parent credentials, chain verification, attenuation rules
