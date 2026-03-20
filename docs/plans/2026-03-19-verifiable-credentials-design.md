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

W3C JSON-LD interop is explicitly out of scope — filed as future bead
harmony-8ws.

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
resolution for key lookup and revocation checking. No I/O, no async, no
runtime coupling.

### Strict UCAN Separation

- **UCANs** = capability delegation ("you can do X")
- **Credentials** = identity claims ("you are X" / "you have property X")

These are complementary systems with no overlap. A UCAN grants permission
to act; a credential proves an attribute. They share infrastructure
(`IdentityRef`, `CryptoSuite`) but have distinct semantics and
verification flows.

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
    pub issuer: IdentityRef,            // who signed this
    pub subject: IdentityRef,           // who this is about
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
  verifier passes this to the key resolver so it can determine which
  signing key was active at issuance time. See Key Resolution below.

- **`status_list_index`** — `Option` because not all credentials need
  revocation. Short-lived credentials may rely solely on expiry.

- **`nonce`** — 16 random bytes for replay protection and correlation
  resistance.

- **`signature`** — variable-length to support Ed25519 (64 bytes) and
  ML-DSA-65 (3309 bytes). The `issuer.suite` disambiguates.

### Self-Issued Credentials

Credentials where `issuer == subject` are allowed with no special
handling. This supports self-sovereign identity claims (e.g., an
identity asserting its own metadata). Verification follows the same
path — the signature is still checked against the issuer's key.

### Content Hash

```rust
impl Credential {
    /// BLAKE3 hash of the signable payload. Stable identifier for this
    /// credential, usable for indexing, deduplication, and future
    /// delegation chain references.
    pub fn content_hash(&self) -> [u8; 32] { ... }
}
```

### Signable Payload

Everything except `signature`, serialized canonically with postcard.
The issuer signs this payload; the verifier reconstructs and checks
the signature.

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

### Known Limitation: Presentation Replay

The `Presentation` struct has no verifier binding or challenge-response
nonce. A presentation intercepted in transit could be replayed to a
different verifier. The credential's `nonce` is issuer-set, not
verifier-set. This is acceptable for V1 — adding a verifier challenge
field is a straightforward future enhancement if needed.

## Key Resolution

### The Problem

The existing `IdentityResolver` trait in `harmony-identity` returns
`Option<Identity>`, where `Identity` is Ed25519-only. Credential
verification must support all three crypto suites: Ed25519, ML-DSA-65,
and ML-DSA-65 with key rotation (KEL).

### Solution: Credential-Specific Resolver Trait

Rather than modifying `harmony-identity`, the credential crate defines
its own resolver trait that returns raw public key bytes for any suite:

```rust
/// Resolve an issuer's signing public key for credential verification.
///
/// The `issued_at` parameter enables KEL-backed resolvers to return
/// the signing key that was active at credential issuance time.
/// For non-rotatable identities, `issued_at` can be ignored.
pub trait CredentialKeyResolver {
    fn resolve(&self, issuer: &IdentityRef, issued_at: u64) -> Option<Vec<u8>>;
}
```

The `CryptoSuite` in `issuer` tells the verification function which
algorithm to use. The resolver just provides the raw key bytes —
Ed25519 (32B), ML-DSA-65 (1952B), or the historical ML-DSA-65 key
from a KEL.

**Why not modify `IdentityResolver`?** It's used by UCAN verification
and other consumers. Changing its return type would be a breaking change
across the workspace. The credential-specific trait is additive and
focused.

**KEL integration note:** `KeyEventLog` currently exposes only
`current_signing_key()`. A `CredentialKeyResolver` implementation for
KEL-backed identities would need to walk the event chain to find the
key active at `issued_at`. This is a consumer implementation concern —
the credential crate defines the trait, not the KEL integration.

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

Following the `RevocationSet` pattern from UCANs (which answers a
yes/no question rather than returning borrowed data), the status list
resolver provides a direct query interface:

```rust
/// Check whether a credential has been revoked.
///
/// Returns `Some(true)` if revoked, `Some(false)` if valid,
/// `None` if the issuer's status list could not be resolved.
pub trait StatusListResolver {
    fn is_revoked(&self, issuer: &IdentityRef, index: u32) -> Option<bool>;
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
    keys: &impl CredentialKeyResolver,
    status_lists: &impl StatusListResolver,
) -> Result<(), CredentialError>
```

**Steps, in order:**
1. **Time bounds** — `not_before <= now < expires_at`
2. **Issuer resolution** — look up issuer's public key via
   `CredentialKeyResolver`, passing `credential.issued_at`
3. **Signature verification** — reconstruct signable payload, verify
   against issuer's key using algorithm from `issuer.suite`
4. **Revocation check** — if `status_list_index` is `Some(idx)`, call
   `status_lists.is_revoked(issuer, idx)`

### Presentation Verification

```rust
pub fn verify_presentation(
    presentation: &Presentation,
    now: u64,
    keys: &impl CredentialKeyResolver,
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
    IndexOutOfBounds,
    SerializeError(&'static str),
    DeserializeError(&'static str),
}
```

Implements `Display` and `std::error::Error` (under `std` feature),
following the pattern in `KelError` and `TrustError`.

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
    /// All time bounds are required at construction.
    /// `not_before` defaults to `issued_at`; `expires_at` must be > `not_before`.
    pub fn new(
        issuer: IdentityRef,
        subject: IdentityRef,
        issued_at: u64,
        expires_at: u64,
        nonce: [u8; 16],
    ) -> Self { ... }

    pub fn not_before(&mut self, not_before: u64) -> &mut Self { ... }
    pub fn add_claim(&mut self, type_id: u16, value: Vec<u8>, salt: [u8; 16]) -> &mut Self { ... }
    pub fn status_list_index(&mut self, index: u32) -> &mut Self { ... }

    /// Produce the signable payload bytes.
    pub fn signable_payload(&self) -> Vec<u8> { ... }

    /// Finalize with a signature to produce Credential + SaltedClaims.
    /// The SaltedClaims should be stored by the holder for later disclosure.
    pub fn build(self, signature: Vec<u8>) -> (Credential, Vec<SaltedClaim>) { ... }
}
```

`not_before` defaults to `issued_at` when not explicitly set.
`expires_at` and `nonce` are required constructor parameters to prevent
accidental omission.

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

[dev-dependencies]
postcard = { workspace = true }
rand = { workspace = true }

[features]
default = ["std"]
std = ["harmony-identity/std", "harmony-crypto/std", "serde/std", "postcard/use-std"]
test-utils = ["hashbrown"]

[dependencies.hashbrown]
workspace = true
optional = true
```

### Re-exports (lib.rs)

```rust
pub use claim::{Claim, SaltedClaim};
pub use credential::{Credential, CredentialBuilder};
pub use disclosure::Presentation;
pub use error::CredentialError;
pub use status_list::{StatusList, StatusListResolver};
pub use verify::{verify_credential, verify_presentation, CredentialKeyResolver};

#[cfg(any(test, feature = "test-utils"))]
pub use status_list::MemoryStatusListResolver;
#[cfg(any(test, feature = "test-utils"))]
pub use verify::MemoryKeyResolver;
```

## Changes to Existing Crates

None. This is a new additive crate. All dependencies (`IdentityRef`,
`CryptoSuite`, BLAKE3, ML-DSA-65) already exist. The new
`CredentialKeyResolver` trait replaces `IdentityResolver` usage,
avoiding changes to `harmony-identity`.

## Testing Strategy

### Claim tests
- `SaltedClaim::digest()` is deterministic for same inputs
- Different salts produce different digests
- Different type_ids with same value produce different digests

### Credential tests
- Builder produces valid credential with correct digest count
- `signable_payload()` is deterministic (same inputs -> same bytes)
- `content_hash()` is deterministic
- Serde round-trip preserves all fields
- Self-issued credential (`issuer == subject`) builds and verifies

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
- Non-revocable credential skips status list check

### StatusList tests
- New list has all bits unset
- `revoke(idx)` sets the bit
- `is_revoked(idx)` returns correct state
- Out-of-bounds index handling
- Serde round-trip preserves bit state
- Custom capacity works

### Integration tests
- Full flow: build credential -> create presentation -> verify
- Both Ed25519 and ML-DSA-65 issuer paths
- Revocation after issuance

## Future Beads

- **harmony-8ws:** W3C JSON-LD bridge — optional `std` feature for
  JSON-LD serialization, DID-based identifiers, external VC wallet
  interop
- **harmony-2fs:** Credential delegation chains — optional `proof`
  field referencing parent credentials, chain verification, attenuation
  rules
