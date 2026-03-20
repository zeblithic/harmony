# Profiles & Endorsements Design

**Date:** 2026-03-20
**Status:** Approved
**Scope:** New `harmony-profile` crate + additive `harmony-zenoh` namespace patterns
**Bead:** harmony-3xn

## Overview

Harmony needs public identity metadata (profiles) and signed
attestations between identities (endorsements). Profiles let
identities publish human-facing information — display name, status,
avatar. Endorsements let identities make verifiable statements about
each other — "I can verify this person holds credential X."

This design introduces a `harmony-profile` crate that provides:

1. **Profile records** — self-signed identity metadata published via
   Zenoh pub/sub
2. **Endorsement records** — signed attestations from one identity
   about another, published via Zenoh pub/sub
3. **Verification** — signature checking for both record types via
   a key resolver trait

Accusations (cryptographic proof of misbehavior) are deferred to a
future bead — they require evidence infrastructure that warrants
separate design.

## Design Philosophy

### Profiles Are Published Metadata, Not Identity

Identity (`harmony-identity`) is who you are cryptographically.
Profiles are what you choose to tell the world — display name,
status text, avatar. They live in a separate crate and namespace
because they have different update cadence, privacy requirements,
and trust characteristics than identity announcements.

### Endorsements Are Attestations, Not Trust

Endorsements are verifiable statements of fact: "I can confirm
Identity B holds a driver's license" or "I verified Identity B's
degree." They are NOT trust mutations — receiving an endorsement
does not automatically change anyone's `TrustScore`. The trust
store remains a local, subjective assessment. Endorsements are
public data that consumers (including future EigenTrust computation)
can use to *inform* trust decisions.

### Endorser Ownership

Endorsements live under the endorser's namespace
(`harmony/endorsement/{endorser_hex}/...`). The endorser owns their
voice — they control what they publish. The endorsee curates which
endorsements they accept/display by referencing them from their
profile or client UI. An endorsement exists whether or not the
endorsee acknowledges it — it's public information.

### Sans-I/O

Following the established Harmony pattern, all operations are pure
functions. Signing is the caller's responsibility (builders produce
signable payloads, callers sign externally). Verification uses
trait-based key resolution. No I/O, no async, no runtime coupling.

### Post-Quantum First

ML-DSA-65 is the primary signing suite. Ed25519 is supported for
Reticulum compatibility but is not the default path. Documentation
and examples lead with post-quantum identity construction.

## ProfileRecord

```rust
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProfileRecord {
    pub identity_ref: IdentityRef,
    pub display_name: Option<String>,
    pub status_text: Option<String>,
    pub avatar_cid: Option<[u8; 32]>,
    pub published_at: u64,       // unix epoch seconds
    pub expires_at: u64,         // prevents indefinite caching
    pub nonce: [u8; 16],         // replay protection, crypto random
    pub signature: Vec<u8>,      // self-signed
}
```

### Field Rationale

- **`identity_ref`** — the identity that owns this profile.
  `CryptoSuite` tag determines signature verification algorithm.

- **`display_name`** — human-readable name. Optional — an identity
  can exist without a display name (identified by address hash or
  identicon).

- **`status_text`** — freetext status. Future convention: if it
  matches the form `/book/<256-bit-address>`, clients can resolve
  it as a CAS reference to a rich content bundle containing
  arbitrarily complex profile data (static + dynamic state bundled
  into a single addressable entity). The crate treats it as an
  opaque string.

- **`avatar_cid`** — 32-byte CID pointing to image content in the
  CAS layer. Resolution and rendering are consumer concerns.

- **`published_at`** / **`expires_at`** — unix epoch seconds.
  Same TTL pattern as announce records. Profiles are refreshed
  periodically by the owning identity.

- **`nonce`** — 16 cryptographically random bytes. Must be unique
  per publication, especially when `published_at` has not advanced.

- **`signature`** — self-signed by the owning identity's private
  key. Only the identity owner can publish their own profile.

### Self-Signed Verification

Unlike credentials (issuer signs about a subject), profiles are
always self-signed. The verifier checks the signature against the
identity's public key, resolved via `ProfileKeyResolver`. No
`public_key` field on the struct — the verifier already knows the
peer (from a prior announce, contact, or key exchange).

### Serialization

Postcard with format version byte prefix (same pattern as
`Credential`, `AnnounceRecord`).

## ProfileBuilder

```rust
pub struct ProfileBuilder {
    identity_ref: IdentityRef,
    display_name: Option<String>,
    status_text: Option<String>,
    avatar_cid: Option<[u8; 32]>,
    published_at: u64,
    expires_at: u64,
    nonce: [u8; 16],
}

impl ProfileBuilder {
    /// All timestamps are unix epoch seconds. `nonce` must be
    /// cryptographically random.
    ///
    /// # Panics
    ///
    /// Panics if `expires_at <= published_at`.
    pub fn new(
        identity_ref: IdentityRef,
        published_at: u64,
        expires_at: u64,
        nonce: [u8; 16],
    ) -> Self { ... }

    pub fn display_name(&mut self, name: String) -> &mut Self { ... }
    pub fn status_text(&mut self, text: String) -> &mut Self { ... }
    pub fn avatar_cid(&mut self, cid: [u8; 32]) -> &mut Self { ... }
    pub fn signable_payload(&self) -> Vec<u8> { ... }
    pub fn build(self, signature: Vec<u8>) -> ProfileRecord { ... }
}
```

## EndorsementRecord

```rust
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EndorsementRecord {
    pub endorser: IdentityRef,
    pub endorsee: IdentityRef,
    pub type_id: u32,
    pub reason: Option<String>,
    pub published_at: u64,       // unix epoch seconds
    pub expires_at: u64,
    pub nonce: [u8; 16],
    pub signature: Vec<u8>,      // signed by endorser
}
```

### Field Rationale

- **`endorser`** — the identity making the attestation. Owns the
  signature. `CryptoSuite` tag determines verification algorithm.

- **`endorsee`** — the identity being endorsed.

- **`type_id: u32`** — endorsement category. Following the Q8 page
  address convention (2 mode bits, 28 hash bits, 2 checksum bits),
  the type ID can be interpreted as a content-addressed reference
  to a page that defines the endorsement schema. This makes
  endorsement types self-describing and immutable — two independent
  parties defining the same endorsement structure get the same ID
  because it's derived from the content hash. For V1, consumers can
  also use plain integer type IDs; the crate does not interpret the
  value.

- **`reason`** — optional human-readable context for the
  endorsement ("I verified their license in person"). Not structured
  data.

- **`published_at`** / **`expires_at`** — unix epoch seconds. An
  endorsement expires naturally; to "revoke" before expiry, stop
  refreshing it or publish a replacement with `expires_at` in the
  past.

- **`nonce`** — 16 cryptographically random bytes.

- **`signature`** — signed by the endorser's private key.

### Endorsement Semantics

Endorsements are attestations of fact, not expressions of trust.
They're closer to credentials than to trust scores:

- **Trust score endorsement dimension** (bits 1-0 of `TrustScore`)
  reflects your *subjective assessment* of someone's credibility.
- **Endorsement records** are *objective, verifiable statements*
  that an endorser makes about an endorsee.

These may be correlated but one doesn't force the other. You could
endorse someone's qualifications without trusting them personally.

### Ownership & Curation

- **Endorser publishes** at `harmony/endorsement/{endorser_hex}/{endorsee_hex}`
- **Endorsee curates** which endorsements they display by referencing
  them from their profile/client UI
- **Public can query** `harmony/endorsement/*/{endorsee_hex}` to
  find all endorsements about someone, regardless of curation

## EndorsementBuilder

```rust
pub struct EndorsementBuilder {
    endorser: IdentityRef,
    endorsee: IdentityRef,
    type_id: u32,
    reason: Option<String>,
    published_at: u64,
    expires_at: u64,
    nonce: [u8; 16],
}

impl EndorsementBuilder {
    /// # Panics
    ///
    /// Panics if `expires_at <= published_at`.
    pub fn new(
        endorser: IdentityRef,
        endorsee: IdentityRef,
        type_id: u32,
        published_at: u64,
        expires_at: u64,
        nonce: [u8; 16],
    ) -> Self { ... }

    pub fn reason(&mut self, reason: String) -> &mut Self { ... }
    pub fn signable_payload(&self) -> Vec<u8> { ... }
    pub fn build(self, signature: Vec<u8>) -> EndorsementRecord { ... }
}
```

## Verification & Key Resolution

```rust
/// Resolve an identity's verifying public key for profile and
/// endorsement verification.
///
/// Returns the raw verifying key bytes:
/// - Ed25519: 32-byte verifying key
/// - ML-DSA-65: 1952-byte signing public key
///
/// No `issued_at` parameter — profiles and endorsements are always
/// verified against the identity's current key. If the identity has
/// rotated keys, the record should be re-signed.
pub trait ProfileKeyResolver {
    fn resolve(&self, identity: &IdentityRef) -> Option<Vec<u8>>;
}
```

### Verification Functions

```rust
pub fn verify_profile(
    record: &ProfileRecord,
    now: u64,
    keys: &impl ProfileKeyResolver,
) -> Result<(), ProfileError>

pub fn verify_endorsement(
    record: &EndorsementRecord,
    now: u64,
    keys: &impl ProfileKeyResolver,
) -> Result<(), ProfileError>
```

Both follow the same pattern:

1. Structural validity (`published_at < expires_at`)
2. Not expired (`now < expires_at`)
3. Not future-stamped (`published_at <= now + MAX_CLOCK_SKEW`)
4. Key resolution (resolve signer's public key)
5. Signature verification (dispatch on `CryptoSuite`)

Profile verification resolves `record.identity_ref` (self-signed).
Endorsement verification resolves `record.endorser` (endorser-signed).

### MemoryKeyResolver

Test helper behind `cfg(any(test, feature = "test-utils"))`,
same pattern as `harmony-credential` and `harmony-discovery`.

## Error Handling

```rust
pub enum ProfileError {
    Expired,
    FutureTimestamp,
    InvalidRecord,
    SignatureInvalid,
    KeyNotFound,
    SerializeError(&'static str),
    DeserializeError(&'static str),
}
```

`KeyNotFound` is the profile-specific equivalent of credential's
`IssuerNotFound` — the resolver couldn't find the signer's public
key.

Implements `Display` and `std::error::Error` (under `std` feature).

## Zenoh Namespace

### Profile (existing, unchanged)

```
harmony/profile/{address_hex}              — individual profile
harmony/profile/*                          — subscribe to all profiles
```

### Endorsement (new)

```
harmony/endorsement/{endorser_hex}/{endorsee_hex}  — specific endorsement
harmony/endorsement/{endorser_hex}/*               — all by endorser
harmony/endorsement/*/{endorsee_hex}               — all of endorsee
```

Key expression builders added to `harmony-zenoh/src/namespace.rs`:

```rust
pub mod endorsement {
    pub const PREFIX: &str = "harmony/endorsement";

    /// All endorsements by a specific endorser.
    pub fn by_endorser(endorser_hex: &str) -> String { ... }

    /// All endorsements of a specific endorsee.
    pub fn of_endorsee(endorsee_hex: &str) -> String { ... }

    /// Specific endorsement from endorser to endorsee.
    pub fn key(endorser_hex: &str, endorsee_hex: &str) -> String { ... }
}
```

`address_hex` parameters must be the 32-character lowercase hex
encoding of the 16-byte `IdentityHash`.

## Crate Structure

```
harmony-profile/
  Cargo.toml
  src/
    lib.rs              — no_std setup, re-exports
    error.rs            — ProfileError enum
    profile.rs          — ProfileRecord, ProfileBuilder, serialization
    endorsement.rs      — EndorsementRecord, EndorsementBuilder, serialization
    verify.rs           — verify_profile(), verify_endorsement(),
                          ProfileKeyResolver trait, MemoryKeyResolver
```

### Dependencies

```toml
[dependencies]
ed25519-dalek = { workspace = true }
harmony-crypto = { workspace = true, features = ["serde"] }
harmony-identity = { workspace = true }
serde = { workspace = true, default-features = false, features = ["derive", "alloc"] }
postcard = { workspace = true }

[dev-dependencies]
rand = { workspace = true }
hashbrown = { workspace = true }

[features]
default = ["std"]
std = [
    "harmony-crypto/std",
    "harmony-identity/std",
    "postcard/use-std",
    "serde/std",
]
test-utils = ["hashbrown"]

[dependencies.hashbrown]
workspace = true
optional = true
```

No dependency on `harmony-credential`, `harmony-trust`,
`harmony-discovery`, or `harmony-zenoh`. All integration happens
at the runtime layer.

### Re-exports (lib.rs)

```rust
pub use profile::{ProfileRecord, ProfileBuilder};
pub use endorsement::{EndorsementRecord, EndorsementBuilder};
pub use verify::{verify_profile, verify_endorsement, ProfileKeyResolver};
pub use error::ProfileError;

#[cfg(any(test, feature = "test-utils"))]
pub use verify::MemoryKeyResolver;
```

## Changes to Existing Crates

**harmony-zenoh:** Additive only — new `endorsement` module in
`namespace.rs` with key expression constants and builders. No
changes to the existing `profile` module. No logic changes.

**Workspace root Cargo.toml:** Add `harmony-profile` to members
and workspace dependencies.

No other crate modifications.

## Testing Strategy

### ProfileRecord tests
- Builder produces valid record with all fields set
- `signable_payload()` is deterministic
- Serialization round-trip preserves all fields
- Builder panics if `expires_at <= published_at`
- Sparse profile (all Optional fields None) works
- Deserialize rejects corrupt data

### EndorsementRecord tests
- Builder produces valid record with all fields set
- `signable_payload()` is deterministic
- Serialization round-trip preserves all fields
- Builder panics if `expires_at <= published_at`
- Endorsement with and without reason
- Deserialize rejects corrupt data

### Verification tests (ML-DSA-65 primary)
- Valid profile passes (ML-DSA-65)
- Valid endorsement passes (ML-DSA-65)
- Ed25519 profile verification (compat)
- Ed25519 endorsement verification (compat)
- Expired record rejected
- Future-stamped record rejected
- Invalid record (published_at >= expires_at) rejected
- Tampered signature rejected
- Unknown key rejected (KeyNotFound)

### Key expression tests (in harmony-zenoh)
- `endorsement::key()` produces correct format
- `endorsement::by_endorser()` produces correct wildcard
- `endorsement::of_endorsee()` produces correct wildcard

## Future Beads

- **Accusations** — cryptographic proof of misbehavior with "truth
  as absolute defense." Requires evidence infrastructure (logged
  misbehavior, signed proof bundles).
- **EigenTrust** — transitive trust computation that consumes
  endorsements, trust scores, and accusations to produce global
  reputation rankings.
- **Profile CAS bundles** — `/book/<address>` convention for rich
  profile content bundles, resolved through the content-addressed
  storage layer.
- **Endorsement type registry** — well-known Q8 page addresses for
  common endorsement types (identity verification, credential
  attestation, skill certification).
