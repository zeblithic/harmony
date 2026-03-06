# UCAN Capability Tokens — Design

**Date:** 2026-03-06
**Status:** Approved
**Bead:** harmony-7w4

## Problem

Harmony's concentric ring architecture needs a formal authorization primitive. Every syscall, IPC message, and resource access in Rings 2-3 is capability-gated, but Ring 0 has no capability token format. The mesh OS design specifies "UCAN-style delegation chains signed by Ed25519" rooted in `PrivateIdentity`, but no implementation exists.

## Solution

A `ucan` module in harmony-identity implementing Harmony-native capability tokens that follow UCAN authorization principles (delegation chains, attenuation, Ed25519 signatures) without adopting the web-ecosystem format (no DIDs, no JWT, no DAG-CBOR, no jq policy language). Compact binary tokens with content-addressed proof chains, optimized for kernel verification and `no_std`.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Capability model | Harmony-native with UCAN principles | Compact binary format around 7 Ring 2 types. No DIDs/JWT. Optimized for kernel verification and no_std. |
| Revocation | Expiry-first + signed revocation records | Mandatory expiration as primary security. Local revocation set for immediate invalidation. Distributed via Zenoh pub/sub. |
| Resource scoping | Generic opaque bytes | Ring 0 handles signing/verification/chains. Ring 2 interprets resource bytes per capability type. Decoupled evolution. |
| Chain depth | Configurable max, default 5 | Bounds verification cost in kernel hot path. 5 hops covers root→service→subprocess→worker→ephemeral. |
| Crate location | `ucan` module in harmony-identity | UCANs are identity-derived. Keeps dependency graph flat. ~400-600 lines. |
| Proof chain | Content-addressed (BLAKE3 hash references) | Compact fixed-size tokens. Proofs stored in content store (local cache + mesh). Deduplication. Aligns with Harmony's content-addressed architecture. |
| Attenuation at Ring 0 | Capability type match + time bound narrowing only | Resource-level attenuation (port ranges, CID ranges) deferred to Ring 2. |

## Token Structure

```rust
pub struct UcanToken {
    /// Issuer: address hash of the signing identity (16 bytes)
    pub issuer: [u8; 16],

    /// Audience: address hash of the recipient identity (16 bytes)
    pub audience: [u8; 16],

    /// Capability type (1 byte enum)
    pub capability: CapabilityType,

    /// Resource scope (opaque bytes, interpreted by Ring 2, max 256 bytes)
    pub resource: Vec<u8>,

    /// Not-before: Unix timestamp in seconds (8 bytes)
    pub not_before: u64,

    /// Expires-at: Unix timestamp in seconds (8 bytes, 0 = never)
    pub expires_at: u64,

    /// Nonce: 16 random bytes for replay protection
    pub nonce: [u8; 16],

    /// Parent proof: BLAKE3 hash of the parent token (32 bytes)
    /// None for root tokens (issued by the resource owner)
    pub proof: Option<[u8; 32]>,

    /// Ed25519 signature over all preceding fields (64 bytes)
    pub signature: [u8; 64],
}

#[repr(u8)]
pub enum CapabilityType {
    Memory    = 0,
    Endpoint  = 1,
    Interrupt = 2,
    IOPort    = 3,
    Identity  = 4,
    Content   = 5,
    Compute   = 6,
}
```

**Wire size:** Root token with 32-byte resource ≈ 177 bytes. Delegated token ≈ 209 bytes. Both fit in 500-byte MTU.

## Binary Wire Format

Big-endian, fixed field order, deterministic:

```
[16B issuer][16B audience][1B capability_type][2B resource_len][NB resource]
[8B not_before][8B expires_at][16B nonce][1B has_proof][32B proof (if has_proof)]
[64B signature]
```

The signature covers everything before it (`signable_bytes`). Content hash = `BLAKE3(entire_serialized_token_including_signature)`.

## Content Addressing

A token's content ID is `BLAKE3(to_bytes())`. Parent proofs are referenced by this hash. The proof chain is stored in the content store (harmony-content's blob store), which functions as a local + mesh-distributed cache. "Offline" verification works because the local content store always has proofs for locally-held capabilities.

## Verification

Three resolver traits form the sans-I/O boundary:

```rust
/// Resolves parent tokens by their content hash.
pub trait ProofResolver {
    fn resolve(&self, hash: &[u8; 32]) -> Option<UcanToken>;
}

/// Resolves identities by their address hash.
pub trait IdentityResolver {
    fn resolve(&self, address_hash: &[u8; 16]) -> Option<Identity>;
}

/// Checks whether a token has been revoked.
pub trait RevocationSet {
    fn is_revoked(&self, token_hash: &[u8; 32]) -> bool;
}
```

**Verification algorithm:**

1. Check time bounds: `not_before <= now <= expires_at`
2. Verify Ed25519 signature against issuer's public key (via `IdentityResolver`)
3. Check local revocation set for the token's content hash
4. If `proof` is `Some(parent_hash)`:
   a. Resolve parent token via `ProofResolver`
   b. Verify parent's `audience` == this token's `issuer` (chain continuity)
   c. Verify attenuation: capability type match + time bounds narrowed
   d. Recursively verify parent (up to max depth)
5. If `proof` is `None`: root token — issuer claims resource ownership

**Verifier signature:**

```rust
pub fn verify_token(
    token: &UcanToken,
    now: u64,
    proofs: &impl ProofResolver,
    identities: &impl IdentityResolver,
    revocations: &impl RevocationSet,
    max_depth: usize,
) -> Result<(), UcanError>
```

## Delegation and Attenuation

Delegation creates a child token where:
- `issuer` = delegator's address hash
- `proof` = BLAKE3 hash of the parent token
- Capability must be equal to or narrower than parent's

**Ring 0 attenuation rules** (lightweight):
- Capability type must match exactly
- `child.not_before >= parent.not_before`
- `parent.expires_at == 0 || child.expires_at <= parent.expires_at`

**Resource-level attenuation** (e.g., "parent grants ports 80-443, child gets port 80") is deferred to Ring 2, which understands the resource byte format per capability type.

**Delegation API:**

```rust
impl PrivateIdentity {
    pub fn issue_root_token(
        &self,
        rng: &mut impl EntropySource,
        audience: &[u8; 16],
        capability: CapabilityType,
        resource: &[u8],
        not_before: u64,
        expires_at: u64,
    ) -> Result<UcanToken, UcanError>;

    pub fn delegate(
        &self,
        rng: &mut impl EntropySource,
        parent: &UcanToken,
        audience: &[u8; 16],
        capability: CapabilityType,
        resource: &[u8],
        not_before: u64,
        expires_at: u64,
    ) -> Result<UcanToken, UcanError>;
}
```

## Revocation

**Revocation record:**

```rust
pub struct Revocation {
    /// Address hash of the revoking identity
    pub issuer: [u8; 16],
    /// BLAKE3 hash of the token being revoked
    pub token_hash: [u8; 32],
    /// Timestamp of revocation
    pub revoked_at: u64,
    /// Ed25519 signature
    pub signature: [u8; 64],
}
```

**Rules:**
- Only the issuer of a token can revoke it (verified by signature)
- Revoking a token implicitly invalidates all downstream delegations (chain verification fails if any link is revoked)
- Revocations are immutable

**Distribution:** Published via Zenoh at `harmony/{issuer_hash}/revocations`. Each kernel subscribes and maintains a local `HashSet<[u8; 32]>` for O(1) lookup.

## Error Type

```rust
#[derive(Debug, thiserror::Error)]
pub enum UcanError {
    #[error("token has expired")]
    Expired,
    #[error("token is not yet valid")]
    NotYetValid,
    #[error("signature verification failed")]
    SignatureInvalid,
    #[error("delegation chain exceeds maximum depth of {0}")]
    ChainTooDeep(usize),
    #[error("proof token not found")]
    ProofNotFound,
    #[error("chain continuity broken: parent audience != child issuer")]
    ChainBroken,
    #[error("capability type mismatch in delegation")]
    CapabilityMismatch,
    #[error("time bounds not attenuated")]
    AttenuationViolation,
    #[error("token has been revoked")]
    Revoked,
    #[error("issuer identity not found")]
    IssuerNotFound,
    #[error("invalid token encoding")]
    InvalidEncoding,
    #[error("resource field exceeds maximum size")]
    ResourceTooLarge,
}
```

## In-Memory Test Implementations

- `MemoryProofStore` — `HashMap<[u8; 32], UcanToken>` implementing `ProofResolver`
- `MemoryIdentityStore` — `HashMap<[u8; 16], Identity>` implementing `IdentityResolver`
- `MemoryRevocationSet` — `HashSet<[u8; 32]>` implementing `RevocationSet`

## What This Bead Does NOT Do

- **No resource-level attenuation** — Ring 2 interprets resource bytes per capability type
- **No Zenoh revocation distribution** — separate concern, uses existing Zenoh session layer
- **No kernel integration** — Ring 2 wires traits into the syscall path
- **No content store integration** — `ProofResolver` is the abstraction; content-store-backed impl is Ring 2
- **No process migration serialization** — future bead
- **No flat inline proof mode** — potential future addition for constrained scenarios
