# Unified Identity Reference Design

**Date:** 2026-03-19
**Status:** Approved
**Scope:** Modifications to `harmony-identity` crate
**Bead:** harmony-dnp

## Overview

Harmony has two fully implemented identity types — `Identity`
(Ed25519/X25519, classical) and `PqIdentity` (ML-KEM-768/ML-DSA-65,
post-quantum) — but no unified way to reference "any identity" while
carrying crypto suite information. Consumers like `harmony-trust` and
`harmony-contacts` currently use `IdentityHash` (`[u8; 16]`), which
identifies an identity by address but cannot distinguish classical
from post-quantum keys.

This design introduces:

1. **`CryptoSuite`** — promoted from UCAN-only to crate-level, with
   multicodec conversion for future DID interoperability
2. **`IdentityRef`** — a 17-byte lightweight reference carrying the
   address hash + crypto suite tag

Both types are additive. No existing consumer is forced to change.

## Design Philosophy

### PQ-Native, Classical as Compat

Post-quantum identities (ML-KEM-768/ML-DSA-65) are Harmony-native and
fully trusted. Classical identities (Ed25519/X25519) are supported for
backward compatibility with the existing Reticulum network but treated
as cryptographically weaker — the key material is vulnerable to quantum
attack. The `CryptoSuite` tag makes this distinction explicit so
consumers can apply appropriate policy.

### Reference, Not Replacement

`IdentityRef` is a thin reference (hash + suite), not an enum wrapping
the full key material. Crypto operations (sign, verify, encrypt, decrypt)
still happen on the concrete `Identity` / `PqIdentity` types. The
reference is what you pass around when you need to know "which identity"
and "what kind" without carrying kilobytes of public key material.

### Additive, Not Breaking

`IdentityHash` stays. `Identity` and `PqIdentity` stay. UCAN's use of
`CryptoSuite` stays. Everything is backward compatible. Consumers adopt
`IdentityRef` when they need suite awareness, or keep using `IdentityHash`
when they don't.

## CryptoSuite

### Current state

`CryptoSuite` exists in `ucan.rs` as a UCAN-specific enum with values
`Ed25519 = 0x00` and `MlDsa65 = 0x01`. It's only used for UCAN token
wire format dispatch.

### Change

Promote to a top-level module `crypto_suite.rs`. Add multicodec
conversion and quantum-resistance query. Re-export from `ucan.rs` for
backward compatibility.

```rust
/// The cryptographic algorithm suite backing an identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CryptoSuite {
    /// Ed25519 signing + X25519 encryption (Reticulum-compatible).
    /// Backward-compatibility layer — NOT post-quantum secure.
    Ed25519 = 0x00,
    /// ML-DSA-65 signing + ML-KEM-768 encryption (NIST FIPS 203/204).
    /// Harmony-native, post-quantum secure.
    MlDsa65 = 0x01,
}

impl CryptoSuite {
    /// Whether this suite is post-quantum secure.
    pub fn is_post_quantum(self) -> bool {
        matches!(self, Self::MlDsa65)
    }

    /// Multicodec identifier for the signing algorithm.
    /// Ed25519 = 0x00ed, ML-DSA-65 = 0x1211.
    pub fn signing_multicodec(self) -> u16 {
        match self {
            Self::Ed25519 => 0x00ed,
            Self::MlDsa65 => 0x1211,
        }
    }

    /// Multicodec identifier for the encryption/KEM algorithm.
    /// X25519 = 0x00ec, ML-KEM-768 = 0x120c.
    pub fn encryption_multicodec(self) -> u16 {
        match self {
            Self::Ed25519 => 0x00ec,
            Self::MlDsa65 => 0x120c,
        }
    }

    /// Construct from a signing multicodec identifier.
    pub fn from_signing_multicodec(code: u16) -> Option<Self> {
        match code {
            0x00ed => Some(Self::Ed25519),
            0x1211 => Some(Self::MlDsa65),
            _ => None,
        }
    }

    /// Construct from an encryption multicodec identifier.
    pub fn from_encryption_multicodec(code: u16) -> Option<Self> {
        match code {
            0x00ec => Some(Self::Ed25519),
            0x120c => Some(Self::MlDsa65),
            _ => None,
        }
    }

    /// Construct from the wire discriminant byte (0x00 or 0x01).
    pub fn from_byte(byte: u8) -> Option<Self> {
        match byte {
            0x00 => Some(Self::Ed25519),
            0x01 => Some(Self::MlDsa65),
            _ => None,
        }
    }
}
```

### Multicodec values

Ed25519 (`0x00ed`) and X25519 (`0x00ec`) are finalized in the multicodec
table. ML-DSA-65 (`0x1211`) and ML-KEM-768 (`0x120c`) are draft entries
as of 2026-03 — subject to finalization. The conversion methods will be
updated if the values change before ratification.

### Wire byte values preserved

The discriminant values (`0x00` and `0x01`) are unchanged from the UCAN
wire format. Existing serialized UCAN tokens remain valid.

## IdentityRef

```rust
/// A lightweight reference to an identity: address hash + crypto suite.
///
/// 17 bytes total. Use this when you need to know "which identity" and
/// "what kind" without carrying full public key material.
///
/// Construct from concrete identity types via From impls, or directly
/// via IdentityRef::new().
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct IdentityRef {
    /// The 128-bit address hash: SHA256(pub_keys)[:16].
    pub hash: IdentityHash,
    /// The cryptographic suite backing this identity.
    pub suite: CryptoSuite,
}

impl IdentityRef {
    pub fn new(hash: IdentityHash, suite: CryptoSuite) -> Self {
        Self { hash, suite }
    }

    /// Whether this identity uses post-quantum cryptography.
    pub fn is_post_quantum(&self) -> bool {
        self.suite.is_post_quantum()
    }
}

impl From<&Identity> for IdentityRef {
    fn from(id: &Identity) -> Self {
        Self {
            hash: id.address_hash,
            suite: CryptoSuite::Ed25519,
        }
    }
}

impl From<&PqIdentity> for IdentityRef {
    fn from(id: &PqIdentity) -> Self {
        Self {
            hash: id.address_hash,
            suite: CryptoSuite::MlDsa65,
        }
    }
}
```

### Relationship to IdentityHash

`IdentityHash` (`[u8; 16]`) remains the universal key for maps where
crypto suite doesn't matter:

| Use case | Right type |
|----------|-----------|
| Path table routing | `DestinationHash` (unchanged) |
| Trust store keys | `IdentityHash` (suite-aware policy is consumer-side) |
| Contact store keys | `IdentityHash` (can adopt `IdentityRef` later) |
| "Who is this and what kind?" | `IdentityRef` |

You can always extract the hash: `identity_ref.hash` is an `IdentityHash`.

### Trust integration (consumer-side policy, not stored)

The trust store's Identity dimension (bits [7:6]) relates to crypto suite.
A classical identity has an inherently lower verification ceiling because
the key material is quantum-vulnerable. But this is a **policy decision**,
not a store decision:

```rust
// Example consumer policy (NOT in harmony-trust)
fn max_identity_for_suite(suite: CryptoSuite) -> u8 {
    match suite {
        CryptoSuite::Ed25519 => 1,  // "weakly verified" ceiling
        CryptoSuite::MlDsa65 => 3,  // full range available
    }
}
```

The trust store stores whatever score you set. `IdentityRef` gives
consumers the information to apply suite-aware policies on top.

## Changes to harmony-identity

### New files

- `src/crypto_suite.rs` — `CryptoSuite` enum with multicodec conversion
- `src/identity_ref.rs` — `IdentityRef` struct with `From` impls
  (imports `crate::identity::Identity` and `crate::pq_identity::PqIdentity`)

### Modified files

- `Cargo.toml` — add `serde` dependency (workspace, default-features = false,
  features = ["derive", "alloc"]) and `postcard` for tests. Add `serde/std`
  to the `std` feature gate. This follows the same pattern as `harmony-trust`
  and `harmony-contacts`.
- `src/ucan.rs` — remove local `CryptoSuite` definition, add
  `use crate::crypto_suite::CryptoSuite;` at the top. Existing code uses
  bare `CryptoSuite` (not qualified), so the import resolves all references.
- `src/lib.rs` — add `pub mod crypto_suite;` and `pub mod identity_ref;`.
  Update re-exports: change `pub use ucan::{..., CryptoSuite, ...}` to
  `pub use crypto_suite::CryptoSuite;` and add `pub use identity_ref::IdentityRef;`.

### Files NOT changed

- `src/identity.rs` — `Identity` type untouched
- `src/pq_identity.rs` — `PqIdentity` type untouched
- All downstream crates — no changes required (verified: no downstream
  crate imports `CryptoSuite` directly; the public API path
  `harmony_identity::CryptoSuite` is preserved through `lib.rs` re-export)

## Testing Strategy

### CryptoSuite tests

- `is_post_quantum()`: Ed25519 returns false, MlDsa65 returns true
- Multicodec values: signing and encryption match known standards
  (Ed25519 = 0x00ed, X25519 = 0x00ec, ML-DSA-65 = 0x1211, ML-KEM-768 = 0x120c)
- `from_signing_multicodec` / `from_encryption_multicodec` round-trip
- Unknown multicodec returns None
- Wire byte discriminant: Ed25519 = 0x00, MlDsa65 = 0x01
- Serde round-trip

### IdentityRef tests

- Construction from `Identity` sets `CryptoSuite::Ed25519`
- Construction from `PqIdentity` sets `CryptoSuite::MlDsa65`
- `is_post_quantum()` delegates correctly
- Hash extraction matches source identity's address_hash
- Equality: same hash + same suite = equal
- Inequality: same hash + different suite = not equal
- Serde round-trip

### UCAN backward compatibility

- All existing UCAN tests pass unchanged (CryptoSuite re-exported)
- Full workspace test suite passes
