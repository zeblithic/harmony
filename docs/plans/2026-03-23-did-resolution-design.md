# DID Resolution: did:key and did:jwk

**Date:** 2026-03-23
**Status:** Approved
**Scope:** Parse DID identifiers to raw public key bytes + CryptoSuite
**Bead:** harmony-95u

## Overview

Resolve `did:key` and `did:jwk` identifiers into raw public key
bytes and a `CryptoSuite` discriminant. This is the inverse of
`identity_to_did_key` (from harmony-credential's JSON-LD bridge)
and the missing link that makes the SD-JWT import pipeline usable
end-to-end: parse SD-JWT → resolve issuer DID → verify signature.

A `DidResolver` trait enables extensibility for network-backed
methods (e.g., did:web via harmony-fyr's Zenoh gateway). did:web
returns `UnsupportedMethod` in this bead.

## Module Location

`crates/harmony-identity/src/did.rs` — no feature flag for did:key
(purely algorithmic, no_std + alloc). did:jwk requires JSON parsing
and is gated behind `std` feature.

## Data Types

### ResolvedDid

```rust
pub struct ResolvedDid {
    pub suite: CryptoSuite,
    pub public_key: Vec<u8>,
}
```

### DidError

```rust
pub enum DidError {
    UnsupportedMethod(String),
    MalformedDid(String),
    DecodingError(String),
    UnknownMulticodec(u32),
}
```

## Public API

### resolve_did

```rust
pub fn resolve_did(did: &str) -> Result<ResolvedDid, DidError>
```

Dispatches on the DID method prefix:
- `did:key:` → `resolve_did_key`
- `did:jwk:` → `resolve_did_jwk` (requires `std`)
- `did:web:` → `Err(UnsupportedMethod("web"))`
- anything else → `Err(UnsupportedMethod(method))`

### DidResolver trait

```rust
pub trait DidResolver {
    fn resolve(&self, did: &str) -> Result<ResolvedDid, DidError>;
}
```

A `DefaultDidResolver` struct implements this trait by delegating
to `resolve_did`. This lets callers use the trait uniformly whether
they have a network-backed resolver or just the built-in methods.

```rust
pub struct DefaultDidResolver;

impl DidResolver for DefaultDidResolver {
    fn resolve(&self, did: &str) -> Result<ResolvedDid, DidError> {
        resolve_did(did)
    }
}
```

## did:key Resolution

Per the did:key spec:

1. Strip `did:key:` prefix
2. Verify multibase prefix is `z` (base58btc)
3. Base58-decode the remainder
4. Read unsigned varint (LEB128) multicodec prefix
5. Map the reconstructed multicodec integer → CryptoSuite:
   - `0x00ed` → Ed25519 (LEB128 wire bytes: `[0xed, 0x01]`)
   - `0x1211` → MlDsa65 (LEB128 wire bytes: `[0x91, 0x24]`)
   - other → `UnknownMulticodec`
6. Return remaining bytes as `public_key`

This is the exact inverse of `identity_to_did_key` from
harmony-credential's JSON-LD bridge. Uses
`CryptoSuite::from_signing_multicodec()` which maps `0x1211`
to `MlDsa65` (not `MlDsa65Rotatable` — the round-trip is
intentionally lossy, matching `crypto_suite.rs`'s
`multicodec_round_trip_lossy_for_rotatable` test).

### Key Length Validation

After extracting the multicodec and key bytes, validate the key
length matches the expected size for the suite:
- Ed25519: exactly 32 bytes
- MlDsa65: exactly 1952 bytes
- (MlDsa65Rotatable is decoded as MlDsa65 — same multicodec)

Mismatch returns `MalformedDid`.

## did:jwk Resolution

Per the did:jwk spec (requires `std` for JSON parsing):

1. Strip `did:jwk:` prefix
2. Base64url-decode → JSON string
3. Parse as JWK object
4. Extract `kty` and `crv`:
   - `kty: "OKP"`, `crv: "Ed25519"` → Ed25519
   - Other combinations → `MalformedDid("unsupported JWK key type: …")`
5. Base64url-decode the `x` field → raw public key bytes
6. Validate key length

## Dependencies

- `bs58` — base58btc decoding for did:key. Must be added to
  `[workspace.dependencies]` in root `Cargo.toml` (currently
  only a direct dep in harmony-credential) and to
  `harmony-identity/Cargo.toml`.
- `base64` — base64url decoding for did:jwk (already a workspace dep)
- `serde_json` — JWK parsing for did:jwk (behind `std`, already
  a workspace dep)

## Testing

1. did:key Ed25519 round-trip (encode → decode → same bytes)
2. did:key ML-DSA-65 round-trip
3. did:key unknown multicodec → `UnknownMulticodec`
4. did:key malformed (bad prefix, bad base58) → errors
5. did:key wrong key length → `MalformedDid`
6. did:jwk Ed25519 → correct key bytes
7. did:jwk malformed JSON → error
8. did:web → `UnsupportedMethod`
9. Unknown method → `UnsupportedMethod`
10. DefaultDidResolver delegates to resolve_did

## What This Bead Delivers

- `did.rs` module in `harmony-identity`
- `resolve_did`, `resolve_did_key`, `resolve_did_jwk` functions
- `ResolvedDid`, `DidError` types
- `DidResolver` trait + `DefaultDidResolver`
- Re-exports from `lib.rs`
- ~150 lines production code, ~200 lines tests
- no_std + alloc for did:key, std for did:jwk

## What This Bead Does NOT Deliver

- did:web resolution (harmony-fyr)
- did:ethr, did:ion, or other blockchain-based methods
- DID Document construction (only key extraction)
- Key rotation / versioning
