# Announce Pubkey→Hash Binding

**Date:** 2026-03-22
**Status:** Draft
**Scope:** `harmony-discovery` (AnnounceRecord, verify_announce, AnnounceBuilder)
**Bead:** harmony-3bu

## Overview

V1 limitation: `verify_announce()` checks that the record's signature matches the included `public_key`, but does NOT verify that the `public_key` hashes to the claimed `identity_ref.hash`. An attacker can generate their own ML-DSA keypair, craft an announce with the victim's `identity_hash` but the attacker's pubkey, and it passes verification. This enables pubkey cache poisoning for token forgery.

The root cause: `address_hash = SHA256(ML-KEM-768_pub || ML-DSA-65_pub)[:16]` requires BOTH the encryption key and the signing key to re-derive, but the announce only carries the signing key.

## Fix

Include the ML-KEM-768 encryption key in the announce record. The verifier re-derives `SHA256(enc_pub || sign_pub)[:16]` and compares against `identity_ref.hash`.

## Changes

### AnnounceRecord — new field

```rust
pub struct AnnounceRecord {
    pub identity_ref: IdentityRef,
    pub public_key: Vec<u8>,        // ML-DSA-65 verifying key (1952 B) — existing
    pub encryption_key: Vec<u8>,    // ML-KEM-768 public key (1184 B) — NEW
    pub routing_hints: Vec<RoutingHint>,
    pub published_at: u64,
    pub expires_at: u64,
    pub nonce: [u8; 16],
    pub signature: Vec<u8>,
}
```

The `encryption_key` is included in `SignablePayload` (signed, tamper-proof).

### verify_announce() — binding check

After signature verification passes:

```rust
// Re-derive address hash from the included public keys.
let mut combined = Vec::with_capacity(
    record.encryption_key.len() + record.public_key.len()
);
combined.extend_from_slice(&record.encryption_key);
combined.extend_from_slice(&record.public_key);
let derived_hash = harmony_crypto::hash::truncated_hash(&combined);
if derived_hash != record.identity_ref.hash {
    return Err(DiscoveryError::AddressMismatch);
}
```

`DiscoveryError::AddressMismatch` already exists (`#[allow(dead_code)]`) — reserved for this purpose.

### AnnounceBuilder — accept encryption_key

```rust
pub fn new(
    identity_ref: IdentityRef,
    public_key: Vec<u8>,
    encryption_key: Vec<u8>,  // NEW
    published_at: u64,
    expires_at: u64,
    nonce: [u8; 16],
) -> Self
```

Include `encryption_key` in `signable_payload()`.

### FORMAT_VERSION bump

Bump from 2 to 3. V2 records (without `encryption_key`) are rejected — unverified announces should not be trusted.

### harmony-node runtime — downgrade security warning

The `pubkey_cache` SECURITY(V1) warnings can be downgraded to informational comments noting the binding is now enforced.

## Wire Size Impact

| Component | Old | New |
|-----------|-----|-----|
| ML-DSA-65 verifying key | 1952 B | 1952 B |
| ML-KEM-768 encryption key | — | 1184 B |
| ML-DSA-65 signature | 3309 B | 3309 B |
| Other fields | ~100 B | ~100 B |
| **Total** | **~5.4 KB** | **~6.5 KB** |

Announces are infrequent (on connect + periodic refresh). 1.1 KB increase is negligible.

## What this fixes

- Forged announce records can no longer poison the `pubkey_cache`
- Token-gated serving is secure against cache poisoning attacks
- `DiscoveryError::AddressMismatch` is no longer dead code

## What this doesn't change

- Ed25519 suite binding (same principle: `SHA256(X25519_pub || Ed25519_pub)[:16]` — announce would need to carry both keys for Ed25519 too, but Ed25519 is legacy/Reticulum-compat only)
- Announce format for the Ed25519 suite is left as-is (V1 limitation remains for legacy)
