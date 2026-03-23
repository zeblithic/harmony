# SD-JWT Disclosure Verification and Claim Mapping

**Date:** 2026-03-22
**Status:** Approved
**Scope:** Verify SD-JWT disclosure hashes, map to Harmony `SaltedClaim`
**Bead:** harmony-b3m

## Overview

Add disclosure hash verification and claim mapping to `harmony-sdjwt`.
Given a parsed `SdJwt` with verified signature, this module:
1. Hashes each disclosure and verifies it appears in the signed `_sd` list
2. Maps verified disclosures to Harmony's native `SaltedClaim` type

Trust vector injection is out of scope — the caller decides what to
do with the verified claims.

## Module Location

New file `crates/harmony-sdjwt/src/claims.rs` behind a `credential`
feature flag in `harmony-sdjwt`. This adds `harmony-credential` as
an optional dependency.

```toml
[features]
credential = ["dep:harmony-credential", "std"]
```

## Disclosure Hash Verification

Per RFC 9901 §6.3:

```rust
pub fn verify_disclosures(sd_jwt: &SdJwt) -> Result<Vec<&Disclosure>, SdJwtError>
```

For each disclosure in `sd_jwt.disclosures`:
1. Compute `base64url_no_pad(SHA-256(ASCII_bytes(disclosure.raw)))`
   where `disclosure.raw` is the base64url-encoded disclosure string
   as it appeared in the SD-JWT token (the segment between `~`
   separators, per RFC 9901 §4.2.3). The hash input is the ASCII
   bytes of this encoded string, NOT the decoded JSON.
   Uses `sd_jwt.payload.sd_alg` if present, defaulting to `sha-256`.
2. Check that the resulting digest string appears in `sd_jwt.payload.sd`
3. If ALL disclosures match: return them
4. If ANY disclosure does NOT match: return error

Unmatched disclosures are an error (not silently filtered) — a
disclosure that doesn't match any `_sd` entry means the presentation
is malformed or tampered.

### New Error Variant

```rust
SdJwtError::DisclosureHashMismatch
```

### Algorithm Support

Only `sha-256` is supported initially (the RFC default and the only
algorithm used in practice). If `sd_alg` is present and not
`"sha-256"`, return `SdJwtError::UnsupportedAlgorithm`.

## Claim Mapping

```rust
pub fn map_claims(disclosures: &[&Disclosure]) -> Vec<harmony_credential::SaltedClaim>
```

For each verified disclosure:

### type_id Derivation

1. Look up `claim_name` in a static vocabulary dictionary → `type_id`
2. If not found: derive `type_id = BLAKE3(claim_name)[0..2] | 0x8000`
   (high bit marks it as hash-derived, not well-known)
3. Array element disclosures (`claim_name` is `None`): `type_id = 0x0000`

### Static Vocabulary Dictionary

Reserved range `0x0100–0x01FF`:

| Claim name | type_id | Source |
|---|---|---|
| `given_name` | `0x0100` | EUDI PID |
| `family_name` | `0x0101` | EUDI PID |
| `birthdate` | `0x0102` | EUDI PID |
| `age_over_18` | `0x0103` | EUDI PID |
| `nationality` | `0x0104` | EUDI PID |
| `email` | `0x0110` | OpenID4VP |
| `phone_number` | `0x0111` | OpenID4VP |
| `address` | `0x0112` | OpenID4VP |

Implemented as a `const` match statement (no heap, no_std compatible
within the function body).

Hash-derived range: `0x8000–0xFFFF`.

### Salt Conversion

SD-JWT salts are base64url-encoded random bytes (typically 16 bytes
encoded as ~22 characters). Harmony's `SaltedClaim` requires
`[u8; 16]`:

- Base64url-decode the salt string to get raw bytes
- ≤ 16 bytes: zero-pad
- \> 16 bytes: truncate to 16

This is lossy. The original salt string is preserved in
`Disclosure::salt` (and `Disclosure::raw` contains the full
base64url-encoded disclosure for digest recomputation).

### Value

`Disclosure::claim_value` (re-serialized JSON bytes) is used as the
`SaltedClaim::claim.value`.

## Dependencies

```toml
[dependencies]
harmony-credential = { workspace = true, optional = true }
sha2 = { workspace = true }  # for SHA-256 disclosure hashing
```

`sha2` is needed because SD-JWT uses SHA-256 for disclosure hashing
(not BLAKE3). `harmony-crypto` already has `sha2` in its deps, but
we need it directly here since we're not going through harmony-crypto's
API for this specific operation.

## Testing

1. Disclosure hash verification — matching disclosures accepted
2. Unmatched disclosure → `DisclosureHashMismatch` error
3. Known vocabulary mapping — `given_name` → `0x0100`
4. Unknown claim name → hash-derived `type_id` with high bit set
5. Array element → `type_id = 0x0000`
6. Salt truncation (> 16 bytes) and padding (< 16 bytes)
7. Default `sd_alg` (None → SHA-256)

## What This Bead Delivers

- `claims.rs` module in `harmony-sdjwt` (behind `credential` feature)
- `verify_disclosures` function
- `map_claims` function
- `DisclosureHashMismatch` error variant
- Static EUDI PID / OpenID4VP vocabulary dictionary
- ~150 lines production code, ~200 lines tests
- No changes to existing types or API
- `harmony-sdjwt` already in `[workspace.members]` (added by harmony-yfk)
