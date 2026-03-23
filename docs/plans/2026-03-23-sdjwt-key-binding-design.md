# SD-JWT Key Binding (KB-JWT)

**Date:** 2026-03-23
**Status:** Approved
**Scope:** Verify KB-JWT holder binding in SD-JWT presentations
**Bead:** harmony-lth

## Overview

Verify the optional Key Binding JWT (KB-JWT) in SD-JWT presentations.
The KB-JWT proves the holder possesses a specific private key, binding
the presentation to a verifier session via nonce and audience. Required
by EUDI wallet flows to prevent replay attacks.

## Module Location

`crates/harmony-sdjwt/src/key_binding.rs`, behind the `std` feature
(KB-JWT parsing requires JSON).

## KB-JWT Structure (RFC 9901 §11.6)

The KB-JWT is a compact JWS appended after the last disclosure:

```
<issuer-jwt>~<disc1>~...~<discN>~<kb-jwt>
```

**Header:**
- `typ`: MUST be `"kb+jwt"`
- `alg`: signing algorithm

**Payload:**
- `aud`: intended audience (verifier identifier)
- `nonce`: verifier-provided nonce (prevents replay)
- `iat`: issued-at (Unix epoch seconds)
- `sd_hash`: `base64url(SHA-256(sd-jwt-without-kb-jwt))` — binds
  the KB-JWT to the specific disclosure set

## Public API

```rust
pub fn verify_key_binding(
    sd_jwt: &SdJwt,
    holder_key: &[u8],
    holder_suite: CryptoSuite,
    expected_nonce: &str,
    expected_aud: &str,
    now: u64,
) -> Result<(), SdJwtError>
```

### Algorithm

1. Check `sd_jwt.key_binding_jwt` is `Some` → else `KeyBindingInvalid`
2. Split KB-JWT on `.` into header, payload, signature (3 parts)
3. Base64url-decode and parse header JSON
4. Verify `typ == "kb+jwt"` (case-insensitive, per RFC 7515 §4.1.9)
5. Base64url-decode and parse payload JSON
6. Verify `nonce == expected_nonce` → else `KeyBindingInvalid`
7. Verify `aud == expected_aud` → else `KeyBindingInvalid`
8. Verify `iat <= now + 60` (not in the future with 60s skew)
9. Compute `sd_hash`:
   - Reconstruct the SD-JWT without the KB-JWT: take everything
     from `sd_jwt.signing_input`, the signature, and all disclosures
   - Actually: the `sd_hash` covers the entire compact serialization
     up to (but not including) the final KB-JWT segment. This is
     available by stripping the last `~kb-jwt` from the original
     compact string. Since `SdJwt` stores `signing_input` and
     `disclosures[].raw`, we can reconstruct it.
   - SHA-256 hash, base64url encode, compare to payload `sd_hash`
10. Verify KB-JWT signature: `verify_signature(holder_suite, holder_key,
    kb_signing_input, kb_signature)`

### sd_hash Reconstruction

The SD-JWT without KB-JWT is: `<jws>~<disc1>~...~<discN>~`

We reconstruct it from `SdJwt` fields:
```
{signing_input}.{base64url(signature)}~{disc1.raw}~...~{discN.raw}~
```

This avoids needing the original compact string.

## New Error Variant

```rust
#[cfg(feature = "std")]
KeyBindingInvalid(String),
```

Descriptive message for each failure mode. Signature failures use
existing `SignatureInvalid`.

## Holder Key Discovery

The issuer's payload `cnf` claim contains the holder's key. Extracting
it is the caller's responsibility — `verify_key_binding` takes raw
key bytes. The `cnf` JWK can be resolved via `resolve_did_jwk` or
extracted directly from the payload's `extra` field.

## Dependencies

- `sha2` — SHA-256 for sd_hash (already available via `credential` feature)
- `base64` — base64url encoding (already available)
- `serde_json` — JSON parsing (behind `std`)

sha2 needs to be available under the `std` feature (not just `credential`).
The implementation should check if it's already accessible or add it
as a direct optional dep behind `std`.

## Testing

1. Valid KB-JWT — full round-trip verification
2. Missing KB-JWT → `KeyBindingInvalid`
3. Wrong `typ` → `KeyBindingInvalid`
4. Wrong `nonce` → `KeyBindingInvalid`
5. Wrong `aud` → `KeyBindingInvalid`
6. Wrong `sd_hash` → `KeyBindingInvalid`
7. Wrong holder key → `SignatureInvalid`
8. Future `iat` → `KeyBindingInvalid`

## What This Bead Delivers

- `key_binding.rs` module in `harmony-sdjwt` (behind `std`)
- `verify_key_binding` function
- `KeyBindingInvalid(String)` error variant
- ~100 lines production code, ~200 lines tests
- No changes to existing types
