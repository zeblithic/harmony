# SD-JWT Parser and JWS Verification

**Date:** 2026-03-22
**Status:** Approved
**Scope:** New `harmony-sdjwt` crate — parse and verify SD-JWT compact serialization
**Bead:** harmony-yfk

## Overview

New `no_std + alloc` crate for parsing SD-JWT (Selective Disclosure
for JWTs, RFC 9901) compact serialization and verifying JWS
signatures. This is the ingestion layer for external credentials —
it splits the wire format, decodes headers/payloads/disclosures,
and verifies issuer signatures. Claim mapping to Harmony's native
types is out of scope (that's harmony-b3m).

## Motivation

The EU's EUDI wallet mandate has converged on SD-JWT as the primary
format for high-assurance credentials. SD-JWT's salted-hash
selective disclosure is structurally isomorphic to Harmony's BLAKE3
claim digests. By parsing SD-JWTs, Harmony nodes can consume
credentials from the global VC ecosystem without implementing
heavyweight JSON-LD/RDF processing.

## Crate: `harmony-sdjwt`

Location: `crates/harmony-sdjwt/`
Features: `no_std + alloc` core. `serde_json` gated behind `std`
feature (same pattern as `harmony-credential`'s `jsonld` feature).

Must be added to `[workspace] members` and `[workspace.dependencies]`
in the root `Cargo.toml`.

### Dependencies

- `serde` — serialization (no_std compatible)
- `serde_json` (optional, behind `std` feature) — JSON deserialization
- `base64` — base64url decoding (no_std with `default-features = false`)
- `harmony-identity` — `verify_signature` for JWS verification

### Compact Serialization Format

SD-JWT wire format (RFC 9901 Section 5):

```
<header>.<payload>.<signature>~<disclosure1>~<disclosure2>~...~
```

- `header.payload.signature` — standard JWS (base64url-encoded JSON)
- `~` separates disclosures (base64url-encoded JSON arrays)
- Trailing `~` is optional

Each disclosure is a JSON array: `[salt, claim_name, value]` for
object properties, or `[salt, value]` for array elements.

## Data Types

### JwsHeader

```rust
pub struct JwsHeader {
    pub alg: String,
    pub typ: Option<String>,
    pub kid: Option<String>,
}
```

### JwtPayload

```rust
pub struct JwtPayload {
    pub iss: Option<String>,
    pub sub: Option<String>,
    pub iat: Option<u64>,
    pub exp: Option<u64>,
    pub nbf: Option<u64>,
    pub sd: Vec<String>,
    pub sd_alg: Option<String>,
    pub extra: Vec<(String, serde_json::Value)>,
}
```

`sd` contains the **top-level** `_sd` array entries only
(base64url-encoded hash digests). Nested `_sd` arrays within
sub-objects are captured in `extra` — harmony-b3m can walk them
if needed. `sd_alg` is the hash algorithm (defaults to `sha-256`
per spec). `extra` captures all other claims as key-value pairs.

### Disclosure

```rust
pub struct Disclosure {
    pub raw: String,
    pub salt: String,
    pub claim_name: Option<String>,
    pub claim_value: serde_json::Value,
}
```

`raw` is the original base64url string (needed for hash
verification by downstream crates). `claim_name` is `None` for
array element disclosures.

### SdJwt

```rust
pub struct SdJwt {
    pub header: JwsHeader,
    pub payload: JwtPayload,
    pub signature: Vec<u8>,
    pub disclosures: Vec<Disclosure>,
    /// Raw `base64url(header).base64url(payload)` string preserved
    /// from parsing. Used for signature verification — avoids lossy
    /// re-encoding of deserialized JSON.
    pub signing_input: String,
}
```

## Public API

### parse

```rust
pub fn parse(compact: &str) -> Result<SdJwt, SdJwtError>
```

1. Split the entire input on `~` into segments
2. Take segment 0 as the JWS compact serialization
3. Split the JWS on `.` into header, payload, signature (3 parts)
4. Store the raw `header.payload` substring as `signing_input`
   (before base64url decoding — needed for signature verification)
5. Base64url-decode header, payload, and signature
6. Deserialize header and payload as JSON
7. Extract `_sd` and `_sd_alg` from payload (top-level only)
8. Take segments 1..N, filter empty strings (trailing `~`),
   base64url-decode each as a JSON array → `Disclosure`
9. Return `SdJwt` with all components

### verify

```rust
pub fn verify(
    sd_jwt: &SdJwt,
    suite: CryptoSuite,
    public_key: &[u8],
) -> Result<(), SdJwtError>
```

Uses `sd_jwt.signing_input` (the raw `base64url(header).base64url(payload)`
string preserved during parsing) as the message, then calls
`harmony_identity::verify_signature` with the provided suite and key.
The caller is responsible for resolving the correct public key
(via DID resolution — harmony-95u).

### signing_input

```rust
pub fn signing_input(compact: &str) -> Result<(&str, &str), SdJwtError>
```

Extracts the `header` and `payload` base64url strings from compact
serialization without decoding them. Useful for external verification
workflows.

## Error Types

```rust
pub enum SdJwtError {
    EmptyInput,
    MalformedCompact,
    Base64Error,
    JsonError,
    MissingAlgorithm,
    UnsupportedAlgorithm(String),
    InvalidDisclosure,
    SignatureInvalid(harmony_identity::IdentityError),
}
```

## Algorithm Mapping

JWS `alg` header → `CryptoSuite` for verification:

| JWS alg | CryptoSuite | Notes |
|---------|-------------|-------|
| `EdDSA` | `Ed25519` | Standard Ed25519 |
| `MLDSA65` | `MlDsa65` | Draft, NIST FIPS 204 |
| `ES256` | — | Not natively supported (future) |

`MLDSA65` maps to `MlDsa65` (not `MlDsa65Rotatable`). Rotation
awareness is out of scope for this crate — the caller can override
the suite if needed. Unsupported algorithms return
`UnsupportedAlgorithm`.

## What's NOT in Scope

- **Key Binding (SD-JWT+KB)** — deferred to harmony-lth
- **Disclosure hash verification** — deferred to harmony-b3m
  (comparing SHA-256(salt+name+value) against `_sd` digests)
- **Nested disclosures** — not needed for MVP
- **Decoy digest handling** — transparent (ignored during parse)
- **DID resolution** — deferred to harmony-95u
- **Claim mapping to Harmony types** — deferred to harmony-b3m

## Testing

1. Parse valid compact serialization with disclosures
2. Parse JWS without disclosures (no `~` segments)
3. Header extraction (`alg`, `typ`, `kid`)
4. Payload extraction (`iss`, `sub`, `iat`, `exp`, `_sd`, `_sd_alg`)
5. Disclosure decoding (`[salt, name, value]` array)
6. Array element disclosure (`[salt, value]`, no name)
7. Error: empty input
8. Error: missing signature segment
9. Error: invalid base64url
10. Error: invalid JSON
11. Error: disclosure not an array
12. Verify with Ed25519 — sign test JWT, verify with correct key
13. Verify rejects wrong key
14. ML-DSA-65 verification

## What This Bead Delivers

- `harmony-sdjwt` crate at `crates/harmony-sdjwt/`
- `parse`, `verify`, `signing_input` functions
- `SdJwt`, `JwsHeader`, `JwtPayload`, `Disclosure` types
- `SdJwtError` error enum
- ~200 lines of production code, ~300 lines of tests
- `no_std + alloc` compatible
