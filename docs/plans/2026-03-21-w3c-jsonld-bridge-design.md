# W3C JSON-LD Bridge for Verifiable Credentials

**Date:** 2026-03-21
**Status:** Approved
**Scope:** Export-only JSON-LD serialization of harmony-credential types
**Bead:** harmony-8ws

## Overview

Add export-only conversion of `Credential` and `Presentation` to
W3C Verifiable Credentials Data Model 2.0 JSON-LD format. This
enables Harmony credentials to be consumed by external VC wallets
and verifiers without those systems needing to understand Harmony's
compact binary format.

Import (W3C → Harmony) is deferred to harmony-6wt.

## Identity Mapping: did:key

Harmony `IdentityRef` (16-byte hash + CryptoSuite) maps to `did:key`
(self-resolving DID method, no infrastructure needed):

- **Ed25519** (`0x00ed`): `did:key:z6Mk...` (base58btc + multicodec + 32B key)
- **ML-DSA-65** (`0x1211`): `did:key:z...` (same encoding, 1952B key)

The `did:key` is computed at export time from the full public key,
not stored on the credential. The existing
`CryptoSuite::signing_multicodec()` provides the codec prefix.

The export functions require the caller to provide raw public key
bytes (from a key resolver or local storage) since `IdentityRef`
only stores the 16-byte hash, not the full key.

## Output Format

### Credential

```json
{
  "@context": [
    "https://www.w3.org/ns/credentials/v2",
    "https://w3id.org/security/data-integrity/v2"
  ],
  "type": ["VerifiableCredential"],
  "issuer": "did:key:z6Mk...",
  "validFrom": "2026-03-20T00:00:00Z",
  "validUntil": "2027-03-20T00:00:00Z",
  "credentialSubject": {
    "id": "did:key:z6Mk...",
    "claims": [
      { "digest": "base64url-encoded-blake3-hash" }
    ]
  },
  "credentialStatus": {
    "type": "BitstringStatusListEntry",
    "statusListIndex": "42",
    "statusListCredential": "harmony:status-list:<issuer-hash-hex>"
  },
  "proof": {
    "type": "DataIntegrityProof",
    "cryptosuite": "eddsa-2022",
    "proofValue": "base64url-encoded-signature"
  }
}
```

### Presentation (with disclosed claims)

Same structure, wrapped in a Verifiable Presentation envelope.

Disclosed `SaltedClaim` entries appear in `credentialSubject.claims`
with their full content:

```json
{
  "claims": [
    { "typeId": 1, "value": "base64url-encoded-value", "digest": "base64url-hash" },
    { "digest": "base64url-hash-only" }
  ]
}
```

- Disclosed claims include `typeId` (claim type_id as integer),
  `value` (base64url-encoded claim value), and `digest`
- Undisclosed claims include only `digest`
- `salt` is NOT included — it's a privacy primitive for the holder,
  not meaningful to the verifier (who recomputes the digest from
  typeId + value + salt if needed for selective disclosure proofs)

The `nonce` field is always present (Harmony credentials always
have a 16-byte nonce). It is emitted as lowercase hex in
`proof.nonce`.

### Field Mapping

| Harmony field | W3C JSON-LD field | Notes |
|---|---|---|
| `issuer` (IdentityRef) | `issuer` (did:key) | Requires public key bytes |
| `subject` (IdentityRef) | `credentialSubject.id` (did:key) | Requires public key bytes |
| `issued_at` (u64 epoch) | `validFrom` (ISO 8601) | |
| `expires_at` (u64 epoch) | `validUntil` (ISO 8601) | |
| `not_before` (u64 epoch) | `validFrom` (ISO 8601) | Uses `not_before` if > `issued_at` |
| `claim_digests` | `credentialSubject.claims[].digest` | Base64url-encoded BLAKE3 |
| `status_list_index` | `credentialStatus` | Omitted when `None` |
| `signature` | `proof.proofValue` | Base64url-encoded |
| `nonce` | `proof.nonce` | Hex-encoded |
| `proof` (delegation) | `delegationProof` | Harmony-specific field, outside W3C `proof` |

### Naming Conflict: `proof`

W3C VCs use `proof` for the cryptographic signature. Harmony's
`Credential.proof` field is the delegation chain reference (parent
credential's content hash). In the JSON-LD export:

- Harmony's `signature` → W3C `proof` object
- Harmony's `proof` (delegation) → `delegationProof` (Harmony-specific)

### Cryptosuite Selection

| CryptoSuite | W3C cryptosuite | Context URL |
|---|---|---|
| Ed25519 | `eddsa-2022` | `https://w3id.org/security/data-integrity/v2` |
| MlDsa65 | `mldsa65-2025` (draft) | TBD — use a Harmony-specific context for now |
| MlDsa65Rotatable | `mldsa65-2025` | Same as MlDsa65 |

## Module Structure

New module `jsonld.rs` in `harmony-credential`, behind the `jsonld`
feature flag (requires `std`).

### Feature Flag

```toml
[features]
jsonld = ["dep:serde_json", "dep:base64", "dep:bs58", "std"]
```

The `jsonld` feature implies `std` since `serde_json` requires it.

### Dependencies (all optional, behind `jsonld`)

- `serde_json` — JSON serialization
- `base64` — base64url encoding for signatures and digests
- `bs58` — base58btc encoding for did:key (simpler than full `multibase` crate)

## Public API

```rust
/// Export a Credential to W3C VC JSON-LD.
pub fn credential_to_jsonld(
    credential: &Credential,
    issuer_key: &[u8],
    subject_key: &[u8],
) -> Result<serde_json::Value, CredentialError>

/// Export a Presentation to W3C VP JSON-LD.
pub fn presentation_to_jsonld(
    presentation: &Presentation,
    issuer_key: &[u8],
    subject_key: &[u8],
) -> Result<serde_json::Value, CredentialError>

/// Encode an IdentityRef + public key as a did:key string.
pub fn identity_to_did_key(
    identity: &IdentityRef,
    public_key: &[u8],
) -> String
```

Returns `serde_json::Value` — callers serialize to string themselves.

## Testing

1. **Credential export** — verify all required W3C fields present
2. **DID key encoding** — Ed25519 → `did:key:z6Mk...`, ML-DSA-65
   → valid `did:key:z...` with multicodec `0x1211`
3. **Timestamps** — Unix epoch (u64) → ISO 8601 round-trip
4. **Status list** — present when `status_list_index` is `Some`,
   absent when `None`
5. **Presentation** — disclosed claims as plaintext, undisclosed
   as digests
6. **Delegation proof** — `delegationProof` field present when
   `credential.proof` is `Some`, absent when `None`
7. **ML-DSA-65 export** — PQ credential with `mldsa65-2025`

## What This Bead Delivers

- `jsonld.rs` module in `harmony-credential` (behind `jsonld` feature)
- `credential_to_jsonld`, `presentation_to_jsonld`, `identity_to_did_key`
- `serde_json`, `base64`, `multibase` as optional dependencies
- 7 tests covering all export paths
- No changes to existing types or binary format
- Re-exports from `lib.rs` behind the `jsonld` feature

## What This Bead Does NOT Deliver

- Import (W3C → Harmony) — deferred to harmony-6wt
- DID document generation
- Credential schema registry
- Refresh service endpoints
- JSON-LD context hosting
