# W3C JSON-LD Credential Import

**Date:** 2026-03-23
**Status:** Draft
**Scope:** `harmony-credential` (new `import.rs` module, `jsonld` feature)
**Bead:** harmony-6wt

## Problem

Harmony can export credentials to W3C JSON-LD format (`jsonld.rs`) but
cannot import them. External issuers produce W3C Verifiable Credentials
with standard Data Integrity proofs (`eddsa-jcs-2022`, `mldsa65-jcs-2024`)
that Harmony nodes cannot consume. There is also no way to round-trip
Harmony's own exported VCs back into the binary credential system.

## Solution

New module `import.rs` in `harmony-credential` behind the `jsonld` feature.
A single function parses a JSON-LD VC, verifies the Data Integrity proof
(dispatching on cryptosuite), extracts claims, and produces a Harmony
`Credential` + `Vec<SaltedClaim>`.

Supports two proof families:
- **Harmony custom** (`harmony-eddsa-2022`, `harmony-mldsa65-2025`) —
  reconstructs postcard payload, verifies signature. Round-trip import.
- **Standard JCS** (`eddsa-jcs-2022`, `mldsa65-jcs-2024`) — canonicalizes
  via `harmony-jcs`, hashes, verifies Data Integrity proof. External import.

And two claims formats:
- **Harmony format** (`credentialSubject.claims` array with digests/typeIds/values)
- **External format** (arbitrary JSON-LD fields mapped through vocabulary)

Note: the exact W3C cryptosuite names for ML-DSA JCS proofs may evolve
as the Data Integrity ML-DSA spec matures. The implementation should
accept both `mldsa65-jcs-2024` and `di-mldsa-jcs-2025` (draft names).

## Design Decisions

### Dual proof dispatch (not one-or-the-other)

The `proof.cryptosuite` field determines which verification path is used.
Harmony custom proofs sign postcard bytes; JCS proofs sign canonicalized
JSON. Supporting both maximizes interop — Harmony nodes can consume
credentials from each other AND from external W3C issuers.

### Sentinel salt for imported claims

External VCs don't use Harmony's salted selective disclosure model. Imported
claims use `[0u8; 16]` (all-zeros) as the salt, producing deterministic
digests: `BLAKE3([0u8; 16] || type_id.to_le_bytes() || value)`. This:
- Preserves claim-binding (verifier can recompute digest from disclosed claim)
- Signals "imported, not selectively disclosable" to inspecting code
- Avoids misleading random salts on a credential whose signature doesn't
  cover Harmony's digest format

Note: the `type_id` is encoded as 2-byte little-endian in the digest,
matching the existing `SaltedClaim::digest()` implementation in `claim.rs`.

The sentinel salt also applies to Harmony-format claims that lack an
explicit `salt` field. The current export bridge (`presentation_to_jsonld`)
does not include salts for disclosed claims — so round-tripped Harmony
credentials lose selective disclosure capability. A future export
enhancement (adding `salt` to disclosed claims) would restore full
round-trip fidelity, but is out of scope for this bead.

### Identity hash derivation from DID

Harmony's native `IdentityRef.hash` is `SHA256(enc_key || sign_key)[:16]`,
requiring both the encryption key (X25519 or ML-KEM) and signing key. But
`did:key` only encodes the signing key — there's no encryption key.

For imported credentials, we derive the identity hash from the signing key
alone: `SHA256(sign_key)[:16]`. This produces a different hash than
locally-derived Harmony identities, which is correct — imported external
identities are NOT the same as local Harmony identities, even if they
share the same signing key. The hash serves as a unique identifier within
the imported credential, not as a Harmony network address.

For Ed25519 `did:key` where both the signing and X25519 key are derivable,
a future enhancement could optionally compute the full Harmony hash. But
the single-key derivation is simpler and works uniformly for all suites.

### Claims format detection

If `credentialSubject` contains a `claims` key with an array value, treat
it as Harmony format (structured digests, type IDs, values). Otherwise,
treat each key in `credentialSubject` (except `id`, `type`, and `@context`)
as an external claim, mapped through the vocabulary.

### @context validation

The import does NOT validate `@context` entries. JCS-based Data Integrity
suites operate on the JSON structure directly (no RDF expansion), so
context URIs are informational, not functional. Validating them would
require maintaining a list of known context URIs with no security benefit
for the JCS verification path. The Harmony custom path similarly doesn't
need context validation (it verifies postcard bytes).

### Module in harmony-credential (not a new crate)

Import is the natural inverse of export. Both live in `harmony-credential`
behind the `jsonld` feature, which already gates `serde_json`, `base64`,
and `bs58`. The feature gains `harmony-jcs` and `harmony-identity` as
additional dependencies for canonicalization and DID resolution.

## Architecture

### Public API

```rust
/// Result of importing a W3C JSON-LD Verifiable Credential.
pub struct ImportedCredential {
    /// The Harmony binary credential with verified signature.
    pub credential: Credential,
    /// Extracted claims (with sentinel salt for imported/external VCs).
    pub claims: Vec<SaltedClaim>,
    /// Which proof verification path was used.
    pub proof_type: ImportedProofType,
}

/// How the imported credential's proof was verified.
pub enum ImportedProofType {
    /// Harmony custom cryptosuite (postcard payload signature).
    HarmonyCustom,
    /// Standard W3C JCS Data Integrity proof.
    JcsDataIntegrity,
}

/// Import a W3C Verifiable Credential from JSON-LD format.
///
/// Parses the VC envelope, resolves issuer/subject DIDs, verifies the
/// Data Integrity proof, extracts claims, and produces a Harmony credential.
pub fn import_jsonld_vc(
    vc_json: &serde_json::Value,
    resolver: &impl DidResolver,
) -> Result<ImportedCredential, ImportError>
```

### Import Pipeline

1. **Parse VC envelope**
   - `issuer` → string (DID) or object with `id` field
   - `credentialSubject` → object with optional `id` field
   - `proof` → object with `type`, `cryptosuite`, `proofValue`,
     `verificationMethod`, `created`, `proofPurpose`, `nonce`
   - `validFrom` → `not_before` (ISO 8601 → Unix seconds)
   - `validUntil` → `expires_at` (ISO 8601 → Unix seconds)

2. **Resolve issuer DID**
   - Extract DID from `issuer` field (string or `issuer.id`)
   - `resolver.resolve(issuer_did)` → `ResolvedDid { suite, public_key }`
   - Derive identity hash: `SHA256(public_key)[:16]`
   - Build `IdentityRef { hash, suite }`

3. **Resolve subject DID** (if `credentialSubject.id` present)
   - Same resolution path: resolve → derive hash → build `IdentityRef`
   - If absent, subject = issuer (self-attestation)

4. **Dispatch on cryptosuite**
   - Read `proof.cryptosuite` string
   - `"harmony-eddsa-2022"` | `"harmony-mldsa65-2025"` → Harmony path
   - `"eddsa-jcs-2022"` | `"mldsa65-jcs-2024"` | `"di-mldsa-jcs-2025"` → JCS path
   - Other → `ImportError::UnsupportedCryptosuite`

5. **Verify proof**

   **Harmony custom path:**
   - Extract timestamp, nonce, claim digests from the VC fields
   - Use `Credential::signable_bytes()` (or equivalent `pub(crate)` method)
     to reconstruct the postcard payload. Do NOT duplicate the serialization
     logic — reuse the existing builder path.
   - Decode `proofValue` (multibase base58btc → strip leading 'z', then
     base58 decode → signature bytes)
   - Verify signature with issuer's public key

   **JCS path** (W3C Data Integrity Cryptosuites v1.0 §4.3):
   - Clone the VC document, remove `proof` key → `unsigned_doc`
   - Clone the `proof` object, remove `proofValue` key → `proof_options`
   - `proof_hash = SHA-256(harmony_jcs::canonicalize(&proof_options))`
   - `doc_hash = SHA-256(harmony_jcs::canonicalize(&unsigned_doc))`
   - `verify_bytes = proof_hash || doc_hash` (64 bytes)
   - Decode `proofValue` (multibase base58btc → signature bytes)
   - Verify signature over `verify_bytes` with issuer's public key

6. **Extract claims**

   **Harmony format** (detected by `credentialSubject.claims` array):
   - Each element has `digest` (base64url → `[u8; 32]`)
   - Disclosed elements may also have `typeId` (u16), `value` (base64url)
   - If `salt` field present: decode hex → `[u8; 16]` (full round-trip)
   - If `salt` absent: use sentinel `[0u8; 16]` (current export doesn't
     include salt — disclosed claims lose selective disclosure on round-trip)
   - Build `SaltedClaim` for disclosed, digest-only for undisclosed

   **External format** (arbitrary fields):
   - For each key in `credentialSubject` (skip `id`, `type`, `@context`):
     - Map name through vocabulary: known names → fixed type ID,
       unknown → `BLAKE3(name)[0..2] | 0x8000`
     - Value = compact JSON serialization of the field value (bytes)
     - Salt = `[0u8; 16]` (sentinel)
   - Build `SaltedClaim` for each field
   - Compute `claim_digests` via `SaltedClaim::digest()` per claim

7. **Build Credential**
   - `issuer` / `subject` from resolved DIDs (step 2/3)
   - `claim_digests` from extracted claims
   - `not_before` / `expires_at` from parsed timestamps
   - `issued_at` from `proof.created` (ISO 8601 → Unix seconds)
   - `nonce` from `proof.nonce` (hex → `[u8; 16]`, or zeros if absent)
   - `signature` from decoded `proofValue`
   - `proof` (delegation parent) = None for imported credentials

### Error Types

```rust
pub enum ImportError {
    /// JSON structure doesn't match W3C VC format.
    MalformedVc(String),
    /// DID resolution failed.
    DidResolution(DidError),
    /// Proof cryptosuite not supported.
    UnsupportedCryptosuite(String),
    /// Data Integrity proof verification failed.
    ProofInvalid,
    /// Timestamp parsing failed.
    InvalidTimestamp(String),
    /// Claim extraction failed.
    ClaimError(String),
    /// Base encoding (base58, base64url) failed.
    EncodingError(String),
}
```

## Changes

### harmony-credential

New file: `src/import.rs`
- `ImportedCredential`, `ImportedProofType`, `ImportError` types
- `import_jsonld_vc()` main entry point
- Internal helpers: `parse_vc_envelope()`, `verify_harmony_proof()`,
  `verify_jcs_proof()`, `extract_harmony_claims()`,
  `extract_external_claims()`, `parse_iso8601()`,
  `derive_import_identity_hash()`

Modify `src/lib.rs`: export `import` module under `jsonld` feature.

Modify `Cargo.toml`: add `harmony-jcs` and `harmony-identity` to
`jsonld` feature dependencies.

### Dependencies added to `jsonld` feature

```toml
harmony-jcs = { workspace = true }
harmony-identity = { workspace = true, features = ["std"] }
```

`harmony-crypto` is already a dependency (for SHA-256 in JCS proof hash
and BLAKE3 for claim digest computation).

## Testing

- `import_harmony_eddsa_roundtrip` — export then import a Harmony VC
- `import_harmony_mldsa65_roundtrip` — same for ML-DSA-65
- `import_jcs_eddsa_proof` — construct a JCS-signed VC, verify import
- `import_jcs_mldsa65_proof` — same for ML-DSA-65
- `import_external_claims_vocabulary` — external VC with known claim names
- `import_external_claims_unknown` — unknown claim names get hash-derived IDs
- `import_sentinel_salt` — verify imported claims use `[0u8; 16]` salt
- `import_missing_subject` — subject defaults to issuer
- `import_invalid_proof_rejected` — tampered proofValue fails
- `import_unsupported_cryptosuite` — unknown suite returns error
- `import_malformed_vc_rejected` — missing required fields
- `import_timestamps_parsed` — ISO 8601 validFrom/validUntil → Unix seconds
- `import_identity_hash_derivation` — verify SHA256(sign_key)[:16] hash

## What is NOT in Scope

- RDFC-2022 canonicalization (much more complex than JCS, separate bead)
- VP (Verifiable Presentation) import (only VC import)
- Status list verification during import (checked at credential verification time)
- Credential re-signing (imported credentials keep the original signature)
- `did:web` resolution (requires network fetch — rejected by resolver)
- Export enhancement to include salts (future bead for full round-trip fidelity)
- `@context` validation (JCS suites don't need RDF expansion)
