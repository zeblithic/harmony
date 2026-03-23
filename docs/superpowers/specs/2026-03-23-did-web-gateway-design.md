# did:web Gateway Service Design

## Goal

Enable constrained mesh nodes to verify enterprise/government issuer signatures by delegating `did:web` HTTPS resolution to a full harmony-node gateway. The gateway fetches DID Documents, extracts public keys, and returns them over Zenoh — so mesh nodes never need a TLS stack.

## Architecture

The implementation spans two crates:

- **harmony-identity** gains DID Document parsing (pure logic, no I/O) and a pluggable `WebDidFetcher` trait for the HTTP boundary.
- **harmony-node** implements the fetcher with `reqwest`, declares a Zenoh queryable, and caches results with a configurable TTL.

The sans-I/O pattern is preserved: all DID Document parsing and key extraction lives in the library crate. The binary crate provides the HTTP implementation and Zenoh wiring.

## Data Types

### ResolvedDidDocument

New type in `harmony-identity/src/did.rs`:

```rust
pub struct ResolvedDidDocument {
    pub id: String,
    pub verification_methods: Vec<ResolvedDid>,
}
```

`ResolvedDid` is unchanged (`suite: CryptoSuite`, `public_key: Vec<u8>`). The document wraps multiple resolved keys with the DID identifier.

### WebDidFetcher Trait

Sans-I/O boundary for HTTP:

```rust
pub trait WebDidFetcher {
    fn fetch(&self, url: &str) -> Result<Vec<u8>, DidError>;
}
```

The library parses the response bytes. The caller provides the HTTP implementation. This keeps harmony-identity free of `reqwest`/`tokio` dependencies.

### DidResolver Trait Extension

```rust
pub trait DidResolver {
    fn resolve(&self, did: &str) -> Result<ResolvedDid, DidError>;
    fn resolve_document(&self, did: &str) -> Result<ResolvedDidDocument, DidError>;
}
```

`DefaultDidResolver` implements `resolve_document` by wrapping the single-key result. A new `WebDidResolver` accepts a `WebDidFetcher` and implements both methods, delegating HTTP to the injected fetcher.

## DID Document Parsing

New file: `harmony-identity/src/did_document.rs`.

### DID-to-URL Mapping

Per the W3C did:web specification:

| DID | URL |
|---|---|
| `did:web:example.com` | `https://example.com/.well-known/did.json` |
| `did:web:example.com:issuers:1` | `https://example.com/issuers/1/did.json` |
| `did:web:example.com%3A8443` | `https://example.com:8443/.well-known/did.json` |

Colons after the method-specific identifier are path separators. Percent-encoded characters are decoded for the domain component.

### Verification Method Extraction

Scan the `verificationMethod` array in the DID Document JSON. Two key formats are supported:

- **`publicKeyJwk`** — reuses JWK parsing from `resolve_did_jwk`
- **`publicKeyMultibase`** — reuses multibase/multicodec parsing from `resolve_did_key`

Methods with unsupported key types (P-256, RSA, etc.) are skipped without error — a document may contain keys we don't support alongside ones we do.

### Validation

- The `id` field in the document must match the DID being resolved (prevents document substitution attacks).
- At least one supported verification method must be present. If none, return `DidError::NoSupportedKeys`.

All parsing is pure function logic — no I/O, fully testable with fixture JSON strings.

## Gateway Service (harmony-node)

### Zenoh Queryable

Declared at startup on `harmony/identity/web/**`. A mesh node resolving `did:web:example.com:issuers:1` sends a Zenoh query to `harmony/identity/web/example.com/issuers/1`.

### Query Flow

1. Extract domain + path segments from the Zenoh key expression.
2. Reconstruct the DID: `did:web:{domain}:{path_segments_joined_by_colon}`.
3. Check TTL cache — if hit and not expired, return cached result.
4. Call `WebDidResolver::resolve_document()` with a `reqwest`-backed fetcher.
5. Cache the result with configurable TTL (default 300 seconds).
6. Serialize `ResolvedDidDocument` via postcard and reply.

### Fetcher Implementation

A struct wrapping `reqwest::blocking::Client` that implements `WebDidFetcher`. Enforces HTTPS-only — rejects `http://` URLs.

### Cache

`HashMap<String, (ResolvedDidDocument, u64)>` keyed by DID string, with `u64` expiry timestamp (Unix seconds). Checked on each query, lazy eviction (no background task). TTL configurable via the node's TOML config.

### Error Handling

If the fetch fails or the document has no supported keys, the gateway replies with an empty Zenoh payload (convention for "not found") and logs with `tracing::warn!`.

## Response Format

No gateway signature on responses. Trust relies on the authenticated Zenoh transport between mesh node and gateway. Multi-gateway attestation and trust-vector integration are future work.

The response payload is `ResolvedDidDocument` serialized via postcard, consistent with other Zenoh message formats in harmony-node.

## Testing Strategy

### Unit Tests (harmony-identity)

- DID-to-URL mapping: root domain, path-based, percent-encoded port
- DID Document parsing: fixture JSON with Ed25519, ML-DSA-65, mixed/unsupported keys, malformed documents
- `resolve_document` returning multiple keys from a multi-method document
- `resolve` returning first supported key from the same document
- Validation: `id` mismatch rejection, no-supported-keys error
- Reuse of existing `publicKeyJwk` and `publicKeyMultibase` parsing paths

### Integration Tests (harmony-node)

- Test `WebDidFetcher` returning fixture JSON (no real HTTP)
- Full query flow: DID string -> URL construction -> document parse -> `ResolvedDidDocument`
- Cache behavior: second call returns cached, expired entry triggers re-fetch
- Error cases: fetch failure, invalid JSON, HTTPS enforcement

No real network tests. The `WebDidFetcher` trait boundary makes mocking clean.

## Scope Exclusions

- **No gateway response signing** — deferred to trust-vector integration
- **No multi-gateway consensus** — single gateway per mesh segment for now
- **No did:web DID Document caching persistence** — in-memory only, lost on restart
- **No background cache refresh** — lazy eviction on query
- **No real HTTP integration tests** — mocked via `WebDidFetcher` trait
