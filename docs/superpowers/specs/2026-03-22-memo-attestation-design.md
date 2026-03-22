# Signed Memo Attestations: Memoize Once, Trust Everywhere

**Bead:** harmony-m5y
**Date:** 2026-03-22
**Status:** Draft

## Problem

Deterministic computations (Nix builds, compiler runs, data transforms) are
repeated across devices that could share results. The CAS layer stores content
by hash, but there's no way to attest that a given output hash is the correct
result of a given input — you either trust the source or recompute from scratch.

## Solution

A signed memo layer where any identity can attest: "I computed f(input) and
got output." Multiple signers independently verify and add their signatures,
building a pile of attestations. Consumers trust the output if enough signers
they trust have attested. The memo format is generic — Nix derivations are
just one kind of input. Any deterministic function works.

The key insight: once you've verified a computation, signing a memo costs
nothing and helps everyone else. "You bothered to check, so why not add your
name to the pile?"

## Design Decisions

### Generic memo format (not build-specific)

A memo is `input_cid → output_cid + signature`. It doesn't know what the
computation was. A Nix build, a video transcode, a proof verification —
all produce the same structure. The input CID references a computation
recipe stored in CAS; the output CID references the result. Both are just
ContentIds.

### Independent credentials per signer

Each signer issues their own `Credential` with the same claim. No
coordination between signers, no ordering dependencies. The verifier
collects attestations and decides their own trust threshold ("I need 1
signer from Mozilla" or "I need 3 signers I personally trust"). Uses
the existing Credential infrastructure without modification.

### Namespace encodes the trust model

`harmony/memo/<input>/<output>/sign/<signer>` makes disagreements
structurally visible. Multiple `<output>` paths under the same `<input>`
means signers disagree — the fork is in the path topology, not hidden in
data. The signer ID in the path means publishing is self-authenticating.

### CID references only (not stored content)

The memo path uses CID hex strings but doesn't require content to be
stored locally. The memo layer is about attestations, not storage. Content
availability is handled by the existing CAS + Bloom filter infrastructure.
Once you know the output CID, you fetch it from the closest/fastest source.

### Separate Bloom filter for memos

A dedicated memo filter advertises which input CIDs you have attestations
for. This answers "does this peer know about computations for input X?"
separately from "does this peer have content Y?" The consumer queries the
Zenoh namespace to learn specific outputs and signers.

The filter advertises input CIDs (not output CIDs) because you don't know
the output until you've found a memo — that's the whole point.

## Data Model

### New crate: `harmony-memo`

```rust
/// A signed attestation: "I computed f(input) and got output."
pub struct Memo {
    pub input: ContentId,
    pub output: ContentId,
    pub credential: Credential,
}
```

**Claim encoding:** The input→output mapping is encoded as a `SaltedClaim`
inside the credential. Claim `type_id = 0x4D45` ("ME"). Claim value is
`input.to_bytes() || output.to_bytes()` (64 bytes). This reuses selective
disclosure and the existing verification pipeline.

**MemoStore:**

```rust
pub struct MemoStore {
    /// Memos keyed by input CID. Each input may have multiple memos
    /// (different outputs from different signers, or same output from
    /// multiple signers).
    by_input: HashMap<ContentId, Vec<Memo>>,
}
```

Methods:
- `insert(memo: Memo)` — add a memo, deduplicate by (input, output, signer)
  where signer = `credential.issuer.hash`
- `get_by_input(input: &ContentId) -> &[Memo]` — all memos for an input
- `get_by_input_and_signer(input, signer) -> Option<&Memo>` — specific
- `outputs_for_input(input) -> Vec<(ContentId, Vec<IdentityRef>)>` — grouped
  by output, with list of signers for each
- `input_cids() -> impl Iterator<Item = &ContentId>` — for Bloom filter

### Memo construction and verification

```rust
/// Create a signed memo. Caller provides RNG (sans-I/O pattern).
/// Issuer == subject (self-attestation: "I computed this").
/// Signer identity = credential.issuer.
pub fn create_memo(
    input: ContentId,
    output: ContentId,
    identity: &PqPrivateIdentity,  // or PrivateIdentity
    rng: &mut impl rand_core::CryptoRngCore,
    now: u64,
    expires_at: u64,
) -> Memo

/// Verify a memo's credential signature, claim encoding, and time bounds.
/// Uses a no-op StatusListResolver (memos don't use revocation lists —
/// distrust is expressed by not endorsing, not by revoking individual memos).
pub fn verify_memo(
    memo: &Memo,
    now: u64,
    keys: &impl CredentialKeyResolver,
) -> Result<(), MemoError>
```

**Error type:**

```rust
pub enum MemoError {
    /// Credential signature invalid or expired.
    Credential(CredentialError),
    /// Claim type_id is not 0x4D45 or claim value is not 64 bytes.
    ClaimDecodingFailed,
    /// Decoded input/output CIDs don't match the Memo's fields.
    InputOutputMismatch,
}
```

**Wire format:** Version byte prefix (starting at `1`) + postcard serialization
of `(input, output, credential)`. Enables future format evolution.

## Zenoh Namespace

**Path structure:** `harmony/memo/<input_hex>/<output_hex>/sign/<signer_hex>`

- `<input_hex>` — 64 hex chars (32-byte ContentId)
- `<output_hex>` — 64 hex chars (32-byte ContentId)
- `<signer_hex>` — 32 hex chars (16-byte IdentityHash)

**Payload:** Serialized `Credential` (postcard format).

**Discovery queries:**
- `harmony/memo/<input>/**` — all memos for this input
- `harmony/memo/<input>/<output>/sign/**` — all signers for this output
- `harmony/memo/<input>/*/sign/<trusted_id>` — what does a trusted signer
  say this input produces? (fastest path to a trusted answer)

**Conflict detection:** Multiple `<output>` paths under the same `<input>`
means disagreement — structurally visible in the path topology.

**Namespace helpers in `harmony-zenoh`:**

```rust
pub mod memo {
    pub const PREFIX: &str = "harmony/memo";
    pub const SUB: &str = "harmony/memo/**";

    pub fn sign_key(input_hex: &str, output_hex: &str, signer_hex: &str) -> String;
    pub fn input_query(input_hex: &str) -> String;
    pub fn output_query(input_hex: &str, output_hex: &str) -> String;
    pub fn signer_query(input_hex: &str, signer_hex: &str) -> String;
}
```

## Bloom Filter

**Namespace:** `harmony/filters/memo/{node_addr}`

Advertises input CIDs that the node has attestations for. Same `BloomFilter`
infrastructure as content filters (wire format, parameters, broadcast
interval). Maintained by the `MemoStore` — input CID inserted on each
`insert()`.

**Filter lifecycle:** Bloom filters are append-only — removed or superseded
memos leave stale entries. The filter is rebuilt periodically from
`MemoStore::input_cids()` on the same schedule as content filter rebuilds
(every `filter_broadcast_ticks`). This keeps false positive rates bounded.

**Filter subscription key in `harmony-zenoh`:**

```rust
pub mod filters {
    pub const MEMO_PREFIX: &str = "harmony/filters/memo";
    pub fn memo_key(node_addr: &str) -> String;
    pub const MEMO_SUB: &str = "harmony/filters/memo/*";
}
```

## CLI

**New subcommands on `harmony`:**

```
harmony memo sign --input <cid_hex> --output <cid_hex>
    Sign a memo attesting input produces output. Uses node identity.

harmony memo list <input_cid_hex>
    Show all known memos for this input (outputs + signers).

harmony memo verify <input_cid_hex>
    Show trusted outputs with signer counts.
```

## Node Runtime Integration

**New event/action variants in `RuntimeEvent` / `RuntimeAction`:**

- `MemoReceived { input, output, credential }` — peer published a memo
- `PublishMemo { key_expr, payload }` — publish signed memo to Zenoh

**Event loop:** Subscribes to `harmony/memo/**`. Incoming memos pushed to
runtime via `MemoReceived`. The runtime stores in `MemoStore` and updates
the memo Bloom filter.

## File Changes

| File | Change |
|------|--------|
| `Cargo.toml` (root) | Add harmony-memo to workspace members |
| `crates/harmony-memo/Cargo.toml` | New crate |
| `crates/harmony-memo/src/lib.rs` | Memo struct, claim encoding, create/verify |
| `crates/harmony-memo/src/store.rs` | MemoStore: insert, query, dedup |
| `crates/harmony-zenoh/src/namespace.rs` | Add `pub mod memo` + filter keys |
| `crates/harmony-node/src/main.rs` | `harmony memo sign/list/verify` subcommands |
| `crates/harmony-node/src/runtime.rs` | MemoStore, new event/action variants |
| `crates/harmony-node/src/event_loop.rs` | Zenoh subscription for memo namespace |

## What is NOT in Scope

- No Nix integration (bead `harmony-s1i`)
- No slashing/whistleblower protocol (bead `harmony-bkz`)
- No social graph propagation for discovery (bead `harmony-boa`)
- No persistence of memos across restarts (in-memory MemoStore for v1)
- No trust threshold logic (consumer decides their own policy)
- No content storage — memos reference CIDs, content lives in CAS

## Testing

**Unit tests for harmony-memo:**
- Create memo, verify signature
- Reject memo with invalid signature
- Reject expired memo
- MemoStore insert and dedup (same signer, same input+output = no dup)
- MemoStore query by input, by signer
- Claim encoding/decoding round-trip
- outputs_for_input grouping

**Integration tests:**
- CLI: `harmony memo sign` creates and publishes
- Zenoh namespace: published memo discoverable via subscription
- Bloom filter: input CID appears in memo filter after signing
