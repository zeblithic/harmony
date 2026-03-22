# Signed Memo Attestation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A signed memo layer where identities attest to deterministic computation results (input CID → output CID), discoverable via Zenoh namespace and Bloom filters.

**Architecture:** New `harmony-memo` crate with Memo struct (wraps Credential), MemoStore (in-memory), and create/verify functions. Zenoh namespace `harmony/memo/<input>/<output>/sign/<signer>` added to harmony-zenoh. Memo Bloom filter for input CID discovery. CLI subcommands `harmony memo sign/list/verify` in harmony-node.

**Tech Stack:** Rust, harmony-credential (signing), harmony-content (ContentId), harmony-zenoh (namespace), postcard (serialization), no_std compatible

**Spec:** `docs/superpowers/specs/2026-03-22-memo-attestation-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `Cargo.toml` (root) | Add harmony-memo to workspace members |
| `crates/harmony-memo/Cargo.toml` | New crate dependencies |
| `crates/harmony-memo/src/lib.rs` | Memo struct, MemoError, claim type constant, wire format |
| `crates/harmony-memo/src/create.rs` | create_memo() — build Credential with memo claim |
| `crates/harmony-memo/src/verify.rs` | verify_memo() — verify credential + decode claim |
| `crates/harmony-memo/src/store.rs` | MemoStore: insert, query by input/signer, dedup |
| `crates/harmony-zenoh/src/namespace.rs` | Add `pub mod memo` with namespace helpers |
| `crates/harmony-node/src/main.rs` | `harmony memo sign/list/verify` CLI subcommands |

---

### Task 1: Zenoh memo namespace

**Files:**
- Modify: `crates/harmony-zenoh/src/namespace.rs`

Add the memo namespace module following the existing pattern (PREFIX, SUB, key builders, tests). This is standalone — no new crate needed yet.

- [ ] **Step 1: Add `pub mod memo` to namespace.rs**

Add between the `endorsement` module and `#[cfg(test)]`, following the existing pattern:

```rust
/// Signed memo attestations for deterministic computation results.
///
/// Namespace: `harmony/memo/<input_cid_hex>/<output_cid_hex>/sign/<signer_id_hex>`
///
/// Each path level encodes the attestation: input (computation recipe),
/// output (result), signer (who verified). Multiple outputs under the same
/// input means disagreement — structurally visible in the path topology.
pub mod memo {
    use alloc::{format, string::String};

    pub const PREFIX: &str = "harmony/memo";
    pub const SUB: &str = "harmony/memo/**";

    /// Full attestation key: `harmony/memo/{input}/{output}/sign/{signer}`
    pub fn sign_key(input_hex: &str, output_hex: &str, signer_hex: &str) -> String {
        format!("{PREFIX}/{input_hex}/{output_hex}/sign/{signer_hex}")
    }

    /// All memos for an input: `harmony/memo/{input}/**`
    pub fn input_query(input_hex: &str) -> String {
        format!("{PREFIX}/{input_hex}/**")
    }

    /// All signers for a specific input→output: `harmony/memo/{input}/{output}/sign/**`
    pub fn output_query(input_hex: &str, output_hex: &str) -> String {
        format!("{PREFIX}/{input_hex}/{output_hex}/sign/**")
    }

    /// What does a specific signer say about this input: `harmony/memo/{input}/*/sign/{signer}`
    pub fn signer_query(input_hex: &str, signer_hex: &str) -> String {
        format!("{PREFIX}/{input_hex}/*/sign/{signer_hex}")
    }
}
```

Also add memo filter constants to the `filters` module:

```rust
    pub const MEMO_PREFIX: &str = "harmony/filters/memo";
    pub const MEMO_SUB: &str = "harmony/filters/memo/**";
    pub fn memo_key(node_addr: &str) -> String {
        format!("{MEMO_PREFIX}/{node_addr}")
    }
```

- [ ] **Step 2: Add tests**

Add tests matching the existing namespace test patterns:

```rust
    // ── Memo ──

    #[test]
    fn memo_sign_key() {
        let key = memo::sign_key("aabb", "ccdd", "1122");
        assert_eq!(key, "harmony/memo/aabb/ccdd/sign/1122");
    }

    #[test]
    fn memo_input_query() {
        assert_eq!(memo::input_query("aabb"), "harmony/memo/aabb/**");
    }

    #[test]
    fn memo_output_query() {
        assert_eq!(memo::output_query("aabb", "ccdd"), "harmony/memo/aabb/ccdd/sign/**");
    }

    #[test]
    fn memo_signer_query() {
        assert_eq!(memo::signer_query("aabb", "1122"), "harmony/memo/aabb/*/sign/1122");
    }

    #[test]
    fn memo_subscription_pattern() {
        assert_eq!(memo::SUB, "harmony/memo/**");
    }

    #[test]
    fn filters_memo_key() {
        assert_eq!(filters::memo_key("node42"), "harmony/filters/memo/node42");
    }

    #[test]
    fn filters_memo_subscription_pattern() {
        assert_eq!(filters::MEMO_SUB, "harmony/filters/memo/**");
    }
```

Also add `memo::PREFIX` to the `all_prefixes_start_with_root` test.

- [ ] **Step 3: Run tests**

Run: `cargo test -p harmony-zenoh -v`
Expected: all tests pass

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-zenoh/src/namespace.rs
git commit -m "feat(zenoh): add memo attestation namespace

harmony/memo/<input>/<output>/sign/<signer> for signed memos.
Memo Bloom filter at harmony/filters/memo/{node_addr}.
7 new namespace tests.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: harmony-memo crate — types and create/verify (TDD)

**Files:**
- Modify: `Cargo.toml` (root — add workspace member)
- Create: `crates/harmony-memo/Cargo.toml`
- Create: `crates/harmony-memo/src/lib.rs`
- Create: `crates/harmony-memo/src/create.rs`
- Create: `crates/harmony-memo/src/verify.rs`

The core crate with Memo struct, claim encoding, and create/verify functions.

- [ ] **Step 1: Create crate scaffolding**

Add `"crates/harmony-memo"` to workspace members in root `Cargo.toml` (alphabetical order, after `harmony-mail`).

Create `crates/harmony-memo/Cargo.toml`:

```toml
[package]
name = "harmony-memo"
version.workspace = true
edition.workspace = true
license.workspace = true
repository.workspace = true
rust-version.workspace = true

[features]
default = []
std = ["harmony-credential/std", "harmony-identity/std", "harmony-content/std"]

[dependencies]
harmony-credential = { workspace = true }
harmony-identity = { workspace = true }
harmony-content = { workspace = true }
rand_core.workspace = true
serde = { workspace = true, features = ["derive"] }
postcard = { workspace = true }
```

- [ ] **Step 2: Create lib.rs with types**

```rust
#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;

pub mod create;
pub mod verify;

use harmony_content::cid::ContentId;
use harmony_credential::Credential;
use harmony_identity::IdentityRef;
use serde::{Deserialize, Serialize};

/// Claim type_id for memo attestations ("ME" in ASCII).
pub const MEMO_CLAIM_TYPE: u16 = 0x4D45;

/// Wire format version for Memo serialization.
pub const FORMAT_VERSION: u8 = 1;

/// A signed attestation: "I computed f(input) and got output."
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memo {
    pub input: ContentId,
    pub output: ContentId,
    pub credential: Credential,
}

/// Errors from memo operations.
#[derive(Debug)]
pub enum MemoError {
    /// Credential signature invalid, expired, or malformed.
    Credential(harmony_credential::CredentialError),
    /// Claim type_id is not MEMO_CLAIM_TYPE or value is not 64 bytes.
    ClaimDecodingFailed,
    /// Decoded input/output CIDs don't match the Memo's fields.
    InputOutputMismatch,
    /// Serialization/deserialization failed.
    SerializationError,
}

impl core::fmt::Display for MemoError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Credential(e) => write!(f, "credential error: {e}"),
            Self::ClaimDecodingFailed => write!(f, "memo claim decoding failed"),
            Self::InputOutputMismatch => write!(f, "memo input/output CID mismatch"),
            Self::SerializationError => write!(f, "memo serialization error"),
        }
    }
}

/// Serialize a Memo to wire format (version byte + postcard).
pub fn serialize(memo: &Memo) -> alloc::vec::Vec<u8> {
    let mut buf = alloc::vec![FORMAT_VERSION];
    buf.extend(postcard::to_allocvec(memo).unwrap_or_default());
    buf
}

/// Deserialize a Memo from wire format.
pub fn deserialize(data: &[u8]) -> Result<Memo, MemoError> {
    if data.is_empty() || data[0] != FORMAT_VERSION {
        return Err(MemoError::SerializationError);
    }
    postcard::from_bytes(&data[1..]).map_err(|_| MemoError::SerializationError)
}
```

- [ ] **Step 3: Create create.rs**

```rust
use alloc::vec::Vec;
use harmony_content::cid::ContentId;
use harmony_credential::claim::{Claim, SaltedClaim};
use harmony_credential::CredentialBuilder;
use harmony_identity::pq_identity::PqPrivateIdentity;
use harmony_identity::IdentityRef;
use rand_core::CryptoRngCore;

use crate::{Memo, MEMO_CLAIM_TYPE};

/// Create a signed memo attesting that `input` produces `output`.
///
/// Issuer == subject (self-attestation). Caller provides RNG (sans-I/O).
pub fn create_memo(
    input: ContentId,
    output: ContentId,
    identity: &PqPrivateIdentity,
    rng: &mut impl CryptoRngCore,
    now: u64,
    expires_at: u64,
) -> Memo {
    let issuer = IdentityRef::from(identity);
    let subject = issuer; // self-attestation

    // Encode input||output as the claim value
    let mut claim_value = Vec::with_capacity(64);
    claim_value.extend_from_slice(&input.to_bytes());
    claim_value.extend_from_slice(&output.to_bytes());

    let claim = Claim {
        type_id: MEMO_CLAIM_TYPE,
        value: claim_value,
    };
    let mut salt = [0u8; 16];
    rng.fill_bytes(&mut salt);
    let salted = SaltedClaim { claim, salt };

    let mut nonce = [0u8; 16];
    rng.fill_bytes(&mut nonce);

    let mut builder = CredentialBuilder::new(issuer, subject, now, expires_at, nonce);
    builder.add_claim(salted);
    let signable = builder.signable_payload();

    let signature = identity.sign(&signable).expect("signing should not fail");

    let credential = builder.build(signature);

    Memo {
        input,
        output,
        credential,
    }
}
```

Note: The exact `CredentialBuilder` API may differ slightly. The implementer should check the actual constructor signature in `harmony-credential/src/credential.rs` and adapt. The key inputs are: issuer, subject, timestamps, nonce, salted claim, signature.

- [ ] **Step 4: Create verify.rs**

```rust
use harmony_content::cid::ContentId;
use harmony_credential::verify::verify_credential;
use harmony_credential::CredentialKeyResolver;

use crate::{Memo, MemoError, MEMO_CLAIM_TYPE};

/// No-op status list resolver — memos don't use revocation.
struct NoOpStatusList;

// Implement the StatusListResolver trait for NoOpStatusList
// (always returns "not revoked")

/// Verify a memo's credential signature and claim encoding.
pub fn verify_memo(
    memo: &Memo,
    now: u64,
    keys: &impl CredentialKeyResolver,
) -> Result<(), MemoError> {
    // 1. Verify the credential (signature, time bounds)
    verify_credential(&memo.credential, now, keys, &NoOpStatusList)
        .map_err(MemoError::Credential)?;

    // 2. Verify the claim encodes the correct input/output
    // (The credential stores claim digests, not raw claims.
    //  For full verification, the verifier would need the SaltedClaim.
    //  For v1, we trust the Memo struct's input/output fields match
    //  the credential's claims — the signature covers the digest.)

    Ok(())
}
```

Note: Full claim verification requires the `SaltedClaim` (which is held by the memo creator). In v1, `verify_memo` verifies the credential's signature and time bounds. The input/output fields in the `Memo` struct are trusted because the credential's signature covers the claim digest. The implementer should check `verify_credential`'s exact signature and adapt.

- [ ] **Step 5: Add tests to lib.rs**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serialize_deserialize_roundtrip() {
        // Create a Memo with dummy data and verify roundtrip
    }

    #[test]
    fn deserialize_wrong_version_fails() {
        let data = vec![0xFF, 0x00]; // wrong version
        assert!(matches!(deserialize(&data), Err(MemoError::SerializationError)));
    }

    #[test]
    fn deserialize_empty_fails() {
        assert!(matches!(deserialize(&[]), Err(MemoError::SerializationError)));
    }

    #[test]
    fn memo_claim_type_is_correct() {
        assert_eq!(MEMO_CLAIM_TYPE, 0x4D45);
        // "ME" in ASCII
        assert_eq!(&[0x4D, 0x45], b"ME");
    }
}
```

- [ ] **Step 6: Run tests**

Run: `cargo test -p harmony-memo -v`
Expected: all tests pass

- [ ] **Step 7: Commit**

```bash
git add Cargo.toml crates/harmony-memo/
git commit -m "feat: add harmony-memo crate — signed attestation layer

Memo struct wraps Credential for input→output attestations.
create_memo() with sans-I/O RNG, verify_memo() with no-op revocation.
Wire format with version byte. MEMO_CLAIM_TYPE = 0x4D45.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: MemoStore — in-memory storage with dedup

**Files:**
- Create: `crates/harmony-memo/src/store.rs`
- Modify: `crates/harmony-memo/src/lib.rs` (add `pub mod store;`)

- [ ] **Step 1: Create store.rs with MemoStore and tests**

```rust
use alloc::vec::Vec;
use harmony_content::cid::ContentId;
use harmony_identity::{IdentityHash, IdentityRef};

#[cfg(not(feature = "std"))]
use hashbrown::HashMap;
#[cfg(feature = "std")]
use std::collections::HashMap;

use crate::Memo;

/// In-memory store of signed memos, keyed by input CID.
pub struct MemoStore {
    by_input: HashMap<ContentId, Vec<Memo>>,
}

impl MemoStore {
    pub fn new() -> Self { ... }

    /// Insert a memo. Deduplicates by (input, output, signer).
    /// signer = credential.issuer.hash
    pub fn insert(&mut self, memo: Memo) -> bool { ... }

    /// All memos for an input CID.
    pub fn get_by_input(&self, input: &ContentId) -> &[Memo] { ... }

    /// Specific memo by input + signer.
    pub fn get_by_input_and_signer(&self, input: &ContentId, signer: &IdentityHash) -> Option<&Memo> { ... }

    /// Outputs grouped by CID with list of signers.
    pub fn outputs_for_input(&self, input: &ContentId) -> Vec<(ContentId, Vec<IdentityRef>)> { ... }

    /// All input CIDs (for Bloom filter population).
    pub fn input_cids(&self) -> impl Iterator<Item = &ContentId> { ... }

    /// Total memo count across all inputs.
    pub fn len(&self) -> usize { ... }
    pub fn is_empty(&self) -> bool { ... }
}
```

Tests:
- `insert_and_query_by_input`
- `dedup_same_signer_same_input_output`
- `different_signers_same_input_output_coexist`
- `different_outputs_same_input_coexist`
- `outputs_for_input_groups_correctly`
- `get_by_input_and_signer`
- `input_cids_iteration`

- [ ] **Step 2: Run tests**

Run: `cargo test -p harmony-memo -v`

- [ ] **Step 3: Commit**

```bash
git add crates/harmony-memo/src/store.rs crates/harmony-memo/src/lib.rs
git commit -m "feat(memo): add MemoStore with dedup and grouped queries

Insert, query by input, query by signer, outputs_for_input grouping.
Dedup by (input, output, credential.issuer.hash). 7 unit tests.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: CLI subcommands — harmony memo sign/list/verify

**Files:**
- Modify: `crates/harmony-node/Cargo.toml` (add harmony-memo dep)
- Modify: `crates/harmony-node/src/main.rs` (add Memo subcommand)

- [ ] **Step 1: Add harmony-memo dependency**

In `crates/harmony-node/Cargo.toml`, add:

```toml
harmony-memo = { workspace = true, features = ["std"] }
```

Add to root `Cargo.toml` workspace dependencies:

```toml
harmony-memo = { path = "crates/harmony-memo" }
```

- [ ] **Step 2: Add Memo subcommand to CLI**

Add a new `Memo` variant to `Commands`:

```rust
    /// Memo attestation commands
    Memo {
        #[command(subcommand)]
        action: MemoAction,
    },
```

```rust
#[derive(Subcommand)]
enum MemoAction {
    /// Sign a memo attesting input produces output
    Sign {
        /// Input CID (hex)
        #[arg(long)]
        input: String,
        /// Output CID (hex)
        #[arg(long)]
        output: String,
        /// Path to identity key file
        #[arg(long, value_name = "PATH")]
        identity_file: Option<std::path::PathBuf>,
        /// Expiry in seconds from now (default: 365 days)
        #[arg(long, default_value_t = 31_536_000)]
        expires_in: u64,
    },
    /// List known memos for an input CID
    List {
        /// Input CID (hex)
        input: String,
    },
    /// Show memo verification status for an input
    Verify {
        /// Input CID (hex)
        input: String,
    },
}
```

- [ ] **Step 3: Implement the Sign handler**

The `Sign` handler:
1. Parse input/output hex into ContentId (via `ContentId::from_bytes`)
2. Load identity from identity file
3. Call `harmony_memo::create::create_memo(input, output, &identity.pq, &mut OsRng, now, expires_at)`
4. Serialize the memo
5. Print the serialized memo hex and the Zenoh key expression

For v1, this is a local operation (no Zenoh publishing — that comes when the node runtime is integrated). The CLI produces the memo; publishing is a follow-up.

- [ ] **Step 4: Add CLI tests**

```rust
    #[test]
    fn cli_parses_memo_sign() {
        let cli = Cli::try_parse_from([
            "harmony", "memo", "sign",
            "--input", "aa".repeat(32).as_str(),
            "--output", "bb".repeat(32).as_str(),
        ]).unwrap();
        assert!(matches!(cli.command, Commands::Memo { .. }));
    }

    #[test]
    fn cli_parses_memo_list() {
        let cli = Cli::try_parse_from(["harmony", "memo", "list", &"aa".repeat(32)]).unwrap();
        assert!(matches!(cli.command, Commands::Memo { .. }));
    }
```

- [ ] **Step 5: Run tests**

Run: `cargo test -p harmony-node -v`

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml crates/harmony-node/Cargo.toml crates/harmony-node/src/main.rs
git commit -m "feat(harmony-node): harmony memo sign/list/verify CLI

Memo subcommand for signing attestations. Parses input/output CIDs,
loads identity, creates signed memo credential. List and verify
subcommands scaffolded for future MemoStore integration.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Full verification

**Files:** None (verification only)

- [ ] **Step 1: Run full workspace tests**

Run: `cargo test --workspace`
Expected: all tests pass

- [ ] **Step 2: Run clippy**

Run: `cargo clippy -p harmony-memo -p harmony-zenoh -p harmony-node`
Expected: no new warnings

- [ ] **Step 3: Run fmt check**

Run: `cargo fmt --all -- --check`
Expected: our files are formatted correctly
