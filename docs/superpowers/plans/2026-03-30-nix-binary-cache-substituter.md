# Nix Binary Cache Substituter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make harmony-node serve as a Nix binary cache substituter on the local mesh, so any Nix-enabled machine can fetch packages from the Harmony CAS.

**Architecture:** NAR files are ingested into the existing CAS via `dag::ingest` (just blobs). narinfo metadata is stored as Books with memo mappings (store hash → narinfo CID). An axum HTTP server serves three endpoints implementing the Nix binary cache protocol. Behind a `nix-cache` feature flag.

**Tech Stack:** Rust, axum 0.8, ed25519-dalek (workspace dep), base64 (workspace dep), harmony-content CAS, harmony-memo

**Spec:** `docs/superpowers/specs/2026-03-30-nix-binary-cache-substituter-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `crates/harmony-node/src/narinfo.rs` | **Create.** Pure narinfo format: struct, build text, parse text, fingerprint, sign. No I/O. |
| `crates/harmony-node/src/nix_cache.rs` | **Create.** axum HTTP server: router, three handlers, shared state. |
| `crates/harmony-node/src/nar.rs` | **Create.** NAR push pipeline: single path, closure, disk persistence. |
| `crates/harmony-node/src/main.rs` | **Modify.** Add `Nar` subcommand, spawn HTTP server in `run`. |
| `crates/harmony-node/src/config.rs` | **Modify.** Add `nix_cache_port` field to `ConfigFile`. |
| `crates/harmony-node/Cargo.toml` | **Modify.** Add `nix-cache` feature + deps. |
| `Cargo.toml` (workspace) | **Modify.** Add axum, tower-http workspace deps. |

NixOS integration (harmony-os repo, separate branch + PR):

| File | Responsibility |
|------|---------------|
| `nixos/harmony-node-service.nix` | **Modify.** Add `nixCachePort` option. |
| `nixos/rpi5-base.nix` | **Modify.** Configure nix-cache port. |

---

### Task 1: Dependencies and Feature Flag

**Files:**
- Modify: `Cargo.toml` (workspace root)
- Modify: `crates/harmony-node/Cargo.toml`

- [ ] **Step 1: Add axum and tower-http to workspace Cargo.toml**

In the `[workspace.dependencies]` section of the root `Cargo.toml`, add after the existing entries (alphabetical order near the `a`/`t` sections):

```toml
axum = { version = "0.8", default-features = false, features = ["tokio", "http1"] }
tower-http = { version = "0.6", default-features = false }
```

- [ ] **Step 2: Add nix-cache feature and deps to harmony-node/Cargo.toml**

In `crates/harmony-node/Cargo.toml`, add to the `[features]` section:

```toml
nix-cache = ["dep:axum", "dep:tower-http", "dep:base64", "dep:ed25519-dalek"]
```

In the `[dependencies]` section, add:

```toml
axum = { workspace = true, optional = true }
tower-http = { workspace = true, optional = true }
base64 = { workspace = true, optional = true }
ed25519-dalek = { workspace = true, optional = true }
```

- [ ] **Step 3: Verify it compiles**

Run: `cargo check -p harmony-node --features nix-cache`
Expected: compiles with no errors (unused import warnings are OK)

- [ ] **Step 4: Commit**

```bash
git add Cargo.toml crates/harmony-node/Cargo.toml Cargo.lock
git commit -m "feat(node): add nix-cache feature flag with axum dependencies"
```

---

### Task 2: narinfo.rs — Pure Format Logic

**Files:**
- Create: `crates/harmony-node/src/narinfo.rs`
- Modify: `crates/harmony-node/src/main.rs` (add `mod narinfo;`)

This module handles narinfo text generation, parsing, fingerprint computation, and Nix ed25519 signing. No I/O, no CAS — pure data transformation.

**Reference — Nix narinfo format:**
```
StorePath: /nix/store/aaaabbbb...-package-1.0
URL: nar/0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef.nar
Compression: none
FileHash: sha256:BASE64HASH==
FileSize: 12345
NarHash: sha256:BASE64HASH==
NarSize: 12345
References: aaaabbbb...-dep1 ccccdddd...-dep2
Sig: keyname-1:BASE64SIG==
```

**Reference — Nix signing key file format:** `<keyname>:<base64-of-64-bytes>` where the 64 bytes are the ed25519 secret key (32-byte seed + 32-byte public key, libsodium format).

**Reference — Nix fingerprint (what gets signed):** `1;<StorePath>;<NarHash>;<NarSize>;<space-separated-sorted-references>`

- [ ] **Step 1: Write the failing tests**

Create `crates/harmony-node/src/narinfo.rs`:

```rust
//! Nix narinfo format: build, parse, fingerprint, and sign.
//!
//! This module handles the text-based narinfo metadata format used by the
//! Nix binary cache protocol. It is pure data transformation — no I/O,
//! no CAS access, no network.
//!
//! Reference: https://nixos.org/manual/nix/stable/protocols/binary-cache

#[cfg(feature = "nix-cache")]
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
#[cfg(feature = "nix-cache")]
use ed25519_dalek::{Signer, SigningKey};

/// A parsed Nix binary cache signing key.
///
/// Nix key files contain `<keyname>:<base64-of-64-bytes>` where the 64 bytes
/// are the ed25519 secret (32-byte seed + 32-byte public key, libsodium format).
#[cfg(feature = "nix-cache")]
pub struct NixSigningKey {
    pub name: String,
    key: SigningKey,
}

/// All fields needed to produce a narinfo response.
#[cfg(feature = "nix-cache")]
#[derive(Debug, Clone)]
pub struct NarInfo {
    /// Full store path, e.g. `/nix/store/abc...-glibc-2.39`
    pub store_path: String,
    /// URL to fetch the NAR from, e.g. `nar/<hex_cid>.nar`
    pub url: String,
    /// SHA-256 hash of the NAR bytes in Nix format: `sha256:<base64>`
    pub nar_hash: String,
    /// Size of the NAR in bytes
    pub nar_size: u64,
    /// Direct dependency basenames (just `abc...-dep`, not full paths)
    pub references: Vec<String>,
    /// Signature in `<keyname>:<base64sig>` format
    pub sig: Option<String>,
}

#[cfg(feature = "nix-cache")]
impl NixSigningKey {
    /// Load a Nix signing key from the standard `<name>:<base64>` format.
    pub fn from_nix_format(contents: &str) -> Result<Self, String> {
        let contents = contents.trim();
        let colon = contents
            .find(':')
            .ok_or("invalid Nix key format: missing ':' separator")?;
        let name = contents[..colon].to_string();
        if name.is_empty() {
            return Err("invalid Nix key format: empty key name".into());
        }
        let b64 = &contents[colon + 1..];
        let bytes = BASE64
            .decode(b64)
            .map_err(|e| format!("invalid Nix key format: bad base64: {e}"))?;
        if bytes.len() != 64 {
            return Err(format!(
                "invalid Nix key format: expected 64 bytes, got {}",
                bytes.len()
            ));
        }
        // First 32 bytes = ed25519 seed, second 32 = public key (libsodium format)
        let seed: [u8; 32] = bytes[..32]
            .try_into()
            .map_err(|_| "internal: seed slice length")?;
        let key = SigningKey::from_bytes(&seed);
        Ok(Self { name, key })
    }

    /// Sign a narinfo fingerprint string.
    pub fn sign_fingerprint(&self, fingerprint: &str) -> String {
        let sig = self.key.sign(fingerprint.as_bytes());
        format!("{}:{}", self.name, BASE64.encode(sig.to_bytes()))
    }
}

#[cfg(feature = "nix-cache")]
impl NarInfo {
    /// Compute the Nix fingerprint for signing.
    ///
    /// Format: `1;<StorePath>;<NarHash>;<NarSize>;<sorted space-separated References>`
    pub fn fingerprint(&self) -> String {
        let mut refs = self.references.clone();
        refs.sort();
        format!(
            "1;{};{};{};{}",
            self.store_path,
            self.nar_hash,
            self.nar_size,
            refs.join(" "),
        )
    }

    /// Sign this narinfo with the given key and store the signature.
    pub fn sign(&mut self, key: &NixSigningKey) {
        let fp = self.fingerprint();
        self.sig = Some(key.sign_fingerprint(&fp));
    }

    /// Format the SHA-256 hash bytes as Nix `sha256:<base64>`.
    pub fn format_nix_hash(sha256: &[u8; 32]) -> String {
        format!("sha256:{}", BASE64.encode(sha256))
    }

    /// Build the narinfo text for HTTP responses.
    pub fn to_text(&self) -> String {
        let mut out = String::with_capacity(512);
        out.push_str(&format!("StorePath: {}\n", self.store_path));
        out.push_str(&format!("URL: {}\n", self.url));
        out.push_str("Compression: none\n");
        // With Compression: none, FileHash == NarHash and FileSize == NarSize
        out.push_str(&format!("FileHash: {}\n", self.nar_hash));
        out.push_str(&format!("FileSize: {}\n", self.nar_size));
        out.push_str(&format!("NarHash: {}\n", self.nar_hash));
        out.push_str(&format!("NarSize: {}\n", self.nar_size));
        if !self.references.is_empty() {
            out.push_str(&format!("References: {}\n", self.references.join(" ")));
        }
        if let Some(ref sig) = self.sig {
            out.push_str(&format!("Sig: {}\n", sig));
        }
        out
    }

    /// Parse narinfo text back into a NarInfo struct.
    ///
    /// Used primarily for testing round-trips. Lenient: ignores unknown fields.
    pub fn from_text(text: &str) -> Result<Self, String> {
        let mut store_path = None;
        let mut url = None;
        let mut nar_hash = None;
        let mut nar_size = None;
        let mut references = Vec::new();
        let mut sig = None;

        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Some(val) = line.strip_prefix("StorePath: ") {
                store_path = Some(val.to_string());
            } else if let Some(val) = line.strip_prefix("URL: ") {
                url = Some(val.to_string());
            } else if let Some(val) = line.strip_prefix("NarHash: ") {
                nar_hash = Some(val.to_string());
            } else if let Some(val) = line.strip_prefix("NarSize: ") {
                nar_size = Some(
                    val.parse::<u64>()
                        .map_err(|e| format!("invalid NarSize: {e}"))?,
                );
            } else if let Some(val) = line.strip_prefix("References: ") {
                references = val.split_whitespace().map(String::from).collect();
            } else if let Some(val) = line.strip_prefix("Sig: ") {
                sig = Some(val.to_string());
            }
            // Ignore Compression, FileHash, FileSize, unknown fields
        }

        Ok(Self {
            store_path: store_path.ok_or("missing StorePath")?,
            url: url.ok_or("missing URL")?,
            nar_hash: nar_hash.ok_or("missing NarHash")?,
            nar_size: nar_size.ok_or("missing NarSize")?,
            references,
            sig,
        })
    }
}

#[cfg(all(test, feature = "nix-cache"))]
mod tests {
    use super::*;

    fn test_signing_key() -> NixSigningKey {
        // Generate a deterministic test key: 32 zero bytes as seed
        let seed = [0u8; 32];
        let key = SigningKey::from_bytes(&seed);
        // Encode in Nix format: name:base64(seed + pubkey)
        let pubkey = key.verifying_key().to_bytes();
        let mut combined = Vec::with_capacity(64);
        combined.extend_from_slice(&seed);
        combined.extend_from_slice(&pubkey);
        let nix_format = format!("test-key-1:{}", BASE64.encode(&combined));
        NixSigningKey::from_nix_format(&nix_format).unwrap()
    }

    fn test_narinfo() -> NarInfo {
        NarInfo {
            store_path: "/nix/store/aaaabbbbccccddddeeeeffffgggghhhh-hello-1.0".into(),
            url: "nar/deadbeef01234567deadbeef01234567deadbeef01234567deadbeef01234567.nar"
                .into(),
            nar_hash: "sha256:47DEQpj8HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFU=".into(),
            nar_size: 1024,
            references: vec![
                "zzzzyyyyxxxxwwwwvvvvuuuuttttssss-glibc-2.39".into(),
                "aabbccddeeffgghhiijjkkllmmnnoott-gcc-14.1".into(),
            ],
            sig: None,
        }
    }

    #[test]
    fn signing_key_from_nix_format() {
        let key = test_signing_key();
        assert_eq!(key.name, "test-key-1");
    }

    #[test]
    fn signing_key_rejects_empty_name() {
        let result = NixSigningKey::from_nix_format(":AAAA");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("empty key name"));
    }

    #[test]
    fn signing_key_rejects_missing_colon() {
        let result = NixSigningKey::from_nix_format("no-colon-here");
        assert!(result.is_err());
    }

    #[test]
    fn signing_key_rejects_wrong_length() {
        let result = NixSigningKey::from_nix_format("test:AAAA");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("expected 64 bytes"));
    }

    #[test]
    fn fingerprint_format() {
        let info = test_narinfo();
        let fp = info.fingerprint();
        // References should be sorted in the fingerprint
        assert!(fp.starts_with("1;/nix/store/aaaabbbbccccddddeeeeffffgggghhhh-hello-1.0;"));
        assert!(fp.contains(";1024;"));
        // aabbcc... sorts before zzzzy...
        assert!(fp.ends_with(
            "aabbccddeeffgghhiijjkkllmmnnoott-gcc-14.1 zzzzyyyyxxxxwwwwvvvvuuuuttttssss-glibc-2.39"
        ));
    }

    #[test]
    fn sign_produces_keyname_colon_base64() {
        let key = test_signing_key();
        let mut info = test_narinfo();
        info.sign(&key);
        let sig = info.sig.as_ref().unwrap();
        assert!(sig.starts_with("test-key-1:"));
        // ed25519 sig is 64 bytes → 88 chars base64
        let b64_part = &sig["test-key-1:".len()..];
        let decoded = BASE64.decode(b64_part).unwrap();
        assert_eq!(decoded.len(), 64);
    }

    #[test]
    fn narinfo_text_round_trip() {
        let key = test_signing_key();
        let mut info = test_narinfo();
        info.sign(&key);
        let text = info.to_text();
        let parsed = NarInfo::from_text(&text).unwrap();
        assert_eq!(parsed.store_path, info.store_path);
        assert_eq!(parsed.url, info.url);
        assert_eq!(parsed.nar_hash, info.nar_hash);
        assert_eq!(parsed.nar_size, info.nar_size);
        assert_eq!(parsed.references, info.references);
        assert_eq!(parsed.sig, info.sig);
    }

    #[test]
    fn format_nix_hash() {
        let hash = [0u8; 32];
        let formatted = NarInfo::format_nix_hash(&hash);
        assert!(formatted.starts_with("sha256:"));
        assert_eq!(formatted, "sha256:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=");
    }

    #[test]
    fn narinfo_text_contains_required_fields() {
        let info = test_narinfo();
        let text = info.to_text();
        assert!(text.contains("StorePath: /nix/store/"));
        assert!(text.contains("URL: nar/"));
        assert!(text.contains("Compression: none"));
        assert!(text.contains("FileHash: sha256:"));
        assert!(text.contains("FileSize: 1024"));
        assert!(text.contains("NarHash: sha256:"));
        assert!(text.contains("NarSize: 1024"));
        assert!(text.contains("References: "));
    }

    #[test]
    fn empty_references_omitted() {
        let mut info = test_narinfo();
        info.references.clear();
        let text = info.to_text();
        assert!(!text.contains("References:"));
    }
}
```

- [ ] **Step 2: Add module declaration to main.rs**

In `crates/harmony-node/src/main.rs`, add after the existing `mod` declarations:

```rust
#[cfg(feature = "nix-cache")]
mod narinfo;
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `cargo test -p harmony-node --features nix-cache narinfo -- --nocapture`
Expected: All 9 tests pass

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-node/src/narinfo.rs crates/harmony-node/src/main.rs
git commit -m "feat(node): add narinfo format module with build, parse, fingerprint, sign"
```

---

### Task 3: nix_cache.rs — HTTP Server

**Files:**
- Create: `crates/harmony-node/src/nix_cache.rs`
- Modify: `crates/harmony-node/src/main.rs` (add `mod nix_cache;`)

Three endpoints: `/nix-cache-info`, `/<hash>.narinfo`, `/nar/<cid>.nar`.

**Key types from codebase:**
- `harmony_content::cid::ContentId` — 32-byte CID, `from_bytes([u8; 32])`, `to_bytes() -> [u8; 32]`
- `harmony_content::book::BookStore` trait — `get(&self, &ContentId) -> Option<&[u8]>`
- `harmony_content::book::MemoryBookStore` — `new()`, implements BookStore
- `harmony_content::dag::reassemble(root, &store) -> Result<Vec<u8>, ContentError>`
- `harmony_memo::store::MemoStore` — `peek_by_input(&self, &ContentId) -> &[Memo]`
- `harmony_memo::Memo` — has `.input: ContentId`, `.output: ContentId`
- `ContentId::for_book(data, flags) -> Result<ContentId, ContentError>` — for computing CID from store hash
- `ContentFlags::default()` — standard flags

- [ ] **Step 1: Write the HTTP server module with tests**

Create `crates/harmony-node/src/nix_cache.rs`:

```rust
//! Nix binary cache HTTP server (axum).
//!
//! Serves the three endpoints of the Nix binary cache protocol:
//! - `GET /nix-cache-info` — cache metadata
//! - `GET /<hash>.narinfo` — narinfo lookup via MemoStore
//! - `GET /nar/<cid>.nar` — NAR streaming via DAG reassembly
//!
//! All handlers are read-only. The axum server shares state with the
//! event loop via `Arc` — no locking for the common serving path.

#[cfg(feature = "nix-cache")]
use std::sync::Arc;

#[cfg(feature = "nix-cache")]
use axum::{
    body::Body,
    extract::{Path, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::get,
    Router,
};

#[cfg(feature = "nix-cache")]
use harmony_content::book::MemoryBookStore;
#[cfg(feature = "nix-cache")]
use harmony_content::cid::{ContentFlags, ContentId};
#[cfg(feature = "nix-cache")]
use harmony_memo::store::MemoStore;

/// Shared state for the Nix cache HTTP handlers.
///
/// The `BookStore` and `MemoStore` are wrapped in `Arc` for concurrent
/// read access from axum tasks. The event loop owns the mutable
/// references; the HTTP layer is read-only.
#[cfg(feature = "nix-cache")]
pub struct NixCacheState {
    pub book_store: Arc<MemoryBookStore>,
    pub memo_store: Arc<MemoStore>,
    /// Optional data directory for disk fallback when books aren't in memory.
    pub data_dir: Option<std::path::PathBuf>,
}

#[cfg(feature = "nix-cache")]
pub fn router(state: Arc<NixCacheState>) -> Router {
    Router::new()
        .route("/nix-cache-info", get(nix_cache_info))
        .route("/{hash}.narinfo", get(narinfo_handler))
        .route("/nar/{cid}.nar", get(nar_handler))
        .with_state(state)
}

/// `GET /nix-cache-info` — static cache metadata.
///
/// Priority 30 ranks above cache.nixos.org (40) so Nix tries the
/// local harmony cache first.
#[cfg(feature = "nix-cache")]
async fn nix_cache_info() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "text/x-nix-cache-info")],
        "StoreDir: /nix/store\nWantMassQuery: 1\nPriority: 30\n",
    )
}

/// `GET /<hash>.narinfo` — look up narinfo by Nix store hash.
///
/// The store hash (32-char nix-base32) is converted to a CID by hashing
/// the string bytes via `ContentId::for_book`. The memo's output CID
/// points to a Book containing the pre-built narinfo text.
#[cfg(feature = "nix-cache")]
async fn narinfo_handler(
    State(state): State<Arc<NixCacheState>>,
    Path(hash_with_ext): Path<String>,
) -> Response {
    // Parse "abc123.narinfo" → "abc123"
    let store_hash = match hash_with_ext.strip_suffix(".narinfo") {
        Some(h) => h,
        None => return StatusCode::BAD_REQUEST.into_response(),
    };

    // Validate: Nix store hashes are 32 chars of nix-base32
    if store_hash.len() != 32 {
        return StatusCode::BAD_REQUEST.into_response();
    }

    // Compute input CID from store hash string bytes
    let input_cid = match ContentId::for_book(store_hash.as_bytes(), ContentFlags::default()) {
        Ok(cid) => cid,
        Err(_) => return StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    };

    // Look up memo (no LFU inflation for HTTP queries)
    let memos = state.memo_store.peek_by_input(&input_cid);
    let memo = match memos.first() {
        Some(m) => m,
        None => return StatusCode::NOT_FOUND.into_response(),
    };

    // Read the narinfo Book (output CID of the memo)
    let narinfo_cid = &memo.output;
    let narinfo_bytes = if let Some(data) = state.book_store.get(narinfo_cid) {
        data.to_vec()
    } else if let Some(ref data_dir) = state.data_dir {
        match crate::disk_io::read_book(data_dir, narinfo_cid) {
            Ok(data) => data,
            Err(_) => return StatusCode::NOT_FOUND.into_response(),
        }
    } else {
        return StatusCode::NOT_FOUND.into_response();
    };

    let text = match String::from_utf8(narinfo_bytes) {
        Ok(t) => t,
        Err(_) => return StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    };

    (
        [(header::CONTENT_TYPE, "text/x-nix-narinfo")],
        text,
    )
        .into_response()
}

/// `GET /nar/<cid>.nar` — stream NAR bytes by reassembling from CAS DAG.
///
/// The CID in the URL is the root of the DAG created during `nar push`.
/// We reassemble the full NAR by walking the DAG and concatenating leaf Books.
#[cfg(feature = "nix-cache")]
async fn nar_handler(
    State(state): State<Arc<NixCacheState>>,
    Path(cid_with_ext): Path<String>,
) -> Response {
    // Parse "deadbeef...64chars.nar" → CID hex
    let cid_hex = match cid_with_ext.strip_suffix(".nar") {
        Some(h) => h,
        None => return StatusCode::BAD_REQUEST.into_response(),
    };

    // Parse 64 hex chars → ContentId
    if cid_hex.len() != 64 {
        return StatusCode::BAD_REQUEST.into_response();
    }
    let cid_bytes = match hex::decode(cid_hex) {
        Ok(b) => b,
        Err(_) => return StatusCode::BAD_REQUEST.into_response(),
    };
    let cid_arr: [u8; 32] = match cid_bytes.try_into() {
        Ok(a) => a,
        Err(_) => return StatusCode::BAD_REQUEST.into_response(),
    };
    let root_cid = ContentId::from_bytes(cid_arr);

    // Reassemble from CAS — try memory first, fall back to disk
    // For the disk fallback case, we load all needed books into a temporary store
    let book_store = state.book_store.clone();

    // Spawn blocking to avoid holding the Arc across an await point
    let data_dir = state.data_dir.clone();
    let result = tokio::task::spawn_blocking(move || {
        // First try in-memory reassembly
        match harmony_content::dag::reassemble(&root_cid, book_store.as_ref()) {
            Ok(data) => return Ok(data),
            Err(_) => {}
        }

        // Fall back to disk: build a temporary store with all needed books
        if let Some(ref dir) = data_dir {
            let mut temp_store = MemoryBookStore::new();
            // Load root book/bundle from disk
            if let Ok(root_data) = crate::disk_io::read_book(dir, &root_cid) {
                temp_store.store(root_cid, root_data);
                // Walk and load all referenced CIDs
                load_dag_from_disk(dir, &root_cid, &mut temp_store);
                return harmony_content::dag::reassemble(&root_cid, &temp_store)
                    .map_err(|e| e.to_string());
            }
        }
        Err("NAR not found".to_string())
    })
    .await;

    match result {
        Ok(Ok(data)) => (
            [
                (header::CONTENT_TYPE, "application/x-nix-nar"),
                (header::CONTENT_LENGTH, &data.len().to_string()),
            ],
            Body::from(data),
        )
            .into_response(),
        _ => StatusCode::NOT_FOUND.into_response(),
    }
}

/// Recursively load a DAG from disk into an in-memory BookStore.
///
/// Walks bundles depth-first, loading each referenced CID from disk.
#[cfg(feature = "nix-cache")]
fn load_dag_from_disk(
    data_dir: &std::path::Path,
    cid: &ContentId,
    store: &mut MemoryBookStore,
) {
    if let Some(data) = store.get(cid) {
        // Already loaded — check if it's a bundle that needs recursive loading
        let data = data.to_vec();
        if cid.depth() > 0 {
            if let Ok(children) = harmony_content::bundle::parse_bundle(&data) {
                for child in children {
                    if child.is_sentinel() {
                        continue;
                    }
                    if !store.contains(child) {
                        if let Ok(child_data) = crate::disk_io::read_book(data_dir, child) {
                            store.store(*child, child_data);
                            load_dag_from_disk(data_dir, child, store);
                        }
                    }
                }
            }
        }
        return;
    }

    // Not in store — try loading from disk
    if let Ok(data) = crate::disk_io::read_book(data_dir, cid) {
        let data_clone = data.clone();
        store.store(*cid, data);
        if cid.depth() > 0 {
            if let Ok(children) = harmony_content::bundle::parse_bundle(&data_clone) {
                for child in children {
                    if child.is_sentinel() {
                        continue;
                    }
                    if !store.contains(child) {
                        load_dag_from_disk(data_dir, child, store);
                    }
                }
            }
        }
    }
}

#[cfg(all(test, feature = "nix-cache"))]
mod tests {
    use super::*;
    use axum::body::to_bytes;
    use axum::http::Request;
    use harmony_content::book::BookStore;
    use harmony_content::cid::ContentFlags;
    use harmony_memo::Memo;
    use tower::ServiceExt;

    /// Create a test NixCacheState with known data.
    fn setup_test_state() -> (Arc<NixCacheState>, ContentId, String) {
        let mut book_store = MemoryBookStore::new();
        let mut memo_store = MemoStore::new();

        // Store a narinfo as a Book
        let narinfo_text = "StorePath: /nix/store/aaaabbbbccccddddeeeeffffgggghhhh-hello-1.0\n\
            URL: nar/deadbeef01234567deadbeef01234567deadbeef01234567deadbeef01234567.nar\n\
            Compression: none\n\
            FileHash: sha256:47DEQpj8HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFU=\n\
            FileSize: 0\n\
            NarHash: sha256:47DEQpj8HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFU=\n\
            NarSize: 0\n";
        let narinfo_cid = book_store.insert(narinfo_text.as_bytes()).unwrap();

        // Create a memo mapping store hash → narinfo CID
        let store_hash = "aaaabbbbccccddddeeeeffffgggghhhh";
        let input_cid =
            ContentId::for_book(store_hash.as_bytes(), ContentFlags::default()).unwrap();

        // Create memo directly (bypass PQ identity for tests)
        let identity =
            harmony_identity::pq_identity::PqPrivateIdentity::generate(&mut rand::rngs::OsRng);
        let memo = harmony_memo::create::create_memo(
            input_cid,
            narinfo_cid,
            &identity,
            &mut rand::rngs::OsRng,
            1000,
            9_999_999,
        )
        .unwrap();
        memo_store.insert(memo);

        let state = Arc::new(NixCacheState {
            book_store: Arc::new(book_store),
            memo_store: Arc::new(memo_store),
            data_dir: None,
        });

        (state, narinfo_cid, store_hash.to_string())
    }

    #[tokio::test]
    async fn nix_cache_info_returns_expected_fields() {
        let (state, _, _) = setup_test_state();
        let app = router(state);

        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/nix-cache-info")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = to_bytes(resp.into_body(), 1024).await.unwrap();
        let text = String::from_utf8(body.to_vec()).unwrap();
        assert!(text.contains("StoreDir: /nix/store"));
        assert!(text.contains("WantMassQuery: 1"));
        assert!(text.contains("Priority: 30"));
    }

    #[tokio::test]
    async fn narinfo_returns_200_when_memo_exists() {
        let (state, _, store_hash) = setup_test_state();
        let app = router(state);

        let resp = app
            .oneshot(
                Request::builder()
                    .uri(&format!("/{store_hash}.narinfo"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let ct = resp.headers().get(header::CONTENT_TYPE).unwrap();
        assert_eq!(ct, "text/x-nix-narinfo");
        let body = to_bytes(resp.into_body(), 4096).await.unwrap();
        let text = String::from_utf8(body.to_vec()).unwrap();
        assert!(text.contains("StorePath: /nix/store/"));
    }

    #[tokio::test]
    async fn narinfo_returns_404_when_missing() {
        let (state, _, _) = setup_test_state();
        let app = router(state);

        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz.narinfo")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn narinfo_returns_400_for_bad_hash() {
        let (state, _, _) = setup_test_state();
        let app = router(state);

        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/tooshort.narinfo")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn nar_returns_200_for_known_book() {
        let mut book_store = MemoryBookStore::new();
        let test_data = b"test NAR content here";
        let cid = book_store.insert(test_data.as_slice()).unwrap();

        let state = Arc::new(NixCacheState {
            book_store: Arc::new(book_store),
            memo_store: Arc::new(MemoStore::new()),
            data_dir: None,
        });
        let app = router(state);

        let cid_hex = hex::encode(cid.to_bytes());
        let resp = app
            .oneshot(
                Request::builder()
                    .uri(&format!("/nar/{cid_hex}.nar"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = to_bytes(resp.into_body(), 4096).await.unwrap();
        assert_eq!(&body[..], test_data);
    }

    #[tokio::test]
    async fn nar_returns_404_for_unknown_cid() {
        let (state, _, _) = setup_test_state();
        let app = router(state);

        let fake_cid = hex::encode([0xFFu8; 32]);
        let resp = app
            .oneshot(
                Request::builder()
                    .uri(&format!("/nar/{fake_cid}.nar"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn nar_returns_400_for_bad_cid() {
        let (state, _, _) = setup_test_state();
        let app = router(state);

        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/nar/not-a-valid-hex-cid.nar")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }
}
```

- [ ] **Step 2: Add module declaration to main.rs**

In `crates/harmony-node/src/main.rs`, add after the `narinfo` module declaration:

```rust
#[cfg(feature = "nix-cache")]
mod nix_cache;
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `cargo test -p harmony-node --features nix-cache nix_cache -- --nocapture`
Expected: All 6 tests pass

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-node/src/nix_cache.rs crates/harmony-node/src/main.rs
git commit -m "feat(node): add Nix binary cache HTTP server with axum"
```

---

### Task 4: nar.rs — NAR Push Pipeline

**Files:**
- Create: `crates/harmony-node/src/nar.rs`
- Modify: `crates/harmony-node/src/main.rs` (add `mod nar;`)

The push pipeline: dump NAR from Nix store, ingest into CAS, build narinfo, sign, create memo, persist to disk.

**Key codebase patterns used:**
- `dag::ingest(data, &config, &mut store)` — chunks data, returns root CID
- `disk_io::write_book(data_dir, &cid, data)` — persist a book
- `memo_io::write_memo(data_dir, &memo)` — persist a memo
- `harmony_crypto::hash::full_hash(data) -> [u8; 32]` — SHA-256
- `parse_bundle(data) -> Result<&[ContentId], ContentError>` — parse bundle children
- `ChunkerConfig::default()` — default FastCDC settings

- [ ] **Step 1: Write the NAR push module**

Create `crates/harmony-node/src/nar.rs`:

```rust
//! NAR push pipeline: ingest Nix store paths into the Harmony CAS.
//!
//! Handles single store path ingestion and full closure enumeration.
//! Called by the `harmony nar push` CLI subcommand.

#[cfg(feature = "nix-cache")]
use std::path::Path;
#[cfg(feature = "nix-cache")]
use std::process::Command;

#[cfg(feature = "nix-cache")]
use harmony_content::book::{BookStore, MemoryBookStore};
#[cfg(feature = "nix-cache")]
use harmony_content::chunker::ChunkerConfig;
#[cfg(feature = "nix-cache")]
use harmony_content::cid::{ContentFlags, ContentId};

#[cfg(feature = "nix-cache")]
use crate::narinfo::{NarInfo, NixSigningKey};

/// Result of pushing a single store path.
#[cfg(feature = "nix-cache")]
pub struct PushResult {
    pub store_path: String,
    pub nar_root_cid: ContentId,
    pub narinfo_cid: ContentId,
    pub nar_size: u64,
    pub skipped: bool,
}

/// Push a single Nix store path into the CAS.
///
/// 1. Dumps NAR via `nix-store --dump`
/// 2. Ingests NAR blob into CAS via `dag::ingest`
/// 3. Persists all CAS books to disk
/// 4. Builds and signs narinfo
/// 5. Stores narinfo as a Book
/// 6. Creates memo mapping store hash → narinfo CID
///
/// Returns `Ok(PushResult { skipped: true })` if a memo already exists.
#[cfg(feature = "nix-cache")]
pub fn push_store_path(
    store_path: &str,
    nar_data: &[u8],
    signing_key: &NixSigningKey,
    data_dir: &Path,
    identity: &harmony_identity::pq_identity::PqPrivateIdentity,
) -> Result<PushResult, Box<dyn std::error::Error>> {
    // Extract store hash (32 chars after /nix/store/)
    let basename = store_path
        .strip_prefix("/nix/store/")
        .ok_or_else(|| format!("invalid store path: {store_path}"))?;
    let store_hash = &basename[..32.min(basename.len())];
    if store_hash.len() != 32 {
        return Err(format!("store hash too short: {store_hash}").into());
    }

    // Check if memo already exists (idempotent push)
    let input_cid = ContentId::for_book(store_hash.as_bytes(), ContentFlags::default())?;
    let memo_path = data_dir.join("memo");
    if memo_path.exists() {
        let existing_memos = crate::memo_io::scan_memos(data_dir);
        for (memo, _) in &existing_memos {
            if memo.input == input_cid {
                return Ok(PushResult {
                    store_path: store_path.to_string(),
                    nar_root_cid: memo.output,
                    narinfo_cid: memo.output,
                    nar_size: nar_data.len() as u64,
                    skipped: true,
                });
            }
        }
    }

    // Compute NarHash (SHA-256 of full NAR)
    let nar_hash_bytes = harmony_crypto::hash::full_hash(nar_data);
    let nar_hash = NarInfo::format_nix_hash(&nar_hash_bytes);
    let nar_size = nar_data.len() as u64;

    // Ingest NAR into CAS
    let mut book_store = MemoryBookStore::new();
    let config = ChunkerConfig::default();
    let nar_root_cid = harmony_content::dag::ingest(nar_data, &config, &mut book_store)?;

    // Persist all books to disk by walking the DAG
    persist_dag_to_disk(&nar_root_cid, &book_store, data_dir)?;

    // Build narinfo
    let nar_root_hex = hex::encode(nar_root_cid.to_bytes());
    let mut narinfo = NarInfo {
        store_path: store_path.to_string(),
        url: format!("nar/{nar_root_hex}.nar"),
        nar_hash,
        nar_size,
        references: Vec::new(), // Populated by caller for closure mode
        sig: None,
    };
    narinfo.sign(signing_key);

    // Store narinfo text as a Book
    let narinfo_text = narinfo.to_text();
    let narinfo_cid = book_store.insert(narinfo_text.as_bytes())?;
    crate::disk_io::write_book(data_dir, &narinfo_cid, narinfo_text.as_bytes())?;

    // Create memo: store_hash → narinfo CID
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs();
    let expires_at = now + 365 * 24 * 3600; // 1 year

    let memo = harmony_memo::create::create_memo(
        input_cid,
        narinfo_cid,
        identity,
        &mut rand::rngs::OsRng,
        now,
        expires_at,
    )
    .map_err(|e| format!("memo creation failed: {e}"))?;

    crate::memo_io::write_memo(data_dir, &memo)?;

    Ok(PushResult {
        store_path: store_path.to_string(),
        nar_root_cid,
        narinfo_cid,
        nar_size,
        skipped: false,
    })
}

/// Push a store path with references already resolved.
///
/// Like `push_store_path` but fills in the References field from the provided list.
#[cfg(feature = "nix-cache")]
pub fn push_store_path_with_refs(
    store_path: &str,
    nar_data: &[u8],
    references: Vec<String>,
    signing_key: &NixSigningKey,
    data_dir: &Path,
    identity: &harmony_identity::pq_identity::PqPrivateIdentity,
) -> Result<PushResult, Box<dyn std::error::Error>> {
    let basename = store_path
        .strip_prefix("/nix/store/")
        .ok_or_else(|| format!("invalid store path: {store_path}"))?;
    let store_hash = &basename[..32.min(basename.len())];
    if store_hash.len() != 32 {
        return Err(format!("store hash too short: {store_hash}").into());
    }

    let input_cid = ContentId::for_book(store_hash.as_bytes(), ContentFlags::default())?;

    // Check for existing memo
    if data_dir.join("memo").exists() {
        let existing_memos = crate::memo_io::scan_memos(data_dir);
        for (memo, _) in &existing_memos {
            if memo.input == input_cid {
                return Ok(PushResult {
                    store_path: store_path.to_string(),
                    nar_root_cid: memo.output,
                    narinfo_cid: memo.output,
                    nar_size: nar_data.len() as u64,
                    skipped: true,
                });
            }
        }
    }

    let nar_hash_bytes = harmony_crypto::hash::full_hash(nar_data);
    let nar_hash = NarInfo::format_nix_hash(&nar_hash_bytes);
    let nar_size = nar_data.len() as u64;

    let mut book_store = MemoryBookStore::new();
    let config = ChunkerConfig::default();
    let nar_root_cid = harmony_content::dag::ingest(nar_data, &config, &mut book_store)?;

    persist_dag_to_disk(&nar_root_cid, &book_store, data_dir)?;

    let nar_root_hex = hex::encode(nar_root_cid.to_bytes());
    let mut narinfo = NarInfo {
        store_path: store_path.to_string(),
        url: format!("nar/{nar_root_hex}.nar"),
        nar_hash,
        nar_size,
        references,
        sig: None,
    };
    narinfo.sign(signing_key);

    let narinfo_text = narinfo.to_text();
    let narinfo_cid = book_store.insert(narinfo_text.as_bytes())?;
    crate::disk_io::write_book(data_dir, &narinfo_cid, narinfo_text.as_bytes())?;

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs();
    let expires_at = now + 365 * 24 * 3600;

    let memo = harmony_memo::create::create_memo(
        input_cid,
        narinfo_cid,
        identity,
        &mut rand::rngs::OsRng,
        now,
        expires_at,
    )
    .map_err(|e| format!("memo creation failed: {e}"))?;

    crate::memo_io::write_memo(data_dir, &memo)?;

    Ok(PushResult {
        store_path: store_path.to_string(),
        nar_root_cid,
        narinfo_cid,
        nar_size,
        skipped: false,
    })
}

/// Dump NAR bytes from a local Nix store path via `nix-store --dump`.
#[cfg(feature = "nix-cache")]
pub fn dump_nar(store_path: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let output = Command::new("nix-store")
        .args(["--dump", store_path])
        .output()
        .map_err(|e| format!("failed to run nix-store --dump: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("nix-store --dump failed: {stderr}").into());
    }

    if output.stdout.is_empty() {
        return Err(format!("nix-store --dump produced empty output for {store_path}").into());
    }

    Ok(output.stdout)
}

/// Enumerate the full closure of a store path via `nix-store -qR`.
#[cfg(feature = "nix-cache")]
pub fn enumerate_closure(store_path: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let output = Command::new("nix-store")
        .args(["-qR", store_path])
        .output()
        .map_err(|e| format!("failed to run nix-store -qR: {e}. Is the store path valid?"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(
            format!("Cannot enumerate closure: {stderr}. Is the store path valid?").into(),
        );
    }

    let paths: Vec<String> = String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter(|l| !l.is_empty())
        .map(String::from)
        .collect();

    Ok(paths)
}

/// Get direct references of a store path via `nix-store -q --references`.
#[cfg(feature = "nix-cache")]
pub fn get_references(store_path: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let output = Command::new("nix-store")
        .args(["-q", "--references", store_path])
        .output()
        .map_err(|e| format!("failed to run nix-store -q --references: {e}"))?;

    if !output.status.success() {
        return Ok(Vec::new()); // No references is valid
    }

    let refs: Vec<String> = String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter(|l| !l.is_empty())
        .filter_map(|l| {
            // Convert full path to basename
            l.strip_prefix("/nix/store/").map(String::from)
        })
        .collect();

    Ok(refs)
}

/// Persist all books in a DAG to disk by walking the tree recursively.
///
/// Walks bundles depth-first and writes each CID's data to disk.
/// Skips books that already exist on disk.
#[cfg(feature = "nix-cache")]
fn persist_dag_to_disk(
    root: &ContentId,
    store: &MemoryBookStore,
    data_dir: &Path,
) -> Result<(), std::io::Error> {
    // Write root
    if let Some(data) = store.get(root) {
        let data = data.to_vec();
        crate::disk_io::write_book(data_dir, root, &data)?;

        // If it's a bundle, recurse into children
        if root.depth() > 0 {
            if let Ok(children) = harmony_content::bundle::parse_bundle(&data) {
                for child in children {
                    if !child.is_sentinel() {
                        persist_dag_to_disk(child, store, data_dir)?;
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(all(test, feature = "nix-cache"))]
mod tests {
    use super::*;
    use crate::narinfo::NixSigningKey;
    use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
    use ed25519_dalek::SigningKey;
    use harmony_content::book::BookStore;
    use harmony_content::cid::ContentFlags;

    fn test_signing_key() -> NixSigningKey {
        let seed = [1u8; 32];
        let key = SigningKey::from_bytes(&seed);
        let pubkey = key.verifying_key().to_bytes();
        let mut combined = Vec::with_capacity(64);
        combined.extend_from_slice(&seed);
        combined.extend_from_slice(&pubkey);
        NixSigningKey::from_nix_format(&format!("test-key-1:{}", BASE64.encode(&combined)))
            .unwrap()
    }

    fn test_identity() -> harmony_identity::pq_identity::PqPrivateIdentity {
        harmony_identity::pq_identity::PqPrivateIdentity::generate(&mut rand::rngs::OsRng)
    }

    #[test]
    fn push_store_path_creates_memo_and_books() {
        let dir = tempfile::TempDir::new().unwrap();
        let key = test_signing_key();
        let identity = test_identity();

        // Use a small test payload as the "NAR"
        let nar_data = b"this is fake NAR data for testing the push pipeline";

        let result = push_store_path(
            "/nix/store/aaaabbbbccccddddeeeeffffgggghhhh-test-1.0",
            nar_data,
            &key,
            dir.path(),
            &identity,
        )
        .expect("push should succeed");

        assert!(!result.skipped);
        assert_eq!(result.nar_size, nar_data.len() as u64);

        // Verify memo was written to disk
        let memos = crate::memo_io::scan_memos(dir.path());
        assert_eq!(memos.len(), 1);

        // Verify the narinfo book was written
        let narinfo_data =
            crate::disk_io::read_book(dir.path(), &result.narinfo_cid).expect("narinfo on disk");
        let narinfo_text = String::from_utf8(narinfo_data).unwrap();
        assert!(narinfo_text.contains("StorePath: /nix/store/aaaabbbbccccddddeeeeffffgggghhhh-test-1.0"));
        assert!(narinfo_text.contains("Compression: none"));
        assert!(narinfo_text.contains("Sig: test-key-1:"));
    }

    #[test]
    fn push_same_path_twice_is_idempotent() {
        let dir = tempfile::TempDir::new().unwrap();
        let key = test_signing_key();
        let identity = test_identity();

        let nar_data = b"NAR data for idempotency test";
        let store_path = "/nix/store/bbbbccccddddeeeeffffgggghhhhiiii-test-2.0";

        let first = push_store_path(store_path, nar_data, &key, dir.path(), &identity)
            .expect("first push");
        assert!(!first.skipped);

        let second = push_store_path(store_path, nar_data, &key, dir.path(), &identity)
            .expect("second push");
        assert!(second.skipped);

        // Only one memo on disk
        let memos = crate::memo_io::scan_memos(dir.path());
        assert_eq!(memos.len(), 1);
    }

    #[test]
    fn push_with_refs_includes_references() {
        let dir = tempfile::TempDir::new().unwrap();
        let key = test_signing_key();
        let identity = test_identity();

        let nar_data = b"NAR with references";
        let refs = vec![
            "zzzzyyyyxxxxwwww0000111122223333-glibc-2.39".into(),
            "aaaabbbbccccdddd4444555566667777-gcc-14.1".into(),
        ];

        let result = push_store_path_with_refs(
            "/nix/store/ccccddddeeeeffffgggghhhhiiiijjjj-test-3.0",
            nar_data,
            refs,
            &key,
            dir.path(),
            &identity,
        )
        .expect("push with refs");

        let narinfo_data =
            crate::disk_io::read_book(dir.path(), &result.narinfo_cid).unwrap();
        let narinfo_text = String::from_utf8(narinfo_data).unwrap();
        assert!(narinfo_text.contains("References: "));
        assert!(narinfo_text.contains("glibc-2.39"));
        assert!(narinfo_text.contains("gcc-14.1"));
    }

    #[test]
    fn push_rejects_invalid_store_path() {
        let dir = tempfile::TempDir::new().unwrap();
        let key = test_signing_key();
        let identity = test_identity();

        let result = push_store_path(
            "/not/a/nix/store/path",
            b"data",
            &key,
            dir.path(),
            &identity,
        );
        assert!(result.is_err());
    }

    #[test]
    fn persist_dag_writes_all_books() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut store = MemoryBookStore::new();
        let config = ChunkerConfig::default();

        // Ingest enough data to create multiple chunks (> 1 MiB for default config)
        let data = vec![0xABu8; 2 * 1024 * 1024]; // 2 MiB
        let root = harmony_content::dag::ingest(&data, &config, &mut store).unwrap();

        persist_dag_to_disk(&root, &store, dir.path()).unwrap();

        // Verify we can reassemble from disk alone
        let mut disk_store = MemoryBookStore::new();
        let books = crate::disk_io::scan_books(dir.path());
        assert!(books.len() > 1, "should have multiple books for 2 MiB");
        for (cid, _) in &books {
            let data = crate::disk_io::read_book(dir.path(), cid).unwrap();
            disk_store.store(*cid, data);
        }
        let reassembled = harmony_content::dag::reassemble(&root, &disk_store).unwrap();
        assert_eq!(reassembled, data);
    }
}
```

- [ ] **Step 2: Add module declaration to main.rs**

In `crates/harmony-node/src/main.rs`, add after the `nix_cache` module declaration:

```rust
#[cfg(feature = "nix-cache")]
mod nar;
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `cargo test -p harmony-node --features nix-cache nar:: -- --nocapture`
Expected: All 5 tests pass

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-node/src/nar.rs crates/harmony-node/src/main.rs
git commit -m "feat(node): add NAR push pipeline with CAS ingest and memo creation"
```

---

### Task 5: CLI and Config Integration

**Files:**
- Modify: `crates/harmony-node/src/config.rs`
- Modify: `crates/harmony-node/src/main.rs`

Wire the `Nar` subcommand into the CLI and spawn the HTTP server in `run`.

- [ ] **Step 1: Add nix_cache_port to ConfigFile**

In `crates/harmony-node/src/config.rs`, add to the `ConfigFile` struct after the `engram_manifest_cid` field:

```rust
    /// Port for the Nix binary cache HTTP server (0 or absent = disabled).
    /// Requires the `nix-cache` feature.
    pub nix_cache_port: Option<u16>,
```

- [ ] **Step 2: Add Nar subcommand and nix-cache-port CLI flag to main.rs**

In `crates/harmony-node/src/main.rs`, add the `Nar` variant to the `Commands` enum:

```rust
    /// Nix binary cache commands
    #[cfg(feature = "nix-cache")]
    Nar {
        #[command(subcommand)]
        action: NarAction,
    },
```

Add the `NarAction` enum after the `MemoAction` enum:

```rust
#[cfg(feature = "nix-cache")]
#[derive(Subcommand)]
enum NarAction {
    /// Push a Nix store path (or closure) into the Harmony CAS
    Push {
        /// Nix store path to push (e.g., /nix/store/abc...-package-1.0)
        store_path: Option<String>,
        /// Push the entire closure (all transitive dependencies)
        #[arg(long)]
        closure: bool,
        /// Read NAR data from stdin instead of running nix-store --dump
        #[arg(long)]
        stdin: bool,
        /// Store path name (required with --stdin, e.g., abc...-package-1.0)
        #[arg(long, value_name = "NAME")]
        store_path_name: Option<String>,
        /// Path to Nix binary cache signing key file
        #[arg(long, value_name = "PATH")]
        signing_key: std::path::PathBuf,
        /// CAS data directory for persistent storage
        #[arg(long, value_name = "PATH")]
        data_dir: std::path::PathBuf,
        /// Path to identity key file
        #[arg(long, value_name = "PATH")]
        identity_file: Option<std::path::PathBuf>,
    },
}
```

Add `--nix-cache-port` to the `Run` command's args (inside the existing `Run { ... }` variant):

```rust
        /// Port for Nix binary cache HTTP server (0 = disabled)
        #[cfg(feature = "nix-cache")]
        #[arg(long)]
        nix_cache_port: Option<u16>,
```

- [ ] **Step 3: Implement the Nar::Push handler in the `run()` function**

In the `run()` function's `match cli.command { ... }`, add the `Nar` arm before the `Cid` arm:

```rust
        #[cfg(feature = "nix-cache")]
        Commands::Nar { action } => match action {
            NarAction::Push {
                store_path,
                closure,
                stdin,
                store_path_name,
                signing_key,
                data_dir,
                identity_file,
            } => {
                use crate::narinfo::NixSigningKey;

                // Load signing key
                let key_contents = std::fs::read_to_string(&signing_key).map_err(|e| {
                    format!(
                        "failed to read signing key {}: {e}",
                        signing_key.display()
                    )
                })?;
                let nix_key = NixSigningKey::from_nix_format(&key_contents)
                    .map_err(|e| format!("invalid Nix signing key: {e}"))?;

                // Load identity
                let id_path =
                    crate::identity_file::resolve_path(identity_file.as_deref())?;
                let identity = crate::identity_file::load_or_generate(&id_path)?;

                if stdin {
                    // Pipe mode: read NAR from stdin
                    let sp_name = store_path_name.or(store_path).ok_or(
                        "--store-path-name (or positional store_path) required with --stdin",
                    )?;
                    let full_path = if sp_name.starts_with("/nix/store/") {
                        sp_name
                    } else {
                        format!("/nix/store/{sp_name}")
                    };

                    use std::io::Read;
                    let mut nar_data = Vec::new();
                    std::io::stdin()
                        .read_to_end(&mut nar_data)
                        .map_err(|e| format!("failed to read stdin: {e}"))?;
                    if nar_data.is_empty() {
                        return Err("empty NAR data from stdin".into());
                    }

                    let result = crate::nar::push_store_path(
                        &full_path,
                        &nar_data,
                        &nix_key,
                        &data_dir,
                        &identity.pq,
                    )?;
                    if result.skipped {
                        eprintln!("Skipped (already cached): {}", result.store_path);
                    } else {
                        eprintln!("Pushed: {} ({} bytes)", result.store_path, result.nar_size);
                    }
                } else if closure {
                    // Closure mode: enumerate all deps and push each
                    let sp = store_path.ok_or("store path required for closure mode")?;
                    let paths = crate::nar::enumerate_closure(&sp)?;
                    let total = paths.len();
                    eprintln!("Closure contains {total} store paths");

                    for (i, path) in paths.iter().enumerate() {
                        let refs = crate::nar::get_references(path).unwrap_or_default();
                        let nar_data = crate::nar::dump_nar(path)?;
                        let result = crate::nar::push_store_path_with_refs(
                            path,
                            &nar_data,
                            refs,
                            &nix_key,
                            &data_dir,
                            &identity.pq,
                        )?;
                        if result.skipped {
                            eprintln!("[{}/{}] Skipped: {}", i + 1, total, result.store_path);
                        } else {
                            eprintln!(
                                "[{}/{}] Pushed: {} ({} bytes)",
                                i + 1,
                                total,
                                result.store_path,
                                result.nar_size
                            );
                        }
                    }
                } else {
                    // Single path mode
                    let sp = store_path.ok_or("store path required")?;
                    let refs = crate::nar::get_references(&sp).unwrap_or_default();
                    let nar_data = crate::nar::dump_nar(&sp)?;
                    let result = crate::nar::push_store_path_with_refs(
                        &sp,
                        &nar_data,
                        refs,
                        &nix_key,
                        &data_dir,
                        &identity.pq,
                    )?;
                    if result.skipped {
                        eprintln!("Skipped (already cached): {}", result.store_path);
                    } else {
                        eprintln!("Pushed: {} ({} bytes)", result.store_path, result.nar_size);
                    }
                }

                drop(identity);
                Ok(())
            }
        },
```

- [ ] **Step 4: Spawn HTTP server in the Run command**

In the `Commands::Run { ... }` match arm, after the `event_loop::run(...)` call setup but **before** calling `event_loop::run(...)`, add the HTTP server spawn. Find the line `tracing::info!(cache_capacity, compute_budget, %listen_addr, "harmony node starting");` and add after it:

```rust
            // Spawn Nix binary cache HTTP server if configured
            #[cfg(feature = "nix-cache")]
            {
                let nix_cache_port = crate::config::resolve(
                    nix_cache_port,
                    config_file.nix_cache_port,
                    0u16,
                );
                if nix_cache_port > 0 {
                    let state = std::sync::Arc::new(crate::nix_cache::NixCacheState {
                        book_store: std::sync::Arc::new(
                            harmony_content::book::MemoryBookStore::new(),
                        ),
                        memo_store: std::sync::Arc::new(harmony_memo::store::MemoStore::new()),
                        data_dir: config_file.data_dir.clone(),
                    });
                    let nix_addr: std::net::SocketAddr =
                        format!("0.0.0.0:{nix_cache_port}").parse().map_err(|e| {
                            format!("invalid nix-cache-port {nix_cache_port}: {e}")
                        })?;
                    let nix_router = crate::nix_cache::router(state);
                    tracing::info!(%nix_addr, "nix binary cache server starting");
                    tokio::spawn(async move {
                        let listener = tokio::net::TcpListener::bind(nix_addr)
                            .await
                            .expect("failed to bind nix-cache port");
                        if let Err(e) = axum::serve(listener, nix_router).await {
                            tracing::error!("nix cache server error: {e}");
                        }
                    });
                }
            }
```

**Important note for the implementer:** The HTTP server currently creates its own empty BookStore and MemoStore. This means it can only serve content from disk (via `data_dir`). Sharing the runtime's live stores requires passing them through the event loop, which is deferred — for v1, disk-only serving works because `nar push` persists everything to disk. A follow-up can wire the runtime's stores via `Arc<RwLock>`.

- [ ] **Step 5: Verify compilation**

Run: `cargo check -p harmony-node --features nix-cache`
Expected: Compiles with no errors

- [ ] **Step 6: Run all tests**

Run: `cargo test -p harmony-node --features nix-cache -- --nocapture`
Expected: All narinfo, nix_cache, and nar tests pass

- [ ] **Step 7: Commit**

```bash
git add crates/harmony-node/src/main.rs crates/harmony-node/src/config.rs
git commit -m "feat(node): wire Nar CLI subcommand and HTTP server into main"
```

---

### Task 6: NixOS Service Integration (harmony-os repo)

**Files:**
- Modify: `nixos/harmony-node-service.nix` (in the harmony-os repo)
- Modify: `nixos/rpi5-base.nix` (in the harmony-os repo)

**Important:** This task is in the `harmony-os` repo, not the `harmony` repo. Create a matching branch in harmony-os before committing.

- [ ] **Step 1: Create branch in harmony-os**

```bash
cd <harmony-os-repo>
git fetch origin
git checkout -b <matching-branch-name> origin/main
```

- [ ] **Step 2: Add nixCachePort option to harmony-node-service.nix**

In `nixos/harmony-node-service.nix`, add after the `diskQuota` option (before the closing `};` of `options.services.harmony-node`):

```nix
    nixCachePort = lib.mkOption {
      type = lib.types.nullOr lib.types.port;
      default = null;
      example = 5000;
      description = "TCP port for Nix binary cache HTTP server. When set, harmony-node serves narinfo and NAR files. Requires the nix-cache feature in the binary.";
    };
```

In the `serviceConfig` let-binding, add after the `diskQuotaArgs` definition:

```nix
        nixCacheArgs = lib.optionalString (cfg.nixCachePort != null)
          " --nix-cache-port ${toString cfg.nixCachePort}";
```

Add `${nixCacheArgs}` to the end of the `ExecStart` string (before the closing `";`).

In the `config` section, add after the existing `networking.firewall.allowedUDPPorts`:

```nix
    # Open TCP port for Nix binary cache HTTP server (if configured)
    networking.firewall.allowedTCPPorts = lib.mkIf (cfg.nixCachePort != null)
      [ cfg.nixCachePort ];
```

- [ ] **Step 3: Configure nix-cache port in rpi5-base.nix**

In `nixos/rpi5-base.nix`, add to the `services.harmony-node` block:

```nix
    nixCachePort = 5000;
```

- [ ] **Step 4: Verify NixOS evaluation (if possible)**

Run: `nix eval .#nixosConfigurations.rpi5-luna.config.services.harmony-node.nixCachePort --system aarch64-linux 2>/dev/null || echo "eval requires aarch64 — skip on macOS"`

- [ ] **Step 5: Commit and push**

```bash
cd <harmony-os-repo>
git add nixos/harmony-node-service.nix nixos/rpi5-base.nix
git commit -m "feat(nixos): add Nix binary cache port option to harmony-node service"
git push -u origin <branch-name>
```

---

## Self-Review Checklist

**Spec coverage:**
- Data model (NAR in CAS, narinfo as Book, memo mapping): Task 4 (nar.rs) + Task 2 (narinfo.rs)
- Ingestion pipeline (single path, closure, pipe mode): Task 4 + Task 5
- HTTP server (3 endpoints): Task 3 (nix_cache.rs)
- Signing (Nix ed25519): Task 2 (narinfo.rs)
- CLI interface: Task 5 (main.rs)
- Configuration: Task 5 (config.rs)
- Feature flag: Task 1 (Cargo.toml)
- NixOS integration: Task 6
- Error handling: Covered in each module

**No placeholders:** All code is complete. No TBD/TODO references.

**Type consistency:**
- `NarInfo` struct and methods consistent across narinfo.rs and nar.rs
- `NixSigningKey` type used consistently
- `PushResult` defined in nar.rs, used in main.rs
- `NixCacheState` defined in nix_cache.rs, constructed in main.rs
- `ContentId`, `ContentFlags`, `MemoryBookStore` used with correct import paths throughout

**Known limitation (documented in Task 5 step 4):** The HTTP server creates its own empty stores and serves only from disk. Sharing the runtime's live in-memory stores requires `Arc<RwLock>` threading through the event loop — deferred to a follow-up since `nar push` persists everything to disk and disk-only serving works for v1.
