# S3 Great Library Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Archive durable Harmony books to S3 via an archivist service that subscribes to Zenoh content publish events.

**Architecture:** New `harmony-s3` crate with async S3 client (`S3Library`), plus an archivist task in harmony-node that filters for durable content and uploads it. Stateless — dedup via S3 HEAD before PUT.

**Tech Stack:** Rust, aws-sdk-s3 v1, aws-config v1, tokio, harmony-content (ContentId, ContentFlags)

**Spec:** `docs/superpowers/specs/2026-03-24-s3-great-library-design.md`

---

### Task 1: Scaffold harmony-s3 crate with S3Library

Create the new crate with the S3 client wrapper. PUT/GET/HEAD operations on books keyed by hex-encoded ContentId.

**Files:**
- Modify: `Cargo.toml` (workspace root) — add member + AWS deps
- Create: `crates/harmony-s3/Cargo.toml`
- Create: `crates/harmony-s3/src/lib.rs`
- Create: `crates/harmony-s3/src/error.rs`

- [ ] **Step 1: Add workspace member and dependencies**

In root `Cargo.toml`, add `"crates/harmony-s3"` to the workspace members list (after `"crates/harmony-rawlink"`).

Add to `[workspace.dependencies]`:
```toml
aws-config = "1"
aws-sdk-s3 = "1"
```

- [ ] **Step 2: Create crate Cargo.toml**

```toml
[package]
name = "harmony-s3"
version.workspace = true
edition.workspace = true
license.workspace = true
repository.workspace = true
rust-version.workspace = true

[dependencies]
aws-config = { workspace = true }
aws-sdk-s3 = { workspace = true }
hex.workspace = true
tokio = { workspace = true }
tracing.workspace = true

[dev-dependencies]
tokio = { workspace = true, features = ["rt", "macros"] }
```

Note: `harmony-s3` does NOT depend on `harmony-content` — it takes raw `&[u8; 32]` CID bytes and `&[u8]` data. This keeps the crate minimal and avoids pulling in the content layer.

- [ ] **Step 3: Create error.rs**

```rust
//! Error types for S3 operations.

use std::fmt;

#[derive(Debug)]
pub enum S3Error {
    /// S3 PUT failed.
    PutFailed(String),
    /// S3 GET failed.
    GetFailed(String),
    /// S3 HEAD failed.
    HeadFailed(String),
    /// AWS SDK configuration error.
    ConfigError(String),
}

impl fmt::Display for S3Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PutFailed(msg) => write!(f, "S3 PUT failed: {msg}"),
            Self::GetFailed(msg) => write!(f, "S3 GET failed: {msg}"),
            Self::HeadFailed(msg) => write!(f, "S3 HEAD failed: {msg}"),
            Self::ConfigError(msg) => write!(f, "AWS config error: {msg}"),
        }
    }
}

impl std::error::Error for S3Error {}
```

- [ ] **Step 4: Create lib.rs with S3Library**

```rust
//! S3 backing store for Harmony's content-addressed storage.
//!
//! Books (immutable, ≤1MB, BLAKE3-addressed) are stored as S3 objects
//! keyed by hex-encoded ContentId. Uses Intelligent-Tiering for
//! automatic cost optimization.

pub mod error;

use error::S3Error;

/// S3-backed book store.
///
/// Each book is stored at `{prefix}book/{cid_hex}` where `cid_hex` is
/// the 64-character hex encoding of the 32-byte ContentId.
pub struct S3Library {
    client: aws_sdk_s3::Client,
    bucket: String,
    prefix: String,
}

impl S3Library {
    /// Create a new S3Library.
    ///
    /// AWS credentials are resolved via the standard SDK chain:
    /// environment variables, ~/.aws/credentials, IAM role, etc.
    pub async fn new(bucket: String, prefix: String, region: Option<String>) -> Result<Self, S3Error> {
        let mut config_loader = aws_config::from_env();
        if let Some(ref region) = region {
            config_loader = config_loader.region(aws_config::Region::new(region.clone()));
        }
        let sdk_config = config_loader.load().await;
        let client = aws_sdk_s3::Client::new(&sdk_config);
        Ok(Self { client, bucket, prefix })
    }

    /// S3 object key for a given ContentId.
    pub fn object_key(&self, cid_bytes: &[u8; 32]) -> String {
        format!("{}book/{}", self.prefix, hex::encode(cid_bytes))
    }

    /// Upload a book to S3 with Intelligent-Tiering.
    ///
    /// Idempotent — writing the same CID twice produces the same object.
    pub async fn put_book(&self, cid_bytes: &[u8; 32], data: &[u8]) -> Result<(), S3Error> {
        let key = self.object_key(cid_bytes);
        self.client
            .put_object()
            .bucket(&self.bucket)
            .key(&key)
            .body(aws_sdk_s3::primitives::ByteStream::from(data.to_vec()))
            .storage_class(aws_sdk_s3::types::StorageClass::IntelligentTiering)
            .send()
            .await
            .map_err(|e| S3Error::PutFailed(format!("{e}")))?;
        Ok(())
    }

    /// Fetch a book from S3.
    ///
    /// Returns `None` if the object doesn't exist (NoSuchKey).
    pub async fn get_book(&self, cid_bytes: &[u8; 32]) -> Result<Option<Vec<u8>>, S3Error> {
        let key = self.object_key(cid_bytes);
        match self.client
            .get_object()
            .bucket(&self.bucket)
            .key(&key)
            .send()
            .await
        {
            Ok(output) => {
                let bytes = output
                    .body
                    .collect()
                    .await
                    .map_err(|e| S3Error::GetFailed(format!("body read: {e}")))?
                    .into_bytes()
                    .to_vec();
                Ok(Some(bytes))
            }
            Err(sdk_err) => {
                // Check for NoSuchKey
                let service_err = sdk_err.into_service_error();
                if service_err.is_no_such_key() {
                    Ok(None)
                } else {
                    Err(S3Error::GetFailed(format!("{service_err}")))
                }
            }
        }
    }

    /// Check if a book exists in S3 (HEAD request).
    ///
    /// Used for dedup before upload.
    pub async fn exists(&self, cid_bytes: &[u8; 32]) -> Result<bool, S3Error> {
        let key = self.object_key(cid_bytes);
        match self.client
            .head_object()
            .bucket(&self.bucket)
            .key(&key)
            .send()
            .await
        {
            Ok(_) => Ok(true),
            Err(sdk_err) => {
                let service_err = sdk_err.into_service_error();
                if service_err.is_not_found() {
                    Ok(false)
                } else {
                    Err(S3Error::HeadFailed(format!("{service_err}")))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn object_key_format() {
        // Can't create a real S3Library without AWS config,
        // so test the key formatting logic directly.
        let cid_bytes = [0xAA; 32];
        let key = format!("{}book/{}", "harmony-zenoh/", hex::encode(cid_bytes));
        assert_eq!(
            key,
            "harmony-zenoh/book/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        );
        assert_eq!(key.len(), "harmony-zenoh/book/".len() + 64);
    }

    #[test]
    fn object_key_different_prefix() {
        let cid_bytes = [0x00; 32];
        let key = format!("{}book/{}", "test/", hex::encode(cid_bytes));
        assert!(key.starts_with("test/book/"));
        assert_eq!(key.len(), "test/book/".len() + 64);
    }
}
```

**Important:** The `aws_sdk_s3` error handling API may differ slightly from what's shown. The subagent implementing this MUST read the `aws-sdk-s3` crate docs for the correct error matching patterns. Key methods to verify:
- `SdkError::into_service_error()` — converts to service-level error
- `GetObjectError::is_no_such_key()` — checks for missing object
- `HeadObjectError::is_not_found()` — checks for missing object in HEAD
- `ByteStream::from(Vec<u8>)` — creates body from bytes
- `StorageClass::IntelligentTiering` — verify the exact variant name

- [ ] **Step 5: Verify compilation**

```bash
cargo check -p harmony-s3
```

This will pull in the AWS SDK (~200 crates). May take a few minutes.

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml crates/harmony-s3/
git commit -m "feat(s3): scaffold harmony-s3 crate with S3Library"
```

---

### Task 2: Archivist module

Add the archivist logic to harmony-s3 — a function that subscribes to Zenoh content publish events and uploads durable books to S3.

**Files:**
- Create: `crates/harmony-s3/src/archivist.rs`
- Modify: `crates/harmony-s3/src/lib.rs` — add module
- Modify: `crates/harmony-s3/Cargo.toml` — add zenoh dependency

**Context:**
- Content publish key pattern: `harmony/content/publish/{cid_hex}` (namespace.rs:145)
- CID is 32 bytes, hex-encoded as 64 chars in the key expression suffix
- ContentFlags byte is the first byte of the CID (header[0] >> 4 gives the 4 flag bits)
- Ephemeral flag: bit 6 (0x40) of header[0]. If set → skip. If clear → archive.
- The archivist function takes an `S3Library` and a `zenoh::Session` and runs forever.

- [ ] **Step 1: Add zenoh dependency**

In `crates/harmony-s3/Cargo.toml`, add:
```toml
zenoh = { workspace = true }
```

- [ ] **Step 2: Create archivist.rs**

```rust
//! Archivist service — subscribes to Zenoh content publish events
//! and archives durable books to S3.

use crate::S3Library;
use tracing::{debug, info, warn};

/// Publish key expression pattern for content events.
const PUBLISH_SUB: &str = "harmony/content/publish/*";

/// Check if a ContentId (32 bytes) represents durable content.
///
/// Durable = ephemeral flag is NOT set (bit 6 of header byte 0).
fn is_durable(cid_bytes: &[u8; 32]) -> bool {
    // ContentFlags are in the top nibble of header[0]:
    //   bit 7 = encrypted, bit 6 = ephemeral, bit 5 = sha224, bit 4 = lsb_mode
    let ephemeral = cid_bytes[0] & 0x40 != 0;
    !ephemeral
}

/// Extract CID hex from a Zenoh key expression.
///
/// Key format: `harmony/content/publish/{cid_hex}`
/// Returns the 64-char hex suffix, or None if invalid.
fn extract_cid_hex(key_expr: &str) -> Option<&str> {
    let suffix = key_expr.rsplit('/').next()?;
    if suffix.len() == 64 {
        Some(suffix)
    } else {
        None
    }
}

/// Parse a CID hex string into 32 bytes.
fn parse_cid(hex_str: &str) -> Option<[u8; 32]> {
    let bytes = hex::decode(hex_str).ok()?;
    bytes.as_slice().try_into().ok()
}

/// Run the archivist loop.
///
/// Subscribes to content publish events on Zenoh, filters for durable
/// content, and uploads books to S3. Runs forever.
pub async fn run(s3: S3Library, session: zenoh::Session) {
    let subscriber = match session.declare_subscriber(PUBLISH_SUB).await {
        Ok(sub) => sub,
        Err(e) => {
            warn!(err = %e, "archivist failed to subscribe — exiting");
            return;
        }
    };

    info!("archivist started — archiving durable content to S3");

    while let Ok(sample) = subscriber.recv_async().await {
        let key_expr = sample.key_expr().as_str();

        // Extract and parse CID from key expression
        let cid_hex = match extract_cid_hex(key_expr) {
            Some(hex) => hex,
            None => {
                debug!(key_expr, "archivist: invalid publish key — skipping");
                continue;
            }
        };
        let cid_bytes = match parse_cid(cid_hex) {
            Some(bytes) => bytes,
            None => {
                debug!(cid_hex, "archivist: invalid CID hex — skipping");
                continue;
            }
        };

        // Check if durable
        if !is_durable(&cid_bytes) {
            debug!(cid = cid_hex, "archivist: ephemeral content — skipping");
            continue;
        }

        // Check if already archived (HEAD — cheap)
        match s3.exists(&cid_bytes).await {
            Ok(true) => {
                debug!(cid = cid_hex, "archivist: already in S3 — skipping");
                continue;
            }
            Ok(false) => {} // proceed to upload
            Err(e) => {
                warn!(cid = cid_hex, err = %e, "archivist: S3 HEAD failed — skipping");
                continue;
            }
        }

        // Upload to S3
        let data = sample.payload().to_bytes().to_vec();
        match s3.put_book(&cid_bytes, &data).await {
            Ok(()) => {
                info!(
                    cid = cid_hex,
                    size = data.len(),
                    "archivist: book archived to S3"
                );
            }
            Err(e) => {
                warn!(cid = cid_hex, err = %e, "archivist: S3 PUT failed");
            }
        }
    }

    warn!("archivist: subscriber closed — exiting");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_durable_public_durable() {
        // Header byte 0: encrypted=0, ephemeral=0 → durable
        let mut cid = [0u8; 32];
        cid[0] = 0x00; // no flags set
        assert!(is_durable(&cid));
    }

    #[test]
    fn is_durable_encrypted_durable() {
        // Header byte 0: encrypted=1, ephemeral=0 → durable
        let mut cid = [0u8; 32];
        cid[0] = 0x80; // encrypted flag set
        assert!(is_durable(&cid));
    }

    #[test]
    fn is_not_durable_ephemeral() {
        // Header byte 0: ephemeral=1 → not durable
        let mut cid = [0u8; 32];
        cid[0] = 0x40; // ephemeral flag set
        assert!(!is_durable(&cid));
    }

    #[test]
    fn is_not_durable_encrypted_ephemeral() {
        // Header byte 0: encrypted=1, ephemeral=1 → not durable
        let mut cid = [0u8; 32];
        cid[0] = 0xC0; // both flags set
        assert!(!is_durable(&cid));
    }

    #[test]
    fn extract_cid_hex_valid() {
        let key = "harmony/content/publish/00aabbccddee11223344556677889900aabbccddee11223344556677889900aa";
        let hex = extract_cid_hex(key).unwrap();
        assert_eq!(hex.len(), 64);
        assert_eq!(hex, "00aabbccddee11223344556677889900aabbccddee11223344556677889900aa");
    }

    #[test]
    fn extract_cid_hex_wrong_length() {
        let key = "harmony/content/publish/tooshort";
        assert!(extract_cid_hex(key).is_none());
    }

    #[test]
    fn parse_cid_valid() {
        let hex = "00aabbccddee11223344556677889900aabbccddee11223344556677889900aa";
        let cid = parse_cid(hex).unwrap();
        assert_eq!(cid[0], 0x00);
        assert_eq!(cid[1], 0xAA);
        assert_eq!(cid[31], 0xAA);
    }

    #[test]
    fn parse_cid_invalid_hex() {
        assert!(parse_cid("not_valid_hex_at_all_xxxx").is_none());
    }
}
```

- [ ] **Step 3: Add module to lib.rs**

Add to `lib.rs`:
```rust
pub mod archivist;
```

- [ ] **Step 4: Run tests**

```bash
cargo test -p harmony-s3
```
Expected: all tests pass (2 from Task 1 + 7 from archivist).

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-s3/
git commit -m "feat(s3): add archivist module with durability filtering"
```

---

### Task 3: Event loop integration and config

Wire the archivist into harmony-node with configuration.

**Files:**
- Modify: `crates/harmony-node/Cargo.toml` — add harmony-s3 dependency
- Modify: `crates/harmony-node/src/config.rs` — add ArchivistConfig
- Modify: `crates/harmony-node/src/event_loop.rs` — spawn archivist
- Modify: `crates/harmony-node/src/main.rs` — thread config

**Context:**
- ConfigFile at config.rs:56-76, uses `#[serde(deny_unknown_fields)]`
- Session created at event_loop.rs:238
- The archivist should spawn after the session is created, alongside the rawlink bridge
- `run()` function signature at event_loop.rs:190-201

- [ ] **Step 1: Add dependency**

In `crates/harmony-node/Cargo.toml`, add:
```toml
harmony-s3 = { workspace = true, optional = true }
```

Add feature:
```toml
archivist = ["harmony-s3"]
```

In root `Cargo.toml`, add to workspace dependencies:
```toml
harmony-s3 = { path = "crates/harmony-s3" }
```

- [ ] **Step 2: Add ArchivistConfig to config.rs**

```rust
#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct ArchivistConfig {
    pub bucket: String,
    pub prefix: String,
    pub region: Option<String>,
}
```

Add to `ConfigFile`:
```rust
pub archivist: Option<ArchivistConfig>,
```

- [ ] **Step 3: Spawn archivist in event_loop.rs**

Add `archivist_config: Option<crate::config::ArchivistConfig>` parameter to `run()`.

After the Zenoh session is created (~line 238), add:

```rust
#[cfg(feature = "archivist")]
if let Some(ref archivist) = archivist_config {
    match harmony_s3::S3Library::new(
        archivist.bucket.clone(),
        archivist.prefix.clone(),
        archivist.region.clone(),
    ).await {
        Ok(s3) => {
            let session = session.clone();
            tokio::spawn(async move {
                harmony_s3::archivist::run(s3, session).await;
            });
            tracing::info!(
                bucket = %archivist.bucket,
                prefix = %archivist.prefix,
                "S3 archivist started"
            );
        }
        Err(e) => {
            tracing::warn!(err = %e, "S3 archivist failed to start — continuing without archival");
        }
    }
}
```

- [ ] **Step 4: Thread config from main.rs**

Find where config fields are extracted and passed to `event_loop::run()`. Add:
```rust
let archivist_config = config_file.archivist;
```

Pass as a new parameter to `event_loop::run()`. On non-archivist builds, the parameter is `Option<ArchivistConfig>` set to `None` via the config file.

Note: The `archivist_config` parameter should NOT be cfg-gated — it's always `Option<ArchivistConfig>`. Only the spawn code is cfg-gated. This way the function signature is stable across features.

- [ ] **Step 5: Compile and test**

```bash
cargo test -p harmony-node
cargo test -p harmony-s3
```

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-node/ crates/harmony-s3/ Cargo.toml
git commit -m "feat(node): integrate S3 archivist with config and feature gate"
```
