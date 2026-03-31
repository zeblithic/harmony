//! NAR push pipeline — dump, ingest, sign, persist.
//!
//! Provides functions for pushing Nix store paths into the Harmony CAS:
//! - Dumping NAR archives from the Nix store
//! - Ingesting NAR bytes into the CAS (chunked via FastCDC)
//! - Building and signing narinfo metadata
//! - Creating memos linking store hashes to narinfo CIDs
//!
//! No async: all I/O is synchronous (intended for spawn_blocking contexts).

#![cfg(feature = "nix-cache")]

use std::error::Error;
use std::io;
use std::path::Path;
use std::process::Command;

use harmony_content::{
    book::{BookStore, MemoryBookStore},
    bundle,
    chunker::ChunkerConfig,
    cid::{ContentFlags, ContentId},
    dag,
};
use harmony_crypto::hash::full_hash;
use harmony_identity::pq_identity::PqPrivateIdentity;
use harmony_memo::create::create_memo;

use crate::{
    disk_io::write_book,
    memo_io::{scan_memos, write_memo},
    narinfo::{NarInfo, NixSigningKey},
};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Result of pushing a single Nix store path into the CAS.
#[derive(Debug)]
pub struct PushResult {
    /// The full store path that was pushed (e.g. `/nix/store/<hash>-<name>`).
    pub store_path: String,
    /// CID of the root NAR book/bundle in the CAS.
    pub nar_root_cid: ContentId,
    /// CID of the narinfo book in the CAS.
    pub narinfo_cid: ContentId,
    /// Size of the NAR data in bytes.
    pub nar_size: u64,
    /// `true` if the push was skipped because the memo already exists.
    pub skipped: bool,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Push a store path into the Harmony CAS.
///
/// Steps:
/// 1. Extract the 32-char store hash from `store_path`.
/// 2. Check for an existing memo (idempotent skip).
/// 3. Hash the NAR bytes → NarHash.
/// 4. Ingest NAR bytes into a [`MemoryBookStore`] via FastCDC DAG.
/// 5. Persist all books to disk.
/// 6. Build, sign, and store the narinfo text.
/// 7. Create and persist a memo linking store-hash CID → narinfo CID.
pub fn push_store_path(
    store_path: &str,
    nar_data: &[u8],
    signing_key: &NixSigningKey,
    data_dir: &Path,
    identity: &PqPrivateIdentity,
) -> Result<PushResult, Box<dyn Error>> {
    push_store_path_with_refs(store_path, nar_data, &[], signing_key, data_dir, identity)
}

/// Push a store path with explicit references into the Harmony CAS.
///
/// Like [`push_store_path`] but fills in the `References` field in the
/// generated narinfo with the provided basenames (no `/nix/store/` prefix).
pub fn push_store_path_with_refs(
    store_path: &str,
    nar_data: &[u8],
    references: &[String],
    signing_key: &NixSigningKey,
    data_dir: &Path,
    identity: &PqPrivateIdentity,
) -> Result<PushResult, Box<dyn Error>> {
    // --- 1. Extract the 32-char Nix store hash ---
    let store_hash = extract_store_hash(store_path)?;

    // --- 2. Idempotency check: skip if memo already exists ---
    let input_cid = ContentId::for_book(store_hash.as_bytes(), ContentFlags::default())
        .map_err(|e| format!("input CID derivation failed: {e}"))?;

    let existing = scan_memos(data_dir);
    if existing
        .iter()
        .any(|(m, _)| m.input.to_bytes() == input_cid.to_bytes())
    {
        // The memo already exists; return a skipped result with placeholder CIDs.
        let placeholder = ContentId::from_bytes([0u8; 32]);
        return Ok(PushResult {
            store_path: store_path.to_string(),
            nar_root_cid: placeholder,
            narinfo_cid: placeholder,
            nar_size: nar_data.len() as u64,
            skipped: true,
        });
    }

    // --- 3. Compute NarHash ---
    let sha256: [u8; 32] = full_hash(nar_data);
    let nar_hash = NarInfo::format_nix_hash(&sha256);

    // --- 4. Ingest NAR into CAS ---
    let mut store = MemoryBookStore::new();
    let config = ChunkerConfig::DEFAULT;
    let nar_root_cid = dag::ingest(nar_data, &config, &mut store)
        .map_err(|e| format!("dag::ingest failed: {e}"))?;

    // --- 5. Persist all DAG books to disk ---
    persist_dag_to_disk(&nar_root_cid, &store, data_dir)?;

    // --- 6. Build narinfo, sign, store as a book ---
    let nar_size = nar_data.len() as u64;
    let nar_url = format!("nar/{}.nar", hex::encode(nar_root_cid.to_bytes()));

    let mut narinfo = NarInfo {
        store_path: store_path.to_string(),
        url: nar_url,
        nar_hash,
        nar_size,
        references: references.to_vec(),
        sig: None,
    };
    narinfo.sign(signing_key);

    let narinfo_text = narinfo.to_text();
    let narinfo_bytes = narinfo_text.as_bytes();

    // Store the narinfo text as its own leaf book.
    let narinfo_cid = store
        .insert(narinfo_bytes)
        .map_err(|e| format!("narinfo book insert failed: {e}"))?;

    // Write the narinfo book to disk.
    write_book(data_dir, &narinfo_cid, narinfo_bytes)?;

    // --- 7. Create and persist memo ---
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs();
    let expires_at = now + 365 * 24 * 3600;

    let memo = create_memo(
        input_cid,
        narinfo_cid,
        identity,
        &mut rand::rngs::OsRng,
        now,
        expires_at,
    )
    .map_err(|e| format!("create_memo failed: {e}"))?;

    write_memo(data_dir, &memo)?;

    Ok(PushResult {
        store_path: store_path.to_string(),
        nar_root_cid,
        narinfo_cid,
        nar_size,
        skipped: false,
    })
}

/// Run `nix-store --dump <store_path>` and return the NAR bytes.
pub fn dump_nar(store_path: &str) -> Result<Vec<u8>, Box<dyn Error>> {
    let output = Command::new("nix-store")
        .arg("--dump")
        .arg(store_path)
        .output()?;

    if !output.status.success() {
        return Err(format!(
            "nix-store --dump failed (exit {}): {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }

    Ok(output.stdout)
}

/// Run `nix-store -qR <store_path>` and return the full closure as store paths.
pub fn enumerate_closure(store_path: &str) -> Result<Vec<String>, Box<dyn Error>> {
    let output = Command::new("nix-store")
        .args(["-qR", store_path])
        .output()?;

    if !output.status.success() {
        return Err(format!(
            "nix-store -qR failed (exit {}): {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }

    let paths = String::from_utf8(output.stdout)?
        .lines()
        .filter(|l| !l.is_empty())
        .map(str::to_string)
        .collect();

    Ok(paths)
}

/// Run `nix-store -q --references <store_path>` and return basenames (no `/nix/store/` prefix).
pub fn get_references(store_path: &str) -> Result<Vec<String>, Box<dyn Error>> {
    let output = Command::new("nix-store")
        .args(["-q", "--references", store_path])
        .output()?;

    if !output.status.success() {
        return Err(format!(
            "nix-store -q --references failed (exit {}): {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }

    let basenames = String::from_utf8(output.stdout)?
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| {
            l.strip_prefix("/nix/store/")
                .unwrap_or(l)
                .to_string()
        })
        .collect();

    Ok(basenames)
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Extract the 32-character Nix store hash from a full store path.
///
/// The Nix store path format is `/nix/store/<32chars>-<name>`.
/// Returns an error if the path does not begin with `/nix/store/` or if the
/// hash portion is not exactly 32 characters.
fn extract_store_hash(store_path: &str) -> Result<String, Box<dyn Error>> {
    let rest = store_path
        .strip_prefix("/nix/store/")
        .ok_or_else(|| format!("store path must start with /nix/store/, got: {store_path:?}"))?;

    // The hash is the first 32 characters before the first `-`.
    let hash_part = rest.split('-').next().unwrap_or("");
    if hash_part.len() != 32 {
        return Err(format!(
            "store hash must be exactly 32 characters, got {} in path {store_path:?}",
            hash_part.len()
        )
        .into());
    }

    Ok(hash_part.to_string())
}

/// Recursively walk the DAG from `root` and write every book to disk.
///
/// For bundle CIDs (depth > 0), parses children via [`bundle::parse_bundle`]
/// and recurses. Sentinel CIDs are skipped. Already-written books are
/// detected by presence in `store`; on-disk deduplication is handled by the
/// filesystem (write-to-temp-then-rename is idempotent).
fn persist_dag_to_disk(
    root: &ContentId,
    store: &MemoryBookStore,
    data_dir: &Path,
) -> Result<(), io::Error> {
    persist_dag_recursive(root, store, data_dir)
}

fn persist_dag_recursive(
    cid: &ContentId,
    store: &MemoryBookStore,
    data_dir: &Path,
) -> Result<(), io::Error> {
    // Skip sentinels — they carry no on-disk data.
    if cid.is_sentinel() {
        return Ok(());
    }

    let data = store
        .get(cid)
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, format!("CID not in store: {cid:?}")))?;

    // Write this node to disk.
    write_book(data_dir, cid, data)?;

    // For bundles, recurse into children.
    if cid.depth() > 0 {
        let children: Vec<ContentId> = {
            let children_slice = bundle::parse_bundle(data)
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid bundle"))?;
            children_slice.to_vec()
        };
        for child in &children {
            persist_dag_recursive(child, store, data_dir)?;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
    use crate::disk_io::read_book;
    use crate::memo_io::scan_memos;
    use crate::narinfo::NarInfo;
    use harmony_content::{
        book::{BookStore, MemoryBookStore},
        chunker::ChunkerConfig,
        dag,
    };

    fn test_signing_key() -> NixSigningKey {
        let seed = [1u8; 32];
        let key = ed25519_dalek::SigningKey::from_bytes(&seed);
        let pubkey = key.verifying_key().to_bytes();
        let mut combined = Vec::with_capacity(64);
        combined.extend_from_slice(&seed);
        combined.extend_from_slice(&pubkey);
        NixSigningKey::from_nix_format(&format!("test-key-1:{}", BASE64.encode(&combined)))
            .unwrap()
    }

    fn test_identity() -> PqPrivateIdentity {
        PqPrivateIdentity::generate(&mut rand::rngs::OsRng)
    }

    /// Minimal valid-ish NAR data (enough bytes to avoid empty-data errors).
    fn fake_nar_data() -> Vec<u8> {
        // FastCDC needs a non-empty slice; real NAR format not required for unit tests.
        b"fake-nar-content-for-testing-purposes-0000000000000000".to_vec()
    }

    fn valid_store_path() -> &'static str {
        "/nix/store/aaaabbbbccccddddeeeeffffgggghhhh-test-package-1.0"
    }

    // -----------------------------------------------------------------------

    #[test]
    fn push_store_path_creates_memo_and_books() {
        let dir = tempfile::TempDir::new().unwrap();
        let nar_data = fake_nar_data();
        let key = test_signing_key();
        let identity = test_identity();

        let result = push_store_path(
            valid_store_path(),
            &nar_data,
            &key,
            dir.path(),
            &identity,
        )
        .expect("push_store_path should succeed");

        assert!(!result.skipped, "first push should not be skipped");
        assert_eq!(result.nar_size, nar_data.len() as u64);

        // Memo should be on disk.
        let memos = scan_memos(dir.path());
        assert_eq!(memos.len(), 1, "exactly one memo should be written");

        // Narinfo book should be readable and parseable.
        let narinfo_bytes = read_book(dir.path(), &result.narinfo_cid)
            .expect("narinfo book should be on disk");
        let narinfo_text = std::str::from_utf8(&narinfo_bytes).expect("narinfo must be UTF-8");
        let parsed = NarInfo::from_text(narinfo_text).expect("narinfo must parse");
        assert_eq!(parsed.store_path, valid_store_path());

        // Narinfo must be signed.
        assert!(
            parsed.sig.is_some(),
            "narinfo should have a signature after push"
        );
    }

    #[test]
    fn push_same_path_twice_is_idempotent() {
        let dir = tempfile::TempDir::new().unwrap();
        let nar_data = fake_nar_data();
        let key = test_signing_key();
        let identity = test_identity();

        let first = push_store_path(
            valid_store_path(),
            &nar_data,
            &key,
            dir.path(),
            &identity,
        )
        .expect("first push should succeed");
        assert!(!first.skipped);

        let second = push_store_path(
            valid_store_path(),
            &nar_data,
            &key,
            dir.path(),
            &identity,
        )
        .expect("second push should succeed");
        assert!(second.skipped, "second push should be skipped");

        // Only one memo should exist on disk.
        let memos = scan_memos(dir.path());
        assert_eq!(memos.len(), 1, "only one memo should exist after two pushes");
    }

    #[test]
    fn push_with_refs_includes_references() {
        let dir = tempfile::TempDir::new().unwrap();
        let nar_data = fake_nar_data();
        let key = test_signing_key();
        let identity = test_identity();

        let refs = vec![
            "aaaa0000111122223333bbbbccccdddd-dep-a-1.0".to_string(),
            "bbbb0000111122223333aaaa44445555-dep-b-2.3".to_string(),
        ];

        let result = push_store_path_with_refs(
            valid_store_path(),
            &nar_data,
            &refs,
            &key,
            dir.path(),
            &identity,
        )
        .expect("push_with_refs should succeed");

        assert!(!result.skipped);

        let narinfo_bytes = read_book(dir.path(), &result.narinfo_cid)
            .expect("narinfo should be on disk");
        let narinfo_text = std::str::from_utf8(&narinfo_bytes).unwrap();
        let parsed = NarInfo::from_text(narinfo_text).expect("narinfo must parse");

        assert_eq!(parsed.references.len(), 2);
        assert!(
            narinfo_text.contains("References:"),
            "narinfo text must contain References line"
        );
        for r in &refs {
            assert!(
                narinfo_text.contains(r.as_str()),
                "narinfo must contain reference {r}"
            );
        }
    }

    #[test]
    fn push_rejects_invalid_store_path() {
        let dir = tempfile::TempDir::new().unwrap();
        let nar_data = fake_nar_data();
        let key = test_signing_key();
        let identity = test_identity();

        let err = push_store_path(
            "/not/a/nix/store/path",
            &nar_data,
            &key,
            dir.path(),
            &identity,
        )
        .expect_err("should fail for non-nix-store path");

        assert!(
            err.to_string().contains("/nix/store/"),
            "error should mention the expected prefix: {err}"
        );
    }

    #[test]
    fn persist_dag_writes_all_books() {
        let dir = tempfile::TempDir::new().unwrap();

        // Ingest 2 MiB of data to force multiple chunks and a bundle tree.
        let large_data: Vec<u8> = (0u32..=524287u32)
            .flat_map(|i| i.to_le_bytes())
            .collect();
        assert_eq!(large_data.len(), 2 * 1024 * 1024);

        let mut store = MemoryBookStore::new();
        let config = ChunkerConfig::DEFAULT;
        let root_cid = dag::ingest(&large_data, &config, &mut store)
            .expect("ingest should succeed for large data");

        // Persist all books.
        persist_dag_to_disk(&root_cid, &store, dir.path()).expect("persist should succeed");

        // Verify: reassemble from a disk-only store.
        let mut disk_store = MemoryBookStore::new();
        load_dag_from_disk_for_test(dir.path(), &root_cid, &mut disk_store)
            .expect("load from disk should succeed");

        let reassembled = dag::reassemble(&root_cid, &disk_store)
            .expect("reassemble from disk-loaded store should succeed");

        assert_eq!(
            reassembled, large_data,
            "reassembled data must match original"
        );
    }

    /// Test helper: load all books for a DAG from disk into a MemoryBookStore.
    fn load_dag_from_disk_for_test(
        data_dir: &Path,
        cid: &ContentId,
        store: &mut MemoryBookStore,
    ) -> Result<(), io::Error> {
        if cid.is_sentinel() {
            return Ok(());
        }
        if store.contains(cid) {
            return Ok(());
        }
        let data = read_book(data_dir, cid)?;
        if cid.depth() > 0 {
            let children: Vec<ContentId> = {
                let s = bundle::parse_bundle(&data)
                    .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid bundle"))?;
                s.to_vec()
            };
            store.store(*cid, data);
            for child in &children {
                load_dag_from_disk_for_test(data_dir, child, store)?;
            }
        } else {
            store.store(*cid, data);
        }
        Ok(())
    }
}
