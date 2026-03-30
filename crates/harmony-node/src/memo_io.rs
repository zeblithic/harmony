//! Synchronous disk I/O helpers for memo persistence.
//!
//! These functions are intended to be called inside `tokio::task::spawn_blocking`.
//!
//! File layout: `{data_dir}/memo/{input_hex[8..10]}/{input_hex}_{output_hex}_{signer_hex}`
//!
//! The fan-out prefix `input_hex[8..10]` = byte 4 of the raw input CID bytes,
//! giving 256 possible subdirectories.

use harmony_memo::Memo;
use std::{
    fs,
    io::{self, Write as _},
    path::PathBuf,
};

// ---------------------------------------------------------------------------
// Path helper
// ---------------------------------------------------------------------------

/// Return the filesystem path for a memo under `data_dir`.
///
/// Layout: `{data_dir}/memo/{input_hex[8..10]}/{input_hex}_{output_hex}_{signer_hex}`
///
/// - `input_hex` = 64-char hex of `memo.input` (32 bytes)
/// - `output_hex` = 64-char hex of `memo.output` (32 bytes)
/// - `signer_hex` = 32-char hex of `memo.credential.issuer.hash` (16 bytes)
/// - `input_hex[8..10]` = 2-char sharding prefix (byte 4 of input CID)
pub fn memo_path(data_dir: &std::path::Path, memo: &Memo) -> PathBuf {
    let input_hex = hex::encode(memo.input.to_bytes());
    let output_hex = hex::encode(memo.output.to_bytes());
    let signer_hex = hex::encode(memo.credential.issuer.hash);
    let prefix = &input_hex[8..10];
    let filename = format!("{input_hex}_{output_hex}_{signer_hex}");
    data_dir.join("memo").join(prefix).join(filename)
}

// ---------------------------------------------------------------------------
// Write
// ---------------------------------------------------------------------------

/// Serialize and write `memo` under `data_dir`, creating parent directories as needed.
///
/// Uses write-to-temp-then-rename for crash safety: a power loss during write
/// leaves the temp file (not the final path), so `scan_memos` never sees a
/// truncated memo.
///
/// Returns the number of bytes written.
pub fn write_memo(data_dir: &std::path::Path, memo: &Memo) -> Result<u64, io::Error> {
    let bytes = harmony_memo::serialize(memo)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;

    let path = memo_path(data_dir, memo);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp_path = path.with_extension("tmp");
    {
        let mut file = fs::File::create(&tmp_path)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
    }
    fs::rename(&tmp_path, &path)?;
    Ok(bytes.len() as u64)
}

// ---------------------------------------------------------------------------
// Read
// ---------------------------------------------------------------------------

/// Read and deserialize a memo from an absolute `path`.
///
/// Returns `Err` with `ErrorKind::NotFound` if the file does not exist,
/// or `ErrorKind::InvalidData` if deserialization fails.
pub fn read_memo(path: &std::path::Path) -> Result<Memo, io::Error> {
    let bytes = fs::read(path)?;
    harmony_memo::deserialize(&bytes)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
}

// ---------------------------------------------------------------------------
// Delete
// ---------------------------------------------------------------------------

/// Delete the memo file from `data_dir`.
///
/// Returns `Err` with `ErrorKind::NotFound` if the file does not exist.
pub fn delete_memo(data_dir: &std::path::Path, memo: &Memo) -> Result<(), io::Error> {
    fs::remove_file(memo_path(data_dir, memo))
}

// ---------------------------------------------------------------------------
// Scan
// ---------------------------------------------------------------------------

/// Walk `{data_dir}/memo/` and return all valid `Memo`s found, with their on-disk sizes.
///
/// Files ending in `.tmp` (interrupted writes) are silently skipped.
/// Files that fail to deserialize are skipped with a `tracing::warn!`.
pub fn scan_memos(data_dir: &std::path::Path) -> Vec<(Memo, u64)> {
    let memo_root = PathBuf::from(data_dir).join("memo");
    let mut memos = Vec::new();

    let prefix_dir = match fs::read_dir(&memo_root) {
        Ok(rd) => rd,
        Err(e) => {
            if e.kind() != io::ErrorKind::NotFound {
                tracing::warn!("scan_memos: cannot read {}: {}", memo_root.display(), e);
            }
            return memos;
        }
    };

    for prefix_entry in prefix_dir {
        let prefix_entry = match prefix_entry {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!("scan_memos: directory entry error: {}", e);
                continue;
            }
        };
        let prefix_path = prefix_entry.path();
        if !prefix_path.is_dir() {
            continue;
        }

        let entries = match fs::read_dir(&prefix_path) {
            Ok(rd) => rd,
            Err(e) => {
                tracing::warn!("scan_memos: cannot read {}: {}", prefix_path.display(), e);
                continue;
            }
        };

        for entry in entries {
            let entry = match entry {
                Ok(e) => e,
                Err(e) => {
                    tracing::warn!("scan_memos: entry error: {}", e);
                    continue;
                }
            };
            let file_path = entry.path();
            let file_name = entry.file_name();
            let name = file_name.to_string_lossy();

            // Skip interrupted writes.
            if name.ends_with(".tmp") {
                continue;
            }

            let size = match entry.metadata() {
                Ok(meta) => meta.len(),
                Err(e) => {
                    tracing::warn!(path = %file_path.display(), error = %e, "scan_memos: skipping memo: metadata read failed");
                    continue;
                }
            };

            match read_memo(&file_path) {
                Ok(memo) => {
                    tracing::debug!("scan_memos: discovered memo {}", name);
                    memos.push((memo, size));
                }
                Err(e) => {
                    tracing::warn!(path = %file_path.display(), error = %e, "scan_memos: skipping invalid memo file");
                }
            }
        }
    }

    memos
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use harmony_content::cid::ContentId;
    use harmony_identity::pq_identity::PqPrivateIdentity;
    use harmony_memo::create::create_memo;

    fn make_cid(byte: u8) -> ContentId {
        ContentId::from_bytes([byte; 32])
    }

    fn make_test_memo(input_byte: u8, output_byte: u8) -> Memo {
        let identity = PqPrivateIdentity::generate(&mut rand::rngs::OsRng);
        create_memo(
            make_cid(input_byte),
            make_cid(output_byte),
            &identity,
            &mut rand::rngs::OsRng,
            1000,
            9999,
        )
        .expect("create_memo")
    }

    #[test]
    fn write_and_read_roundtrip() {
        let dir = tempfile::TempDir::new().unwrap();
        let memo = make_test_memo(0x11, 0x22);

        let path = memo_path(dir.path(), &memo);
        write_memo(dir.path(), &memo).expect("write should succeed");

        let restored = read_memo(&path).expect("read should succeed");
        assert_eq!(restored.input, memo.input);
        assert_eq!(restored.output, memo.output);
        assert_eq!(restored.credential.issuer, memo.credential.issuer);
    }

    #[test]
    fn scan_discovers_written_memos() {
        let dir = tempfile::TempDir::new().unwrap();

        let memo_a = make_test_memo(0x01, 0x02);
        let memo_b = make_test_memo(0x03, 0x04);
        let memo_c = make_test_memo(0x05, 0x06);

        write_memo(dir.path(), &memo_a).unwrap();
        write_memo(dir.path(), &memo_b).unwrap();
        write_memo(dir.path(), &memo_c).unwrap();

        let found = scan_memos(dir.path());
        assert_eq!(found.len(), 3);

        // Verify all three input CIDs are present.
        let mut inputs: Vec<_> = found.iter().map(|(m, _)| m.input).collect();
        inputs.sort_by_key(|c| c.to_bytes());
        let mut expected_inputs = vec![memo_a.input, memo_b.input, memo_c.input];
        expected_inputs.sort_by_key(|c| c.to_bytes());
        assert_eq!(inputs, expected_inputs);
    }

    #[test]
    fn delete_removes_file() {
        let dir = tempfile::TempDir::new().unwrap();
        let memo = make_test_memo(0xAB, 0xCD);

        write_memo(dir.path(), &memo).unwrap();
        let path = memo_path(dir.path(), &memo);
        assert!(path.exists(), "file should exist after write");

        delete_memo(dir.path(), &memo).unwrap();
        assert!(!path.exists(), "file should be gone after delete");
    }

    #[test]
    fn scan_skips_invalid_files() {
        let dir = tempfile::TempDir::new().unwrap();

        // Write a valid memo.
        let memo = make_test_memo(0x77, 0x88);
        write_memo(dir.path(), &memo).unwrap();

        // Inject a garbage file in the same prefix directory.
        let input_hex = hex::encode(memo.input.to_bytes());
        let prefix = &input_hex[8..10];
        let bad_file = dir
            .path()
            .join("memo")
            .join(prefix)
            .join("garbage_file.bin");
        fs::write(&bad_file, b"this is not a valid memo").unwrap();

        let found = scan_memos(dir.path());

        // Only the valid memo should be returned.
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].0.input, memo.input);
        assert_eq!(found[0].0.output, memo.output);
    }

    #[test]
    fn scan_empty_returns_empty() {
        let dir = tempfile::TempDir::new().unwrap();
        let found = scan_memos(dir.path());
        assert!(found.is_empty());
    }

    #[test]
    fn write_returns_byte_size() {
        let dir = tempfile::TempDir::new().unwrap();
        let memo = make_test_memo(0x33, 0x44);

        let reported_size = write_memo(dir.path(), &memo).expect("write should succeed");

        let path = memo_path(dir.path(), &memo);
        let disk_size = fs::metadata(&path).unwrap().len();

        assert_eq!(reported_size, disk_size);
        assert!(reported_size > 0);
    }
}
