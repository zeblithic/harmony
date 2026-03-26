//! Synchronous disk I/O helpers for CAS book persistence.
//!
//! These functions are intended to be called inside `tokio::task::spawn_blocking`.
//!
//! File layout: `{data_dir}/book/{hex_cid[8..10]}/{hex_cid}`
//!
//! The fan-out prefix `hex_cid[8..10]` corresponds to byte 4 of the CID
//! (the first hash byte), giving 256 possible subdirectories.

use harmony_content::cid::ContentId;
use std::{
    fs,
    io::{self, Write as _},
    path::PathBuf,
};

// ---------------------------------------------------------------------------
// Path helper
// ---------------------------------------------------------------------------

/// Return the filesystem path for a CAS book CID under `data_dir`.
///
/// Layout: `{data_dir}/book/{hex_cid[8..10]}/{hex_cid}`
pub fn book_path(data_dir: &str, cid: &ContentId) -> PathBuf {
    let hex_cid = hex::encode(cid.to_bytes());
    let prefix = &hex_cid[8..10];
    PathBuf::from(data_dir)
        .join("book")
        .join(prefix)
        .join(&hex_cid)
}

// ---------------------------------------------------------------------------
// Write
// ---------------------------------------------------------------------------

/// Write `data` for `cid` under `data_dir`, creating parent directories as needed.
pub fn write_book(data_dir: &str, cid: &ContentId, data: &[u8]) -> Result<(), io::Error> {
    let path = book_path(data_dir, cid);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = fs::File::create(&path)?;
    file.write_all(data)?;
    file.sync_all()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Read
// ---------------------------------------------------------------------------

/// Read raw bytes for `cid` from `data_dir`.
///
/// Returns `Err` with `ErrorKind::NotFound` if the file does not exist.
pub fn read_book(data_dir: &str, cid: &ContentId) -> Result<Vec<u8>, io::Error> {
    let path = book_path(data_dir, cid);
    fs::read(path)
}

// ---------------------------------------------------------------------------
// Scan
// ---------------------------------------------------------------------------

/// Walk `{data_dir}/book/` and return all valid `ContentId`s found.
///
/// Files whose names are not valid 64-hex-char CIDs are skipped with a warning.
pub fn scan_books(data_dir: &str) -> Vec<ContentId> {
    let book_root = PathBuf::from(data_dir).join("book");
    let mut cids = Vec::new();

    let prefix_dir = match fs::read_dir(&book_root) {
        Ok(rd) => rd,
        Err(e) => {
            if e.kind() != io::ErrorKind::NotFound {
                tracing::warn!("scan_books: cannot read {}: {}", book_root.display(), e);
            }
            return cids;
        }
    };

    for prefix_entry in prefix_dir {
        let prefix_entry = match prefix_entry {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!("scan_books: directory entry error: {}", e);
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
                tracing::warn!("scan_books: cannot read {}: {}", prefix_path.display(), e);
                continue;
            }
        };

        for entry in entries {
            let entry = match entry {
                Ok(e) => e,
                Err(e) => {
                    tracing::warn!("scan_books: entry error: {}", e);
                    continue;
                }
            };
            let file_name = entry.file_name();
            let name = file_name.to_string_lossy();

            // CID filenames are exactly 64 hex chars (32 bytes).
            if name.len() != 64 {
                tracing::warn!("scan_books: skipping unexpected file: {}", name);
                continue;
            }

            let bytes = match hex::decode(name.as_ref()) {
                Ok(b) => b,
                Err(_) => {
                    tracing::warn!("scan_books: skipping non-hex filename: {}", name);
                    continue;
                }
            };

            let arr: [u8; 32] = match bytes.try_into() {
                Ok(a) => a,
                Err(_) => {
                    tracing::warn!("scan_books: skipping filename with wrong length: {}", name);
                    continue;
                }
            };

            tracing::debug!("scan_books: discovered CID {}", name);
            cids.push(ContentId::from_bytes(arr));
        }
    }

    cids
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use harmony_content::cid::ContentFlags;

    fn make_cid(data: &[u8]) -> ContentId {
        ContentId::for_book(data, ContentFlags::default()).unwrap()
    }

    fn dir_str(dir: &tempfile::TempDir) -> String {
        dir.path().to_string_lossy().into_owned()
    }

    #[test]
    fn write_and_read_round_trip() {
        let dir = tempfile::TempDir::new().unwrap();
        let data = b"hello harmony disk_io";
        let cid = make_cid(data);

        write_book(&dir_str(&dir), &cid, data).expect("write should succeed");
        let read_back = read_book(&dir_str(&dir), &cid).expect("read should succeed");

        assert_eq!(read_back, data);
    }

    #[test]
    fn write_creates_prefix_directory() {
        let dir = tempfile::TempDir::new().unwrap();
        let data = b"prefix dir test";
        let cid = make_cid(data);

        write_book(&dir_str(&dir), &cid, data).expect("write should succeed");

        let path = book_path(&dir_str(&dir), &cid);
        assert!(path.exists(), "book file should exist at expected path");

        let prefix_dir = path.parent().unwrap();
        assert!(prefix_dir.is_dir(), "prefix directory should have been created");
    }

    #[test]
    fn scan_discovers_written_books() {
        let dir = tempfile::TempDir::new().unwrap();

        let cid_a = make_cid(b"book a");
        let cid_b = make_cid(b"book b");
        let cid_c = make_cid(b"book c");

        write_book(&dir_str(&dir), &cid_a, b"book a").unwrap();
        write_book(&dir_str(&dir), &cid_b, b"book b").unwrap();
        write_book(&dir_str(&dir), &cid_c, b"book c").unwrap();

        let mut found = scan_books(&dir_str(&dir));
        found.sort_by_key(|c| c.to_bytes());

        let mut expected = vec![cid_a, cid_b, cid_c];
        expected.sort_by_key(|c| c.to_bytes());

        assert_eq!(found, expected);
    }

    #[test]
    fn scan_skips_invalid_filenames() {
        let dir = tempfile::TempDir::new().unwrap();
        let data_dir = dir_str(&dir);

        // Write a valid book first.
        let cid = make_cid(b"valid book");
        write_book(&data_dir, &cid, b"valid book").unwrap();

        // Inject a garbage file in the same prefix directory.
        let prefix = &hex::encode(cid.to_bytes())[8..10].to_owned();
        let bad_file = PathBuf::from(&data_dir).join("book").join(prefix).join("not-a-cid.txt");
        fs::write(&bad_file, b"garbage").unwrap();

        // Inject a file with wrong length (too short) in a fresh prefix dir.
        let short_dir = PathBuf::from(&data_dir).join("book").join("ff");
        fs::create_dir_all(&short_dir).unwrap();
        fs::write(short_dir.join("tooshort"), b"data").unwrap();

        let found = scan_books(&data_dir);

        // Only the one valid CID should be returned.
        assert_eq!(found.len(), 1);
        assert_eq!(found[0], cid);
    }

    #[test]
    fn read_missing_file_returns_error() {
        let dir = tempfile::TempDir::new().unwrap();
        let cid = make_cid(b"this was never written");

        let err = read_book(&dir_str(&dir), &cid).expect_err("read of missing file should fail");
        assert_eq!(err.kind(), io::ErrorKind::NotFound);
    }

    #[test]
    fn scan_empty_directory_returns_empty() {
        let dir = tempfile::TempDir::new().unwrap();
        let found = scan_books(&dir_str(&dir));
        assert!(found.is_empty());
    }
}
