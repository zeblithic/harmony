//! Binary CID journal for crash-safe resume.
//!
//! Each entry is a raw 32-byte ContentId, appended after each shard upload.
//! On resume, the journal is read to recover previously collected CIDs.

use harmony_content::cid::ContentId;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// Append-only binary journal of 32-byte CID entries.
pub struct CidJournal {
    #[allow(dead_code)] // used by entry_count (test-only API)
    path: PathBuf,
    file: File,
}

impl CidJournal {
    /// Create a fresh journal, truncating any existing file.
    ///
    /// Use this for fresh runs (not resuming). Any prior journal content
    /// is discarded.
    pub fn create(path: &Path) -> Result<Self, String> {
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)
            .map_err(|e| format!("journal create: {e}"))?;

        Ok(Self {
            path: path.to_path_buf(),
            file,
        })
    }

    /// Open an existing journal for appending.  Truncates any partial last entry.
    ///
    /// Use this when resuming (`--resume-from`). Preserves existing entries.
    pub fn open(path: &Path) -> Result<Self, String> {
        if path.exists() {
            let len = std::fs::metadata(path)
                .map_err(|e| format!("journal metadata: {e}"))?
                .len();
            let aligned = len - (len % 32);
            if aligned < len {
                let file = OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(path)
                    .map_err(|e| format!("journal open for truncate: {e}"))?;
                file.set_len(aligned)
                    .map_err(|e| format!("journal truncate: {e}"))?;
            }
        }

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| format!("journal open: {e}"))?;

        Ok(Self {
            path: path.to_path_buf(),
            file,
        })
    }

    /// Append a CID to the journal.
    pub fn append(&mut self, cid: &ContentId) -> Result<(), String> {
        self.file
            .write_all(&cid.to_bytes())
            .map_err(|e| format!("journal write: {e}"))?;
        self.file
            .sync_data()
            .map_err(|e| format!("journal sync: {e}"))?;
        Ok(())
    }

    /// Read all CIDs from the journal.
    pub fn read_all(path: &Path) -> Result<Vec<[u8; 32]>, String> {
        if !path.exists() {
            return Ok(Vec::new());
        }
        let mut data = Vec::new();
        File::open(path)
            .map_err(|e| format!("journal read: {e}"))?
            .read_to_end(&mut data)
            .map_err(|e| format!("journal read: {e}"))?;

        let aligned = data.len() - (data.len() % 32);
        let count = aligned / 32;
        let mut cids = Vec::with_capacity(count);
        for chunk in data[..aligned].chunks_exact(32) {
            let mut cid = [0u8; 32];
            cid.copy_from_slice(chunk);
            cids.push(cid);
        }
        Ok(cids)
    }

    /// Number of entries written so far (based on file size).
    #[allow(dead_code)]
    pub fn entry_count(&self) -> Result<u64, String> {
        let len = std::fs::metadata(&self.path)
            .map_err(|e| format!("journal metadata: {e}"))?
            .len();
        Ok(len / 32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use harmony_content::cid::ContentFlags;

    fn make_cid(data: &[u8]) -> ContentId {
        ContentId::for_book(data, ContentFlags::default()).unwrap()
    }

    #[test]
    fn write_and_read_back() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.journal");

        let cid1 = make_cid(b"shard one");
        let cid2 = make_cid(b"shard two");

        {
            let mut journal = CidJournal::open(&path).unwrap();
            journal.append(&cid1).unwrap();
            journal.append(&cid2).unwrap();
            assert_eq!(journal.entry_count().unwrap(), 2);
        }

        let cids = CidJournal::read_all(&path).unwrap();
        assert_eq!(cids.len(), 2);
        assert_eq!(cids[0], cid1.to_bytes());
        assert_eq!(cids[1], cid2.to_bytes());
    }

    #[test]
    fn truncates_partial_entry() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.journal");

        let cid = make_cid(b"complete");
        {
            let mut f = File::create(&path).unwrap();
            f.write_all(&cid.to_bytes()).unwrap();
            f.write_all(&[0xFF; 10]).unwrap();
        }

        let _journal = CidJournal::open(&path).unwrap();

        let cids = CidJournal::read_all(&path).unwrap();
        assert_eq!(cids.len(), 1);
        assert_eq!(cids[0], cid.to_bytes());
    }

    #[test]
    fn empty_journal() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nonexistent.journal");
        let cids = CidJournal::read_all(&path).unwrap();
        assert!(cids.is_empty());
    }

    #[test]
    fn reopen_and_append() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.journal");

        let cid1 = make_cid(b"first");
        let cid2 = make_cid(b"second");

        {
            let mut journal = CidJournal::open(&path).unwrap();
            journal.append(&cid1).unwrap();
        }
        {
            let mut journal = CidJournal::open(&path).unwrap();
            journal.append(&cid2).unwrap();
        }

        let cids = CidJournal::read_all(&path).unwrap();
        assert_eq!(cids.len(), 2);
    }
}
