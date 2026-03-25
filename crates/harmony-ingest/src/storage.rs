//! Storage backends for shard and manifest book uploads.
//!
//! Supports S3 (via harmony-s3) and/or local directory with two-level
//! hex prefix layout for filesystem scalability.

use harmony_content::cid::ContentId;
use std::path::{Path, PathBuf};

/// Write a book to the local directory with two-level hex prefix.
///
/// Layout: `{base}/book/{hex[0..2]}/{hex_cid}`
pub fn write_local_book(base: &Path, cid: &ContentId, data: &[u8]) -> Result<(), String> {
    let hex_cid = hex::encode(cid.to_bytes());
    let prefix_dir = base.join("book").join(&hex_cid[..2]);
    std::fs::create_dir_all(&prefix_dir)
        .map_err(|e| format!("create dir {}: {e}", prefix_dir.display()))?;
    let file_path = prefix_dir.join(&hex_cid);
    std::fs::write(&file_path, data)
        .map_err(|e| format!("write {}: {e}", file_path.display()))?;
    Ok(())
}

/// Compute the expected local path for a book.
pub fn local_book_path(base: &Path, cid: &ContentId) -> PathBuf {
    let hex_cid = hex::encode(cid.to_bytes());
    base.join("book").join(&hex_cid[..2]).join(&hex_cid)
}

/// Upload a book to S3 with retry (exponential backoff).
///
/// Retries up to 3 times with 1s, 2s, 4s delays.
pub async fn upload_s3_book(
    s3: &harmony_s3::S3Library,
    cid: &ContentId,
    data: Vec<u8>,
) -> Result<(), String> {
    let cid_bytes = cid.to_bytes();
    let delays = [1, 2, 4];
    let mut last_err = String::new();

    for (attempt, delay_secs) in delays.iter().enumerate() {
        match s3.put_book(&cid_bytes, data.clone()).await {
            Ok(()) => return Ok(()),
            Err(e) => {
                last_err = format!("{e}");
                tracing::warn!(
                    attempt = attempt + 1,
                    delay_secs,
                    err = %e,
                    "S3 upload failed, retrying"
                );
                tokio::time::sleep(std::time::Duration::from_secs(*delay_secs)).await;
            }
        }
    }

    Err(format!("S3 upload failed after 3 attempts: {last_err}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use harmony_content::cid::ContentFlags;

    fn make_cid(data: &[u8]) -> ContentId {
        ContentId::for_book(data, ContentFlags::default()).unwrap()
    }

    #[test]
    fn local_book_path_two_level_prefix() {
        let cid = make_cid(b"test data");
        let path = local_book_path(Path::new("/mnt/usb"), &cid);
        let hex_cid = hex::encode(cid.to_bytes());
        let expected = format!("/mnt/usb/book/{}/{}", &hex_cid[..2], hex_cid);
        assert_eq!(path.to_str().unwrap(), expected);
    }

    #[test]
    fn write_and_read_local_book() {
        let dir = tempfile::tempdir().unwrap();
        let data = b"shard bytes here";
        let cid = make_cid(data);

        write_local_book(dir.path(), &cid, data).unwrap();

        let path = local_book_path(dir.path(), &cid);
        let read_back = std::fs::read(&path).unwrap();
        assert_eq!(read_back, data);
    }

    #[test]
    fn write_same_book_twice_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let data = b"same content";
        let cid = make_cid(data);

        write_local_book(dir.path(), &cid, data).unwrap();
        write_local_book(dir.path(), &cid, data).unwrap();

        let path = local_book_path(dir.path(), &cid);
        let read_back = std::fs::read(&path).unwrap();
        assert_eq!(read_back, data);
    }
}
