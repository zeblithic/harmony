//! Zenoh-driven archivist that durably archives content-addressed books to S3.
//!
//! The archivist subscribes to `harmony/content/publish/*` and uploads each
//! durable (non-ephemeral) book that is not already present in the S3 library.
//! Ephemeral content (ContentFlag bit 6 set) is silently skipped.

use crate::S3Library;

/// Checks whether a content identifier refers to durable content.
///
/// ContentFlags occupy the top nibble of header byte 0:
/// - bit 7 (0x80) = encrypted
/// - bit 6 (0x40) = **ephemeral** ← this flag governs durability
/// - bit 5 (0x20) = sha224
/// - bit 4 (0x10) = lsb_mode
///
/// Returns `true` when bit 6 is **clear** (content is durable).
pub fn is_durable(cid_bytes: &[u8; 32]) -> bool {
    (cid_bytes[0] & 0x40) == 0
}

/// Extracts the 64-character hex CID from the last segment of a Zenoh key
/// expression such as `harmony/content/publish/{cid_hex}`.
///
/// Returns `None` if the last segment is not exactly 64 characters.
pub fn extract_cid_hex(key_expr: &str) -> Option<&str> {
    let segment = key_expr.rsplit('/').next()?;
    if segment.len() == 64 {
        Some(segment)
    } else {
        None
    }
}

/// Hex-decodes a 64-character string into a 32-byte content identifier.
///
/// Returns `None` if `hex_str` is not valid lowercase/uppercase hexadecimal
/// or is not exactly 64 characters long.
pub fn parse_cid(hex_str: &str) -> Option<[u8; 32]> {
    if hex_str.len() != 64 {
        return None;
    }
    let mut bytes = [0u8; 32];
    hex::decode_to_slice(hex_str, &mut bytes).ok()?;
    Some(bytes)
}

/// Run the archivist loop.
///
/// Subscribes to `harmony/content/publish/*` on the provided Zenoh session
/// and uploads every durable, not-yet-archived book to `s3`.
///
/// The function returns when the Zenoh subscriber is closed (typically on
/// session shutdown).
pub async fn run(s3: S3Library, session: zenoh::Session) {
    const KEY_PATTERN: &str = "harmony/content/publish/*";

    let subscriber = match session.declare_subscriber(KEY_PATTERN).await {
        Ok(s) => s,
        Err(e) => {
            tracing::error!(err = %e, "archivist: failed to declare subscriber");
            return;
        }
    };

    tracing::info!(key_pattern = KEY_PATTERN, "archivist: subscriber active");

    while let Ok(sample) = subscriber.recv_async().await {
        let key_expr = sample.key_expr().to_string();

        // Extract CID hex from key expression.
        let hex = match extract_cid_hex(&key_expr) {
            Some(h) => h,
            None => {
                tracing::warn!(key_expr, "archivist: could not extract CID hex from key");
                continue;
            }
        };

        // Parse hex into 32-byte CID.
        let cid = match parse_cid(hex) {
            Some(c) => c,
            None => {
                tracing::warn!(key_expr, hex, "archivist: invalid CID hex");
                continue;
            }
        };

        // Skip ephemeral content.
        if !is_durable(&cid) {
            tracing::debug!(key_expr, "archivist: skipping ephemeral content");
            continue;
        }

        // Skip already-archived content (HEAD request).
        match s3.exists(&cid).await {
            Ok(true) => {
                tracing::debug!(key_expr, "archivist: already archived, skipping");
                continue;
            }
            Ok(false) => {}
            Err(e) => {
                tracing::warn!(key_expr, err = %e, "archivist: exists check failed, skipping");
                continue;
            }
        }

        // Upload the book.
        // TODO(harmony-aa7): verify BLAKE3(payload) matches the hash portion of the CID
        // before archiving. Currently trusts the publisher — verification is deferred to
        // the fallback resolver which hashes on download.
        let payload = sample.payload().to_bytes().to_vec();
        match s3.put_book(&cid, payload).await {
            Ok(()) => {
                tracing::info!(key_expr, "archivist: archived book");
            }
            Err(e) => {
                tracing::warn!(key_expr, err = %e, "archivist: put_book failed");
            }
        }
    }

    tracing::info!("archivist: subscriber closed, exiting");
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── is_durable ─────────────────────────────────────────────────────────────

    #[test]
    fn is_durable_public_durable() {
        // No flags set — public, durable content.
        let mut cid = [0u8; 32];
        cid[0] = 0x00;
        assert!(is_durable(&cid), "header[0]=0x00 must be durable");
    }

    #[test]
    fn is_durable_encrypted_durable() {
        // Encrypted (bit 7) but NOT ephemeral (bit 6 clear) → durable.
        let mut cid = [0u8; 32];
        cid[0] = 0x80;
        assert!(is_durable(&cid), "header[0]=0x80 (encrypted only) must be durable");
    }

    #[test]
    fn is_not_durable_ephemeral() {
        // Ephemeral flag (bit 6) set, not encrypted.
        let mut cid = [0u8; 32];
        cid[0] = 0x40;
        assert!(!is_durable(&cid), "header[0]=0x40 (ephemeral) must NOT be durable");
    }

    #[test]
    fn is_not_durable_encrypted_ephemeral() {
        // Both encrypted and ephemeral.
        let mut cid = [0u8; 32];
        cid[0] = 0xC0;
        assert!(
            !is_durable(&cid),
            "header[0]=0xC0 (encrypted+ephemeral) must NOT be durable"
        );
    }

    // ── extract_cid_hex ────────────────────────────────────────────────────────

    #[test]
    fn extract_cid_hex_valid() {
        let cid_hex = "a".repeat(64);
        let key_expr = format!("harmony/content/publish/{cid_hex}");
        let result = extract_cid_hex(&key_expr);
        assert_eq!(result, Some(cid_hex.as_str()));
    }

    #[test]
    fn extract_cid_hex_wrong_length() {
        // 63-char suffix — one char short of a valid CID hex.
        let short = "b".repeat(63);
        let key_expr = format!("harmony/content/publish/{short}");
        assert_eq!(extract_cid_hex(&key_expr), None);
    }

    // ── parse_cid ──────────────────────────────────────────────────────────────

    #[test]
    fn parse_cid_valid() {
        // Round-trip: encode 32 bytes to hex, then parse back.
        let original: [u8; 32] = core::array::from_fn(|i| i as u8);
        let hex_str = hex::encode(original);
        let result = parse_cid(&hex_str);
        assert_eq!(result, Some(original));
    }

    #[test]
    fn parse_cid_invalid_hex() {
        // 64 chars but contains non-hex characters.
        let bad = "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz";
        assert_eq!(parse_cid(bad), None);
    }
}
