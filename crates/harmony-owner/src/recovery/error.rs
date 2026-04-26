//! `RecoveryError` enum — operator-readable failure modes for mnemonic and
//! encrypted-file encode/decode paths. Sibling to `OwnerError` (NOT folded
//! into it; recovery is a self-contained concern).

#[derive(Debug, thiserror::Error)]
pub enum RecoveryError {
    // ── Mnemonic parse ─────────────────────────────────────────────
    #[error("expected 24 BIP39 words, got {0}")]
    WrongWordCount(usize),

    #[error("unknown word at position {position}: {word:?}")]
    UnknownWord { position: usize, word: String },

    #[error("mnemonic checksum mismatch — likely a typo somewhere in the 24 words")]
    BadChecksum,

    #[error("mnemonic contains non-ASCII characters; BIP39 English wordlist is ASCII-only")]
    NonAsciiInput,

    // ── Encrypted-file decode ──────────────────────────────────────
    #[error("recovery file is too small ({0} bytes; minimum 69)")]
    TooSmall(usize),

    #[error("recovery file is too large ({0} bytes; maximum 1024)")]
    TooLarge(usize),

    #[error("not a harmony recovery file (magic mismatch)")]
    UnrecognizedFormat,

    #[error("recovery file format version {0:#x} is not supported by this build")]
    UnsupportedVersion(u8),

    #[error("recovery file uses unsupported KDF id {0:#x}")]
    UnsupportedKdfId(u8),

    #[error("recovery file KDF id {id:#x} present but parameters are non-standard")]
    UnsupportedKdfParams { id: u8 },

    #[error("wrong passphrase or corrupted recovery file (AEAD tag rejected)")]
    WrongPassphraseOrCorrupt,

    #[error("recovery file payload could not be decoded: {0}")]
    PayloadDecodeFailed(String),

    #[error("recovery file payload has unexpected format string {found:?}; expected {expected:?}")]
    UnexpectedPayloadFormat { found: String, expected: &'static str },

    // ── Encrypted-file encode ──────────────────────────────────────
    #[error("comment is {actual} bytes; max allowed is {max}")]
    CommentTooLong { actual: usize, max: usize },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_messages_render_expected_text() {
        assert_eq!(
            RecoveryError::WrongWordCount(13).to_string(),
            "expected 24 BIP39 words, got 13"
        );
        assert_eq!(
            RecoveryError::UnknownWord { position: 7, word: "harmonny".into() }.to_string(),
            r#"unknown word at position 7: "harmonny""#
        );
        assert_eq!(
            RecoveryError::BadChecksum.to_string(),
            "mnemonic checksum mismatch — likely a typo somewhere in the 24 words"
        );
        assert_eq!(
            RecoveryError::TooSmall(50).to_string(),
            "recovery file is too small (50 bytes; minimum 69)"
        );
        assert_eq!(
            RecoveryError::UnsupportedVersion(0x02).to_string(),
            "recovery file format version 0x2 is not supported by this build"
        );
        assert_eq!(
            RecoveryError::UnsupportedKdfId(0x99).to_string(),
            "recovery file uses unsupported KDF id 0x99"
        );
        assert_eq!(
            RecoveryError::UnsupportedKdfParams { id: 0x01 }.to_string(),
            "recovery file KDF id 0x1 present but parameters are non-standard"
        );
        assert_eq!(
            RecoveryError::UnexpectedPayloadFormat {
                found: "harmony-foo-v9".into(),
                expected: "harmony-owner-recovery-v1",
            }.to_string(),
            r#"recovery file payload has unexpected format string "harmony-foo-v9"; expected "harmony-owner-recovery-v1""#
        );
        assert_eq!(
            RecoveryError::CommentTooLong { actual: 300, max: 256 }.to_string(),
            "comment is 300 bytes; max allowed is 256"
        );
    }
}
