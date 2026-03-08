use sha2::{Digest, Sha224, Sha256};

/// Full SHA-256 hash length in bytes.
pub const HASH_LENGTH: usize = 32;

/// Truncated hash length in bytes (Reticulum address hash).
pub const TRUNCATED_HASH_LENGTH: usize = 16;

/// Compute the full SHA-256 hash of `data` (32 bytes).
pub fn full_hash(data: &[u8]) -> [u8; HASH_LENGTH] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().into()
}

/// Compute a truncated SHA-256 hash of `data` (first 16 bytes).
///
/// This matches Reticulum's address derivation: `SHA256(data)[:16]`.
pub fn truncated_hash(data: &[u8]) -> [u8; TRUNCATED_HASH_LENGTH] {
    let full = full_hash(data);
    let mut truncated = [0u8; TRUNCATED_HASH_LENGTH];
    truncated.copy_from_slice(&full[..TRUNCATED_HASH_LENGTH]);
    truncated
}

/// Compute a BLAKE3 hash of `data` (32 bytes).
pub fn blake3_hash(data: &[u8]) -> [u8; 32] {
    blake3::hash(data).into()
}

/// Name hash length in bytes (Reticulum).
pub const NAME_HASH_LENGTH: usize = 10;

/// Compute the Reticulum name hash: `SHA256(data)[:10]`.
pub fn name_hash(data: &[u8]) -> [u8; NAME_HASH_LENGTH] {
    let full = full_hash(data);
    let mut result = [0u8; NAME_HASH_LENGTH];
    result.copy_from_slice(&full[..NAME_HASH_LENGTH]);
    result
}

/// SHA-224 hash length in bytes.
pub const SHA224_HASH_LENGTH: usize = 28;

/// Compute the SHA-224 hash of `data` (28 bytes).
///
/// SHA-224 uses different initialization vectors from SHA-256, making them
/// independent hash families. Used by ContentId for power-of-choice collision
/// resistance (the `alt_hash` flag).
pub fn sha224_hash(data: &[u8]) -> [u8; SHA224_HASH_LENGTH] {
    let mut hasher = Sha224::new();
    hasher.update(data);
    hasher.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_known_vector_empty() {
        // SHA-256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
        let hash = full_hash(b"");
        assert_eq!(
            hex::encode(hash),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn sha256_known_vector_abc() {
        // SHA-256("abc") = ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
        let hash = full_hash(b"abc");
        assert_eq!(
            hex::encode(hash),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn truncated_hash_is_first_16_bytes() {
        let full = full_hash(b"abc");
        let truncated = truncated_hash(b"abc");
        assert_eq!(truncated.len(), 16);
        assert_eq!(&full[..16], &truncated);
    }

    #[test]
    fn name_hash_is_first_10_bytes() {
        let full = full_hash(b"test_app_name");
        let nh = name_hash(b"test_app_name");
        assert_eq!(nh.len(), 10);
        assert_eq!(&full[..10], &nh);
    }

    #[test]
    fn blake3_known_vector_empty() {
        // BLAKE3("") known hash
        let hash = blake3_hash(b"");
        let expected = blake3::hash(b"");
        assert_eq!(hash, <[u8; 32]>::from(expected));
    }

    #[test]
    fn blake3_deterministic() {
        let h1 = blake3_hash(b"hello harmony");
        let h2 = blake3_hash(b"hello harmony");
        assert_eq!(h1, h2);
    }

    #[test]
    fn different_inputs_produce_different_hashes() {
        let h1 = full_hash(b"alice");
        let h2 = full_hash(b"bob");
        assert_ne!(h1, h2);

        let t1 = truncated_hash(b"alice");
        let t2 = truncated_hash(b"bob");
        assert_ne!(t1, t2);
    }

    #[test]
    fn sha224_known_vector_empty() {
        // SHA-224("") = d14a028c2a3a2bc9476102bb288234c415a2b01f828ea62ac5b3e42f
        let hash = sha224_hash(b"");
        assert_eq!(
            hex::encode(hash),
            "d14a028c2a3a2bc9476102bb288234c415a2b01f828ea62ac5b3e42f"
        );
    }

    #[test]
    fn sha224_known_vector_abc() {
        // SHA-224("abc") = 23097d223405d8228642a477bda255b32aadbce4bda0b3f7e36c9da7
        let hash = sha224_hash(b"abc");
        assert_eq!(
            hex::encode(hash),
            "23097d223405d8228642a477bda255b32aadbce4bda0b3f7e36c9da7"
        );
    }

    #[test]
    fn sha224_deterministic() {
        let h1 = sha224_hash(b"hello harmony");
        let h2 = sha224_hash(b"hello harmony");
        assert_eq!(h1, h2);
    }

    #[test]
    fn sha224_differs_from_sha256() {
        let s256 = full_hash(b"test");
        let s224 = sha224_hash(b"test");
        assert_ne!(&s256[..28], &s224[..]);
    }
}
