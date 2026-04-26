//! BIP39-24 mnemonic encode/decode for the 32-byte recovery seed.
//!
//! Uses the `bip39` crate (v2) with the English wordlist. The 32-byte seed
//! maps to exactly 24 BIP39 words: 256 bits payload + 8-bit checksum
//! (`SHA256(seed)[0]`) split into 24 × 11-bit groups. No PBKDF2 expansion
//! (the seed is the raw 32 bytes, not a BIP39-derived 64-byte expansion).

/// Encode a 32-byte seed as a 24-word BIP39 mnemonic (English wordlist).
pub(crate) fn to_mnemonic_inner(seed: &[u8; 32]) -> String {
    bip39::Mnemonic::from_entropy(seed)
        .expect("32 bytes is always valid BIP39-24 entropy")
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_produces_24_words() {
        let seed = [0u8; 32];
        let m = to_mnemonic_inner(&seed);
        let words: Vec<&str> = m.split_whitespace().collect();
        assert_eq!(words.len(), 24);
    }

    /// BIP39 canonical test vector: 32 bytes of zeroes maps to a known
    /// 24-word mnemonic starting with "abandon ... art". This locks
    /// interop with external BIP39 implementations.
    #[test]
    fn bip39_test_vector_all_zero_seed() {
        let seed = [0u8; 32];
        let m = to_mnemonic_inner(&seed);
        assert_eq!(
            m,
            "abandon abandon abandon abandon abandon abandon abandon abandon \
             abandon abandon abandon abandon abandon abandon abandon abandon \
             abandon abandon abandon abandon abandon abandon abandon art"
        );
    }
}
