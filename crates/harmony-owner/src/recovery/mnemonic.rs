//! BIP39-24 mnemonic encode/decode for the 32-byte recovery seed.
//!
//! Uses the `bip39` crate (v2) with the English wordlist. The 32-byte seed
//! maps to exactly 24 BIP39 words: 256 bits payload + 8-bit checksum
//! (`SHA256(seed)[0]`) split into 24 × 11-bit groups. No PBKDF2 expansion
//! (the seed is the raw 32 bytes, not a BIP39-derived 64-byte expansion).

use crate::recovery::error::RecoveryError;

/// Encode a 32-byte seed as a 24-word BIP39 mnemonic (English wordlist).
pub(crate) fn to_mnemonic_inner(seed: &[u8; 32]) -> String {
    bip39::Mnemonic::from_entropy(seed)
        .expect("32 bytes is always valid BIP39-24 entropy")
        .to_string()
}

/// Parse a 24-word BIP39 mnemonic (English wordlist) into a 32-byte seed.
///
/// Leniency rules:
/// - Whitespace-tolerant: any run of whitespace collapses to a single space.
/// - Case-insensitive: lowercased before wordlist lookup.
/// - NO Unicode normalization: non-ASCII input is rejected outright.
///
/// Errors map explicitly to `RecoveryError::{NonAsciiInput, WrongWordCount,
/// UnknownWord, BadChecksum}`. We do not blanket-`From<bip39::Error>` so a
/// future bip39 crate update cannot widen our error contract.
pub(crate) fn from_mnemonic_inner(s: &str) -> Result<[u8; 32], RecoveryError> {
    if !s.is_ascii() {
        return Err(RecoveryError::NonAsciiInput);
    }
    // Whitespace normalize + lowercase.
    let normalized = s.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase();
    let words: Vec<&str> = normalized.split(' ').collect();
    if words.len() != 24 {
        return Err(RecoveryError::WrongWordCount(words.len()));
    }
    let mnemonic = bip39::Mnemonic::parse_in_normalized(
        bip39::Language::English,
        &normalized,
    )
    .map_err(map_bip39_err_with_words(&words))?;
    let entropy = mnemonic.to_entropy();
    debug_assert_eq!(entropy.len(), 32, "BIP39-24 always decodes to 32 bytes");
    let mut seed = [0u8; 32];
    seed.copy_from_slice(&entropy);
    Ok(seed)
}

fn map_bip39_err_with_words<'a>(words: &'a [&'a str]) -> impl FnOnce(bip39::Error) -> RecoveryError + 'a {
    move |e| match e {
        bip39::Error::UnknownWord(idx) => RecoveryError::UnknownWord {
            position: idx + 1, // 1-indexed for human display
            word: words.get(idx).map(|s| s.to_string()).unwrap_or_default(),
        },
        bip39::Error::InvalidChecksum => RecoveryError::BadChecksum,
        bip39::Error::BadWordCount(n) => RecoveryError::WrongWordCount(n),
        // Other variants (BadEntropyBitCount, AmbiguousLanguages, etc.) are
        // unreachable for our 24-word English-only path. Map to BadChecksum
        // as a conservative fallback rather than panicking.
        _ => RecoveryError::BadChecksum,
    }
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

#[cfg(test)]
mod decode_tests {
    use super::*;

    #[test]
    fn seed_to_mnemonic_roundtrips() {
        let mut seed = [0u8; 32];
        for (i, b) in seed.iter_mut().enumerate() {
            *b = (i * 7) as u8;
        }
        let m = to_mnemonic_inner(&seed);
        let restored = from_mnemonic_inner(&m).unwrap();
        assert_eq!(restored, seed);
    }

    #[test]
    fn case_insensitive_input_succeeds() {
        let m = to_mnemonic_inner(&[0u8; 32]);
        let upper = m.to_uppercase();
        assert!(from_mnemonic_inner(&upper).is_ok());
        let mixed: String = m.chars().enumerate()
            .map(|(i, c)| if i % 2 == 0 { c.to_ascii_uppercase() } else { c })
            .collect();
        assert!(from_mnemonic_inner(&mixed).is_ok());
    }

    #[test]
    fn whitespace_normalization() {
        let m = to_mnemonic_inner(&[0u8; 32]);
        // Collapse all spaces to triples-of-various-whitespace.
        let weird = m.replace(' ', "\t \n  ");
        assert!(from_mnemonic_inner(&weird).is_ok());
        // Leading and trailing whitespace
        let padded = format!("   \n{m}\t\t   ");
        assert!(from_mnemonic_inner(&padded).is_ok());
    }
}
