//! Nix narinfo format — pure data transformation, no I/O.
//!
//! Handles narinfo text generation, parsing, fingerprint computation, and
//! Nix ed25519 signing. No disk I/O, no CAS, no network.

#![cfg(feature = "nix-cache")]

use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use ed25519_dalek::{Signer, SigningKey};

// ---------------------------------------------------------------------------
// Nix base-32 encoding
// ---------------------------------------------------------------------------

/// Nix-specific base-32 alphabet (omits e, o, t, u from lowercase ASCII).
const NIX_BASE32_CHARS: &[u8; 32] = b"0123456789abcdfghijklmnpqrsvwxyz";

/// Encode raw bytes using Nix's custom base-32 encoding.
///
/// This is NOT RFC 4648 base32. Nix processes bits from LSB to MSB in 5-bit
/// groups, then reverses the output. For a SHA-256 hash (32 bytes / 256 bits)
/// this produces 52 characters.
///
/// Reference: `nix/src/libutil/hash.cc` — `printHash32()`
fn nix_base32_encode(bytes: &[u8]) -> String {
    let bit_count = bytes.len() * 8;
    let len = (bit_count + 4) / 5; // ceil(bits / 5)
    let mut out = String::with_capacity(len);

    for n in (0..len).rev() {
        let b = n * 5;
        let i = b / 8;
        let j = b % 8;
        let mut c = (bytes[i] >> j) as u16;
        if j > 3 && i + 1 < bytes.len() {
            c |= (bytes[i + 1] as u16) << (8 - j);
        }
        out.push(NIX_BASE32_CHARS[(c & 0x1f) as usize] as char);
    }

    out
}

/// A Nix binary cache signing key loaded from the standard `<name>:<base64>` format.
///
/// The 64-byte payload is the ed25519 secret key in libsodium format:
/// first 32 bytes are the seed, next 32 bytes are the public key.
pub struct NixSigningKey {
    name: String,
    signing_key: SigningKey,
}

impl NixSigningKey {
    /// Parse a Nix signing key from the `<keyname>:<base64-of-64-bytes>` format.
    pub fn from_nix_format(contents: &str) -> Result<Self, String> {
        let colon_pos = contents
            .find(':')
            .ok_or_else(|| "signing key must contain ':' separator".to_string())?;

        let name = &contents[..colon_pos];
        let b64 = &contents[colon_pos + 1..];

        if name.is_empty() {
            return Err("signing key name must not be empty".to_string());
        }

        let bytes = BASE64
            .decode(b64.trim())
            .map_err(|e| format!("signing key base64 decode error: {e}"))?;

        if bytes.len() != 64 {
            return Err(format!(
                "signing key must be 64 bytes (seed + pubkey), got {}",
                bytes.len()
            ));
        }

        // The first 32 bytes are the seed.
        let seed: [u8; 32] = bytes[..32]
            .try_into()
            .map_err(|_| "seed slice length mismatch".to_string())?;

        let signing_key = SigningKey::from_bytes(&seed);

        Ok(Self {
            name: name.to_string(),
            signing_key,
        })
    }

    /// Sign a fingerprint string and return `<keyname>:<base64sig>`.
    pub fn sign_fingerprint(&self, fingerprint: &str) -> String {
        let sig = self.signing_key.sign(fingerprint.as_bytes());
        format!("{}:{}", self.name, BASE64.encode(sig.to_bytes()))
    }

    /// The key name (the part before the colon in the Nix key file).
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// All fields of a Nix narinfo file.
#[derive(Debug, Clone, PartialEq)]
pub struct NarInfo {
    pub store_path: String,
    pub url: String,
    /// `sha256:<nix-base32>` — hash of the NAR archive
    pub nar_hash: String,
    pub nar_size: u64,
    pub references: Vec<String>,
    pub sig: Option<String>,
}

impl NarInfo {
    /// Format a raw SHA-256 digest as a Nix hash string: `sha256:<nix-base32>`.
    ///
    /// Nix uses a custom base-32 encoding (not RFC 4648 and not standard base64)
    /// for hashes in narinfo files and fingerprints. A 32-byte SHA-256 digest
    /// produces a 52-character nix-base32 string.
    pub fn format_nix_hash(sha256: &[u8; 32]) -> String {
        format!("sha256:{}", nix_base32_encode(sha256))
    }

    /// Compute the Nix fingerprint that gets signed:
    /// `1;<StorePath>;<NarHash>;<NarSize>;<space-separated-sorted-references>`
    pub fn fingerprint(&self) -> String {
        let mut sorted_refs = self.references.clone();
        sorted_refs.sort();
        format!(
            "1;{};{};{};{}",
            self.store_path,
            self.nar_hash,
            self.nar_size,
            sorted_refs.join(" ")
        )
    }

    /// Sign this narinfo with the given key and store the resulting signature.
    pub fn sign(&mut self, key: &NixSigningKey) {
        self.sig = Some(key.sign_fingerprint(&self.fingerprint()));
    }

    /// Render the narinfo as HTTP response text.
    pub fn to_text(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!("StorePath: {}", self.store_path));
        lines.push(format!("URL: {}", self.url));
        lines.push("Compression: none".to_string());
        // FileHash and FileSize equal NarHash/NarSize for uncompressed NARs.
        lines.push(format!("FileHash: {}", self.nar_hash));
        lines.push(format!("FileSize: {}", self.nar_size));
        lines.push(format!("NarHash: {}", self.nar_hash));
        lines.push(format!("NarSize: {}", self.nar_size));

        if !self.references.is_empty() {
            let mut sorted = self.references.clone();
            sorted.sort();
            lines.push(format!("References: {}", sorted.join(" ")));
        }

        if let Some(sig) = &self.sig {
            lines.push(format!("Sig: {}", sig));
        }

        // Nix expects a trailing newline.
        let mut text = lines.join("\n");
        text.push('\n');
        text
    }

    /// Parse a narinfo text back into a `NarInfo` struct (used in tests and verification).
    pub fn from_text(text: &str) -> Result<Self, String> {
        let mut store_path: Option<String> = None;
        let mut url: Option<String> = None;
        let mut nar_hash: Option<String> = None;
        let mut nar_size: Option<u64> = None;
        let mut references: Vec<String> = Vec::new();
        let mut sig: Option<String> = None;

        for raw_line in text.lines() {
            let line = raw_line.trim();
            if line.is_empty() {
                continue;
            }
            let (key, value) = line
                .split_once(": ")
                .ok_or_else(|| format!("malformed narinfo line: {line:?}"))?;
            match key {
                "StorePath" => store_path = Some(value.to_string()),
                "URL" => url = Some(value.to_string()),
                "NarHash" => nar_hash = Some(value.to_string()),
                "NarSize" => {
                    nar_size = Some(
                        value
                            .parse()
                            .map_err(|e| format!("invalid NarSize: {e}"))?,
                    )
                }
                "References" => {
                    if !value.is_empty() {
                        references = value.split_whitespace().map(str::to_string).collect();
                    }
                }
                "Sig" => sig = Some(value.to_string()),
                // Ignore Compression, FileHash, FileSize — derived from NarHash/NarSize.
                _ => {}
            }
        }

        Ok(NarInfo {
            store_path: store_path.ok_or("missing StorePath")?,
            url: url.ok_or("missing URL")?,
            nar_hash: nar_hash.ok_or("missing NarHash")?,
            nar_size: nar_size.ok_or("missing NarSize")?,
            references,
            sig,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_signing_key() -> NixSigningKey {
        let seed = [0u8; 32];
        let key = SigningKey::from_bytes(&seed);
        let pubkey = key.verifying_key().to_bytes();
        let mut combined = Vec::with_capacity(64);
        combined.extend_from_slice(&seed);
        combined.extend_from_slice(&pubkey);
        let nix_format = format!("test-key-1:{}", BASE64.encode(&combined));
        NixSigningKey::from_nix_format(&nix_format).unwrap()
    }

    fn sample_narinfo() -> NarInfo {
        // Use a real nix-base32 hash (all-zero SHA-256 → 52 nix-base32 chars)
        let zero_hash = NarInfo::format_nix_hash(&[0u8; 32]);
        NarInfo {
            store_path: "/nix/store/aaaabbbbccccdddd0000111122223333-test-package-1.0".to_string(),
            url: "nar/aabbccdd.nar".to_string(),
            nar_hash: zero_hash,
            nar_size: 12345,
            references: vec![
                "zzzzwwwwxxxxyyyyaaaa0000111122223333-dep1".to_string(),
                "aaaabbbbccccdddd0000111122223333-dep0".to_string(),
            ],
            sig: None,
        }
    }

    #[test]
    fn signing_key_from_nix_format() {
        let key = test_signing_key();
        assert_eq!(key.name(), "test-key-1");
    }

    #[test]
    fn signing_key_rejects_empty_name() {
        let seed = [0u8; 32];
        let sk = SigningKey::from_bytes(&seed);
        let pubkey = sk.verifying_key().to_bytes();
        let mut combined = Vec::with_capacity(64);
        combined.extend_from_slice(&seed);
        combined.extend_from_slice(&pubkey);
        let nix_format = format!(":{}", BASE64.encode(&combined));
        let err = NixSigningKey::from_nix_format(&nix_format)
            .err()
            .expect("should reject empty key name");
        assert!(err.contains("empty"), "error should mention 'empty': {err}");
    }

    #[test]
    fn signing_key_rejects_missing_colon() {
        assert!(
            NixSigningKey::from_nix_format("nokeyhere").is_err(),
            "should reject input without ':'"
        );
    }

    #[test]
    fn signing_key_rejects_wrong_length() {
        // Only 32 bytes instead of required 64.
        let short = BASE64.encode([0u8; 32]);
        let nix_format = format!("mykey:{short}");
        let err = NixSigningKey::from_nix_format(&nix_format)
            .err()
            .expect("should reject wrong-length key");
        assert!(err.contains("64"), "error should mention 64 bytes: {err}");
    }

    #[test]
    fn fingerprint_format() {
        let ni = sample_narinfo();
        let fp = ni.fingerprint();
        // References must be sorted.
        assert!(
            fp.starts_with("1;/nix/store/"),
            "fingerprint must start with '1;'"
        );
        // Sorted: dep0 comes before dep1.
        let refs_part = fp.split(';').nth(4).expect("fingerprint has 5 semicolon parts");
        let refs: Vec<&str> = refs_part.split_whitespace().collect();
        assert_eq!(refs.len(), 2);
        assert!(
            refs[0] < refs[1],
            "references must be sorted in fingerprint"
        );
        assert!(fp.contains(&ni.nar_hash));
        assert!(fp.contains(&ni.nar_size.to_string()));
    }

    #[test]
    fn sign_produces_keyname_colon_base64() {
        let key = test_signing_key();
        let mut ni = sample_narinfo();
        ni.sign(&key);

        let sig = ni.sig.as_ref().expect("sig should be set after sign()");
        let (name_part, b64_part) = sig.split_once(':').expect("sig must contain ':'");
        assert_eq!(name_part, "test-key-1");

        let sig_bytes = BASE64
            .decode(b64_part)
            .expect("sig base64 must be valid");
        assert_eq!(sig_bytes.len(), 64, "ed25519 signature must be 64 bytes");
    }

    #[test]
    fn narinfo_text_round_trip() {
        let key = test_signing_key();
        let mut original = sample_narinfo();
        original.sign(&key);

        let text = original.to_text();
        let parsed = NarInfo::from_text(&text).expect("round-trip parse must succeed");

        assert_eq!(parsed.store_path, original.store_path);
        assert_eq!(parsed.url, original.url);
        assert_eq!(parsed.nar_hash, original.nar_hash);
        assert_eq!(parsed.nar_size, original.nar_size);
        assert_eq!(parsed.sig, original.sig);

        // References may come back in a different order; compare sorted.
        let mut orig_refs = original.references.clone();
        let mut parsed_refs = parsed.references.clone();
        orig_refs.sort();
        parsed_refs.sort();
        assert_eq!(orig_refs, parsed_refs);
    }

    #[test]
    fn format_nix_hash() {
        let hash = [0u8; 32];
        let result = NarInfo::format_nix_hash(&hash);
        assert!(result.starts_with("sha256:"), "must start with 'sha256:'");
        // SHA-256 (32 bytes / 256 bits) → 52 nix-base32 chars
        let nix32_part = result.strip_prefix("sha256:").unwrap();
        assert_eq!(nix32_part.len(), 52, "nix-base32 of 32 bytes should be 52 chars");
        // All-zero hash should produce all-zero nix-base32 (all '0' chars)
        assert!(
            nix32_part.chars().all(|c| c == '0'),
            "all-zero hash should encode to all-zero nix-base32: {nix32_part}"
        );
    }

    #[test]
    fn nix_base32_properties() {
        // Verify encoding properties: correct length, alphabet, determinism
        let hash = [0xABu8; 32];
        let encoded = super::nix_base32_encode(&hash);

        // SHA-256 (32 bytes / 256 bits) → ceil(256/5) = 52 chars
        assert_eq!(encoded.len(), 52, "nix-base32 of 32 bytes should be 52 chars");

        // All chars must be in the nix-base32 alphabet (no e, o, t, u)
        let alphabet = "0123456789abcdfghijklmnpqrsvwxyz";
        for ch in encoded.chars() {
            assert!(
                alphabet.contains(ch),
                "char '{ch}' not in nix-base32 alphabet"
            );
        }

        // Deterministic: same input → same output
        assert_eq!(encoded, super::nix_base32_encode(&hash));

        // Different input → different output
        let hash2 = [0xCDu8; 32];
        assert_ne!(encoded, super::nix_base32_encode(&hash2));
    }

    #[test]
    fn nix_base32_small_known_vectors() {
        // Single byte: 0xFF = 0b11111111
        // n=1: b=5, i=0, j=5 → 0xFF >> 5 = 7 → '7'
        // n=0: b=0, i=0, j=0 → 0xFF & 0x1f = 31 → 'z'
        assert_eq!(super::nix_base32_encode(&[0xFF]), "7z");

        // Single byte: 0x00 → "00"
        assert_eq!(super::nix_base32_encode(&[0x00]), "00");

        // Two bytes: 0x00 0x00 → "0000" (ceil(16/5)=4)
        assert_eq!(super::nix_base32_encode(&[0x00, 0x00]), "0000");
    }

    #[test]
    fn narinfo_text_contains_required_fields() {
        let ni = sample_narinfo();
        let text = ni.to_text();
        assert!(text.contains("StorePath:"), "must contain StorePath");
        assert!(text.contains("URL:"), "must contain URL");
        assert!(text.contains("Compression: none"), "must contain Compression");
        assert!(text.contains("FileHash:"), "must contain FileHash");
        assert!(text.contains("FileSize:"), "must contain FileSize");
        assert!(text.contains("NarHash:"), "must contain NarHash");
        assert!(text.contains("NarSize:"), "must contain NarSize");
    }

    #[test]
    fn empty_references_omitted() {
        let mut ni = sample_narinfo();
        ni.references = Vec::new();
        let text = ni.to_text();
        assert!(
            !text.contains("References:"),
            "References line must be omitted when refs is empty"
        );
    }
}
