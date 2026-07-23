//! Ed25519 ↔ X25519 key conversions (RFC 7748 §5 birational map).
//!
//! A single Ed25519 keypair yields a matching X25519 (Diffie-Hellman) keypair
//! by mapping the Edwards curve point/scalar onto the birationally-equivalent
//! Montgomery curve. Harmony uses this so a device's one signing identity can
//! also receive [`crate::sealed_box`] traffic without holding a second stored
//! secret — the X25519 private key is derived on demand and never persisted.
//!
//! ```text
//! public:  x25519_pub  = montgomery_u( decompress_edwards(ed25519_verify_key) )
//! private: x25519_priv = clamp( sha512(seed)[..32] )   // = SigningKey::to_scalar_bytes, clamped
//! ```
//!
//! # Warning
//!
//! These conversions are a **frozen wire contract**: X25519 keys derived here
//! address every sealed blob in the network. Changing the derivation orphans
//! existing ciphertext. The public side and the clamped-scalar private side
//! MUST agree — [`crate::sealed_box::open`] with the private key must recover
//! traffic sealed to the public key. That parity is pinned by a known-answer
//! test below and re-asserted by harmony-owner and harmony-client.

use curve25519_dalek::edwards::CompressedEdwardsY;
use ed25519_dalek::SigningKey;
use zeroize::Zeroizing;

/// Convert an Ed25519 verify key to its birational X25519 public key
/// (RFC 7748 §5). Returns `None` when the bytes fail Edwards decompression
/// (off-curve) or decode to a small-order (torsion) point.
///
/// NOT a canonicality gate: curve25519-dalek's decompression reduces the `y`
/// field element mod p, so some non-canonical encodings decompress and return
/// `Some` — never treat `Some(_)` as proof of a canonical Ed25519 encoding.
/// Callers constructing keys from freshly generated material may `expect()`;
/// callers handling external bytes MUST propagate the `None`.
pub fn ed25519_pub_to_x25519(ed25519_pub: &[u8; 32]) -> Option<[u8; 32]> {
    let edwards = CompressedEdwardsY(*ed25519_pub).decompress()?;
    if edwards.is_small_order() {
        return None;
    }
    Some(edwards.to_montgomery().to_bytes())
}

/// Derive the X25519 private (Diffie-Hellman) key matching an Ed25519 signing
/// key (RFC 7748 §5), returning it zeroize-on-drop.
///
/// This is `clamp(SigningKey::to_scalar_bytes())`: `to_scalar_bytes` is the low
/// 32 bytes of `SHA-512(seed)` (the same expansion Ed25519 signing uses), and
/// the RFC 7748 clamp (`&= 248` on byte 0, `&= 127 | 64` on byte 31) makes it a
/// valid X25519 scalar. Its X25519 public key equals
/// [`ed25519_pub_to_x25519`] of the signing key's verify key.
pub fn ed25519_priv_to_x25519(signing_key: &SigningKey) -> Zeroizing<[u8; 32]> {
    let mut x_priv = Zeroizing::new(signing_key.to_scalar_bytes());
    x_priv[0] &= 248;
    x_priv[31] &= 127;
    x_priv[31] |= 64;
    x_priv
}

#[cfg(test)]
mod tests {
    use super::*;

    /// THE load-bearing parity test: the pub-side birational map must equal the
    /// public key of the priv-side clamped scalar (what [`crate::sealed_box`]
    /// decrypts with). If these diverge, sealed traffic is undecryptable.
    #[test]
    fn pub_conversion_matches_priv_derived_public() {
        for seed in [[0x42u8; 32], [0x01u8; 32], [0xEEu8; 32]] {
            let sk = SigningKey::from_bytes(&seed);
            let via_pub =
                ed25519_pub_to_x25519(&sk.verifying_key().to_bytes()).expect("valid key converts");
            // The clamped private scalar times the X25519 base point must give
            // the same public key the birational pub map produces.
            let x_priv = ed25519_priv_to_x25519(&sk);
            let via_priv =
                curve25519_dalek::montgomery::MontgomeryPoint::mul_base_clamped(*x_priv).to_bytes();
            assert_eq!(
                via_pub, via_priv,
                "birational pub/priv parity (seed {:02x})",
                seed[0]
            );
        }
    }

    /// Reference vector — pins the public-side derivation scheme. DO NOT
    /// regenerate casually: changing this derivation orphans every sealed blob
    /// addressed to existing keys. Also asserted by harmony-owner's
    /// `x25519::tests::reference_vector_seed_42` and harmony-client's
    /// `dm_signing` parity test — the three MUST stay identical.
    #[test]
    fn pub_reference_vector_seed_42() {
        let sk = SigningKey::from_bytes(&[0x42u8; 32]);
        let x = ed25519_pub_to_x25519(&sk.verifying_key().to_bytes()).unwrap();
        const EXPECTED: &str = "cc4f2cdb695dd766f34118eb67b98652fed1d8bc49c330b119bbfa8a64989378";
        assert_eq!(hex::encode(x), EXPECTED, "ZEB-372 frozen derivation vector");
    }

    /// Private-side reference vector — pins the clamped-scalar derivation.
    /// Frozen alongside the public vector above.
    #[test]
    fn priv_reference_vector_seed_42() {
        let sk = SigningKey::from_bytes(&[0x42u8; 32]);
        let x_priv = ed25519_priv_to_x25519(&sk);
        const EXPECTED: &str = "90e7595fc89e52fdfddce9c6a43d74dbf6047025ee0462d2d172e8b6a2841d6e";
        assert_eq!(
            hex::encode(*x_priv),
            EXPECTED,
            "clamped-scalar frozen derivation vector"
        );
    }

    /// Small-order / invalid inputs must be rejected (torsion check).
    #[test]
    fn rejects_invalid_and_small_order_points() {
        // y = 2 is not on the curve: decompression fails.
        let mut off_curve = [0u8; 32];
        off_curve[0] = 2;
        assert!(
            ed25519_pub_to_x25519(&off_curve).is_none(),
            "off-curve point"
        );
        // Compressed identity point (small order).
        let mut identity = [0u8; 32];
        identity[0] = 1;
        assert!(
            ed25519_pub_to_x25519(&identity).is_none(),
            "small-order point"
        );
    }
}
