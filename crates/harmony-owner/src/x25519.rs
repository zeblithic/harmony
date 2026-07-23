//! RFC 7748 §5 birational map: Ed25519 verify key → X25519 public key.
//!
//! ZEB-372 / ZEB-738. The implementation now lives in
//! [`harmony_crypto::x25519`] and is re-exported here so existing
//! `harmony_owner::x25519::ed25519_pub_to_x25519` call sites are unchanged. The
//! matching PRIVATE key is `harmony_crypto::x25519::ed25519_priv_to_x25519`
//! (`clamp(SigningKey::to_scalar_bytes())`) — derivable on demand from the
//! Ed25519 signing key, never stored. harmony-client's `dm_signing` wrappers
//! delegate to the same core module, so all three stay byte-identical by
//! construction rather than by a hand-maintained contract.

pub use harmony_crypto::x25519::ed25519_pub_to_x25519;

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;

    /// THE load-bearing parity test: the pub-side birational map must equal
    /// the public key of the priv-side clamped scalar (what harmony-client
    /// decrypts with). If these diverge, sealed deposits are undecryptable.
    #[test]
    fn pub_conversion_matches_priv_derived_public() {
        for seed in [[0x42u8; 32], [0x01u8; 32], [0xEEu8; 32]] {
            let sk = SigningKey::from_bytes(&seed);
            let via_pub =
                ed25519_pub_to_x25519(&sk.verifying_key().to_bytes()).expect("valid key converts");
            // mul_base_clamped applies RFC 7748 clamping internally — identical
            // to harmony-client's ed25519_priv_to_x25519 + X25519 base-point mul.
            let via_priv = curve25519_dalek::montgomery::MontgomeryPoint::mul_base_clamped(
                sk.to_scalar_bytes(),
            )
            .to_bytes();
            assert_eq!(
                via_pub, via_priv,
                "birational pub/priv parity (seed {:02x})",
                seed[0]
            );
        }
    }

    /// Reference vector — pins the derivation scheme. DO NOT regenerate
    /// casually: changing this derivation orphans every sealed blob addressed
    /// to existing bundles (mirrors the harmony-pkarr derive.rs case-vector
    /// discipline). Identical to `harmony_crypto::x25519`'s frozen vector.
    #[test]
    fn reference_vector_seed_42() {
        let sk = SigningKey::from_bytes(&[0x42u8; 32]);
        let x = ed25519_pub_to_x25519(&sk.verifying_key().to_bytes()).unwrap();
        // Pin: computed once from the first verified run of the parity test
        // above (2026-06-09). Regenerating breaks sealed-blob addressing.
        const EXPECTED: &str = "cc4f2cdb695dd766f34118eb67b98652fed1d8bc49c330b119bbfa8a64989378";
        assert_eq!(hex::encode(x), EXPECTED, "ZEB-372 frozen derivation vector");
    }

    /// Small-order / invalid inputs must be rejected, mirroring the client's
    /// torsion check.
    #[test]
    fn rejects_invalid_and_small_order_points() {
        // y = 2 is not on the curve: (y²-1)/(dy²+1) is a non-square, so
        // decompression fails. (Note: curve25519-dalek's decompress reduces
        // the y field element mod p rather than rejecting non-canonical
        // encodings, so e.g. [0xFF; 32] decompresses as y ≡ 18 — an
        // off-curve y is the right "invalid input" probe.)
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
