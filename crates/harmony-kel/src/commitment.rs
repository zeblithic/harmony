use alloc::vec::Vec;
use harmony_crypto::hash::blake3_hash;
use harmony_crypto::ml_dsa::MlDsaPublicKey;
use harmony_crypto::ml_kem::MlKemPublicKey;

pub fn compute_commitment(
    signing_key: &MlDsaPublicKey,
    encryption_key: &MlKemPublicKey,
) -> [u8; 32] {
    let mut buf =
        Vec::with_capacity(signing_key.as_bytes().len() + encryption_key.as_bytes().len());
    buf.extend_from_slice(&signing_key.as_bytes());
    buf.extend_from_slice(&encryption_key.as_bytes());
    blake3_hash(&buf)
}

pub fn verify_commitment(
    signing_key: &MlDsaPublicKey,
    encryption_key: &MlKemPublicKey,
    commitment: &[u8; 32],
) -> bool {
    compute_commitment(signing_key, encryption_key) == *commitment
}

#[cfg(test)]
mod tests {
    use super::*;
    use harmony_crypto::ml_dsa;
    use harmony_crypto::ml_kem;

    #[test]
    fn commitment_is_deterministic() {
        let (sign_pk, _) = ml_dsa::generate(&mut rand::rngs::OsRng);
        let (enc_pk, _) = ml_kem::generate(&mut rand::rngs::OsRng);
        let c1 = compute_commitment(&sign_pk, &enc_pk);
        let c2 = compute_commitment(&sign_pk, &enc_pk);
        assert_eq!(c1, c2);
    }

    #[test]
    fn different_keys_produce_different_commitments() {
        let (sign_pk1, _) = ml_dsa::generate(&mut rand::rngs::OsRng);
        let (enc_pk1, _) = ml_kem::generate(&mut rand::rngs::OsRng);
        let (sign_pk2, _) = ml_dsa::generate(&mut rand::rngs::OsRng);
        let (enc_pk2, _) = ml_kem::generate(&mut rand::rngs::OsRng);
        assert_ne!(
            compute_commitment(&sign_pk1, &enc_pk1),
            compute_commitment(&sign_pk2, &enc_pk2)
        );
    }

    #[test]
    fn verify_commitment_matches() {
        let (sign_pk, _) = ml_dsa::generate(&mut rand::rngs::OsRng);
        let (enc_pk, _) = ml_kem::generate(&mut rand::rngs::OsRng);
        let commitment = compute_commitment(&sign_pk, &enc_pk);
        assert!(verify_commitment(&sign_pk, &enc_pk, &commitment));
    }

    #[test]
    fn verify_commitment_rejects_wrong_keys() {
        let (sign_pk1, _) = ml_dsa::generate(&mut rand::rngs::OsRng);
        let (enc_pk1, _) = ml_kem::generate(&mut rand::rngs::OsRng);
        let (sign_pk2, _) = ml_dsa::generate(&mut rand::rngs::OsRng);
        let (enc_pk2, _) = ml_kem::generate(&mut rand::rngs::OsRng);
        let commitment = compute_commitment(&sign_pk1, &enc_pk1);
        assert!(!verify_commitment(&sign_pk2, &enc_pk2, &commitment));
    }
}
