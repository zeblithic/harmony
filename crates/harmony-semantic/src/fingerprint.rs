// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Model fingerprint — first 4 bytes of SHA-256(model_id).

/// A 4-byte model fingerprint identifying the embedding model.
///
/// Computed as `SHA-256(model_id_string)[:4]`. Enables mixed-model
/// coexistence: sidecars from different models can coexist, and the
/// fingerprint disambiguates during overlay merge and search.
pub type ModelFingerprint = [u8; 4];

/// Compute a model fingerprint from a model identifier string.
pub fn model_fingerprint(model_id: &str) -> ModelFingerprint {
    let hash = harmony_crypto::hash::full_hash(model_id.as_bytes());
    let mut fp = [0u8; 4];
    fp.copy_from_slice(&hash[..4]);
    fp
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_fingerprint_deterministic() {
        let fp1 = model_fingerprint("all-MiniLM-L6-v2");
        let fp2 = model_fingerprint("all-MiniLM-L6-v2");
        assert_eq!(fp1, fp2);

        // Different model IDs must produce different fingerprints.
        let fp3 = model_fingerprint("nomic-embed-text-v1.5");
        assert_ne!(fp1, fp3);
    }
}
