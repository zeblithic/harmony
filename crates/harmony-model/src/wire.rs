//! Wire encoding/decoding for model metadata.
//!
//! Uses postcard for compact binary serialization (matches harmony-engram convention).

use harmony_content::{ContentFlags, ContentId};

use crate::manifest::{ModelAdvertisement, ModelManifest};

/// Maximum encoded manifest size (safety check).
const MAX_MANIFEST_SIZE: usize = 64 * 1024;

/// Errors from model wire operations.
#[derive(Debug)]
pub enum ModelError {
    /// Postcard serialization failed.
    EncodeFailed,
    /// Postcard deserialization failed.
    DecodeFailed,
    /// Encoded manifest exceeds 64KB safety limit.
    ManifestTooLarge,
}

impl core::fmt::Display for ModelError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EncodeFailed => write!(f, "manifest encode failed"),
            Self::DecodeFailed => write!(f, "manifest decode failed"),
            Self::ManifestTooLarge => write!(f, "encoded manifest exceeds 64KB limit"),
        }
    }
}

impl std::error::Error for ModelError {}

/// Serialize a [`ModelManifest`] to postcard bytes.
///
/// Returns [`ModelError::ManifestTooLarge`] if the encoded size exceeds 64KB.
pub fn encode_manifest(manifest: &ModelManifest) -> Result<Vec<u8>, ModelError> {
    let bytes = postcard::to_allocvec(manifest).map_err(|_| ModelError::EncodeFailed)?;
    if bytes.len() > MAX_MANIFEST_SIZE {
        return Err(ModelError::ManifestTooLarge);
    }
    Ok(bytes)
}

/// Deserialize a [`ModelManifest`] from postcard bytes.
pub fn decode_manifest(data: &[u8]) -> Result<ModelManifest, ModelError> {
    postcard::from_bytes(data).map_err(|_| ModelError::DecodeFailed)
}

/// Compute the [`ContentId`] for an encoded manifest.
///
/// Manifests are always public durable (flags `00`).
pub fn manifest_cid(data: &[u8]) -> ContentId {
    ContentId::for_book(data, ContentFlags::default())
        .expect("encoded manifest should be within book size limit")
}

/// Serialize a [`ModelAdvertisement`] to postcard bytes.
pub fn encode_advertisement(ad: &ModelAdvertisement) -> Result<Vec<u8>, ModelError> {
    postcard::to_allocvec(ad).map_err(|_| ModelError::EncodeFailed)
}

/// Deserialize a [`ModelAdvertisement`] from postcard bytes.
pub fn decode_advertisement(data: &[u8]) -> Result<ModelAdvertisement, ModelError> {
    postcard::from_bytes(data).map_err(|_| ModelError::DecodeFailed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{ModelAdvertisement, ModelFormat, ModelManifest, ModelTask};
    use harmony_content::{ContentFlags, ContentId};

    fn sample_manifest() -> ModelManifest {
        let data_cid = ContentId::for_book(b"model-data", ContentFlags::default()).unwrap();
        ModelManifest {
            name: "Qwen3-0.6B-Q4_K_M".into(),
            family: "qwen3".into(),
            parameter_count: 600_000_000,
            format: ModelFormat::Gguf,
            quantization: Some("Q4_K_M".into()),
            context_length: 32768,
            vocab_size: 151936,
            memory_estimate: 512_000_000,
            tasks: vec![ModelTask::TextGeneration],
            data_cid,
            tokenizer_cid: None,
        }
    }

    fn sample_advertisement() -> ModelAdvertisement {
        let manifest_cid = ContentId::for_book(b"manifest-bytes", ContentFlags::default()).unwrap();
        ModelAdvertisement {
            manifest_cid,
            name: "Qwen3-0.6B-Q4_K_M".into(),
            family: "qwen3".into(),
            parameter_count: 600_000_000,
            quantization: Some("Q4_K_M".into()),
            tasks: vec![ModelTask::TextGeneration],
            memory_estimate: 512_000_000,
        }
    }

    #[test]
    fn manifest_round_trip() {
        let manifest = sample_manifest();
        let encoded = encode_manifest(&manifest).unwrap();
        let decoded = decode_manifest(&encoded).unwrap();
        assert_eq!(manifest, decoded);
    }

    #[test]
    fn manifest_cid_deterministic() {
        let manifest = sample_manifest();
        let encoded = encode_manifest(&manifest).unwrap();
        let cid1 = manifest_cid(&encoded);
        let cid2 = manifest_cid(&encoded);
        assert_eq!(cid1, cid2);
    }

    #[test]
    fn advertisement_round_trip() {
        let ad = sample_advertisement();
        let encoded = encode_advertisement(&ad).unwrap();
        let decoded = decode_advertisement(&encoded).unwrap();
        assert_eq!(ad, decoded);
    }

    #[test]
    fn manifest_too_large_rejected() {
        let mut manifest = sample_manifest();
        manifest.name = "x".repeat(70_000);
        let result = encode_manifest(&manifest);
        assert!(matches!(result, Err(ModelError::ManifestTooLarge)));
    }

    #[test]
    fn decode_manifest_empty_fails() {
        let result = decode_manifest(&[]);
        assert!(matches!(result, Err(ModelError::DecodeFailed)));
    }

    #[test]
    fn decode_advertisement_empty_fails() {
        let result = decode_advertisement(&[]);
        assert!(matches!(result, Err(ModelError::DecodeFailed)));
    }
}
