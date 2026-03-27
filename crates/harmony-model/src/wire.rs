//! Wire encoding/decoding for model metadata.

use crate::manifest::{ModelAdvertisement, ModelManifest};

/// Errors from model wire operations.
#[derive(Debug)]
pub enum ModelError {}

pub fn encode_manifest(_manifest: &ModelManifest) -> Result<Vec<u8>, ModelError> {
    todo!()
}
pub fn decode_manifest(_data: &[u8]) -> Result<ModelManifest, ModelError> {
    todo!()
}
pub fn manifest_cid(_data: &[u8]) -> harmony_content::ContentId {
    todo!()
}
pub fn encode_advertisement(_ad: &ModelAdvertisement) -> Result<Vec<u8>, ModelError> {
    todo!()
}
pub fn decode_advertisement(_data: &[u8]) -> Result<ModelAdvertisement, ModelError> {
    todo!()
}
