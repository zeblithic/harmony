//! Tensor slicing into fixed-size shards with CID computation.
//!
//! Handles f32/bf16 → f16 conversion for non-f16 checkpoints.

use half::f16;
use harmony_content::cid::{ContentFlags, ContentId};

/// A single shard ready for storage.
#[derive(Debug)]
pub struct Shard {
    /// The shard's content identifier.
    pub cid: ContentId,
    /// Raw shard bytes (contiguous f16 vectors).
    pub data: Vec<u8>,
}

/// Source tensor dtype for conversion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SourceDtype {
    F16,
    F32,
    BF16,
}

/// Compute the number of shards needed for the given table.
pub fn num_shards(total_entries: u64, shard_size: u32) -> u64 {
    total_entries.div_ceil(shard_size as u64)
}

/// Convert a slice of source-dtype bytes to f16 bytes.
///
/// For F16 input, returns a copy. For F32/BF16, converts each element.
pub fn convert_to_f16(src: &[u8], dtype: SourceDtype) -> Vec<u8> {
    match dtype {
        SourceDtype::F16 => src.to_vec(),
        SourceDtype::F32 => {
            let mut out = Vec::with_capacity(src.len() / 2);
            for chunk in src.chunks_exact(4) {
                let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                out.extend_from_slice(&f16::from_f32(val).to_le_bytes());
            }
            out
        }
        SourceDtype::BF16 => {
            let mut out = Vec::with_capacity(src.len());
            for chunk in src.chunks_exact(2) {
                // bf16 → f32: place bf16 bytes in upper 16 bits of f32.
                let val = f32::from_bits((chunk[1] as u32) << 24 | (chunk[0] as u32) << 16);
                out.extend_from_slice(&f16::from_f32(val).to_le_bytes());
            }
            out
        }
    }
}

/// Bytes per element for the source dtype.
pub fn source_dtype_bytes(dtype: SourceDtype) -> usize {
    match dtype {
        SourceDtype::F16 | SourceDtype::BF16 => 2,
        SourceDtype::F32 => 4,
    }
}

/// Slice a single shard from the tensor's raw bytes, converting to f16.
///
/// If the shard is the last one and has fewer than `shard_size` entries,
/// the output is zero-padded to the full shard size.
pub fn slice_shard(
    tensor_bytes: &[u8],
    shard_index: u64,
    shard_size: u32,
    embedding_dim: usize,
    dtype: SourceDtype,
) -> Shard {
    let src_vector_bytes = embedding_dim * source_dtype_bytes(dtype);
    let src_shard_bytes = shard_size as usize * src_vector_bytes;
    let start = shard_index as usize * src_shard_bytes;
    let available = tensor_bytes.len().saturating_sub(start);
    let copy_len = available.min(src_shard_bytes);

    let src_slice = &tensor_bytes[start..start + copy_len];
    let f16_data = convert_to_f16(src_slice, dtype);

    // Target shard size in f16 bytes.
    let f16_vector_bytes = embedding_dim * 2;
    let f16_shard_bytes = shard_size as usize * f16_vector_bytes;

    let mut data = vec![0u8; f16_shard_bytes];
    let copy = f16_data.len().min(f16_shard_bytes);
    data[..copy].copy_from_slice(&f16_data[..copy]);

    let cid = ContentId::for_book(&data, ContentFlags::default())
        .expect("shard size is well under 1MB limit");

    Shard { cid, data }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn num_shards_exact_division() {
        assert_eq!(num_shards(12, 3), 4);
        assert_eq!(num_shards(200, 200), 1);
    }

    #[test]
    fn num_shards_remainder() {
        assert_eq!(num_shards(7, 3), 3);
        assert_eq!(num_shards(1, 200), 1);
    }

    #[test]
    fn slice_shard_basic() {
        let tensor_bytes: Vec<u8> = (0..16).collect();
        let shard = slice_shard(&tensor_bytes, 0, 2, 2, SourceDtype::F16);
        assert_eq!(shard.data, &tensor_bytes[0..8]);
        assert_eq!(shard.data.len(), 8);
    }

    #[test]
    fn slice_shard_second() {
        let tensor_bytes: Vec<u8> = (0..16).collect();
        let shard = slice_shard(&tensor_bytes, 1, 2, 2, SourceDtype::F16);
        assert_eq!(shard.data, &tensor_bytes[8..16]);
    }

    #[test]
    fn slice_shard_last_padded() {
        let vector_bytes = 2 * 2; // dim=2, f16=2 bytes
        let tensor_bytes: Vec<u8> = (0..(5 * vector_bytes) as u8).collect();
        let shard = slice_shard(&tensor_bytes, 1, 3, 2, SourceDtype::F16);
        assert_eq!(shard.data.len(), 12);
        assert_eq!(&shard.data[0..8], &tensor_bytes[12..20]);
        assert_eq!(&shard.data[8..12], &[0, 0, 0, 0]);
    }

    #[test]
    fn slice_shard_deterministic_cid() {
        let tensor_bytes: Vec<u8> = (0..16).collect();
        let s1 = slice_shard(&tensor_bytes, 0, 2, 2, SourceDtype::F16);
        let s2 = slice_shard(&tensor_bytes, 0, 2, 2, SourceDtype::F16);
        assert_eq!(s1.cid, s2.cid);
    }

    #[test]
    fn different_shards_different_cids() {
        let tensor_bytes: Vec<u8> = (0..16).collect();
        let s1 = slice_shard(&tensor_bytes, 0, 2, 2, SourceDtype::F16);
        let s2 = slice_shard(&tensor_bytes, 1, 2, 2, SourceDtype::F16);
        assert_ne!(s1.cid, s2.cid);
    }

    #[test]
    fn convert_f32_to_f16() {
        let mut src = Vec::new();
        src.extend_from_slice(&1.0f32.to_le_bytes());
        src.extend_from_slice(&2.0f32.to_le_bytes());
        let f16_bytes = convert_to_f16(&src, SourceDtype::F32);
        assert_eq!(f16_bytes.len(), 4);
        let v0 = f16::from_le_bytes([f16_bytes[0], f16_bytes[1]]);
        let v1 = f16::from_le_bytes([f16_bytes[2], f16_bytes[3]]);
        assert_eq!(v0.to_f32(), 1.0);
        assert_eq!(v1.to_f32(), 2.0);
    }

    #[test]
    fn slice_shard_f32_converts_to_f16() {
        let mut tensor_bytes = Vec::new();
        for val in &[1.0f32, 2.0, 3.0, 4.0] {
            tensor_bytes.extend_from_slice(&val.to_le_bytes());
        }
        let shard = slice_shard(&tensor_bytes, 0, 2, 2, SourceDtype::F32);
        assert_eq!(shard.data.len(), 8);
        let v0 = f16::from_le_bytes([shard.data[0], shard.data[1]]);
        assert_eq!(v0.to_f32(), 1.0);
    }
}
