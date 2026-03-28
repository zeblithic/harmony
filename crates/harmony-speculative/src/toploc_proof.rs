//! TOPLOC proof generation and verification.
//!
//! Wraps the integer math core (toploc.rs) with candle tensor extraction
//! and deterministic sampling. Feature-gated behind `prefill`.

use candle_core::{DType, IndexOp, Tensor};
use harmony_crypto::hash::full_hash;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::prefill::{PrefillCacheHeader, PrefillError};
use crate::toploc::{self, CHUNK_SIZE, TOP_K};
use harmony_inference::InferenceCache;

// ── Serde helpers for [u16; TOP_K] (128 elements) ───────────────────
// serde_core (the workspace fork) only derives arrays up to size 32.
// We serialize as a flat byte sequence: 128 * 2 = 256 bytes (little-endian).

fn serialize_coefficients<S: Serializer>(
    arr: &[u16; TOP_K],
    s: S,
) -> Result<S::Ok, S::Error> {
    let mut bytes = [0u8; TOP_K * 2];
    for (i, &v) in arr.iter().enumerate() {
        let le = v.to_le_bytes();
        bytes[i * 2] = le[0];
        bytes[i * 2 + 1] = le[1];
    }
    s.serialize_bytes(&bytes)
}

fn deserialize_coefficients<'de, D: Deserializer<'de>>(
    d: D,
) -> Result<[u16; TOP_K], D::Error> {
    use serde::de::Error;
    let bytes: &[u8] = serde::de::Deserialize::deserialize(d)?;
    if bytes.len() != TOP_K * 2 {
        return Err(D::Error::invalid_length(
            bytes.len(),
            &"256 bytes for 128 u16 coefficients",
        ));
    }
    let mut arr = [0u16; TOP_K];
    for i in 0..TOP_K {
        arr[i] = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
    }
    Ok(arr)
}

/// A single TOPLOC proof: polynomial encoding of top-128 mantissas
/// from a KV tensor slice.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TocProof {
    /// Injective modulus for the finite field Z_m.
    pub modulus: u16,
    /// 128 polynomial coefficients in monomial form over Z_m.
    #[serde(
        serialize_with = "serialize_coefficients",
        deserialize_with = "deserialize_coefficients"
    )]
    pub coefficients: [u16; TOP_K],
    /// Which layer was sampled.
    pub layer: u16,
    /// Which KV head was sampled.
    pub head: u16,
    /// Starting token index for this chunk.
    pub token_offset: u32,
}

/// Result of TOPLOC verification.
#[derive(Debug, Clone)]
pub struct VerifyResult {
    pub proofs_checked: usize,
    pub proofs_passed: usize,
    pub details: Vec<ProofCheckDetail>,
}

impl VerifyResult {
    pub fn is_valid(&self) -> bool {
        self.proofs_checked > 0 && self.proofs_passed == self.proofs_checked
    }
}

#[derive(Debug, Clone)]
pub struct ProofCheckDetail {
    pub layer: u16,
    pub head: u16,
    pub token_offset: u32,
    pub agreement_rate: f32,
    pub mean_mantissa_diff: f32,
    pub median_mantissa_diff: f32,
    pub passed: bool,
}

// ── Thresholds ───────────────────────────────────────────────────────

const AGREEMENT_THRESHOLD: f32 = 0.5;
const MEAN_DIFF_THRESHOLD: f32 = 256.0;
const MEDIAN_DIFF_THRESHOLD: f32 = 128.0;

// ── Deterministic sampling ───────────────────────────────────────────

/// Derive (layer, head) for a given chunk index from the header hash.
fn sample_coordinates(
    header: &PrefillCacheHeader,
    chunk_idx: usize,
    num_layers: usize,
    num_kv_heads: usize,
) -> (usize, usize) {
    let mut seed_input = Vec::with_capacity(68);
    seed_input.extend_from_slice(&header.model_cid);
    seed_input.extend_from_slice(&header.token_hash);
    seed_input.extend_from_slice(&(chunk_idx as u32).to_le_bytes());
    let hash = full_hash(&seed_input);
    let layer = u16::from_le_bytes([hash[0], hash[1]]) as usize % num_layers;
    let head = u16::from_le_bytes([hash[2], hash[3]]) as usize % num_kv_heads;
    (layer, head)
}

// ── Tensor extraction helper ─────────────────────────────────────────

/// Extract a flattened f32 slice from a KV tensor at a specific head and token range.
/// Input shape: [1, num_kv_heads, seq_len, head_dim]
fn extract_kv_slice(
    tensor: &Tensor,
    head: usize,
    token_offset: usize,
    chunk_size: usize,
) -> Result<Vec<f32>, PrefillError> {
    let (_b, _h, seq_len, _head_dim) = tensor
        .dims4()
        .map_err(|e| PrefillError::SerializationFailed(format!("tensor shape: {e}")))?;
    if token_offset >= seq_len {
        return Err(PrefillError::SerializationFailed(format!(
            "token_offset {token_offset} >= seq_len {seq_len}"
        )));
    }
    let end = (token_offset + chunk_size).min(seq_len);
    let sliced = tensor
        .i((0, head, token_offset..end, ..))
        .map_err(|e| PrefillError::SerializationFailed(e.to_string()))?;
    let as_f32 = sliced
        .to_dtype(DType::F32)
        .map_err(|e| PrefillError::SerializationFailed(e.to_string()))?;
    let flat = as_f32
        .flatten_all()
        .map_err(|e| PrefillError::SerializationFailed(e.to_string()))?;
    flat.to_vec1::<f32>()
        .map_err(|e| PrefillError::SerializationFailed(e.to_string()))
}

// ── Proof generation ─────────────────────────────────────────────────

/// Generate TOPLOC proofs from a full-precision (uncompressed) KV cache.
/// Must be called BEFORE compress().
pub fn generate_proofs(
    cache: &InferenceCache,
    header: &PrefillCacheHeader,
) -> Result<Vec<TocProof>, PrefillError> {
    if cache.is_compressed() {
        return Err(PrefillError::SerializationFailed(
            "cache must not be compressed for proof generation".into(),
        ));
    }

    let num_layers = header.num_layers as usize;
    let num_kv_heads = header.num_kv_heads as usize;
    let seq_len = cache.position;
    let num_chunks = (seq_len + CHUNK_SIZE - 1) / CHUNK_SIZE;

    let mut proofs = Vec::with_capacity(num_chunks);

    for chunk_idx in 0..num_chunks {
        let (layer, head) = sample_coordinates(header, chunk_idx, num_layers, num_kv_heads);
        let token_offset = chunk_idx * CHUNK_SIZE;

        let (k_tensor, _v_tensor) = match &cache.layers[layer] {
            Some((k, v)) => (k, v),
            None => continue,
        };

        let flat = extract_kv_slice(k_tensor, head, token_offset, CHUNK_SIZE)?;

        if flat.len() < TOP_K {
            continue;
        }

        let (raw_indices, mantissas) = toploc::extract_top_k(&flat, TOP_K);

        let mut idx_arr = [0u16; TOP_K];
        let mut man_arr = [0u16; TOP_K];
        for i in 0..TOP_K {
            idx_arr[i] = raw_indices[i];
            man_arr[i] = mantissas[i];
        }

        let modulus = toploc::find_injective_modulus(&idx_arr);

        for idx in &mut idx_arr {
            *idx = (*idx as u32 % modulus as u32) as u16;
        }

        let coefficients = toploc::ndd_interpolate(&idx_arr, &man_arr, modulus);

        proofs.push(TocProof {
            modulus,
            coefficients,
            layer: layer as u16,
            head: head as u16,
            token_offset: token_offset as u32,
        });
    }

    Ok(proofs)
}

// ── Proof verification ───────────────────────────────────────────────

/// Verify TOPLOC proofs against a locally-computed KV cache.
pub fn verify_proofs(
    local_cache: &InferenceCache,
    header: &PrefillCacheHeader,
) -> Result<VerifyResult, PrefillError> {
    if local_cache.is_compressed() {
        return Err(PrefillError::SerializationFailed(
            "cache must not be compressed for verification".into(),
        ));
    }

    if header.num_layers == 0 || header.num_kv_heads == 0 {
        return Err(PrefillError::SerializationFailed(
            "header num_layers and num_kv_heads must be non-zero".into(),
        ));
    }

    let proofs = &header.proofs;

    if proofs.is_empty() {
        return Ok(VerifyResult {
            proofs_checked: 0,
            proofs_passed: 0,
            details: vec![],
        });
    }

    let mut details = Vec::with_capacity(proofs.len());

    for proof in proofs {
        let layer = proof.layer as usize;
        let head = proof.head as usize;
        let token_offset = proof.token_offset as usize;

        // Validate routing: re-derive expected (layer, head) from deterministic sampler.
        // Reject proofs that claim a different layer/head than the sampler dictates.
        let chunk_idx = token_offset / CHUNK_SIZE;
        let (expected_layer, expected_head) = sample_coordinates(
            header,
            chunk_idx,
            header.num_layers as usize,
            header.num_kv_heads as usize,
        );
        if layer != expected_layer || head != expected_head {
            details.push(ProofCheckDetail {
                layer: proof.layer,
                head: proof.head,
                token_offset: proof.token_offset,
                agreement_rate: 0.0,
                mean_mantissa_diff: f32::INFINITY,
                median_mantissa_diff: f32::INFINITY,
                passed: false,
            });
            continue;
        }

        // Bounds check: proof.layer is untrusted u16 from remote header.
        if layer >= local_cache.layers.len() {
            details.push(ProofCheckDetail {
                layer: proof.layer,
                head: proof.head,
                token_offset: proof.token_offset,
                agreement_rate: 0.0,
                mean_mantissa_diff: f32::INFINITY,
                median_mantissa_diff: f32::INFINITY,
                passed: false,
            });
            continue;
        }

        // Modulus guard: proof.modulus is untrusted. Must be a large prime
        // (legitimate proofs use primes near 65521). A small modulus collapses
        // all values to a tiny range, making all diffs trivially small and
        // defeating the verification threshold checks.
        const MIN_MODULUS: u16 = 1000;
        if proof.modulus < MIN_MODULUS {
            details.push(ProofCheckDetail {
                layer: proof.layer,
                head: proof.head,
                token_offset: proof.token_offset,
                agreement_rate: 0.0,
                mean_mantissa_diff: f32::INFINITY,
                median_mantissa_diff: f32::INFINITY,
                passed: false,
            });
            continue;
        }

        let (k_tensor, _) = match &local_cache.layers[layer] {
            Some((k, v)) => (k, v),
            None => {
                details.push(ProofCheckDetail {
                    layer: proof.layer,
                    head: proof.head,
                    token_offset: proof.token_offset,
                    agreement_rate: 0.0,
                    mean_mantissa_diff: f32::INFINITY,
                    median_mantissa_diff: f32::INFINITY,
                    passed: false,
                });
                continue;
            }
        };

        let flat = extract_kv_slice(k_tensor, head, token_offset, CHUNK_SIZE)?;

        if flat.len() < TOP_K {
            details.push(ProofCheckDetail {
                layer: proof.layer,
                head: proof.head,
                token_offset: proof.token_offset,
                agreement_rate: 0.0,
                mean_mantissa_diff: f32::INFINITY,
                median_mantissa_diff: f32::INFINITY,
                passed: false,
            });
            continue;
        }

        let (local_indices, local_mantissas) = toploc::extract_top_k(&flat, TOP_K);

        let mut diffs: Vec<f32> = Vec::new();
        for i in 0..TOP_K {
            let x = (local_indices[i] as u32 % proof.modulus as u32) as u16;
            let claimed = toploc::horner_evaluate(&proof.coefficients, x, proof.modulus);
            // Reduce local mantissa mod m to match NDD encoding (which reduces y-values mod m).
            let local = (local_mantissas[i] as u32 % proof.modulus as u32) as u16;
            let diff = (claimed as i32 - local as i32).unsigned_abs() as f32;
            diffs.push(diff);
        }

        let agreeing = diffs.iter().filter(|&&d| d <= MEAN_DIFF_THRESHOLD).count();
        let agreement_rate = agreeing as f32 / TOP_K as f32;

        let mut agreeing_diffs: Vec<f32> = diffs
            .iter()
            .copied()
            .filter(|&d| d <= MEAN_DIFF_THRESHOLD)
            .collect();

        let (mean_diff, median_diff) = if agreeing_diffs.is_empty() {
            (f32::INFINITY, f32::INFINITY)
        } else {
            let sum: f32 = agreeing_diffs.iter().sum();
            let mean = sum / agreeing_diffs.len() as f32;
            agreeing_diffs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median = agreeing_diffs[agreeing_diffs.len() / 2];
            (mean, median)
        };

        // Note: mean_diff <= MEAN_DIFF_THRESHOLD is tautologically true since
        // agreeing_diffs is pre-filtered to values <= MEAN_DIFF_THRESHOLD.
        // The effective criteria are agreement rate + median diff.
        let passed = agreement_rate >= AGREEMENT_THRESHOLD
            && median_diff <= MEDIAN_DIFF_THRESHOLD;

        details.push(ProofCheckDetail {
            layer: proof.layer,
            head: proof.head,
            token_offset: proof.token_offset,
            agreement_rate,
            mean_mantissa_diff: mean_diff,
            median_mantissa_diff: median_diff,
            passed,
        });
    }

    let proofs_passed = details.iter().filter(|d| d.passed).count();

    Ok(VerifyResult {
        proofs_checked: details.len(),
        proofs_passed,
        details,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use harmony_content::book::MemoryBookStore;
    use harmony_content::cid::ContentId;
    use crate::prefill::{
        PREFILL_MAGIC_V2, store_prefill_cache_with_proofs, load_prefill_cache, token_hash,
    };

    fn make_cache_with_data(num_layers: usize, num_kv_heads: usize, head_dim: usize, n_tokens: usize) -> InferenceCache {
        let mut cache = InferenceCache::new(num_layers, head_dim, num_kv_heads);
        if n_tokens > 0 {
            let shape = (1, num_kv_heads, n_tokens, head_dim);
            for layer in cache.layers.iter_mut() {
                let k = Tensor::rand(0f32, 1f32, shape, &Device::Cpu)
                    .unwrap().to_dtype(DType::F16).unwrap();
                let v = Tensor::rand(0f32, 1f32, shape, &Device::Cpu)
                    .unwrap().to_dtype(DType::F16).unwrap();
                *layer = Some((k, v));
            }
            cache.position = n_tokens;
        }
        cache
    }

    fn make_header(cache: &InferenceCache, token_ids: &[u32]) -> PrefillCacheHeader {
        PrefillCacheHeader {
            magic: PREFILL_MAGIC_V2,
            model_cid: [0xAB; 32],
            token_hash: token_hash(token_ids),
            token_count: cache.position as u32,
            num_layers: cache.num_layers as u16,
            num_kv_heads: cache.num_kv_heads as u16,
            head_dim: cache.head_dim as u16,
            quant_bits: 3,
            proofs: vec![],
        }
    }

    #[test]
    fn generate_verify_roundtrip() {
        // 32 tokens * 128 head_dim = 4096 elements >= TOP_K (128)
        let cache = make_cache_with_data(2, 8, 128, 32);
        let token_ids: Vec<u32> = (0..32).collect();
        let header = make_header(&cache, &token_ids);

        let proofs = generate_proofs(&cache, &header).unwrap();
        assert!(!proofs.is_empty(), "should generate at least one proof");

        let mut header_with_proofs = header;
        header_with_proofs.proofs = proofs;

        let result = verify_proofs(&cache, &header_with_proofs).unwrap();
        assert!(result.is_valid(), "verification should pass for identical cache: {:?}", result.details);
    }

    #[test]
    fn verify_rejects_tampered_cache() {
        let cache = make_cache_with_data(2, 8, 128, 32);
        let token_ids: Vec<u32> = (0..32).collect();
        let header = make_header(&cache, &token_ids);

        let proofs = generate_proofs(&cache, &header).unwrap();
        assert!(!proofs.is_empty());

        // Create a different cache (different random data)
        let tampered = make_cache_with_data(2, 8, 128, 32);

        let mut header_with_proofs = header;
        header_with_proofs.proofs = proofs;

        let result = verify_proofs(&tampered, &header_with_proofs).unwrap();
        assert!(!result.is_valid(), "verification should fail for tampered cache");
    }

    #[test]
    fn proofs_in_header_roundtrip() {
        let mut cache = make_cache_with_data(2, 8, 128, 32);
        let token_ids: Vec<u32> = (0..32).collect();
        let header = make_header(&cache, &token_ids);

        let proofs = generate_proofs(&cache, &header).unwrap();
        let proof_count = proofs.len();

        // Compress and store with proofs
        cache.compress().unwrap();
        let model_cid = ContentId::for_book(b"test-model", Default::default()).unwrap();
        let mut store = MemoryBookStore::new();

        let root = store_prefill_cache_with_proofs(
            &cache, &model_cid, &token_ids, proofs, &mut store
        ).unwrap();

        // Load and check proofs survived
        let (_, loaded_header) = load_prefill_cache(&root, &model_cid, &store).unwrap();
        assert_eq!(loaded_header.magic, PREFILL_MAGIC_V2);
        assert_eq!(loaded_header.proofs.len(), proof_count);
    }

    #[test]
    fn sampling_deterministic() {
        let header = PrefillCacheHeader {
            magic: *b"HKV\x02",
            model_cid: [0xAB; 32],
            token_hash: [0xCD; 32],
            token_count: 64,
            num_layers: 28,
            num_kv_heads: 8,
            head_dim: 128,
            quant_bits: 3,
            proofs: vec![],
        };
        let (l1, h1) = sample_coordinates(&header, 0, 28, 8);
        let (l2, h2) = sample_coordinates(&header, 0, 28, 8);
        assert_eq!((l1, h1), (l2, h2));
    }

    #[test]
    fn sampling_varies_across_chunks() {
        let header = PrefillCacheHeader {
            magic: *b"HKV\x02",
            model_cid: [0xAB; 32],
            token_hash: [0xCD; 32],
            token_count: 128,
            num_layers: 28,
            num_kv_heads: 8,
            head_dim: 128,
            quant_bits: 3,
            proofs: vec![],
        };
        let coords: Vec<_> = (0..4)
            .map(|i| sample_coordinates(&header, i, 28, 8))
            .collect();
        let unique: std::collections::HashSet<_> = coords.into_iter().collect();
        assert!(unique.len() >= 2, "sampling should vary across chunks");
    }

    #[test]
    fn proof_serde_roundtrip() {
        let proof = TocProof {
            modulus: 65521,
            coefficients: [42u16; TOP_K],
            layer: 14,
            head: 3,
            token_offset: 64,
        };
        let bytes = postcard::to_allocvec(&proof).unwrap();
        let restored: TocProof = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(proof, restored);
    }
}
