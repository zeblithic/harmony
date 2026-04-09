//! Integration test: Engram lookup → resolve → inject through the full pipeline.
//!
//! Creates a small f16 Engram table, constructs an EngramClient, runs the
//! bridge (N-gram extraction + embedding resolution), and validates injection
//! via EngramGatedResidual.  No external files or network access needed.

use std::collections::HashMap;

use candle_core::{Device, IndexOp, Tensor};
use harmony_engram::EngramConfig;
use harmony_inference::engram_bridge::{prepare_engram_request, resolve_engram_embeddings};
use harmony_inference::engram_residual::EngramGatedResidual;

/// Table parameters for tests.
const EMBEDDING_DIM: usize = 8;
const NUM_HEADS: u32 = 2;
const SHARD_SIZE: u32 = 4;
const TOTAL_ENTRIES: u64 = 12;
const NUM_SHARDS: u64 = 3; // ceil(12 / 4)
const HASH_SEEDS: [u64; 2] = [42, 99];

/// Build an EngramConfig for testing.
fn test_config() -> EngramConfig {
    EngramConfig {
        version: "v1".into(),
        embedding_dim: EMBEDDING_DIM,
        dtype_bytes: 2, // f16
        num_heads: NUM_HEADS,
        shard_size: SHARD_SIZE,
        num_shards: NUM_SHARDS,
        total_entries: TOTAL_ENTRIES,
        hash_seeds: HASH_SEEDS.to_vec(),
    }
}

/// Build an EngramClient with dummy CIDs.
fn test_client() -> harmony_engram::EngramClient {
    let config = test_config();
    let shard_cids: Vec<[u8; 32]> = (0..NUM_SHARDS)
        .map(|i| {
            let mut cid = [0u8; 32];
            cid[0] = i as u8;
            cid
        })
        .collect();
    harmony_engram::EngramClient::from_manifest(config, shard_cids)
}

/// Encode an f32 value as IEEE 754 binary16 (f16) little-endian bytes.
fn f32_to_f16_bytes(val: f32) -> [u8; 2] {
    // Simplified f32→f16 conversion for test values in [0, 2] range.
    let bits = val.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let frac = bits & 0x7FFFFF;

    if exp == 0 {
        // Zero or subnormal f32 → f16 zero
        return (sign as u16 * 0x8000).to_le_bytes();
    }

    let f16_exp = exp - 127 + 15;
    let f16_frac = (frac >> 13) as u16;

    let h = if f16_exp <= 0 {
        (sign as u16) << 15 // Underflow → signed zero
    } else if f16_exp >= 31 {
        ((sign as u16) << 15) | 0x7C00 // Overflow → infinity
    } else {
        ((sign as u16) << 15) | ((f16_exp as u16) << 10) | f16_frac
    };
    h.to_le_bytes()
}

/// Generate shard data with deterministic non-zero f16 values.
///
/// Each entry's embedding has all components set to `(entry_index + 1) * 0.1`,
/// so entry 0 → 0.1, entry 1 → 0.2, etc.  This makes it easy to verify that
/// the right entries were looked up and aggregated.
fn generate_shard_data() -> HashMap<u64, Vec<u8>> {
    let vector_bytes = EMBEDDING_DIM * 2; // f16
    let shard_bytes = SHARD_SIZE as usize * vector_bytes;

    let mut shards = HashMap::new();
    for shard_idx in 0..NUM_SHARDS {
        let mut data = vec![0u8; shard_bytes];
        for entry in 0..SHARD_SIZE as usize {
            let global_idx = shard_idx as usize * SHARD_SIZE as usize + entry;
            if global_idx >= TOTAL_ENTRIES as usize {
                break; // Last shard may be partially filled
            }
            let val_bytes = f32_to_f16_bytes((global_idx as f32 + 1.0) * 0.1);
            let offset = entry * vector_bytes;
            for d in 0..EMBEDDING_DIM {
                data[offset + d * 2] = val_bytes[0];
                data[offset + d * 2 + 1] = val_bytes[1];
            }
        }
        shards.insert(shard_idx, data);
    }
    shards
}

#[test]
fn lookup_resolve_produces_correct_shape() {
    let client = test_client();
    let shard_data = generate_shard_data();
    let tokens = [1u32, 2, 3, 4, 5];

    let request = prepare_engram_request(&client, &tokens).unwrap();
    let tensor =
        resolve_engram_embeddings(&client, &request, &shard_data, &Device::Cpu).unwrap();

    // Output: [1, seq_len, embedding_dim]
    assert_eq!(tensor.dims(), &[1, tokens.len(), EMBEDDING_DIM]);
}

#[test]
fn lookup_resolve_position_zero_is_zero() {
    // Position 0 has no N-gram coverage (bigrams start at position 1,
    // trigrams at position 2), so its embedding should be all zeros.
    let client = test_client();
    let shard_data = generate_shard_data();
    let tokens = [10u32, 20, 30];

    let request = prepare_engram_request(&client, &tokens).unwrap();
    let tensor =
        resolve_engram_embeddings(&client, &request, &shard_data, &Device::Cpu).unwrap();

    // Extract position 0
    let pos0 = tensor
        .i((0, 0, ..))
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    let max_abs: f32 = pos0.iter().map(|v: &f32| v.abs()).fold(0f32, f32::max);
    assert!(
        max_abs < 1e-6,
        "position 0 should be zero (no N-gram coverage), got max abs {max_abs}"
    );
}

#[test]
fn lookup_resolve_later_positions_are_nonzero() {
    // Positions 1+ should have non-zero embeddings from bigram/trigram lookups.
    let client = test_client();
    let shard_data = generate_shard_data();
    let tokens = [1u32, 2, 3, 4];

    let request = prepare_engram_request(&client, &tokens).unwrap();
    let tensor =
        resolve_engram_embeddings(&client, &request, &shard_data, &Device::Cpu).unwrap();

    for pos in 1..tokens.len() {
        let pos_vec = tensor
            .i((0, pos, ..))
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let max_abs: f32 = pos_vec.iter().map(|v: &f32| v.abs()).fold(0f32, f32::max);
        assert!(
            max_abs > 1e-6,
            "position {pos} should be non-zero (has N-gram coverage), got max abs {max_abs}"
        );
    }
}

#[test]
fn lookup_is_deterministic() {
    // Same tokens must produce identical embeddings across calls.
    let client = test_client();
    let shard_data = generate_shard_data();
    let tokens = [5u32, 10, 15, 20];

    let request1 = prepare_engram_request(&client, &tokens).unwrap();
    let tensor1 =
        resolve_engram_embeddings(&client, &request1, &shard_data, &Device::Cpu).unwrap();

    let request2 = prepare_engram_request(&client, &tokens).unwrap();
    let tensor2 =
        resolve_engram_embeddings(&client, &request2, &shard_data, &Device::Cpu).unwrap();

    let diff: f32 = (&tensor1 - &tensor2)
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar()
        .unwrap();
    assert!(diff < 1e-6, "determinism violated: max diff = {diff}");
}

#[test]
fn different_tokens_produce_different_embeddings() {
    let client = test_client();
    let shard_data = generate_shard_data();

    let tokens_a = [1u32, 2, 3];
    let tokens_b = [100u32, 200, 300];

    let req_a = prepare_engram_request(&client, &tokens_a).unwrap();
    let tensor_a =
        resolve_engram_embeddings(&client, &req_a, &shard_data, &Device::Cpu).unwrap();

    let req_b = prepare_engram_request(&client, &tokens_b).unwrap();
    let tensor_b =
        resolve_engram_embeddings(&client, &req_b, &shard_data, &Device::Cpu).unwrap();

    // Compare position 1 (first bigram position) — should differ
    let a1 = tensor_a.i((0, 1, ..)).unwrap().to_vec1::<f32>().unwrap();
    let b1 = tensor_b.i((0, 1, ..)).unwrap().to_vec1::<f32>().unwrap();

    let differs = a1
        .iter()
        .zip(b1.iter())
        .any(|(a, b): (&f32, &f32)| (a - b).abs() > 1e-6);
    assert!(
        differs,
        "different tokens should produce different embeddings at the same position"
    );
}

#[test]
fn full_pipeline_lookup_resolve_inject() {
    // End-to-end: lookup → resolve → EngramGatedResidual → residual addition.
    let device = Device::Cpu;
    let client = test_client();
    let shard_data = generate_shard_data();
    let tokens = [1u32, 2, 3, 4, 5, 6];
    let seq_len = tokens.len();
    let hidden_dim = 32;

    // Phase 1 & 2: lookup and resolve
    let request = prepare_engram_request(&client, &tokens).unwrap();
    let engram_embeddings =
        resolve_engram_embeddings(&client, &request, &shard_data, &device).unwrap();
    assert_eq!(engram_embeddings.dims(), &[1, seq_len, EMBEDDING_DIM]);

    // Phase 3: inject via EngramGatedResidual
    // Use from_tensors with non-zero conv weights (::new() zeros the conv,
    // which would make silu(conv(x)) = silu(0) = 0 for all inputs).
    let module = EngramGatedResidual::from_tensors(
        Tensor::randn(0f32, 1f32, (hidden_dim, EMBEDDING_DIM), &device).unwrap(),
        Tensor::randn(0f32, 1f32, (hidden_dim, EMBEDDING_DIM), &device).unwrap(),
        Tensor::ones(hidden_dim, candle_core::DType::F32, &device).unwrap(),
        Tensor::ones(hidden_dim, candle_core::DType::F32, &device).unwrap(),
        Tensor::ones((hidden_dim, 1, 3), candle_core::DType::F32, &device).unwrap(),
        hidden_dim,
        1e-6,
    )
    .unwrap();

    let hidden_state = Tensor::randn(0f32, 1f32, (1, seq_len, hidden_dim), &device).unwrap();

    let residual = module.forward(&hidden_state, &engram_embeddings).unwrap();
    assert_eq!(residual.dims(), &[1, seq_len, hidden_dim]);

    // Apply residual (as the model would)
    let output = (&hidden_state + &residual).unwrap();
    assert_eq!(output.dims(), &[1, seq_len, hidden_dim]);

    // Output should differ from input (engram injection had effect)
    let diff: f32 = (&output - &hidden_state)
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar()
        .unwrap();
    assert!(
        diff > 1e-8,
        "engram injection should modify hidden state, got max diff {diff}"
    );
}

#[test]
fn inject_position_zero_unchanged() {
    // Position 0 has zero engram embedding → residual should be zero there.
    let device = Device::Cpu;
    let client = test_client();
    let shard_data = generate_shard_data();
    let tokens = [1u32, 2, 3, 4];
    let hidden_dim = 16;

    let request = prepare_engram_request(&client, &tokens).unwrap();
    let engram_embeddings =
        resolve_engram_embeddings(&client, &request, &shard_data, &device).unwrap();

    // Use from_tensors with non-zero conv weights so the module is active —
    // ::new() zero-inits the conv, which would make all positions zero and
    // the test vacuously true.
    let module = EngramGatedResidual::from_tensors(
        Tensor::randn(0f32, 1f32, (hidden_dim, EMBEDDING_DIM), &device).unwrap(),
        Tensor::randn(0f32, 1f32, (hidden_dim, EMBEDDING_DIM), &device).unwrap(),
        Tensor::ones(hidden_dim, candle_core::DType::F32, &device).unwrap(),
        Tensor::ones(hidden_dim, candle_core::DType::F32, &device).unwrap(),
        Tensor::ones((hidden_dim, 1, 3), candle_core::DType::F32, &device).unwrap(),
        hidden_dim,
        1e-6,
    )
    .unwrap();

    let hidden_state =
        Tensor::randn(0f32, 1f32, (1, tokens.len(), hidden_dim), &device).unwrap();

    let residual = module.forward(&hidden_state, &engram_embeddings).unwrap();

    // Position 0 has zero engram embedding (no N-gram coverage) → zero value
    // projection → zero gated value → zero conv input at this position →
    // silu(0) = 0.
    let res0 = residual
        .i((0, 0, ..))
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let max_at_0: f32 = res0.iter().map(|v: &f32| v.abs()).fold(0f32, f32::max);
    assert!(
        max_at_0 < 1e-6,
        "position 0 residual should be zero, got max abs {max_at_0}"
    );

    // Verify the module IS active at later positions (proves the test is
    // discriminative — position 0 is zero because of the zero engram, not
    // because the module is dead).
    let last_pos = tokens.len() - 1;
    let res_last = residual
        .i((0, last_pos, ..))
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let max_at_last: f32 = res_last.iter().map(|v: &f32| v.abs()).fold(0f32, f32::max);
    assert!(
        max_at_last > 1e-6,
        "last position should be non-zero (module is active), got max abs {max_at_last}"
    );
}

#[test]
fn single_token_produces_zero_embeddings() {
    // A single token can't form any N-grams → all-zero embeddings.
    let client = test_client();
    let shard_data = generate_shard_data();
    let tokens = [42u32];

    let request = prepare_engram_request(&client, &tokens).unwrap();
    assert!(request.lookups.is_empty(), "single token has no N-grams");

    let tensor =
        resolve_engram_embeddings(&client, &request, &shard_data, &Device::Cpu).unwrap();
    assert_eq!(tensor.dims(), &[1, 1, EMBEDDING_DIM]);

    let max_val: f32 = tensor
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar()
        .unwrap();
    assert!(max_val < 1e-6, "single token should produce zero embeddings");
}
