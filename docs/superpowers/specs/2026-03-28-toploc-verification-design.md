# TOPLOC Verification for Prefill KV Cache Integrity

**Date:** 2026-03-28
**Status:** Draft
**Bead:** harmony-mssf
**Depends on:** harmony-hbf0 (prefill KV cache sharing)

## Problem

When a node fetches a prefill KV cache from the mesh via CAS, it currently trusts the producer blindly. A malicious node could inject poisoned attention states — fabricated KV data that causes the model to generate biased, hallucinated, or adversarial text without altering model weights. TOPLOC (Locality Sensitive Hashing for Trustless Verifiable Inference) provides lightweight, hardware-robust proofs that the cache was genuinely produced by a specific model from a specific prompt.

## Algorithm Summary

TOPLOC encodes the top-128 activations from a KV tensor slice into an integer polynomial over a finite field Z_m. The 258-byte proof (2-byte modulus + 128 × 2-byte coefficients) acts as a locality-sensitive hash of the tensor's most salient features. Verification re-runs the forward pass, evaluates the polynomial at locally-derived indices, and compares against local mantissas using statistical thresholds.

**Key properties:**
- ~8 bytes/token overhead (258 bytes per 32-token chunk)
- Zero external dependencies (pure integer arithmetic, no crypto)
- Hardware-robust (IEEE 754 mantissa encoding tolerates cross-platform float differences)
- 100% detection rate, 0% false positives in published experiments (ICML 2025)
- Verification cost: one full prefill forward pass (same cost the verifier would pay anyway)

## Constraints

- **No new crates.** All code in harmony-speculative behind the existing `prefill` feature.
- **No new dependencies.** Pure integer arithmetic + candle tensor extraction. SHA-256 (already available via harmony-crypto) for deterministic sampling.
- **Stack-allocated math.** The NDD and modulus search use fixed-size arrays (`[u16; 128]`, `[bool; 65536]`) — no heap allocation in the hot path.
- **Proofs in the header.** `PrefillCacheHeader` gains a `proofs: Vec<TocProof>` field. No separate CAS blob.

## Architecture

```
Producer node:                               Consumer node:

  forward(tokens, &mut cache)                 load_prefill_cache(cid, model_cid)
  → cache has full-precision KV tensors       → (compressed cache, header with proofs)
  proofs = generate_proofs(cache, header)     decompress()
  cache.compress()                            forward(tokens, &mut local_cache)
  store_prefill_cache_with_proofs(            → local_cache has fresh KV tensors
    cache, model_cid, tokens, proofs, store)  verify_proofs(local_cache, header)
  → CAS blob with proofs in header            → VerifyResult { passed/failed }
                                              if valid: continue generation
                                              if invalid: discard, prefill locally
```

**Call ordering on producer:** `forward()` → `generate_proofs()` → `compress()` → `store_prefill_cache_with_proofs()`. Proofs must be generated before compression because they need full-precision tensors.

## TocProof Type

```rust
/// A single TOPLOC proof: polynomial encoding of top-128 mantissas
/// from a KV tensor slice. Core proof is 258 bytes (modulus + coefficients),
/// plus 8 bytes of routing metadata.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TocProof {
    /// Injective modulus for the finite field Z_m. Always ≤ 65536.
    pub modulus: u16,
    /// 128 polynomial coefficients in monomial form over Z_m.
    /// c_0 + c_1*x + c_2*x^2 + ... + c_127*x^127 (mod m).
    pub coefficients: [u16; 128],
    /// Which layer was sampled (0..num_layers).
    pub layer: u16,
    /// Which KV head was sampled (0..num_kv_heads).
    pub head: u16,
    /// Starting token index for this 32-token chunk.
    pub token_offset: u32,
}
```

Serialized size: ~266 bytes per proof. For 2048 tokens: 64 proofs × 266 bytes = ~17 KB.

## TocProofSet and Results

```rust
/// Collection of proofs covering a prefill cache.
/// Stored in PrefillCacheHeader.proofs.
pub type TocProofSet = Vec<TocProof>;

/// Result of TOPLOC verification.
#[derive(Debug, Clone)]
pub struct VerifyResult {
    pub proofs_checked: usize,
    pub proofs_passed: usize,
    pub details: Vec<ProofCheckDetail>,
}

#[derive(Debug, Clone)]
pub struct ProofCheckDetail {
    pub layer: u16,
    pub head: u16,
    pub token_offset: u32,
    /// Fraction of verifier's top-128 indices where the polynomial
    /// evaluation agrees with the local mantissa (within threshold).
    pub agreement_rate: f32,
    /// Mean mantissa difference for agreeing indices.
    pub mean_mantissa_diff: f32,
    /// Median mantissa difference for agreeing indices.
    pub median_mantissa_diff: f32,
    pub passed: bool,
}

impl VerifyResult {
    /// Whether all proofs passed verification thresholds.
    pub fn is_valid(&self) -> bool {
        self.proofs_checked > 0 && self.proofs_passed == self.proofs_checked
    }
}
```

## PrefillCacheHeader Change

Add the proofs field and bump the magic version:

```rust
/// Magic for proof-bearing blobs (version 2).
pub const PREFILL_MAGIC_V2: [u8; 4] = *b"HKV\x02";

pub struct PrefillCacheHeader {
    // ... existing fields unchanged ...

    /// TOPLOC proofs for KV cache integrity verification.
    /// Empty if producer did not generate proofs.
    pub proofs: Vec<TocProof>,
}
```

**Versioning:** Blobs with proofs use magic `HKV\x02`. The existing `HKV\x01` blobs (no proofs) remain loadable — `load_prefill_cache` accepts both magic versions. When loading `HKV\x01`, the `proofs` field is set to an empty Vec. This avoids postcard backward-compatibility issues (postcard cannot default missing trailing fields).

`store_prefill_cache` (no proofs) continues to emit `HKV\x01`. The new `store_prefill_cache_with_proofs` emits `HKV\x02`.

## store_prefill_cache API Change

The existing `store_prefill_cache` is unchanged (emits `HKV\x01`, no proofs). A new function handles proof-bearing blobs:

```rust
/// Store a compressed KV cache with TOPLOC proofs in CAS. Returns the root CID.
/// Emits HKV\x02 format with proofs embedded in the header.
pub fn store_prefill_cache_with_proofs(
    cache: &InferenceCache,
    model_cid: &ContentId,
    token_ids: &[u32],
    proofs: Vec<TocProof>,
    store: &mut dyn BookStore,
) -> Result<ContentId, PrefillError>;
```

Internally identical to `store_prefill_cache` except it sets `magic = PREFILL_MAGIC_V2` and `proofs = proofs` on the header. `load_prefill_cache` already returns the header, so consumers inspect `header.proofs` to decide whether to verify.

## Deterministic Sampling

For each 32-token chunk, determine which (layer, head) to audit:

```rust
fn sample_coordinates(
    header: &PrefillCacheHeader,
    chunk_idx: usize,
    num_layers: usize,
    num_kv_heads: usize,
) -> (usize, usize)
```

**Seed derivation:** `SHA-256(model_cid || token_hash || chunk_idx as u32 LE)`. The first two bytes select the layer (mod num_layers), the next two select the head (mod num_kv_heads). This ensures:
- Same cache → same sampling pattern (deterministic)
- Different chunks → different (layer, head) pairs (coverage)
- Adversary can't predict sampling without knowing model_cid + token_hash (which they already know if they produced the cache — but they can't forge the proofs without matching activations)

## Math Internals

All functions are private to the `toploc` module. Pure integer arithmetic, no heap allocation.

### find_injective_modulus

```rust
/// Find a large m ≤ 65536 such that all indices are unique mod m.
/// Searches downward from 65536 (succeeds on first try ~88% of the time).
fn find_injective_modulus(indices: &[u16; 128]) -> u16
```

Uses a stack-allocated `[bool; 65536]` collision array. Starts at 65536 and decrements until all `indices[i] % m` are distinct. Starting high maximizes the chance of first-try success and keeps the modular field large.

### ndd_interpolate (Newton Divided Differences → monomial form)

```rust
/// Compute 128 polynomial coefficients in MONOMIAL form over Z_m.
/// Internally uses Newton Divided Differences to find the interpolating
/// polynomial, then converts Newton basis to monomial (power) basis.
/// O(k²) = O(16384) multiplications.
fn ndd_interpolate(xs: &[u16; 128], ys: &[u16; 128], m: u16) -> [u16; 128]
```

**Two-phase computation:**
1. **NDD phase:** Build the Newton divided difference table over Z_m using modular inverse (Extended Euclidean Algorithm) for division. Produces 128 Newton basis coefficients.
2. **Conversion phase:** Convert Newton basis `c_0 + c_1(x-x_0) + c_2(x-x_0)(x-x_1) + ...` to monomial basis `a_0 + a_1*x + a_2*x^2 + ...` via synthetic expansion. This is O(k²) and runs in-place on a `[u16; 128]` buffer.

The monomial form is stored in the proof so the verifier can evaluate with standard Horner's method without needing the original x_i interpolation nodes.

### extended_gcd

```rust
/// Extended Euclidean Algorithm. Returns (gcd, x) where a*x ≡ gcd (mod b).
fn extended_gcd(a: u32, b: u32) -> (u32, i32)
```

Used to compute modular inverse: `inverse(a, m) = extended_gcd(a, m).1.rem_euclid(m)`.

### horner_evaluate

```rust
/// Evaluate polynomial in monomial form at x using Horner's method over Z_m.
/// p(x) = c_0 + c_1*x + c_2*x^2 + ... evaluated as c_0 + x*(c_1 + x*(c_2 + ...))
/// O(k) = O(128). All arithmetic in u32 to prevent overflow on 32-bit MIPS.
fn horner_evaluate(coeffs: &[u16; 128], x: u16, m: u16) -> u16
```

### extract_top_k_mantissas

```rust
/// Extract top-128 elements by absolute magnitude from an f32 slice.
/// Returns (indices, mantissa_values) where mantissa is the lower 16 bits
/// of the 23-bit IEEE 754 fraction field.
/// Panics if the slice has fewer than k elements.
fn extract_top_k_mantissas(data: &[f32], k: usize) -> ([u16; 128], [u16; 128])
```

Uses `f32::to_bits()` for zero-cost IEEE 754 decomposition. Mantissa = `(bits & 0x007FFFFF) as u16` (lower 16 bits of the 23-bit fraction). Exponent = `((bits >> 23) & 0xFF) as u8`.

Returns fixed-size arrays (not Vec) to match `find_injective_modulus` and `ndd_interpolate` signatures.

## Proof Generation

```rust
/// Generate TOPLOC proofs from a full-precision (uncompressed) KV cache.
/// Must be called BEFORE compress().
pub fn generate_proofs(
    cache: &InferenceCache,
    header: &PrefillCacheHeader,
) -> Result<Vec<TocProof>, PrefillError>
```

**Flow per 32-token chunk:**
1. `sample_coordinates(header, chunk_idx)` → (layer, head)
2. Extract K tensor from `cache.layers[layer]` → `[1, num_kv_heads, seq_len, head_dim]`
3. Narrow to head and 32-token window → `[32, 128]` → flatten to 4096 f32 values
4. `extract_top_k_mantissas(flat, 128)` → (indices, mantissas) as `[u16; 128]`
5. `find_injective_modulus(&indices)` → m
6. Reduce indices in-place: `indices[i] = indices[i] % m`
7. `ndd_interpolate(&indices, &mantissas, m)` → 128 monomial coefficients
8. Pack `TocProof { modulus: m, coefficients, layer, head, token_offset }`

**Edge cases:**
- If `seq_len < 32` for the last chunk, use whatever tokens remain.
- If a layer is not populated (`layers[layer]` is None), skip that chunk's proof.

## Proof Verification

```rust
/// Verify TOPLOC proofs against a locally-computed KV cache.
/// The local cache must be from running forward() with the same tokens.
pub fn verify_proofs(
    local_cache: &InferenceCache,
    header: &PrefillCacheHeader,
) -> Result<VerifyResult, PrefillError>
```

**How verification works:** The polynomial encodes the producer's top-128 (index, mantissa) pairs. The verifier extracts its own top-128 from the local forward pass, evaluates the proof polynomial at each of its indices, and checks whether the polynomial output matches the local mantissa. If the producer and verifier ran the same model on the same tokens, their top-128 sets overlap substantially, and the polynomial values at overlapping indices agree closely.

**Flow per proof:**
1. Extract same KV slice from `local_cache` (layer, head, token window)
2. Flatten to f32, `extract_top_k_mantissas` → local `(indices, mantissas)` as `[u16; 128]`
3. Decompose local values: extract exponents via `(bits >> 23) & 0xFF`
4. For each of the 128 local indices: `horner_evaluate(proof.coefficients, index % proof.modulus, proof.modulus)` → polynomial output
5. Compare polynomial output against local mantissa. If `|poly_output - local_mantissa| ≤ mantissa_threshold`, this index "agrees"
6. `agreement_rate` = count of agreeing indices / 128
7. For agreeing indices: compute mean and median of `|poly_output - local_mantissa|`

**Acceptance thresholds (preliminary, may need tuning with real model data):**
- Agreement rate ≥ 0.5 (at least 64 of 128 indices agree)
- Mean mantissa difference ≤ 256
- Median mantissa difference ≤ 128

## Testing Strategy

### Math unit tests

| Test | What |
|---|---|
| `injective_modulus_finds_valid` | 128 random indices → modulus found, all unique mod m |
| `injective_modulus_deterministic` | Same indices → same modulus |
| `ndd_interpolate_roundtrip` | Interpolate then Horner-evaluate at original x → original y |
| `horner_matches_direct` | Horner matches naive sum-of-products evaluation |
| `extended_gcd_known_vectors` | Known pairs → correct modular inverse |
| `extract_top_k_selects_highest` | Synthetic f32 slice → correct top-128 by magnitude |
| `mantissa_extraction_correct` | Known f32 → correct IEEE 754 mantissa bits |

### Integration tests

| Test | What |
|---|---|
| `generate_verify_roundtrip` | Generate from cache, verify same cache → all pass |
| `verify_rejects_tampered_cache` | Generate, corrupt one layer, verify → fails |
| `sampling_deterministic` | Same header → same (layer, head) per chunk |
| `sampling_varies_across_chunks` | Different chunks → different coordinates |
| `proof_serde_roundtrip` | TocProof postcard roundtrip, correct size |
| `proofs_in_header_roundtrip` | Header with proofs via store_with_proofs → load → proofs intact |
| `load_v1_blob_has_empty_proofs` | Load HKV\x01 blob → header.proofs is empty |

## Out of Scope

- **Proof generation during forward pass** (inline hooking) — would require changes to qwen3_ext.rs. Current approach extracts from the cache after forward.
- **Selective verification** (only verify a random subset of proofs) — all proofs are verified. The per-proof cost is trivial (Horner evaluation).
- **Configurable thresholds** — hardcoded for now. Can be made configurable in a follow-up if needed.
- **f16 mantissa handling** — the current implementation converts to f32 before extraction. Direct f16 mantissa extraction (10-bit mantissa) is a minor optimization for later.
