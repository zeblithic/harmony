# Phase 0h: Candle Validation Loop Design

> The capstone of Phase 0. Proves that the full pipeline works: PyTorch weights
> exported to GGUF produce identical outputs when loaded and run in candle.

**Dependencies:** Phase 0e (HarmonyModel in candle), Phase 0f (training scaffold in PyTorch), Phase 0g (GGUF export/import)

## Goal

Verify that PyTorch and candle produce numerically identical logits from the same
weights, proving weight portability and architectural alignment across the two
implementations.

## Architecture

Two-step fixture pipeline with clean language separation:

1. **Python fixture generator** creates a deterministic model, exports GGUF, runs
   forward pass, and writes reference logits.
2. **Rust integration test** loads the GGUF, runs the same input through candle,
   and compares logit vectors.

No cross-language runtime calls. Python produces files, Rust reads them.

## Comparison Strategy

**Single-token comparison:** Feed the full input sequence through both models.
Extract last-position logits from PyTorch (`logits[0, -1, :]`). Candle's
`forward()` already returns last-position logits when given a full sequence on
a fresh cache. Compare these two `[vocab_size]` vectors.

If last-position logits match, intermediate positions would too -- same weights,
same math, same input.

## Tolerance

- **Primary check:** Max absolute difference < 1e-5 across all logit values.
  The tiny model (4 layers, 32 hidden dim) has minimal f32 accumulation, so
  this strict threshold should hold.
- **Diagnostic:** Cosine similarity between the two logit vectors, printed but
  not asserted. Provides a quick "how close?" signal. Expected to be ~1.0 when
  the absolute check passes.

## Python Fixture Generator

New file: `training/ct87/generate_validation_fixtures.py`

### Flow

1. `torch.manual_seed(42)` -- deterministic initialization.
2. Create `HarmonyModel(MICRO_CONFIG)` -- same micro config as existing test
   fixtures (4 layers, 32 hidden, 128 vocab, `tie_embeddings=True`).
3. Export to GGUF via `export_gguf()` --> `validation_model.gguf`.
4. Hardcoded input tokens: `[1, 5, 12, 3, 7, 42, 18, 100]` -- 8 tokens, all
   within vocab_size=128. Hardcoded for inspectability and reproducibility.
5. `model.eval()` + `torch.no_grad()` -- disable dropout/grad tracking.
6. Forward pass --> `logits[0, -1, :]` --> 128 float values.
7. Write `validation_reference.json`:
   ```json
   {
     "input_tokens": [1, 5, 12, 3, 7, 42, 18, 100],
     "last_logits": [0.123, -0.456, ...]
   }
   ```

### CLI

```
python3 -m ct87.generate_validation_fixtures
```

No arguments. Fully deterministic. Outputs to `crates/harmony-inference/tests/fixtures/`.

### Key Detail

The model is constructed and exported in the same Python process that produces
the reference logits, guaranteeing the weights in the GGUF and the logits
correspond to exactly the same model instance.

## Rust Integration Test

New file: `crates/harmony-inference/tests/validate_candle.rs`

### Flow

1. Load `validation_model.gguf` via `std::fs::read` with `CARGO_MANIFEST_DIR`
   (avoids embedding ~168KB into the test binary).
2. Parse `validation_reference.json` via `serde_json` into:
   ```rust
   struct ValidationReference {
       input_tokens: Vec<u32>,
       last_logits: Vec<f64>,
   }
   ```
   Logits stored as f64 in JSON to preserve full precision from Python's
   `json.dumps` (values originate from f32 PyTorch tensors).
3. `gguf_file::Content::read()` --> `HarmonyModel::from_gguf()` on CPU.
4. Create fresh `InferenceCache` --> `model.forward(&input_tensor, &mut cache, None)`.
5. **Absolute diff check:** For each logit index, assert `|rust - python| < 1e-5`.
   On failure, print which index diverged and both values.
6. **Cosine similarity:** Compute and print `dot(rust, python) / (norm(rust) * norm(python))`.
   Diagnostic only, not asserted.

### Dependencies

`serde` (with `derive`) and `serde_json` added to `[dev-dependencies]` in
`Cargo.toml`. Test-only -- no production dependency changes.

### Direct Model Access

The test uses `HarmonyModel::from_gguf()` and calls `forward()` directly with
a manually-constructed `InferenceCache`, bypassing `HarmonyEngine`. This keeps
the test focused on weight/math parity rather than engine plumbing (already
tested in Phase 0g).

## Fixture Files

All generated fixtures land in `crates/harmony-inference/tests/fixtures/`:

| File | Size | Contents |
|---|---|---|
| `validation_model.gguf` | ~168KB | Deterministic tiny model weights (tied embeddings) |
| `validation_reference.json` | ~2KB | Input tokens + last-position logits |

These are committed alongside the existing `tiny_harmony.gguf` and
`tiny_harmony_untied.gguf` fixtures from Phase 0g.

## Files Changed

| File | Action | ~Lines |
|---|---|---|
| `training/ct87/generate_validation_fixtures.py` | Create | 50 |
| `crates/harmony-inference/tests/validate_candle.rs` | Create | 80 |
| `crates/harmony-inference/tests/fixtures/validation_model.gguf` | Create | binary |
| `crates/harmony-inference/tests/fixtures/validation_reference.json` | Create | ~2KB |
| `crates/harmony-inference/Cargo.toml` | Modify | +2 lines |

## Testing

The Rust integration test (`validate_candle.rs`) IS the test for this phase.
It runs as part of `cargo test` in the harmony-inference crate. No additional
test files needed.

The Python fixture generator is validated indirectly: if it produces wrong
logits, the Rust test fails. The existing `test_export_gguf.py` tests already
validate GGUF export correctness.

## Scope Boundary

**In scope:**
- Python fixture generator (GGUF + reference logits)
- Rust integration test (load + forward + compare)
- `serde`/`serde_json` dev-dependencies
- Committed fixture files

**Out of scope:**
- Untied embedding parity (structurally tested in 0g)
- Multi-step KV cache validation (inference behavior, not weight parity)
- Engram injection parity (no weights yet)
- UQ head parity (no weights yet)
- Target config validation (tiny is sufficient proof)
- Quantized weight validation (0g is f32-only)
- Trained checkpoint validation (no checkpoints exist yet)
