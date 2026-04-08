# Phase 0h: Candle Validation Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove that PyTorch and candle produce identical logits from the same GGUF weights, closing out Phase 0.

**Architecture:** A Python fixture generator creates a deterministic tiny model, exports to GGUF, runs a forward pass, and writes reference logits as JSON. A Rust integration test loads that GGUF, runs the same input through candle, and asserts the logit vectors match within 1e-5 absolute tolerance, also reporting cosine similarity.

**Tech Stack:** Python (PyTorch, gguf, json), Rust (candle, serde, serde_json)

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `training/ct87/generate_validation_fixtures.py` | Create | Python script: build deterministic tiny model, export GGUF, run forward, write reference logits JSON |
| `crates/harmony-inference/tests/validate_candle.rs` | Create | Rust integration test: load GGUF, run forward, compare logits |
| `crates/harmony-inference/tests/fixtures/validation_model.gguf` | Create (binary) | GGUF weights from deterministic tiny model |
| `crates/harmony-inference/tests/fixtures/validation_reference.json` | Create | Input tokens + expected last-position logits |
| `crates/harmony-inference/Cargo.toml` | Modify | Add serde + serde_json to dev-dependencies |

---

### Task 1: Python Fixture Generator + Fixtures

**Files:**
- Create: `training/ct87/generate_validation_fixtures.py`
- Create: `crates/harmony-inference/tests/fixtures/validation_model.gguf`
- Create: `crates/harmony-inference/tests/fixtures/validation_reference.json`

**Context:** This script reuses the existing `MICRO_CONFIG` pattern from `training/ct87/generate_test_fixtures.py` and the `export_gguf()` function from `training/ct87/export_gguf.py`. It creates the same tiny model (4 layers, 32 hidden, 128 vocab, tied embeddings) used for the Phase 0g GGUF test fixtures, but additionally runs a forward pass and saves reference logits.

The Python model's `forward()` returns `[batch, seq_len, vocab_size]` logits for all positions. We extract `logits[0, -1, :]` (last position) because the Rust model's `forward()` only returns last-position logits `[1, vocab_size]`.

- [ ] **Step 1: Create the fixture generator script**

Create `training/ct87/generate_validation_fixtures.py`:

```python
"""Generate validation fixtures for Phase 0h candle validation loop.

Creates a deterministic tiny model, exports to GGUF, runs a forward pass,
and writes reference logits as JSON. The Rust integration test in
crates/harmony-inference/tests/validate_candle.rs loads these fixtures
and verifies candle produces identical logits.

Run from the training/ directory:
    python3 -m ct87.generate_validation_fixtures
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from ct87.export_gguf import export_gguf
from ct87.model import HarmonyModel, HarmonyModelConfig

# Micro config matching test_config() in harmony_model.rs and MICRO_CONFIG
# in generate_test_fixtures.py.
MICRO_CONFIG = HarmonyModelConfig(
    num_layers=4,
    hidden_dim=32,
    num_query_heads=4,
    num_kv_heads=2,
    head_dim=8,
    ffn_dim=64,
    vocab_size=128,
    max_seq_len=64,
    rope_theta=10000.0,
    rms_norm_eps=1e-6,
    layers_per_block=2,
    engram_injection_layer=1,
    engram_dim=16,
    tie_embeddings=True,
)

# Hardcoded input tokens -- all within vocab_size=128, chosen for
# inspectability and reproducibility without seed management.
INPUT_TOKENS = [1, 5, 12, 3, 7, 42, 18, 100]

FIXTURE_DIR = (
    Path(__file__).resolve().parents[2]
    / "crates"
    / "harmony-inference"
    / "tests"
    / "fixtures"
)


def main() -> None:
    os.makedirs(FIXTURE_DIR, exist_ok=True)

    # Deterministic model construction
    torch.manual_seed(42)
    model = HarmonyModel(MICRO_CONFIG)

    # Export GGUF
    gguf_path = FIXTURE_DIR / "validation_model.gguf"
    export_gguf(model.state_dict(), MICRO_CONFIG, gguf_path, "validation-micro")
    print(f"Generated {gguf_path}")

    # Forward pass -- extract last-position logits
    model.eval()
    input_ids = torch.tensor([INPUT_TOKENS], dtype=torch.long)  # [1, 8]
    with torch.no_grad():
        logits = model(input_ids)  # [1, 8, 128]
    last_logits = logits[0, -1, :].tolist()  # 128 float values

    # Write reference JSON
    reference = {
        "input_tokens": INPUT_TOKENS,
        "last_logits": last_logits,
    }
    json_path = FIXTURE_DIR / "validation_reference.json"
    with open(json_path, "w") as f:
        json.dump(reference, f, indent=2)
    print(f"Generated {json_path}")
    print(f"Logit vector length: {len(last_logits)}")
    print("Done!")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the fixture generator**

Run from the repo root:

```bash
cd training
python3 -m ct87.generate_validation_fixtures
```

Expected output:

```
Generated /path/to/crates/harmony-inference/tests/fixtures/validation_model.gguf
Generated /path/to/crates/harmony-inference/tests/fixtures/validation_reference.json
Logit vector length: 128
Done!
```

- [ ] **Step 3: Verify the generated fixtures**

Check that both files exist and the JSON is well-formed:

```bash
ls -la crates/harmony-inference/tests/fixtures/validation_model.gguf
```

```bash
ls -la crates/harmony-inference/tests/fixtures/validation_reference.json
```

```bash
python3 -c "
import json
d = json.load(open('crates/harmony-inference/tests/fixtures/validation_reference.json'))
print(f'tokens: {len(d[\"input_tokens\"])}, logits: {len(d[\"last_logits\"])}')
"
```

Expected: GGUF file ~168KB, JSON file ~2KB, output `tokens: 8, logits: 128`.

- [ ] **Step 4: Commit**

```bash
git add training/ct87/generate_validation_fixtures.py
git add crates/harmony-inference/tests/fixtures/validation_model.gguf
git add crates/harmony-inference/tests/fixtures/validation_reference.json
git commit -m "feat: add Python validation fixture generator for Phase 0h

Generates deterministic tiny model GGUF + reference logits JSON
for cross-runtime validation between PyTorch and candle."
```

---

### Task 2: Rust Integration Test

**Files:**
- Create: `crates/harmony-inference/tests/validate_candle.rs`
- Modify: `crates/harmony-inference/Cargo.toml`

**Context:** This integration test loads the fixtures generated by Task 1 and verifies that candle produces the same logits as PyTorch. It uses `HarmonyModel::from_gguf()` (implemented in Phase 0g) and the model's `forward()` method directly, bypassing `HarmonyEngine`.

The Rust `forward()` signature is:
```rust
pub fn forward(
    &self,
    input: &Tensor,           // [1, seq_len] u32 tensor
    cache: &mut InferenceCache,
    engram_fn: Option<EngramFn<'_>>,
) -> Result<HarmonyForwardOutput>
```

It returns `HarmonyForwardOutput { logits: Tensor, layer_norms: Vec<f32> }` where `logits` shape is `[1, vocab_size]` (last-position only). We compare this against the Python `last_logits` from the reference JSON.

`InferenceCache::new(num_layers, head_dim, num_kv_heads)` creates a fresh empty cache. The config values come from the loaded model via `model.config()`.

- [ ] **Step 1: Add dev-dependencies to Cargo.toml**

In `crates/harmony-inference/Cargo.toml`, add a `[dev-dependencies]` section after the existing `[dependencies]` section:

```toml
[dev-dependencies]
serde = { workspace = true, features = ["derive"] }
serde_json = { workspace = true }
```

**Important:** Check that `serde_json` is in the workspace root `Cargo.toml` under `[workspace.dependencies]`. If it is not there yet, add it:

```toml
serde_json = "1"
```

(`serde` is already a workspace dependency since it's used as an optional dep in harmony-inference.)

- [ ] **Step 2: Verify Cargo.toml change compiles**

```bash
cargo check -p harmony-inference --tests
```

Expected: compiles with no errors.

- [ ] **Step 3: Write the integration test**

Create `crates/harmony-inference/tests/validate_candle.rs`:

```rust
//! Phase 0h: Candle Validation Loop
//!
//! Verifies that candle produces the same logits as PyTorch for a
//! deterministic tiny model exported to GGUF.
//!
//! Fixtures generated by: python3 -m ct87.generate_validation_fixtures
//! (run from the training/ directory)

use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use harmony_inference::{HarmonyModel, InferenceCache};
use serde::Deserialize;

#[derive(Deserialize)]
struct ValidationReference {
    input_tokens: Vec<u32>,
    last_logits: Vec<f64>,
}

/// Load the validation GGUF fixture.
fn load_validation_gguf() -> Vec<u8> {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/validation_model.gguf");
    std::fs::read(&path).expect(
        "validation fixture not found; \
         run `python3 -m ct87.generate_validation_fixtures` from training/",
    )
}

/// Load the validation reference JSON fixture.
fn load_validation_reference() -> ValidationReference {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/validation_reference.json");
    let data = std::fs::read_to_string(&path).expect(
        "validation reference not found; \
         run `python3 -m ct87.generate_validation_fixtures` from training/",
    );
    serde_json::from_str(&data).expect("failed to parse validation_reference.json")
}

/// Compute cosine similarity between two f64 slices.
fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

#[test]
fn candle_matches_pytorch_logits() {
    // Load fixtures
    let gguf_data = load_validation_gguf();
    let reference = load_validation_reference();

    // Parse GGUF and load model
    let mut cursor = std::io::Cursor::new(&gguf_data);
    let content =
        gguf_file::Content::read(&mut cursor).expect("failed to parse GGUF");
    let model = HarmonyModel::from_gguf(&content, &mut cursor, &Device::Cpu)
        .expect("failed to load model");

    let config = model.config();
    assert_eq!(
        reference.last_logits.len(),
        config.vocab_size,
        "reference logit count must match vocab_size"
    );

    // Prepare input tensor: [1, seq_len] u32
    let input =
        Tensor::new(reference.input_tokens.as_slice(), &Device::Cpu)
            .expect("failed to create input tensor")
            .reshape((1, reference.input_tokens.len()))
            .expect("failed to reshape input");

    // Forward pass with fresh cache, no engram
    let mut cache = InferenceCache::new(
        config.num_layers,
        config.head_dim,
        config.num_kv_heads,
    );
    let output = model
        .forward(&input, &mut cache, None)
        .expect("forward pass failed");

    // Extract rust logits as f64 for comparison
    let rust_logits_f32: Vec<f32> = output
        .logits
        .flatten_all()
        .expect("flatten failed")
        .to_vec1()
        .expect("to_vec1 failed");
    let rust_logits: Vec<f64> =
        rust_logits_f32.iter().map(|&v| v as f64).collect();

    assert_eq!(
        rust_logits.len(),
        reference.last_logits.len(),
        "logit vector lengths must match"
    );

    // Absolute difference check: max |rust - python| < 1e-5
    let mut max_diff: f64 = 0.0;
    let mut max_diff_idx: usize = 0;
    for (i, (r, p)) in rust_logits
        .iter()
        .zip(reference.last_logits.iter())
        .enumerate()
    {
        let diff = (r - p).abs();
        if diff > max_diff {
            max_diff = diff;
            max_diff_idx = i;
        }
    }

    // Cosine similarity (diagnostic)
    let cos_sim =
        cosine_similarity(&rust_logits, &reference.last_logits);
    println!("Phase 0h validation results:");
    println!(
        "  max absolute diff: {max_diff:.2e} at index {max_diff_idx}"
    );
    println!("  cosine similarity: {cos_sim:.10}");
    println!(
        "  rust logit[{max_diff_idx}]: {:.8}, \
         python logit[{max_diff_idx}]: {:.8}",
        rust_logits[max_diff_idx],
        reference.last_logits[max_diff_idx]
    );

    assert!(
        max_diff < 1e-5,
        "logit mismatch exceeds tolerance: \
         max_diff={max_diff:.2e} at index {max_diff_idx}, \
         rust={:.8}, python={:.8}, \
         cosine_similarity={cos_sim:.10}",
        rust_logits[max_diff_idx],
        reference.last_logits[max_diff_idx],
    );
}
```

- [ ] **Step 4: Run the integration test**

```bash
cargo test -p harmony-inference --test validate_candle -- --nocapture
```

Expected output (values will vary):

```
Phase 0h validation results:
  max absolute diff: X.XXe-XX at index NN
  cosine similarity: 0.99999XXXXX
...
test candle_matches_pytorch_logits ... ok
```

If the test fails with a logit mismatch, the error message includes the index, both values, and cosine similarity for debugging.

- [ ] **Step 5: Run the full test suite to verify no regressions**

```bash
cargo test -p harmony-inference
```

Expected: all existing tests + the new `candle_matches_pytorch_logits` test pass.

Also run the Python tests:

```bash
cd training
python3 -m pytest tests/ -v
```

Expected: all Python tests pass (no changes to Python test files).

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-inference/Cargo.toml
git add crates/harmony-inference/tests/validate_candle.rs
git commit -m "feat: add Rust integration test for Phase 0h candle validation

Loads GGUF + reference logits from Python fixtures and asserts
candle forward pass matches PyTorch within 1e-5 absolute tolerance.
Reports cosine similarity as diagnostic."
```
