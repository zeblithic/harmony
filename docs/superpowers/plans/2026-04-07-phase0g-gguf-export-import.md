# Phase 0g: GGUF Export/Import Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bridge PyTorch training and candle inference by exporting safetensors checkpoints to GGUF and loading them in `HarmonyModel`/`HarmonyEngine`.

**Architecture:** Python exporter converts `state_dict` keys to GGUF tensor names per a deterministic mapping, writing f32 tensors and harmony-namespaced metadata. Rust loader reads GGUF metadata to reconstruct `HarmonyModelConfig`, loads tensors to build `HarmonyModel` via a new `from_gguf()` constructor. `HarmonyEngine::load_gguf()` is wired up to replace its "not yet implemented" placeholder.

**Tech Stack:** Python (`gguf`, `safetensors`, `torch`, `numpy`), Rust (`candle_core::quantized::gguf_file`, `candle_nn`)

**Spec:** `docs/superpowers/specs/2026-04-07-phase0g-gguf-export-import-design.md`

**Note:** This plan adds `harmony.engram_injection_layer` (u32) and `harmony.engram_dim` (u32) to the GGUF metadata beyond what the spec listed, because `HarmonyModelConfig` requires these fields and the GGUF should be self-describing.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `training/pyproject.toml` | Modify | Add `gguf>=0.6` dependency |
| `training/ct87/export_gguf.py` | Create | GGUF exporter: naming map, metadata writer, export function, CLI |
| `training/ct87/generate_test_fixtures.py` | Create | Generate deterministic GGUF fixtures for Rust tests |
| `training/tests/test_export_gguf.py` | Create | Python exporter tests |
| `crates/harmony-inference/tests/fixtures/tiny_harmony.gguf` | Create | Tied-embedding GGUF fixture for Rust tests |
| `crates/harmony-inference/tests/fixtures/tiny_harmony_untied.gguf` | Create | Untied-embedding GGUF fixture for Rust tests |
| `crates/harmony-inference/src/harmony_model.rs` | Modify | Add `from_gguf()` to `Mlp`, `Attention`, `TransformerLayer`, `HarmonyModel` |
| `crates/harmony-inference/src/harmony_engine.rs` | Modify | Wire up `HarmonyEngine::load_gguf()` |
| `docs/superpowers/specs/2026-04-07-phase0g-gguf-export-import-design.md` | Modify | Add engram metadata fields |

---

### Task 1: Python GGUF Exporter + Tests + Fixtures

**Files:**
- Modify: `training/pyproject.toml`
- Modify: `docs/superpowers/specs/2026-04-07-phase0g-gguf-export-import-design.md`
- Create: `training/ct87/export_gguf.py`
- Create: `training/tests/test_export_gguf.py`
- Create: `training/ct87/generate_test_fixtures.py`
- Create: `crates/harmony-inference/tests/fixtures/tiny_harmony.gguf`
- Create: `crates/harmony-inference/tests/fixtures/tiny_harmony_untied.gguf`

- [ ] **Step 1: Add `gguf` dependency to pyproject.toml**

In `training/pyproject.toml`, add `"gguf>=0.6"` to the dependencies list:

```toml
[project]
name = "ct87"
version = "0.1.0"
description = "PyTorch training scaffold for the ct87 custom model"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.2",
    "safetensors>=0.4",
    "datasets>=2.16",
    "gguf>=0.6",
]
```

Install the updated dependencies:

Run: `cd training && pip install -e '.[dev]'`

- [ ] **Step 2: Update spec with engram metadata fields**

In `docs/superpowers/specs/2026-04-07-phase0g-gguf-export-import-design.md`, add two rows to the GGUF Metadata table, after `harmony.tie_embeddings`:

```
| `harmony.engram_injection_layer` | u32 | 2 | `config.engram_injection_layer` |
| `harmony.engram_dim` | u32 | 256 | `config.engram_dim` |
```

- [ ] **Step 3: Write the failing tests**

Create `training/tests/test_export_gguf.py`:

```python
"""Tests for the GGUF exporter."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from ct87.export_gguf import build_naming_map, export_gguf
from ct87.model import HarmonyModel, HarmonyModelConfig


def _test_config(tie: bool = True) -> HarmonyModelConfig:
    """Micro config matching Rust test_config() in harmony_model.rs."""
    return HarmonyModelConfig(
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
        tie_embeddings=tie,
    )


# ---- Naming map tests ----


class TestBuildNamingMap:
    def test_tied_excludes_output_weight(self):
        config = _test_config(tie=True)
        nm = build_naming_map(config)
        assert "lm_head.weight" not in nm
        assert "output.weight" not in nm.values()

    def test_untied_includes_output_weight(self):
        config = _test_config(tie=False)
        nm = build_naming_map(config)
        assert "lm_head.weight" in nm
        assert nm["lm_head.weight"] == "output.weight"

    def test_layer_count_matches(self):
        config = _test_config()
        nm = build_naming_map(config)
        layer_keys = [k for k in nm if k.startswith("layers.")]
        # 11 tensor keys per transformer layer
        assert len(layer_keys) == config.num_layers * 11

    def test_block_attnres_query_count(self):
        config = _test_config()
        nm = build_naming_map(config)
        query_keys = [k for k in nm if "block_attnres" in k]
        # 2 blocks -> 1 boundary query
        assert len(query_keys) == config.num_blocks - 1

    def test_embed_and_norm_present(self):
        config = _test_config()
        nm = build_naming_map(config)
        assert nm["embed_tokens.weight"] == "token_embd.weight"
        assert nm["final_norm.weight"] == "output_norm.weight"


# ---- Export tests ----


class TestExportGguf:
    def test_roundtrip_tensor_values(self):
        """Export model to GGUF and read back -- verify tensor values match."""
        from gguf import GGUFReader

        config = _test_config(tie=True)
        torch.manual_seed(42)
        model = HarmonyModel(config)
        state_dict = model.state_dict()

        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_path = Path(tmpdir) / "test.gguf"
            export_gguf(state_dict, config, gguf_path, name="test")

            reader = GGUFReader(str(gguf_path))
            gguf_tensors = {t.name: t for t in reader.tensors}

            naming_map = build_naming_map(config)
            for pytorch_key, gguf_name in naming_map.items():
                assert gguf_name in gguf_tensors, f"Missing: {gguf_name}"
                original = state_dict[pytorch_key].detach().float().numpy()
                if "block_attnres.queries" in pytorch_key:
                    original = original.squeeze()
                loaded = np.array(gguf_tensors[gguf_name].data)
                np.testing.assert_allclose(
                    loaded.reshape(original.shape),
                    original,
                    rtol=1e-6,
                    err_msg=f"Mismatch for {gguf_name}",
                )

    def test_metadata_present(self):
        """Verify all metadata keys are present."""
        from gguf import GGUFReader

        config = _test_config()
        torch.manual_seed(42)
        model = HarmonyModel(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_path = Path(tmpdir) / "test.gguf"
            export_gguf(model.state_dict(), config, gguf_path, name="test-model")

            reader = GGUFReader(str(gguf_path))
            fields = reader.fields

            expected_keys = [
                "general.architecture",
                "general.name",
                "harmony.block_count",
                "harmony.embedding_length",
                "harmony.attention.head_count",
                "harmony.attention.head_count_kv",
                "harmony.attention.key_length",
                "harmony.feed_forward_length",
                "harmony.context_length",
                "harmony.rope.freq_base",
                "harmony.attention.layer_norm_rms_epsilon",
                "harmony.vocab_size",
                "harmony.layers_per_block",
                "harmony.tie_embeddings",
                "harmony.engram_injection_layer",
                "harmony.engram_dim",
            ]
            for key in expected_keys:
                assert key in fields, f"Missing metadata key: {key}"

    def test_tied_skips_output_weight(self):
        """When tie_embeddings=True, output.weight absent from GGUF."""
        from gguf import GGUFReader

        config = _test_config(tie=True)
        torch.manual_seed(42)
        model = HarmonyModel(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_path = Path(tmpdir) / "test.gguf"
            export_gguf(model.state_dict(), config, gguf_path)

            reader = GGUFReader(str(gguf_path))
            names = {t.name for t in reader.tensors}
            assert "output.weight" not in names
            assert "token_embd.weight" in names

    def test_untied_includes_output_weight(self):
        """When tie_embeddings=False, output.weight present in GGUF."""
        from gguf import GGUFReader

        config = _test_config(tie=False)
        torch.manual_seed(42)
        model = HarmonyModel(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_path = Path(tmpdir) / "test.gguf"
            export_gguf(model.state_dict(), config, gguf_path)

            reader = GGUFReader(str(gguf_path))
            names = {t.name for t in reader.tensors}
            assert "output.weight" in names
            assert "token_embd.weight" in names

    def test_missing_keys_raises(self):
        """Exporter rejects state_dict with missing keys."""
        config = _test_config()
        state_dict = {"embed_tokens.weight": torch.zeros(128, 32)}

        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_path = Path(tmpdir) / "test.gguf"
            with pytest.raises(ValueError, match="Missing keys"):
                export_gguf(state_dict, config, gguf_path)

    def test_extra_keys_raises(self):
        """Exporter rejects state_dict with extra keys."""
        config = _test_config()
        torch.manual_seed(42)
        model = HarmonyModel(config)
        state_dict = model.state_dict()
        state_dict["spurious.weight"] = torch.zeros(10)

        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_path = Path(tmpdir) / "test.gguf"
            with pytest.raises(ValueError, match="Extra keys"):
                export_gguf(state_dict, config, gguf_path)
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `cd training && python -m pytest tests/test_export_gguf.py -v`
Expected: FAIL (ImportError — `ct87.export_gguf` does not exist yet)

- [ ] **Step 5: Implement the exporter**

Create `training/ct87/export_gguf.py`:

```python
"""GGUF exporter for ct87 -- converts safetensors checkpoints to GGUF format.

Usage:
    python -m ct87.export_gguf --checkpoint model.safetensors --config tiny --output model.gguf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from gguf import GGUFWriter
from safetensors.torch import load_file

from ct87.model import HarmonyModelConfig


def build_naming_map(config: HarmonyModelConfig) -> dict[str, str]:
    """Build the PyTorch state_dict key -> GGUF tensor name mapping.

    Returns a dict where keys are PyTorch state_dict keys and values are
    the corresponding GGUF tensor names. When tie_embeddings is True,
    lm_head.weight is excluded (the loader falls back to token_embd.weight).
    """
    mapping: dict[str, str] = {}
    mapping["embed_tokens.weight"] = "token_embd.weight"

    for i in range(config.num_layers):
        mapping[f"layers.{i}.attn_norm.weight"] = f"blk.{i}.attn_norm.weight"
        mapping[f"layers.{i}.attn.q_proj.weight"] = f"blk.{i}.attn_q.weight"
        mapping[f"layers.{i}.attn.k_proj.weight"] = f"blk.{i}.attn_k.weight"
        mapping[f"layers.{i}.attn.v_proj.weight"] = f"blk.{i}.attn_v.weight"
        mapping[f"layers.{i}.attn.o_proj.weight"] = f"blk.{i}.attn_output.weight"
        mapping[f"layers.{i}.attn.q_norm.weight"] = f"blk.{i}.attn_q_norm.weight"
        mapping[f"layers.{i}.attn.k_norm.weight"] = f"blk.{i}.attn_k_norm.weight"
        mapping[f"layers.{i}.ffn_norm.weight"] = f"blk.{i}.ffn_norm.weight"
        mapping[f"layers.{i}.mlp.gate_proj.weight"] = f"blk.{i}.ffn_gate.weight"
        mapping[f"layers.{i}.mlp.up_proj.weight"] = f"blk.{i}.ffn_up.weight"
        mapping[f"layers.{i}.mlp.down_proj.weight"] = f"blk.{i}.ffn_down.weight"

    mapping["final_norm.weight"] = "output_norm.weight"

    if not config.tie_embeddings:
        mapping["lm_head.weight"] = "output.weight"

    for i in range(config.num_blocks - 1):
        mapping[f"block_attnres.queries.{i}"] = (
            f"harmony.block_attnres.query.{i}.weight"
        )

    return mapping


def write_metadata(
    writer: GGUFWriter, config: HarmonyModelConfig, name: str,
) -> None:
    """Write all harmony GGUF metadata keys."""
    writer.add_name(name)
    writer.add_uint32("harmony.block_count", config.num_layers)
    writer.add_uint32("harmony.embedding_length", config.hidden_dim)
    writer.add_uint32("harmony.attention.head_count", config.num_query_heads)
    writer.add_uint32("harmony.attention.head_count_kv", config.num_kv_heads)
    writer.add_uint32("harmony.attention.key_length", config.head_dim)
    writer.add_uint32("harmony.feed_forward_length", config.ffn_dim)
    writer.add_uint32("harmony.context_length", config.max_seq_len)
    writer.add_float32("harmony.rope.freq_base", config.rope_theta)
    writer.add_float32(
        "harmony.attention.layer_norm_rms_epsilon", config.rms_norm_eps,
    )
    writer.add_uint32("harmony.vocab_size", config.vocab_size)
    writer.add_uint32("harmony.layers_per_block", config.layers_per_block)
    writer.add_bool("harmony.tie_embeddings", config.tie_embeddings)
    writer.add_uint32(
        "harmony.engram_injection_layer", config.engram_injection_layer,
    )
    writer.add_uint32("harmony.engram_dim", config.engram_dim)


def export_gguf(
    state_dict: dict[str, torch.Tensor],
    config: HarmonyModelConfig,
    output_path: str | Path,
    name: str | None = None,
) -> None:
    """Export a ct87 state_dict to GGUF format.

    Args:
        state_dict: Model state_dict (from safetensors or model.state_dict()).
        config: Model configuration matching the checkpoint.
        output_path: Path to write the GGUF file.
        name: Optional model name for metadata. Defaults to "ct87".

    Raises:
        ValueError: If state_dict keys don't match expected keys for config.
    """
    if name is None:
        name = "ct87"

    naming_map = build_naming_map(config)

    # Validate completeness
    expected_keys = set(naming_map.keys())
    actual_keys = set(state_dict.keys())

    # When tied, lm_head.weight exists in state_dict but isn't in naming_map
    if config.tie_embeddings and "lm_head.weight" in actual_keys:
        actual_keys = actual_keys - {"lm_head.weight"}

    missing = expected_keys - actual_keys
    extra = actual_keys - expected_keys
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"Missing keys: {sorted(missing)}")
        if extra:
            parts.append(f"Extra keys: {sorted(extra)}")
        raise ValueError("; ".join(parts))

    writer = GGUFWriter(str(output_path), arch="harmony")
    write_metadata(writer, config, name)

    for pytorch_key, gguf_name in naming_map.items():
        tensor = state_dict[pytorch_key]
        # BlockAttnRes queries: squeeze from [1, 1, hidden] to [hidden]
        if "block_attnres.queries" in pytorch_key:
            tensor = tensor.squeeze()
        arr = tensor.detach().cpu().float().numpy()
        writer.add_tensor(gguf_name, arr)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ct87 checkpoint to GGUF")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to safetensors checkpoint",
    )
    parser.add_argument("--config", choices=["tiny", "target"], required=True)
    parser.add_argument("--output", type=str, required=True, help="Output GGUF path")
    parser.add_argument(
        "--name", type=str, default=None, help="Model name for metadata",
    )
    args = parser.parse_args()

    config = (
        HarmonyModelConfig.tiny()
        if args.config == "tiny"
        else HarmonyModelConfig.target()
    )
    state_dict = load_file(args.checkpoint)
    name = args.name or f"ct87-{args.config}"
    export_gguf(state_dict, config, args.output, name)
    print(f"Exported to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd training && python -m pytest tests/test_export_gguf.py -v`
Expected: All 12 tests PASS

- [ ] **Step 7: Generate Rust test fixtures**

Create `training/ct87/generate_test_fixtures.py`:

```python
"""Generate GGUF test fixtures for Rust integration tests.

Run from the training/ directory:
    python -m ct87.generate_test_fixtures
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import torch

from ct87.export_gguf import export_gguf
from ct87.model import HarmonyModel, HarmonyModelConfig

# Micro config matching test_config() in harmony_model.rs
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

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "crates" / "harmony-inference" / "tests" / "fixtures"


def generate(tie: bool, filename: str, name: str) -> None:
    torch.manual_seed(42)
    config = replace(MICRO_CONFIG, tie_embeddings=tie)
    model = HarmonyModel(config)
    export_gguf(model.state_dict(), config, FIXTURE_DIR / filename, name)
    print(f"Generated {FIXTURE_DIR / filename}")


def main() -> None:
    os.makedirs(FIXTURE_DIR, exist_ok=True)
    generate(tie=True, filename="tiny_harmony.gguf", name="test-tied")
    generate(tie=False, filename="tiny_harmony_untied.gguf", name="test-untied")
    print("Done!")


if __name__ == "__main__":
    main()
```

Run from the `training/` directory:

Run: `cd training && python -m ct87.generate_test_fixtures`
Expected: Two files created:
- `crates/harmony-inference/tests/fixtures/tiny_harmony.gguf`
- `crates/harmony-inference/tests/fixtures/tiny_harmony_untied.gguf`

- [ ] **Step 8: Commit**

```bash
git add training/pyproject.toml
git add training/ct87/export_gguf.py
git add training/ct87/generate_test_fixtures.py
git add training/tests/test_export_gguf.py
git add crates/harmony-inference/tests/fixtures/tiny_harmony.gguf
git add crates/harmony-inference/tests/fixtures/tiny_harmony_untied.gguf
git add docs/superpowers/specs/2026-04-07-phase0g-gguf-export-import-design.md
git commit -m "feat(training): add GGUF exporter and test fixtures for Phase 0g"
```

---

### Task 2: Rust GGUF Loader + Engine Wiring + Tests

**Files:**
- Modify: `crates/harmony-inference/src/harmony_model.rs`
- Modify: `crates/harmony-inference/src/harmony_engine.rs`

**Reference files (read, not modified):**
- `crates/harmony-inference/src/qwen3_ext.rs` — existing GGUF loader pattern
- `crates/harmony-inference/src/block_attnres.rs` — `BlockAttnRes::from_tensors()`
- `crates/harmony-inference/src/lib.rs` — `InferenceEngine` trait, `InferenceCache`

- [ ] **Step 1: Write failing tests in harmony_model.rs**

Add these tests to the existing `#[cfg(test)] mod tests` block at the end of `crates/harmony-inference/src/harmony_model.rs`, after the existing tests:

```rust
    // -----------------------------------------------------------------------
    // GGUF loading tests (Task 2)
    // -----------------------------------------------------------------------

    fn load_test_gguf() -> Vec<u8> {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/tiny_harmony.gguf");
        std::fs::read(&path)
            .expect("test fixture not found; run `python -m ct87.generate_test_fixtures` from training/")
    }

    fn load_test_gguf_untied() -> Vec<u8> {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/tiny_harmony_untied.gguf");
        std::fs::read(&path)
            .expect("test fixture not found; run `python -m ct87.generate_test_fixtures` from training/")
    }

    #[test]
    fn from_gguf_loads_tied_model() {
        let data = load_test_gguf();
        let mut cursor = std::io::Cursor::new(&data);
        let content =
            candle_core::quantized::gguf_file::Content::read(&mut cursor).unwrap();
        let model =
            HarmonyModel::from_gguf(&content, &mut cursor, &Device::Cpu).unwrap();
        assert_eq!(model.config().num_layers, 4);
        assert_eq!(model.config().hidden_dim, 32);
        assert_eq!(model.config().num_query_heads, 4);
        assert_eq!(model.config().num_kv_heads, 2);
        assert_eq!(model.config().head_dim, 8);
        assert_eq!(model.config().ffn_dim, 64);
        assert_eq!(model.config().vocab_size, 128);
        assert!(model.config().tie_embeddings);
        assert_eq!(model.layers.len(), 4);
    }

    #[test]
    fn from_gguf_forward_produces_logits() {
        let data = load_test_gguf();
        let mut cursor = std::io::Cursor::new(&data);
        let content =
            candle_core::quantized::gguf_file::Content::read(&mut cursor).unwrap();
        let model =
            HarmonyModel::from_gguf(&content, &mut cursor, &Device::Cpu).unwrap();

        let cfg = model.config();
        let mut cache = InferenceCache::new(cfg.num_layers, cfg.head_dim, cfg.num_kv_heads);
        let input = Tensor::new(&[1u32, 2, 3], &Device::Cpu)
            .unwrap()
            .reshape((1, 3))
            .unwrap();
        let output = model.forward(&input, &mut cache, None).unwrap();
        assert_eq!(output.logits.dims(), &[1, cfg.vocab_size]);
        assert_eq!(output.layer_norms.len(), cfg.num_layers);
    }

    #[test]
    fn from_gguf_untied_model() {
        let data = load_test_gguf_untied();
        let mut cursor = std::io::Cursor::new(&data);
        let content =
            candle_core::quantized::gguf_file::Content::read(&mut cursor).unwrap();
        let model =
            HarmonyModel::from_gguf(&content, &mut cursor, &Device::Cpu).unwrap();
        assert!(!model.config().tie_embeddings);

        // Forward pass should still work
        let cfg = model.config();
        let mut cache = InferenceCache::new(cfg.num_layers, cfg.head_dim, cfg.num_kv_heads);
        let input = Tensor::new(&[1u32, 2, 3], &Device::Cpu)
            .unwrap()
            .reshape((1, 3))
            .unwrap();
        let output = model.forward(&input, &mut cache, None).unwrap();
        assert_eq!(output.logits.dims(), &[1, cfg.vocab_size]);
    }

    #[test]
    fn from_gguf_wrong_architecture_errors() {
        let data = load_test_gguf();
        let mut cursor = std::io::Cursor::new(&data);
        let mut content =
            candle_core::quantized::gguf_file::Content::read(&mut cursor).unwrap();
        // Overwrite the architecture metadata
        content.metadata.insert(
            "general.architecture".to_string(),
            candle_core::quantized::gguf_file::Value::String("wrong".to_string()),
        );
        cursor.set_position(0);
        let result = HarmonyModel::from_gguf(&content, &mut cursor, &Device::Cpu);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("harmony"),
            "error should mention 'harmony': {err}"
        );
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-inference from_gguf`
Expected: FAIL (no method named `from_gguf` found)

- [ ] **Step 3: Add imports and `load_tensor` helper**

At the top of `crates/harmony-inference/src/harmony_model.rs`, add these imports alongside the existing ones:

```rust
use candle_core::quantized::gguf_file;
use std::io::{Read, Seek};
```

Add the `load_tensor` helper function after the existing `random_rms_norm()` function (in the "Weight construction helpers" section around line 600):

```rust
/// Load a tensor from GGUF content and dequantize to f32.
fn load_tensor<R: Read + Seek>(
    ct: &gguf_file::Content,
    reader: &mut R,
    name: &str,
    device: &Device,
) -> Result<Tensor> {
    let qtensor = ct.tensor(reader, name, device)?;
    qtensor.dequantize(device)
}
```

- [ ] **Step 4: Add `from_gguf()` to `Mlp`, `Attention`, `TransformerLayer`**

Add to the `impl Mlp` block (after the existing `new()` method around line 187):

```rust
    fn from_gguf<R: Read + Seek>(
        ct: &gguf_file::Content,
        reader: &mut R,
        prefix: &str,
        device: &Device,
    ) -> Result<Self> {
        let gate_proj = Linear::new(
            load_tensor(ct, reader, &format!("{prefix}.ffn_gate.weight"), device)?,
            None,
        );
        let up_proj = Linear::new(
            load_tensor(ct, reader, &format!("{prefix}.ffn_up.weight"), device)?,
            None,
        );
        let down_proj = Linear::new(
            load_tensor(ct, reader, &format!("{prefix}.ffn_down.weight"), device)?,
            None,
        );
        Ok(Self { gate_proj, up_proj, down_proj })
    }
```

Add to the `impl Attention` block (after the existing `new()` method around line 255):

```rust
    fn from_gguf<R: Read + Seek>(
        ct: &gguf_file::Content,
        reader: &mut R,
        num_query_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
        rotary_emb: Arc<RotaryEmbedding>,
        prefix: &str,
        device: &Device,
    ) -> Result<Self> {
        let q_proj = Linear::new(
            load_tensor(ct, reader, &format!("{prefix}.attn_q.weight"), device)?,
            None,
        );
        let k_proj = Linear::new(
            load_tensor(ct, reader, &format!("{prefix}.attn_k.weight"), device)?,
            None,
        );
        let v_proj = Linear::new(
            load_tensor(ct, reader, &format!("{prefix}.attn_v.weight"), device)?,
            None,
        );
        let o_proj = Linear::new(
            load_tensor(ct, reader, &format!("{prefix}.attn_output.weight"), device)?,
            None,
        );

        let q_norm_w = load_tensor(ct, reader, &format!("{prefix}.attn_q_norm.weight"), device)?;
        let k_norm_w = load_tensor(ct, reader, &format!("{prefix}.attn_k_norm.weight"), device)?;
        let q_norm = RmsNorm::new(q_norm_w, rms_norm_eps);
        let k_norm = RmsNorm::new(k_norm_w, rms_norm_eps);

        let num_kv_groups = num_query_heads / num_kv_heads;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads: num_query_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            rotary_emb,
        })
    }
```

Add to the `impl TransformerLayer` block (after the existing `new()` method around line 355):

```rust
    fn from_gguf<R: Read + Seek>(
        ct: &gguf_file::Content,
        reader: &mut R,
        config: &HarmonyModelConfig,
        rotary_emb: Arc<RotaryEmbedding>,
        layer_idx: usize,
        device: &Device,
    ) -> Result<Self> {
        let prefix = format!("blk.{layer_idx}");

        let attn_norm_w = load_tensor(ct, reader, &format!("{prefix}.attn_norm.weight"), device)?;
        let attn_norm = RmsNorm::new(attn_norm_w, config.rms_norm_eps);

        let ffn_norm_w = load_tensor(ct, reader, &format!("{prefix}.ffn_norm.weight"), device)?;
        let ffn_norm = RmsNorm::new(ffn_norm_w, config.rms_norm_eps);

        let attn = Attention::from_gguf(
            ct, reader,
            config.num_query_heads, config.num_kv_heads,
            config.head_dim, config.rms_norm_eps,
            rotary_emb, &prefix, device,
        )?;

        let mlp = Mlp::from_gguf(ct, reader, &prefix, device)?;

        Ok(Self { attn_norm, attn, ffn_norm, mlp })
    }
```

- [ ] **Step 5: Add `HarmonyModel::from_gguf()`**

Add to the `impl HarmonyModel` block (after the existing `new()` method, around line 456):

```rust
    /// Load a HarmonyModel from GGUF data.
    ///
    /// Reads metadata to reconstruct [`HarmonyModelConfig`], then loads all
    /// tensors by their GGUF names. Verifies `general.architecture == "harmony"`.
    ///
    /// The GGUF must have been produced by `ct87.export_gguf` (Phase 0g Python
    /// exporter) or follow the same tensor naming and metadata conventions.
    pub fn from_gguf<R: Read + Seek>(
        ct: &gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        // Verify architecture
        let arch = match ct.metadata.get("general.architecture") {
            Some(gguf_file::Value::String(s)) => s.as_str(),
            _ => "",
        };
        if arch != "harmony" {
            candle_core::bail!("expected architecture 'harmony', got '{arch}'");
        }

        // Read metadata helper
        let md = |key: &str| -> Result<&gguf_file::Value> {
            ct.metadata.get(key).ok_or_else(|| {
                candle_core::Error::Msg(format!("missing GGUF metadata key: {key}"))
            })
        };

        let num_layers = md("harmony.block_count")?.to_u32()? as usize;
        let hidden_dim = md("harmony.embedding_length")?.to_u32()? as usize;
        let num_query_heads = md("harmony.attention.head_count")?.to_u32()? as usize;
        let num_kv_heads = md("harmony.attention.head_count_kv")?.to_u32()? as usize;
        let head_dim = md("harmony.attention.key_length")?.to_u32()? as usize;
        let ffn_dim = md("harmony.feed_forward_length")?.to_u32()? as usize;
        let vocab_size = md("harmony.vocab_size")?.to_u32()? as usize;
        let max_seq_len = md("harmony.context_length")?.to_u32()? as usize;
        let rope_theta = md("harmony.rope.freq_base")?.to_f32()? as f64;
        let rms_norm_eps =
            md("harmony.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let layers_per_block = md("harmony.layers_per_block")?.to_u32()? as usize;
        let tie_embeddings = md("harmony.tie_embeddings")?.to_bool()?;
        let engram_injection_layer =
            md("harmony.engram_injection_layer")?.to_u32()? as usize;
        let engram_dim = md("harmony.engram_dim")?.to_u32()? as usize;

        let config = HarmonyModelConfig {
            num_layers,
            hidden_dim,
            num_query_heads,
            num_kv_heads,
            head_dim,
            ffn_dim,
            vocab_size,
            max_seq_len,
            rope_theta,
            rms_norm_eps,
            layers_per_block,
            engram_injection_layer,
            engram_dim,
            tie_embeddings,
        };

        if layers_per_block == 0 || num_layers % layers_per_block != 0 {
            candle_core::bail!(
                "num_layers ({num_layers}) must be divisible by layers_per_block ({layers_per_block})"
            );
        }
        if engram_injection_layer >= num_layers {
            candle_core::bail!(
                "engram_injection_layer ({engram_injection_layer}) must be < num_layers ({num_layers})"
            );
        }

        // Embedding
        let embed_weight = load_tensor(ct, reader, "token_embd.weight", device)?;
        let embed_tokens = Embedding::new(embed_weight.clone(), hidden_dim);

        // Shared rotary embedding (recomputed from config, not stored in GGUF)
        let rotary_emb = Arc::new(RotaryEmbedding::new(
            head_dim, max_seq_len, rope_theta, device,
        )?);

        // Transformer layers
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            layers.push(TransformerLayer::from_gguf(
                ct, reader, &config, rotary_emb.clone(), i, device,
            )?);
        }

        // Final norm
        let final_norm_w = load_tensor(ct, reader, "output_norm.weight", device)?;
        let final_norm = RmsNorm::new(final_norm_w, rms_norm_eps);

        // lm_head: use output.weight if untied, else reuse embedding weight
        let lm_head = if tie_embeddings {
            Linear::new(embed_weight, None)
        } else {
            let w = load_tensor(ct, reader, "output.weight", device)?;
            Linear::new(w, None)
        };

        // BlockAttnRes
        let num_blocks = num_layers / layers_per_block;
        let bar_config = BlockAttnResConfig {
            num_blocks,
            layers_per_block,
            hidden_dim,
        };
        let mut query_tensors = Vec::with_capacity(num_blocks.saturating_sub(1));
        for i in 0..num_blocks.saturating_sub(1) {
            let name = format!("harmony.block_attnres.query.{i}.weight");
            let q = load_tensor(ct, reader, &name, device)?;
            // Reshape from [hidden_dim] to [1, 1, hidden_dim]
            let q = q.reshape((1, 1, hidden_dim))?;
            query_tensors.push(q);
        }
        let block_attnres = BlockAttnRes::from_tensors(&bar_config, query_tensors)?;

        Ok(Self {
            config,
            embed_tokens,
            layers,
            final_norm,
            lm_head,
            block_attnres,
            device: device.clone(),
        })
    }
```

- [ ] **Step 6: Run model tests to verify they pass**

Run: `cargo test -p harmony-inference from_gguf`
Expected: All 4 GGUF tests PASS

- [ ] **Step 7: Wire up `HarmonyEngine::load_gguf()`**

In `crates/harmony-inference/src/harmony_engine.rs`, add these imports at the top alongside existing ones:

```rust
use std::io::Cursor;
use candle_core::quantized::gguf_file;
```

Replace the `load_gguf` implementation in the `impl InferenceEngine for HarmonyEngine` block. The current code (around line 213) is:

```rust
    fn load_gguf(&mut self, _gguf_data: &[u8]) -> Result<(), InferenceError> {
        Err(InferenceError::InvalidGguf(
            "harmony GGUF loading not yet implemented (Phase 0g)".into(),
        ))
    }
```

Replace it with:

```rust
    fn load_gguf(&mut self, gguf_data: &[u8]) -> Result<(), InferenceError> {
        let mut cursor = Cursor::new(gguf_data);
        let content = gguf_file::Content::read(&mut cursor)
            .map_err(|e| InferenceError::InvalidGguf(e.to_string()))?;
        let model = HarmonyModel::from_gguf(&content, &mut cursor, &self.device)
            .map_err(|e| InferenceError::InvalidGguf(e.to_string()))?;
        self.config = model.config().clone();
        self.uq_feature_config = UqFeatureConfig::for_num_layers(self.config.num_layers);
        self.model = Some(model);
        Ok(())
    }
```

- [ ] **Step 8: Add engine loading test**

Add this test to the `#[cfg(test)] mod tests` block in `crates/harmony-inference/src/harmony_engine.rs`, after the existing tests:

```rust
    // -----------------------------------------------------------------------
    // GGUF loading tests (Task 2)
    // -----------------------------------------------------------------------

    #[test]
    fn load_gguf_creates_model() {
        let data = std::fs::read(
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/tiny_harmony.gguf"),
        )
        .expect("test fixture missing");

        // Use a dummy config — load_gguf will overwrite it
        let dummy_config = test_config();
        let mut engine = HarmonyEngine::new(dummy_config, Device::Cpu);
        engine.load_gguf(&data).expect("load_gguf should succeed");

        // Engine should now have a model with the fixture's config
        assert!(engine.model.is_some());
        let cache = engine.new_cache().unwrap();
        assert_eq!(cache.num_layers, 4);
        assert_eq!(cache.head_dim, 8);
        assert_eq!(cache.num_kv_heads, 2);
    }

    #[test]
    fn load_gguf_then_forward() {
        let data = std::fs::read(
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/tiny_harmony.gguf"),
        )
        .expect("test fixture missing");

        let mut engine = HarmonyEngine::new(test_config(), Device::Cpu);
        engine.load_gguf(&data).unwrap();

        let mut cache = engine.new_cache().unwrap();
        let logits = engine.forward(&[1, 2, 3], &mut cache).unwrap();
        assert_eq!(logits.len(), 128); // vocab_size from fixture
    }

    #[test]
    fn load_gguf_replaces_not_implemented() {
        // Verify the placeholder error is gone
        let mut engine = HarmonyEngine::new(test_config(), Device::Cpu);
        let result = engine.load_gguf(b"not valid gguf");
        // Should fail with GGUF parse error, not "not yet implemented"
        match result {
            Err(InferenceError::InvalidGguf(msg)) => {
                assert!(
                    !msg.contains("not yet implemented"),
                    "placeholder should be replaced: {msg}"
                );
            }
            _ => panic!("expected InvalidGguf error, got {result:?}"),
        }
    }
```

- [ ] **Step 9: Run all tests**

Run: `cargo test -p harmony-inference`
Expected: All existing tests PASS + 7 new GGUF tests PASS

- [ ] **Step 10: Commit**

```bash
git add crates/harmony-inference/src/harmony_model.rs
git add crates/harmony-inference/src/harmony_engine.rs
git commit -m "feat(inference): add HarmonyModel::from_gguf() and wire up HarmonyEngine::load_gguf()"
```
