# Phase 0f: PyTorch Training Scaffold — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Define ct87 in PyTorch and prove it trains with Muon+WSD on pre-tokenized data.

**Architecture:** Three Python files — `model.py` (config + all nn.Modules + HarmonyModel), `optim.py` (Muon optimizer + WSD schedule), `train.py` (CLI + data loading + training loop). Lives in a new top-level `training/` directory alongside the Rust `crates/`.

**Tech Stack:** Python 3.10+, PyTorch >= 2.2, safetensors, HuggingFace datasets, pytest

---

## File Structure

| File | Responsibility |
|------|----------------|
| `training/pyproject.toml` | Package definition, dependencies, entry points |
| `training/ct87/__init__.py` | Package marker |
| `training/ct87/model.py` | `HarmonyModelConfig`, `RMSNorm`, `RotaryEmbedding`, `Attention`, `Mlp`, `TransformerLayer`, `BlockAttnRes`, `HarmonyModel` |
| `training/ct87/optim.py` | `Muon`, `WSDSchedule`, `partition_params` |
| `training/ct87/train.py` | CLI, data loading, training loop, checkpointing |
| `training/tests/test_model.py` | Model architecture tests |
| `training/tests/test_optim.py` | Optimizer and schedule tests |
| `training/tests/test_train.py` | End-to-end integration tests |

---

### Task 1: Project Scaffolding + HarmonyModelConfig

**Files:**
- Create: `training/pyproject.toml`
- Create: `training/ct87/__init__.py`
- Create: `training/ct87/model.py`
- Create: `training/tests/test_model.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "ct87"
version = "0.1.0"
description = "PyTorch training scaffold for the ct87 custom model"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.2",
    "safetensors>=0.4",
    "datasets>=2.16",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create `ct87/__init__.py`**

```python
"""ct87 -- PyTorch training scaffold for the Harmony ct87 custom model."""
```

- [ ] **Step 3: Write config tests in `tests/test_model.py`**

These tests verify exact parity with the Rust `HarmonyModelConfig::target()` and
`HarmonyModelConfig::tiny()` constructors in
`crates/harmony-inference/src/harmony_model.rs:58-95`.

```python
"""Tests for ct87 model architecture."""

import torch
import pytest
from ct87.model import HarmonyModelConfig


class TestHarmonyModelConfig:
    """Config values must match Rust HarmonyModelConfig exactly."""

    def test_target_config_values(self):
        c = HarmonyModelConfig.target()
        assert c.num_layers == 24
        assert c.hidden_dim == 1280
        assert c.num_query_heads == 16
        assert c.num_kv_heads == 8
        assert c.head_dim == 80
        assert c.ffn_dim == 3413
        assert c.vocab_size == 32000
        assert c.max_seq_len == 32768
        assert c.rope_theta == pytest.approx(1e6)
        assert c.rms_norm_eps == pytest.approx(1e-6)
        assert c.layers_per_block == 3
        assert c.engram_injection_layer == 2
        assert c.engram_dim == 256
        assert c.tie_embeddings is True

    def test_tiny_config_values(self):
        c = HarmonyModelConfig.tiny()
        assert c.num_layers == 8
        assert c.hidden_dim == 512
        assert c.num_query_heads == 8
        assert c.num_kv_heads == 4
        assert c.head_dim == 64
        assert c.ffn_dim == 1365
        assert c.vocab_size == 32000
        assert c.max_seq_len == 4096
        assert c.rope_theta == pytest.approx(1e6)
        assert c.rms_norm_eps == pytest.approx(1e-6)
        assert c.layers_per_block == 2
        assert c.engram_injection_layer == 2
        assert c.engram_dim == 128
        assert c.tie_embeddings is True

    def test_num_blocks_target(self):
        c = HarmonyModelConfig.target()
        assert c.num_blocks == 8  # 24 / 3

    def test_num_blocks_tiny(self):
        c = HarmonyModelConfig.tiny()
        assert c.num_blocks == 4  # 8 / 2

    def test_num_kv_groups_target(self):
        c = HarmonyModelConfig.target()
        assert c.num_kv_groups == 2  # 16 / 8

    def test_num_kv_groups_tiny(self):
        c = HarmonyModelConfig.tiny()
        assert c.num_kv_groups == 2  # 8 / 4
```

- [ ] **Step 4: Run tests to verify they fail**

```bash
cd training
pip install -e ".[dev]"
pytest tests/test_model.py::TestHarmonyModelConfig -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'ct87.model'` (model.py doesn't exist yet)

- [ ] **Step 5: Write `HarmonyModelConfig` in `ct87/model.py`**

```python
"""ct87 model architecture -- mirrors the candle HarmonyModel exactly.

Config values, layer modules, and forward pass must produce identical results
to crates/harmony-inference/src/harmony_model.rs so that weights are portable
via GGUF (Phase 0g).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HarmonyModelConfig:
    """Full configuration for the ct87 HarmonyModel.

    Field names and values match the Rust HarmonyModelConfig in
    crates/harmony-inference/src/harmony_model.rs.
    """

    num_layers: int
    hidden_dim: int
    num_query_heads: int
    num_kv_heads: int
    head_dim: int
    ffn_dim: int
    vocab_size: int
    max_seq_len: int
    rope_theta: float
    rms_norm_eps: float
    layers_per_block: int
    engram_injection_layer: int
    engram_dim: int
    tie_embeddings: bool

    @property
    def num_blocks(self) -> int:
        return self.num_layers // self.layers_per_block

    @property
    def num_kv_groups(self) -> int:
        return self.num_query_heads // self.num_kv_heads

    @staticmethod
    def target() -> HarmonyModelConfig:
        """Target (production) config -- 24-layer, 1280-hidden ct87 model."""
        return HarmonyModelConfig(
            num_layers=24,
            hidden_dim=1280,
            num_query_heads=16,
            num_kv_heads=8,
            head_dim=80,
            ffn_dim=3413,
            vocab_size=32000,
            max_seq_len=32768,
            rope_theta=1e6,
            rms_norm_eps=1e-6,
            layers_per_block=3,
            engram_injection_layer=2,
            engram_dim=256,
            tie_embeddings=True,
        )

    @staticmethod
    def tiny() -> HarmonyModelConfig:
        """Tiny config -- 8-layer, 512-hidden model for fast iteration."""
        return HarmonyModelConfig(
            num_layers=8,
            hidden_dim=512,
            num_query_heads=8,
            num_kv_heads=4,
            head_dim=64,
            ffn_dim=1365,
            vocab_size=32000,
            max_seq_len=4096,
            rope_theta=1e6,
            rms_norm_eps=1e-6,
            layers_per_block=2,
            engram_injection_layer=2,
            engram_dim=128,
            tie_embeddings=True,
        )
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
cd training
pytest tests/test_model.py::TestHarmonyModelConfig -v
```

Expected: 6 PASSED

- [ ] **Step 7: Commit**

```bash
cd training
git add pyproject.toml ct87/__init__.py ct87/model.py tests/test_model.py
git commit -m "feat(training): add project scaffold and HarmonyModelConfig"
```

---

### Task 2: Layer Building Blocks (RMSNorm, RotaryEmbedding, Attention, Mlp, TransformerLayer)

**Files:**
- Modify: `training/ct87/model.py`
- Modify: `training/tests/test_model.py`

**Reference:** The Rust implementations are in `crates/harmony-inference/src/harmony_model.rs:132-372`. Each PyTorch module must match the Rust version exactly.

- [ ] **Step 1: Write layer tests in `tests/test_model.py`**

Append to the existing test file:

```python
from ct87.model import (
    HarmonyModelConfig,
    RMSNorm,
    RotaryEmbedding,
    Attention,
    Mlp,
    TransformerLayer,
)


def _tiny_config() -> HarmonyModelConfig:
    """Shared tiny config for layer tests."""
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
        tie_embeddings=True,
    )


class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(32, eps=1e-6)
        x = torch.randn(2, 5, 32)
        out = norm(x)
        assert out.shape == (2, 5, 32)

    def test_weight_initialized_to_ones(self):
        norm = RMSNorm(32, eps=1e-6)
        assert torch.allclose(norm.weight, torch.ones(32))

    def test_normalizes_magnitude(self):
        norm = RMSNorm(32, eps=1e-6)
        x = torch.randn(1, 1, 32) * 100.0
        out = norm(x)
        # RMS of output (before weight scaling, which is ones) should be ~1
        rms = out.pow(2).mean(dim=-1).sqrt()
        assert rms.item() == pytest.approx(1.0, abs=0.1)


class TestRotaryEmbedding:
    def test_output_shape(self):
        rope = RotaryEmbedding(head_dim=8, max_seq_len=64, theta=10000.0)
        q = torch.randn(1, 4, 5, 8)  # [batch, heads, seq, head_dim]
        k = torch.randn(1, 2, 5, 8)
        q_rot, k_rot = rope(q, k, offset=0)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_different_offsets_produce_different_results(self):
        rope = RotaryEmbedding(head_dim=8, max_seq_len=64, theta=10000.0)
        q = torch.randn(1, 4, 1, 8)
        k = torch.randn(1, 2, 1, 8)
        q0, _ = rope(q, k, offset=0)
        q5, _ = rope(q, k, offset=5)
        assert not torch.allclose(q0, q5)

    def test_preserves_norm(self):
        """RoPE is a rotation -- it should preserve vector norms."""
        rope = RotaryEmbedding(head_dim=8, max_seq_len=64, theta=10000.0)
        q = torch.randn(1, 4, 3, 8)
        k = torch.randn(1, 2, 3, 8)
        q_rot, k_rot = rope(q, k, offset=0)
        assert torch.allclose(q.norm(dim=-1), q_rot.norm(dim=-1), atol=1e-5)
        assert torch.allclose(k.norm(dim=-1), k_rot.norm(dim=-1), atol=1e-5)


class TestAttention:
    def test_output_shape(self):
        cfg = _tiny_config()
        rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        attn = Attention(cfg, rope)
        x = torch.randn(2, 5, cfg.hidden_dim)
        out = attn(x)
        assert out.shape == (2, 5, cfg.hidden_dim)

    def test_causal_masking(self):
        """Output at position i must not change when tokens after i change."""
        cfg = _tiny_config()
        rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        attn = Attention(cfg, rope)
        attn.eval()

        x = torch.randn(1, 4, cfg.hidden_dim)
        out1 = attn(x)

        # Modify position 3 (last token)
        x2 = x.clone()
        x2[0, 3, :] = torch.randn(cfg.hidden_dim)
        out2 = attn(x2)

        # Positions 0-2 should be identical
        assert torch.allclose(out1[0, :3], out2[0, :3], atol=1e-5)


class TestMlp:
    def test_output_shape(self):
        cfg = _tiny_config()
        mlp = Mlp(cfg)
        x = torch.randn(2, 5, cfg.hidden_dim)
        out = mlp(x)
        assert out.shape == (2, 5, cfg.hidden_dim)


class TestTransformerLayer:
    def test_output_shape(self):
        cfg = _tiny_config()
        rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        layer = TransformerLayer(cfg, rope)
        x = torch.randn(2, 5, cfg.hidden_dim)
        out = layer(x)
        assert out.shape == (2, 5, cfg.hidden_dim)

    def test_residual_connection(self):
        """Output should not be identical to input (layers transform) but
        should be correlated (residual connection)."""
        cfg = _tiny_config()
        rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        layer = TransformerLayer(cfg, rope)
        x = torch.randn(1, 3, cfg.hidden_dim)
        out = layer(x)
        # Not identical
        assert not torch.allclose(x, out)
        # Correlated (cosine similarity > 0 on average due to residual)
        cos_sim = F.cosine_similarity(x.flatten(), out.flatten(), dim=0)
        assert cos_sim.item() > 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd training
pytest tests/test_model.py -k "not TestHarmonyModelConfig" -v
```

Expected: FAIL with `ImportError: cannot import name 'RMSNorm'`

- [ ] **Step 3: Implement `RMSNorm`**

Append to `ct87/model.py`:

```python
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Matches candle_nn::RmsNorm -- weight only, no bias.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight
```

- [ ] **Step 4: Implement `RotaryEmbedding`**

Mirrors `harmony_model.rs:132-169`. Uses the split-half convention matching
candle_nn's `rope()` function.

Append to `ct87/model.py`:

```python
class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding.

    Precomputes sin/cos tables. Applied to Q and K in attention.
    Matches crates/harmony-inference/src/harmony_model.rs:137-169.
    """

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 1e6):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to q and k.

        Args:
            q: [batch, heads, seq_len, head_dim]
            k: [batch, heads, seq_len, head_dim]
            offset: position offset (not used in training, always 0)
        """
        seq_len = q.shape[2]
        cos = self.cos_cached[offset : offset + seq_len]  # [seq, head_dim/2]
        sin = self.sin_cached[offset : offset + seq_len]
        q_rot = _apply_rope(q, cos, sin)
        k_rot = _apply_rope(k, cos, sin)
        return q_rot, k_rot


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embedding to x. Shape: [batch, heads, seq, head_dim].

    Uses split-half convention: first half and second half of head_dim.
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    # cos/sin shape: [seq, half] -> broadcast to [1, 1, seq, half]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
```

- [ ] **Step 5: Implement `Attention`**

Mirrors `harmony_model.rs:202-323`. GQA with per-head QK norm, no bias.
Uses `F.scaled_dot_product_attention(is_causal=True)` for causal masking.

Append to `ct87/model.py`:

```python
def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads for grouped query attention.

    Input: [batch, num_kv_heads, seq_len, head_dim]
    Output: [batch, num_kv_heads * n_rep, seq_len, head_dim]
    """
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    x = x.unsqueeze(2).expand(b, h, n_rep, s, d)
    return x.reshape(b, h * n_rep, s, d)


class Attention(nn.Module):
    """Grouped Query Attention with per-head QK norm and RoPE.

    Matches crates/harmony-inference/src/harmony_model.rs:202-323.
    No bias on projections (Qwen3 style).
    """

    def __init__(self, config: HarmonyModelConfig, rotary_emb: RotaryEmbedding):
        super().__init__()
        self.num_heads = config.num_query_heads
        self.num_kv_heads = config.num_kv_heads
        self.num_kv_groups = config.num_kv_groups
        self.head_dim = config.head_dim

        self.q_proj = nn.Linear(config.hidden_dim, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_dim, bias=False)

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = rotary_emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, seq_len, _ = x.shape

        q = self.q_proj(x).reshape(b, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(b, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(b, seq_len, self.num_kv_heads, self.head_dim)

        # Per-head QK norm (Rust flattens heads into batch, norms, reshapes back)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # [b, seq, heads, dim] -> [b, heads, seq, dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # RoPE
        q, k = self.rotary_emb(q, k)

        # GQA: repeat KV heads to match query heads
        k = _repeat_kv(k, self.num_kv_groups)
        v = _repeat_kv(v, self.num_kv_groups)

        # Scaled dot-product attention with causal mask
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # [b, heads, seq, dim] -> [b, seq, heads * dim]
        attn_out = attn_out.transpose(1, 2).reshape(b, seq_len, -1)
        return self.o_proj(attn_out)
```

- [ ] **Step 6: Implement `Mlp`**

Mirrors `harmony_model.rs:175-196`. SwiGLU with no bias.

Append to `ct87/model.py`:

```python
class Mlp(nn.Module):
    """SwiGLU MLP.

    Matches crates/harmony-inference/src/harmony_model.rs:175-196.
    """

    def __init__(self, config: HarmonyModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_dim, config.ffn_dim, bias=False)
        self.up_proj = nn.Linear(config.hidden_dim, config.ffn_dim, bias=False)
        self.down_proj = nn.Linear(config.ffn_dim, config.hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

- [ ] **Step 7: Implement `TransformerLayer`**

Mirrors `harmony_model.rs:330-371`. Pre-norm, attention, residual, pre-norm, MLP, residual.

Append to `ct87/model.py`:

```python
class TransformerLayer(nn.Module):
    """Single transformer layer: pre-norm attention + pre-norm SwiGLU MLP.

    Matches crates/harmony-inference/src/harmony_model.rs:330-371.
    """

    def __init__(self, config: HarmonyModelConfig, rotary_emb: RotaryEmbedding):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.attn = Attention(config, rotary_emb)
        self.ffn_norm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.mlp = Mlp(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x + self.attn(self.attn_norm(x))
        h = h + self.mlp(self.ffn_norm(h))
        return h
```

- [ ] **Step 8: Run tests to verify they pass**

```bash
cd training
pytest tests/test_model.py -v
```

Expected: All tests PASS (config tests + layer tests)

- [ ] **Step 9: Commit**

```bash
cd training
git add ct87/model.py tests/test_model.py
git commit -m "feat(training): add layer building blocks (RMSNorm, RoPE, Attention, Mlp, TransformerLayer)"
```

---

### Task 3: BlockAttnRes + HarmonyModel

**Files:**
- Modify: `training/ct87/model.py`
- Modify: `training/tests/test_model.py`

**Reference:** BlockAttnRes in `crates/harmony-inference/src/block_attnres.rs:63-233`. HarmonyModel in `crates/harmony-inference/src/harmony_model.rs:378-534`.

- [ ] **Step 1: Write BlockAttnRes and HarmonyModel tests**

Append to `tests/test_model.py`:

```python
from ct87.model import BlockAttnRes, HarmonyModel


class TestBlockAttnRes:
    def test_query_count(self):
        """num_blocks - 1 queries (block 0 has no query)."""
        bar = BlockAttnRes(num_blocks=4, hidden_dim=32)
        assert len(bar.queries) == 3

    def test_block_zero_passthrough(self):
        """block_input for block 0 returns hidden_state unchanged."""
        bar = BlockAttnRes(num_blocks=4, hidden_dim=32)
        h = torch.randn(1, 5, 32)
        state = []
        result = bar.block_input(0, h, state)
        assert torch.equal(result, h)

    def test_block_input_mixes_summaries(self):
        """block_input for block > 0 produces a different tensor than passthrough."""
        bar = BlockAttnRes(num_blocks=4, hidden_dim=32)
        h = torch.randn(1, 5, 32)
        state = [torch.randn(1, 5, 32)]  # One completed block summary
        result = bar.block_input(1, h, state)
        assert result.shape == h.shape
        # Result should differ from raw hidden state (mixing happened)
        assert not torch.allclose(result, h, atol=1e-6)

    def test_notify_stores_at_block_end(self):
        """notify_layer_output stores summary at block boundaries only."""
        bar = BlockAttnRes(num_blocks=4, hidden_dim=32)
        state = []
        h = torch.randn(1, 5, 32)
        layers_per_block = 2

        bar.notify_layer_output(0, h, state, layers_per_block)  # Not end of block
        assert len(state) == 0

        bar.notify_layer_output(1, h, state, layers_per_block)  # End of block 0
        assert len(state) == 1

        bar.notify_layer_output(2, h, state, layers_per_block)  # Not end of block
        assert len(state) == 1

        bar.notify_layer_output(3, h, state, layers_per_block)  # End of block 1
        assert len(state) == 2


class TestHarmonyModel:
    def test_forward_output_shape(self):
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 5))
        logits = model(input_ids)
        assert logits.shape == (2, 5, cfg.vocab_size)

    def test_tied_embeddings(self):
        cfg = _tiny_config()
        assert cfg.tie_embeddings is True
        model = HarmonyModel(cfg)
        assert model.lm_head.weight is model.embed_tokens.weight

    def test_untied_embeddings(self):
        cfg = _tiny_config()
        cfg.tie_embeddings = False
        model = HarmonyModel(cfg)
        assert model.lm_head.weight is not model.embed_tokens.weight

    def test_correct_layer_count(self):
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        assert len(model.layers) == cfg.num_layers

    def test_block_attnres_affects_output(self):
        """Model with BlockAttnRes should produce different outputs than one
        without (single block = no mixing)."""
        cfg = _tiny_config()
        torch.manual_seed(42)
        model_with = HarmonyModel(cfg)

        cfg_no_bar = _tiny_config()
        cfg_no_bar.layers_per_block = cfg_no_bar.num_layers  # Single block
        torch.manual_seed(42)
        model_without = HarmonyModel(cfg_no_bar)

        input_ids = torch.randint(0, cfg.vocab_size, (1, 5))
        out_with = model_with(input_ids)
        out_without = model_without(input_ids)
        # Outputs should differ because BlockAttnRes mixing changes hidden states
        assert not torch.allclose(out_with, out_without, atol=1e-5)

    def test_causal_masking(self):
        """Output at position i must not change when tokens after i change."""
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        model.eval()

        input_ids = torch.randint(0, cfg.vocab_size, (1, 6))
        logits1 = model(input_ids)

        input_ids2 = input_ids.clone()
        input_ids2[0, 4:] = torch.randint(0, cfg.vocab_size, (2,))
        logits2 = model(input_ids2)

        # Positions 0-3 should be identical
        assert torch.allclose(logits1[0, :4], logits2[0, :4], atol=1e-5)

    def test_weight_init_linear_scale(self):
        """Linear weights should have std approximately 1/sqrt(fan_in)."""
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        layer = model.layers[0]
        q_weight = layer.attn.q_proj.weight
        fan_in = q_weight.shape[1]  # [out, in]
        expected_std = 1.0 / math.sqrt(fan_in)
        actual_std = q_weight.std().item()
        # Allow 30% tolerance for random init
        assert abs(actual_std - expected_std) / expected_std < 0.3
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd training
pytest tests/test_model.py -k "TestBlockAttnRes or TestHarmonyModel" -v
```

Expected: FAIL with `ImportError: cannot import name 'BlockAttnRes'`

- [ ] **Step 3: Implement `BlockAttnRes`**

Append to `ct87/model.py`:

```python
class BlockAttnRes(nn.Module):
    """Block Attention Residuals -- learned depth-wise attention at block boundaries.

    At block boundaries, computes attention over previous block summaries to let
    deep layers recall early-layer features. Solves PreNorm dilution.

    Matches crates/harmony-inference/src/block_attnres.rs.
    """

    def __init__(self, num_blocks: int, hidden_dim: int):
        super().__init__()
        # num_blocks - 1 queries: block 0 has no preceding boundary
        self.queries = nn.ParameterList([
            nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
            for _ in range(num_blocks - 1)
        ])
        self.hidden_dim = hidden_dim
        self.scale = math.sqrt(hidden_dim)

    def notify_layer_output(
        self, layer_idx: int, hidden_state: torch.Tensor,
        state: list[torch.Tensor], layers_per_block: int,
    ) -> None:
        """Store block summary at block boundaries."""
        if (layer_idx + 1) % layers_per_block == 0:
            state.append(hidden_state)

    def block_input(
        self, block_idx: int, hidden_state: torch.Tensor,
        state: list[torch.Tensor],
    ) -> torch.Tensor:
        """Mix previous block summaries at block boundary.

        Block 0: passthrough. Block k>0: attention-weighted sum of all
        preceding summaries + current hidden state.
        """
        if block_idx == 0:
            return hidden_state

        query = self.queries[block_idx - 1]  # [1, 1, hidden_dim]

        # Collect candidates: all completed summaries + current hidden state
        candidates = state + [hidden_state]

        # Score each candidate: dot(query, candidate) / sqrt(hidden_dim)
        # candidate shape: [batch, seq_len, hidden_dim]
        # query shape: [1, 1, hidden_dim] -- broadcasts over batch and seq
        scores = []
        for candidate in candidates:
            # [batch, seq_len, hidden_dim] * [1, 1, hidden_dim] -> sum -> [batch, seq_len, 1]
            score = (candidate * query).sum(dim=-1, keepdim=True) / self.scale
            scores.append(score)

        # [batch, seq_len, num_candidates]
        stacked = torch.cat(scores, dim=-1)
        weights = F.softmax(stacked, dim=-1)

        # Weighted sum: [batch, seq_len, hidden_dim]
        result = torch.zeros_like(hidden_state)
        for i, candidate in enumerate(candidates):
            result = result + weights[..., i : i + 1] * candidate

        return result
```

- [ ] **Step 4: Implement `HarmonyModel`**

Append to `ct87/model.py`:

```python
class HarmonyModel(nn.Module):
    """The ct87 custom model -- Qwen3-derived transformer with BlockAttnRes.

    Forward pass mirrors crates/harmony-inference/src/harmony_model.rs:473-534.
    Training only -- no KV cache, no Engram injection, no UQ collection.
    """

    def __init__(self, config: HarmonyModelConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.embed_scale = 1.0 / math.sqrt(config.hidden_dim)

        rotary_emb = RotaryEmbedding(config.head_dim, config.max_seq_len, config.rope_theta)
        self.layers = nn.ModuleList([
            TransformerLayer(config, rotary_emb) for _ in range(config.num_layers)
        ])

        self.final_norm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        self.block_attnres = BlockAttnRes(config.num_blocks, config.hidden_dim)

        # Tied embeddings
        if config.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self._init_weights()

    def _init_weights(self):
        """Initialize weights matching candle HarmonyModel::new().

        - Linear: Kaiming uniform, scale 1/sqrt(fan_in)
        - RMSNorm: ones (already done in RMSNorm.__init__)
        - Embedding: normal, std 1/sqrt(hidden_dim)
        - BlockAttnRes queries: small normal, std 0.02 (already done in BlockAttnRes.__init__)
        """
        std = 1.0 / math.sqrt(self.config.hidden_dim)
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=std)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                fan_in = module.weight.shape[1]
                nn.init.uniform_(module.weight, -1.0 / math.sqrt(fan_in), 1.0 / math.sqrt(fan_in))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: [batch, seq_len] token IDs

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        h = self.embed_tokens(input_ids) * self.embed_scale
        attnres_state: list[torch.Tensor] = []
        layers_per_block = self.config.layers_per_block

        for i, layer in enumerate(self.layers):
            # Block boundary mixing (blocks > 0)
            if i > 0 and i % layers_per_block == 0:
                block_idx = i // layers_per_block
                h = self.block_attnres.block_input(block_idx, h, attnres_state)

            # Standard transformer layer
            h = layer(h)

            # Store block summary at block end
            self.block_attnres.notify_layer_output(i, h, attnres_state, layers_per_block)

        return self.lm_head(self.final_norm(h))
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd training
pytest tests/test_model.py -v
```

Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
cd training
git add ct87/model.py tests/test_model.py
git commit -m "feat(training): add BlockAttnRes and HarmonyModel with full forward pass"
```

---

### Task 4: Muon Optimizer + WSD Schedule

**Files:**
- Create: `training/ct87/optim.py`
- Create: `training/tests/test_optim.py`

**Reference:** Muon optimizer from the Muon paper (Jordan et al.). WSD schedule is
a standard piecewise linear schedule. Both described in the design spec.

- [ ] **Step 1: Write optimizer and schedule tests**

```python
"""Tests for Muon optimizer and WSD schedule."""

import torch
import torch.nn as nn
import pytest
from ct87.optim import Muon, WSDSchedule, partition_params, newton_schulz_orthogonalize
from ct87.model import HarmonyModel, HarmonyModelConfig


def _tiny_config() -> HarmonyModelConfig:
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
        tie_embeddings=True,
    )


class TestNewtonSchulz:
    def test_near_orthogonal(self):
        """After Newton-Schulz, X @ X^T should be close to identity."""
        X = torch.randn(32, 32)
        X_orth = newton_schulz_orthogonalize(X)
        identity = torch.eye(32)
        result = X_orth @ X_orth.T
        assert torch.allclose(result, identity, atol=0.05)

    def test_non_square_tall(self):
        """Newton-Schulz should work on tall matrices (more rows than cols)."""
        X = torch.randn(64, 32)
        X_orth = newton_schulz_orthogonalize(X)
        # For tall matrices, check X^T @ X is close to identity
        result = X_orth.T @ X_orth
        identity = torch.eye(32)
        assert torch.allclose(result, identity, atol=0.1)


class TestMuon:
    def test_step_decreases_loss(self):
        """One Muon step should decrease loss on a simple problem."""
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        muon_params, adam_params = partition_params(model)
        optimizer = Muon(muon_params, adam_params, lr=1e-3, adam_lr=1e-3)

        input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
        targets = torch.randint(0, cfg.vocab_size, (2, 8))

        logits = model(input_ids)
        loss1 = torch.nn.functional.cross_entropy(
            logits.view(-1, cfg.vocab_size), targets.view(-1),
        )
        loss1.backward()
        optimizer.step()
        optimizer.zero_grad()

        logits = model(input_ids)
        loss2 = torch.nn.functional.cross_entropy(
            logits.view(-1, cfg.vocab_size), targets.view(-1),
        )
        assert loss2.item() < loss1.item()

    def test_adam_fallback_for_1d(self):
        """1D parameters (norms, embeddings) should use Adam, not Muon."""
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        muon_params, adam_params = partition_params(model)

        # All adam_params should be 1D or embedding
        for p in adam_params:
            assert p.dim() <= 2, f"Adam param should be <=2D, got {p.dim()}D shape {p.shape}"

        # All muon_params should be 2D (Linear weights)
        for p in muon_params:
            assert p.dim() == 2, f"Muon param should be 2D, got {p.dim()}D shape {p.shape}"


class TestPartitionParams:
    def test_all_params_accounted_for(self):
        """partition_params should cover all model parameters."""
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        muon_params, adam_params = partition_params(model)

        total_partitioned = sum(p.numel() for p in muon_params) + sum(p.numel() for p in adam_params)
        total_model = sum(p.numel() for p in model.parameters())
        assert total_partitioned == total_model

    def test_no_duplicates(self):
        """No parameter should appear in both groups."""
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        muon_params, adam_params = partition_params(model)

        muon_ids = {id(p) for p in muon_params}
        adam_ids = {id(p) for p in adam_params}
        assert len(muon_ids & adam_ids) == 0


class TestWSDSchedule:
    def test_warmup_starts_at_zero(self):
        sched = WSDSchedule(warmup_steps=100, total_steps=1000)
        assert sched.get_lr_multiplier(0) == pytest.approx(0.0)

    def test_warmup_linear(self):
        sched = WSDSchedule(warmup_steps=100, total_steps=1000)
        assert sched.get_lr_multiplier(50) == pytest.approx(0.5)

    def test_warmup_end(self):
        sched = WSDSchedule(warmup_steps=100, total_steps=1000)
        assert sched.get_lr_multiplier(100) == pytest.approx(1.0)

    def test_stable_phase(self):
        sched = WSDSchedule(warmup_steps=100, total_steps=1000, decay_fraction=0.1)
        # Stable phase: steps 100 to 900
        assert sched.get_lr_multiplier(500) == pytest.approx(1.0)

    def test_decay_start(self):
        sched = WSDSchedule(warmup_steps=100, total_steps=1000, decay_fraction=0.1)
        # Decay starts at step 900
        assert sched.get_lr_multiplier(900) == pytest.approx(1.0)

    def test_decay_end(self):
        sched = WSDSchedule(warmup_steps=100, total_steps=1000, decay_fraction=0.1, min_lr_ratio=0.0)
        # At step 999 (last step), should be close to 0
        assert sched.get_lr_multiplier(999) == pytest.approx(0.0, abs=0.02)

    def test_decay_midpoint(self):
        sched = WSDSchedule(warmup_steps=100, total_steps=1000, decay_fraction=0.1, min_lr_ratio=0.0)
        # Decay from step 900 to 999 (100 steps), midpoint at 950
        assert sched.get_lr_multiplier(950) == pytest.approx(0.5, abs=0.02)

    def test_min_lr_ratio(self):
        sched = WSDSchedule(warmup_steps=10, total_steps=100, decay_fraction=0.1, min_lr_ratio=0.1)
        # At final step, LR should be min_lr_ratio
        assert sched.get_lr_multiplier(99) == pytest.approx(0.1, abs=0.02)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd training
pytest tests/test_optim.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'ct87.optim'`

- [ ] **Step 3: Implement `ct87/optim.py`**

```python
"""Muon optimizer and WSD learning rate schedule for ct87 training.

Muon applies Newton-Schulz orthogonalization to the momentum buffer before
updating matrix-shaped parameters (Linear weights). Non-matrix parameters
(embeddings, norms, BlockAttnRes queries) use standard AdamW.

WSD (Warmup-Stable-Decay) is a piecewise linear LR schedule: linear warmup,
constant stable phase, linear decay.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn


def newton_schulz_orthogonalize(X: torch.Tensor, num_iters: int = 5) -> torch.Tensor:
    """Orthogonalize a matrix via Newton-Schulz iteration.

    Approximates the polar decomposition. 5 iterations is sufficient.

    Args:
        X: matrix to orthogonalize [m, n]
        num_iters: number of iterations (default 5)

    Returns:
        Orthogonalized matrix with same shape.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    # Normalize so spectral norm < 1 (required for convergence)
    X = X / (X.norm() + 1e-7)
    for _ in range(num_iters):
        A = X @ X.T
        X = a * X + b * (A @ X) + c * (A @ (A @ X))
    return X


def partition_params(model: nn.Module) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Partition model parameters into Muon (2D matrices) and Adam (everything else).

    Muon group: 2D weight matrices from Linear layers (q/k/v/o_proj, gate/up/down_proj).
    Adam group: embeddings, RMSNorm weights, BlockAttnRes queries, lm_head if tied.

    Returns:
        (muon_params, adam_params)
    """
    muon_params = []
    adam_params = []
    seen: set[int] = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in seen:
            continue  # Skip tied params (lm_head.weight == embed_tokens.weight)
        seen.add(id(param))

        if param.dim() == 2 and "embed" not in name and "norm" not in name:
            muon_params.append(param)
        else:
            adam_params.append(param)

    return muon_params, adam_params


class Muon(torch.optim.Optimizer):
    """Muon optimizer with AdamW fallback for non-matrix parameters.

    Muon applies Newton-Schulz orthogonalization to the momentum buffer
    before updating 2D weight matrices. Non-matrix parameters use AdamW.

    Args:
        muon_params: list of 2D parameters for Muon update rule
        adam_params: list of non-matrix parameters for AdamW
        lr: learning rate for Muon parameters
        momentum: Muon momentum coefficient (default: 0.95)
        adam_lr: learning rate for Adam parameters
        adam_betas: Adam beta coefficients (default: (0.9, 0.95))
        adam_eps: Adam epsilon (default: 1e-8)
        adam_wd: Adam weight decay (default: 0.0)
    """

    def __init__(
        self,
        muon_params: list[torch.Tensor],
        adam_params: list[torch.Tensor],
        lr: float = 3e-4,
        momentum: float = 0.95,
        adam_lr: float = 3e-4,
        adam_betas: tuple[float, float] = (0.9, 0.95),
        adam_eps: float = 1e-8,
        adam_wd: float = 0.0,
    ):
        defaults: dict[str, Any] = dict(lr=lr, momentum=momentum)
        params = []

        # Muon group
        if muon_params:
            params.append({"params": muon_params, "lr": lr, "momentum": momentum, "is_muon": True})

        # Adam group
        if adam_params:
            params.append({
                "params": adam_params,
                "lr": adam_lr,
                "betas": adam_betas,
                "eps": adam_eps,
                "weight_decay": adam_wd,
                "is_muon": False,
            })

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any = None) -> None:
        for group in self.param_groups:
            if group.get("is_muon", False):
                self._muon_step(group)
            else:
                self._adam_step(group)

    def _muon_step(self, group: dict) -> None:
        lr = group["lr"]
        momentum = group["momentum"]

        for p in group["params"]:
            if p.grad is None:
                continue

            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(p.data)

            buf = state["momentum_buffer"]
            buf.mul_(momentum).add_(p.grad)

            # Orthogonalize the momentum buffer
            update = newton_schulz_orthogonalize(buf)

            p.data.add_(update, alpha=-lr)

    def _adam_step(self, group: dict) -> None:
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        wd = group["weight_decay"]

        for p in group["params"]:
            if p.grad is None:
                continue

            state = self.state[p]
            if "step" not in state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p.data)
                state["exp_avg_sq"] = torch.zeros_like(p.data)

            state["step"] += 1
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]

            # Decoupled weight decay
            if wd != 0:
                p.data.mul_(1.0 - lr * wd)

            exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)

            # Bias correction
            bc1 = 1 - beta1 ** state["step"]
            bc2 = 1 - beta2 ** state["step"]
            step_size = lr / bc1

            denom = (exp_avg_sq.sqrt() / math.sqrt(bc2)).add_(eps)
            p.data.addcdiv_(exp_avg, denom, value=-step_size)


class WSDSchedule:
    """Warmup-Stable-Decay learning rate schedule.

    Three phases:
    1. Warmup: linear from 0 to max LR over warmup_steps
    2. Stable: constant at max LR
    3. Decay: linear from max LR to min_lr_ratio * max_lr over last
       decay_fraction * total_steps steps

    Args:
        warmup_steps: number of warmup steps
        total_steps: total number of training steps
        decay_fraction: fraction of total_steps for decay phase (default: 0.1)
        min_lr_ratio: minimum LR as fraction of max LR (default: 0.0)
    """

    def __init__(
        self,
        warmup_steps: int,
        total_steps: int,
        decay_fraction: float = 0.1,
        min_lr_ratio: float = 0.0,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.decay_steps = int(total_steps * decay_fraction)
        self.decay_start = total_steps - self.decay_steps
        self.min_lr_ratio = min_lr_ratio

    def get_lr_multiplier(self, step: int) -> float:
        """Get the learning rate multiplier for a given step.

        Returns a value in [min_lr_ratio, 1.0].
        """
        if step < self.warmup_steps:
            # Linear warmup: 0 -> 1
            return step / self.warmup_steps if self.warmup_steps > 0 else 1.0
        elif step < self.decay_start:
            # Stable: constant at 1.0
            return 1.0
        else:
            # Linear decay: 1.0 -> min_lr_ratio
            progress = (step - self.decay_start) / self.decay_steps if self.decay_steps > 0 else 1.0
            return 1.0 - (1.0 - self.min_lr_ratio) * progress
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd training
pytest tests/test_optim.py -v
```

Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
cd training
git add ct87/optim.py tests/test_optim.py
git commit -m "feat(training): add Muon optimizer with Newton-Schulz and WSD schedule"
```

---

### Task 5: Training Loop + Integration Tests

**Files:**
- Create: `training/ct87/train.py`
- Create: `training/tests/test_train.py`

- [ ] **Step 1: Write training loop and integration tests**

```python
"""Tests for the training loop."""

import os
import tempfile

import torch
import pytest
from ct87.model import HarmonyModel, HarmonyModelConfig
from ct87.optim import Muon, WSDSchedule, partition_params
from ct87.train import save_checkpoint, load_checkpoint, make_synthetic_dataloader


def _tiny_config() -> HarmonyModelConfig:
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
        tie_embeddings=True,
    )


class TestOverfit:
    def test_loss_decreases_on_tiny_batch(self):
        """Tiny model overfits a small batch -- loss drops below 2.0 in 50 steps."""
        torch.manual_seed(42)
        cfg = _tiny_config()
        model = HarmonyModel(cfg)

        muon_params, adam_params = partition_params(model)
        optimizer = Muon(muon_params, adam_params, lr=1e-3, adam_lr=1e-3)

        # Fixed tiny batch: 2 sequences of length 16
        input_ids = torch.randint(0, cfg.vocab_size, (2, 17))
        x = input_ids[:, :-1]
        targets = input_ids[:, 1:]

        initial_loss = None
        final_loss = None

        for step in range(50):
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, cfg.vocab_size), targets.view(-1),
            )
            if step == 0:
                initial_loss = loss.item()
            final_loss = loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        assert final_loss < initial_loss, f"Loss should decrease: {initial_loss} -> {final_loss}"
        assert final_loss < 2.0, f"Loss should drop below 2.0 after 50 steps, got {final_loss}"


class TestCheckpoint:
    def test_save_load_roundtrip(self):
        """Save and load a checkpoint -- logits should match exactly."""
        torch.manual_seed(42)
        cfg = _tiny_config()
        model = HarmonyModel(cfg)

        input_ids = torch.randint(0, cfg.vocab_size, (1, 8))
        logits_before = model(input_ids).detach()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint(model, None, 0, tmpdir)

            model2 = HarmonyModel(cfg)
            load_checkpoint(model2, tmpdir, step=0)
            logits_after = model2(input_ids).detach()

        assert torch.allclose(logits_before, logits_after, atol=1e-6)


class TestSyntheticDataloader:
    def test_batch_shape(self):
        dl = make_synthetic_dataloader(vocab_size=128, seq_len=16, batch_size=4, seed=42)
        batch = next(dl)
        assert batch.shape == (4, 17)  # seq_len + 1

    def test_values_in_range(self):
        dl = make_synthetic_dataloader(vocab_size=128, seq_len=16, batch_size=4, seed=42)
        batch = next(dl)
        assert batch.min().item() >= 0
        assert batch.max().item() < 128

    def test_reproducible(self):
        dl1 = make_synthetic_dataloader(vocab_size=128, seq_len=16, batch_size=4, seed=42)
        dl2 = make_synthetic_dataloader(vocab_size=128, seq_len=16, batch_size=4, seed=42)
        assert torch.equal(next(dl1), next(dl2))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd training
pytest tests/test_train.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'ct87.train'`

- [ ] **Step 3: Implement `ct87/train.py`**

```python
"""Training loop for ct87 -- CLI entry point.

Usage:
    python -m ct87.train --config tiny --data <path> --steps 200

For testing without a real dataset, use --synthetic to generate random tokens.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterator

import torch
import torch.nn.functional as F

from ct87.model import HarmonyModel, HarmonyModelConfig
from ct87.optim import Muon, WSDSchedule, partition_params


def make_synthetic_dataloader(
    vocab_size: int, seq_len: int, batch_size: int, seed: int = 42,
) -> Iterator[torch.Tensor]:
    """Infinite dataloader of random token sequences for testing.

    Yields batches of shape [batch_size, seq_len + 1] (extra token for targets).
    """
    rng = torch.Generator()
    rng.manual_seed(seed)
    while True:
        yield torch.randint(0, vocab_size, (batch_size, seq_len + 1), generator=rng)


def make_hf_dataloader(
    data_path: str, seq_len: int, batch_size: int, seed: int = 42,
) -> Iterator[torch.Tensor]:
    """Infinite dataloader from a pre-tokenized HuggingFace dataset.

    Loads the dataset, concatenates all token sequences into one long stream,
    then yields random windows of [seq_len + 1] tokens packed into batches.
    """
    from datasets import load_from_disk

    dataset = load_from_disk(data_path)
    # Expect a column called "input_ids" with token ID lists
    all_tokens: list[int] = []
    for example in dataset:
        all_tokens.extend(example["input_ids"])

    all_tokens_t = torch.tensor(all_tokens, dtype=torch.long)
    total = len(all_tokens_t)
    window = seq_len + 1

    rng = torch.Generator()
    rng.manual_seed(seed)

    while True:
        starts = torch.randint(0, total - window, (batch_size,), generator=rng)
        batch = torch.stack([all_tokens_t[s : s + window] for s in starts])
        yield batch


def save_checkpoint(
    model: HarmonyModel,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    output_dir: str,
) -> None:
    """Save model weights (safetensors) and optionally optimizer state."""
    from safetensors.torch import save_file

    os.makedirs(output_dir, exist_ok=True)
    weights_path = os.path.join(output_dir, f"model_step_{step}.safetensors")
    save_file(model.state_dict(), weights_path)

    if optimizer is not None:
        opt_path = os.path.join(output_dir, f"optimizer_step_{step}.pt")
        torch.save(optimizer.state_dict(), opt_path)


def load_checkpoint(
    model: HarmonyModel,
    output_dir: str,
    step: int,
) -> None:
    """Load model weights from a safetensors checkpoint."""
    from safetensors.torch import load_file

    weights_path = os.path.join(output_dir, f"model_step_{step}.safetensors")
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict)


def set_lr(optimizer: torch.optim.Optimizer, multiplier: float, base_lr: float) -> None:
    """Set learning rate for all param groups based on schedule multiplier."""
    for group in optimizer.param_groups:
        if "_base_lr" not in group:
            group["_base_lr"] = group["lr"]
        group["lr"] = group["_base_lr"] * multiplier


def detect_device(requested: str | None) -> torch.device:
    """Auto-detect best available device."""
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ct87 model")
    parser.add_argument("--config", choices=["tiny", "target"], default="tiny")
    parser.add_argument("--data", type=str, default=None, help="Path to pre-tokenized HF dataset")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic random data")
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--output-dir", type=str, default="training/checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.data is None and not args.synthetic:
        print("Error: must provide --data <path> or --synthetic", file=sys.stderr)
        sys.exit(1)

    # Config
    config = HarmonyModelConfig.tiny() if args.config == "tiny" else HarmonyModelConfig.target()
    seq_len = args.seq_len or (512 if args.config == "tiny" else 2048)

    # Device and seed
    device = detect_device(args.device)
    torch.manual_seed(args.seed)
    print(f"Config: {args.config}, device: {device}, seq_len: {seq_len}")

    # Model
    model = HarmonyModel(config).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # Optimizer
    muon_params, adam_params = partition_params(model)
    optimizer = Muon(muon_params, adam_params, lr=args.lr, adam_lr=args.lr)
    schedule = WSDSchedule(warmup_steps=args.warmup, total_steps=args.steps)

    # Data
    if args.synthetic:
        dataloader = make_synthetic_dataloader(config.vocab_size, seq_len, args.batch_size, args.seed)
    else:
        dataloader = make_hf_dataloader(args.data, seq_len, args.batch_size, args.seed)

    # Training loop
    for step in range(args.steps):
        lr_mult = schedule.get_lr_multiplier(step)
        set_lr(optimizer, lr_mult, args.lr)

        batch = next(dataloader).to(device)
        input_ids = batch[:, :-1]
        targets = batch[:, 1:]

        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, config.vocab_size), targets.view(-1))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 10 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"step={step:5d}  loss={loss.item():.4f}  lr={current_lr:.6f}")

        if args.save_every > 0 and step > 0 and step % args.save_every == 0:
            save_checkpoint(model, optimizer, step, args.output_dir)
            print(f"  -> checkpoint saved at step {step}")

    # Final checkpoint
    save_checkpoint(model, optimizer, args.steps, args.output_dir)
    print(f"Training complete. Final checkpoint at step {args.steps}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd training
pytest tests/test_train.py -v
```

Expected: All tests PASS

- [ ] **Step 5: Run a quick synthetic training to verify end-to-end**

```bash
cd training
python -m ct87.train --config tiny --synthetic --steps 50 --seq-len 32 --batch-size 2 --warmup 10 --save-every 0
```

Expected: Loss printed every 10 steps, decreasing over time. Something like:
```
Config: tiny, device: cpu, seq_len: 32
Model parameters: ~2,500,000
step=    0  loss=4.8520  lr=0.000000
step=   10  loss=4.7103  lr=0.000030
step=   20  loss=4.5321  lr=0.000060
step=   30  loss=4.2891  lr=0.000090
step=   40  loss=3.9542  lr=0.000120
```

- [ ] **Step 6: Run all tests**

```bash
cd training
pytest tests/ -v
```

Expected: All tests PASS across test_model.py, test_optim.py, test_train.py

- [ ] **Step 7: Commit**

```bash
cd training
git add ct87/train.py tests/test_train.py
git commit -m "feat(training): add training loop with CLI and synthetic data support"
```

---

## Self-Review Checklist

### Spec coverage

| Spec requirement | Task |
|-----------------|------|
| HarmonyModelConfig with target() and tiny() | Task 1, Step 5 |
| RMSNorm, RotaryEmbedding, Attention, Mlp, TransformerLayer | Task 2, Steps 3-7 |
| BlockAttnRes with block_input() and notify_layer_output() | Task 3, Step 3 |
| HarmonyModel with forward pass and BlockAttnRes orchestration | Task 3, Step 4 |
| Tied embeddings, embedding scaling | Task 3, Step 4 |
| Weight initialization (Kaiming, ones, normal, 0.02) | Task 3, Step 4 |
| Muon optimizer with Newton-Schulz | Task 4, Step 3 |
| WSD schedule | Task 4, Step 3 |
| Parameter partitioning | Task 4, Step 3 |
| Training loop + CLI | Task 5, Step 3 |
| Pre-tokenized data loading | Task 5, Step 3 (make_hf_dataloader) |
| Safetensors checkpointing | Task 5, Step 3 |
| pyproject.toml | Task 1, Step 1 |
| All unit tests | Tasks 1-5 |
| Integration overfit test | Task 5, Step 1 |

### Placeholder scan

No TBD, TODO, or "implement later" found. All code blocks are complete.

### Type consistency

- `HarmonyModelConfig` used consistently across all files
- `partition_params(model)` returns `tuple[list[torch.Tensor], list[torch.Tensor]]` -- used in tests and train.py
- `WSDSchedule.get_lr_multiplier(step)` returns `float` -- used in train.py
- `BlockAttnRes.notify_layer_output()` takes `layers_per_block: int` param -- used in HarmonyModel.forward() and tests
- `save_checkpoint` / `load_checkpoint` signatures match between train.py and test_train.py
- `make_synthetic_dataloader` signature matches between train.py and test_train.py
