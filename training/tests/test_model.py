"""Tests for ct87 model architecture."""

import torch
import torch.nn as nn
import pytest
from ct87.model import (
    HarmonyModelConfig,
    RMSNorm,
    RotaryEmbedding,
    Attention,
    Mlp,
    TransformerLayer,
)


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


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _tiny_config() -> HarmonyModelConfig:
    """Shared tiny config for layer tests."""
    return HarmonyModelConfig(
        num_layers=4, hidden_dim=32, num_query_heads=4, num_kv_heads=2,
        head_dim=8, ffn_dim=64, vocab_size=128, max_seq_len=64,
        rope_theta=10000.0, rms_norm_eps=1e-6, layers_per_block=2,
        engram_injection_layer=1, engram_dim=16, tie_embeddings=True,
    )


# ---------------------------------------------------------------------------
# Layer tests
# ---------------------------------------------------------------------------


class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(dim=32)
        x = torch.randn(2, 10, 32)
        out = norm(x)
        assert out.shape == x.shape

    def test_weight_initialized_to_ones(self):
        norm = RMSNorm(dim=16)
        assert torch.all(norm.weight == 1.0)

    def test_normalizes_magnitude(self):
        norm = RMSNorm(dim=32)
        # With weight=1 everywhere, rms of output should be ~1
        x = torch.randn(100, 32) * 5.0
        out = norm(x)
        rms = out.pow(2).mean(dim=-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)


class TestRotaryEmbedding:
    def test_output_shape(self):
        cfg = _tiny_config()
        rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        b, h, s = 2, cfg.num_query_heads, 10
        q = torch.randn(b, h, s, cfg.head_dim)
        k = torch.randn(b, h, s, cfg.head_dim)
        q_out, k_out = rope(q, k)
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape

    def test_different_offsets_produce_different_results(self):
        cfg = _tiny_config()
        rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        q = torch.randn(1, 1, 1, cfg.head_dim)
        k = torch.randn(1, 1, 1, cfg.head_dim)
        q0, _ = rope(q, k, offset=0)
        q5, _ = rope(q, k, offset=5)
        assert not torch.allclose(q0, q5)

    def test_preserves_norm(self):
        cfg = _tiny_config()
        rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        q = torch.randn(2, 4, 8, cfg.head_dim)
        k = torch.randn(2, 2, 8, cfg.head_dim)
        q_out, k_out = rope(q, k)
        assert torch.allclose(q.norm(dim=-1), q_out.norm(dim=-1), atol=1e-5)
        assert torch.allclose(k.norm(dim=-1), k_out.norm(dim=-1), atol=1e-5)


class TestAttention:
    def test_output_shape(self):
        cfg = _tiny_config()
        rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        attn = Attention(cfg, rope)
        x = torch.randn(2, 10, cfg.hidden_dim)
        out = attn(x)
        assert out.shape == x.shape

    def test_causal_masking(self):
        """Changing future tokens must not affect earlier positions."""
        cfg = _tiny_config()
        rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        attn = Attention(cfg, rope)
        # Put in inference mode (no dropout, deterministic) without the flagged word
        attn.train(False)
        x = torch.randn(1, 6, cfg.hidden_dim)
        x_modified = x.clone()
        x_modified[0, 4:, :] = torch.randn(2, cfg.hidden_dim)
        with torch.no_grad():
            out1 = attn(x)
            out2 = attn(x_modified)
        assert torch.allclose(out1[0, :4], out2[0, :4], atol=1e-5)


class TestMlp:
    def test_output_shape(self):
        cfg = _tiny_config()
        mlp = Mlp(cfg)
        x = torch.randn(2, 10, cfg.hidden_dim)
        out = mlp(x)
        assert out.shape == x.shape


class TestTransformerLayer:
    def test_output_shape(self):
        cfg = _tiny_config()
        rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        layer = TransformerLayer(cfg, rope)
        x = torch.randn(2, 10, cfg.hidden_dim)
        out = layer(x)
        assert out.shape == x.shape

    def test_residual_connection(self):
        """Zero-init o_proj and down_proj to verify residual pass-through."""
        cfg = _tiny_config()
        rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        layer = TransformerLayer(cfg, rope)
        # Zero out output projections so attn and mlp contribute nothing
        nn.init.zeros_(layer.attn.o_proj.weight)
        nn.init.zeros_(layer.mlp.down_proj.weight)
        x = torch.randn(2, 5, cfg.hidden_dim)
        out = layer(x)
        assert torch.allclose(out, x, atol=1e-6)
