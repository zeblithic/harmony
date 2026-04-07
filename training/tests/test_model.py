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
