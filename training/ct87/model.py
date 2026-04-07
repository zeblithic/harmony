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
