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


# ---------------------------------------------------------------------------
# Layer building blocks
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 1e6):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, offset: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[2]
        cos = self.cos_cached[offset : offset + seq_len]
        sin = self.sin_cached[offset : offset + seq_len]
        return _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    x = x.unsqueeze(2).expand(b, h, n_rep, s, d)
    return x.reshape(b, h * n_rep, s, d)


class Attention(nn.Module):
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
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q, k = self.rotary_emb(q, k)
        k = _repeat_kv(k, self.num_kv_groups)
        v = _repeat_kv(v, self.num_kv_groups)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).reshape(b, seq_len, -1)
        return self.o_proj(attn_out)


class Mlp(nn.Module):
    def __init__(self, config: HarmonyModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_dim, config.ffn_dim, bias=False)
        self.up_proj = nn.Linear(config.hidden_dim, config.ffn_dim, bias=False)
        self.down_proj = nn.Linear(config.ffn_dim, config.hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerLayer(nn.Module):
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


# ---------------------------------------------------------------------------
# Block Attention Residuals + top-level model
# ---------------------------------------------------------------------------


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
        scores = []
        for candidate in candidates:
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
        for module in self.modules():
            if isinstance(module, nn.Linear):
                fan_in = module.weight.shape[1]
                nn.init.uniform_(module.weight, -1.0 / math.sqrt(fan_in), 1.0 / math.sqrt(fan_in))

        # Embedding init runs last so it overwrites the Kaiming uniform that
        # was applied to the tied lm_head weight (which shares this tensor).
        std = 1.0 / math.sqrt(self.config.hidden_dim)
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=std)

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
