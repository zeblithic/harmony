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
from torch.utils.checkpoint import checkpoint as _gradient_checkpoint


@dataclass
class HarmonyModelConfig:
    """Full configuration for the ct87 HarmonyModel.

    Field names and values match the Rust HarmonyModelConfig in
    crates/harmony-inference/src/harmony_model.rs.

    The `use_ann_engram` flag is research-only (not mirrored in Rust):
    when True, the model skips construction of the production
    `EngramGatedResidual` module and expects the training script to
    attach an `EngramANNInjection` module via `attach_engram_ann()`. Used
    by ZEB-117 Model γ. Configs that set this flag are not GGUF-portable.
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
    think_token_id: int | None = None
    ct_max_steps: int | None = None
    ct_confidence_threshold: float | None = None
    use_ann_engram: bool = False

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

    @staticmethod
    def tiny_engram_ann() -> HarmonyModelConfig:
        """Model γ — gated-residual + ANN retrieval + anti-collapse (ZEB-117).

        Structurally identical to tiny(); differs only in that the model
        expects an externally-attached `EngramANNInjection` module instead
        of constructing the production `EngramGatedResidual`. Not
        GGUF-portable — no Rust mirror.

        See docs/research/2026-04-14-engram-injection-mechanism-findings.md
        for the anti-collapse measures (§3.3) and the ZEB-117 bake-off
        design.
        """
        base = HarmonyModelConfig.tiny()
        base.use_ann_engram = True
        return base


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
        # Scale matches Rust block_attnres.rs:77-78: 1/sqrt(hidden_dim)
        query_scale = 1.0 / math.sqrt(hidden_dim)
        self.queries = nn.ParameterList([
            nn.Parameter(torch.randn(1, 1, hidden_dim) * query_scale)
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
    Training only -- no KV cache, no UQ collection.
    """

    def __init__(self, config: HarmonyModelConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim)

        rotary_emb = RotaryEmbedding(config.head_dim, config.max_seq_len, config.rope_theta)
        self.layers = nn.ModuleList([
            TransformerLayer(config, rotary_emb) for _ in range(config.num_layers)
        ])

        self.final_norm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        self.block_attnres = BlockAttnRes(config.num_blocks, config.hidden_dim)

        # Validate engram_injection_layer (matches Rust harmony_model.rs:511-516)
        if config.engram_injection_layer >= config.num_layers:
            raise ValueError(
                f"engram_injection_layer ({config.engram_injection_layer}) must be "
                f"< num_layers ({config.num_layers})"
            )

        # Engram gated residual injection module. The production
        # `EngramGatedResidual` is always constructed (keeps weight layout
        # stable for GGUF export even when the ANN engram is attached).
        # Model γ sets `config.use_ann_engram=True` and attaches an
        # `EngramANNInjection` via `attach_engram_ann()` after model
        # construction; when both are present, the ANN path is used at
        # the injection layer and `engram_embeddings` is ignored.
        from ct87.engram import EngramGatedResidual
        self.engram_residual = EngramGatedResidual(config)
        self.engram_ann: "EngramANNInjection | None" = None
        # Side channel for the training loop to read the gate after each
        # forward (used for entropy regularization and optional
        # reconstruction loss). Reset every forward that uses the ANN
        # engram.
        self._last_ann_gate: torch.Tensor | None = None

        # Tied embeddings
        if config.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.gradient_checkpointing = False

        self._init_weights()

    def attach_engram_ann(self, module: "EngramANNInjection") -> None:
        """Attach a Model-γ ANN engram injection module.

        Called by the training script after constructing the model and
        loading the corpus table. The module is registered as a
        submodule so its parameters are discovered by the optimizer and
        its buffers move with `.to(device)`.

        When an ANN module is attached, the forward pass uses it at the
        configured injection layer and ignores any `engram_embeddings`
        argument passed to `forward()`.
        """
        if not self.config.use_ann_engram:
            raise ValueError(
                "attach_engram_ann() called but config.use_ann_engram is "
                "False — set the flag (e.g. via "
                "HarmonyModelConfig.tiny_engram_ann()) before attaching."
            )
        self.engram_ann = module

    def _init_weights(self):
        """Initialize weights matching candle HarmonyModel::new().

        - Linear: Kaiming uniform, scale 1/sqrt(fan_in)
        - Conv1d: zeros (Engram conv1d starts zeroed, matching Rust new())
        - RMSNorm: ones (already done in RMSNorm.__init__)
        - Embedding: normal, std 1/sqrt(hidden_dim)
        - BlockAttnRes queries: small normal, std 0.02 (already done in BlockAttnRes.__init__)
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Matches Rust random_linear(): scaled_randn with std=1/sqrt(fan_in)
                fan_in = module.weight.shape[1]
                nn.init.normal_(module.weight, mean=0.0, std=1.0 / math.sqrt(fan_in))
            elif isinstance(module, nn.Conv1d):
                # Engram depthwise conv1d starts zeroed (matches Rust new())
                nn.init.zeros_(module.weight)

        # Embedding init runs last so it overwrites the Kaiming uniform that
        # was applied to the tied lm_head weight (which shares this tensor).
        std = 1.0 / math.sqrt(self.config.hidden_dim)
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=std)

    def set_gradient_checkpointing(self, enable: bool = True) -> None:
        """Enable or disable gradient checkpointing for transformer layers.

        When enabled, intermediate activations inside each TransformerLayer are
        recomputed during the backward pass instead of stored, trading ~30%
        extra compute for significant VRAM savings. Only effective during
        training (model.training=True).
        """
        self.gradient_checkpointing = enable

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        engram_embeddings: torch.Tensor | None = None,
        input_embeds: torch.Tensor | None = None,
        return_hidden_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            input_ids: [batch, seq_len] token IDs. Mutually exclusive with
                input_embeds.
            engram_embeddings: optional [batch, seq_len, engram_dim] Engram
                embeddings from an EngramTable lookup. When provided, the
                EngramGatedResidual module injects them after the configured
                engram_injection_layer.
            input_embeds: optional [batch, seq_len, hidden_dim] embeddings to
                use instead of token lookup. Used by COCONUT continuous thought
                to feed hidden states back as input.
            return_hidden_states: if True, return (logits, hidden_states) where
                hidden_states is the pre-final_norm output of the last layer.

        Returns:
            logits: [batch, seq_len, vocab_size], or
            (logits, hidden_states) if return_hidden_states=True.
        """
        if input_embeds is not None:
            if input_ids is not None:
                raise ValueError("Cannot provide both input_ids and input_embeds")
            h = input_embeds
        elif input_ids is not None:
            h = self.embed_tokens(input_ids)
        else:
            raise ValueError("Must provide either input_ids or input_embeds")

        attnres_state: list[torch.Tensor] = []
        layers_per_block = self.config.layers_per_block
        # Reset gate side-channel at the start of every forward
        self._last_ann_gate = None

        for i, layer in enumerate(self.layers):
            # Block boundary mixing (blocks > 0)
            if i > 0 and i % layers_per_block == 0:
                block_idx = i // layers_per_block
                h = self.block_attnres.block_input(block_idx, h, attnres_state)

            # Standard transformer layer (with optional gradient checkpointing)
            if self.gradient_checkpointing and self.training:
                h = _gradient_checkpoint(layer, h, use_reentrant=False)
            else:
                h = layer(h)

            # Engram injection at the configured layer.
            # Model γ: ANN path takes precedence when attached.
            if i == self.config.engram_injection_layer:
                if self.engram_ann is not None:
                    residual, gate = self.engram_ann(h)
                    h = h + residual
                    self._last_ann_gate = gate
                elif engram_embeddings is not None:
                    h = h + self.engram_residual(h, engram_embeddings)

            # Store block summary at block end
            self.block_attnres.notify_layer_output(i, h, attnres_state, layers_per_block)

        logits = self.lm_head(self.final_norm(h))
        if return_hidden_states:
            return logits, h
        return logits
