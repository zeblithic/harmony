"""ct87 model architecture -- mirrors the candle HarmonyModel exactly.

Config values, layer modules, and forward pass must produce identical results
to crates/harmony-inference/src/harmony_model.rs so that weights are portable
via GGUF (Phase 0g).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _gradient_checkpoint

if TYPE_CHECKING:
    from ct87.engram import (
        EngramANNInjection,
        EngramCrossAttention,
        GatedEngramInjection,
    )


@dataclass
class HarmonyModelConfig:
    """Full configuration for the ct87 HarmonyModel.

    Field names and values match the Rust HarmonyModelConfig in
    crates/harmony-inference/src/harmony_model.rs.

    The `ffn_dim_overrides` field is research-only (not mirrored in Rust):
    when set to a non-empty mapping, the listed layer indices use a
    different feed-forward intermediate dimension. Used by ZEB-117 Model
    beta as a params-matched dense control for the engram-injection
    bake-off. Configs that set overrides are not GGUF-portable and will
    fail the Rust-parity tests.

    The `use_ann_engram` flag is also research-only (not mirrored in
    Rust): when True, the model expects the training script to attach an
    `EngramANNInjection` module via `attach_engram_ann()` after
    construction. Used by ZEB-117 Model gamma. Configs that set this
    flag are not GGUF-portable.

    The `use_xattn_engram` flag is research-only (not mirrored in Rust):
    when True, the model expects the training script to attach an
    `EngramCrossAttention` module via `attach_engram_xattn()` after
    construction. Used by ZEB-117 Model delta. Configs that set this
    flag are not GGUF-portable. Mutually exclusive with `use_ann_engram`.
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
    use_xattn_engram: bool = False
    use_head_gates: bool = False
    ffn_dim_overrides: dict[int, int] | None = None
    # η-B / ZEB-130: multi-layer gated cross-attention injection.
    # Non-empty means use the GatedEngramInjection ModuleDict path
    # (mutually exclusive with use_xattn_engram / use_ann_engram).
    engram_inject_layers: tuple[int, ...] = ()
    engram_gate_init: float = 0.0
    # θ-V-contrast (ZEB-130): V-contrastive auxiliary loss. Adds a second
    # xattn forward against a per-step row-permutation of the primary table
    # and penalizes cosine alignment between real-branch and shuffled-branch
    # post-o_proj outputs. Only meaningful when engram_inject_layers is set
    # (rides on top of the multi-layer gated injection path).
    engram_vcontrast_enabled: bool = False
    engram_vcontrast_lambda: float = 1.0
    engram_vcontrast_warmup_steps: int = 200
    # ι-Q-diversity (ZEB-130): MoE load-balancing auxiliary loss on the
    # retrieval-row marginal distribution. Penalizes imbalance in the softmax
    # of Q @ K^T across retrieval rows (importance weighting toward uniform
    # coverage). Only meaningful when engram_inject_layers is non-empty
    # (Q-div operates on the retrieval softmax which only exists when
    # engram cross-attention is injected into some layer).
    engram_qdiv_enabled: bool = False
    engram_qdiv_lambda: float = 0.01
    engram_qdiv_warmup_steps: int = 200

    def __post_init__(self) -> None:
        """Validate `ffn_dim_overrides` up front so misconfigurations fail fast.

        Catches silent typos (out-of-range layer indices that would never
        be queried) and invalid dims (<=0) before model construction.
        """
        if self.use_ann_engram and self.use_xattn_engram:
            raise ValueError(
                "use_ann_engram and use_xattn_engram are mutually "
                "exclusive - only one research-only engram injection "
                "module can be attached at a time."
            )
        # η-B multi-layer injection is mutually exclusive with the legacy
        # single-point paths (model delta xattn, model gamma ANN).
        if self.engram_inject_layers:
            if self.use_xattn_engram or self.use_ann_engram:
                raise ValueError(
                    "engram_inject_layers (multi-layer gated injection) is "
                    "mutually exclusive with use_xattn_engram / use_ann_engram "
                    "(single-point legacy paths)."
                )
            # Type check must come BEFORE the duplicate check: set() equality
            # would treat 2 and 2.0 as the same element (so duplicate detection
            # passes), but ModuleDict keys str(2) vs str(2.0) would collide
            # with forward() probing for str(layer_idx) using an integer
            # loop counter, silently never firing the injection.
            for layer_idx in self.engram_inject_layers:
                # `type(x) is int` rejects bool too (True/False are 1/0 under
                # isinstance but would stringify to "True"/"False" keys).
                if type(layer_idx) is not int:
                    raise TypeError(
                        "engram_inject_layers entries must be plain int "
                        f"(got {layer_idx!r} of type "
                        f"{type(layer_idx).__name__}); ModuleDict keys "
                        "would not match str(layer_idx) probed by forward()."
                    )
            if len(self.engram_inject_layers) != len(set(self.engram_inject_layers)):
                raise ValueError(
                    "engram_inject_layers contains duplicate layer indices: "
                    f"{self.engram_inject_layers!r}. Each injection point must "
                    "be a unique layer index (ModuleDict keys would collide)."
                )
            for layer_idx in self.engram_inject_layers:
                if not (0 <= layer_idx < self.num_layers):
                    raise ValueError(
                        f"engram_inject_layers has layer_idx={layer_idx} "
                        f"outside [0, {self.num_layers}) - would be silently "
                        "ignored at model construction."
                    )
        if self.engram_vcontrast_enabled:
            if not self.engram_inject_layers:
                raise ValueError(
                    "engram_vcontrast_enabled=True requires "
                    "engram_inject_layers to be non-empty (V-contrast lives "
                    "on top of the multi-layer gated injection path)."
                )
            if self.engram_vcontrast_lambda < 0.0:
                raise ValueError(
                    "engram_vcontrast_lambda must be >= 0.0, got "
                    f"{self.engram_vcontrast_lambda!r}"
                )
            if self.engram_vcontrast_warmup_steps < 0:
                raise ValueError(
                    "engram_vcontrast_warmup_steps must be >= 0, got "
                    f"{self.engram_vcontrast_warmup_steps!r}"
                )
        if self.engram_qdiv_enabled:
            if not self.engram_inject_layers:
                raise ValueError(
                    "engram_qdiv_enabled=True requires "
                    "engram_inject_layers to be non-empty (Q-div operates "
                    "on retrieval softmax which only exists in the "
                    "multi-layer cross-attention engram path)."
                )
            if self.engram_qdiv_lambda < 0:
                raise ValueError(
                    f"engram_qdiv_lambda must be >= 0, got {self.engram_qdiv_lambda}"
                )
            if self.engram_qdiv_warmup_steps < 0:
                raise ValueError(
                    f"engram_qdiv_warmup_steps must be >= 0, got "
                    f"{self.engram_qdiv_warmup_steps}"
                )
        if self.ffn_dim_overrides is None:
            return
        for layer_idx, dim in self.ffn_dim_overrides.items():
            if not (0 <= layer_idx < self.num_layers):
                raise ValueError(
                    f"ffn_dim_overrides has layer_idx={layer_idx} outside "
                    f"[0, {self.num_layers}) - would be silently ignored "
                    "at model construction."
                )
            if not isinstance(dim, int) or dim <= 0:
                raise ValueError(
                    f"ffn_dim_overrides[{layer_idx}]={dim!r} must be a "
                    "positive int."
                )

    def layer_ffn_dim(self, layer_idx: int) -> int:
        """Effective FFN intermediate dim for a given layer.

        Returns the override if set for this layer, else the global `ffn_dim`.
        """
        if not (0 <= layer_idx < self.num_layers):
            raise ValueError(
                f"layer_idx={layer_idx} must be in [0, {self.num_layers})"
            )
        if self.ffn_dim_overrides is not None and layer_idx in self.ffn_dim_overrides:
            return self.ffn_dim_overrides[layer_idx]
        return self.ffn_dim

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
    def tiny_ffn_expanded() -> HarmonyModelConfig:
        """Model beta - params-matched dense control for the ZEB-117 bake-off.

        Identical to tiny() except the engram-injection layer's FFN
        intermediate dimension is expanded from 1365 to 1877. This adds
        3 x 512 x 512 = 786,432 parameters to that single MLP, matching
        the overhead of the Model delta cross-attention block (independent
        W_k, W_v, W_o). Isolates the retrieval contribution in Models
        gamma and delta from the free-parameter regularization
        contribution established by ZEB-102 Phase 0 ablations.

        See docs/research/2026-04-14-engram-injection-mechanism-findings.md
        for methodology (Table 3 / section 5.2).
        """
        base = HarmonyModelConfig.tiny()
        base.ffn_dim_overrides = {base.engram_injection_layer: 1877}
        # Re-validate since we mutated the dataclass after construction.
        base.__post_init__()
        return base

    @staticmethod
    def tiny_engram_xattn() -> HarmonyModelConfig:
        """Model delta - cross-attention to memory + top-k retrieval (ZEB-117).

        Structurally identical to tiny(); differs only in that the model
        expects an externally-attached `EngramCrossAttention` module
        instead of constructing the production `EngramGatedResidual`.
        Not GGUF-portable - no Rust mirror.

        See docs/research/2026-04-14-engram-injection-mechanism-findings.md
        for the cross-attention injection rationale (s4) and the ZEB-117
        bake-off design.
        """
        base = HarmonyModelConfig.tiny()
        base.use_xattn_engram = True
        return base

    @staticmethod
    def tiny_engram_xattn_routed() -> HarmonyModelConfig:
        """Model epsilon - cross-attention + per-head gate scalars (ZEB-127).

        Extends Model delta with learned per-head gate scalars that give
        the model N_heads degrees of freedom to selectively route engram
        signal. Inspired by thalamic routing literature.
        """
        base = HarmonyModelConfig.tiny()
        base.use_xattn_engram = True
        base.use_head_gates = True
        return base

    @staticmethod
    def tiny_engram_ann() -> HarmonyModelConfig:
        """Model gamma - gated-residual + ANN retrieval + anti-collapse (ZEB-117).

        Structurally identical to tiny(); differs only in that the model
        expects an externally-attached `EngramANNInjection` module instead
        of constructing the production `EngramGatedResidual`. Not
        GGUF-portable - no Rust mirror.

        See docs/research/2026-04-14-engram-injection-mechanism-findings.md
        for the anti-collapse measures (s3.3) and the ZEB-117 bake-off
        design.
        """
        base = HarmonyModelConfig.tiny()
        base.use_ann_engram = True
        return base

    @staticmethod
    def tiny_engram_ann_routed() -> HarmonyModelConfig:
        """Model epsilon ablation 3: gamma gated-residual + per-head gates (ZEB-127).

        Tests whether per-head routing helps the gamma injection mechanism.
        """
        base = HarmonyModelConfig.tiny()
        base.use_ann_engram = True
        base.use_head_gates = True
        return base

    @staticmethod
    def tiny_engram_xattn_consol_online() -> HarmonyModelConfig:
        """Zeta-A: cross-attention + online consolidation (ZEB-128)."""
        base = HarmonyModelConfig.tiny()
        base.use_xattn_engram = True
        return base

    @staticmethod
    def tiny_engram_xattn_consol_phased() -> HarmonyModelConfig:
        """Zeta-B: cross-attention + phased consolidation (ZEB-128)."""
        base = HarmonyModelConfig.tiny()
        base.use_xattn_engram = True
        return base

    @staticmethod
    def tiny_engram_xattn_ctrl() -> HarmonyModelConfig:
        """Zeta-ctrl: cross-attention control, no consolidation (ZEB-128)."""
        base = HarmonyModelConfig.tiny()
        base.use_xattn_engram = True
        return base

    @staticmethod
    def tiny_engram_xattn_capgap() -> HarmonyModelConfig:
        """η-B capacity-gap: multi-layer zero-init gated xattn injection (ZEB-130).

        Attaches gated cross-attention injection at layers 2 and 5 (early +
        mid-late, matching DeepSeek Engram's U-shaped scaling). Gate alpha
        initialized to 0 so tanh(0)=0 produces no perturbation at step 0 —
        preserves the frozen pretrained baseline's behavior until the
        optimizer learns to open the gate.

        Used with --freeze-backbone and --init-from <beta_checkpoint> to test
        H2: can a well-trained backbone leverage a gated engram signal that
        the from-scratch co-training setup could not?
        """
        base = HarmonyModelConfig.tiny()
        base.engram_inject_layers = (2, 5)
        base.engram_gate_init = 0.0
        base.__post_init__()  # re-validate after mutation (matches tiny_ffn_expanded pattern)
        return base

    @staticmethod
    def tiny_engram_xattn_capgap_vcontrast() -> HarmonyModelConfig:
        """θ-V-contrast: η-B capgap + V-contrastive auxiliary loss (ZEB-130).

        Extends `tiny_engram_xattn_capgap` with a per-step shuffled-table
        contrastive auxiliary loss on the post-o_proj pre-gate outputs of
        every injection layer. Used to address the (D*) DISTRIBUTIONAL
        ALIGNMENT verdict from the 2026-04-17 (W)/(A) forensic — V's
        per-token output directions are diverse within a run but align
        across runs despite different retrievals; the aux loss pressures
        V toward content-sensitivity.
        """
        base = HarmonyModelConfig.tiny_engram_xattn_capgap()
        base.engram_vcontrast_enabled = True
        base.engram_vcontrast_lambda = 1.0
        base.engram_vcontrast_warmup_steps = 200
        base.__post_init__()  # re-validate after mutation
        return base

    @staticmethod
    def tiny_engram_xattn_capgap_qdiv() -> "HarmonyModelConfig":
        """ι_1: capgap baseline + Q-div aux only (no V-contrast).

        Ablation test: does load-balancing Q's retrieval distribution alone
        unstick the η-B content-invariance, or is V-side pressure also needed?
        Spec: docs/superpowers/specs/2026-04-17-iota-q-diversity-design.md
        """
        config = HarmonyModelConfig.tiny_engram_xattn_capgap()
        config.engram_qdiv_enabled = True
        config.__post_init__()
        return config

    @staticmethod
    def tiny_engram_xattn_capgap_vcontrast_qdiv() -> "HarmonyModelConfig":
        """ι_2: capgap + V-contrast + Q-div together.

        Combined shortcut-closure test: does pressuring V toward content-
        sensitivity AND Q toward diversity jointly content-route at 40M?
        Spec: docs/superpowers/specs/2026-04-17-iota-q-diversity-design.md
        """
        config = HarmonyModelConfig.tiny_engram_xattn_capgap_vcontrast()
        config.engram_qdiv_enabled = True
        config.__post_init__()
        return config


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
    def __init__(self, config: HarmonyModelConfig, ffn_dim: int | None = None):
        super().__init__()
        effective_ffn_dim = config.ffn_dim if ffn_dim is None else ffn_dim
        self.gate_proj = nn.Linear(config.hidden_dim, effective_ffn_dim, bias=False)
        self.up_proj = nn.Linear(config.hidden_dim, effective_ffn_dim, bias=False)
        self.down_proj = nn.Linear(effective_ffn_dim, config.hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerLayer(nn.Module):
    def __init__(
        self,
        config: HarmonyModelConfig,
        rotary_emb: RotaryEmbedding,
        layer_idx: int,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.attn = Attention(config, rotary_emb)
        self.ffn_norm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.mlp = Mlp(config, ffn_dim=config.layer_ffn_dim(layer_idx))

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
            TransformerLayer(config, rotary_emb, layer_idx=i)
            for i in range(config.num_layers)
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
        # Model gamma sets `config.use_ann_engram=True` and attaches an
        # `EngramANNInjection` via `attach_engram_ann()` after model
        # construction; when both are present, the ANN path is used at
        # the injection layer and `engram_embeddings` is ignored.
        from ct87.engram import EngramGatedResidual
        self.engram_residual = EngramGatedResidual(config)
        self.engram_ann: EngramANNInjection | None = None
        self.engram_xattn: EngramCrossAttention | None = None
        # η-B multi-layer injection path (ZEB-130).
        # Populated by attach_gated_engram_injections(). ModuleDict keys are
        # str(layer_idx). Mutually exclusive with engram_ann / engram_xattn.
        self.engram_injections: nn.ModuleDict | None = None
        # Side channel for the training loop to read the gate after each
        # forward (used for entropy regularization and optional
        # reconstruction loss). Reset every forward that uses the ANN
        # engram.
        self._last_ann_gate: torch.Tensor | None = None
        self._last_xattn_output: torch.Tensor | None = None
        self._last_pre_injection_hidden: torch.Tensor | None = None
        # θ-V-contrast (ZEB-130): aux-loss sink populated by
        # GatedEngramInjection wrappers (when vcontrast_sink is set) during
        # the forward pass. Owned by the training script (which assigns the
        # same list reference into both this attribute and each wrapper's
        # `_vcontrast_sink`); cleared at the start of every training-mode
        # forward.
        self._contrastive_aux_losses: list[torch.Tensor] | None = None
        # ι-Q-diversity (ZEB-130): aux-loss sink populated by
        # GatedEngramInjection wrappers (when qdiv_sink is set) during
        # the forward pass. Owned by the training script (which assigns the
        # same list reference into both this attribute and each wrapper's
        # `_qdiv_sink`); cleared at the start of every training-mode
        # forward.
        self._qdiv_aux_losses: list[torch.Tensor] = []
        self.engram_inject_mult: float = 1.0
        # ZEB-134 Skip-to-Logit router. When attached, the model captures
        # the engram injection output at the LAST entry of
        # engram_inject_layers and feeds it through the router to produce
        # an additive contribution to the final logits. The router itself
        # lives in ct87.engram to keep all research modules co-located.
        self.engram_skip_router: nn.Module | None = None
        self._last_engram_skip_input: torch.Tensor | None = None

        # Tied embeddings
        if config.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.gradient_checkpointing = False

        self._init_weights()

    def attach_engram_ann(self, module: EngramANNInjection) -> None:
        """Attach a Model-gamma ANN engram injection module.

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
                "False - set the flag (e.g. via "
                "HarmonyModelConfig.tiny_engram_ann()) before attaching."
            )
        if self.engram_xattn is not None:
            raise ValueError(
                "Cannot attach Model gamma (ANN) after Model delta "
                "(cross-attention) - the two research engram modules are "
                "mutually exclusive."
            )
        self.engram_ann = module

    def attach_engram_xattn(self, module: EngramCrossAttention) -> None:
        """Attach a Model-delta cross-attention engram injection module.

        Called by the training script after constructing the model and
        loading the corpus table. The module is registered as a
        submodule so its parameters are discovered by the optimizer and
        its buffers move with `.to(device)`.

        When an xattn module is attached, the forward pass uses it at
        the configured injection layer and ignores any `engram_embeddings`
        argument passed to `forward()`.
        """
        if not self.config.use_xattn_engram:
            raise ValueError(
                "attach_engram_xattn() called but config.use_xattn_engram "
                "is False - set the flag (e.g. via "
                "HarmonyModelConfig.tiny_engram_xattn()) before attaching."
            )
        if self.engram_ann is not None:
            raise ValueError(
                "Cannot attach Model delta (cross-attention) after Model "
                "gamma (ANN) - the two research engram modules are "
                "mutually exclusive."
            )
        self.engram_xattn = module

    def attach_gated_engram_injections(
        self, injections_by_layer: dict[int, GatedEngramInjection]
    ) -> None:
        """Attach the η-B multi-layer gated injection modules (ZEB-130).

        Called by the training script after constructing the model.
        `injections_by_layer` must cover exactly the layer indices listed
        in `config.engram_inject_layers`.

        Registers the injections as submodules (keyed by str(layer_idx))
        so their parameters are discovered by the optimizer and their
        buffers move with `.to(device)`.
        """
        from ct87.engram import GatedEngramInjection  # avoid top-level circular

        if self.engram_injections is not None:
            raise ValueError(
                "attach_gated_engram_injections() called but engram_injections "
                "is already set - re-attach would orphan existing parameters "
                "from the module tree and desync any optimizer built against them."
            )
        if not self.config.engram_inject_layers:
            raise ValueError(
                "attach_gated_engram_injections() called but "
                "config.engram_inject_layers is empty - use "
                "HarmonyModelConfig.tiny_engram_xattn_capgap() (or set the "
                "field directly) before attaching."
            )
        expected = set(self.config.engram_inject_layers)
        got = set(injections_by_layer.keys())
        if got != expected:
            raise ValueError(
                f"attach_gated_engram_injections() got layer_idx keys "
                f"{sorted(got)} but config declares {sorted(expected)}."
            )
        # Track object identity to reject reuse of the same
        # GatedEngramInjection across layer slots. nn.ModuleDict silently
        # accepts duplicate module references (it registers the module once
        # and points both keys at it), which would cause the shared alpha +
        # xattn params to receive two gradient contributions per step and
        # break the "one injection per layer" contract the experiment relies
        # on.
        seen_ids: dict[int, int] = {}
        for layer_idx, inj in injections_by_layer.items():
            if not isinstance(inj, GatedEngramInjection):
                raise TypeError(
                    "attach_gated_engram_injections() values must be "
                    "GatedEngramInjection instances."
                )
            prior = seen_ids.get(id(inj))
            if prior is not None:
                raise ValueError(
                    "attach_gated_engram_injections() got the same "
                    "GatedEngramInjection instance for layers "
                    f"{prior} and {layer_idx}; each configured layer must "
                    "have its own module so parameters aren't shared."
                )
            seen_ids[id(inj)] = layer_idx
        self.engram_injections = nn.ModuleDict({
            str(layer_idx): injections_by_layer[layer_idx]
            for layer_idx in self.config.engram_inject_layers
        })

    def attach_engram_skip_router(self, router: nn.Module) -> None:
        """Attach the ZEB-134 SkipToLogitEngramRouter.

        The router takes the engram injection output at the last entry
        of config.engram_inject_layers and adds a delta-logits tensor
        to the model's main logits. Must be called after
        attach_gated_engram_injections() so the injection path is
        ready to feed the router.
        """
        if self.engram_skip_router is not None:
            raise ValueError(
                "attach_engram_skip_router() called but engram_skip_router "
                "is already set - re-attach would orphan existing router "
                "params from the module tree and desync any optimizer "
                "built against them."
            )
        if self.engram_injections is None:
            raise ValueError(
                "attach_engram_skip_router() requires attach_gated_engram_"
                "injections() to have been called first - the router "
                "reads the injection output of the last configured "
                "injection layer."
            )
        self.engram_skip_router = router

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
                engram_injection_layer. Ignored when η-B multi-layer injection
                is attached (via attach_gated_engram_injections) or when a
                research engram module (ANN/xattn) is attached — the active
                engram path takes precedence.
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
        self._last_xattn_output = None
        self._last_pre_injection_hidden = None
        self._last_engram_skip_input = None
        # V-contrast aux-loss sink: clear only in training mode so eval-time
        # callers (e.g. forensic) can inspect a pre-loaded list without it
        # being wiped out.
        if self._contrastive_aux_losses is not None and self.training:
            self._contrastive_aux_losses.clear()
        # ι-Q-diversity aux-loss sink: clear in training mode.
        if self.training:
            self._qdiv_aux_losses.clear()

        # η-B misuse guard: a multi-layer capgap config without an attach call
        # would silently fall through to the legacy single-point elif below
        # (which probes engram_injection_layer, typically still set from the
        # base preset) and do nothing, producing a model that behaves as if
        # no injection were configured. Fail here so tests/training notice
        # the missing attach_gated_engram_injections() call immediately.
        if self.config.engram_inject_layers and self.engram_injections is None:
            raise RuntimeError(
                "config.engram_inject_layers="
                f"{self.config.engram_inject_layers!r} declares multi-layer "
                "engram injection but attach_gated_engram_injections() was "
                "never called. The forward pass would otherwise be a no-op "
                "at every configured layer."
            )

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

            # Engram injection at configured layer(s).
            # Precedence (mutually exclusive by construction):
            #   η-B multi-layer > Model delta xattn > Model gamma ANN > production
            if self.engram_injections is not None:
                key = str(i)
                if key in self.engram_injections:
                    injection_out = self.engram_injections[key](h)
                    h = h + self.engram_inject_mult * injection_out
                    # ZEB-134: snapshot the LAST injection site's output
                    # for skip-to-logit routing. Overwriting is intended:
                    # only the final injection participates in the skip
                    # path because the decoder layers after that site
                    # are the ones the router is meant to bypass.
                    if self.engram_skip_router is not None:
                        self._last_engram_skip_input = injection_out
            elif i == self.config.engram_injection_layer:
                if self.engram_xattn is not None:
                    self._last_pre_injection_hidden = h
                    xattn_out = self.engram_xattn(h)
                    self._last_xattn_output = xattn_out
                    h = h + xattn_out * self.engram_inject_mult
                elif self.engram_ann is not None:
                    residual, gate = self.engram_ann(h)
                    h = h + residual
                    self._last_ann_gate = gate
                elif engram_embeddings is not None:
                    h = h + self.engram_residual(h, engram_embeddings)

            # Store block summary at block end
            self.block_attnres.notify_layer_output(i, h, attnres_state, layers_per_block)

        logits = self.lm_head(self.final_norm(h))

        # ZEB-134 Skip-to-Logit additive path. The router produces a
        # delta-logits tensor of shape [B, L, vocab] that bypasses the
        # frozen L6/L7 + final_norm pipeline. W_align starts at zero so
        # at step 0 the router contributes zero regardless of input;
        # gradient flow opens through the nonzero alpha scalar first.
        if (
            self.engram_skip_router is not None
            and self._last_engram_skip_input is not None
            and self.engram_inject_mult != 0.0
        ):
            skip_logits = self.engram_skip_router(self._last_engram_skip_input)
            logits = logits + skip_logits

        if return_hidden_states:
            return logits, h
        return logits
