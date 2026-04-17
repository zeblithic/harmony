"""Tests for the ZEB-133 layer-wise content-decay probe.

The probe measures how much of the real-vs-shuffled injection delta
survives each transformer layer after the final L5 injection site,
finally projecting it onto the LM-head rows to see whether the surviving
delta aligns with any vocabulary direction.

Tests use tiny synthetic HarmonyModel instances so that the hook
mechanics and bookkeeping are exercised end-to-end without needing the
full 40M checkpoint or oracle tables. A toy "table swap" is simulated
by hot-swapping a random tensor buffer on the engram module between
runs — the hook math is identical whether the swap is an oracle or a
random tensor.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from ct87.engram import EngramCrossAttention, GatedEngramInjection
from ct87.model import HarmonyModel, HarmonyModelConfig
from scripts.probe_layer_decay import (
    LAYER_DECAY_SITES,
    direction_cosines,
    lm_head_row_alignment,
    measure_layer_decay,
    verdict_from_stats,
)


def _tiny_config(num_layers: int = 6, hidden_dim: int = 32) -> HarmonyModelConfig:
    """HarmonyModelConfig small enough to construct and forward in < 1s.

    Defaults to 6 layers with injection at (1, 3) so the probe has
    exactly two post-injection transformer layers (matching ι₂'s
    8-layer config with last injection at L5 and decay across L6, L7).
    num_layers must be divisible by layers_per_block=2 so block_attnres
    has the right number of inter-block queries.
    """
    cfg = HarmonyModelConfig(
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        num_query_heads=4,
        num_kv_heads=2,
        head_dim=8,
        ffn_dim=64,
        vocab_size=257,
        max_seq_len=64,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        layers_per_block=2,
        engram_injection_layer=1,
        engram_dim=16,
        tie_embeddings=True,
        use_xattn_engram=False,
        use_ann_engram=False,
    )
    # Last injection must be < num_layers - 1 so post-injection decay
    # has something to measure.
    cfg.engram_inject_layers = (1, 3)
    return cfg


def _build_tiny_model(
    cfg: HarmonyModelConfig,
    table: torch.Tensor,
    k_retrieved: int = 4,
) -> HarmonyModel:
    """Build a tiny HarmonyModel with multi-layer engram injection attached."""
    model = HarmonyModel(cfg)
    injections: dict[int, GatedEngramInjection] = {}
    for layer_idx in cfg.engram_inject_layers:
        xattn = EngramCrossAttention(
            cfg, table,
            num_heads=cfg.num_query_heads,
            k_retrieved=k_retrieved,
            retrieval_bias_weight=1.0,
            retrieval_temperature=None,
        )
        # Non-zero gate so the injection actually modifies the residual.
        injections[layer_idx] = GatedEngramInjection(xattn, alpha_init=0.5)
    model.attach_gated_engram_injections(injections)
    model.eval()
    return model


def _clone_weights(src: HarmonyModel, dst: HarmonyModel) -> None:
    """Copy all parameters + persistent buffers from src into dst.

    Leaves non-persistent buffers (EngramCrossAttention.table /
    table_normalized) untouched so the two models share weights but
    differ only in their oracle tables.
    """
    dst.load_state_dict(src.state_dict(), strict=False)


def test_measure_layer_decay_returns_all_expected_keys() -> None:
    """The probe must return every documented site with fraction + dir_cos keys."""
    cfg = _tiny_config()
    torch.manual_seed(0)
    table_real = torch.randn(32, cfg.engram_dim)
    table_shuf = table_real[torch.randperm(32)].clone() + torch.randn_like(table_real) * 0.5

    real_model = _build_tiny_model(cfg, table_real)
    shuf_model = _build_tiny_model(cfg, table_shuf)
    _clone_weights(real_model, shuf_model)

    batch = torch.randint(0, cfg.vocab_size, (2, 16))
    out = measure_layer_decay(real_model, shuf_model, batch)

    for site in LAYER_DECAY_SITES:
        assert site in out, f"Missing site {site!r} in probe output"
        assert "fraction" in out[site] and "direction_cos_vs_L5" in out[site], \
            f"Site {site!r} missing expected sub-keys"


def test_pre_injection_fraction_is_zero_for_matched_tokens() -> None:
    """Pre-last-injection, real and shuf share the computation — diff must be 0.

    The two models share every parameter; the only divergence is the
    oracle table. Before the final injection site runs, neither model
    has read from its oracle table at the final site, so the hidden
    state must be bit-equal.
    """
    cfg = _tiny_config()
    torch.manual_seed(1)
    table_real = torch.randn(32, cfg.engram_dim)
    table_shuf = torch.randn(32, cfg.engram_dim)

    real_model = _build_tiny_model(cfg, table_real)
    shuf_model = _build_tiny_model(cfg, table_shuf)
    _clone_weights(real_model, shuf_model)

    # The shared-weights invariant requires that prior injection sites
    # use the same table too — but our tables differ, so earlier L1
    # injection already diverges the two. Temporarily remove the earlier
    # injection so the ONLY difference is the L2 (last) site.
    final = cfg.engram_inject_layers[-1]
    for m in (real_model, shuf_model):
        for k in list(m.engram_injections.keys()):
            if int(k) != final:
                del m.engram_injections[k]
        m.config = type(m.config)(**{**m.config.__dict__,
                                    "engram_inject_layers": (final,)})

    batch = torch.randint(0, cfg.vocab_size, (2, 16))
    out = measure_layer_decay(real_model, shuf_model, batch)
    assert out["L_pre_inj"]["fraction"] == pytest.approx(0.0, abs=1e-5), (
        "Pre-injection fraction must be zero when weights + table usage "
        "prior to the final injection are identical."
    )


def test_direction_cosine_is_in_unit_range() -> None:
    """All cosine values must lie in [-1, 1] (sanity for numerical correctness)."""
    cfg = _tiny_config()
    torch.manual_seed(2)
    table_real = torch.randn(32, cfg.engram_dim)
    table_shuf = torch.randn(32, cfg.engram_dim) * 2.0 + 0.3

    real_model = _build_tiny_model(cfg, table_real)
    shuf_model = _build_tiny_model(cfg, table_shuf)
    _clone_weights(real_model, shuf_model)

    batch = torch.randint(0, cfg.vocab_size, (2, 16))
    out = measure_layer_decay(real_model, shuf_model, batch)
    for site, stats in out.items():
        dc = stats["direction_cos_vs_L5"]
        if not math.isnan(dc):
            assert -1.0 - 1e-5 <= dc <= 1.0 + 1e-5, \
                f"Site {site}: direction_cos_vs_L5 = {dc} outside [-1, 1]"


def test_real_vs_shuf_shapes_match() -> None:
    """Smoke test: different batch sizes should not blow up the probe."""
    cfg = _tiny_config()
    torch.manual_seed(3)
    table_real = torch.randn(32, cfg.engram_dim)
    table_shuf = torch.randn(32, cfg.engram_dim)

    real_model = _build_tiny_model(cfg, table_real)
    shuf_model = _build_tiny_model(cfg, table_shuf)
    _clone_weights(real_model, shuf_model)

    for B, L in [(1, 8), (4, 16)]:
        batch = torch.randint(0, cfg.vocab_size, (B, L))
        out = measure_layer_decay(real_model, shuf_model, batch)
        for site in LAYER_DECAY_SITES:
            assert math.isfinite(out[site]["fraction"]), (
                f"fraction NaN/inf at site {site} for B={B} L={L}"
            )


def test_post_injection_fraction_is_positive_when_tables_differ() -> None:
    """If the oracle tables actually differ, the post-injection diff must be nonzero.

    Guards against the "probe reports 0 no matter what" bug that would
    silently pass every other test.
    """
    cfg = _tiny_config()
    torch.manual_seed(4)
    table_real = torch.randn(64, cfg.engram_dim) * 2.0
    table_shuf = torch.randn(64, cfg.engram_dim) * 2.0 - 1.0

    real_model = _build_tiny_model(cfg, table_real)
    shuf_model = _build_tiny_model(cfg, table_shuf)
    _clone_weights(real_model, shuf_model)

    batch = torch.randint(0, cfg.vocab_size, (2, 16))
    out = measure_layer_decay(real_model, shuf_model, batch)
    assert out["L_post_inj"]["fraction"] > 1e-4, (
        f"L_post_inj fraction should be meaningfully > 0 when tables "
        f"actually differ; got {out['L_post_inj']['fraction']:.2e}."
    )


def test_direction_cosines_floats_only() -> None:
    """direction_cosines() helper must return a plain Python float per site."""
    d_pre = torch.zeros(2, 8, 16)
    d_post = torch.randn(2, 8, 16)
    d_final = torch.randn(2, 8, 16)
    cosines = direction_cosines(
        {"pre": d_pre, "post": d_post, "final": d_final},
        reference_key="post",
    )
    for k, v in cosines.items():
        assert isinstance(v, float), f"{k}: expected float, got {type(v).__name__}"
    assert cosines["post"] == pytest.approx(1.0, abs=1e-5)


def test_lm_head_row_alignment_bounds() -> None:
    """max row cos ≤ 1; mean-top-10 cos ≤ max row cos."""
    torch.manual_seed(0)
    delta = torch.randn(2, 8, 16)
    lm_head_weight = torch.randn(100, 16)
    stats = lm_head_row_alignment(delta, lm_head_weight, top_k=10)
    assert 0.0 <= stats["max_cos"] <= 1.0 + 1e-5
    assert stats["top10_mean_cos"] <= stats["max_cos"] + 1e-5


def test_lm_head_row_alignment_small_vocab() -> None:
    """top_k larger than vocab should clamp rather than crash."""
    delta = torch.randn(1, 2, 4)
    lm_head_weight = torch.randn(3, 4)
    stats = lm_head_row_alignment(delta, lm_head_weight, top_k=10)
    assert math.isfinite(stats["max_cos"]) and math.isfinite(stats["top10_mean_cos"])


def test_verdict_hysteresis_has_fallback() -> None:
    """verdict_from_stats must return one of the known verdicts for any input."""
    from scripts.probe_layer_decay import VERDICT_LABELS
    good_decay = {
        "L_post_inj": {"fraction": 0.10, "direction_cos_vs_L5": 1.0},
        "L6_out": {"fraction": 0.01, "direction_cos_vs_L5": 0.3},
        "L7_out": {"fraction": 0.005, "direction_cos_vs_L5": 0.1},
        "LM_head_in": {"fraction": 0.002, "direction_cos_vs_L5": 0.05},
    }
    lm_stats = {"max_cos": 0.1, "top10_mean_cos": 0.05}
    verdict = verdict_from_stats(good_decay, lm_stats)
    assert verdict in VERDICT_LABELS

    survived_but_blind = {
        "L_post_inj": {"fraction": 0.10, "direction_cos_vs_L5": 1.0},
        "L6_out": {"fraction": 0.09, "direction_cos_vs_L5": 0.9},
        "L7_out": {"fraction": 0.09, "direction_cos_vs_L5": 0.85},
        "LM_head_in": {"fraction": 0.08, "direction_cos_vs_L5": 0.8},
    }
    lm_low = {"max_cos": 0.02, "top10_mean_cos": 0.015}
    assert verdict_from_stats(survived_but_blind, lm_low) in VERDICT_LABELS
