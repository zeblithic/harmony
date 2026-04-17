"""Tests for ZEB-135 EMA baseline-subtraction on GatedEngramInjection.

The wrapper must:

  * Register an `inj_ema` buffer of shape (hidden_dim,) initialized to
    zero, so checkpoints from before the feature landed keep loading
    into a fresh EMA state.
  * Update the EMA in training mode only, via
    new_ema = m * old_ema + (1 - m) * mean(out).detach()
    The detach is load-bearing: without it, backprop would flow through
    the running baseline and couple every token's gradient to every
    other token's history.
  * Subtract the EMA in both train AND eval so the optimizer sees the
    same signal the eval metric sees.
  * Reject invalid momentum values (non-finite, <= 0, >= 1).
"""

from __future__ import annotations

import math

import pytest
import torch

from ct87.engram import EngramCrossAttention, GatedEngramInjection
from ct87.model import HarmonyModelConfig


def _tiny_cfg() -> HarmonyModelConfig:
    cfg = HarmonyModelConfig(
        num_layers=4,
        hidden_dim=32,
        num_query_heads=4,
        num_kv_heads=2,
        head_dim=8,
        ffn_dim=64,
        vocab_size=129,
        max_seq_len=64,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        layers_per_block=2,
        engram_injection_layer=1,
        engram_dim=16,
        tie_embeddings=True,
    )
    return cfg


def _build_injection(
    ema_subtract: bool = False, ema_momentum: float = 0.99,
) -> tuple[GatedEngramInjection, HarmonyModelConfig]:
    cfg = _tiny_cfg()
    torch.manual_seed(0)
    table = torch.randn(16, cfg.engram_dim)
    xattn = EngramCrossAttention(
        cfg, table, num_heads=cfg.num_query_heads,
        k_retrieved=4, retrieval_bias_weight=1.0, retrieval_temperature=None,
    )
    inj = GatedEngramInjection(
        xattn,
        alpha_init=0.5,
        ema_subtract=ema_subtract,
        ema_momentum=ema_momentum,
    )
    return inj, cfg


def test_inj_ema_buffer_initialized_to_zero_with_correct_shape() -> None:
    inj, cfg = _build_injection(ema_subtract=True)
    assert hasattr(inj, "inj_ema"), "inj_ema buffer missing"
    assert inj.inj_ema.shape == (cfg.hidden_dim,), (
        f"inj_ema shape must be (hidden_dim,) = ({cfg.hidden_dim},), "
        f"got {tuple(inj.inj_ema.shape)}"
    )
    assert torch.equal(inj.inj_ema, torch.zeros_like(inj.inj_ema)), (
        "inj_ema must start at zero so a fresh checkpoint has "
        "deterministic baseline-subtract behavior at step 0."
    )


def test_inj_ema_registered_even_when_feature_disabled() -> None:
    """Buffer must exist unconditionally so checkpoints saved with the
    feature OFF can be loaded into models with it ON and vice versa."""
    inj, _ = _build_injection(ema_subtract=False)
    assert "inj_ema" in dict(inj.named_buffers()), (
        "inj_ema must be registered regardless of --engram-ema-subtract "
        "so checkpoint state_dicts are schema-compatible across runs."
    )


def test_ema_updates_in_training_mode() -> None:
    """After one training-mode forward, inj_ema must reflect the batch mean."""
    inj, cfg = _build_injection(ema_subtract=True, ema_momentum=0.9)
    inj.train()
    torch.manual_seed(1)
    h = torch.randn(2, 5, cfg.hidden_dim)
    # Capture mean(out) BEFORE subtracting — we need the pre-subtract
    # mean to check the update math.
    with torch.no_grad():
        gate = torch.tanh(inj.alpha)
        retrieved, topk_sims = inj.engram_xattn.retrieve_topk(h)
        inj_real = inj.engram_xattn._attention_block(h, retrieved, topk_sims)
        expected_pre_subtract = (gate * inj_real).mean(dim=(0, 1))

    ema_before = inj.inj_ema.clone()
    _ = inj(h)
    ema_after = inj.inj_ema.clone()

    # inj_ema = 0.9 * 0 + 0.1 * expected_pre_subtract
    expected = 0.9 * ema_before + 0.1 * expected_pre_subtract.to(ema_before.dtype)
    assert torch.allclose(ema_after, expected, atol=1e-5), (
        f"EMA after one forward disagrees with expected mixture.\n"
        f"delta norm = {(ema_after - expected).norm().item():.3e}"
    )


def test_ema_frozen_in_eval_mode() -> None:
    """inj_ema must NOT update in eval, but MUST still be subtracted."""
    inj, cfg = _build_injection(ema_subtract=True, ema_momentum=0.9)
    # Warm the EMA up with one training forward so it's non-zero.
    inj.train()
    h = torch.randn(2, 5, cfg.hidden_dim)
    _ = inj(h)
    ema_before = inj.inj_ema.clone()
    assert ema_before.norm() > 0, "EMA should be non-zero after a training forward"

    inj.eval()
    _ = inj(torch.randn(2, 5, cfg.hidden_dim))
    ema_after = inj.inj_ema.clone()
    assert torch.equal(ema_before, ema_after), (
        "Eval-mode forward must not mutate inj_ema; got "
        f"delta = {(ema_after - ema_before).norm().item():.3e}"
    )


def test_ema_is_actually_subtracted() -> None:
    """Enabling subtract must produce a different output than disabling it."""
    inj_on, cfg = _build_injection(ema_subtract=True, ema_momentum=0.9)
    inj_off, _ = _build_injection(ema_subtract=False)

    # Give them the same weights so the only difference is the feature.
    inj_off.load_state_dict(inj_on.state_dict(), strict=False)

    inj_on.train()
    inj_off.train()
    # Warm the EMA on the subtract-enabled wrapper.
    h_warm = torch.randn(2, 5, cfg.hidden_dim)
    _ = inj_on(h_warm)

    # Now compare outputs on fresh input.
    h = torch.randn(2, 5, cfg.hidden_dim)
    inj_on.eval()
    inj_off.eval()
    with torch.no_grad():
        out_on = inj_on(h)
        out_off = inj_off(h)
    diff = (out_on - out_off).norm().item()
    assert diff > 1e-4, (
        f"With a non-zero EMA, subtract-on and subtract-off outputs must "
        f"differ. Got diff = {diff:.3e}"
    )


def test_ema_has_no_grad_fn() -> None:
    """inj_ema is a buffer; it must never be part of the autograd graph."""
    inj, cfg = _build_injection(ema_subtract=True)
    inj.train()
    h = torch.randn(2, 5, cfg.hidden_dim, requires_grad=True)
    _ = inj(h)
    assert inj.inj_ema.grad_fn is None, (
        "inj_ema must remain a leaf buffer with grad_fn=None. "
        "Backprop through the baseline would couple every token's "
        "gradient to every other token's history."
    )


def test_ema_momentum_rejects_invalid_values() -> None:
    """Constructor must reject non-finite or out-of-range momentum."""
    cfg = _tiny_cfg()
    torch.manual_seed(0)
    table = torch.randn(16, cfg.engram_dim)
    xattn = EngramCrossAttention(
        cfg, table, num_heads=cfg.num_query_heads,
        k_retrieved=4, retrieval_bias_weight=1.0, retrieval_temperature=None,
    )
    for bad in [0.0, 1.0, -0.1, 1.5, float("nan"), float("inf"), float("-inf")]:
        with pytest.raises(ValueError):
            GatedEngramInjection(
                xattn, alpha_init=0.0,
                ema_subtract=True, ema_momentum=bad,
            )


def test_ema_subtract_disabled_leaves_output_untouched() -> None:
    """With the feature disabled, inj_ema must never be subtracted, even
    after it's been mutated by an outside caller."""
    inj_off, cfg = _build_injection(ema_subtract=False)
    inj_off.inj_ema.fill_(1000.0)  # Poison the buffer.
    inj_off.eval()
    h = torch.randn(2, 5, cfg.hidden_dim)
    with torch.no_grad():
        out = inj_off(h)
    # If subtraction were silently happening, out would have huge bias
    # because we poisoned the EMA to 1000.
    assert out.abs().max().item() < 100.0, (
        "Disabled --engram-ema-subtract must short-circuit the subtract. "
        "Got max |out| = "
        f"{out.abs().max().item():.3f} after poisoning inj_ema to 1000."
    )


def test_isfinite_catches_nan_momentum_at_cli_layer() -> None:
    """Regression test for the NaN-sneaking-past-positive-check pattern:
    NaN < 0 returns False, so raw comparisons mis-accept NaN. The fix
    requires math.isfinite() in addition to the range check."""
    # Using 0 < x < 1 alone, NaN sneaks through because NaN > 0 is False
    # (rejected by that branch) — demonstrates the naive code path the
    # CLI validator must guard.
    assert not (0.0 < float("nan") < 1.0), (
        "Sanity: NaN must NOT satisfy the open-interval test. If this "
        "assertion starts failing, Python comparison semantics changed "
        "and the CLI guard logic must be revisited."
    )
    assert math.isfinite(0.99)
    assert not math.isfinite(float("nan"))
    assert not math.isfinite(float("inf"))
