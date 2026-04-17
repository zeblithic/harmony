"""Tests for θ-V-contrast V-contrastive engram injection (ZEB-130)."""
from __future__ import annotations

import pytest

from ct87.model import HarmonyModelConfig


def _tiny_config() -> HarmonyModelConfig:
    """Minimal config for V-contrast tests (matches test_capacity_gap pattern)."""
    c = HarmonyModelConfig(
        num_layers=4, hidden_dim=64, num_query_heads=2, num_kv_heads=2,
        head_dim=32, ffn_dim=128, vocab_size=128, max_seq_len=64,
        rope_theta=10000.0, rms_norm_eps=1e-6, layers_per_block=2,
        engram_injection_layer=1, engram_dim=32, tie_embeddings=True,
    )
    return c


class TestVContrastConfig:

    def test_default_disabled(self):
        c = _tiny_config()
        assert c.engram_vcontrast_enabled is False
        assert c.engram_vcontrast_lambda == 1.0
        assert c.engram_vcontrast_warmup_steps == 200

    def test_enabled_passes_post_init(self):
        c = _tiny_config()
        c.engram_inject_layers = (1,)
        c.engram_vcontrast_enabled = True
        c.engram_vcontrast_lambda = 0.5
        c.engram_vcontrast_warmup_steps = 100
        c.__post_init__()  # should not raise

    def test_negative_lambda_rejected(self):
        c = _tiny_config()
        c.engram_inject_layers = (1,)
        c.engram_vcontrast_enabled = True
        c.engram_vcontrast_lambda = -1.0
        with pytest.raises(ValueError, match="engram_vcontrast_lambda"):
            c.__post_init__()

    def test_negative_warmup_rejected(self):
        c = _tiny_config()
        c.engram_inject_layers = (1,)
        c.engram_vcontrast_enabled = True
        c.engram_vcontrast_warmup_steps = -1
        with pytest.raises(ValueError, match="engram_vcontrast_warmup_steps"):
            c.__post_init__()

    def test_enabled_without_inject_layers_rejected(self):
        """V-contrast lives on top of the multi-layer gated injection path."""
        c = _tiny_config()
        c.engram_vcontrast_enabled = True  # but engram_inject_layers stays empty
        with pytest.raises(ValueError, match="engram_inject_layers"):
            c.__post_init__()


class TestVContrastPreset:

    def test_preset_extends_capgap(self):
        c = HarmonyModelConfig.tiny_engram_xattn_capgap_vcontrast()
        # capgap-inherited fields:
        assert c.engram_inject_layers == (2, 5)
        assert c.engram_gate_init == 0.0
        # V-contrast-specific fields:
        assert c.engram_vcontrast_enabled is True
        assert c.engram_vcontrast_lambda == 1.0
        assert c.engram_vcontrast_warmup_steps == 200

    def test_preset_passes_post_init(self):
        # Re-validates after __post_init__ runs (preset calls it explicitly).
        HarmonyModelConfig.tiny_engram_xattn_capgap_vcontrast()


class TestLambdaSchedule:

    def test_step_zero_returns_zero(self):
        from ct87.train import lambda_schedule
        assert lambda_schedule(0, warmup=200, target=1.0) == 0.0

    def test_warmup_linear(self):
        from ct87.train import lambda_schedule
        # 100/200 = 0.5 of target
        assert lambda_schedule(100, warmup=200, target=1.0) == pytest.approx(0.5)
        # 50/200 = 0.25
        assert lambda_schedule(50, warmup=200, target=2.0) == pytest.approx(0.5)

    def test_at_warmup_boundary(self):
        from ct87.train import lambda_schedule
        # Spec: returns target AT and past the warmup boundary.
        assert lambda_schedule(200, warmup=200, target=1.0) == 1.0

    def test_past_warmup_returns_target(self):
        from ct87.train import lambda_schedule
        assert lambda_schedule(500, warmup=200, target=1.5) == 1.5
        assert lambda_schedule(10000, warmup=200, target=0.3) == 0.3

    def test_zero_warmup_returns_target_immediately(self):
        from ct87.train import lambda_schedule
        # Edge case: warmup=0 should never enter the linear ramp branch
        # (avoids division-by-zero and matches "no warmup" semantics).
        assert lambda_schedule(0, warmup=0, target=1.0) == 1.0
        assert lambda_schedule(100, warmup=0, target=1.0) == 1.0
