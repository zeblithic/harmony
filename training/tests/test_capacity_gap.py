"""Tests for η-B capacity-gap pretraining feature (ZEB-130)."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from ct87.engram import EngramCrossAttention, GatedEngramInjection
from ct87.model import HarmonyModel, HarmonyModelConfig


def _tiny_config() -> HarmonyModelConfig:
    """Minimal config for GatedEngramInjection tests."""
    c = HarmonyModelConfig(
        num_layers=4, hidden_dim=64, num_query_heads=2, num_kv_heads=2,
        head_dim=32, ffn_dim=128, vocab_size=128, max_seq_len=64,
        rope_theta=10000.0, rms_norm_eps=1e-6, layers_per_block=2,
        engram_injection_layer=1, engram_dim=32, tie_embeddings=True,
    )
    c.use_xattn_engram = True
    return c


class TestGatedEngramInjection:
    """GatedEngramInjection wraps EngramCrossAttention with a learnable tanh gate."""

    def _make_xattn(self) -> EngramCrossAttention:
        c = _tiny_config()
        table = torch.randn(16, c.engram_dim)
        xattn = EngramCrossAttention(c, table, num_heads=2, k_retrieved=4)
        # EngramCrossAttention zero-inits o_proj by design (step-0 no-op
        # contract). Override here so tests that require non-zero xattn
        # output can observe the gate signal.
        nn.init.xavier_uniform_(xattn.o_proj.weight)
        return xattn

    def test_forward_zero_at_init(self):
        """With alpha_init=0, tanh(0)=0 so the gate output is the zero tensor."""
        torch.manual_seed(0)
        wrapper = GatedEngramInjection(self._make_xattn(), alpha_init=0.0)
        wrapper.train(False)
        h = torch.randn(2, 5, 64)
        out = wrapper(h)
        assert torch.allclose(out, torch.zeros_like(h), atol=1e-6), (
            "With alpha_init=0, gate output must be zero (tanh(0)=0)"
        )

    def test_alpha_is_learnable_parameter(self):
        """Alpha is registered as a parameter and is discoverable by optimizers."""
        wrapper = GatedEngramInjection(self._make_xattn(), alpha_init=0.5)
        param_names = dict(wrapper.named_parameters())
        assert "alpha" in param_names
        assert param_names["alpha"].requires_grad
        assert torch.allclose(
            param_names["alpha"].detach(), torch.tensor(0.5), atol=1e-6
        )

    def test_gradient_flows_to_alpha(self):
        """A loss depending on the wrapper output must produce a non-zero grad on alpha."""
        torch.manual_seed(0)
        wrapper = GatedEngramInjection(self._make_xattn(), alpha_init=0.1)
        wrapper.train(True)
        h = torch.randn(2, 5, 64, requires_grad=False)
        out = wrapper(h)
        loss = out.pow(2).mean()
        loss.backward()
        assert wrapper.alpha.grad is not None
        assert wrapper.alpha.grad.abs().item() > 1e-8

    def test_forward_nonzero_when_alpha_nonzero(self):
        """With alpha != 0, output must be non-zero (injection is active)."""
        torch.manual_seed(0)
        wrapper = GatedEngramInjection(self._make_xattn(), alpha_init=1.0)
        wrapper.train(False)
        h = torch.randn(2, 5, 64)
        out = wrapper(h)
        assert out.abs().max().item() > 1e-4

    def test_alpha_init_default_is_zero(self):
        """Default alpha_init is 0.0 (the safe zero-perturbation value)."""
        wrapper = GatedEngramInjection(self._make_xattn())
        assert wrapper.alpha.detach().item() == 0.0


class TestCapacityGapConfig:
    """Config fields and preset for η-B."""

    def test_config_has_inject_layers_default_empty(self):
        """Default config has empty engram_inject_layers (legacy behavior preserved)."""
        c = HarmonyModelConfig.tiny()
        assert c.engram_inject_layers == ()

    def test_config_has_gate_init_default_zero(self):
        """Default config has engram_gate_init=0.0."""
        c = HarmonyModelConfig.tiny()
        assert c.engram_gate_init == 0.0

    def test_capgap_preset_sets_inject_layers(self):
        """tiny_engram_xattn_capgap uses multi-layer injection at layers 2 and 5."""
        c = HarmonyModelConfig.tiny_engram_xattn_capgap()
        assert c.engram_inject_layers == (2, 5)
        assert c.engram_gate_init == 0.0

    def test_capgap_preset_does_not_set_legacy_xattn_flag(self):
        """capgap preset uses the new path, not the legacy single-point path."""
        c = HarmonyModelConfig.tiny_engram_xattn_capgap()
        assert c.use_xattn_engram is False
        assert c.use_ann_engram is False

    def test_config_rejects_mixing_legacy_and_multi_layer(self):
        """Config must not set use_xattn_engram=True alongside non-empty inject_layers."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            HarmonyModelConfig(
                num_layers=8, hidden_dim=512, num_query_heads=8, num_kv_heads=4,
                head_dim=64, ffn_dim=1365, vocab_size=32000, max_seq_len=4096,
                rope_theta=1e6, rms_norm_eps=1e-6, layers_per_block=2,
                engram_injection_layer=2, engram_dim=128, tie_embeddings=True,
                use_xattn_engram=True,
                engram_inject_layers=(2, 5),
            )

    def test_config_rejects_out_of_range_inject_layer(self):
        """engram_inject_layers must reference valid layer indices."""
        with pytest.raises(ValueError, match="outside"):
            HarmonyModelConfig(
                num_layers=8, hidden_dim=512, num_query_heads=8, num_kv_heads=4,
                head_dim=64, ffn_dim=1365, vocab_size=32000, max_seq_len=4096,
                rope_theta=1e6, rms_norm_eps=1e-6, layers_per_block=2,
                engram_injection_layer=2, engram_dim=128, tie_embeddings=True,
                engram_inject_layers=(2, 99),
            )

    def test_config_rejects_mixing_ann_and_multi_layer(self):
        """Config must not set use_ann_engram=True alongside non-empty inject_layers."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            HarmonyModelConfig(
                num_layers=8, hidden_dim=512, num_query_heads=8, num_kv_heads=4,
                head_dim=64, ffn_dim=1365, vocab_size=32000, max_seq_len=4096,
                rope_theta=1e6, rms_norm_eps=1e-6, layers_per_block=2,
                engram_injection_layer=2, engram_dim=128, tie_embeddings=True,
                use_ann_engram=True,
                engram_inject_layers=(2, 5),
            )

    def test_config_rejects_duplicate_inject_layers(self):
        """Duplicate layer indices would cause ModuleDict key collisions."""
        with pytest.raises(ValueError, match="duplicate"):
            HarmonyModelConfig(
                num_layers=8, hidden_dim=512, num_query_heads=8, num_kv_heads=4,
                head_dim=64, ffn_dim=1365, vocab_size=32000, max_seq_len=4096,
                rope_theta=1e6, rms_norm_eps=1e-6, layers_per_block=2,
                engram_injection_layer=2, engram_dim=128, tie_embeddings=True,
                engram_inject_layers=(2, 2, 5),
            )

    def test_config_rejects_first_position_out_of_range(self):
        """First layer index out-of-range is still caught (not just last)."""
        with pytest.raises(ValueError, match="outside"):
            HarmonyModelConfig(
                num_layers=8, hidden_dim=512, num_query_heads=8, num_kv_heads=4,
                head_dim=64, ffn_dim=1365, vocab_size=32000, max_seq_len=4096,
                rope_theta=1e6, rms_norm_eps=1e-6, layers_per_block=2,
                engram_injection_layer=2, engram_dim=128, tie_embeddings=True,
                engram_inject_layers=(-1, 2),
            )
