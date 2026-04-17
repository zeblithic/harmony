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


class TestHarmonyModelMultiLayerInjection:
    """HarmonyModel supports GatedEngramInjection at multiple layers via ModuleDict."""

    def _build_model(self) -> HarmonyModel:
        config = HarmonyModelConfig.tiny_engram_xattn_capgap()
        return HarmonyModel(config)

    def _build_injections(
        self, config: HarmonyModelConfig
    ) -> dict[int, GatedEngramInjection]:
        """Build injections using the real EngramCrossAttention API, with
        o_proj overridden to xavier_uniform so the injection produces a
        non-trivial signal (otherwise xattn's zero-init o_proj would make
        all 'gate opens' tests pass trivially)."""
        out: dict[int, GatedEngramInjection] = {}
        table = torch.randn(16, config.engram_dim)
        for layer_idx in config.engram_inject_layers:
            xattn = EngramCrossAttention(
                config, table, num_heads=2, k_retrieved=4,
            )
            nn.init.xavier_uniform_(xattn.o_proj.weight)
            out[layer_idx] = GatedEngramInjection(
                xattn, alpha_init=config.engram_gate_init,
            )
        return out

    def test_engram_injections_default_none(self):
        """Model with empty inject_layers has engram_injections=None."""
        config = HarmonyModelConfig.tiny()  # empty inject_layers
        model = HarmonyModel(config)
        assert model.engram_injections is None

    def test_attach_gated_engram_injections(self):
        """Attaching populates engram_injections as a ModuleDict keyed by layer index."""
        model = self._build_model()
        injections = self._build_injections(model.config)
        model.attach_gated_engram_injections(injections)
        assert isinstance(model.engram_injections, nn.ModuleDict)
        assert set(model.engram_injections.keys()) == {"2", "5"}

    def test_attach_rejects_wrong_layer_indices(self):
        """Attach rejects layer indices not declared in config.engram_inject_layers."""
        model = self._build_model()
        table = torch.randn(16, model.config.engram_dim)
        wrong = {
            7: GatedEngramInjection(
                EngramCrossAttention(
                    model.config, table, num_heads=2, k_retrieved=4,
                )
            ),
        }
        with pytest.raises(ValueError, match="layer_idx"):
            model.attach_gated_engram_injections(wrong)

    def test_attach_rejects_when_config_flag_absent(self):
        """Attach fails if config.engram_inject_layers is empty."""
        config = HarmonyModelConfig.tiny()
        model = HarmonyModel(config)
        with pytest.raises(ValueError, match="engram_inject_layers"):
            model.attach_gated_engram_injections({})

    def test_attach_rejects_empty_dict_when_layers_declared(self):
        """Empty dict against non-empty config raises — prevents a silent
        dead-zone where engram_injections is an empty ModuleDict and the
        forward's `elif` would permanently skip all legacy injection paths."""
        model = self._build_model()
        with pytest.raises(ValueError, match="layer_idx"):
            model.attach_gated_engram_injections({})

    def test_attach_rejects_double_attach(self):
        """Second attach call raises — prevents orphaning params of the
        first ModuleDict from PyTorch's module tree (which would desync
        any optimizer built against those params)."""
        model = self._build_model()
        injections = self._build_injections(model.config)
        model.attach_gated_engram_injections(injections)
        # Second call must raise.
        injections2 = self._build_injections(model.config)
        with pytest.raises(ValueError, match="already set"):
            model.attach_gated_engram_injections(injections2)

    def test_forward_zero_perturbation_at_init(self):
        """With alpha_init=0 on all injections, forward must match a no-injection baseline."""
        torch.manual_seed(123)
        config = HarmonyModelConfig.tiny_engram_xattn_capgap()
        model_capgap = HarmonyModel(config)
        injections = self._build_injections(config)
        model_capgap.attach_gated_engram_injections(injections)
        model_capgap.train(False)

        torch.manual_seed(123)
        config_plain = HarmonyModelConfig.tiny()
        model_plain = HarmonyModel(config_plain)
        model_plain.train(False)

        ids = torch.randint(0, config.vocab_size, (2, 16))
        with torch.no_grad():
            out_capgap = model_capgap(ids)
            out_plain = model_plain(ids)
        assert torch.allclose(out_capgap, out_plain, atol=1e-5)

    def test_forward_diverges_when_gate_opened(self):
        """Opening the gate must cause outputs to diverge from the no-injection baseline."""
        torch.manual_seed(123)
        config = HarmonyModelConfig.tiny_engram_xattn_capgap()
        model = HarmonyModel(config)
        injections = self._build_injections(config)
        model.attach_gated_engram_injections(injections)
        model.train(False)

        ids = torch.randint(0, config.vocab_size, (2, 16))
        with torch.no_grad():
            out_closed = model(ids)
            for injection in model.engram_injections.values():
                injection.alpha.data = torch.tensor(1.0)
            out_open = model(ids)
        assert not torch.allclose(out_closed, out_open, atol=1e-3)

    def test_inject_mult_zero_disables_all_layers(self):
        """Setting engram_inject_mult=0 must zero all multi-layer injections."""
        torch.manual_seed(123)
        config = HarmonyModelConfig.tiny_engram_xattn_capgap()
        model = HarmonyModel(config)
        injections = self._build_injections(config)
        model.attach_gated_engram_injections(injections)
        for injection in model.engram_injections.values():
            injection.alpha.data = torch.tensor(1.0)
        model.train(False)

        torch.manual_seed(123)
        config_plain = HarmonyModelConfig.tiny()
        model_plain = HarmonyModel(config_plain)
        model_plain.train(False)

        ids = torch.randint(0, config.vocab_size, (2, 16))
        with torch.no_grad():
            model.engram_inject_mult = 0.0
            out_zeroed = model(ids)
            out_plain = model_plain(ids)
        assert torch.allclose(out_zeroed, out_plain, atol=1e-5)
