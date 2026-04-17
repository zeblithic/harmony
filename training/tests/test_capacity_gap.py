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


class TestInitFromFlag:
    """--init-from loads model weights but not training state."""

    def test_init_from_loads_backbone_weights(self, tmp_path):
        """After --init-from, a new model can load a source checkpoint via strict=False."""
        # Build a source checkpoint: tiny baseline (no engram injections).
        src_ckpt = tmp_path / "src" / "checkpoint.pt"
        src_ckpt.parent.mkdir()
        config = HarmonyModelConfig.tiny()
        model = HarmonyModel(config)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": {},
                "step": 500,
                "rng_state": torch.get_rng_state(),
                "config": config,
            },
            src_ckpt,
        )

        # Load into a capgap-configured target (which has extra engram_injections keys
        # after attach — but with no attach called, state_dict is equivalent).
        ckpt = torch.load(src_ckpt, map_location="cpu", weights_only=False)
        new_config = HarmonyModelConfig.tiny_engram_xattn_capgap()
        new_model = HarmonyModel(new_config)
        missing, unexpected = new_model.load_state_dict(
            ckpt["model_state_dict"], strict=False
        )
        # No unexpected keys (source is strict subset of target)
        assert unexpected == []
        # Missing, if any, must only be engram_injections keys.
        for k in missing:
            assert k.startswith("engram_injections"), (
                f"Unexpected missing key outside engram_injections: {k}"
            )

    def test_train_rejects_both_init_from_and_resume_from(self):
        """CLI validation rejects using both flags simultaneously."""
        import os
        import subprocess
        import sys
        # Resolve `training/` (the parent of `tests/`) so the subprocess can
        # `python -m ct87.train` regardless of where pytest was invoked from.
        training_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        result = subprocess.run(
            [
                sys.executable, "-m", "ct87.train",
                "--init-from", "/tmp/nonexistent.pt",
                "--resume-from", "/tmp/nonexistent.pt",
                "--config", "tiny_engram_xattn_capgap",
                "--synthetic",
                "--output-dir", "/tmp/capgap_test",
                "--steps", "1",
            ],
            capture_output=True, text=True, timeout=30,
            cwd=training_root,
        )
        assert result.returncode != 0, f"Expected non-zero exit but got stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        combined = result.stderr + result.stdout
        assert (
            "--init-from" in combined
            and "--resume-from" in combined
            and "mutually exclusive" in combined.lower()
        ), f"Error message did not mention both flags + 'mutually exclusive':\n{combined}"


class TestFreezeBackbone:
    """--freeze-backbone disables grad on all non-engram params."""

    def _build_and_attach(self) -> HarmonyModel:
        config = HarmonyModelConfig.tiny_engram_xattn_capgap()
        model = HarmonyModel(config)
        table = torch.randn(16, config.engram_dim)
        injections = {
            layer_idx: GatedEngramInjection(
                EngramCrossAttention(
                    config, table, num_heads=2, k_retrieved=4,
                ),
                alpha_init=0.0,
            )
            for layer_idx in config.engram_inject_layers
        }
        model.attach_gated_engram_injections(injections)
        return model

    def test_freeze_sets_requires_grad_correctly(self):
        """After freezing, only engram_injections params have requires_grad=True."""
        model = self._build_and_attach()
        from ct87.train import freeze_backbone_for_capgap
        freeze_backbone_for_capgap(model)
        for name, p in model.named_parameters():
            if name.startswith("engram_injections"):
                assert p.requires_grad, f"{name} should be trainable"
            else:
                assert not p.requires_grad, f"{name} should be frozen"

    def test_trainable_param_count_is_small(self):
        """Frozen capgap model has O(engram_injections) trainable params, not O(backbone)."""
        model = self._build_and_attach()
        from ct87.train import freeze_backbone_for_capgap
        freeze_backbone_for_capgap(model)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        # Trainable should be << frozen: the tiny_engram_xattn_capgap config has
        # ~1.4M engram params vs ~39.6M backbone params (~3.6%), well under 10%.
        # (Production backbones are much larger, making this ratio even smaller.)
        assert trainable * 10 < frozen, (
            f"Trainable ({trainable}) should be under 10% of frozen ({frozen})"
        )
