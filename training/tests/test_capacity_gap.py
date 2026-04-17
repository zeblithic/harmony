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
        # No o_proj override needed: GatedEngramInjection.__init__ now
        # re-initializes o_proj to xavier_uniform itself, because the gate
        # takes over the step-0 no-op contract from the wrapped xattn.
        return EngramCrossAttention(c, table, num_heads=2, k_retrieved=4)

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

    def test_no_gradient_dead_zone_with_zero_alpha(self):
        """At alpha_init=0, gradient must still flow to alpha on the first
        backward. Guards against the dual-zero-init dead zone: if the wrapped
        xattn's o_proj were left at its zero-init, we'd have
        tanh(0) * o_proj @ attn = 0 * 0 = 0, with BOTH partials vanishing
        (∂/∂alpha = sech²(0) * 0 = 0 and ∂/∂o_proj = tanh(0) * attn = 0),
        leaving the injection frozen for the entire run. GatedEngramInjection
        breaks this by xavier-initializing the wrapped o_proj."""
        torch.manual_seed(0)
        # Build EngramCrossAttention directly (no _make_xattn helper) to
        # prove the wrapper itself is doing the fix, not the test setup.
        c = _tiny_config()
        table = torch.randn(16, c.engram_dim)
        raw_xattn = EngramCrossAttention(c, table, num_heads=2, k_retrieved=4)
        wrapper = GatedEngramInjection(raw_xattn, alpha_init=0.0)
        assert wrapper.engram_xattn.o_proj.weight.abs().sum().item() > 0.0, (
            "Wrapped o_proj should have been xavier-inited by "
            "GatedEngramInjection.__init__"
        )
        wrapper.train(True)
        h = torch.randn(2, 5, 64)
        out = wrapper(h)
        # out.sum() gives uniform ∂L/∂out = 1 per element, so
        # ∂L/∂alpha = sech²(0) * sum(xattn_out) = sum(xattn_out). This stays
        # well-defined even though out itself is identically zero at
        # alpha_init=0. Using out.pow(2).mean() would zero the upstream
        # gradient and give a false-pass regardless of the bug.
        loss = out.sum()
        loss.backward()
        assert wrapper.alpha.grad is not None
        assert wrapper.alpha.grad.abs().item() > 1e-8, (
            f"alpha.grad vanished (={wrapper.alpha.grad.item()!r}); "
            "the injection would be frozen for the entire run."
        )

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

    def test_config_rejects_float_inject_layer(self):
        """Float entries (2.0) would stringify to "2.0" and miss str(2) probing."""
        with pytest.raises(TypeError, match="plain int"):
            HarmonyModelConfig(
                num_layers=8, hidden_dim=512, num_query_heads=8, num_kv_heads=4,
                head_dim=64, ffn_dim=1365, vocab_size=32000, max_seq_len=4096,
                rope_theta=1e6, rms_norm_eps=1e-6, layers_per_block=2,
                engram_injection_layer=2, engram_dim=128, tie_embeddings=True,
                engram_inject_layers=(2.0, 5),
            )

    def test_config_rejects_bool_inject_layer(self):
        """Bool entries (True/False) pass isinstance(int) but stringify to
        'True'/'False', which forward() would never probe."""
        with pytest.raises(TypeError, match="plain int"):
            HarmonyModelConfig(
                num_layers=8, hidden_dim=512, num_query_heads=8, num_kv_heads=4,
                head_dim=64, ffn_dim=1365, vocab_size=32000, max_seq_len=4096,
                rope_theta=1e6, rms_norm_eps=1e-6, layers_per_block=2,
                engram_injection_layer=2, engram_dim=128, tie_embeddings=True,
                engram_inject_layers=(True, 5),
            )


class TestHarmonyModelMultiLayerInjection:
    """HarmonyModel supports GatedEngramInjection at multiple layers via ModuleDict."""

    def _build_model(self) -> HarmonyModel:
        config = HarmonyModelConfig.tiny_engram_xattn_capgap()
        return HarmonyModel(config)

    def _build_injections(
        self, config: HarmonyModelConfig
    ) -> dict[int, GatedEngramInjection]:
        """Build injections using the real EngramCrossAttention API.

        No manual o_proj override is needed: GatedEngramInjection.__init__
        re-initializes the wrapped o_proj to xavier_uniform itself (the
        gate takes over xattn's step-0 no-op contract). This mirrors the
        production path in train.py."""
        out: dict[int, GatedEngramInjection] = {}
        table = torch.randn(16, config.engram_dim)
        for layer_idx in config.engram_inject_layers:
            xattn = EngramCrossAttention(
                config, table, num_heads=2, k_retrieved=4,
            )
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

    def test_forward_rejects_configured_capgap_without_attach(self):
        """Multi-layer capgap config without attach_gated_engram_injections()
        must raise in forward() — otherwise the `elif` branch would silently
        evaluate to a no-op at every layer and the misuse would go unnoticed."""
        model = self._build_model()
        assert model.engram_injections is None
        x = torch.randint(0, model.config.vocab_size, (1, 16))
        with pytest.raises(RuntimeError, match="attach_gated_engram_injections"):
            model(x)

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

    def test_attach_rejects_shared_injection_instance(self):
        """The same GatedEngramInjection instance cannot back two layer slots.
        nn.ModuleDict silently accepts duplicate module references, which would
        make both layers share a single alpha + xattn parameter set (double
        gradient contribution per step, broken per-layer gate logging)."""
        model = self._build_model()
        table = torch.randn(16, model.config.engram_dim)
        shared = GatedEngramInjection(
            EngramCrossAttention(
                model.config, table, num_heads=2, k_retrieved=4,
            ),
            alpha_init=0.0,
        )
        reused = {layer_idx: shared for layer_idx in model.config.engram_inject_layers}
        with pytest.raises(ValueError, match="same GatedEngramInjection instance"):
            model.attach_gated_engram_injections(reused)

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

        # Load into a capgap-configured target WITH injections attached, so
        # engram_injections.* keys actually exist in the target state dict.
        # Without this attach, missing would be trivially empty and this test
        # would not observe regressions in the --init-from path's handling of
        # engram-injection keys.
        ckpt = torch.load(src_ckpt, map_location="cpu", weights_only=False)
        new_config = HarmonyModelConfig.tiny_engram_xattn_capgap()
        new_model = HarmonyModel(new_config)
        table = torch.randn(16, new_config.engram_dim)
        new_model.attach_gated_engram_injections({
            layer_idx: GatedEngramInjection(
                EngramCrossAttention(
                    new_config, table, num_heads=2, k_retrieved=4,
                ),
                alpha_init=0.0,
            )
            for layer_idx in new_config.engram_inject_layers
        })
        missing, unexpected = new_model.load_state_dict(
            ckpt["model_state_dict"], strict=False
        )
        # No unexpected keys (source is strict subset of target).
        assert unexpected == []
        # Missing must be non-empty (the engram_injections.* params we just
        # attached are not in the source checkpoint) and must be *only*
        # engram_injections.* keys.
        assert len(missing) > 0, (
            "Expected engram_injections.* to be missing after attach; empty "
            "missing list means the test can't catch regressions in the "
            "--init-from handling of engram-injection keys."
        )
        for k in missing:
            assert k.startswith("engram_injections."), (
                f"Unexpected missing key outside engram_injections: {k}"
            )

    def test_train_rejects_both_init_from_and_resume_from(self, tmp_path):
        """CLI validation rejects using both flags simultaneously."""
        import os
        import subprocess
        import sys
        # Resolve `training/` (the parent of `tests/`) so the subprocess can
        # `python -m ct87.train` regardless of where pytest was invoked from.
        training_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Paths don't need to exist — argparse mutex fires before any file IO.
        # Use tmp_path rather than hardcoded /tmp locations so parallel runs
        # can't collide and the tests stay OS-agnostic.
        missing_ckpt = str(tmp_path / "nonexistent.pt")
        result = subprocess.run(
            [
                sys.executable, "-m", "ct87.train",
                "--init-from", missing_ckpt,
                "--resume-from", missing_ckpt,
                "--config", "tiny_engram_xattn_capgap",
                "--synthetic",
                "--output-dir", str(tmp_path / "capgap_test"),
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
            if name.startswith("engram_injections."):
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

    def test_freeze_rejects_model_without_injections(self):
        """Freezing without injections would leave the optimizer empty — must error."""
        config = HarmonyModelConfig.tiny()  # no engram_inject_layers, no attach
        model = HarmonyModel(config)
        from ct87.train import freeze_backbone_for_capgap
        with pytest.raises(RuntimeError, match="engram_injections"):
            freeze_backbone_for_capgap(model)


class TestCapgapTrainingLoopWiring:
    """Verify the capgap preset attaches GatedEngramInjection modules at the
    declared layers when the training loop runs."""

    def test_capgap_config_dispatch_attaches_injections(self, tmp_path):
        """Running train.py with --config tiny_engram_xattn_capgap must
        attach engram_injections as a ModuleDict before the first step."""
        import os
        import subprocess
        import sys

        # Build a minimal "source" checkpoint so --init-from has something
        # to load (needed because the capgap path implies init-from in practice).
        src_ckpt = tmp_path / "src" / "checkpoint.pt"
        src_ckpt.parent.mkdir()
        src_config = HarmonyModelConfig.tiny()
        torch.manual_seed(0)
        src_model = HarmonyModel(src_config)
        torch.save(
            {
                "model_state_dict": src_model.state_dict(),
                "optimizer_state_dict": {},
                "step": 0,
                "rng_state": torch.get_rng_state(),
                "config": src_config,
            },
            src_ckpt,
        )

        training_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out_dir = tmp_path / "out"
        result = subprocess.run(
            [
                sys.executable, "-m", "ct87.train",
                "--init-from", str(src_ckpt),
                "--freeze-backbone",
                "--config", "tiny_engram_xattn_capgap",
                "--synthetic",
                "--output-dir", str(out_dir),
                "--steps", "1",
                "--batch-size", "2",
                "--seq-len", "32",
                "--checkpoint-interval", "1",
            ],
            capture_output=True, text=True, timeout=180,
            cwd=training_root,
        )
        # If this fails, likely the capgap wiring didn't actually build + attach
        # the injections. Print full output for diagnosis.
        assert result.returncode == 0, (
            f"train.py exited {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
        # Should see the capgap attach log line.
        assert "[capgap]" in (result.stdout + result.stderr), (
            f"Expected [capgap] log line in output:\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
        # Checkpoint must contain engram_injections.* params.
        out_ckpt_path = out_dir / "checkpoint.pt"
        assert out_ckpt_path.exists(), "train.py didn't save checkpoint"
        out_ckpt = torch.load(out_ckpt_path, map_location="cpu", weights_only=False)
        injection_params = [
            k for k in out_ckpt["model_state_dict"].keys()
            if k.startswith("engram_injections.")
        ]
        assert len(injection_params) > 0, (
            f"No engram_injections params in saved state dict. Got keys: "
            f"{list(out_ckpt['model_state_dict'].keys())[:20]}..."
        )


class TestCapgapSmokeIntegration:
    """End-to-end: init from source checkpoint, train N steps frozen, eval."""

    def test_frozen_backbone_params_unchanged_after_training(self, tmp_path):
        """After N training steps with --freeze-backbone, backbone weights must be bitwise unchanged."""
        import copy
        import os
        import subprocess
        import sys

        # Step A: build a source checkpoint with random weights (substitute for β).
        src_ckpt = tmp_path / "src" / "checkpoint.pt"
        src_ckpt.parent.mkdir()
        src_config = HarmonyModelConfig.tiny()
        torch.manual_seed(42)
        src_model = HarmonyModel(src_config)
        torch.save(
            {
                "model_state_dict": src_model.state_dict(),
                "optimizer_state_dict": {},
                "step": 1000,
                "rng_state": torch.get_rng_state(),
                "config": src_config,
            },
            src_ckpt,
        )

        # Snapshot the source state dict for later comparison.
        src_state = copy.deepcopy(src_model.state_dict())

        # Step B: run train.py with --init-from + --freeze-backbone + capgap preset.
        training_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out_dir = tmp_path / "run"
        result = subprocess.run(
            [
                sys.executable, "-m", "ct87.train",
                "--init-from", str(src_ckpt),
                "--freeze-backbone",
                "--config", "tiny_engram_xattn_capgap",
                "--synthetic",
                "--output-dir", str(out_dir),
                "--steps", "3",
                "--batch-size", "2",
                "--seq-len", "32",
                "--checkpoint-interval", "1",
            ],
            capture_output=True, text=True, timeout=180,
            cwd=training_root,
        )
        assert result.returncode == 0, (
            f"train.py failed (exit {result.returncode}):\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

        # Step C: load the output checkpoint and confirm backbone is bit-identical.
        out_ckpt_path = out_dir / "checkpoint.pt"
        assert out_ckpt_path.exists(), "checkpoint.pt not written after training"
        out_ckpt = torch.load(out_ckpt_path, map_location="cpu", weights_only=False)
        out_state = out_ckpt["model_state_dict"]
        for name, src_tensor in src_state.items():
            out_tensor = out_state.get(name)
            assert out_tensor is not None, (
                f"Source param '{name}' missing from output checkpoint"
            )
            assert torch.equal(src_tensor, out_tensor), (
                f"Backbone param '{name}' changed despite --freeze-backbone"
            )

        # Step D: confirm engram_injections params DO exist in the output.
        injection_param_names = [
            n for n in out_state.keys() if n.startswith("engram_injections.")
        ]
        assert len(injection_param_names) > 0, (
            "No engram_injections params saved — check attach wiring"
        )

    def test_zero_injection_eval_works_on_capgap_checkpoint(self, tmp_path):
        """After a capgap run, --zero-injection-eval must produce a sensible delta."""
        import os
        import subprocess
        import sys

        src_ckpt = tmp_path / "src" / "checkpoint.pt"
        src_ckpt.parent.mkdir()
        src_config = HarmonyModelConfig.tiny()
        torch.manual_seed(7)
        src_model = HarmonyModel(src_config)
        torch.save(
            {
                "model_state_dict": src_model.state_dict(),
                "optimizer_state_dict": {},
                "step": 100,
                "rng_state": torch.get_rng_state(),
                "config": src_config,
            },
            src_ckpt,
        )
        training_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        run_dir = tmp_path / "run"
        r = subprocess.run(
            [
                sys.executable, "-m", "ct87.train",
                "--init-from", str(src_ckpt),
                "--freeze-backbone",
                "--config", "tiny_engram_xattn_capgap",
                "--synthetic",
                "--output-dir", str(run_dir),
                "--steps", "5",
                "--batch-size", "2",
                "--seq-len", "32",
                "--checkpoint-interval", "1",
            ],
            capture_output=True, text=True, timeout=180,
            cwd=training_root,
        )
        assert r.returncode == 0, (
            f"train failed (exit {r.returncode}):\n"
            f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
        )

        eval_dir = tmp_path / "eval"
        r2 = subprocess.run(
            [
                sys.executable, "-m", "ct87.train",
                "--resume-from", str(run_dir / "checkpoint.pt"),
                "--zero-injection-eval",
                "--config", "tiny_engram_xattn_capgap",
                "--synthetic",
                "--output-dir", str(eval_dir),
                "--batch-size", "2",
                "--seq-len", "32",
            ],
            capture_output=True, text=True, timeout=180,
            cwd=training_root,
        )
        assert r2.returncode == 0, (
            f"eval failed (exit {r2.returncode}):\n"
            f"STDOUT:\n{r2.stdout}\nSTDERR:\n{r2.stderr}"
        )
        # Must report val_loss with/without and a delta.
        combined = r2.stdout.lower() + r2.stderr.lower()
        assert "delta" in combined, (
            f"--zero-injection-eval did not report delta:\n"
            f"STDOUT:\n{r2.stdout}\nSTDERR:\n{r2.stderr}"
        )


class TestHardFailGuards:
    """Guardrails added after PR #247 review: silent misconfiguration must become
    a hard exit so GPU-hours are not burned on confounded experiments."""

    def _run_train(self, *argv, timeout=30):
        import os
        import subprocess
        import sys

        training_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return subprocess.run(
            [sys.executable, "-m", "ct87.train", *argv],
            capture_output=True, text=True, timeout=timeout,
            cwd=training_root,
        )

    def _save_ckpt(self, path, config, with_config_key: bool, with_all_keys: bool = True):
        """Write a .pt checkpoint for --init-from.

        with_config_key=False simulates a legacy checkpoint missing config metadata.
        with_all_keys=False simulates a checkpoint missing non-engram backbone keys.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        model = HarmonyModel(config)
        state = model.state_dict()
        if not with_all_keys:
            # Drop half the backbone keys to simulate partial/renamed source.
            keys = sorted(state.keys())
            for k in keys[: len(keys) // 2]:
                state.pop(k)
        payload = {
            "model_state_dict": state,
            "optimizer_state_dict": {},
            "step": 0,
            "rng_state": torch.get_rng_state(),
        }
        if with_config_key:
            payload["config"] = config
        torch.save(payload, path)

    def test_init_from_rejects_checkpoint_without_config(self, tmp_path):
        """Legacy --init-from source missing a 'config' field must exit 2."""
        src = tmp_path / "legacy.pt"
        self._save_ckpt(src, HarmonyModelConfig.tiny(), with_config_key=False)
        r = self._run_train(
            "--init-from", str(src),
            "--config", "tiny_engram_xattn_capgap",
            "--synthetic",
            "--output-dir", str(tmp_path / "out"),
            "--steps", "1",
        )
        assert r.returncode == 2, (
            f"Expected exit 2 on missing config metadata, got {r.returncode}\n"
            f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
        )
        assert "no config metadata" in (r.stderr + r.stdout).lower()

    def test_init_from_accepts_no_config_with_allow_partial_init(self, tmp_path):
        """--allow-partial-init permits loading a legacy checkpoint missing config."""
        src = tmp_path / "legacy.pt"
        self._save_ckpt(src, HarmonyModelConfig.tiny(), with_config_key=False)
        r = self._run_train(
            "--init-from", str(src),
            "--allow-partial-init",
            "--config", "tiny_engram_xattn_capgap",
            "--synthetic",
            "--output-dir", str(tmp_path / "out"),
            "--steps", "1",
            "--batch-size", "2",
            "--seq-len", "32",
            "--checkpoint-interval", "1",
            timeout=180,
        )
        assert r.returncode == 0, (
            f"--allow-partial-init should permit missing config, got exit "
            f"{r.returncode}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
        )

    def test_init_from_rejects_missing_backbone_keys(self, tmp_path):
        """--init-from must exit 2 when non-engram backbone keys are missing."""
        src = tmp_path / "partial.pt"
        self._save_ckpt(
            src, HarmonyModelConfig.tiny(),
            with_config_key=True, with_all_keys=False,
        )
        r = self._run_train(
            "--init-from", str(src),
            "--config", "tiny_engram_xattn_capgap",
            "--synthetic",
            "--output-dir", str(tmp_path / "out"),
            "--steps", "1",
            "--batch-size", "2",
            "--seq-len", "32",
        )
        assert r.returncode == 2, (
            f"Expected exit 2 on missing backbone keys, got {r.returncode}\n"
            f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
        )
        lower = (r.stderr + r.stdout).lower()
        assert "missing" in lower and "non-engram" in lower

    def test_capgap_requires_xattn_table_without_synthetic(self, tmp_path):
        """Capgap preset must refuse to build a random placeholder table in real runs."""
        src = tmp_path / "src.pt"
        self._save_ckpt(src, HarmonyModelConfig.tiny(), with_config_key=True)
        # Use a non-existent --data path so we never reach the dataloader —
        # the xattn-table guard fires first. tmp_path keeps the fake path
        # per-test and OS-agnostic.
        r = self._run_train(
            "--init-from", str(src),
            "--config", "tiny_engram_xattn_capgap",
            "--data", str(tmp_path / "does-not-exist"),
            "--output-dir", str(tmp_path / "out"),
            "--steps", "1",
        )
        assert r.returncode == 2, (
            f"Expected exit 2 when capgap run is missing --engram-xattn-table "
            f"and not --synthetic, got {r.returncode}\n"
            f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
        )
        assert "--engram-xattn-table" in (r.stderr + r.stdout)

    def test_freeze_backbone_rejects_coconut(self, tmp_path):
        """--freeze-backbone + --coconut must exit 2 (aux modules would still train)."""
        r = self._run_train(
            "--freeze-backbone",
            "--coconut",
            "--think-token-id", "0",
            "--config", "tiny_engram_xattn_capgap",
            "--synthetic",
            "--output-dir", str(tmp_path / "out"),
            "--steps", "1",
        )
        assert r.returncode == 2, (
            f"Expected exit 2 on --freeze-backbone + --coconut, got "
            f"{r.returncode}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
        )
        combined = r.stderr + r.stdout
        assert "--freeze-backbone" in combined and "--coconut" in combined

    def test_freeze_backbone_rejects_uq_head(self, tmp_path):
        """--freeze-backbone + --uq-head must also fail fast."""
        r = self._run_train(
            "--freeze-backbone",
            "--uq-head",
            "--config", "tiny_engram_xattn_capgap",
            "--synthetic",
            "--output-dir", str(tmp_path / "out"),
            "--steps", "1",
        )
        assert r.returncode == 2, (
            f"Expected exit 2 on --freeze-backbone + --uq-head, got "
            f"{r.returncode}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
        )
        assert "--uq-head" in (r.stderr + r.stdout)
