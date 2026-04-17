"""Tests for θ-V-contrast V-contrastive engram injection (ZEB-130)."""
from __future__ import annotations

import os
import pytest
import torch

from ct87.engram import (
    ContrastiveGatedEngramInjection,
    EngramCrossAttention,
    GatedEngramInjection,
)
from ct87.model import HarmonyModel, HarmonyModelConfig


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


class TestAttentionBlockRefactor:
    """The post-retrieval attention pipeline must be callable on caller-supplied
    (retrieved, topk_sims) so the V-contrast subclass can run it against a
    shuffled table without a second `retrieve_topk` call duplicating the
    matmul against `self.table_normalized`."""

    def test_attention_block_matches_forward(self):
        torch.manual_seed(0)
        c = _tiny_config()
        c.use_xattn_engram = True
        table = torch.randn(16, c.engram_dim)
        xattn = EngramCrossAttention(c, table, num_heads=2, k_retrieved=4)
        # Init o_proj non-zero so the residual is non-trivial (it's
        # zero-init'd by default for the legacy delta path; we override
        # here so the test catches a refactor that drops a step).
        torch.nn.init.xavier_uniform_(xattn.o_proj.weight)
        xattn.eval()

        h = torch.randn(2, 5, c.hidden_dim)
        retrieved, topk_sims = xattn.retrieve_topk(h)

        out_via_forward = xattn(h)
        out_via_block = xattn._attention_block(h, retrieved, topk_sims)
        assert torch.allclose(out_via_forward, out_via_block, atol=1e-6), (
            "_attention_block must reproduce forward() bit-for-bit when "
            "fed the same retrieved tensors."
        )


class TestContrastiveGatedEngramInjection:

    def _make_xattn(self) -> EngramCrossAttention:
        c = _tiny_config()
        c.use_xattn_engram = True
        table = torch.randn(16, c.engram_dim)
        return EngramCrossAttention(c, table, num_heads=2, k_retrieved=4)

    def test_residual_matches_parent_when_eval(self):
        """In eval mode, the contrastive subclass must produce the same residual
        as the parent — the shuffled branch is training-only."""
        torch.manual_seed(0)
        xattn = self._make_xattn()
        sink: list = []
        wrapper = ContrastiveGatedEngramInjection(
            xattn, alpha_init=0.5, aux_loss_sink=sink,
        )
        wrapper.eval()
        h = torch.randn(2, 5, 64)

        out_contrastive = wrapper(h)
        # Compare to the parent's forward against the same xattn module by
        # re-running just the parent path:
        gate = torch.tanh(wrapper.alpha).to(dtype=out_contrastive.dtype)
        expected = gate * xattn(h)
        assert torch.allclose(out_contrastive, expected, atol=1e-6)
        assert sink == [], "Sink must be empty in eval mode"

    def test_training_appends_one_aux_loss(self):
        torch.manual_seed(0)
        xattn = self._make_xattn()
        sink: list = []
        wrapper = ContrastiveGatedEngramInjection(
            xattn, alpha_init=0.5, aux_loss_sink=sink,
        )
        wrapper.train(True)
        h = torch.randn(2, 5, 64)
        _ = wrapper(h)
        assert len(sink) == 1
        aux = sink[0]
        assert aux.dim() == 0, "aux loss must be a scalar"
        assert 0.0 <= aux.item() <= 1.0, "(cos)^2 is bounded in [0, 1]"

    def test_training_with_no_sink_skips_aux_compute(self):
        """If aux_loss_sink is None, training mode must not allocate the
        shuffled branch — used as a guard for accidentally double-paying
        for the aux compute when V-contrast is disabled."""
        torch.manual_seed(0)
        xattn = self._make_xattn()
        wrapper = ContrastiveGatedEngramInjection(
            xattn, alpha_init=0.5, aux_loss_sink=None,
        )
        wrapper.train(True)
        h = torch.randn(2, 5, 64)
        # Should not raise and should produce a valid residual.
        out = wrapper(h)
        assert out.shape == h.shape

    def test_per_step_permutation_differs(self):
        """Two consecutive forwards in training mode must produce different
        aux-loss values almost-surely (different perms → different shuf branches).
        We assert different aux scalars rather than reach into the perm tensor
        directly so the test stays robust to internal refactors."""
        torch.manual_seed(0)
        xattn = self._make_xattn()
        sink: list = []
        wrapper = ContrastiveGatedEngramInjection(
            xattn, alpha_init=0.5, aux_loss_sink=sink,
        )
        wrapper.train(True)
        h = torch.randn(2, 5, 64)
        _ = wrapper(h)
        _ = wrapper(h)
        assert len(sink) == 2
        assert sink[0].item() != sink[1].item(), (
            "Per-step re-permutation must produce two distinct aux losses; "
            "if these match exactly, the perm is being cached or reused."
        )

    def test_gradient_flows_to_v_proj_through_aux(self):
        """The aux loss must produce non-zero gradient on v_proj — that's
        the load-bearing pressure on V toward content-sensitivity."""
        torch.manual_seed(0)
        xattn = self._make_xattn()
        sink: list = []
        wrapper = ContrastiveGatedEngramInjection(
            xattn, alpha_init=0.0, aux_loss_sink=sink,
        )
        wrapper.train(True)
        h = torch.randn(2, 5, 64)
        _ = wrapper(h)
        assert len(sink) == 1
        sink[0].backward()
        v_grad = wrapper.engram_xattn.v_proj.weight.grad
        assert v_grad is not None
        assert v_grad.abs().sum().item() > 1e-8, (
            "Aux loss must flow gradient to v_proj — that's the whole point."
        )

    def test_residual_does_not_depend_on_shuf_branch(self):
        """The shuffled branch must NOT perturb the residual. Two training-mode
        forwards on the same input (with sink reset between) must produce the
        same residual, modulo differences in the shuffle's effect on the residual
        — which must be zero. We verify by setting torch.manual_seed before each
        forward AND ensuring the residual depends only on the unshuffled path."""
        torch.manual_seed(0)
        xattn = self._make_xattn()
        sink: list = []
        wrapper = ContrastiveGatedEngramInjection(
            xattn, alpha_init=0.5, aux_loss_sink=sink,
        )
        wrapper.train(True)
        h = torch.randn(2, 5, 64)

        # Two forwards: the residual must be identical even though the
        # internal randperm produces different shuf branches.
        torch.manual_seed(123)
        out1 = wrapper(h).clone()
        torch.manual_seed(456)
        out2 = wrapper(h).clone()
        assert torch.allclose(out1, out2, atol=1e-6), (
            "Residual changed between forwards even though only the shuffled "
            "branch's RNG should differ — the shuf branch is leaking into "
            "the residual."
        )


class TestModelAuxLossSink:

    def _make_capgap_vcontrast_model(self):
        """Build a tiny model + ContrastiveGatedEngramInjection wrappers
        without going through train.py — keeps the test self-contained."""
        c = _tiny_config()
        c.engram_inject_layers = (1,)
        c.engram_vcontrast_enabled = True
        c.__post_init__()
        model = HarmonyModel(c)
        table = torch.randn(16, c.engram_dim)
        sink: list = []
        injections = {}
        for layer_idx in c.engram_inject_layers:
            xattn = EngramCrossAttention(c, table, num_heads=2, k_retrieved=4)
            injections[layer_idx] = ContrastiveGatedEngramInjection(
                xattn, alpha_init=0.0, aux_loss_sink=sink,
            )
        model.attach_gated_engram_injections(injections)
        model._contrastive_aux_losses = sink
        return model, sink

    def test_default_attribute_is_none(self):
        c = _tiny_config()
        model = HarmonyModel(c)
        assert model._contrastive_aux_losses is None

    def test_forward_clears_list_in_training(self):
        model, sink = self._make_capgap_vcontrast_model()
        # Pre-load the sink with a stale value to verify forward clears it
        # before the wrappers append.
        sink.append(torch.tensor(99.0))
        model.train(True)
        input_ids = torch.randint(0, model.config.vocab_size, (2, 5))
        _ = model(input_ids)
        # After forward, the sink should contain ONLY the per-layer entries
        # appended during this forward (one per injection layer), not the
        # stale 99.0 we put in.
        assert len(sink) == len(model.config.engram_inject_layers)
        for entry in sink:
            assert entry.item() != 99.0

    def test_forward_does_not_clear_in_eval(self):
        """In eval mode the wrappers don't append, but model.forward must
        also not clear — so a caller's pre-loaded list survives. (Edge case
        for resumable forensic flows that may want to inspect the sink.)"""
        model, sink = self._make_capgap_vcontrast_model()
        sink.append(torch.tensor(42.0))
        model.eval()
        input_ids = torch.randint(0, model.config.vocab_size, (2, 5))
        _ = model(input_ids)
        assert len(sink) == 1
        assert sink[0].item() == 42.0


class TestOneStepIntegration:
    """One forward + backward + optimizer step exercises the full V-contrast wiring
    end-to-end. Catches integration bugs that the unit tests miss (e.g. sink not
    cleared between steps, lambda not applied, gradient not flowing)."""

    def test_one_step_with_vcontrast(self):
        torch.manual_seed(0)
        c = _tiny_config()
        c.engram_inject_layers = (1,)
        c.engram_vcontrast_enabled = True
        c.engram_vcontrast_lambda = 1.0
        c.engram_vcontrast_warmup_steps = 1  # use full lambda from step 0
        c.__post_init__()

        model = HarmonyModel(c)
        table = torch.randn(16, c.engram_dim)
        sink: list = []
        injections = {}
        for layer_idx in c.engram_inject_layers:
            xattn = EngramCrossAttention(c, table, num_heads=2, k_retrieved=4)
            injections[layer_idx] = ContrastiveGatedEngramInjection(
                xattn, alpha_init=0.0, aux_loss_sink=sink,
            )
        model.attach_gated_engram_injections(injections)
        model._contrastive_aux_losses = sink

        # Force the optimizer to see only the engram_injections params, mirroring
        # the --freeze-backbone path so this test is sensitive to V-contrast
        # gradient flow rather than to backbone gradient noise.
        engram_params = list(model.engram_injections.parameters())
        opt = torch.optim.SGD(engram_params, lr=1e-2)

        model.train(True)
        input_ids = torch.randint(0, c.vocab_size, (2, 5))
        targets = torch.randint(0, c.vocab_size, (2, 5))

        # Snapshot v_proj before the step.
        v_before = (
            model.engram_injections["1"]
            .engram_xattn.v_proj.weight.detach().clone()
        )

        logits = model(input_ids)
        lm_loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, c.vocab_size), targets.reshape(-1),
        )
        # At alpha_init=0, the residual is identically zero so LM loss has
        # NO gradient path to engram params via the residual. The aux loss
        # is therefore the *only* source of engram-param gradient — perfect
        # for verifying V-contrast wiring in isolation.
        assert len(sink) == 1, "Aux sink must hold exactly one entry per inject layer"
        aux = sink[0]
        # Lambda is 1.0 with warmup=1, so step=0 schedule returns 0.0 at step 0;
        # but step 1 returns full lambda. Exercise that path explicitly.
        from ct87.train import lambda_schedule
        lam_step1 = lambda_schedule(1, 1, 1.0)
        assert lam_step1 == 1.0
        total_loss = lm_loss + lam_step1 * aux

        opt.zero_grad()
        total_loss.backward()
        opt.step()

        v_after = (
            model.engram_injections["1"]
            .engram_xattn.v_proj.weight.detach().clone()
        )
        delta = (v_after - v_before).abs().sum().item()
        assert delta > 1e-8, (
            "v_proj weight unchanged after a V-contrast aux-loss step — the "
            "aux gradient is not reaching V."
        )


import subprocess
import sys
from pathlib import Path


class TestTrainPyVContrastSmoke:

    def test_train_steps_with_vcontrast(self, tmp_path):
        """Run train.py for 2 steps with --synthetic + --engram-vcontrast and
        verify the CSV log has the new vcontrast_* columns populated."""
        log_file = tmp_path / "train.csv"
        output_dir = tmp_path / "output"
        result = subprocess.run(
            [
                sys.executable, "-m", "ct87.train",
                "--config", "tiny_engram_xattn_capgap_vcontrast",
                "--engram-vcontrast",
                "--synthetic",
                "--steps", "2",
                "--save-every", "0",
                "--log-file", str(log_file),
                "--output-dir", str(output_dir),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=Path(__file__).parent.parent,  # training/
            env={**os.environ, "PYTHONPATH": str(Path(__file__).parent.parent)},
        )
        assert result.returncode == 0, (
            f"train.py exited {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )
        # Confirm V-contrast banner printed during setup
        assert "ContrastiveGatedEngramInjection" in result.stdout
        # Confirm CSV header includes the V-contrast columns
        lines = log_file.read_text().splitlines()
        header = lines[0].split(",")
        assert "vcontrast_aux_loss" in header
        assert "vcontrast_lambda" in header
