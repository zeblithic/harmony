"""Tests for θ-V-contrast V-contrastive engram injection (ZEB-130)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

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

    def test_retrieve_topk_return_indices(self):
        """retrieve_topk(return_indices=True) must return the gather indices
        used to materialize `retrieved` — this is what the V-contrast branch
        uses to apply a value-only shuffle without recomputing the matmul."""
        torch.manual_seed(0)
        c = _tiny_config()
        c.use_xattn_engram = True
        table = torch.randn(16, c.engram_dim)
        xattn = EngramCrossAttention(c, table, num_heads=2, k_retrieved=4)
        xattn.eval()
        h = torch.randn(2, 5, c.hidden_dim)

        # Two-tuple backward-compat
        two = xattn.retrieve_topk(h)
        assert len(two) == 2

        # Three-tuple with indices
        retrieved, _topk_sims, topk_idx = xattn.retrieve_topk(h, return_indices=True)
        assert topk_idx.shape == (2, 5, 4)
        # Gathering the table with the returned indices must reconstruct the
        # returned `retrieved` — contract the V-contrast path relies on.
        assert torch.allclose(retrieved, xattn.table[topk_idx], atol=0)


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

    def test_seeded_generator_produces_reproducible_aux(self):
        """Two wrappers built with generators seeded the same way must produce
        identical aux-loss sequences across forwards. This is the reproducibility
        contract of --engram-vcontrast-shuffle-seed.

        Note: both the xattn and the wrapper init consume the *global* RNG
        (xavier_uniform_ on o_proj.weight, etc.), so we re-seed before each
        construction to keep module weights bit-identical. The shuffle
        generator is what we're actually testing — it must drive the
        per-step perm independently of global-RNG state.
        """
        def _build(seed):
            torch.manual_seed(0)
            xattn = self._make_xattn()
            sink: list = []
            gen = torch.Generator(device="cpu").manual_seed(seed)
            # Seed global RNG deterministically before wrapper init so
            # o_proj.weight is identical across runs.
            torch.manual_seed(1)
            wrapper = ContrastiveGatedEngramInjection(
                xattn, alpha_init=0.0, aux_loss_sink=sink, shuffle_generator=gen,
            )
            wrapper.train(True)
            return wrapper, sink

        w_a, sink_a = _build(42)
        w_b, sink_b = _build(42)

        torch.manual_seed(7)
        h = torch.randn(2, 5, 64)
        for _ in range(3):
            w_a(h)
            w_b(h)

        assert len(sink_a) == 3 and len(sink_b) == 3
        for a, b in zip(sink_a, sink_b, strict=True):
            assert torch.allclose(a, b, atol=0), (
                "Same-seeded generators must produce bit-identical aux losses; "
                "got divergence — shuffle generator isn't actually driving the perm."
            )

        # Different seed → different aux sequence.
        _, sink_c = _build(99)
        torch.manual_seed(7)
        h2 = torch.randn(2, 5, 64)
        assert torch.equal(h, h2)  # same `h`
        # Re-run both; sink_c should diverge from sink_a on the first forward.
        w_c, sink_c = _build(99)
        w_c(h)
        assert not torch.allclose(sink_a[0], sink_c[0], atol=1e-8), (
            "Different seeds produced identical aux — the seed isn't reaching "
            "the permutation."
        )

    def test_unseeded_shuffle_uses_global_rng(self):
        """With no generator passed, each forward must still randomize — the
        global-RNG path is the default for production runs."""
        torch.manual_seed(0)
        xattn = self._make_xattn()
        sink: list = []
        wrapper = ContrastiveGatedEngramInjection(
            xattn, alpha_init=0.0, aux_loss_sink=sink,
        )
        wrapper.train(True)
        h = torch.randn(2, 5, 64)

        wrapper(h)
        wrapper(h)
        # Two forwards with different global RNG states must give different
        # aux losses (with astronomically high probability for N=16, k=4).
        assert not torch.allclose(sink[0], sink[1], atol=1e-8), (
            "Two unseeded forwards gave identical aux losses — the "
            "per-step permutation isn't actually running."
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

    def test_sink_equivalence_with_ascending_or_out_of_order_declaration(self):
        """train.py relies on `_contrastive_aux_losses` being populated in
        ascending layer-index order regardless of how `engram_inject_layers`
        was declared. Two models built from identical weights but with
        different declaration orders — `(1, 2)` vs `(2, 1)` — must produce
        the same sink sequence. If this broke, the `zip(sorted(...),
        aux_per_layer)` pattern in train.py would silently swap aux values
        between layer-key buckets.
        """
        def build_and_run(inject_layers: tuple[int, ...]) -> list[float]:
            torch.manual_seed(0)  # same RNG → same tables + weights
            c = _tiny_config()
            c.engram_inject_layers = inject_layers
            c.engram_vcontrast_enabled = True
            c.__post_init__()
            model = HarmonyModel(c)
            table = torch.randn(16, c.engram_dim)
            sink: list = []
            injections = {}
            for layer_idx in sorted(inject_layers):  # deterministic build order
                xattn = EngramCrossAttention(c, table, num_heads=2, k_retrieved=4)
                injections[layer_idx] = ContrastiveGatedEngramInjection(
                    xattn, alpha_init=0.0, aux_loss_sink=sink,
                )
            model.attach_gated_engram_injections(injections)
            model._contrastive_aux_losses = sink
            model.train(True)
            torch.manual_seed(1)
            input_ids = torch.randint(0, c.vocab_size, (2, 5))
            _ = model(input_ids)
            return [s.item() for s in sink]

        ascending = build_and_run((1, 2))
        out_of_order = build_and_run((2, 1))
        assert len(ascending) == len(out_of_order) == 2
        # Forward iterates in ascending numerical order, so sink order must
        # be identical whether the declaration was (1, 2) or (2, 1).
        assert ascending == pytest.approx(out_of_order, abs=1e-6), (
            "Sink population order depends on declaration order of "
            "engram_inject_layers — this breaks train.py's per-layer "
            "accumulator zip, which assumes ascending forward order."
        )


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
        # Per-layer columns must match the preset's engram_inject_layers,
        # not the hardcoded (2, 5). Preset `tiny_engram_xattn_capgap_vcontrast`
        # uses the same layers as the capgap preset it extends; whatever those
        # are, their per-layer columns must be present and no extras.
        from ct87.model import HarmonyModelConfig
        cfg = HarmonyModelConfig.tiny_engram_xattn_capgap_vcontrast()
        for i in cfg.engram_inject_layers:
            assert f"vcontrast_aux_L{i}" in header, (
                f"Expected dynamic per-layer column vcontrast_aux_L{i} in CSV "
                f"header; got {header}"
            )
        # Reject any stale hardcoded lowercase-l columns.
        assert "vcontrast_aux_l2" not in header
        assert "vcontrast_aux_l5" not in header


class TestCsvHeaderConditional:
    """V-contrast CSV columns must only appear when the feature is enabled —
    otherwise non-vcontrast runs get polluted with empty columns they don't
    produce data for (Cursor PR 250 review: 'Vcontrast per-layer CSV columns
    emitted when feature disabled')."""

    def test_vcontrast_cols_absent_when_disabled(self, tmp_path):
        """Running a plain synthetic capgap config (no V-contrast) must NOT
        emit any vcontrast_* columns in the CSV header."""
        log_file = tmp_path / "train.csv"
        output_dir = tmp_path / "output"
        result = subprocess.run(
            [
                sys.executable, "-m", "ct87.train",
                "--config", "tiny_engram_xattn_capgap",
                "--synthetic",
                "--steps", "2",
                "--save-every", "0",
                "--log-file", str(log_file),
                "--output-dir", str(output_dir),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=Path(__file__).parent.parent,
            env={**os.environ, "PYTHONPATH": str(Path(__file__).parent.parent)},
        )
        assert result.returncode == 0, (
            f"train.py exited {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )
        header = log_file.read_text().splitlines()[0].split(",")
        assert not any(c.startswith("vcontrast_") for c in header), (
            f"Non-vcontrast run should not emit any vcontrast_* columns; "
            f"got: {[c for c in header if c.startswith('vcontrast_')]}"
        )


class TestCliFlagPresetConsistency:
    """PR 250 CodeRabbit review: --engram-vcontrast must match the preset's
    engram_vcontrast_enabled symmetrically. Preset mutation via flag makes
    CSV logs ambiguous — the recorded --config name should always describe
    what actually ran."""

    def test_flag_without_preset_rejected(self, tmp_path):
        """--engram-vcontrast on a non-vcontrast preset must exit non-zero."""
        result = subprocess.run(
            [
                sys.executable, "-m", "ct87.train",
                "--config", "tiny_engram_xattn_capgap",
                "--engram-vcontrast",
                "--synthetic",
                "--steps", "1",
                "--save-every", "0",
                "--log-file", str(tmp_path / "t.csv"),
                "--output-dir", str(tmp_path / "out"),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=Path(__file__).parent.parent,
            env={**os.environ, "PYTHONPATH": str(Path(__file__).parent.parent)},
        )
        assert result.returncode != 0
        assert "must match the selected preset" in result.stderr

    def test_override_flag_without_vcontrast_rejected(self, tmp_path):
        """--engram-vcontrast-lambda without V-contrast must exit non-zero."""
        result = subprocess.run(
            [
                sys.executable, "-m", "ct87.train",
                "--config", "tiny_engram_xattn_capgap",
                "--engram-vcontrast-lambda", "0.5",
                "--synthetic",
                "--steps", "1",
                "--save-every", "0",
                "--log-file", str(tmp_path / "t.csv"),
                "--output-dir", str(tmp_path / "out"),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=Path(__file__).parent.parent,
            env={**os.environ, "PYTHONPATH": str(Path(__file__).parent.parent)},
        )
        assert result.returncode != 0
        assert "require a V-contrast preset" in result.stderr


class TestShuffleGeneratorCheckpointing:
    """PR 250 CodeRabbit review: capgap_shuffle_gen state must survive
    checkpoint → resume. Otherwise --engram-vcontrast-shuffle-seed replays
    permutations from step 0 on every resume, breaking determinism."""

    def test_capture_and_restore_round_trip(self):
        """Round-trip a generator's state through capture_rng_state +
        restore_rng_state — the restored generator must produce identical
        perms to the original."""
        from ct87.train import capture_rng_state, restore_rng_state

        gen = torch.Generator(device="cpu").manual_seed(42)
        # Advance the generator a few steps so its state differs from the
        # freshly-seeded state — this is what the training loop does.
        for _ in range(7):
            torch.randperm(16, generator=gen, device="cpu")

        state = capture_rng_state(device=None, capgap_shuffle_gen=gen)
        assert "capgap_shuffle_gen" in state

        # Snapshot the next 3 perms from the original generator:
        expected = [
            torch.randperm(16, generator=gen, device="cpu").clone()
            for _ in range(3)
        ]

        # Build a fresh generator, seeded differently, and restore into it.
        gen2 = torch.Generator(device="cpu").manual_seed(9999)
        restore_rng_state(state, device=None, capgap_shuffle_gen=gen2)

        got = [
            torch.randperm(16, generator=gen2, device="cpu").clone()
            for _ in range(3)
        ]

        for e, g in zip(expected, got, strict=True):
            assert torch.equal(e, g), (
                "restore_rng_state didn't restore the shuffle generator's "
                "state — next permutation diverged."
            )

    def test_capture_absent_when_no_generator(self):
        """If V-contrast is off (no generator), the state dict must NOT have
        a capgap_shuffle_gen key — it would be meaningless and would bloat
        checkpoints."""
        from ct87.train import capture_rng_state
        state = capture_rng_state(device=None, capgap_shuffle_gen=None)
        assert "capgap_shuffle_gen" not in state

    def test_restore_silently_skips_when_absent(self):
        """Restoring a pre-θ-V-contrast checkpoint (no capgap_shuffle_gen
        key) must not error when a generator is passed — we degrade to
        'no restore, caller gets a fresh-seeded generator'."""
        from ct87.train import capture_rng_state, restore_rng_state
        legacy_state = capture_rng_state(device=None)  # no gen
        gen = torch.Generator(device="cpu").manual_seed(7)
        # Must not raise:
        restore_rng_state(legacy_state, device=None, capgap_shuffle_gen=gen)
