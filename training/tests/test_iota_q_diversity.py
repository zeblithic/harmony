"""Tests for iota-Q-diversity (ZEB-130): MoE load-balancing aux loss on
retrieval-row marginal distribution. Spec:
docs/superpowers/specs/2026-04-17-iota-q-diversity-design.md
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from ct87.model import HarmonyModelConfig


REPO_TRAINING_ROOT = Path(__file__).resolve().parent.parent


def _run_train_py(args: list[str]) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_TRAINING_ROOT)
    return subprocess.run(
        [sys.executable, "ct87/train.py", *args],
        cwd=REPO_TRAINING_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )


class TestConfigValidation:
    def test_qdiv_defaults(self):
        c = HarmonyModelConfig.tiny_engram_xattn_capgap()
        assert c.engram_qdiv_enabled is False
        assert c.engram_qdiv_lambda == 0.01
        assert c.engram_qdiv_warmup_steps == 200

    def test_qdiv_requires_inject_layers(self):
        c = HarmonyModelConfig.tiny_engram_xattn_capgap()
        c.engram_inject_layers = ()
        c.engram_qdiv_enabled = True
        with pytest.raises(ValueError, match="engram_inject_layers"):
            c.__post_init__()

    def test_qdiv_lambda_must_be_non_negative(self):
        c = HarmonyModelConfig.tiny_engram_xattn_capgap()
        c.engram_qdiv_enabled = True
        c.engram_qdiv_lambda = -0.1
        with pytest.raises(ValueError, match="engram_qdiv_lambda"):
            c.__post_init__()

    def test_qdiv_warmup_must_be_non_negative(self):
        c = HarmonyModelConfig.tiny_engram_xattn_capgap()
        c.engram_qdiv_enabled = True
        c.engram_qdiv_warmup_steps = -1
        with pytest.raises(ValueError, match="engram_qdiv_warmup_steps"):
            c.__post_init__()


class TestComputeQdivAux:
    """Unit tests for the MoE load-balancing loss helper.

    Formula: L = N * sum_i f[i] * P[i], where
      f[i] = fraction of (B*L*k) hard top-k selections on row i (detached)
      P[i] = sum over (B,L,H,j) of attn[b,l,h,j] when topk_idx[b,l,j]==i,
             divided by (B*L*H)
    """

    def test_uniform_selection_gives_unit_loss(self):
        # Construct uniform selection: N=100 rows, B=1 L=100 H=4 k=4.
        # Each row selected exactly 4 times (once per l-position across k),
        # each head's attention uniform 1/k over the 4 selected rows.
        from ct87.engram import compute_qdiv_aux
        N, B, L, H, k = 100, 1, 100, 4, 4
        # Each l-position selects rows l, l+1, l+2, l+3 (mod N) — so each row
        # gets selected k times total across l.
        idx = torch.arange(L).unsqueeze(1) + torch.arange(k).unsqueeze(0)
        idx = (idx % N).unsqueeze(0).expand(B, L, k).contiguous()
        attn = torch.full((B, L, H, k), 1.0 / k)
        loss = compute_qdiv_aux(idx, attn, N)
        # f = 1/N everywhere selected, P = 1/N everywhere selected,
        # sum over rows = N * (1/N) * (1/N) = 1/N, times N = 1.0
        assert loss.item() == pytest.approx(1.0, abs=1e-5)

    def test_full_concentration_gives_n_loss(self):
        from ct87.engram import compute_qdiv_aux
        N, B, L, H, k = 100, 2, 8, 4, 4
        # All top-k selections land on row 0.
        idx = torch.zeros(B, L, k, dtype=torch.int64)
        # All attention mass on slot 0 (which maps to row 0 via idx).
        attn = torch.zeros(B, L, H, k)
        attn[..., 0] = 1.0
        loss = compute_qdiv_aux(idx, attn, N)
        # f[0]=1, P[0]=1, others 0 -> sum = 1, times N = N
        assert loss.item() == pytest.approx(float(N), abs=1e-4)

    def test_partial_concentration_intermediate(self):
        from ct87.engram import compute_qdiv_aux
        N, B, L, H, k = 100, 1, 100, 4, 4
        # Uniform over S=10 rows (rows 0..9).
        S = 10
        idx = (torch.arange(L * k) % S).reshape(B, L, k)
        attn = torch.full((B, L, H, k), 1.0 / k)
        loss = compute_qdiv_aux(idx, attn, N)
        # f[i] = 1/S for i<S, 0 else; P[i] = 1/S similarly
        # sum over i = S * (1/S) * (1/S) = 1/S; times N = N/S
        assert loss.item() == pytest.approx(float(N) / S, rel=0.01)

    def test_gradient_flows_to_attn_only(self):
        from ct87.engram import compute_qdiv_aux
        N, B, L, H, k = 50, 1, 10, 2, 3
        idx = torch.randint(0, N, (B, L, k), dtype=torch.int64)
        attn = torch.rand(B, L, H, k, requires_grad=True)
        attn_softmax = F.softmax(attn, dim=-1)
        loss = compute_qdiv_aux(idx, attn_softmax, N)
        loss.backward()
        assert attn.grad is not None
        assert attn.grad.abs().sum() > 0, "attn should receive gradient"
        # idx is int64 so autograd can't track it; just confirm helper didn't
        # convert f into something grad-requiring by accident.

    def test_deterministic_same_input_same_loss(self):
        from ct87.engram import compute_qdiv_aux
        N, B, L, H, k = 100, 2, 16, 4, 4
        torch.manual_seed(0)
        idx = torch.randint(0, N, (B, L, k), dtype=torch.int64)
        attn = F.softmax(torch.randn(B, L, H, k), dim=-1)
        a = compute_qdiv_aux(idx, attn, N)
        b = compute_qdiv_aux(idx, attn, N)
        assert torch.equal(a, b)


class TestAttentionBlockReturnAttn:
    """API extension for _attention_block: optional return of [B,L,H,k]
    softmax attention weights so Q-div can see them without recomputing.
    Symmetric to retrieve_topk(return_indices=True) from PR #250."""

    def _build_xattn(self, seed=0):
        from ct87.engram import EngramCrossAttention
        torch.manual_seed(seed)
        c = HarmonyModelConfig.tiny_engram_xattn_capgap()
        N, E, H, k = 100, c.engram_dim, 4, 4
        table = torch.randn(N, E)
        return EngramCrossAttention(
            c,
            table,
            num_heads=H,
            k_retrieved=k,
        )

    def test_return_attn_shape(self):
        xattn = self._build_xattn()
        B, L = 2, 7
        hidden = torch.randn(B, L, xattn.hidden_dim)
        retrieved, topk_sims = xattn.retrieve_topk(hidden)
        out, attn = xattn._attention_block(
            hidden, retrieved, topk_sims, return_attn=True,
        )
        assert out.shape == (B, L, xattn.hidden_dim)
        assert attn.shape == (B, L, xattn.num_heads, xattn.k_retrieved)
        assert torch.allclose(attn.sum(dim=-1), torch.ones_like(attn.sum(dim=-1)), atol=1e-5)

    def test_return_attn_false_matches_default(self):
        xattn = self._build_xattn(seed=1)
        B, L = 2, 5
        hidden = torch.randn(B, L, xattn.hidden_dim)
        retrieved, topk_sims = xattn.retrieve_topk(hidden)
        out_default = xattn._attention_block(hidden, retrieved, topk_sims)
        out_with_attn, _ = xattn._attention_block(
            hidden, retrieved, topk_sims, return_attn=True,
        )
        assert torch.equal(out_default, out_with_attn)


class TestGatedEngramInjectionSinkMatrix:
    """Four-cell behavior matrix for the unified GatedEngramInjection.

    No sinks   -> baseline η-B behavior (regression guard for PR #247).
    vcontrast  -> PR #250 V-contrast aux; no Q-div entries.
    qdiv       -> Q-div aux appended; no shuffled-value second forward.
    both       -> both aux losses fire independently.
    """

    def _build(self, seed=0):
        from ct87.engram import EngramCrossAttention
        torch.manual_seed(seed)
        c = HarmonyModelConfig.tiny_engram_xattn_capgap()
        N, E, H, k = 100, c.engram_dim, 4, 4
        table = torch.randn(N, E)
        return EngramCrossAttention(c, table, num_heads=H, k_retrieved=k)

    def test_no_sinks_matches_baseline(self):
        from ct87.engram import GatedEngramInjection
        xattn = self._build()
        wrapper = GatedEngramInjection(xattn, alpha_init=0.1)
        wrapper.train()
        hidden = torch.randn(2, 5, xattn.hidden_dim)

        # Reference: the pre-refactor forward was `gate * xattn(hidden)` where
        # EngramCrossAttention.forward is just retrieve_topk + _attention_block.
        # With no sinks attached, the new forward must produce the same output.
        with torch.no_grad():
            xattn_out = xattn(hidden)
            gate = torch.tanh(wrapper.alpha).to(dtype=xattn_out.dtype)
            reference = gate * xattn_out
            out = wrapper(hidden)

        assert out.shape == hidden.shape
        assert torch.equal(out, reference), (
            "No-sink forward must be bit-identical to pre-refactor "
            "gate * xattn(hidden); got max diff "
            f"{(out - reference).abs().max().item()}"
        )
        # Sinks None confirms no aux compute attached.
        assert wrapper._vcontrast_sink is None
        assert wrapper._qdiv_sink is None

    def test_only_qdiv_sink_appends_one_per_forward(self):
        from ct87.engram import GatedEngramInjection
        xattn = self._build(seed=1)
        sink: list[torch.Tensor] = []
        wrapper = GatedEngramInjection(xattn, alpha_init=0.1, qdiv_sink=sink)
        wrapper.train()
        hidden = torch.randn(2, 5, xattn.hidden_dim)
        _ = wrapper(hidden)
        assert len(sink) == 1
        assert sink[0].ndim == 0
        assert sink[0].item() >= 1.0  # MoE loss floor

    def test_only_vcontrast_sink_appends_one_per_forward(self):
        from ct87.engram import GatedEngramInjection
        xattn = self._build(seed=2)
        sink: list[torch.Tensor] = []
        wrapper = GatedEngramInjection(xattn, alpha_init=0.1, vcontrast_sink=sink)
        wrapper.train()
        hidden = torch.randn(2, 5, xattn.hidden_dim)
        _ = wrapper(hidden)
        assert len(sink) == 1
        assert sink[0].ndim == 0
        assert 0.0 <= sink[0].item() <= 1.0  # cos^2.mean bounded [0, 1]

    def test_both_sinks_fire_independently(self):
        from ct87.engram import GatedEngramInjection
        xattn = self._build(seed=3)
        vsink: list[torch.Tensor] = []
        qsink: list[torch.Tensor] = []
        wrapper = GatedEngramInjection(
            xattn, alpha_init=0.1,
            vcontrast_sink=vsink, qdiv_sink=qsink,
        )
        wrapper.train()
        hidden = torch.randn(2, 5, xattn.hidden_dim)
        _ = wrapper(hidden)
        assert len(vsink) == 1
        assert len(qsink) == 1

    def test_eval_mode_skips_aux(self):
        from ct87.engram import GatedEngramInjection
        xattn = self._build(seed=4)
        vsink: list[torch.Tensor] = []
        qsink: list[torch.Tensor] = []
        wrapper = GatedEngramInjection(
            xattn, alpha_init=0.1,
            vcontrast_sink=vsink, qdiv_sink=qsink,
        )
        wrapper.eval()
        hidden = torch.randn(2, 5, xattn.hidden_dim)
        _ = wrapper(hidden)
        assert len(vsink) == 0
        assert len(qsink) == 0


class TestHarmonyModelQdivSink:
    def test_model_exposes_qdiv_aux_losses_list(self):
        from ct87.model import HarmonyModel, HarmonyModelConfig
        c = HarmonyModelConfig.tiny_engram_xattn_capgap()
        c.engram_qdiv_enabled = True
        c.__post_init__()
        model = HarmonyModel(c)
        assert hasattr(model, "_qdiv_aux_losses")
        assert isinstance(model._qdiv_aux_losses, list)
        assert len(model._qdiv_aux_losses) == 0

    def test_qdiv_sink_receives_one_per_layer_per_forward(self):
        from ct87.engram import EngramCrossAttention, GatedEngramInjection
        from ct87.model import HarmonyModel, HarmonyModelConfig
        c = HarmonyModelConfig.tiny_engram_xattn_capgap()
        c.engram_qdiv_enabled = True
        c.__post_init__()
        model = HarmonyModel(c)

        # Build + attach injections the same way train.py does.
        table = torch.randn(100, c.engram_dim)
        injections = {}
        for layer_idx in c.engram_inject_layers:
            xattn = EngramCrossAttention(
                c, table, num_heads=4, k_retrieved=4
            )
            injections[layer_idx] = GatedEngramInjection(
                xattn,
                alpha_init=c.engram_gate_init,
                qdiv_sink=model._qdiv_aux_losses,
            )
        model.attach_gated_engram_injections(injections)

        model.train()
        B, L = 1, 8
        input_ids = torch.randint(0, c.vocab_size, (B, L))
        model._qdiv_aux_losses.clear()
        _ = model(input_ids)
        assert len(model._qdiv_aux_losses) == len(c.engram_inject_layers)


class TestIotaPresets:
    def test_iota_1_preset_qdiv_on_vcontrast_off(self):
        c = HarmonyModelConfig.tiny_engram_xattn_capgap_qdiv()
        assert c.engram_qdiv_enabled is True
        assert c.engram_vcontrast_enabled is False
        assert len(c.engram_inject_layers) > 0

    def test_iota_2_preset_both_on(self):
        c = HarmonyModelConfig.tiny_engram_xattn_capgap_vcontrast_qdiv()
        assert c.engram_qdiv_enabled is True
        assert c.engram_vcontrast_enabled is True
        assert len(c.engram_inject_layers) > 0

    def test_iota_presets_use_default_lambdas(self):
        c1 = HarmonyModelConfig.tiny_engram_xattn_capgap_qdiv()
        assert c1.engram_qdiv_lambda == 0.01
        assert c1.engram_qdiv_warmup_steps == 200
        c2 = HarmonyModelConfig.tiny_engram_xattn_capgap_vcontrast_qdiv()
        assert c2.engram_qdiv_lambda == 0.01


class TestCliFlagPresetConsistency:
    def test_qdiv_flag_without_preset_rejected(self):
        """Non-qdiv preset + --engram-qdiv must exit non-zero with a clear error."""
        result = _run_train_py([
            "--config", "tiny_engram_xattn_capgap",
            "--engram-qdiv",
            "--steps", "0",
            "--val-data", "/tmp/does-not-exist",
            "--synthetic",
        ])
        assert result.returncode != 0
        combined = result.stderr + result.stdout
        # The error should mention both --engram-qdiv and preset in some form
        assert "engram-qdiv" in combined.lower() or "engram_qdiv" in combined.lower()

    def test_qdiv_lambda_without_enabled_rejected(self):
        """--engram-qdiv-lambda without --engram-qdiv + qdiv preset exits non-zero."""
        result = _run_train_py([
            "--config", "tiny_engram_xattn_capgap",
            "--engram-qdiv-lambda", "0.02",
            "--steps", "0",
            "--val-data", "/tmp/does-not-exist",
            "--synthetic",
        ])
        assert result.returncode != 0
        combined = (result.stderr + result.stdout).lower()
        assert "engram-qdiv" in combined or "engram_qdiv" in combined

    @pytest.mark.parametrize("bad_value", ["nan", "inf", "-inf", "-0.01", "-1.0"])
    def test_qdiv_lambda_invalid_rejected(self, bad_value):
        """NaN/Inf slip past __post_init__'s `< 0` check (NaN comparisons
        always return False). Negative finite values are caught by
        __post_init__ too, but the CLI-layer check gives a fail-fast error
        naming the actual flag (--engram-qdiv-lambda) instead of the
        generic field-name error. Both paths exit non-zero.
        """
        result = _run_train_py([
            "--config", "tiny_engram_xattn_capgap_qdiv",
            "--engram-qdiv",
            "--engram-qdiv-lambda", bad_value,
            "--steps", "0",
            "--synthetic",
        ])
        assert result.returncode != 0, (
            f"--engram-qdiv-lambda={bad_value} should be rejected; "
            f"got returncode={result.returncode}"
        )
        combined = (result.stderr + result.stdout).lower()
        assert "engram" in combined and "lambda" in combined, (
            f"error message should name the flag; got: {combined!r}"
        )

    def test_qdiv_warmup_steps_negative_rejected(self):
        """Negative --engram-qdiv-warmup-steps must fail at the CLI layer."""
        result = _run_train_py([
            "--config", "tiny_engram_xattn_capgap_qdiv",
            "--engram-qdiv",
            "--engram-qdiv-warmup-steps", "-5",
            "--steps", "0",
            "--synthetic",
        ])
        assert result.returncode != 0
        combined = (result.stderr + result.stdout).lower()
        assert "warmup" in combined


class TestQdivTrainingStep:
    """Mid-training behavior: sinks drain per step, lambda warmup applies,
    total_loss includes scaled qdiv term."""

    def test_qdiv_sink_drained_per_step(self):
        """After a forward, sink has len == num_layers; after training-loop
        'drain' (list + clear), sink is empty and forward re-fills it."""
        from ct87.model import HarmonyModel, HarmonyModelConfig

        c = HarmonyModelConfig.tiny_engram_xattn_capgap_qdiv()
        # Build + attach engram injections the same way train.py does.
        from ct87.engram import EngramCrossAttention, GatedEngramInjection

        model = HarmonyModel(c)
        table = torch.randn(100, c.engram_dim)
        injections = {}
        for layer_idx in c.engram_inject_layers:
            xattn = EngramCrossAttention(c, table, num_heads=4, k_retrieved=4)
            injections[layer_idx] = GatedEngramInjection(
                xattn,
                alpha_init=c.engram_gate_init,
                qdiv_sink=model._qdiv_aux_losses,
            )
        model.attach_gated_engram_injections(injections)
        model.train()

        input_ids = torch.randint(0, c.vocab_size, (1, 8))
        # Step 1
        _ = model(input_ids)
        assert len(model._qdiv_aux_losses) == len(c.engram_inject_layers)
        per_layer_qd = list(model._qdiv_aux_losses)
        model._qdiv_aux_losses.clear()
        assert len(model._qdiv_aux_losses) == 0
        total = torch.stack(per_layer_qd).sum()
        # MoE load-balancing loss has a floor of 1.0 per layer.
        assert total.item() >= 1.0 * len(c.engram_inject_layers) - 1e-3

        # Step 2 should also fill the sink after clearing.
        _ = model(input_ids)
        assert len(model._qdiv_aux_losses) == len(c.engram_inject_layers)

    def test_lambda_schedule_warmup_linear(self):
        """lambda_schedule helper (from PR #250) linearly ramps 0 -> target
        over warmup_steps, then stays constant at target."""
        from ct87.train import lambda_schedule
        assert lambda_schedule(step=0, target=0.01, warmup_steps=200) == 0.0
        assert lambda_schedule(step=100, target=0.01, warmup_steps=200) == pytest.approx(0.005)
        assert lambda_schedule(step=200, target=0.01, warmup_steps=200) == pytest.approx(0.01)
        assert lambda_schedule(step=500, target=0.01, warmup_steps=200) == pytest.approx(0.01)


class TestCsvQdivColumns:
    """CSV column presence is gated on engram_qdiv_enabled (matches PR #250's
    pattern for V-contrast columns)."""

    def test_qdiv_cols_present_when_enabled(self, tmp_path):
        # Use iota-1 preset; it has engram_qdiv_enabled=True.
        log_file = tmp_path / "train.csv"
        output_dir = tmp_path / "output"
        result = subprocess.run(
            [
                sys.executable, "-m", "ct87.train",
                "--config", "tiny_engram_xattn_capgap_qdiv",
                "--engram-qdiv",
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
        # Project requires Python 3.10+ (pyproject.toml). Only on that path do
        # we assert end-to-end success; on 3.9 we accept the known
        # zip(strict=True) mid-run crash and verify only header schema.
        if sys.version_info >= (3, 10):
            assert result.returncode == 0, (
                f"train.py exited {result.returncode}\n"
                f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
            )
        assert log_file.exists(), (
            f"No CSV produced; stderr={result.stderr!r}, stdout={result.stdout[-2000:]!r}"
        )
        with open(log_file) as fh:
            header = next(csv.reader(fh))
        assert "qdiv_aux_loss" in header, f"header={header}"
        assert "qdiv_lambda" in header, f"header={header}"
        assert any(h.startswith("qdiv_aux_L") for h in header), f"header={header}"

    def test_qdiv_cols_absent_when_disabled(self, tmp_path):
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
        assert log_file.exists(), (
            f"No CSV produced; stderr={result.stderr!r}, stdout={result.stdout[-2000:]!r}"
        )
        with open(log_file) as fh:
            header = next(csv.reader(fh))
        assert not any(h.startswith("qdiv_") for h in header), (
            f"qdiv columns leaked into non-qdiv preset header: {header}"
        )


class TestSinkLayerOrder:
    """Regression test: sinks receive per-layer aux losses in ascending
    layer order regardless of engram_inject_layers declaration order.
    HarmonyModel.forward iterates i in range(num_layers), so the sink
    append order is forward-order, not declaration-order. train.py
    relies on this to zip(sorted_layers, sink, strict=True) for per-
    layer CSV attribution — if append order ever regresses (e.g. to
    declaration order), that zip mis-attributes."""

    def test_sinks_receive_ascending_layer_order_qdiv(self):
        from ct87.engram import EngramCrossAttention, GatedEngramInjection
        from ct87.model import HarmonyModel, HarmonyModelConfig

        def build_and_run(inject_layers):
            # Per-layer seed: layer 1's module weights are identical across
            # both (1,2) and (2,1) declaration orders, ditto for layer 2.
            # That makes the per-layer loss VALUES directly comparable, so
            # the final assertion can catch append-order regressions (which
            # would shuffle values) rather than structural breakage (which
            # the old enumerate-based assertion would catch but cared less
            # about — the failure mode the zip(strict=True) in train.py
            # actually cares about is value mis-ordering).
            torch.manual_seed(0)
            c = HarmonyModelConfig.tiny_engram_xattn_capgap()
            c.engram_inject_layers = inject_layers
            c.engram_qdiv_enabled = True
            c.__post_init__()
            model = HarmonyModel(c)
            torch.manual_seed(12345)
            table = torch.randn(100, c.engram_dim)
            injections = {}
            for layer_idx in inject_layers:
                torch.manual_seed(1000 + layer_idx)
                xattn = EngramCrossAttention(
                    c, table, num_heads=4, k_retrieved=4,
                )
                injections[layer_idx] = GatedEngramInjection(
                    xattn,
                    alpha_init=c.engram_gate_init,
                    qdiv_sink=model._qdiv_aux_losses,
                )
            model.attach_gated_engram_injections(injections)
            model.train()
            torch.manual_seed(99)
            input_ids = torch.randint(0, c.vocab_size, (1, 8))
            model._qdiv_aux_losses.clear()
            _ = model(input_ids)
            return [t.item() for t in model._qdiv_aux_losses]

        ascending = build_and_run((1, 2))
        out_of_order = build_and_run((2, 1))

        assert len(ascending) == 2
        assert len(out_of_order) == 2
        # Exact equality: per-layer seeding gives identical xattn weights for
        # each layer regardless of declaration order, so if the sink receives
        # losses in consistent ascending-layer-index order, the two lists are
        # bit-identical. If append order ever regressed to declaration order,
        # out_of_order would become [layer-2-loss, layer-1-loss] — detectable.
        assert ascending == out_of_order, (
            f"Sink append order regressed: declaration (1,2) yielded "
            f"{ascending}, (2,1) yielded {out_of_order}. They must match "
            f"because HarmonyModel.forward iterates layers in ascending "
            f"index order. train.py's zip(sorted_layers, sink, strict=True) "
            f"assumes this invariant."
        )


class TestOneStepSmokes:
    """End-to-end: train.py runs one step under each iota preset, writes
    CSV, sinks populate correctly. Subprocess form — isolates train.py
    process state from pytest's process."""

    def test_iota_1_one_step_completes(self, tmp_path):
        log_file = tmp_path / "train.csv"
        output_dir = tmp_path / "output"
        result = _run_train_py([
            "--config", "tiny_engram_xattn_capgap_qdiv",
            "--engram-qdiv",
            "--synthetic",
            "--steps", "1",
            "--batch-size", "1", "--seq-len", "8",
            "--save-every", "0",
            "--log-file", str(log_file),
            "--output-dir", str(output_dir),
        ])

        # Project requires Python 3.10+ (see pyproject.toml). On 3.10+, the
        # one-step run MUST succeed end-to-end; no tolerance for silent
        # crashes that leave an empty CSV. Only Python 3.9 local dev gets
        # the relaxed path because train.py uses zip(strict=True).
        py_is_310_plus = sys.version_info >= (3, 10)
        if py_is_310_plus:
            assert result.returncode == 0, (
                f"train.py exited {result.returncode} on a supported Python "
                f"version.\nstdout: {result.stdout}\nstderr: {result.stderr}"
            )

        assert log_file.exists(), (
            f"Expected a CSV file at {log_file}; stderr={result.stderr!r}"
        )
        with open(log_file) as fh:
            fieldnames = csv.DictReader(fh).fieldnames
        assert fieldnames is not None
        assert "qdiv_aux_loss" in fieldnames
        assert "qdiv_lambda" in fieldnames

        with open(log_file) as fh:
            rows = list(csv.DictReader(fh))
        if py_is_310_plus:
            assert len(rows) >= 1, (
                "Expected at least one data row; train.py ran but wrote no "
                "data."
            )
            qdiv_val = float(rows[0]["qdiv_aux_loss"])
            assert qdiv_val >= 0.0
        elif rows:
            qdiv_val = float(rows[0]["qdiv_aux_loss"])
            assert qdiv_val >= 0.0

    def test_iota_2_one_step_completes(self, tmp_path):
        log_file = tmp_path / "train.csv"
        output_dir = tmp_path / "output"
        result = _run_train_py([
            "--config", "tiny_engram_xattn_capgap_vcontrast_qdiv",
            "--engram-vcontrast",
            "--engram-qdiv",
            "--synthetic",
            "--steps", "1",
            "--batch-size", "1", "--seq-len", "8",
            "--save-every", "0",
            "--log-file", str(log_file),
            "--output-dir", str(output_dir),
        ])

        py_is_310_plus = sys.version_info >= (3, 10)
        if py_is_310_plus:
            assert result.returncode == 0, (
                f"train.py exited {result.returncode} on a supported Python "
                f"version.\nstdout: {result.stdout}\nstderr: {result.stderr}"
            )

        assert log_file.exists(), (
            f"Expected a CSV file at {log_file}; stderr={result.stderr!r}"
        )
        with open(log_file) as fh:
            fieldnames = csv.DictReader(fh).fieldnames
        assert fieldnames is not None
        assert "vcontrast_aux_loss" in fieldnames
        assert "qdiv_aux_loss" in fieldnames

        with open(log_file) as fh:
            rows = list(csv.DictReader(fh))
        if py_is_310_plus:
            assert len(rows) >= 1, (
                "Expected at least one data row; train.py ran but wrote no "
                "data."
            )
            vcontrast_val = float(rows[0]["vcontrast_aux_loss"])
            qdiv_val = float(rows[0]["qdiv_aux_loss"])
            assert 0.0 <= vcontrast_val <= 10.0
            assert qdiv_val >= 0.0
        elif rows:
            vcontrast_val = float(rows[0]["vcontrast_aux_loss"])
            qdiv_val = float(rows[0]["qdiv_aux_loss"])
            assert 0.0 <= vcontrast_val <= 10.0
            assert qdiv_val >= 0.0
