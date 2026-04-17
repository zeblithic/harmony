"""Tests for ι-Q-diversity (ZEB-130): MoE load-balancing aux loss on
retrieval-row marginal distribution. Spec:
docs/superpowers/specs/2026-04-17-iota-q-diversity-design.md
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from ct87.model import HarmonyModelConfig


REPO_TRAINING_ROOT = Path(__file__).resolve().parent.parent


def _run_train_py(args: list[str]) -> subprocess.CompletedProcess:
    import os
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
