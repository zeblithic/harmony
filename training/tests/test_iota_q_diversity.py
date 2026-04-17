"""Tests for ι-Q-diversity (ZEB-130): MoE load-balancing aux loss on
retrieval-row marginal distribution. Spec:
docs/superpowers/specs/2026-04-17-iota-q-diversity-design.md
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from ct87.model import HarmonyModelConfig


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
