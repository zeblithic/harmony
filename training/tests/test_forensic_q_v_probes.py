"""Tests for the (Q-overlap) + (V-rank) forensic probes.

Documents why each probe answers a distinct question beyond (D) unique-rows,
and pins numeric fingerprints for the extreme cases (fixed per-token
retrieval, fully-dispersive per-token retrieval, V-output rank preservation
vs compression).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

_REPO_TRAINING = Path(__file__).resolve().parent.parent
if str(_REPO_TRAINING) not in sys.path:
    sys.path.insert(0, str(_REPO_TRAINING))

from scripts.forensic_eta_b_capgap import (  # noqa: E402
    _effective_rank,
    _q_overlap_stats,
    _v_rank_stats,
)


class TestQOverlapProbe:
    """(Q-overlap) random-pair Jaccard + occupancy CV."""

    def test_fixed_per_token_retrieval_gives_jaccard_one(self) -> None:
        """Every token retrieving the same k rows → Jaccard ≡ 1.

        The H3a fingerprint: marginal |S| = k but per-token retrieval is
        identical across all tokens. The random-pair Jaccard picks this up
        even though the (D) unique-rows probe would only report |S|=k and
        wouldn't distinguish it from "k rows, each token picks a different
        one".
        """
        B, L, k = 4, 8, 8
        fixed_rows = torch.arange(k)
        topk = fixed_rows.view(1, 1, k).expand(B, L, k).contiguous()

        stats = _q_overlap_stats(topk, num_pairs=200)

        assert stats["q_overlap_random_pair_jaccard_mean"] == pytest.approx(1.0, abs=1e-6)
        assert stats["q_overlap_random_pair_jaccard_p50"] == pytest.approx(1.0, abs=1e-6)

    def test_fully_dispersive_retrieval_gives_low_jaccard(self) -> None:
        """Uniformly random per-token top-k → mean Jaccard ≈ k / table_size.

        When top-k is drawn uniformly from a large table independently for
        each token, expected pairwise intersection is k·k/table_size by
        linearity of expectation, and Jaccard ≈ that / (2k - that) → low.
        Pin an upper bound loose enough to absorb sampling noise.
        """
        B, L, k, table_size = 8, 16, 8, 2000
        torch.manual_seed(0)
        topk = torch.stack([
            torch.randperm(table_size)[:k] for _ in range(B * L)
        ]).view(B, L, k)

        stats = _q_overlap_stats(topk, num_pairs=500, seed=0)

        expected_upper_bound = 0.05
        assert stats["q_overlap_random_pair_jaccard_mean"] < expected_upper_bound, (
            f"expected random-pair Jaccard ≪ 1 for dispersive retrieval, "
            f"got {stats['q_overlap_random_pair_jaccard_mean']:.4f}"
        )

    def test_moderate_overlap_gives_intermediate_jaccard(self) -> None:
        """Every token retrieves 4 shared rows + 4 unique rows → Jaccard ≈ 0.5.

        Intersection = 4 (shared), union = 4+4+4 = 12, Jaccard = 4/12 ≈ 0.33.
        The exact number doesn't matter — what matters is that the probe
        produces an intermediate value, not pinned at 0 or 1.
        """
        B, L, k = 2, 16, 8
        shared = torch.arange(4)
        N = B * L
        torch.manual_seed(0)
        per_token_unique = torch.stack([
            4 + torch.randperm(100)[:4] for _ in range(N)
        ])
        topk = torch.cat([
            shared.unsqueeze(0).expand(N, 4),
            per_token_unique,
        ], dim=-1).view(B, L, k)

        stats = _q_overlap_stats(topk, num_pairs=500, seed=0)

        jaccard = stats["q_overlap_random_pair_jaccard_mean"]
        assert 0.2 < jaccard < 0.6, (
            f"expected intermediate Jaccard ~0.33 for 4-shared 4-unique retrieval, "
            f"got {jaccard:.4f}"
        )

    def test_occupancy_cv_zero_on_uniform_usage(self) -> None:
        """Every row in S retrieved the same number of times → CV = 0.

        This is the MoE load-balancing-aux target: uniform marginal usage
        among whatever rows Q decides to use. iota-B's forensic should
        move toward this post-iota-1.

        Construction respects the "distinct per row" precondition of
        _q_overlap_stats by giving every token the same k distinct rows
        [0..k-1]. Each row is then retrieved B*L times → counts all
        equal → CV = 0.
        """
        B, L, k = 4, 4, 4
        shared_rows = torch.arange(k)
        topk = shared_rows.view(1, 1, k).expand(B, L, k).contiguous()

        stats = _q_overlap_stats(topk, num_pairs=100)

        assert stats["q_occupancy_cv_used"] == pytest.approx(0.0, abs=1e-6)

    def test_occupancy_cv_large_on_concentrated_usage(self) -> None:
        """One row dominates, others retrieved once → large CV.

        The "many rows used but traffic skewed" pattern: high |S|, high CV.
        Tests that CV picks up concentration orthogonal to |S|.
        """
        B, L, k = 2, 8, 4
        hot_row = torch.zeros(B * L, 1, dtype=torch.long)
        cold_rows = torch.arange(1, B * L * (k - 1) + 1).view(B * L, k - 1)
        flat_topk = torch.cat([hot_row, cold_rows], dim=-1)
        topk = flat_topk.view(B, L, k)

        stats = _q_overlap_stats(topk, num_pairs=100)

        assert stats["q_occupancy_cv_used"] > 1.0, (
            f"expected CV > 1.0 for concentrated usage, "
            f"got {stats['q_occupancy_cv_used']:.4f}"
        )

    def test_deterministic_given_seed(self) -> None:
        """Same seed + same topk → identical stats (Monte Carlo reproducibility)."""
        B, L, k = 4, 8, 8
        torch.manual_seed(0)
        topk = torch.stack([
            torch.randperm(500)[:k] for _ in range(B * L)
        ]).view(B, L, k)

        stats_a = _q_overlap_stats(topk, num_pairs=200, seed=42)
        stats_b = _q_overlap_stats(topk, num_pairs=200, seed=42)

        assert stats_a["q_overlap_random_pair_jaccard_mean"] == stats_b[
            "q_overlap_random_pair_jaccard_mean"
        ]
        assert stats_a["q_overlap_random_pair_jaccard_p50"] == stats_b[
            "q_overlap_random_pair_jaccard_p50"
        ]

    def test_self_pairs_excluded_from_jaccard(self) -> None:
        """Self-pairs (idx_a == idx_b) would force Jaccard=1 and bias the
        mean upward. With N*(N-1) off-diagonal pairs and a diversified
        retrieval, the mean Jaccard should stay well below the self-pair
        contribution baseline — this test would fail on small-N samples
        if self-pairs crept back in.

        Construction: N=B*L = 8 tokens (deliberately small to amplify
        self-pair bias), each with disjoint k=4 rows. True off-diagonal
        Jaccard is 0 (no shared rows); with a 1/8 self-pair rate the
        biased estimator would show ~0.125. Assert strictly below.
        """
        B, L, k = 2, 4, 4
        N = B * L
        # Disjoint per-token row sets: token i gets rows [i*k, i*k+1, ..., i*k+k-1].
        topk = torch.arange(N * k).view(B, L, k)

        stats = _q_overlap_stats(topk, num_pairs=10000, seed=0)

        assert stats["q_overlap_random_pair_jaccard_mean"] < 0.02, (
            f"disjoint per-token retrieval must yield Jaccard ≈ 0; self-"
            f"pair inclusion would inflate the mean toward 1/N = "
            f"{1.0/N:.3f}. Got "
            f"{stats['q_overlap_random_pair_jaccard_mean']:.4f}."
        )


class _MockXattn:
    """Minimal stand-in exposing the three attributes `_v_rank_stats` reads.

    Avoids constructing the full EngramCrossAttention in tests — just needs
    `.table`, `.retrieval_norm`, `.v_proj`.
    """

    def __init__(
        self,
        table: torch.Tensor,
        v_proj: nn.Module,
        retrieval_norm: nn.Module | None = None,
    ) -> None:
        self.table = table
        self.v_proj = v_proj
        self.retrieval_norm = (
            retrieval_norm if retrieval_norm is not None
            else nn.LayerNorm(table.shape[-1])
        )


class TestVRankProbe:
    """(V-rank) rank(V(E[S])) vs rank(E[S])."""

    def test_identity_v_preserves_rank(self) -> None:
        """V = identity → rank(V(E[S])) = rank(E[S]) = min(|S|, embed_dim)."""
        torch.manual_seed(0)
        n_rows, embed_dim = 100, 32
        table = torch.randn(n_rows, embed_dim)
        v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        with torch.no_grad():
            v_proj.weight.copy_(torch.eye(embed_dim))
        xattn = _MockXattn(table, v_proj, retrieval_norm=nn.Identity())

        topk = torch.arange(50).view(1, 10, 5)

        stats = _v_rank_stats(xattn, topk)

        assert stats["subset_size_S"] == 50
        assert stats["e_rank_hard"] == stats["v_rank_hard"]
        assert stats["v_rank_ratio"] == pytest.approx(1.0)

    def test_low_rank_v_compresses_output(self) -> None:
        """V with rank r → rank(V(E[S])) ≤ r regardless of |S|.

        Captures the θ L5 finding: |S|=198 on the input side, rank-74 on
        the V output. Here we rig V to have rank 4 and verify the output
        rank is capped at 4 even when |S|=50.
        """
        torch.manual_seed(0)
        n_rows, embed_dim = 200, 32
        table = torch.randn(n_rows, embed_dim)
        v_low_rank = 4
        v_weight = torch.randn(embed_dim, v_low_rank) @ torch.randn(v_low_rank, embed_dim)
        v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        with torch.no_grad():
            v_proj.weight.copy_(v_weight)
        xattn = _MockXattn(table, v_proj, retrieval_norm=nn.Identity())

        topk = torch.arange(50).view(1, 10, 5)

        stats = _v_rank_stats(xattn, topk)

        assert stats["v_rank_hard"] <= v_low_rank, (
            f"V has rank {v_low_rank}; V(E[S]) rank must be ≤ {v_low_rank}, "
            f"got {stats['v_rank_hard']}"
        )
        assert stats["v_rank_ratio"] < 0.5, (
            f"expected v/e rank ratio < 0.5 for low-rank V, "
            f"got {stats['v_rank_ratio']:.4f}"
        )

    def test_subset_size_reports_unique_rows(self) -> None:
        """|S| = unique rows across all (B, L, k) top-k slots."""
        torch.manual_seed(0)
        n_rows, embed_dim = 50, 16
        table = torch.randn(n_rows, embed_dim)
        v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        xattn = _MockXattn(table, v_proj, retrieval_norm=nn.Identity())

        topk = torch.tensor([0, 1, 2, 0, 1, 3]).view(1, 2, 3)
        stats = _v_rank_stats(xattn, topk)

        assert stats["subset_size_S"] == 4


class TestEffectiveRank:
    """Entropy-based effective rank sanity checks."""

    def test_uniform_spectrum_matches_matrix_rank(self) -> None:
        """r uniform singular values → effective rank = r.

        exp(H(uniform over r bins)) = r. Confirms `_effective_rank` produces
        the expected limit so readers can calibrate the number.
        """
        r = 8
        u, _ = torch.linalg.qr(torch.randn(32, r))
        v, _ = torch.linalg.qr(torch.randn(16, r))
        singular_values = torch.ones(r)
        matrix = u @ torch.diag(singular_values) @ v.T

        eff = _effective_rank(matrix)

        assert eff == pytest.approx(r, rel=1e-4)

    def test_concentrated_spectrum_gives_low_effective_rank(self) -> None:
        """σ = [1, ε, ε, ε, ...] → effective rank → 1 as ε → 0.

        Lets readers interpret low effective rank as "one direction dominates."
        """
        r = 8
        u, _ = torch.linalg.qr(torch.randn(32, r))
        v, _ = torch.linalg.qr(torch.randn(16, r))
        singular_values = torch.cat([torch.tensor([1.0]), torch.full((r - 1,), 1e-4)])
        matrix = u @ torch.diag(singular_values) @ v.T

        eff = _effective_rank(matrix)

        assert eff < 1.01, (
            f"expected effective rank → 1 for dominant-σ₁ spectrum, "
            f"got {eff:.4f}"
        )

    def test_effective_rank_smooth_where_hard_rank_is_discrete(self) -> None:
        """Effective rank decreases smoothly as σ spectrum concentrates;
        hard rank would cliff at whatever SVD threshold is chosen.

        Demonstrates the added value of reporting both.
        """
        r = 10
        u, _ = torch.linalg.qr(torch.randn(40, r))
        v, _ = torch.linalg.qr(torch.randn(20, r))

        flat = torch.ones(r)
        effective_flat = _effective_rank(u @ torch.diag(flat) @ v.T)

        decaying = torch.tensor([math.exp(-i / 3.0) for i in range(r)])
        effective_decay = _effective_rank(u @ torch.diag(decaying) @ v.T)

        assert effective_flat > effective_decay, (
            f"flat spectrum should have higher effective rank; "
            f"flat={effective_flat:.2f} decay={effective_decay:.2f}"
        )
        assert effective_decay < r
