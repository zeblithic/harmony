"""Tests for Muon optimizer and WSD schedule."""

import torch
import torch.nn as nn
import pytest
from ct87.optim import Muon, WSDSchedule, partition_params, newton_schulz_orthogonalize
from ct87.model import HarmonyModel, HarmonyModelConfig


def _tiny_config() -> HarmonyModelConfig:
    return HarmonyModelConfig(
        num_layers=4, hidden_dim=32, num_query_heads=4, num_kv_heads=2,
        head_dim=8, ffn_dim=64, vocab_size=128, max_seq_len=64,
        rope_theta=10000.0, rms_norm_eps=1e-6, layers_per_block=2,
        engram_injection_layer=1, engram_dim=16, tie_embeddings=True,
    )


class TestNewtonSchulz:
    def test_near_orthogonal(self):
        """After Newton-Schulz, singular values of X should be close to 1.

        The polynomial approximation maps singular values toward 1 but converges
        to an approximate polar factor, not an exact orthogonal matrix.  Checking
        that all singular values land in [0.5, 1.5] verifies the orthogonalizing
        effect without demanding exact convergence in just 5 iterations.
        """
        X = torch.randn(32, 32)
        X_orth = newton_schulz_orthogonalize(X)
        svs = torch.linalg.svdvals(X_orth)
        assert svs.min().item() > 0.5, f"min singular value {svs.min().item():.3f} too small"
        assert svs.max().item() < 1.5, f"max singular value {svs.max().item():.3f} too large"

    def test_non_square_tall(self):
        """Newton-Schulz should work on tall matrices (more rows than cols).

        For tall X [m, n] with m > n, orthogonalization targets column
        orthonormality (X.T @ X ≈ I_n).  After 5 iterations the off-diagonal
        entries of X.T @ X should be small even if the diagonals are not yet 1.
        """
        X = torch.randn(64, 32)
        X_orth = newton_schulz_orthogonalize(X)
        result = X_orth.T @ X_orth
        # Off-diagonal entries should be small (columns becoming orthogonal)
        off_diag = result - torch.diag(result.diag())
        assert off_diag.abs().max().item() < 0.2, (
            f"off-diagonal max {off_diag.abs().max().item():.3f} too large"
        )


class TestMuon:
    def test_step_decreases_loss(self):
        """One Muon step should decrease loss on a simple problem."""
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        muon_params, adam_params = partition_params(model)
        optimizer = Muon(muon_params, adam_params, lr=1e-3, adam_lr=1e-3)

        input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
        targets = torch.randint(0, cfg.vocab_size, (2, 8))

        logits = model(input_ids)
        loss1 = torch.nn.functional.cross_entropy(
            logits.view(-1, cfg.vocab_size), targets.view(-1),
        )
        loss1.backward()
        optimizer.step()
        optimizer.zero_grad()

        logits = model(input_ids)
        loss2 = torch.nn.functional.cross_entropy(
            logits.view(-1, cfg.vocab_size), targets.view(-1),
        )
        assert loss2.item() < loss1.item()

    def test_adam_fallback_for_1d(self):
        """Non-matrix parameters (norms, embeddings, 3-D buffers) use Adam, not Muon.

        Muon only handles exactly 2-D parameters that are not embeddings or norms.
        Everything else — including 1-D norm weights, embedding tables, and 3-D
        tensors like the BlockAttnRes query buffer [1, 1, hidden_dim] — routes to
        the AdamW path.
        """
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        muon_params, adam_params = partition_params(model)

        # Every Muon parameter must be exactly 2-D
        for p in muon_params:
            assert p.dim() == 2, f"Muon param should be 2D, got {p.dim()}D shape {p.shape}"

        # Adam receives everything that is not a plain 2-D weight matrix,
        # including 1-D (norms), 2-D (embeddings), and higher-D (BlockAttnRes queries).
        muon_ids = {id(p) for p in muon_params}
        for p in adam_params:
            assert id(p) not in muon_ids, f"Param {p.shape} appears in both groups"


class TestPartitionParams:
    def test_all_params_accounted_for(self):
        """partition_params should cover all model parameters."""
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        muon_params, adam_params = partition_params(model)

        total_partitioned = sum(p.numel() for p in muon_params) + sum(p.numel() for p in adam_params)
        total_model = sum(p.numel() for p in model.parameters())
        assert total_partitioned == total_model

    def test_no_duplicates(self):
        """No parameter should appear in both groups."""
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        muon_params, adam_params = partition_params(model)

        muon_ids = {id(p) for p in muon_params}
        adam_ids = {id(p) for p in adam_params}
        assert len(muon_ids & adam_ids) == 0


class TestWSDSchedule:
    def test_warmup_starts_at_zero(self):
        sched = WSDSchedule(warmup_steps=100, total_steps=1000)
        assert sched.get_lr_multiplier(0) == pytest.approx(0.0)

    def test_warmup_linear(self):
        sched = WSDSchedule(warmup_steps=100, total_steps=1000)
        assert sched.get_lr_multiplier(50) == pytest.approx(0.5)

    def test_warmup_end(self):
        sched = WSDSchedule(warmup_steps=100, total_steps=1000)
        assert sched.get_lr_multiplier(100) == pytest.approx(1.0)

    def test_stable_phase(self):
        sched = WSDSchedule(warmup_steps=100, total_steps=1000, decay_fraction=0.1)
        assert sched.get_lr_multiplier(500) == pytest.approx(1.0)

    def test_decay_start(self):
        sched = WSDSchedule(warmup_steps=100, total_steps=1000, decay_fraction=0.1)
        assert sched.get_lr_multiplier(900) == pytest.approx(1.0)

    def test_decay_end(self):
        sched = WSDSchedule(warmup_steps=100, total_steps=1000, decay_fraction=0.1, min_lr_ratio=0.0)
        assert sched.get_lr_multiplier(999) == pytest.approx(0.0, abs=0.02)

    def test_decay_midpoint(self):
        sched = WSDSchedule(warmup_steps=100, total_steps=1000, decay_fraction=0.1, min_lr_ratio=0.0)
        assert sched.get_lr_multiplier(950) == pytest.approx(0.5, abs=0.02)

    def test_min_lr_ratio(self):
        # decay_fraction=0.1 of total_steps=100 → decay_steps=10, decay_start=90
        # At step=total_steps (100): progress=(100-90)/10=1.0 → min_lr_ratio exactly
        sched = WSDSchedule(warmup_steps=10, total_steps=100, decay_fraction=0.1, min_lr_ratio=0.1)
        assert sched.get_lr_multiplier(100) == pytest.approx(0.1, abs=0.02)
