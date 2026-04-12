"""Tests for latent projection module and projected Engram lookup."""

import math
import tempfile
from pathlib import Path

import torch
import pytest

from ct87.model import HarmonyModel, HarmonyModelConfig
from ct87.latent_projection import LatentProjection, compute_key_overlap
from ct87.engram import EngramTable


def _tiny_config() -> HarmonyModelConfig:
    return HarmonyModelConfig(
        num_layers=4, hidden_dim=32, num_query_heads=4, num_kv_heads=2,
        head_dim=8, ffn_dim=64, vocab_size=128, max_seq_len=64,
        rope_theta=10000.0, rms_norm_eps=1e-6, layers_per_block=2,
        engram_injection_layer=1, engram_dim=16, tie_embeddings=True,
    )


HIDDEN_DIM = 32
INTERMEDIATE_DIM = 16
LATENT_DIM = 8


def _make_projection() -> LatentProjection:
    torch.manual_seed(42)
    return LatentProjection(HIDDEN_DIM, INTERMEDIATE_DIM, LATENT_DIM)


def _make_table(entries: int = 100, dim: int = 16) -> EngramTable:
    table = torch.randn(entries, dim)
    return EngramTable(table, hash_seeds=[42, 99])


# ---------------------------------------------------------------------------
# LatentProjection module tests
# ---------------------------------------------------------------------------


class TestLatentProjection:
    def test_output_shape(self):
        proj = _make_projection()
        x = torch.randn(1, 4, HIDDEN_DIM)
        out = proj(x)
        assert out.shape == (1, 4, LATENT_DIM)

    def test_output_bounded_by_tanh(self):
        proj = _make_projection()
        x = torch.randn(1, 8, HIDDEN_DIM) * 10.0
        out = proj(x)
        assert out.abs().max().item() <= 1.0 + 1e-6

    def test_deterministic(self):
        proj = _make_projection()
        x = torch.randn(1, 4, HIDDEN_DIM)
        out1 = proj(x)
        out2 = proj(x)
        assert torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# Binary key tests
# ---------------------------------------------------------------------------


class TestBinaryKeys:
    def test_key_length(self):
        proj = _make_projection()
        x = torch.randn(1, 3, HIDDEN_DIM)
        latent = proj(x)
        keys = proj.to_binary_keys(latent.squeeze(0))
        assert len(keys) == 3
        expected_bytes = (LATENT_DIM + 7) // 8
        for key in keys:
            assert len(key) == expected_bytes

    def test_determinism(self):
        proj = _make_projection()
        x = torch.randn(1, 3, HIDDEN_DIM)
        latent = proj(x)
        keys1 = proj.to_binary_keys(latent.squeeze(0))
        keys2 = proj.to_binary_keys(latent.squeeze(0))
        assert keys1 == keys2

    def test_similar_embeddings_same_key(self):
        """Tiny perturbation should produce the same binary key."""
        proj = _make_projection()
        base = torch.randn(1, HIDDEN_DIM)
        noise = torch.randn(1, HIDDEN_DIM) * 1e-4
        similar = base + noise

        lat_a = proj(base.unsqueeze(0)).squeeze(0)
        lat_b = proj(similar.unsqueeze(0)).squeeze(0)
        keys_a = proj.to_binary_keys(lat_a)
        keys_b = proj.to_binary_keys(lat_b)
        assert keys_a == keys_b

    def test_different_embeddings_different_keys(self):
        """Very different inputs should produce different keys."""
        proj = _make_projection()
        a = torch.ones(1, HIDDEN_DIM)
        b = -torch.ones(1, HIDDEN_DIM)

        lat_a = proj(a.unsqueeze(0)).squeeze(0)
        lat_b = proj(b.unsqueeze(0)).squeeze(0)
        keys_a = proj.to_binary_keys(lat_a)
        keys_b = proj.to_binary_keys(lat_b)
        assert keys_a != keys_b

    def test_sign_bit_encoding(self):
        """Verify the sign-bit packing: bit i set if latent[i] >= 0."""
        proj = _make_projection()
        # Manually craft a latent vector
        latent = torch.tensor([[1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]])
        keys = proj.to_binary_keys(latent)
        # Bits 0,2,4,6 should be set → binary 01010101 = 0x55
        assert keys[0] == bytes([0x55])


# ---------------------------------------------------------------------------
# N-gram projection tests
# ---------------------------------------------------------------------------


class TestProjectNgrams:
    def test_bigrams_and_trigrams_count(self):
        """4 positions -> 3 bigrams + 2 trigrams = 5 keys."""
        proj = _make_projection()
        emb = torch.randn(1, 4, HIDDEN_DIM)
        keys, positions = proj.project_ngrams(emb, 4)
        assert len(keys) == 5
        assert len(positions) == 5

    def test_positions_match_rust(self):
        """Bigrams at 1,2,3 and trigrams at 2,3 for seq_len=4."""
        proj = _make_projection()
        emb = torch.randn(1, 4, HIDDEN_DIM)
        _keys, positions = proj.project_ngrams(emb, 4)
        assert positions[:3] == [1, 2, 3]  # bigrams
        assert positions[3:] == [2, 3]  # trigrams

    def test_single_token_returns_empty(self):
        proj = _make_projection()
        emb = torch.randn(1, 1, HIDDEN_DIM)
        keys, positions = proj.project_ngrams(emb, 1)
        assert len(keys) == 0
        assert len(positions) == 0

    def test_two_tokens_bigram_only(self):
        """2 tokens -> 1 bigram, 0 trigrams."""
        proj = _make_projection()
        emb = torch.randn(1, 2, HIDDEN_DIM)
        keys, positions = proj.project_ngrams(emb, 2)
        assert len(keys) == 1
        assert positions == [1]

    def test_key_byte_length(self):
        proj = _make_projection()
        emb = torch.randn(1, 5, HIDDEN_DIM)
        keys, _ = proj.project_ngrams(emb, 5)
        expected_bytes = (LATENT_DIM + 7) // 8
        for key in keys:
            assert len(key) == expected_bytes


# ---------------------------------------------------------------------------
# Checkpoint save/load round-trip
# ---------------------------------------------------------------------------


class TestCheckpointRoundTrip:
    def test_save_load_produces_same_output(self):
        torch.manual_seed(42)
        proj = LatentProjection(HIDDEN_DIM, INTERMEDIATE_DIM, LATENT_DIM)
        x = torch.randn(1, 4, HIDDEN_DIM)
        expected = proj(x)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "projection.pt"
            torch.save(proj.state_dict(), path)

            loaded = LatentProjection.from_checkpoint(
                path, HIDDEN_DIM, INTERMEDIATE_DIM, LATENT_DIM,
            )
            actual = loaded(x)

        assert torch.allclose(expected, actual, atol=1e-6)

    def test_load_with_prefix(self):
        """State dict keys may have a 'latent_projection.' prefix."""
        torch.manual_seed(42)
        proj = LatentProjection(HIDDEN_DIM, INTERMEDIATE_DIM, LATENT_DIM)
        x = torch.randn(1, 4, HIDDEN_DIM)
        expected = proj(x)

        prefixed = {f"latent_projection.{k}": v for k, v in proj.state_dict().items()}

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "projection.pt"
            torch.save(prefixed, path)

            loaded = LatentProjection.from_checkpoint(
                path, HIDDEN_DIM, INTERMEDIATE_DIM, LATENT_DIM,
            )
            actual = loaded(x)

        assert torch.allclose(expected, actual, atol=1e-6)

    def test_load_safetensors_format(self):
        """Round-trip through .safetensors format."""
        from safetensors.torch import save_file

        torch.manual_seed(42)
        proj = LatentProjection(HIDDEN_DIM, INTERMEDIATE_DIM, LATENT_DIM)
        x = torch.randn(1, 4, HIDDEN_DIM)
        expected = proj(x)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "projection.safetensors"
            save_file(proj.state_dict(), str(path))

            loaded = LatentProjection.from_checkpoint(
                path, HIDDEN_DIM, INTERMEDIATE_DIM, LATENT_DIM,
            )
            actual = loaded(x)

        assert torch.allclose(expected, actual, atol=1e-6)


# ---------------------------------------------------------------------------
# Projected lookup tests
# ---------------------------------------------------------------------------


class TestLookupBatchProjected:
    def test_output_shape(self):
        cfg = _tiny_config()
        torch.manual_seed(42)
        model = HarmonyModel(cfg)
        proj = LatentProjection(cfg.hidden_dim, INTERMEDIATE_DIM, LATENT_DIM)
        tbl = _make_table()

        input_ids = torch.randint(0, cfg.vocab_size, (2, 10))
        with torch.no_grad():
            emb = model.embed_tokens(input_ids)
        out = tbl.lookup_batch_projected(input_ids, emb, proj)
        assert out.shape == (2, 10, tbl.engram_dim)

    def test_position_zero_is_zero(self):
        """Position 0 has no N-gram coverage, same as xxhash lookup."""
        cfg = _tiny_config()
        torch.manual_seed(42)
        model = HarmonyModel(cfg)
        proj = LatentProjection(cfg.hidden_dim, INTERMEDIATE_DIM, LATENT_DIM)
        tbl = _make_table()

        input_ids = torch.randint(0, cfg.vocab_size, (1, 5))
        with torch.no_grad():
            emb = model.embed_tokens(input_ids)
        out = tbl.lookup_batch_projected(input_ids, emb, proj)
        assert out[0, 0].abs().max().item() < 1e-6

    def test_later_positions_nonzero(self):
        """Positions 1+ should have non-zero embeddings."""
        cfg = _tiny_config()
        torch.manual_seed(42)
        model = HarmonyModel(cfg)
        proj = LatentProjection(cfg.hidden_dim, INTERMEDIATE_DIM, LATENT_DIM)
        tbl = _make_table()

        input_ids = torch.randint(0, cfg.vocab_size, (1, 5))
        with torch.no_grad():
            emb = model.embed_tokens(input_ids)
        out = tbl.lookup_batch_projected(input_ids, emb, proj)
        for pos in range(1, 5):
            assert out[0, pos].abs().max().item() > 1e-6

    def test_deterministic(self):
        cfg = _tiny_config()
        torch.manual_seed(42)
        model = HarmonyModel(cfg)
        proj = LatentProjection(cfg.hidden_dim, INTERMEDIATE_DIM, LATENT_DIM)
        tbl = _make_table()

        input_ids = torch.randint(0, cfg.vocab_size, (1, 5))
        with torch.no_grad():
            emb = model.embed_tokens(input_ids)
        out1 = tbl.lookup_batch_projected(input_ids, emb, proj)
        out2 = tbl.lookup_batch_projected(input_ids, emb, proj)
        assert torch.allclose(out1, out2)

    def test_different_from_xxhash(self):
        """Projection keys should produce different embeddings than xxhash keys."""
        cfg = _tiny_config()
        torch.manual_seed(42)
        model = HarmonyModel(cfg)
        proj = LatentProjection(cfg.hidden_dim, INTERMEDIATE_DIM, LATENT_DIM)
        tbl = _make_table()

        input_ids = torch.randint(0, cfg.vocab_size, (1, 8))
        with torch.no_grad():
            emb = model.embed_tokens(input_ids)
        xxhash_out = tbl.lookup_batch(input_ids)
        proj_out = tbl.lookup_batch_projected(input_ids, emb, proj)
        # They should generally differ (random projection vs token hashing)
        assert not torch.allclose(xxhash_out, proj_out, atol=1e-4)


# ---------------------------------------------------------------------------
# Key overlap analysis tests
# ---------------------------------------------------------------------------


class TestKeyOverlap:
    def test_returns_expected_keys(self):
        cfg = _tiny_config()
        torch.manual_seed(42)
        model = HarmonyModel(cfg)
        proj = LatentProjection(cfg.hidden_dim, INTERMEDIATE_DIM, LATENT_DIM)
        tbl = _make_table()

        input_ids = torch.randint(0, cfg.vocab_size, (1, 8))
        stats = compute_key_overlap(model, proj, tbl, input_ids)

        expected_keys = {"total_lookups", "matching_indices", "overlap_pct", "unique_xxhash", "unique_projection"}
        assert set(stats.keys()) == expected_keys

    def test_total_lookups_correct(self):
        """For seq_len=8: 7 bigrams + 6 trigrams = 13 n-grams, 2 heads each = 26."""
        cfg = _tiny_config()
        torch.manual_seed(42)
        model = HarmonyModel(cfg)
        proj = LatentProjection(cfg.hidden_dim, INTERMEDIATE_DIM, LATENT_DIM)
        tbl = _make_table()

        input_ids = torch.randint(0, cfg.vocab_size, (1, 8))
        stats = compute_key_overlap(model, proj, tbl, input_ids)

        num_heads = len(tbl.hash_seeds)
        num_bigrams = 7
        num_trigrams = 6
        expected = (num_bigrams + num_trigrams) * num_heads
        assert stats["total_lookups"] == expected

    def test_overlap_percentage_bounded(self):
        cfg = _tiny_config()
        torch.manual_seed(42)
        model = HarmonyModel(cfg)
        proj = LatentProjection(cfg.hidden_dim, INTERMEDIATE_DIM, LATENT_DIM)
        tbl = _make_table()

        input_ids = torch.randint(0, cfg.vocab_size, (2, 10))
        stats = compute_key_overlap(model, proj, tbl, input_ids)

        assert 0.0 <= stats["overlap_pct"] <= 100.0

    def test_self_overlap_is_not_100(self):
        """Random projection should NOT produce the same indices as xxhash."""
        cfg = _tiny_config()
        torch.manual_seed(42)
        model = HarmonyModel(cfg)
        proj = LatentProjection(cfg.hidden_dim, INTERMEDIATE_DIM, LATENT_DIM)
        # Use a large table so collisions are unlikely
        tbl = _make_table(entries=100000, dim=16)

        input_ids = torch.randint(0, cfg.vocab_size, (2, 20))
        stats = compute_key_overlap(model, proj, tbl, input_ids)

        # With a 100K-entry table and random projection, overlap should be very low
        assert stats["overlap_pct"] < 50.0


# ---------------------------------------------------------------------------
# Perplexity measurement with projected lookup (integration)
# ---------------------------------------------------------------------------


class TestEvaluateProjected:
    def test_returns_expected_keys(self):
        from ct87.eval import evaluate_projected
        from ct87.train import make_synthetic_dataloader

        cfg = _tiny_config()
        torch.manual_seed(42)
        model = HarmonyModel(cfg)
        proj = LatentProjection(cfg.hidden_dim, INTERMEDIATE_DIM, LATENT_DIM)
        tbl = _make_table()
        dataloader = make_synthetic_dataloader(cfg.vocab_size, 16, batch_size=2, seed=99)

        metrics = evaluate_projected(
            model, dataloader, cfg.vocab_size, torch.device("cpu"),
            num_batches=3, engram_table=tbl, projection=proj,
        )

        assert set(metrics.keys()) == {"loss", "perplexity", "total_tokens", "tokens_per_sec", "elapsed_sec"}

    def test_loss_is_finite_positive(self):
        from ct87.eval import evaluate_projected
        from ct87.train import make_synthetic_dataloader

        cfg = _tiny_config()
        torch.manual_seed(42)
        model = HarmonyModel(cfg)
        proj = LatentProjection(cfg.hidden_dim, INTERMEDIATE_DIM, LATENT_DIM)
        tbl = _make_table()
        dataloader = make_synthetic_dataloader(cfg.vocab_size, 16, batch_size=2, seed=99)

        metrics = evaluate_projected(
            model, dataloader, cfg.vocab_size, torch.device("cpu"),
            num_batches=3, engram_table=tbl, projection=proj,
        )

        assert metrics["loss"] > 0.0
        assert math.isfinite(metrics["loss"])

    def test_perplexity_is_exp_loss(self):
        from ct87.eval import evaluate_projected
        from ct87.train import make_synthetic_dataloader

        cfg = _tiny_config()
        torch.manual_seed(42)
        model = HarmonyModel(cfg)
        proj = LatentProjection(cfg.hidden_dim, INTERMEDIATE_DIM, LATENT_DIM)
        tbl = _make_table()
        dataloader = make_synthetic_dataloader(cfg.vocab_size, 16, batch_size=2, seed=99)

        metrics = evaluate_projected(
            model, dataloader, cfg.vocab_size, torch.device("cpu"),
            num_batches=3, engram_table=tbl, projection=proj,
        )

        assert abs(metrics["perplexity"] - math.exp(metrics["loss"])) < 0.01

    def test_model_not_modified(self):
        """Projected lookup must not change model weights."""
        from ct87.eval import evaluate_projected
        from ct87.train import make_synthetic_dataloader

        cfg = _tiny_config()
        torch.manual_seed(42)
        model = HarmonyModel(cfg)
        proj = LatentProjection(cfg.hidden_dim, INTERMEDIATE_DIM, LATENT_DIM)
        tbl = _make_table()

        params_before = {k: v.clone() for k, v in model.named_parameters()}
        dataloader = make_synthetic_dataloader(cfg.vocab_size, 16, batch_size=2, seed=99)
        evaluate_projected(
            model, dataloader, cfg.vocab_size, torch.device("cpu"),
            num_batches=3, engram_table=tbl, projection=proj,
        )

        for k, v in model.named_parameters():
            assert torch.allclose(v, params_before[k]), f"Parameter {k} was modified"


# ---------------------------------------------------------------------------
# compute_ngram_averages (pre-projection n-gram extraction)
# ---------------------------------------------------------------------------


class TestComputeNgramAverages:
    def test_output_shapes(self):
        """4 tokens -> 3 bigrams + 2 trigrams = 5 n-gram averages."""
        from ct87.latent_projection import compute_ngram_averages

        emb = torch.randn(1, 4, HIDDEN_DIM)
        avgs, positions = compute_ngram_averages(emb, 4)
        assert avgs.shape == (5, HIDDEN_DIM)
        assert positions == [1, 2, 3, 2, 3]

    def test_single_token_returns_empty(self):
        from ct87.latent_projection import compute_ngram_averages

        avgs, positions = compute_ngram_averages(torch.randn(1, 1, HIDDEN_DIM), 1)
        assert avgs.shape[0] == 0
        assert positions == []

    def test_bigram_average_is_correct(self):
        """Verify the average of two adjacent embeddings."""
        from ct87.latent_projection import compute_ngram_averages

        emb = torch.zeros(1, 3, HIDDEN_DIM)
        emb[0, 0, :] = 2.0
        emb[0, 1, :] = 4.0
        avgs, _ = compute_ngram_averages(emb, 3)
        # First bigram: avg(emb[0], emb[1]) = avg(2, 4) = 3
        assert torch.allclose(avgs[0], torch.full((HIDDEN_DIM,), 3.0))

    def test_preserves_grad(self):
        """Averages should be differentiable w.r.t. input embeddings."""
        from ct87.latent_projection import compute_ngram_averages

        emb = torch.randn(1, 4, HIDDEN_DIM, requires_grad=True)
        avgs, _ = compute_ngram_averages(emb, 4)
        avgs.sum().backward()
        assert emb.grad is not None
        assert emb.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Contrastive loss tests
# ---------------------------------------------------------------------------


class TestContrastiveLoss:
    def test_scalar_output(self):
        from ct87.latent_projection import contrastive_loss

        original = torch.randn(8, HIDDEN_DIM)
        projected = torch.randn(8, LATENT_DIM)
        loss = contrastive_loss(original, projected, temperature=0.07, k=4)
        assert loss.dim() == 0  # scalar

    def test_finite_nonnegative(self):
        from ct87.latent_projection import contrastive_loss

        original = torch.randn(16, HIDDEN_DIM)
        projected = torch.randn(16, LATENT_DIM)
        loss = contrastive_loss(original, projected, temperature=0.07, k=4)
        assert loss.item() >= 0.0
        assert torch.isfinite(loss)

    def test_aligned_lower_than_random(self):
        """Projection that preserves structure should have lower loss."""
        from ct87.latent_projection import contrastive_loss

        torch.manual_seed(42)
        original = torch.randn(16, 128)
        random_proj = torch.randn(16, 32)
        aligned_proj = original[:, :32]

        loss_random = contrastive_loss(original, random_proj, temperature=0.07, k=4)
        loss_aligned = contrastive_loss(original, aligned_proj, temperature=0.07, k=4)
        assert loss_aligned.item() < loss_random.item()

    def test_single_vector_returns_zero(self):
        from ct87.latent_projection import contrastive_loss

        loss = contrastive_loss(torch.randn(1, HIDDEN_DIM), torch.randn(1, LATENT_DIM))
        assert loss.item() == 0.0

    def test_gradient_flows_to_projected(self):
        """Contrastive loss should produce gradients for projected, not original."""
        from ct87.latent_projection import contrastive_loss

        original = torch.randn(8, HIDDEN_DIM)
        projected = torch.randn(8, LATENT_DIM, requires_grad=True)
        loss = contrastive_loss(original, projected, temperature=0.07, k=4)
        loss.backward()
        assert projected.grad is not None
        assert projected.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# lookup_from_keys tests (pre-computed binary keys)
# ---------------------------------------------------------------------------


class TestLookupFromKeys:
    def test_output_shape(self):
        tbl = _make_table()
        proj = _make_projection()
        emb = torch.randn(1, 4, HIDDEN_DIM)
        keys, positions = proj.project_ngrams(emb, 4)
        out = tbl.lookup_from_keys(keys, positions, batch_size=1, seq_len=4)
        assert out.shape == (1, 4, tbl.engram_dim)

    def test_position_zero_is_zero(self):
        tbl = _make_table()
        proj = _make_projection()
        emb = torch.randn(1, 4, HIDDEN_DIM)
        keys, positions = proj.project_ngrams(emb, 4)
        out = tbl.lookup_from_keys(keys, positions, batch_size=1, seq_len=4)
        assert out[0, 0].abs().max().item() < 1e-6

    def test_matches_lookup_batch_projected(self):
        """lookup_from_keys with projection-generated keys should match
        lookup_batch_projected on the same input."""
        cfg = _tiny_config()
        torch.manual_seed(42)
        model = HarmonyModel(cfg)
        proj = LatentProjection(cfg.hidden_dim, INTERMEDIATE_DIM, LATENT_DIM)
        tbl = _make_table()

        input_ids = torch.randint(0, cfg.vocab_size, (1, 8))
        with torch.no_grad():
            emb = model.embed_tokens(input_ids)

        expected = tbl.lookup_batch_projected(input_ids, emb, proj)

        keys, positions = proj.project_ngrams(emb[0:1], 8)
        actual = tbl.lookup_from_keys(keys, positions, batch_size=1, seq_len=8)

        assert torch.allclose(expected, actual, atol=1e-6)
