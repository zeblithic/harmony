"""Tests for ZEB-119 oracle corpus table generator.

These exercise the CPU-only, deterministic parts of the pipeline:
Welford online averaging, n-gram index extraction + student hash
parity, PCA shape/variance, safetensors round-trip, and end-to-end
composition with a mocked teacher. The expensive teacher forward pass
is NOT exercised here; that's validated manually on a mini smoke run
before the full 22-hour KRILE job.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import tempfile
from pathlib import Path

from ct87.engram import (
    _hash_ngram,
    EngramTable,
    EngramANNInjection,
)
try:
    from ct87.engram import EngramCrossAttention
    _HAS_XATTN = True
except ImportError:
    # Model delta (EngramCrossAttention) lives on PR #238; this branch
    # may be based on a main that predates the merge. Xattn-specific
    # tests skip when the class isn't available.
    EngramCrossAttention = None  # type: ignore[assignment]
    _HAS_XATTN = False
from ct87.model import HarmonyModelConfig
from ct87.generate_oracle_table import (
    WelfordTable,
    apply_pca_projection,
    compute_ngram_indices_for_sequence,
    fit_pca_projection,
    save_oracle_table,
)

try:
    import sklearn  # noqa: F401
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

try:
    import transformers  # noqa: F401
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

try:
    import datasets  # noqa: F401
    _HAS_DATASETS = True
except ImportError:
    _HAS_DATASETS = False

try:
    import sentence_transformers  # noqa: F401
    _HAS_ST = True
except ImportError:
    _HAS_ST = False

# fit_pca_projection imports sklearn lazily, so unit tests touching PCA
# paths need this marker to skip cleanly in `.[dev]`-only environments
# that haven't installed scikit-learn.
skip_if_no_sklearn = pytest.mark.skipif(
    not _HAS_SKLEARN, reason="scikit-learn not installed (install .[teacher])"
)
skip_if_no_transformers = pytest.mark.skipif(
    not (_HAS_TRANSFORMERS and _HAS_DATASETS),
    reason="transformers/datasets not installed (install .[teacher])",
)
skip_if_no_sentence_transformers = pytest.mark.skipif(
    not (_HAS_ST and _HAS_SKLEARN),
    reason="sentence-transformers/sklearn not installed (install .[teacher])",
)


# ---------------------------------------------------------------------------
# Welford online mean
# ---------------------------------------------------------------------------


class TestWelfordTable:
    """Welford is the numerical core; a bug here poisons the entire oracle."""

    def test_zeros_shape(self):
        t = WelfordTable.zeros(total_entries=10, teacher_dim=4)
        assert t.means.shape == (10, 4)
        assert t.counts.shape == (10,)
        assert t.means.dtype == np.float32
        assert t.counts.dtype == np.int64
        assert (t.means == 0).all()
        assert (t.counts == 0).all()

    def test_single_update_matches_input(self):
        t = WelfordTable.zeros(5, 3)
        vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        t.update_batch(
            np.array([2], dtype=np.int64),
            vec.reshape(1, 3),
        )
        # With count=1, mean == vec exactly
        np.testing.assert_allclose(t.means[2], vec)
        assert t.counts[2] == 1
        # Other rows untouched
        assert (t.means[[0, 1, 3, 4]] == 0).all()

    def test_multiple_updates_compute_correct_mean(self):
        """The running mean must equal the true arithmetic mean."""
        rng = np.random.default_rng(0)
        t = WelfordTable.zeros(3, 8)
        vectors = rng.standard_normal((50, 8)).astype(np.float32)
        # All 50 updates go to row 1
        indices = np.full(50, 1, dtype=np.int64)
        t.update_batch(indices, vectors)
        expected = vectors.mean(axis=0)
        np.testing.assert_allclose(t.means[1], expected, rtol=1e-5, atol=1e-5)
        assert t.counts[1] == 50
        assert t.counts[0] == 0
        assert t.counts[2] == 0

    def test_updates_across_multiple_rows(self):
        """Updates to different rows must not interact."""
        rng = np.random.default_rng(1)
        t = WelfordTable.zeros(4, 5)
        # Row 0: 10 obs of N(1, 0.1)
        row0 = rng.normal(loc=1.0, scale=0.1, size=(10, 5)).astype(np.float32)
        # Row 2: 20 obs of N(-5, 0.5)
        row2 = rng.normal(loc=-5.0, scale=0.5, size=(20, 5)).astype(np.float32)
        indices = np.concatenate([
            np.zeros(10, dtype=np.int64),
            np.full(20, 2, dtype=np.int64),
        ])
        vectors = np.concatenate([row0, row2], axis=0)
        t.update_batch(indices, vectors)
        np.testing.assert_allclose(t.means[0], row0.mean(axis=0), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(t.means[2], row2.mean(axis=0), rtol=1e-5, atol=1e-5)
        assert (t.means[[1, 3]] == 0).all()
        assert t.counts[0] == 10
        assert t.counts[2] == 20

    def test_update_batch_order_independence(self):
        """Two different orderings of the same observations produce equal means.

        This is the algebraic guarantee Welford is supposed to provide; if a
        future refactor breaks it, retrieval quality becomes seed-dependent.
        """
        rng = np.random.default_rng(2)
        vectors = rng.standard_normal((30, 6)).astype(np.float32)
        indices = rng.integers(0, 4, size=30).astype(np.int64)

        t1 = WelfordTable.zeros(4, 6)
        t1.update_batch(indices, vectors)

        perm = rng.permutation(30)
        t2 = WelfordTable.zeros(4, 6)
        t2.update_batch(indices[perm], vectors[perm])

        np.testing.assert_allclose(t1.means, t2.means, rtol=1e-4, atol=1e-4)
        np.testing.assert_array_equal(t1.counts, t2.counts)

    def test_update_rejects_mismatched_shapes(self):
        t = WelfordTable.zeros(5, 4)
        with pytest.raises(ValueError, match="align"):
            t.update_batch(
                np.array([0, 1], dtype=np.int64),
                np.zeros((3, 4), dtype=np.float32),
            )

    def test_update_rejects_wrong_teacher_dim(self):
        t = WelfordTable.zeros(5, 4)
        with pytest.raises(ValueError, match="teacher_dim"):
            t.update_batch(
                np.array([0], dtype=np.int64),
                np.zeros((1, 7), dtype=np.float32),
            )

    def test_populated_mask_tracks_writes(self):
        t = WelfordTable.zeros(5, 3)
        t.update_batch(
            np.array([1, 3], dtype=np.int64),
            np.ones((2, 3), dtype=np.float32),
        )
        mask = t.populated_mask
        assert mask.tolist() == [False, True, False, True, False]


# ---------------------------------------------------------------------------
# N-gram extraction + student hash parity
# ---------------------------------------------------------------------------


class TestNgramIndices:
    """The oracle MUST produce the exact same row indices the student uses.

    If these diverge, the oracle's "row i" content no longer corresponds to
    the student's "row i" retrieval, and the diagnostic signal is meaningless.
    These tests compare the generator's indices against the student's
    `EngramTable._collect_indices` behavior.
    """

    def test_bigrams_attribute_to_second_token_position(self):
        tokens = [10, 20, 30, 40]
        seeds = (42,)
        N = 100
        rows, positions = compute_ngram_indices_for_sequence(tokens, seeds, N)
        # Three bigrams: (10,20)->pos 1, (20,30)->pos 2, (30,40)->pos 3
        # Three trigrams: (10,20,30)->pos 2, (20,30,40)->pos 3
        # Total = 3 + 2 = 5 entries for 1 seed
        assert len(rows) == 5
        assert len(positions) == 5
        # First three are bigrams; positions 1, 2, 3
        assert positions[:3] == [1, 2, 3]
        # Last two are trigrams; positions 2, 3
        assert positions[3:] == [2, 3]

    def test_multi_seed_multiplies_entries(self):
        tokens = [1, 2, 3]
        # 1 bigram + 1 bigram + 1 trigram = 2 bigrams (pos 1, 2) + 1 trigram (pos 2)
        # = 3 ngrams * 2 seeds = 6 entries
        rows, positions = compute_ngram_indices_for_sequence(tokens, (42, 99), 100)
        assert len(rows) == 6
        assert len(positions) == 6

    def test_parity_with_student_engram_table(self):
        """Generator rows must match student EngramTable._collect_indices exactly."""
        tokens = [7, 13, 29, 41, 53, 67]
        seeds = (42, 99, 137, 251)
        N = 256

        # Generator path
        gen_rows, gen_positions = compute_ngram_indices_for_sequence(
            tokens, seeds, N,
        )

        # Student path: replicate _collect_indices inline, since the method
        # is internal and doesn't expose the flat indices directly.
        student_rows = []
        student_positions = []
        for i in range(len(tokens) - 1):
            bigram = [tokens[i], tokens[i + 1]]
            for seed in seeds:
                student_rows.append(_hash_ngram(bigram, seed) % N)
                student_positions.append(i + 1)
        for i in range(len(tokens) - 2):
            trigram = [tokens[i], tokens[i + 1], tokens[i + 2]]
            for seed in seeds:
                student_rows.append(_hash_ngram(trigram, seed) % N)
                student_positions.append(i + 2)

        assert gen_rows == student_rows, (
            "Generator row indices diverge from student EngramTable — "
            "oracle row i will not correspond to student retrieval row i."
        )
        assert gen_positions == student_positions

    def test_empty_sequence_returns_empty(self):
        rows, positions = compute_ngram_indices_for_sequence([5], (42,), 100)
        assert rows == []
        assert positions == []

    def test_len_2_produces_only_bigrams(self):
        rows, positions = compute_ngram_indices_for_sequence([1, 2], (42,), 100)
        assert len(rows) == 1
        assert positions == [1]


# ---------------------------------------------------------------------------
# PCA projection
# ---------------------------------------------------------------------------


@skip_if_no_sklearn
class TestPCAProjection:
    """PCA fit + apply must preserve shape, respect populated rows, and
    return variance-ratios that are reasonable given the input rank."""

    @staticmethod
    def _make_anisotropic(n_rows: int, dim: int, seed: int) -> np.ndarray:
        """Synthetic high-dim data with a narrow principal subspace."""
        rng = np.random.default_rng(seed)
        # True signal lives in the first 8 dimensions
        signal = rng.standard_normal((n_rows, 8)).astype(np.float32) * 5.0
        noise = rng.standard_normal((n_rows, dim - 8)).astype(np.float32) * 0.01
        return np.concatenate([signal, noise], axis=1)

    def test_fit_and_project_shape(self):
        data = self._make_anisotropic(1000, 64, seed=0)
        mask = np.ones(1000, dtype=bool)
        rng = np.random.default_rng(0)
        components, mean_vec, variance = fit_pca_projection(
            means=data,
            populated_mask=mask,
            target_dim=16,
            subsample_fraction=0.5,
            pca_batch_size=128,
            rng=rng,
        )
        assert components.shape == (16, 64)
        assert mean_vec.shape == (64,)
        assert 0.0 <= variance <= 1.0
        projected = apply_pca_projection(data, components, mean_vec)
        assert projected.shape == (1000, 16)
        assert projected.dtype == np.float32

    def test_high_variance_on_anisotropic_data(self):
        """Top 16 components of 8-dim-signal data must capture >90% variance."""
        data = self._make_anisotropic(2000, 128, seed=1)
        mask = np.ones(2000, dtype=bool)
        rng = np.random.default_rng(1)
        _, _, variance = fit_pca_projection(
            means=data,
            populated_mask=mask,
            target_dim=16,
            subsample_fraction=0.5,
            pca_batch_size=256,
            rng=rng,
        )
        # 8-dim signal + 120-dim tiny noise: top 16 should capture ~all
        # of the signal variance.
        assert variance > 0.9, f"expected >0.9 variance, got {variance}"

    def test_respects_populated_mask(self):
        """Unpopulated rows (all zeros) must not be sampled for the fit."""
        data = np.zeros((500, 32), dtype=np.float32)
        # Only rows 0-99 have real data; 100-499 are zero
        rng_data = np.random.default_rng(3)
        data[:100] = rng_data.standard_normal((100, 32)).astype(np.float32) * 3.0
        mask = np.zeros(500, dtype=bool)
        mask[:100] = True
        rng = np.random.default_rng(4)
        _, mean_vec, _ = fit_pca_projection(
            means=data,
            populated_mask=mask,
            target_dim=8,
            subsample_fraction=1.0,
            pca_batch_size=64,
            rng=rng,
        )
        # Centering should reflect the populated rows' mean, NOT the
        # global mean (which is diluted by 400 zero rows).
        expected_mean = data[:100].mean(axis=0)
        np.testing.assert_allclose(mean_vec, expected_mean, rtol=1e-4, atol=1e-4)

    def test_raises_on_no_populated_rows(self):
        data = np.zeros((10, 4), dtype=np.float32)
        mask = np.zeros(10, dtype=bool)
        rng = np.random.default_rng(5)
        with pytest.raises(ValueError, match="populated"):
            fit_pca_projection(
                means=data, populated_mask=mask, target_dim=2,
                subsample_fraction=0.5, pca_batch_size=4, rng=rng,
            )

    def test_raises_if_target_dim_exceeds_teacher_dim(self):
        data = np.zeros((20, 8), dtype=np.float32)
        mask = np.ones(20, dtype=bool)
        rng = np.random.default_rng(6)
        with pytest.raises(ValueError, match="target_dim"):
            fit_pca_projection(
                means=data, populated_mask=mask, target_dim=16,
                subsample_fraction=0.5, pca_batch_size=4, rng=rng,
            )


# ---------------------------------------------------------------------------
# Safetensors round-trip: generator output must load into student modules
# ---------------------------------------------------------------------------


class TestSafetensorsCompat:
    """The generated table must load verbatim via the student's
    EngramANNInjection / EngramCrossAttention load_corpus_table APIs,
    with no schema mismatch or shape surprises."""

    @staticmethod
    def _tiny_ann_config() -> HarmonyModelConfig:
        # Minimal config purely for the engram modules' shape checks.
        return HarmonyModelConfig(
            num_layers=4, hidden_dim=32, num_query_heads=4, num_kv_heads=2,
            head_dim=8, ffn_dim=64, vocab_size=128, max_seq_len=64,
            rope_theta=10000.0, rms_norm_eps=1e-6, layers_per_block=2,
            engram_injection_layer=1, engram_dim=16, tie_embeddings=True,
            use_ann_engram=True,
        )

    def test_save_and_load_via_ann_injection(self, tmp_path):
        rng = np.random.default_rng(42)
        N, D = 20, 16
        table = rng.standard_normal((N, D)).astype(np.float32)
        out = tmp_path / "oracle.safetensors"
        save_oracle_table(table, out)
        assert out.exists()

        loaded = EngramANNInjection.load_corpus_table(out)
        assert loaded.shape == (N, D)
        np.testing.assert_allclose(loaded.numpy(), table, rtol=1e-6)

    @pytest.mark.skipif(not _HAS_XATTN, reason="Model delta (xattn) lives on PR #238")
    def test_save_and_load_via_xattn(self, tmp_path):
        rng = np.random.default_rng(43)
        N, D = 15, 16
        table = rng.standard_normal((N, D)).astype(np.float32)
        out = tmp_path / "oracle.safetensors"
        save_oracle_table(table, out)
        loaded = EngramCrossAttention.load_corpus_table(out)
        assert loaded.shape == (N, D)
        np.testing.assert_allclose(loaded.numpy(), table, rtol=1e-6)

    def test_save_and_load_via_engram_table(self, tmp_path):
        """The production `EngramTable.from_safetensors()` loader must
        also accept the generator's output (it's the canonical schema
        for student-side retrieval)."""
        rng = np.random.default_rng(44)
        N, D = 30, 16
        table = rng.standard_normal((N, D)).astype(np.float32)
        out = tmp_path / "oracle.safetensors"
        save_oracle_table(table, out)
        engram_table = EngramTable.from_safetensors(
            out, hash_seeds=[42, 99, 137, 251],
        )
        assert engram_table.total_entries == N
        assert engram_table.engram_dim == D
        np.testing.assert_allclose(engram_table.table.numpy(), table, rtol=1e-6)

    def test_loaded_oracle_usable_in_ann_module(self, tmp_path):
        """End-to-end: loaded oracle must construct a valid ANN module."""
        config = self._tiny_ann_config()
        rng = np.random.default_rng(45)
        table_np = rng.standard_normal((50, config.engram_dim)).astype(np.float32)
        out = tmp_path / "oracle.safetensors"
        save_oracle_table(table_np, out)
        loaded = EngramANNInjection.load_corpus_table(out)
        module = EngramANNInjection(config, loaded)
        # Smoke: forward pass runs without shape errors
        h = torch.randn(2, 5, config.hidden_dim)
        residual, gate = module(h)
        assert residual.shape == (2, 5, config.hidden_dim)
        assert gate.shape == (2, 5, 1)

    @pytest.mark.skipif(not _HAS_XATTN, reason="Model delta (xattn) lives on PR #238")
    def test_loaded_oracle_usable_in_xattn_module(self, tmp_path):
        """End-to-end: loaded oracle must construct a valid xattn module."""
        config = self._tiny_ann_config()
        config.use_ann_engram = False
        # Guard for configs that lack the xattn flag (PR #238 pre-merge)
        if hasattr(config, "use_xattn_engram"):
            config.use_xattn_engram = True
        rng = np.random.default_rng(46)
        table_np = rng.standard_normal((50, config.engram_dim)).astype(np.float32)
        out = tmp_path / "oracle.safetensors"
        save_oracle_table(table_np, out)
        loaded = EngramCrossAttention.load_corpus_table(out)
        module = EngramCrossAttention(config, loaded, k_retrieved=4)
        h = torch.randn(2, 5, config.hidden_dim)
        result = module(h)
        assert result.shape == (2, 5, config.hidden_dim)


# ---------------------------------------------------------------------------
# End-to-end composition: Welford + PCA + save, with synthetic data
# ---------------------------------------------------------------------------


@skip_if_no_sklearn
class TestPipelineComposition:
    """A tiny end-to-end test without the teacher model: use synthetic
    'teacher hidden states' + real Welford + real PCA + real save, then
    verify the student can load the output and get sensible retrievals."""

    def test_full_synthetic_pipeline(self, tmp_path):
        # Simulate 200 sequences of 8 tokens each, "teacher dim" 64,
        # target engram_dim 16, N=32 rows.
        rng = np.random.default_rng(100)
        N_rows = 32
        teacher_dim = 64
        target_dim = 16
        seeds = (42, 99)

        table = WelfordTable.zeros(N_rows, teacher_dim)
        for seq_idx in range(200):
            tokens = rng.integers(0, 100, size=8).tolist()
            # Synthetic "teacher hidden states": low-rank + noise
            hidden = (
                rng.standard_normal((8, 4)).astype(np.float32) @
                rng.standard_normal((4, teacher_dim)).astype(np.float32)
            )
            row_ids, positions = compute_ngram_indices_for_sequence(
                tokens, seeds, N_rows,
            )
            if not row_ids:
                continue
            indices = np.array(row_ids, dtype=np.int64)
            vectors = hidden[positions].astype(np.float32)
            table.update_batch(indices, vectors)

        mask = table.populated_mask
        assert mask.any(), "200 random sequences should populate at least one row"

        rng_pca = np.random.default_rng(101)
        components, mean_vec, variance = fit_pca_projection(
            means=table.means,
            populated_mask=mask,
            target_dim=target_dim,
            subsample_fraction=1.0,
            pca_batch_size=16,
            rng=rng_pca,
        )
        assert 0.0 <= variance <= 1.0
        projected = apply_pca_projection(table.means, components, mean_vec)
        assert projected.shape == (N_rows, target_dim)

        out = tmp_path / "synth_oracle.safetensors"
        save_oracle_table(projected, out)
        loaded = EngramANNInjection.load_corpus_table(out)
        np.testing.assert_allclose(loaded.numpy(), projected, rtol=1e-6)


# ---------------------------------------------------------------------------
# Regression tests for PR #239 review feedback
# ---------------------------------------------------------------------------


class TestWelfordVectorizedEquivalence:
    """The vectorized batch updater must produce the same running mean
    as the per-observation Welford identity (modulo floating-point
    accumulation order). A bug here silently corrupts every oracle row.
    """

    def test_batch_equals_scalar_sequence(self):
        rng = np.random.default_rng(7)
        total_entries = 16
        dim = 8
        k_obs = 200

        ref_means = np.zeros((total_entries, dim), dtype=np.float64)
        ref_counts = np.zeros(total_entries, dtype=np.int64)
        indices = rng.integers(0, total_entries, size=k_obs).astype(np.int64)
        vectors = rng.standard_normal((k_obs, dim)).astype(np.float32)
        # Reference: classic per-observation Welford
        for idx, vec in zip(indices.tolist(), vectors):
            ref_counts[idx] += 1
            c = ref_counts[idx]
            ref_means[idx] += (vec.astype(np.float64) - ref_means[idx]) / c

        # Split the observation stream into two batches and feed through
        # the vectorized update path. Result must match ref_means to
        # within f32 precision.
        table = WelfordTable.zeros(total_entries, dim)
        half = k_obs // 2
        table.update_batch(indices[:half], vectors[:half])
        table.update_batch(indices[half:], vectors[half:])

        np.testing.assert_array_equal(table.counts, ref_counts)
        np.testing.assert_allclose(table.means, ref_means.astype(np.float32), atol=1e-5)

    def test_empty_batch_is_noop(self):
        table = WelfordTable.zeros(4, 3)
        table.update_batch(np.array([], dtype=np.int64), np.zeros((0, 3), dtype=np.float32))
        assert (table.counts == 0).all()
        assert (table.means == 0).all()


@skip_if_no_sklearn
class TestPCAPreconditions:
    """Fail loudly before IncrementalPCA would raise AttributeError on
    .components_ from an unfit estimator. Smoke runs and
    misconfigured CLI invocations are the common triggers."""

    def test_raises_when_populated_below_target_dim(self):
        data = np.random.default_rng(0).standard_normal((5, 16)).astype(np.float32)
        mask = np.ones(5, dtype=bool)
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="populated rows"):
            fit_pca_projection(
                means=data, populated_mask=mask, target_dim=8,
                subsample_fraction=1.0, pca_batch_size=16, rng=rng,
            )

    def test_raises_when_pca_batch_below_target_dim(self):
        data = np.random.default_rng(1).standard_normal((100, 16)).astype(np.float32)
        mask = np.ones(100, dtype=bool)
        rng = np.random.default_rng(1)
        with pytest.raises(ValueError, match="pca-batch-size"):
            fit_pca_projection(
                means=data, populated_mask=mask, target_dim=8,
                subsample_fraction=1.0, pca_batch_size=4, rng=rng,
            )


@skip_if_no_sklearn
class TestPCAPreservesUnpopulatedZero:
    """Rows never touched by Welford must remain zero after PCA
    projection when `populated_mask` is passed. Otherwise the student
    reads a constant -mean@W vector for hashes that never occurred,
    silently contaminating retrieval."""

    def test_unpopulated_rows_stay_zero(self):
        rng = np.random.default_rng(99)
        data = np.zeros((200, 32), dtype=np.float32)
        # Only the first 150 rows have real data; last 50 stay zero.
        data[:150] = rng.standard_normal((150, 32)).astype(np.float32) * 4.0
        mask = np.zeros(200, dtype=bool)
        mask[:150] = True

        rng_pca = np.random.default_rng(100)
        components, mean_vec, _ = fit_pca_projection(
            means=data, populated_mask=mask, target_dim=8,
            subsample_fraction=1.0, pca_batch_size=16, rng=rng_pca,
        )
        projected = apply_pca_projection(
            data, components, mean_vec, populated_mask=mask,
        )
        # Populated rows: at least some nonzero entries (sanity).
        assert np.any(projected[:150] != 0.0)
        # Unpopulated rows: must be exactly zero.
        assert (projected[150:] == 0.0).all(), (
            "unpopulated rows leaked a nonzero vector after PCA"
        )

    def test_without_mask_unpopulated_rows_leak(self):
        """Defensive: passing no mask leaves the old-behavior leak
        visible, which is the exact regression we added the mask to
        avoid. Ensures the fix is load-bearing."""
        data = np.zeros((50, 16), dtype=np.float32)
        data[:40] = np.random.default_rng(0).standard_normal((40, 16)).astype(np.float32) * 2.0
        mask = np.zeros(50, dtype=bool)
        mask[:40] = True
        rng_pca = np.random.default_rng(1)
        components, mean_vec, _ = fit_pca_projection(
            means=data, populated_mask=mask, target_dim=4,
            subsample_fraction=1.0, pca_batch_size=8, rng=rng_pca,
        )
        projected_no_mask = apply_pca_projection(data, components, mean_vec)
        # The populated-rows mean is non-trivially nonzero, so centering
        # shifts zero inputs to -mean_vec and projects them to a
        # nonzero vector. If this assertion ever fires, apply_pca has
        # changed its behavior and the other test above may also
        # need review.
        assert not (projected_no_mask[40:] == 0.0).all()


@skip_if_no_transformers
class TestVocabSizeGuard:
    """generate_oracle_table.run_teacher_pass must abort before loading
    the teacher if its tokenizer's vocab_size disagrees with the
    dataset's. Silent garbage here is the worst-case diagnostic failure.
    """

    def test_mismatched_vocab_raises_before_model_load(self, monkeypatch):
        import ct87.generate_oracle_table as gen

        class _FakeTok:
            vocab_size = 151646  # Qwen-style, wrong for Mistral corpus
            pad_token_id = 0

        class _TokCls:
            @staticmethod
            def from_pretrained(_id):
                return _FakeTok()

        class _ModelCls:
            @staticmethod
            def from_pretrained(*_args, **_kwargs):
                raise AssertionError("must not reach model load after vocab mismatch")

        # Patch the imports inside run_teacher_pass. We intercept the
        # two factories it uses and raise from the model path to prove
        # the vocab guard fires first.
        import transformers
        monkeypatch.setattr(transformers, "AutoTokenizer", _TokCls)
        monkeypatch.setattr(transformers, "AutoModel", _ModelCls)
        import datasets as hfds
        monkeypatch.setattr(
            hfds, "load_from_disk",
            lambda _p: (_ for _ in ()).throw(AssertionError("must not load dataset")),
        )

        with pytest.raises(ValueError, match="vocab_size"):
            gen.run_teacher_pass(
                teacher_model_id="mock/teacher",
                tokenized_dataset_path="/tmp/should_not_load",
                layer_index=-2,
                total_entries=10,
                hash_seeds=(42,),
                batch_size=1,
                seq_len=4,
                max_sequences=1,
                device="cpu",
                dtype="float32",
                expected_vocab_size=32000,
            )


class TestHashSeedsArgparse:
    """--hash-seeds should validate at parse time so invalid CLI input
    produces a standard argparse usage error, not a raw traceback."""

    def test_valid_tuple_default(self):
        from ct87.generate_oracle_table import (
            build_argparser, DEFAULT_HASH_SEEDS,
        )
        args = build_argparser().parse_args(
            ["--dataset", "/tmp/fake", "--output", "/tmp/x.safetensors"]
        )
        assert args.hash_seeds == DEFAULT_HASH_SEEDS
        assert isinstance(args.hash_seeds, tuple)

    def test_custom_comma_string_parses_to_tuple(self):
        from ct87.generate_oracle_table import build_argparser
        args = build_argparser().parse_args([
            "--dataset", "/tmp/fake",
            "--output", "/tmp/x.safetensors",
            "--hash-seeds", "1,2,3",
        ])
        assert args.hash_seeds == (1, 2, 3)

    def test_invalid_seeds_raise_systemexit_with_usage(self, capsys):
        from ct87.generate_oracle_table import build_argparser
        with pytest.raises(SystemExit):
            build_argparser().parse_args([
                "--dataset", "/tmp/fake",
                "--output", "/tmp/x.safetensors",
                "--hash-seeds", "not,a,number",
            ])
        captured = capsys.readouterr()
        # argparse writes the usage/error to stderr; the key signal is
        # that we got there via argparse and not a bare ValueError from
        # main().
        assert "hash-seeds" in captured.err.lower() or "invalid" in captured.err.lower()


@skip_if_no_sentence_transformers
class TestSparsePCAGuard:
    """Sparse embedder PCA must fail loudly when n_samples < engram_dim
    instead of letting sklearn raise its less-actionable error."""

    def test_small_n_samples_raises_with_actionable_message(self):
        # 5 vectors, 384-dim native, target 128 -> should refuse.
        from ct87.generate_sparse_uncollided_table import embed_ngram_texts

        # Stub SentenceTransformer so we don't need network/deps: patch
        # via a simple subclass with a fixed encode return value.
        class _FakeST:
            def __init__(self, *_a, **_k):
                pass
            def get_sentence_embedding_dimension(self):
                return 384
            def encode(self, texts, **_kwargs):
                # Return 5 x 384 to trip the guard against target=128.
                return np.random.default_rng(0).standard_normal(
                    (len(texts), 384)
                ).astype(np.float32)

        import sentence_transformers as st_mod
        real_st = st_mod.SentenceTransformer
        st_mod.SentenceTransformer = _FakeST
        try:
            with pytest.raises(ValueError, match="engram-dim"):
                embed_ngram_texts(
                    ngram_texts=["a", "b", "c", "d", "e"],
                    embedder_model_id="mock",
                    engram_dim=128,
                    device="cpu",
                )
        finally:
            st_mod.SentenceTransformer = real_st


@skip_if_no_transformers
class TestLayerBoundsGuard:
    """Out-of-range --layer must fail before the first batch instead of
    discovering the problem via IndexError after 22 hours of teacher
    forward passes."""

    def test_layer_too_large_raises(self, monkeypatch):
        import ct87.generate_oracle_table as gen

        class _FakeTok:
            vocab_size = 32000
            pad_token_id = 0

        class _FakeCfg:
            num_hidden_layers = 12
            hidden_size = 64

        class _FakeModel:
            config = _FakeCfg()
            def to(self, _dev): return self
            def train(self, _flag): return self

        class _TokCls:
            @staticmethod
            def from_pretrained(_id): return _FakeTok()

        class _ModelCls:
            @staticmethod
            def from_pretrained(*_args, **_kwargs): return _FakeModel()

        import transformers
        monkeypatch.setattr(transformers, "AutoTokenizer", _TokCls)
        monkeypatch.setattr(transformers, "AutoModel", _ModelCls)
        import datasets as hfds
        monkeypatch.setattr(
            hfds, "load_from_disk",
            lambda _p: (_ for _ in ()).throw(AssertionError("must not load dataset")),
        )

        # layer_index=50 resolves to 50, out of range [0, 12].
        with pytest.raises(ValueError, match="--layer=50"):
            gen.run_teacher_pass(
                teacher_model_id="mock/teacher",
                tokenized_dataset_path="/tmp/should_not_load",
                layer_index=50,
                total_entries=10,
                hash_seeds=(42,),
                batch_size=1,
                seq_len=4,
                max_sequences=1,
                device="cpu",
                dtype="float32",
                expected_vocab_size=32000,
            )
