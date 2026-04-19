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


class TestSumAccumulatorTable:
    """SumAccumulatorTable replaces WelfordTable for the wide-vector logits
    path. Numerically equivalent (sum/count = running mean) but defers the
    divide to end-of-pass to dodge the per-batch combine cost that made
    WelfordTable ~24x slower than the hidden path at vocab=32000.
    """

    def test_zeros_shape_and_dtypes(self):
        from ct87.generate_oracle_table import SumAccumulatorTable

        t = SumAccumulatorTable.zeros(10, 4)
        assert t.sum.shape == (10, 4)
        assert t.sum.dtype == np.float64
        assert t.counts.shape == (10,)
        assert t.counts.dtype == np.int64
        assert not t.populated_mask.any()
        # Empty means: no NaN — unpopulated rows return zero (so the
        # downstream sidecar consumer can treat the array as dense).
        assert np.all(t.means == 0)

    def test_update_batch_records_sum_and_counts(self):
        from ct87.generate_oracle_table import SumAccumulatorTable

        t = SumAccumulatorTable.zeros(4, 3)
        # Two updates to row 0, one each to rows 1 and 2.
        t.update_batch(
            np.array([0, 0, 1, 2], dtype=np.int64),
            np.array([
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [10.0, 20.0, 30.0],
                [-1.0, -1.0, -1.0],
            ], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            t.counts, np.array([2, 1, 1, 0], dtype=np.int64),
        )
        np.testing.assert_allclose(t.sum[0], [5.0, 7.0, 9.0])
        np.testing.assert_allclose(t.sum[1], [10.0, 20.0, 30.0])
        np.testing.assert_allclose(t.sum[2], [-1.0, -1.0, -1.0])
        np.testing.assert_array_equal(t.sum[3], [0.0, 0.0, 0.0])

    def test_means_match_welford_running_mean(self):
        """Sum/count over multiple batches must equal Welford's running
        mean to within float32 precision. This is the load-bearing
        equivalence claim — if this breaks, the oracle's row values no
        longer mean what the spec says they mean."""
        from ct87.generate_oracle_table import (
            SumAccumulatorTable, WelfordTable,
        )

        rng = np.random.default_rng(0)
        sum_table = SumAccumulatorTable.zeros(16, 64)
        welford = WelfordTable.zeros(16, 64)
        # 5 batches of varying size; some duplicate row indices.
        for _ in range(5):
            k = int(rng.integers(8, 32))
            indices = rng.integers(0, 16, size=k).astype(np.int64)
            vectors = rng.standard_normal((k, 64)).astype(np.float32)
            sum_table.update_batch(indices, vectors)
            welford.update_batch(indices, vectors)

        np.testing.assert_array_equal(sum_table.counts, welford.counts)
        # Both compute mean = sum / count over the same observations,
        # using fp64 accumulators internally; final fp32 values should
        # be very close.
        np.testing.assert_allclose(
            sum_table.means, welford.means, rtol=0, atol=1e-5,
        )

    def test_populated_mask_reflects_counts(self):
        from ct87.generate_oracle_table import SumAccumulatorTable

        t = SumAccumulatorTable.zeros(5, 2)
        t.update_batch(
            np.array([1, 3], dtype=np.int64),
            np.ones((2, 2), dtype=np.float32),
        )
        np.testing.assert_array_equal(
            t.populated_mask,
            np.array([False, True, False, True, False]),
        )

    def test_update_batch_rejects_misaligned_inputs(self):
        from ct87.generate_oracle_table import SumAccumulatorTable

        t = SumAccumulatorTable.zeros(4, 3)
        with pytest.raises(ValueError, match="must align by first dim"):
            t.update_batch(
                np.array([0, 1], dtype=np.int64),
                np.zeros((3, 3), dtype=np.float32),
            )
        with pytest.raises(ValueError, match="dim="):
            t.update_batch(
                np.array([0], dtype=np.int64),
                np.zeros((1, 5), dtype=np.float32),
            )

    def test_empty_indices_is_noop(self):
        from ct87.generate_oracle_table import SumAccumulatorTable

        t = SumAccumulatorTable.zeros(4, 3)
        t.update_batch(np.array([], dtype=np.int64), np.zeros((0, 3)))
        assert not t.populated_mask.any()
        assert np.all(t.sum == 0)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GpuSumAccumulatorTable requires CUDA",
)
class TestGpuSumAccumulatorTable:
    """GPU-resident sum + count accumulator, used in production runs to
    bypass the CPU memory-bandwidth bottleneck on the wide-vector logits
    path. Numerics must match the CPU `SumAccumulatorTable` to within
    fp64 precision (both accumulate in fp64).
    """

    def test_zeros_constructs_on_default_device(self):
        from ct87.generate_oracle_table import GpuSumAccumulatorTable

        t = GpuSumAccumulatorTable.zeros(8, 4)
        assert str(t._sum.device).startswith("cuda")
        assert t._sum.shape == (8, 4)
        assert t._sum.dtype == torch.float64
        assert t._counts.shape == (8,)
        assert t._counts.dtype == torch.long
        assert not t.populated_mask.any()
        assert np.all(t.means == 0)

    def test_update_batch_gpu_records_sum_and_counts(self):
        from ct87.generate_oracle_table import GpuSumAccumulatorTable

        t = GpuSumAccumulatorTable.zeros(4, 3)
        device = t._device
        indices = torch.tensor([0, 0, 1, 2], dtype=torch.long, device=device)
        vectors = torch.tensor([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [10.0, 20.0, 30.0],
            [-1.0, -1.0, -1.0],
        ], dtype=torch.float32, device=device)
        t.update_batch_gpu(indices, vectors)
        np.testing.assert_array_equal(
            t._counts.cpu().numpy(), np.array([2, 1, 1, 0]),
        )
        np.testing.assert_allclose(t._sum[0].cpu().numpy(), [5.0, 7.0, 9.0])
        np.testing.assert_allclose(t._sum[1].cpu().numpy(), [10.0, 20.0, 30.0])

    def test_means_match_cpu_sum_accumulator(self):
        """GPU and CPU accumulators must produce identical means (within
        fp64 precision) given the same inputs. Pins the equivalence
        contract — process_batch should produce bit-equivalent oracle
        artifacts whether running on CPU or GPU."""
        from ct87.generate_oracle_table import (
            GpuSumAccumulatorTable, SumAccumulatorTable,
        )

        rng = np.random.default_rng(0)
        gpu_t = GpuSumAccumulatorTable.zeros(16, 64)
        cpu_t = SumAccumulatorTable.zeros(16, 64)
        device = gpu_t._device
        for _ in range(5):
            k = int(rng.integers(8, 32))
            indices_np = rng.integers(0, 16, size=k).astype(np.int64)
            vectors_np = rng.standard_normal((k, 64)).astype(np.float32)
            cpu_t.update_batch(indices_np, vectors_np)
            gpu_t.update_batch_gpu(
                torch.from_numpy(indices_np).to(device),
                torch.from_numpy(vectors_np).to(device),
            )
        np.testing.assert_array_equal(gpu_t.populated_mask, cpu_t.populated_mask)
        np.testing.assert_allclose(gpu_t.means, cpu_t.means, rtol=0, atol=1e-5)

    def test_update_batch_gpu_rejects_misaligned_inputs(self):
        from ct87.generate_oracle_table import GpuSumAccumulatorTable

        t = GpuSumAccumulatorTable.zeros(4, 3)
        device = t._device
        with pytest.raises(ValueError, match="must align by first dim"):
            t.update_batch_gpu(
                torch.tensor([0, 1], dtype=torch.long, device=device),
                torch.zeros(3, 3, dtype=torch.float32, device=device),
            )
        with pytest.raises(ValueError, match="dim="):
            t.update_batch_gpu(
                torch.tensor([0], dtype=torch.long, device=device),
                torch.zeros(1, 5, dtype=torch.float32, device=device),
            )

    def test_empty_indices_is_noop_gpu(self):
        from ct87.generate_oracle_table import GpuSumAccumulatorTable

        t = GpuSumAccumulatorTable.zeros(4, 3)
        device = t._device
        t.update_batch_gpu(
            torch.tensor([], dtype=torch.long, device=device),
            torch.zeros(0, 3, device=device),
        )
        assert not t.populated_mask.any()


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
        for _ in range(200):
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
        for idx, vec in zip(indices.tolist(), vectors, strict=True):
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

    def test_small_n_samples_raises_with_actionable_message(self, monkeypatch):
        # 5 vectors, 384-dim native, target 128 -> should refuse.
        from ct87.generate_sparse_uncollided_table import embed_ngram_texts

        # Stub SentenceTransformer so we don't need network/weights:
        # a minimal shim with a fixed encode return value is enough.
        class _FakeST:
            def __init__(self, *_a, **_k):
                pass
            def get_sentence_embedding_dimension(self):
                return 384
            def encode(self, texts, **_kwargs):
                # Return len(texts) x 384 to trip the guard against target=128.
                return np.random.default_rng(0).standard_normal(
                    (len(texts), 384)
                ).astype(np.float32)

        # monkeypatch handles restoration automatically; no try/finally
        # or manual state management needed.
        monkeypatch.setattr(
            "sentence_transformers.SentenceTransformer", _FakeST
        )
        with pytest.raises(ValueError, match="engram-dim"):
            embed_ngram_texts(
                ngram_texts=["a", "b", "c", "d", "e"],
                embedder_model_id="mock",
                engram_dim=128,
                device="cpu",
            )


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


class TestHarmonyTeacherURI:
    """ZEB-138 prep: `--teacher harmony:<path>` dispatches to the
    Harmony-architecture loader instead of HuggingFace AutoModel. These
    tests round-trip a tiny HarmonyModel through a temp checkpoint to
    cover the loader, the misuse-guard config-clear, the resolved-layer
    semantics, and the adapter's `outputs.hidden_states[N]` accessor.
    """

    @staticmethod
    def _build_tiny_harmony_checkpoint(tmp_path: Path):
        """Save a tiny HarmonyModel to a checkpoint file in train.py's
        resumable-checkpoint payload format and return its path + the
        live model (handy for shape comparisons).
        """
        from ct87.model import HarmonyModel, HarmonyModelConfig

        cfg = HarmonyModelConfig.tiny()
        model = HarmonyModel(cfg)
        ckpt_path = tmp_path / "tiny_harmony_teacher.pt"
        torch.save(
            {
                "step": 0,
                "model_state_dict": model.state_dict(),
                "config": cfg,
            },
            ckpt_path,
        )
        return ckpt_path, model

    def test_uri_prefix_dispatches_to_harmony_loader(self, monkeypatch):
        import ct87.generate_oracle_table as gen

        captured: dict = {}

        def _stub(ckpt_path, device, dtype, expected_vocab_size, layer_index):
            captured.update(
                ckpt_path=ckpt_path,
                device=device,
                dtype=dtype,
                expected_vocab_size=expected_vocab_size,
                layer_index=layer_index,
            )
            return ("MODEL", "TOK", 5, 64, "TORCH")

        monkeypatch.setattr(gen, "_load_harmony_teacher", _stub)
        result = gen.load_and_validate_teacher(
            teacher_model_id="harmony:/some/path/checkpoint.pt",
            device="cpu",
            dtype="float32",
            expected_vocab_size=32000,
            layer_index=-2,
        )
        assert result == ("MODEL", "TOK", 5, 64, "TORCH")
        # The URI prefix is stripped before reaching the loader.
        assert captured["ckpt_path"] == "/some/path/checkpoint.pt"
        assert captured["layer_index"] == -2

    def test_load_returns_signature_compatible_tuple(self, tmp_path, monkeypatch):
        import ct87.generate_oracle_table as gen

        class _FakeTok:
            vocab_size = 32000
            pad_token_id = 0

        monkeypatch.setattr(
            gen, "_load_harmony_compatible_tokenizer", lambda _v: _FakeTok()
        )
        ckpt_path, _model = self._build_tiny_harmony_checkpoint(tmp_path)

        adapter, tok, resolved_layer, teacher_dim, torch_mod = gen._load_harmony_teacher(
            ckpt_path=str(ckpt_path),
            device="cpu",
            dtype="float32",
            expected_vocab_size=32000,
            layer_index=-2,
        )
        assert tok.vocab_size == 32000
        assert teacher_dim == 512  # tiny.hidden_dim
        assert resolved_layer == 7  # tiny.num_layers (8) + 1 + (-2) = 7
        assert torch_mod is torch

    def test_adapter_forward_returns_hooked_layer(self, tmp_path, monkeypatch):
        """The adapter must run the underlying HarmonyModel and expose
        the resolved layer's hidden state via outputs.hidden_states[N]."""
        import ct87.generate_oracle_table as gen

        class _FakeTok:
            vocab_size = 32000
            pad_token_id = 0

        monkeypatch.setattr(
            gen, "_load_harmony_compatible_tokenizer", lambda _v: _FakeTok()
        )
        ckpt_path, _model = self._build_tiny_harmony_checkpoint(tmp_path)

        adapter, _tok, resolved_layer, teacher_dim, _ = gen._load_harmony_teacher(
            ckpt_path=str(ckpt_path),
            device="cpu",
            dtype="float32",
            expected_vocab_size=32000,
            layer_index=-2,
        )
        input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        outputs = adapter(input_ids=input_ids)
        captured = outputs.hidden_states[resolved_layer]
        assert captured.shape == (1, 4, teacher_dim)
        with pytest.raises(IndexError, match="only captured layer"):
            outputs.hidden_states[resolved_layer + 1]

    def test_misuse_guard_cleared_when_loading_capgap_checkpoint(
        self, tmp_path, monkeypatch,
    ):
        """A capgap config declares engram_inject_layers=(2,5), which
        triggers the misuse guard at HarmonyModel.forward when no
        engram_injections are attached. The teacher loader must clear
        the relevant flags so the bare backbone runs forward."""
        import dataclasses
        import ct87.generate_oracle_table as gen
        from ct87.model import HarmonyModel, HarmonyModelConfig

        class _FakeTok:
            vocab_size = 32000
            pad_token_id = 0

        monkeypatch.setattr(
            gen, "_load_harmony_compatible_tokenizer", lambda _v: _FakeTok()
        )

        cfg = HarmonyModelConfig.tiny_engram_xattn_capgap_vcontrast_qdiv()
        # Build with cleared inject_layers to get a clean state_dict, then
        # save the original cfg to mimic train.py's checkpoint payload.
        cfg_for_save = dataclasses.replace(
            cfg,
            engram_inject_layers=(),
            engram_vcontrast_enabled=False,
            engram_qdiv_enabled=False,
        )
        bare_model = HarmonyModel(cfg_for_save)
        ckpt_path = tmp_path / "capgap_teacher.pt"
        torch.save(
            {"step": 0, "model_state_dict": bare_model.state_dict(), "config": cfg},
            ckpt_path,
        )

        adapter, _, _, _, _ = gen._load_harmony_teacher(
            ckpt_path=str(ckpt_path),
            device="cpu",
            dtype="float32",
            expected_vocab_size=32000,
            layer_index=-2,
        )
        # Forward must not raise the multi-layer-injection misuse guard.
        adapter(input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long))

    def test_vocab_mismatch_in_config_raises(self, tmp_path, monkeypatch):
        import dataclasses
        import ct87.generate_oracle_table as gen
        from ct87.model import HarmonyModel, HarmonyModelConfig

        class _FakeTok:
            vocab_size = 32000
            pad_token_id = 0

        monkeypatch.setattr(
            gen, "_load_harmony_compatible_tokenizer", lambda _v: _FakeTok()
        )
        cfg = dataclasses.replace(HarmonyModelConfig.tiny(), vocab_size=50000)
        model = HarmonyModel(cfg)
        ckpt_path = tmp_path / "wrong_vocab.pt"
        torch.save(
            {"step": 0, "model_state_dict": model.state_dict(), "config": cfg},
            ckpt_path,
        )
        with pytest.raises(ValueError, match="vocab_size"):
            gen._load_harmony_teacher(
                ckpt_path=str(ckpt_path),
                device="cpu",
                dtype="float32",
                expected_vocab_size=32000,
                layer_index=-2,
            )

    def test_payload_without_config_raises_actionable(self, tmp_path, monkeypatch):
        import ct87.generate_oracle_table as gen

        class _FakeTok:
            vocab_size = 32000
            pad_token_id = 0

        monkeypatch.setattr(
            gen, "_load_harmony_compatible_tokenizer", lambda _v: _FakeTok()
        )
        ckpt_path = tmp_path / "stale_format.pt"
        torch.save({"step": 0, "model_state_dict": {}}, ckpt_path)
        with pytest.raises(ValueError, match=r"model_state_dict.*config"):
            gen._load_harmony_teacher(
                ckpt_path=str(ckpt_path),
                device="cpu",
                dtype="float32",
                expected_vocab_size=32000,
                layer_index=-2,
            )

    @pytest.mark.parametrize("malformed_uri", ["harmony:", "harmony:   ", "harmony:\t"])
    def test_empty_harmony_uri_path_raises_actionable(self, malformed_uri):
        """`--teacher harmony:` (or whitespace-only path) must fail with a
        helpful CLI message, not torch.load's opaque '[Errno 2]' OSError."""
        import ct87.generate_oracle_table as gen

        with pytest.raises(ValueError, match=r"non-empty checkpoint path"):
            gen.load_and_validate_teacher(
                teacher_model_id=malformed_uri,
                device="cpu",
                dtype="float32",
                expected_vocab_size=32000,
                layer_index=-2,
            )

    def test_payload_with_wrong_config_type_raises_actionable(self, tmp_path, monkeypatch):
        """Defense in depth: keys present but config deserialized as a plain dict.

        With weights_only=True + our restricted safe_globals allowlist, this
        wouldn't actually round-trip — the unpickle would reject it before
        we got here. But if a future train.py format saves config as a
        dict, the explicit type-check produces an actionable error instead
        of a confusing AttributeError on `config.vocab_size` later.
        """
        import ct87.generate_oracle_table as gen

        class _FakeTok:
            vocab_size = 32000
            pad_token_id = 0

        monkeypatch.setattr(
            gen, "_load_harmony_compatible_tokenizer", lambda _v: _FakeTok()
        )
        # Force the unpickler past safe_globals by writing a payload that
        # only uses primitive types — those are always allowed under
        # weights_only=True regardless of the safe_globals allowlist.
        ckpt_path = tmp_path / "config_as_dict.pt"
        torch.save(
            {"model_state_dict": {"x": torch.zeros(1)}, "config": {"vocab_size": 32000}},
            ckpt_path,
        )
        with pytest.raises(ValueError, match=r"expected payload\['config'\] to be a HarmonyModelConfig"):
            gen._load_harmony_teacher(
                ckpt_path=str(ckpt_path),
                device="cpu",
                dtype="float32",
                expected_vocab_size=32000,
                layer_index=-2,
            )

    def test_layer_zero_hooks_embedding(self, tmp_path, monkeypatch):
        """--layer 0 must capture embedding output, not block 0 output."""
        import ct87.generate_oracle_table as gen

        class _FakeTok:
            vocab_size = 32000
            pad_token_id = 0

        monkeypatch.setattr(
            gen, "_load_harmony_compatible_tokenizer", lambda _v: _FakeTok()
        )
        ckpt_path, _model = self._build_tiny_harmony_checkpoint(tmp_path)
        adapter, _, resolved_layer, _, _ = gen._load_harmony_teacher(
            ckpt_path=str(ckpt_path),
            device="cpu",
            dtype="float32",
            expected_vocab_size=32000,
            layer_index=0,
        )
        assert resolved_layer == 0
        # Adapter's hook target must be embed_tokens, not layers[0].
        assert adapter._hook_module is adapter._model.embed_tokens


@skip_if_no_transformers
class TestSaveTeacherLogits:
    """ZEB-139 prep: --save-teacher-logits gates a second WelfordTable that
    captures the teacher's full LM-head outputs and writes a bf16 sidecar
    alongside the main oracle. The training side (KL+CE retrofit) reads
    that sidecar to compute KL(P_router || P_teacher) without needing
    any extra metadata exchange.
    """

    @staticmethod
    def _patch_hf_for_logits(monkeypatch, *, hidden_dim: int, vocab: int):
        """Install a fake AutoTokenizer + AutoModelForCausalLM whose forward
        returns a tensor with `.hidden_states` (tuple of len num_layers+1)
        and `.logits` (shape [B, T, vocab]). Returns a dict the test can
        inspect to confirm which model class was used.
        """
        import transformers
        import datasets as hfds

        class _FakeTok:
            vocab_size = vocab
            pad_token_id = 0

        class _FakeCfg:
            num_hidden_layers = 4
            hidden_size = hidden_dim
            vocab_size = vocab

        class _FakeOutputs:
            def __init__(self, hidden_states, logits):
                self.hidden_states = hidden_states
                self.logits = logits

        class _FakeModel:
            config = _FakeCfg()
            def to(self, _dev): return self
            def train(self, _flag): return self
            def __call__(self, input_ids):
                B, T = input_ids.shape
                # 5 = num_hidden_layers + 1 (embedding output + 4 blocks).
                hs = tuple(
                    torch.full((B, T, hidden_dim), 0.5 * (i + 1))
                    for i in range(5)
                )
                # Make logits contain the position index in dim 0 so the
                # row-parity test can verify per-(b,p) lookup correctness.
                logits = torch.zeros((B, T, vocab))
                for b in range(B):
                    for t in range(T):
                        logits[b, t, 0] = float(b * 100 + t)
                return _FakeOutputs(hidden_states=hs, logits=logits)

        usage: dict = {"used_for_causal_lm": False, "used_automodel": False}

        class _AutoModelCls:
            @staticmethod
            def from_pretrained(*_args, **_kwargs):
                usage["used_automodel"] = True
                return _FakeModel()

        class _AutoModelForCausalLMCls:
            @staticmethod
            def from_pretrained(*_args, **_kwargs):
                usage["used_for_causal_lm"] = True
                return _FakeModel()

        class _TokCls:
            @staticmethod
            def from_pretrained(_id): return _FakeTok()

        monkeypatch.setattr(transformers, "AutoTokenizer", _TokCls)
        monkeypatch.setattr(transformers, "AutoModel", _AutoModelCls)
        monkeypatch.setattr(
            transformers, "AutoModelForCausalLM", _AutoModelForCausalLMCls,
        )

        # Stub the dataset to return a tiny, deterministic batch.
        class _FakeDataset:
            column_names = ["input_ids"]
            _data = {"input_ids": [
                [1, 2, 3, 4, 5, 6, 7, 8],
                [9, 10, 11, 12, 13, 14, 15, 16],
            ]}
            def __len__(self): return len(self._data["input_ids"])
            def __getitem__(self, key):
                if isinstance(key, slice):
                    return {"input_ids": self._data["input_ids"][key]}
                return self._data["input_ids"][key]

        monkeypatch.setattr(hfds, "load_from_disk", lambda _p: _FakeDataset())
        return usage

    def test_run_teacher_pass_returns_tuple_with_none_when_flag_off(
        self, monkeypatch,
    ):
        """Backward-compat: callers that didn't pass save_teacher_logits
        must get a (table, None) tuple — not a bare WelfordTable, but
        also not a populated logits table."""
        import ct87.generate_oracle_table as gen
        usage = self._patch_hf_for_logits(monkeypatch, hidden_dim=8, vocab=32000)

        table, logits_table = gen.run_teacher_pass(
            teacher_model_id="mock/teacher",
            tokenized_dataset_path="/tmp/fake",
            layer_index=-1,
            total_entries=64,
            hash_seeds=(42,),
            batch_size=2,
            seq_len=8,
            max_sequences=2,
            device="cpu",
            dtype="float32",
            expected_vocab_size=32000,
        )
        assert isinstance(table, gen.WelfordTable)
        assert logits_table is None
        # Default path should NOT load AutoModelForCausalLM.
        assert usage["used_automodel"] is True
        assert usage["used_for_causal_lm"] is False

    def test_flag_on_uses_AutoModelForCausalLM_and_returns_logits_table(
        self, monkeypatch,
    ):
        """--save-teacher-logits triggers the AutoModelForCausalLM upgrade
        AND populates a vocab-wide WelfordTable returned alongside the
        hidden table."""
        import ct87.generate_oracle_table as gen
        usage = self._patch_hf_for_logits(monkeypatch, hidden_dim=8, vocab=32000)

        table, logits_table = gen.run_teacher_pass(
            teacher_model_id="mock/teacher",
            tokenized_dataset_path="/tmp/fake",
            layer_index=-1,
            total_entries=64,
            hash_seeds=(42,),
            batch_size=2,
            seq_len=8,
            max_sequences=2,
            device="cpu",
            dtype="float32",
            expected_vocab_size=32000,
            save_teacher_logits=True,
        )
        assert isinstance(table, gen.WelfordTable)
        # Logits table is the throughput-optimized SumAccumulatorTable
        # (vs WelfordTable for the hidden path) — see SumAccumulatorTable
        # docstring for the rationale.
        assert isinstance(logits_table, gen.SumAccumulatorTable)
        assert logits_table.means.shape == (64, 32000)
        assert logits_table.means.dtype == np.float32
        # The two tables share the xxhash row-population pattern by
        # construction — they update at identical (b, p) positions.
        assert np.array_equal(table.populated_mask, logits_table.populated_mask)
        # Confirm the upgrade actually happened.
        assert usage["used_for_causal_lm"] is True
        assert usage["used_automodel"] is False

    def test_save_teacher_logits_with_harmony_uri_rejected(self, monkeypatch):
        """--save-teacher-logits + --teacher harmony:* must raise BEFORE
        any model load. The Harmony adapter only captures hidden states;
        ZEB-139 (TinyLlama-only) doesn't need this path."""
        import ct87.generate_oracle_table as gen

        # Stub _load_harmony_teacher to fail loudly if reached.
        monkeypatch.setattr(
            gen, "_load_harmony_teacher",
            lambda **_kw: (_ for _ in ()).throw(
                AssertionError("must not reach harmony loader"),
            ),
        )
        with pytest.raises(ValueError, match=r"--save-teacher-logits.*harmony:"):
            gen.load_and_validate_teacher(
                teacher_model_id="harmony:/path/to/ckpt.pt",
                device="cpu",
                dtype="float32",
                expected_vocab_size=32000,
                layer_index=-2,
                save_teacher_logits=True,
            )

    def test_default_sidecar_path_swaps_safetensors_suffix(self):
        from ct87.generate_oracle_table import default_teacher_logits_sidecar_path

        # Standard `.safetensors` output → suffix-swap.
        assert (
            default_teacher_logits_sidecar_path("/tmp/oracle.safetensors")
            == Path("/tmp/oracle_teacher_logits.safetensors")
        )
        # Unusual extension → append rule (still predictable).
        assert (
            default_teacher_logits_sidecar_path("/tmp/oracle.bin")
            == Path("/tmp/oracle.bin_teacher_logits.safetensors")
        )

    def test_save_teacher_logits_sidecar_round_trip_preserves_means_within_bf16(
        self, tmp_path,
    ):
        """save_teacher_logits_sidecar writes bf16 — round-trip must
        preserve the per-row means within bf16's ~3-decimal precision."""
        from safetensors.torch import load_file
        from ct87.generate_oracle_table import (
            TEACHER_LOGITS_SIDECAR_KEY, save_teacher_logits_sidecar,
        )

        rng = np.random.default_rng(0)
        # Logits typically span [-20, +20]; bf16 handles that range cleanly.
        means = rng.standard_normal((16, 256)).astype(np.float32) * 5.0
        out = tmp_path / "logits.safetensors"
        save_teacher_logits_sidecar(means, out)
        assert out.exists()

        loaded = load_file(str(out))
        assert TEACHER_LOGITS_SIDECAR_KEY in loaded
        recovered = loaded[TEACHER_LOGITS_SIDECAR_KEY]
        assert recovered.dtype is torch.bfloat16
        assert tuple(recovered.shape) == (16, 256)
        # bf16 has 7-bit mantissa → ~1e-2 relative precision in this range.
        recovered_f32 = recovered.float().numpy()
        np.testing.assert_allclose(recovered_f32, means, rtol=2e-2, atol=1e-2)

    def test_argparser_accepts_save_teacher_logits_flag(self):
        from ct87.generate_oracle_table import build_argparser

        args = build_argparser().parse_args([
            "--dataset", "/tmp/fake",
            "--output", "/tmp/x.safetensors",
            "--save-teacher-logits",
        ])
        assert args.save_teacher_logits is True
        assert args.teacher_logits_output is None  # uses default derivation

    def test_argparser_default_save_teacher_logits_off(self):
        from ct87.generate_oracle_table import build_argparser

        args = build_argparser().parse_args([
            "--dataset", "/tmp/fake",
            "--output", "/tmp/x.safetensors",
        ])
        assert args.save_teacher_logits is False
        assert args.teacher_logits_output is None

    def test_chunked_logits_update_matches_single_pass(self, monkeypatch):
        """The K-dimension chunk loop in process_batch must produce the
        same Welford means as a single unchunked update_batch call.

        Welford is order-invariant across calls (combines pre-batch mean
        with each chunk's contribution exactly), so chunking K must be
        bit-identical modulo float64 accumulation order. This test pins
        that contract — without it, a future "optimization" that reuses
        the chunk loop for the hidden path could silently change the
        oracle's row values.
        """
        import ct87.generate_oracle_table as gen

        # Force a small chunk size so the loop runs >1 iteration on
        # modest K values without bloating the test.
        monkeypatch.setattr(gen, "LOGITS_UPDATE_CHUNK_K", 7)

        usage = self._patch_hf_for_logits(monkeypatch, hidden_dim=8, vocab=32000)
        del usage  # unused here

        chunked_table, chunked_logits = gen.run_teacher_pass(
            teacher_model_id="mock/teacher",
            tokenized_dataset_path="/tmp/fake",
            layer_index=-1,
            total_entries=64,
            hash_seeds=(42,),
            batch_size=2,
            seq_len=8,
            max_sequences=2,
            device="cpu",
            dtype="float32",
            expected_vocab_size=32000,
            save_teacher_logits=True,
        )

        # Re-run with chunk size larger than any K we'd produce — single
        # pass through update_batch.
        monkeypatch.setattr(gen, "LOGITS_UPDATE_CHUNK_K", 10**9)
        single_table, single_logits = gen.run_teacher_pass(
            teacher_model_id="mock/teacher",
            tokenized_dataset_path="/tmp/fake",
            layer_index=-1,
            total_entries=64,
            hash_seeds=(42,),
            batch_size=2,
            seq_len=8,
            max_sequences=2,
            device="cpu",
            dtype="float32",
            expected_vocab_size=32000,
            save_teacher_logits=True,
        )

        # Hidden path is uneffected by chunking — must be identical.
        np.testing.assert_array_equal(chunked_table.means, single_table.means)
        np.testing.assert_array_equal(chunked_table.counts, single_table.counts)
        # Logits path: equality up to float64 reordering. Welford
        # combines as `(old_mean*old_n + sum)/new_n` per unique row;
        # changing the order rows arrive in within a chunk doesn't
        # change the per-row sum (each update is +=). So means must
        # match bit-exactly.
        np.testing.assert_array_equal(chunked_logits.counts, single_logits.counts)
        np.testing.assert_allclose(
            chunked_logits.means, single_logits.means,
            rtol=0, atol=1e-6,
        )

    def test_main_rejects_teacher_logits_output_without_save_flag(
        self, capsys, tmp_path,
    ):
        """--teacher-logits-output is only consulted when
        --save-teacher-logits is set, so passing the override without
        the flag silently produces no sidecar — a footgun where the
        user expects the path to be honored. main() must fail-fast
        with rc=2 (CLI misuse) before any teacher load."""
        import ct87.generate_oracle_table as gen

        rc = gen.main([
            "--dataset", "/tmp/ignored",
            "--output", str(tmp_path / "oracle.safetensors"),
            # Override path passed but flag NOT set — should reject.
            "--teacher-logits-output",
            str(tmp_path / "explicit_sidecar.safetensors"),
            "--entries", "8",
            "--engram-dim", "2",
        ])
        assert rc == 2, "expected rc=2 (CLI misuse) on flag/path inconsistency"
        captured = capsys.readouterr()
        assert "--teacher-logits-output" in captured.err
        assert "--save-teacher-logits" in captured.err

    def test_main_rejects_path_aliasing_between_oracle_and_sidecar(
        self, monkeypatch, tmp_path,
    ):
        """If --teacher-logits-output resolves to the same file as
        --output, the second save would silently overwrite the first.
        main() must reject this combination at startup with rc=2 (CLI
        misuse), before any write happens."""
        import ct87.generate_oracle_table as gen

        # Stub run_teacher_pass — we don't need a real forward pass to
        # exercise the path-aliasing guard, just two non-empty tables.
        hidden_table = gen.WelfordTable.zeros(8, 4)
        hidden_table.update_batch(
            np.array([0, 1, 2], dtype=np.int64),
            np.ones((3, 4), dtype=np.float32),
        )
        logits_table = gen.SumAccumulatorTable.zeros(8, 16)
        logits_table.update_batch(
            np.array([0, 1, 2], dtype=np.int64),
            np.ones((3, 16), dtype=np.float32),
        )
        monkeypatch.setattr(
            gen, "run_teacher_pass",
            lambda **_kw: (hidden_table, logits_table),
        )
        save_calls: list[str] = []
        monkeypatch.setattr(
            gen, "save_oracle_table",
            lambda *_a, **_k: save_calls.append("oracle"),
        )
        monkeypatch.setattr(
            gen, "save_teacher_logits_sidecar",
            lambda *_a, **_k: save_calls.append("sidecar"),
        )

        out_path = tmp_path / "oracle.safetensors"
        rc = gen.main([
            "--dataset", "/tmp/ignored",
            "--output", str(out_path),
            "--save-teacher-logits",
            # Deliberate aliasing — same file as --output.
            "--teacher-logits-output", str(out_path),
            "--entries", "8",
            "--engram-dim", "2",
            "--pca-batch-size", "2",
            "--pca-subsample-fraction", "1.0",
        ])
        assert rc == 2, "expected rc=2 (CLI misuse) on aliased paths"
        assert save_calls == [], (
            "neither oracle nor sidecar should be written when path "
            f"aliasing is detected; got writes: {save_calls}"
        )

    def test_main_atomic_write_cleans_up_tmps_on_sidecar_failure(
        self, monkeypatch, tmp_path,
    ):
        """If the sidecar save raises after the oracle .tmp is on disk,
        main() must clean up the .tmp files and propagate the error —
        no oracle.safetensors, no oracle.safetensors.tmp, no
        sidecar.safetensors.tmp left lying around."""
        import ct87.generate_oracle_table as gen

        hidden_table = gen.WelfordTable.zeros(8, 4)
        hidden_table.update_batch(
            np.array([0, 1, 2], dtype=np.int64),
            np.ones((3, 4), dtype=np.float32),
        )
        logits_table = gen.SumAccumulatorTable.zeros(8, 16)
        logits_table.update_batch(
            np.array([0, 1, 2], dtype=np.int64),
            np.ones((3, 16), dtype=np.float32),
        )
        monkeypatch.setattr(
            gen, "run_teacher_pass",
            lambda **_kw: (hidden_table, logits_table),
        )

        # Stub save_oracle_table to actually create the .tmp file (so
        # the cleanup path has something to unlink). Using a simple
        # touch — content doesn't matter for this test.
        def _stub_oracle_save(_means, output_path):
            Path(str(output_path)).touch()
        monkeypatch.setattr(gen, "save_oracle_table", _stub_oracle_save)

        # Sidecar save raises — simulates disk full, permission, etc.
        def _stub_sidecar_fail(*_a, **_k):
            raise OSError("simulated sidecar write failure")
        monkeypatch.setattr(
            gen, "save_teacher_logits_sidecar", _stub_sidecar_fail,
        )

        out_path = tmp_path / "oracle.safetensors"
        sidecar_path = tmp_path / "oracle_teacher_logits.safetensors"

        with pytest.raises(OSError, match="simulated sidecar write failure"):
            gen.main([
                "--dataset", "/tmp/ignored",
                "--output", str(out_path),
                "--save-teacher-logits",
                "--entries", "8",
                "--engram-dim", "2",
                "--pca-batch-size", "2",
                "--pca-subsample-fraction", "1.0",
            ])

        # Atomic-write contract: nothing should remain on disk. The
        # .tmp filenames now include PID for concurrent-run safety, so
        # glob the pattern instead of hardcoding the literal name.
        assert not out_path.exists(), (
            "oracle.safetensors must not exist when sidecar save fails"
        )
        oracle_tmps = list(tmp_path.glob("oracle.safetensors.*.tmp"))
        assert oracle_tmps == [], (
            f"oracle .tmp files must be cleaned up after sidecar "
            f"failure; leaked: {oracle_tmps}"
        )
        assert not sidecar_path.exists(), (
            "sidecar must not exist (it was the one that failed)"
        )
        sidecar_tmps = list(
            tmp_path.glob("oracle_teacher_logits.safetensors.*.tmp"),
        )
        assert sidecar_tmps == [], (
            f"sidecar .tmp files must be cleaned up too; leaked: "
            f"{sidecar_tmps}"
        )

    def test_main_does_not_write_oracle_when_logits_parity_fails(
        self, monkeypatch, tmp_path,
    ):
        """If the logits/hidden populated_mask parity check fails, main()
        must abort with rc=1 BEFORE writing any artifact. The reordering
        prevents a half-written `oracle.safetensors` from leaking onto
        disk that a downstream consumer might then pick up unaware."""
        import ct87.generate_oracle_table as gen

        # Build two WelfordTables with deliberately mismatched
        # populated_mask: hidden has rows 0..3 populated, logits has
        # rows 0..2 (missing row 3). Forces the parity guard to fire.
        hidden_table = gen.WelfordTable.zeros(8, 4)
        hidden_table.update_batch(
            np.array([0, 1, 2, 3], dtype=np.int64),
            np.ones((4, 4), dtype=np.float32),
        )
        logits_table = gen.SumAccumulatorTable.zeros(8, 16)
        logits_table.update_batch(
            np.array([0, 1, 2], dtype=np.int64),
            np.ones((3, 16), dtype=np.float32),
        )
        assert not np.array_equal(
            hidden_table.populated_mask, logits_table.populated_mask,
        )

        # Stub run_teacher_pass to skip the actual forward pass.
        monkeypatch.setattr(
            gen, "run_teacher_pass",
            lambda **_kw: (hidden_table, logits_table),
        )

        # Stub the PCA functions so we can detect whether they ran (we
        # want them to run because they're cheap and pure, but the file
        # writes must NOT happen).
        save_calls: list[str] = []
        monkeypatch.setattr(
            gen, "save_oracle_table",
            lambda *_a, **_k: save_calls.append("oracle"),
        )
        monkeypatch.setattr(
            gen, "save_teacher_logits_sidecar",
            lambda *_a, **_k: save_calls.append("sidecar"),
        )

        out_path = tmp_path / "oracle.safetensors"
        rc = gen.main([
            "--dataset", "/tmp/ignored",
            "--output", str(out_path),
            "--save-teacher-logits",
            "--entries", "8",
            "--engram-dim", "2",
            "--pca-batch-size", "2",
            "--pca-subsample-fraction", "1.0",
        ])
        assert rc == 1, "expected non-zero exit on parity mismatch"
        assert save_calls == [], (
            "neither oracle nor sidecar should be written when parity "
            f"check fails; got writes: {save_calls}"
        )
        assert not out_path.exists(), (
            "main() must not leave a half-written oracle.safetensors on "
            "disk after a parity failure"
        )
