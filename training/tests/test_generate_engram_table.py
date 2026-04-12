"""Tests for corpus-based engram table generation."""

import numpy as np
import torch
import pytest

from ct87.generate_engram_table import (
    generate_corpus_table,
    DEFAULT_HASH_SEEDS,
    DEFAULT_ENTRIES,
    DEFAULT_DIM,
)


class TestGenerateCorpusTable:
    def test_output_shape(self):
        """Table should be [entries, dim] float32."""
        chunks = [[1, 2, 3, 4, 5, 6, 7, 8]] * 3
        table = generate_corpus_table(
            chunks, total_entries=100, embedding_dim=16,
            vocab_size=32, hash_seeds=[42, 99],
        )
        assert table.shape == (100, 16)
        assert table.dtype == np.float32

    def test_unit_normalized(self):
        """Non-zero rows should be unit-normalized."""
        chunks = [[1, 2, 3, 4, 5, 6, 7, 8]] * 10
        table = generate_corpus_table(
            chunks, total_entries=100, embedding_dim=16,
            vocab_size=32, hash_seeds=[42, 99],
        )
        for i in range(100):
            norm = np.linalg.norm(table[i])
            if norm > 0:
                assert abs(norm - 1.0) < 1e-5, f"Row {i} norm={norm}"

    def test_zero_rows_for_unused_entries(self):
        """Entries with no n-gram hits should be zero vectors."""
        chunks = [[1, 2, 3, 4]]
        table = generate_corpus_table(
            chunks, total_entries=10000, embedding_dim=16,
            vocab_size=32, hash_seeds=[42],
        )
        zero_count = sum(1 for i in range(10000) if np.allclose(table[i], 0))
        assert zero_count > 0, "Expected some unused entries"

    def test_deterministic(self):
        """Same corpus + seeds should produce identical table."""
        chunks = [[10, 20, 30, 40, 50]] * 5
        t1 = generate_corpus_table(
            chunks, total_entries=100, embedding_dim=16,
            vocab_size=100, hash_seeds=[42, 99],
        )
        t2 = generate_corpus_table(
            chunks, total_entries=100, embedding_dim=16,
            vocab_size=100, hash_seeds=[42, 99],
        )
        np.testing.assert_array_equal(t1, t2)

    def test_different_corpus_different_table(self):
        """Different corpus content should produce different tables."""
        chunks_a = [[1, 2, 3, 4, 5, 6, 7, 8]] * 5
        chunks_b = [[10, 20, 30, 40, 50, 60, 70, 80]] * 5
        t_a = generate_corpus_table(
            chunks_a, total_entries=100, embedding_dim=16,
            vocab_size=100, hash_seeds=[42, 99],
        )
        t_b = generate_corpus_table(
            chunks_b, total_entries=100, embedding_dim=16,
            vocab_size=100, hash_seeds=[42, 99],
        )
        assert not np.allclose(t_a, t_b)

    def test_nonzero_entries_have_signal(self):
        """Entries with n-gram hits should have non-zero vectors."""
        chunks = [[1, 2, 3, 4, 5, 6, 7, 8]] * 20
        table = generate_corpus_table(
            chunks, total_entries=100, embedding_dim=16,
            vocab_size=32, hash_seeds=[42, 99],
        )
        nonzero_count = sum(1 for i in range(100) if not np.allclose(table[i], 0))
        assert nonzero_count > 0, "Expected some non-zero entries"
