"""Tests for the data preprocessing pipeline."""

from __future__ import annotations

import pytest

from ct87.prepare_data import concatenate_and_chunk, split_chunks


class TestConcatenateAndChunk:
    def test_basic_chunking(self):
        """Two short docs with EOS between them, chunked to seq_len=4."""
        documents = [[10, 20, 30], [40, 50]]
        eos_token_id = 2
        # Stream: [10, 20, 30, 2, 40, 50, 2] -> length 7
        # seq_len=4 -> one full chunk [10, 20, 30, 2], remainder [40, 50, 2] discarded
        chunks = concatenate_and_chunk(documents, seq_len=4, eos_token_id=eos_token_id)
        assert len(chunks) == 1
        assert chunks[0] == [10, 20, 30, 2]

    def test_multiple_chunks(self):
        """Enough tokens for two full chunks."""
        documents = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        # Stream: [1, 2, 3, 2, 4, 5, 6, 2, 7, 8, 9, 2] -> length 12
        # seq_len=5 -> chunks: [1,2,3,2,4], [5,6,2,7,8], remainder [9,2] discarded
        chunks = concatenate_and_chunk(documents, seq_len=5, eos_token_id=2)
        assert len(chunks) == 2
        assert chunks[0] == [1, 2, 3, 2, 4]
        assert chunks[1] == [5, 6, 2, 7, 8]

    def test_empty_documents(self):
        """Empty input produces no chunks."""
        chunks = concatenate_and_chunk([], seq_len=4, eos_token_id=2)
        assert chunks == []

    def test_all_tokens_used(self):
        """When total length is exact multiple of seq_len, no tokens wasted."""
        # [10, 2, 20, 2] -> length 4, seq_len=4 -> 1 chunk
        chunks = concatenate_and_chunk([[10], [20]], seq_len=4, eos_token_id=2)
        assert len(chunks) == 1
        assert chunks[0] == [10, 2, 20, 2]

    def test_eos_at_document_boundaries(self):
        """EOS token appears at correct positions between documents."""
        documents = [[100, 200], [300, 400]]
        # Stream: [100, 200, 2, 300, 400, 2] -> length 6
        chunks = concatenate_and_chunk(documents, seq_len=6, eos_token_id=2)
        assert len(chunks) == 1
        assert chunks[0] == [100, 200, 2, 300, 400, 2]


class TestSplitChunks:
    def test_basic_split(self):
        """100 chunks with val_fraction=0.1 -> 90 train, 10 val."""
        chunks = [[i] for i in range(100)]
        train, val = split_chunks(chunks, val_fraction=0.1)
        assert len(train) == 90
        assert len(val) == 10

    def test_no_overlap(self):
        """Train and val sets have no overlapping chunks."""
        chunks = [list(range(i, i + 4)) for i in range(50)]
        train, val = split_chunks(chunks, val_fraction=0.2)
        train_set = {tuple(c) for c in train}
        val_set = {tuple(c) for c in val}
        assert len(train_set & val_set) == 0

    def test_all_chunks_preserved(self):
        """Total chunks in train + val equals input after rounding."""
        chunks = [[i] for i in range(37)]
        train, val = split_chunks(chunks, val_fraction=0.15)
        # int(37 * 0.15) = 5 -> 32 train, 5 val
        assert len(train) == 32
        assert len(val) == 5
        assert len(train) + len(val) == 37

    def test_zero_val_fraction(self):
        """val_fraction=0 puts everything in train."""
        chunks = [[i] for i in range(10)]
        train, val = split_chunks(chunks, val_fraction=0.0)
        assert len(train) == 10
        assert len(val) == 0


import tempfile
import os


class TestEndToEnd:
    @pytest.mark.network
    def test_smoke_prepare_data(self):
        """Run the full pipeline on a tiny slice and verify output loads."""
        from ct87.prepare_data import run_prepare_data

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "test_output")
            stats = run_prepare_data(
                output_dir=output_dir,
                seq_len=64,
                max_tokens=5000,
                val_fraction=0.1,
            )

            assert stats["total_tokens"] >= 5000
            assert stats["num_train_chunks"] > 0
            assert stats["num_val_chunks"] >= 0
            assert stats["num_documents"] > 0

            # Verify train split loads and has correct format
            from datasets import load_from_disk

            train_ds = load_from_disk(os.path.join(output_dir, "train"))
            assert len(train_ds) == stats["num_train_chunks"]
            assert "input_ids" in train_ds.column_names
            assert len(train_ds[0]["input_ids"]) == 64

            # Verify val split loads if it exists
            val_path = os.path.join(output_dir, "val")
            if os.path.exists(val_path):
                val_ds = load_from_disk(val_path)
                assert len(val_ds) == stats["num_val_chunks"]
                assert len(val_ds[0]["input_ids"]) == 64

    @pytest.mark.network
    def test_output_compatible_with_dataloader(self):
        """Output loads via make_hf_dataloader and produces correct batch shapes."""
        from ct87.prepare_data import run_prepare_data
        from ct87.train import make_hf_dataloader

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "test_output")
            run_prepare_data(
                output_dir=output_dir,
                seq_len=64,
                max_tokens=5000,
                val_fraction=0.0,
            )

            train_path = os.path.join(output_dir, "train")
            dl = make_hf_dataloader(train_path, seq_len=32, batch_size=2, seed=42)
            batch = next(dl)
            assert batch.shape == (2, 33)  # batch_size, seq_len + 1
            assert batch.min().item() >= 0
            assert batch.max().item() < 32000
