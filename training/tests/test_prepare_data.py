"""Tests for the data preprocessing pipeline."""

from __future__ import annotations

import pytest


class TestConcatenateAndChunk:
    def test_basic_chunking(self):
        """Two short docs with EOS between them, chunked to seq_len=4."""
        from ct87.prepare_data import concatenate_and_chunk

        documents = [[10, 20, 30], [40, 50]]
        eos_token_id = 2
        # Stream: [10, 20, 30, 2, 40, 50, 2] -> length 7
        # seq_len=4 -> one full chunk [10, 20, 30, 2], remainder [40, 50, 2] discarded
        chunks = concatenate_and_chunk(documents, seq_len=4, eos_token_id=eos_token_id)
        assert len(chunks) == 1
        assert chunks[0] == [10, 20, 30, 2]

    def test_multiple_chunks(self):
        """Enough tokens for two full chunks."""
        from ct87.prepare_data import concatenate_and_chunk

        documents = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        # Stream: [1, 2, 3, 2, 4, 5, 6, 2, 7, 8, 9, 2] -> length 12
        # seq_len=5 -> chunks: [1,2,3,2,4], [5,6,2,7,8], remainder [9,2] discarded
        chunks = concatenate_and_chunk(documents, seq_len=5, eos_token_id=2)
        assert len(chunks) == 2
        assert chunks[0] == [1, 2, 3, 2, 4]
        assert chunks[1] == [5, 6, 2, 7, 8]

    def test_empty_documents(self):
        """Empty input produces no chunks."""
        from ct87.prepare_data import concatenate_and_chunk

        chunks = concatenate_and_chunk([], seq_len=4, eos_token_id=2)
        assert chunks == []

    def test_all_tokens_used(self):
        """When total length is exact multiple of seq_len, no tokens wasted."""
        from ct87.prepare_data import concatenate_and_chunk

        # [10, 2, 20, 2] -> length 4, seq_len=4 -> 1 chunk
        chunks = concatenate_and_chunk([[10], [20]], seq_len=4, eos_token_id=2)
        assert len(chunks) == 1
        assert chunks[0] == [10, 2, 20, 2]

    def test_eos_appears_at_boundaries(self):
        """EOS token appears in the stream between documents."""
        from ct87.prepare_data import concatenate_and_chunk

        documents = [[100, 200], [300, 400]]
        # Stream: [100, 200, 2, 300, 400, 2] -> length 6
        chunks = concatenate_and_chunk(documents, seq_len=6, eos_token_id=2)
        assert len(chunks) == 1
        assert 2 in chunks[0]


class TestSplitChunks:
    def test_basic_split(self):
        """100 chunks with val_fraction=0.1 -> 90 train, 10 val."""
        from ct87.prepare_data import split_chunks

        chunks = [[i] for i in range(100)]
        train, val = split_chunks(chunks, val_fraction=0.1)
        assert len(train) == 90
        assert len(val) == 10

    def test_no_overlap(self):
        """Train and val sets have no overlapping chunks."""
        from ct87.prepare_data import split_chunks

        chunks = [list(range(i, i + 4)) for i in range(50)]
        train, val = split_chunks(chunks, val_fraction=0.2)
        train_set = {tuple(c) for c in train}
        val_set = {tuple(c) for c in val}
        assert len(train_set & val_set) == 0

    def test_all_chunks_preserved(self):
        """Total chunks in train + val equals input."""
        from ct87.prepare_data import split_chunks

        chunks = [[i] for i in range(37)]
        train, val = split_chunks(chunks, val_fraction=0.01)
        assert len(train) + len(val) == 37

    def test_zero_val_fraction(self):
        """val_fraction=0 puts everything in train."""
        from ct87.prepare_data import split_chunks

        chunks = [[i] for i in range(10)]
        train, val = split_chunks(chunks, val_fraction=0.0)
        assert len(train) == 10
        assert len(val) == 0
