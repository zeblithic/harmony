"""Preprocess FineWeb-Edu into tokenized training data for ct87.

Streams text from HuggingFace, tokenizes with the Mistral v0.1 tokenizer,
concatenates documents with EOS separators, chunks into fixed-length
sequences, splits into train/val, and saves as HuggingFace Arrow datasets.

Run from the training/ directory:
    python3 -m ct87.prepare_data \
        --output ../data/fineweb-edu-poc \
        --seq-len 2048 \
        --max-tokens 100_000_000
"""

from __future__ import annotations

import argparse
import os
import sys


def concatenate_and_chunk(
    documents: list[list[int]],
    seq_len: int,
    eos_token_id: int,
) -> list[list[int]]:
    """Concatenate token sequences with EOS separators, then chunk.

    Documents are joined into one stream: [doc1..., EOS, doc2..., EOS, ...].
    The stream is sliced into non-overlapping chunks of exactly seq_len
    tokens. The final partial chunk (if any) is discarded.
    """
    stream: list[int] = []
    for doc in documents:
        stream.extend(doc)
        stream.append(eos_token_id)

    chunks = []
    for start in range(0, len(stream) - seq_len + 1, seq_len):
        chunks.append(stream[start : start + seq_len])
    return chunks


def split_chunks(
    chunks: list[list[int]],
    val_fraction: float,
) -> tuple[list[list[int]], list[list[int]]]:
    """Split chunks into train and validation sets.

    Validation chunks are taken from the end (deterministic, no shuffle).
    """
    n_val = int(len(chunks) * val_fraction)
    if n_val == 0 and val_fraction > 0 and len(chunks) > 0:
        n_val = 0  # not enough chunks for even 1 val sample
    train = chunks[: len(chunks) - n_val] if n_val > 0 else list(chunks)
    val = chunks[len(chunks) - n_val :] if n_val > 0 else []
    return train, val
