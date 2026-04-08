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
    train = chunks[: len(chunks) - n_val] if n_val > 0 else list(chunks)
    val = chunks[len(chunks) - n_val :] if n_val > 0 else []
    return train, val


def run_prepare_data(
    output_dir: str,
    seq_len: int = 2048,
    max_tokens: int | None = None,
    val_fraction: float = 0.01,
) -> dict[str, int]:
    """Run the full preprocessing pipeline.

    Streams documents, tokenizes into a single token stream with EOS
    separators (avoiding a separate documents list), then chunks and splits.

    Returns a stats dict with total_tokens, num_documents,
    num_train_chunks, and num_val_chunks.
    """
    from datasets import Dataset, load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    eos_token_id = tokenizer.eos_token_id
    assert eos_token_id is not None, "Tokenizer must have an EOS token"

    print(f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}, eos_id={eos_token_id}")
    print(f"Streaming FineWeb-Edu (max_tokens={max_tokens})...")

    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu-score-2",
        split="train",
        streaming=True,
    )

    # Stream directly into a flat token stream (avoids accumulating a
    # separate documents list — one fewer full copy in memory).
    token_stream: list[int] = []
    total_tokens = 0
    num_documents = 0

    for example in ds:
        text = example["text"]
        if not text or not text.strip():
            continue

        tokens = tokenizer.encode(text, add_special_tokens=False)
        if not tokens:
            continue

        token_stream.extend(tokens)
        token_stream.append(eos_token_id)
        total_tokens += len(tokens)
        num_documents += 1

        if num_documents % 10_000 == 0:
            print(f"  processed {num_documents:,} documents ({total_tokens:,} tokens)")

        if max_tokens is not None and total_tokens >= max_tokens:
            print(f"  reached max_tokens={max_tokens:,}, stopping")
            break

    print(f"Tokenization complete: {num_documents:,} documents, {total_tokens:,} tokens")

    # Chunk the flat stream directly (same logic as concatenate_and_chunk,
    # but we already have the stream built inline).
    chunks = []
    for start in range(0, len(token_stream) - seq_len + 1, seq_len):
        chunks.append(token_stream[start : start + seq_len])
    print(f"Chunked into {len(chunks):,} sequences of {seq_len} tokens")

    train_chunks, val_chunks = split_chunks(chunks, val_fraction=val_fraction)
    print(f"Split: {len(train_chunks):,} train, {len(val_chunks):,} val")

    os.makedirs(output_dir, exist_ok=True)

    train_ds = Dataset.from_dict({"input_ids": train_chunks})
    train_ds.save_to_disk(os.path.join(output_dir, "train"))
    print(f"Saved train split to {output_dir}/train/")

    if val_chunks:
        val_ds = Dataset.from_dict({"input_ids": val_chunks})
        val_ds.save_to_disk(os.path.join(output_dir, "val"))
        print(f"Saved val split to {output_dir}/val/")

    stats = {
        "total_tokens": total_tokens,
        "num_documents": num_documents,
        "num_train_chunks": len(train_chunks),
        "num_val_chunks": len(val_chunks),
    }
    print(f"Done! Stats: {stats}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess FineWeb-Edu into tokenized training data",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output directory for train/ and val/ splits",
    )
    parser.add_argument(
        "--seq-len", type=int, default=2048,
        help="Tokens per training sequence (default: 2048)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=None,
        help="Stop after this many tokens (default: process all)",
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.01,
        help="Fraction of chunks for validation (default: 0.01)",
    )
    args = parser.parse_args()

    run_prepare_data(
        output_dir=args.output,
        seq_len=args.seq_len,
        max_tokens=args.max_tokens,
        val_fraction=args.val_fraction,
    )


if __name__ == "__main__":
    main()
