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


def chunk_stream(stream: list[int], seq_len: int) -> list[list[int]]:
    """Slice a token stream into non-overlapping fixed-length chunks.

    The final partial chunk (if any) is discarded.
    """
    chunks = []
    for start in range(0, len(stream) - seq_len + 1, seq_len):
        chunks.append(stream[start : start + seq_len])
    return chunks


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

    return chunk_stream(stream, seq_len)


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

    Streams documents, tokenizes into a flat typed buffer with EOS
    separators, then chunks and splits.

    Returns a stats dict with content_tokens, stream_tokens,
    num_documents, num_train_chunks, and num_val_chunks.
    """
    # Fail fast on nonsense parameters — these would otherwise surface hours
    # later after tokenization finishes, burning the entire CPU+network pass.
    if not isinstance(seq_len, int) or seq_len <= 0:
        raise ValueError(f"seq_len must be a positive integer, got {seq_len!r}")
    if not 0.0 <= float(val_fraction) < 1.0:
        raise ValueError(
            f"val_fraction must be in [0.0, 1.0), got {val_fraction!r}"
        )

    import array
    import ctypes
    import gc
    import json

    import numpy as np
    import pyarrow as pa
    from datasets import load_dataset
    from transformers import AutoTokenizer

    # Two memory pools accumulate unbounded during streaming tokenization and
    # must be drained periodically or RSS grows ~40 B/token and OOMs a 46 GB
    # host around 1.4 B tokens:
    #   - glibc malloc holds freed heap segments in arenas after Python frees
    #     the per-document tokenizer allocations (transient list + PyLongs,
    #     ~60 KB/doc). malloc_trim(0) forces return to the OS.
    #   - pyarrow has its own memory pool (used by HF Datasets streaming) that
    #     grows to ~1 GB; release_unused() drops its idle chunks.
    # Empirically, calling both every 10k docs keeps RSS flat at ~2.6 GB for
    # the 100M-token dryrun instead of climbing past 7 GB.
    #
    # malloc_trim is glibc-specific — on macOS/musl the CDLL load fails and
    # we fall back to a no-op. gc + pyarrow release still handle most of the
    # leak on those platforms; only the tokenizer-churn arena fragmentation
    # stays (the 40 B/token regime). If a non-glibc environment ever becomes
    # a real target, revisit with jemalloc_tls/mallctl or similar.
    try:
        _libc = ctypes.CDLL("libc.so.6", use_errno=True)
        _libc.malloc_trim.argtypes = [ctypes.c_int]
        _libc.malloc_trim.restype = ctypes.c_int
        def _malloc_trim() -> None:
            _libc.malloc_trim(0)
    except (OSError, AttributeError):
        def _malloc_trim() -> None:
            pass

    # pa is imported unconditionally above — the pool is always available.
    _pa_pool = pa.default_memory_pool()

    def _release_unused_heap() -> None:
        gc.collect()
        _pa_pool.release_unused()
        _malloc_trim()

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    eos_token_id = tokenizer.eos_token_id
    assert eos_token_id is not None, "Tokenizer must have an EOS token"
    # uint16 ('H') holds 0..65535. Mistral's vocab is 32000 and all known EOS
    # ids fit — OverflowError here would be a loud fail if a larger vocab were
    # ever swapped in, rather than silent truncation. Widen to 'I' (uint32) if
    # you ever wire up a >65535 vocab.
    assert tokenizer.vocab_size <= 65535, (
        f"Tokenizer vocab_size={tokenizer.vocab_size} exceeds uint16 range; "
        "widen the token_stream typecode from 'H' to 'I'."
    )
    assert 0 <= eos_token_id <= 65535

    print(f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}, eos_id={eos_token_id}")
    print(f"Streaming FineWeb-Edu (max_tokens={max_tokens})...")

    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu-score-2",
        split="train",
        streaming=True,
    )

    # Typed 2-byte-per-token buffer. A Python list[int] costs ~36 B/token
    # (28 B PyObject + 8 B list slot), which OOMs a 46 GB host around
    # 1.4 B tokens. array.array('H') is 2 B/token, so 3 B tokens fits in ~6 GB.
    token_stream = array.array("H")
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
            print(f"  processed {num_documents:,} documents ({total_tokens:,} tokens)", flush=True)
            _release_unused_heap()

        if max_tokens is not None and total_tokens >= max_tokens:
            print(f"  reached max_tokens={max_tokens:,}, stopping")
            break

    stream_tokens = total_tokens + num_documents  # includes EOS separators
    print(f"Tokenization complete: {num_documents:,} documents, {total_tokens:,} content tokens ({stream_tokens:,} with EOS)")

    # Zero-copy view into the array.array buffer, then reshape to (n, seq_len)
    # after dropping the trailing partial chunk. Both ops are views — no alloc.
    stream_np = np.frombuffer(token_stream, dtype=np.uint16)
    n_complete = (len(stream_np) // seq_len) * seq_len
    chunks_np = stream_np[:n_complete].reshape(-1, seq_len)
    num_chunks = int(chunks_np.shape[0])
    print(f"Chunked into {num_chunks:,} sequences of {seq_len} tokens")

    # Train/val split via slicing (still views, no alloc).
    n_val = int(num_chunks * val_fraction)
    if n_val > 0:
        train_np = chunks_np[:-n_val]
        val_np = chunks_np[-n_val:]
    else:
        train_np = chunks_np
        val_np = None
    num_train = int(train_np.shape[0])
    num_val = int(val_np.shape[0]) if val_np is not None else 0
    print(f"Split: {num_train:,} train, {num_val:,} val")

    os.makedirs(output_dir, exist_ok=True)

    # Write arrow files directly via pyarrow in batches, bypassing HF
    # Dataset.from_dict which materializes the full dataset in memory during
    # conversion (peak >2x data size, OOMs a 46 GB host at 1.46 M rows).
    #
    # FixedSizeListArray has no per-row offsets — required above ~1.05 M rows
    # because ListArray's default pa.int32() offsets overflow when
    # n_rows * seq_len > 2^31 (≈ 2.15 B items). A 3 B-token prep produces
    # ~1.46 M rows at seq_len=2048 → 2.99 B items → overflow.
    #
    # The on-disk layout (one .arrow file + dataset_info.json + state.json)
    # matches what HF's save_to_disk produces for a single-shard dataset, so
    # load_from_disk() reads our output identically.
    arrow_schema = pa.schema([
        pa.field("input_ids", pa.list_(pa.int32(), list_size=seq_len))
    ])
    dataset_info = {
        "citation": "",
        "description": "",
        "features": {
            "input_ids": {
                "feature": {"dtype": "int32", "_type": "Value"},
                "length": seq_len,
                "_type": "Sequence",
            }
        },
        "homepage": "",
        "license": "",
    }
    rows_per_batch = 10_000  # ~80 MB int32 buffer per batch at seq_len=2048

    def _save_split(arr_uint16: np.ndarray, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        n_rows = int(arr_uint16.shape[0])
        arrow_path = os.path.join(path, "data-00000-of-00001.arrow")

        # HF Datasets uses the streaming IPC format (RecordBatchStreamWriter),
        # NOT the file format. The two wire formats look similar but are not
        # interchangeable — HF's load_from_disk expects stream format.
        with (
            pa.OSFile(arrow_path, "wb") as sink,
            pa.ipc.new_stream(sink, arrow_schema) as writer,
        ):
            for start in range(0, n_rows, rows_per_batch):
                end = min(start + rows_per_batch, n_rows)
                # Per-batch int32 cast: small, short-lived (~80 MB).
                batch_i32 = arr_uint16[start:end].astype(np.int32)
                flat = batch_i32.reshape(-1)
                values = pa.array(flat, type=pa.int32())
                list_arr = pa.FixedSizeListArray.from_arrays(values, list_size=seq_len)
                record_batch = pa.RecordBatch.from_arrays(
                    [list_arr], names=["input_ids"]
                )
                writer.write_batch(record_batch)

        with open(os.path.join(path, "dataset_info.json"), "w") as f:
            json.dump(dataset_info, f)

        # HF state.json — _fingerprint is a content-ish hash; load_from_disk
        # doesn't verify it, it's only used for caching downstream.
        state = {
            "_data_files": [{"filename": "data-00000-of-00001.arrow"}],
            "_fingerprint": f"{n_rows:016x}{seq_len:08x}",
            "_format_columns": None,
            "_format_kwargs": {},
            "_format_type": None,
            "_output_all_columns": False,
            "_split": None,
        }
        with open(os.path.join(path, "state.json"), "w") as f:
            json.dump(state, f)

    _save_split(train_np, os.path.join(output_dir, "train"))
    print(f"Saved train split to {output_dir}/train/")

    if val_np is not None and num_val > 0:
        _save_split(val_np, os.path.join(output_dir, "val"))
        print(f"Saved val split to {output_dir}/val/")

    stats = {
        "content_tokens": total_tokens,
        "stream_tokens": stream_tokens,
        "num_documents": num_documents,
        "num_train_chunks": num_train,
        "num_val_chunks": num_val,
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
