"""Generate a test Engram table as safetensors for the harmony-ingest pipeline.

Produces a [total_entries, embedding_dim] f16 safetensors file with
deterministic, position-dependent embeddings derived from row hashing.
Each row gets a unique vector computed by hashing the row index through
multiple phases, giving the model learnable correlations between N-gram
hash positions and embedding vectors.

Also writes the companion ingest config TOML.

Usage:
    python -m ct87.generate_engram_table [--entries 10000] [--dim 128] [--output-dir engram_test]
"""

from __future__ import annotations

import argparse
import hashlib
import struct
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file


# Default config matching tiny model (engram_dim=128)
DEFAULT_ENTRIES = 10_000
DEFAULT_DIM = 128
DEFAULT_SHARD_SIZE = 200
DEFAULT_HASH_SEEDS = [42, 99, 137, 251]
DEFAULT_TENSOR_NAME = "engram.weight"


def generate_embedding(row: int, dim: int) -> np.ndarray:
    """Generate a deterministic f32 embedding for a table row.

    Uses SHA-256 of the row index to seed a deterministic PRNG sequence.
    The vector is normalized to unit length so all embeddings live on the
    hypersphere — this prevents magnitude bias and matches how the gated
    residual's key/value projections expect to operate.
    """
    # SHA-256 gives 32 bytes of entropy per hash — chain for longer dims
    components = []
    block = 0
    while len(components) < dim:
        digest = hashlib.sha256(
            struct.pack("<II", row, block)
        ).digest()
        # Unpack 8 floats from 32 bytes (interpret as 8 little-endian i32,
        # then map to [-1, 1] range)
        ints = struct.unpack("<8i", digest)
        for v in ints:
            if len(components) < dim:
                components.append(v / (2**31))
        block += 1

    vec = np.array(components[:dim], dtype=np.float32)
    # Normalize to unit length
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def generate_table(total_entries: int, embedding_dim: int) -> np.ndarray:
    """Generate the full embedding table as [total_entries, embedding_dim] f32."""
    table = np.zeros((total_entries, embedding_dim), dtype=np.float32)
    for i in range(total_entries):
        table[i] = generate_embedding(i, embedding_dim)
    return table


def generate_corpus_table(
    chunks: Sequence[list[int]],
    total_entries: int = DEFAULT_ENTRIES,
    embedding_dim: int = DEFAULT_DIM,
    vocab_size: int = 32000,
    hash_seeds: list[int] | None = None,
    projection_seed: int = 0,
) -> np.ndarray:
    """Generate engram table from next-token co-occurrence statistics.

    Scans tokenized chunks for bigrams and trigrams, hashes each to a
    table index (same xxhash64 as training lookup), records the token
    following each n-gram, builds per-entry frequency distributions,
    and compresses via Johnson-Lindenstrauss random projection.

    Args:
        chunks: List of tokenized sequences (each a list of int token IDs).
        total_entries: Number of table entries.
        embedding_dim: Output embedding dimension.
        vocab_size: Tokenizer vocabulary size (for count matrix width).
        hash_seeds: xxhash64 seeds for n-gram hashing (default: [42,99,137,251]).
        projection_seed: Seed for the random projection matrix (default: 0).

    Returns:
        [total_entries, embedding_dim] float32 array, unit-normalized rows.
        Rows with no n-gram hits are zero vectors.
    """
    if total_entries <= 0:
        raise ValueError("total_entries must be > 0")
    if hash_seeds is None:
        hash_seeds = list(DEFAULT_HASH_SEEDS)

    from ct87.engram import _hash_ngram

    # Build count matrix: [total_entries, vocab_size]
    # float32 keeps peak memory ~1.2GB at 10K×32K (vs 2.56GB with float64)
    counts = np.zeros((total_entries, vocab_size), dtype=np.float32)

    for chunk in chunks:
        # Validate token IDs before scanning
        if chunk:
            max_token = max(chunk)
            min_token = min(chunk)
            if min_token < 0 or max_token >= vocab_size:
                raise ValueError(
                    f"Token IDs out of range for vocab_size={vocab_size}: "
                    f"min={min_token}, max={max_token}"
                )
        seq_len = len(chunk)

        # Bigrams: [t[i], t[i+1]] attributed to position i+1, next token at i+2
        for i in range(seq_len - 2):
            bigram = [chunk[i], chunk[i + 1]]
            next_token = chunk[i + 2]
            for seed in hash_seeds:
                idx = _hash_ngram(bigram, seed) % total_entries
                counts[idx, next_token] += 1

        # Trigrams: [t[i], t[i+1], t[i+2]] attributed to position i+2, next token at i+3
        for i in range(seq_len - 3):
            trigram = [chunk[i], chunk[i + 1], chunk[i + 2]]
            next_token = chunk[i + 3]
            for seed in hash_seeds:
                idx = _hash_ngram(trigram, seed) % total_entries
                counts[idx, next_token] += 1

    # Identify rows with actual counts
    row_sums = counts.sum(axis=1)
    has_counts = row_sums > 0

    # Johnson-Lindenstrauss random projection: [vocab_size, embedding_dim]
    rng = np.random.RandomState(projection_seed)
    proj_matrix = rng.randn(vocab_size, embedding_dim).astype(np.float32)
    proj_matrix /= np.sqrt(embedding_dim)

    # Only project rows that have data (avoids overflow warnings on empty rows)
    table = np.zeros((total_entries, embedding_dim), dtype=np.float32)
    if has_counts.any():
        # Normalize active rows to probability distributions (L1)
        active_counts = counts[has_counts]
        active_sums = row_sums[has_counts, np.newaxis]
        probs = active_counts / active_sums

        # Project: [active, vocab_size] @ [vocab_size, embedding_dim]
        # Suppress benign overflow warnings from sparse float32 matmul;
        # any NaN/inf rows are clamped by the L2 normalization below.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            projected = probs @ proj_matrix

        # Clamp any NaN/inf from float32 overflow before normalizing
        np.nan_to_num(projected, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # Unit-normalize (L2)
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        table[has_counts] = projected / norms

    return table


def generate_and_save_corpus_table(
    data_path: str,
    total_entries: int = DEFAULT_ENTRIES,
    embedding_dim: int = DEFAULT_DIM,
    output_dir: str = "engram_corpus",
    shard_size: int = DEFAULT_SHARD_SIZE,
    vocab_size: int = 32000,
    hash_seeds: list[int] | None = None,
    projection_seed: int = 0,
) -> Path:
    """Load tokenized data, generate corpus-based table, and save.

    Args:
        data_path: Path to HuggingFace Arrow dataset with 'input_ids' column.
        total_entries: Number of table entries.
        embedding_dim: Output embedding dimension.
        output_dir: Directory for output files.
        shard_size: Embeddings per shard for ingest config.
        vocab_size: Tokenizer vocabulary size.
        hash_seeds: xxhash64 seeds (default: [42,99,137,251]).
        projection_seed: Seed for random projection (default: 0).

    Returns:
        Path to the generated safetensors file.
    """
    from datasets import load_from_disk

    ds = load_from_disk(data_path)
    if "input_ids" not in ds.column_names:
        raise ValueError(
            f"Dataset at {data_path} has no 'input_ids' column. "
            f"Available columns: {ds.column_names}. "
            f"Use ct87.prepare_data to create a tokenized dataset."
        )
    chunks = ds["input_ids"]

    print(f"Loaded {len(chunks)} chunks from {data_path}")
    print(f"Scanning for n-gram co-occurrences (entries={total_entries}, dim={embedding_dim})...")

    table = generate_corpus_table(
        chunks,
        total_entries=total_entries,
        embedding_dim=embedding_dim,
        vocab_size=vocab_size,
        hash_seeds=hash_seeds,
        projection_seed=projection_seed,
    )

    nonzero = np.count_nonzero(np.linalg.norm(table, axis=1))
    print(f"Table built: {nonzero}/{total_entries} entries have data")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    table_f16 = table.astype(np.float16)
    safetensors_path = out_path / "engram_table.safetensors"
    save_file({DEFAULT_TENSOR_NAME: table_f16}, str(safetensors_path))

    size_mb = safetensors_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {safetensors_path} ({size_mb:.2f} MB)")

    if hash_seeds is None:
        hash_seeds = list(DEFAULT_HASH_SEEDS)
    config_path = write_ingest_config(out_path, shard_size=shard_size, hash_seeds=hash_seeds)
    print(f"Wrote {config_path}")

    return safetensors_path


def write_ingest_config(
    output_dir: Path,
    shard_size: int = DEFAULT_SHARD_SIZE,
    hash_seeds: list[int] | None = None,
    tensor_name: str = DEFAULT_TENSOR_NAME,
) -> Path:
    """Write the ingest config TOML."""
    if hash_seeds is None:
        hash_seeds = list(DEFAULT_HASH_SEEDS)

    seeds_str = ", ".join(str(s) for s in hash_seeds)
    config_path = output_dir / "engram_config.toml"
    config_path.write_text(
        f'version = "v1"\n'
        f"shard_size = {shard_size}\n"
        f"hash_seeds = [{seeds_str}]\n"
        f'tensor = "{tensor_name}"\n'
    )
    return config_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate test Engram table for harmony-ingest"
    )
    parser.add_argument(
        "--entries", type=int, default=DEFAULT_ENTRIES,
        help=f"Total embedding entries (default: {DEFAULT_ENTRIES})",
    )
    parser.add_argument(
        "--dim", type=int, default=DEFAULT_DIM,
        help=f"Embedding dimension (default: {DEFAULT_DIM})",
    )
    parser.add_argument(
        "--output-dir", type=str, default="engram_test",
        help="Output directory (default: engram_test)",
    )
    parser.add_argument(
        "--shard-size", type=int, default=DEFAULT_SHARD_SIZE,
        help=f"Embeddings per shard (default: {DEFAULT_SHARD_SIZE})",
    )
    parser.add_argument(
        "--corpus", type=str, default=None,
        help="Path to tokenized HF dataset (enables corpus-based generation)",
    )
    parser.add_argument(
        "--vocab-size", type=int, default=32000,
        help="Tokenizer vocabulary size (default: 32000, Mistral v0.1)",
    )
    parser.add_argument(
        "--projection-seed", type=int, default=0,
        help="Seed for random projection matrix (default: 0)",
    )
    args = parser.parse_args()

    # Row index is packed as uint32 in SHA-256 seed — validate range.
    if args.entries > 2**32:
        parser.error(f"--entries {args.entries} exceeds uint32 max (2**32)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.corpus is not None:
        generate_and_save_corpus_table(
            data_path=args.corpus,
            total_entries=args.entries,
            embedding_dim=args.dim,
            output_dir=args.output_dir,
            shard_size=args.shard_size,
            vocab_size=args.vocab_size,
            projection_seed=args.projection_seed,
        )
    else:
        print(f"Generating {args.entries} random embeddings, dim={args.dim}...")
        table = generate_table(args.entries, args.dim)

        table_f16 = table.astype(np.float16)

        safetensors_path = output_dir / "engram_table.safetensors"
        save_file({DEFAULT_TENSOR_NAME: table_f16}, str(safetensors_path))

        size_mb = safetensors_path.stat().st_size / (1024 * 1024)
        print(f"Wrote {safetensors_path} ({size_mb:.2f} MB)")

        config_path = write_ingest_config(
            output_dir, shard_size=args.shard_size,
        )
        print(f"Wrote {config_path}")

    num_shards = (args.entries + args.shard_size - 1) // args.shard_size
    print(f"\nTable stats:")
    print(f"  Entries:      {args.entries:,}")
    print(f"  Dimension:    {args.dim}")
    print(f"  Shard size:   {args.shard_size}")
    print(f"  Num shards:   {num_shards}")
    print(f"  Num heads:    {len(DEFAULT_HASH_SEEDS)}")

    safetensors_path = output_dir / "engram_table.safetensors"
    config_path = output_dir / "engram_config.toml"
    print(f"\nNext step:")
    print(f"  cargo run --manifest-path src-tauri/Cargo.toml -p harmony-ingest \\")
    print(f"    -- engram --config {config_path} --input {safetensors_path} \\")
    print(f"    --local-dir {output_dir}/store")


if __name__ == "__main__":
    main()
