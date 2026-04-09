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
    args = parser.parse_args()

    # Row index is packed as uint32 in SHA-256 seed — validate range.
    if args.entries > 2**32:
        parser.error(f"--entries {args.entries} exceeds uint32 max (2**32)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.entries} embeddings, dim={args.dim}...")
    table = generate_table(args.entries, args.dim)

    # Convert to f16 for safetensors (matches Engram table format)
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
    print(f"\nNext step:")
    print(f"  cargo run --manifest-path src-tauri/Cargo.toml -p harmony-ingest \\")
    print(f"    -- engram --config {config_path} --input {safetensors_path} \\")
    print(f"    --local-dir {output_dir}/store")


if __name__ == "__main__":
    main()
