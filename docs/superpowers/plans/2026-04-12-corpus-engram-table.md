# Corpus-Based Engram Table Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an engram table from next-token co-occurrence statistics in the training corpus, replacing the current random-embedding table with one that carries real predictive signal.

**Architecture:** Scan tokenized data to collect which tokens follow each n-gram, hash n-grams to table indices using the same xxhash function as training, build a [total_entries, vocab_size] count matrix, normalize to probabilities, compress to [total_entries, engram_dim] via Johnson-Lindenstrauss random projection, unit-normalize, and save as the same safetensors format.

**Tech Stack:** Python, NumPy, ct87.engram._hash_ngram (xxhash64)

**Spec:** `docs/superpowers/specs/2026-04-12-corpus-engram-table-design.md`

---

### Task 1: `generate_corpus_table()` function with tests

**Files:**
- Modify: `training/ct87/generate_engram_table.py`
- Test: `training/tests/test_generate_engram_table.py` (new)

The core function that scans tokenized data and builds the co-occurrence table.

- [ ] **Step 1: Write the failing tests**

Create `training/tests/test_generate_engram_table.py`:

```python
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
        # Minimal corpus: 3 chunks of 8 tokens
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
        # Very small corpus, large table -> many zero entries
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/zeblith/work/zeblithic/harmony/training && python3 -m pytest tests/test_generate_engram_table.py -v`
Expected: FAIL with `ImportError: cannot import name 'generate_corpus_table'`

- [ ] **Step 3: Write the implementation**

In `training/ct87/generate_engram_table.py`, add these imports at the top (after existing imports):

```python
from collections.abc import Sequence
```

Then add this function after `generate_table()` and before `write_ingest_config()`:

```python
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
    if hash_seeds is None:
        hash_seeds = list(DEFAULT_HASH_SEEDS)

    from ct87.engram import _hash_ngram

    # Build count matrix: [total_entries, vocab_size]
    counts = np.zeros((total_entries, vocab_size), dtype=np.float64)

    for chunk in chunks:
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

    # Normalize rows to probability distributions (L1)
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)  # avoid division by zero
    probs = counts / row_sums

    # Johnson-Lindenstrauss random projection: [vocab_size, embedding_dim]
    rng = np.random.RandomState(projection_seed)
    proj_matrix = rng.randn(vocab_size, embedding_dim).astype(np.float32)
    proj_matrix /= np.sqrt(embedding_dim)

    # Project: [total_entries, vocab_size] @ [vocab_size, embedding_dim] -> [total_entries, embedding_dim]
    table = (probs @ proj_matrix).astype(np.float32)

    # Unit-normalize non-zero rows (L2)
    norms = np.linalg.norm(table, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    # Only normalize rows that had actual counts
    has_counts = (row_sums.squeeze() > 1e-10)
    table[has_counts] = table[has_counts] / norms[has_counts]
    table[~has_counts] = 0.0

    return table
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_generate_engram_table.py -v`
Expected: 6 PASSED

- [ ] **Step 5: Commit**

```bash
git add training/ct87/generate_engram_table.py training/tests/test_generate_engram_table.py
git commit -m "feat: generate_corpus_table() for next-token co-occurrence engram table"
```

---

### Task 2: Wire `--corpus` CLI flag into main()

**Files:**
- Modify: `training/ct87/generate_engram_table.py`
- Test: `training/tests/test_generate_engram_table.py`

Add the `--corpus` CLI flag that switches between random and corpus-based generation.

- [ ] **Step 1: Write the failing test**

Add to `training/tests/test_generate_engram_table.py`:

```python
import tempfile
from pathlib import Path
from safetensors.numpy import load_file


class TestCorpusCLIIntegration:
    def test_corpus_flag_produces_table(self):
        """--corpus should produce a safetensors file with the correct shape."""
        from datasets import Dataset

        # Create minimal tokenized dataset
        chunks = [[1, 2, 3, 4, 5, 6, 7, 8]] * 20
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            ds = Dataset.from_dict({"input_ids": chunks})
            ds.save_to_disk(str(data_dir))

            out_dir = Path(tmpdir) / "output"
            from ct87.generate_engram_table import generate_and_save_corpus_table

            generate_and_save_corpus_table(
                data_path=str(data_dir),
                total_entries=100,
                embedding_dim=16,
                output_dir=str(out_dir),
                vocab_size=32,
            )

            st_path = out_dir / "engram_table.safetensors"
            assert st_path.exists()
            tensors = load_file(str(st_path))
            assert "engram.weight" in tensors
            assert tensors["engram.weight"].shape == (100, 16)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_generate_engram_table.py::TestCorpusCLIIntegration -v`
Expected: FAIL with `ImportError: cannot import name 'generate_and_save_corpus_table'`

- [ ] **Step 3: Write the implementation**

In `training/ct87/generate_engram_table.py`, add this function after `generate_corpus_table()`:

```python
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
```

Then update `main()` to add the `--corpus` flag. Add the argument after `--shard-size`:

```python
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
```

Then replace the generation + save block in `main()` (everything from `print(f"Generating...")` through `print(f"Wrote {config_path}")`) with:

```python
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

        config_path = write_ingest_config(output_dir, shard_size=args.shard_size)
        print(f"Wrote {config_path}")
```

Keep the stats print block after both branches.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_generate_engram_table.py -v`
Expected: 7 PASSED (6 from Task 1 + 1 new)

- [ ] **Step 5: Verify existing tests still pass**

Run: `python3 -m pytest tests/test_latent_projection.py tests/test_eval.py tests/test_engram.py -q`
Expected: All pass, no regressions

- [ ] **Step 6: Commit**

```bash
git add training/ct87/generate_engram_table.py training/tests/test_generate_engram_table.py
git commit -m "feat: --corpus CLI flag for corpus-based engram table generation"
```

---

### Task 3: Push and create PR

**Files:** None (git operations only)

- [ ] **Step 1: Push and create PR**

```bash
git push -u origin HEAD
gh pr create --title "feat: corpus-based engram table from next-token co-occurrence (ZEB-102)" \
  --body "$(cat <<'EOF'
## Summary

- Add `generate_corpus_table()` — builds engram table from next-token co-occurrence statistics
- Add `--corpus <path>` CLI flag to `generate_engram_table.py`
- Compress [vocab_size] frequency distributions to [engram_dim] via Johnson-Lindenstrauss random projection
- Drop-in replacement: same safetensors format, same config TOML, no training changes needed
- 7 new tests

### How it works

1. Scan tokenized corpus for all bigrams/trigrams
2. Hash each n-gram to table index (same xxhash64 + seeds as training)
3. Count which tokens follow each n-gram → `[10000, 32000]` count matrix
4. Normalize to probability distributions
5. Random projection `[32000] → [128]`, unit normalize
6. Save as `engram_table.safetensors`

Each table entry now encodes "what typically comes after the n-grams that hash here" — real predictive signal instead of random vectors.

### Context

All 6 projection-based keying experiments showed the model ignoring engram. Root cause: the synthetic table has no semantic structure. This PR tests whether giving the table real content improves the xxhash-engram result beyond -0.49%.

### Usage

```bash
# Generate corpus-based table
python -m ct87.generate_engram_table \
    --corpus ../data/fineweb-edu-poc/train \
    --entries 10000 --dim 128 --output-dir engram_corpus

# Train with it (same as before — drop-in replacement)
python -m ct87.train --config tiny \
    --data ../data/fineweb-edu-poc/train \
    --engram-table engram_corpus/engram_table.safetensors \
    --steps 10000

# Compare
python -m ct87.eval --compare ...
```

## Test plan

- [ ] All new + existing tests pass
- [ ] KRILE generates corpus table and trains with it
- [ ] Compare: corpus table + xxhash vs random table + xxhash vs baseline

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```
