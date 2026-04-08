# Phase 1a: Data Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a preprocessing pipeline that tokenizes FineWeb-Edu into fixed-length training/validation datasets, and add a validation loss loop to the training script.

**Architecture:** A standalone preprocessing script streams FineWeb-Edu from HuggingFace, tokenizes with Mistral's tokenizer, concatenates documents with EOS separators, chunks into fixed-length sequences, splits train/val, and saves as HuggingFace Arrow datasets. The existing training loop gets a `--val-data` arg that runs periodic validation loss computation.

**Tech Stack:** Python (PyTorch, HuggingFace datasets, HuggingFace transformers, argparse)

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `training/ct87/prepare_data.py` | Create | Preprocessing script: stream, tokenize, chunk, split, save |
| `training/tests/test_prepare_data.py` | Create | Unit tests for chunking, splitting, and end-to-end smoke test |
| `training/ct87/train.py` | Modify | Add `--val-data` arg + validation loss loop |
| `training/tests/test_train.py` | Modify | Add tests for validation loss function |
| `training/pyproject.toml` | Modify | Add `transformers>=4.38` dependency, register `network` pytest marker |
| `.gitignore` | Modify | Add `data/` directory |

---

### Task 1: Dependencies and Gitignore

**Files:**
- Modify: `training/pyproject.toml`
- Modify: `.gitignore`

**Context:** The preprocessing script needs the `transformers` library to load the Mistral tokenizer. The `data/` directory at the repo root will hold large tokenized datasets that must never be committed.

- [ ] **Step 1: Add `transformers` to pyproject.toml**

In `training/pyproject.toml`, add `transformers>=4.38` to the `dependencies` list:

```toml
[project]
name = "ct87"
version = "0.1.0"
description = "PyTorch training scaffold for the ct87 custom model"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.2",
    "safetensors>=0.4",
    "datasets>=2.16",
    "gguf>=0.6",
    "transformers>=4.38",
]
```

- [ ] **Step 2: Add `data/` to .gitignore**

Append to the end of `.gitignore`:

```
# Tokenized training datasets (large binary files)
data/
```

- [ ] **Step 3: Install the updated dependencies**

```bash
cd training
pip install -e '.[dev]' --break-system-packages
```

Expected: installs successfully, `transformers` now available.

- [ ] **Step 4: Verify the tokenizer loads**

```bash
python3 -c "from transformers import AutoTokenizer; t = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1'); print(f'vocab={t.vocab_size}, eos={t.eos_token_id}')"
```

Expected output: `vocab=32000, eos=2`

This downloads and caches the tokenizer files (~1MB) on first run. Subsequent runs use the cache.

- [ ] **Step 5: Commit**

```bash
git add training/pyproject.toml .gitignore
git commit -m "chore: add transformers dep and gitignore data/ for Phase 1a"
```

---

### Task 2: Preprocessing Script — Core Logic + Unit Tests

**Files:**
- Create: `training/ct87/prepare_data.py`
- Create: `training/tests/test_prepare_data.py`

**Context:** The preprocessing script has two separable concerns: (1) the pure chunking/splitting logic, and (2) the I/O layer (streaming from HuggingFace, tokenizing, saving). This task builds the core logic functions and their unit tests. Task 3 wires up the CLI and I/O.

The existing `make_hf_dataloader()` in `training/ct87/train.py` (lines 36-61) loads a HuggingFace Arrow dataset from disk via `load_from_disk()` and expects each example to have an `input_ids` field (list of ints). The preprocessing script must produce exactly this format.

- [ ] **Step 1: Write failing tests for `concatenate_and_chunk`**

Create `training/tests/test_prepare_data.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd training
python3 -m pytest tests/test_prepare_data.py -v
```

Expected: FAIL -- `ModuleNotFoundError: No module named 'ct87.prepare_data'`

- [ ] **Step 3: Implement `concatenate_and_chunk` and `split_chunks`**

Create `training/ct87/prepare_data.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd training
python3 -m pytest tests/test_prepare_data.py -v
```

Expected: all 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add training/ct87/prepare_data.py training/tests/test_prepare_data.py
git commit -m "feat: add concatenate-and-chunk + split logic for Phase 1a

Core preprocessing functions with full unit test coverage.
TDD: tests written first, then implementation."
```

---

### Task 3: Preprocessing Script — CLI and I/O Layer

**Files:**
- Modify: `training/ct87/prepare_data.py`
- Modify: `training/tests/test_prepare_data.py`
- Modify: `training/pyproject.toml` (pytest marker registration)

**Context:** This task adds the HuggingFace streaming, tokenization, and save-to-disk layers on top of the core logic from Task 2. It also adds the CLI entry point (`python3 -m ct87.prepare_data`) and a smoke test that runs the full pipeline on a tiny slice of real data.

The Mistral tokenizer is loaded via `AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")`. FineWeb-Edu is streamed via `load_dataset("HuggingFaceFW/fineweb-edu-score-2", split="train", streaming=True)`. Each example has a `"text"` field containing the document text.

The output is saved via HuggingFace `datasets.Dataset.from_dict()` + `save_to_disk()`, producing Arrow files that `load_from_disk()` in the training loop reads.

- [ ] **Step 1: Register the `network` pytest marker**

In `training/pyproject.toml`, update the pytest section:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "network: tests that require network access",
]
```

- [ ] **Step 2: Write the end-to-end smoke test**

Append to `training/tests/test_prepare_data.py`:

```python
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
```

- [ ] **Step 3: Run the smoke test to verify it fails**

```bash
cd training
python3 -m pytest tests/test_prepare_data.py::TestEndToEnd -v -m network
```

Expected: FAIL -- `ImportError: cannot import name 'run_prepare_data'`

- [ ] **Step 4: Implement `run_prepare_data` and CLI entry point**

Replace the full contents of `training/ct87/prepare_data.py` with:

```python
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


def run_prepare_data(
    output_dir: str,
    seq_len: int = 2048,
    max_tokens: int | None = None,
    val_fraction: float = 0.01,
) -> dict:
    """Run the full preprocessing pipeline.

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

    documents: list[list[int]] = []
    total_tokens = 0
    num_documents = 0

    for example in ds:
        text = example["text"]
        if not text or not text.strip():
            continue

        tokens = tokenizer.encode(text, add_special_tokens=False)
        if not tokens:
            continue

        documents.append(tokens)
        total_tokens += len(tokens)
        num_documents += 1

        if num_documents % 10_000 == 0:
            print(f"  processed {num_documents:,} documents ({total_tokens:,} tokens)")

        if max_tokens is not None and total_tokens >= max_tokens:
            print(f"  reached max_tokens={max_tokens:,}, stopping")
            break

    print(f"Tokenization complete: {num_documents:,} documents, {total_tokens:,} tokens")

    chunks = concatenate_and_chunk(documents, seq_len=seq_len, eos_token_id=eos_token_id)
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
```

- [ ] **Step 5: Run unit tests (no network) to verify core logic still passes**

```bash
cd training
python3 -m pytest tests/test_prepare_data.py -v -m "not network"
```

Expected: all 9 unit tests from Task 2 still PASS.

- [ ] **Step 6: Run the smoke test (requires network)**

```bash
cd training
python3 -m pytest tests/test_prepare_data.py::TestEndToEnd -v -m network
```

Expected: both end-to-end tests PASS. The first run downloads the Mistral tokenizer (~1MB) and streams a few thousand tokens from FineWeb-Edu.

- [ ] **Step 7: Run the full test suite to verify no regressions**

```bash
cd training
python3 -m pytest tests/ -v -m "not network"
```

Expected: all existing tests + new unit tests pass.

- [ ] **Step 8: Commit**

```bash
git add training/ct87/prepare_data.py training/tests/test_prepare_data.py training/pyproject.toml
git commit -m "feat: add FineWeb-Edu preprocessing pipeline for Phase 1a

Streams from HuggingFace, tokenizes with Mistral v0.1, concatenates
with EOS separators, chunks to seq_len, splits train/val, saves as
HuggingFace Arrow datasets compatible with make_hf_dataloader()."
```

---

### Task 4: Validation Loss Loop in Training Script

**Files:**
- Modify: `training/ct87/train.py`
- Modify: `training/tests/test_train.py`

**Context:** The training script (`training/ct87/train.py`) currently only prints training loss every 10 steps. This task adds a `--val-data` argument and a validation loss function that runs at checkpoint boundaries (`--save-every` steps). The validation loop loads data via the same `make_hf_dataloader()` used for training data, iterates over a fixed number of batches (10), and prints the average loss.

The existing training loop structure (lines 153-177 in `train.py`):
```python
for step in range(args.steps):
    # lr schedule, forward, loss, backward, step
    if step % 10 == 0:
        print(f"step={step:5d}  loss={loss.item():.4f}  lr={current_lr:.6f}")
    if args.save_every > 0 and step > 0 and step % args.save_every == 0:
        save_checkpoint(...)
```

- [ ] **Step 1: Write the failing test for validation loss**

Append to `training/tests/test_train.py`:

```python
class TestValidation:
    def test_returns_finite_float(self):
        """compute_validation_loss returns a finite float loss value."""
        from ct87.train import compute_validation_loss, make_synthetic_dataloader

        torch.manual_seed(42)
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        device = torch.device("cpu")

        val_loader = make_synthetic_dataloader(cfg.vocab_size, 16, batch_size=2, seed=99)
        val_loss = compute_validation_loss(model, val_loader, cfg.vocab_size, device, num_batches=3)

        assert isinstance(val_loss, float)
        assert val_loss > 0.0
        assert not torch.isnan(torch.tensor(val_loss))
        assert not torch.isinf(torch.tensor(val_loss))

    def test_no_grad_change(self):
        """Validation does not modify model weights."""
        from ct87.train import compute_validation_loss, make_synthetic_dataloader

        torch.manual_seed(42)
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        device = torch.device("cpu")

        params_before = {n: p.clone() for n, p in model.named_parameters()}

        val_loader = make_synthetic_dataloader(cfg.vocab_size, 16, batch_size=2, seed=99)
        compute_validation_loss(model, val_loader, cfg.vocab_size, device, num_batches=3)

        for name, param in model.named_parameters():
            assert torch.equal(param, params_before[name]), f"param {name} changed during validation"
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd training
python3 -m pytest tests/test_train.py::TestValidation -v
```

Expected: FAIL -- `ImportError: cannot import name 'compute_validation_loss'`

- [ ] **Step 3: Implement `compute_validation_loss` in train.py**

Add this function to `training/ct87/train.py` after the `set_lr` function (after line 99):

```python
def compute_validation_loss(
    model: HarmonyModel,
    val_loader: Iterator[torch.Tensor],
    vocab_size: int,
    device: torch.device,
    num_batches: int = 10,
) -> float:
    """Run validation and return average cross-entropy loss."""
    was_training = model.training
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(num_batches):
            batch = next(val_loader).to(device)
            input_ids = batch[:, :-1]
            targets = batch[:, 1:]
            logits = model(input_ids)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
            total_loss += loss.item()
    if was_training:
        model.train()
    return total_loss / num_batches
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd training
python3 -m pytest tests/test_train.py::TestValidation -v
```

Expected: both tests PASS.

- [ ] **Step 5: Add `--val-data` arg and wire validation into the training loop**

In `training/ct87/train.py`, make these changes:

Add the `--val-data` argument to the argument parser, after the existing `--data` argument (after line 117):

```python
    parser.add_argument("--val-data", type=str, default=None, help="Path to validation HF dataset (optional)")
```

After the training dataloader is created (after line 151), add the validation dataloader setup:

```python
    val_loader = None
    if args.val_data is not None:
        val_loader = make_hf_dataloader(args.val_data, seq_len, args.batch_size, args.seed + 1)
        print(f"Validation data loaded from {args.val_data}")
```

Inside the training loop, after the checkpoint save block (after line 174), add the validation step:

```python
        if val_loader is not None and args.save_every > 0 and step > 0 and step % args.save_every == 0:
            val_loss = compute_validation_loss(model, val_loader, config.vocab_size, device)
            print(f"  -> val_loss={val_loss:.4f}")
```

- [ ] **Step 6: Run the full test suite to verify no regressions**

```bash
cd training
python3 -m pytest tests/ -v -m "not network"
```

Expected: all tests pass, including the two new validation tests and all existing tests.

- [ ] **Step 7: Commit**

```bash
git add training/ct87/train.py training/tests/test_train.py
git commit -m "feat: add validation loss to training loop for Phase 1a

New --val-data arg enables periodic validation loss measurement at
checkpoint boundaries. compute_validation_loss() runs 10 batches
with no_grad and reports average cross-entropy loss."
```

---

### Task 5: Full Integration Verification

**Files:**
- No new files -- this task runs the existing code end-to-end.

**Context:** This task verifies that the complete pipeline works: preprocess data, then train with validation. It is a manual verification step, not a new test file. All automated tests were written in Tasks 2-4.

- [ ] **Step 1: Run the full test suite (non-network)**

```bash
cd training
python3 -m pytest tests/ -v -m "not network"
```

Expected: all tests pass.

- [ ] **Step 2: Run the network tests**

```bash
cd training
python3 -m pytest tests/test_prepare_data.py -v -m network
```

Expected: end-to-end smoke tests pass.

- [ ] **Step 3: Run a real preprocessing on a tiny slice**

```bash
cd training
python3 -m ct87.prepare_data \
    --output ../data/fineweb-edu-test \
    --seq-len 128 \
    --max-tokens 50000 \
    --val-fraction 0.1
```

Expected output (approximate):
```
Tokenizer loaded: vocab_size=32000, eos_id=2
Streaming FineWeb-Edu (max_tokens=50000)...
  reached max_tokens=50000, stopping
Tokenization complete: NN documents, NNNNN tokens
Chunked into NNN sequences of 128 tokens
Split: NNN train, NN val
Saved train split to ../data/fineweb-edu-test/train/
Saved val split to ../data/fineweb-edu-test/val/
Done! Stats: {...}
```

- [ ] **Step 4: Train on the preprocessed data with validation**

```bash
cd training
python3 -m ct87.train \
    --config tiny \
    --data ../data/fineweb-edu-test/train \
    --val-data ../data/fineweb-edu-test/val \
    --seq-len 128 \
    --batch-size 4 \
    --steps 100 \
    --warmup 10 \
    --save-every 50
```

Expected: training runs, loss decreases over steps, validation loss is printed at step 50, training completes at step 100 with final checkpoint.

- [ ] **Step 5: Clean up test data**

```bash
rm -rf data/fineweb-edu-test
```

- [ ] **Step 6: Run the existing Rust test suite to confirm nothing is broken**

```bash
cargo test -p harmony-inference
```

Expected: all 131 Rust tests pass (no Python changes affect Rust).

- [ ] **Step 7: Commit (no-op -- all code is already committed)**

No new files to commit. This task is verification only.
