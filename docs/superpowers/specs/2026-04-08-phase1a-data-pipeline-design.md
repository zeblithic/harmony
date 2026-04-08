# Phase 1a: Data Pipeline Design

> First sub-project of Phase 1 (Actual Training). Builds the preprocessing
> pipeline that turns raw web text into tokenized, chunked training data the
> existing training loop can consume.

**Dependencies:** Phase 0f (training scaffold), Phase 0h (validation loop —
proves the pipeline is architecturally sound)

## Goal

Produce tokenized, fixed-length training and validation datasets from
FineWeb-Edu, using the Mistral v0.1 tokenizer, in a format the existing
`make_hf_dataloader()` can load directly. Add a validation loss assessment
loop to the training script so loss curves are observable from the first
real training run.

## Tokenizer

**Model:** `mistralai/Mistral-7B-v0.1` (tokenizer only — the `transformers`
library downloads ~1MB of tokenizer files, not the 7B model weights).

| Property | Value |
|---|---|
| Type | SentencePiece BPE |
| Vocabulary size | 32,000 (exact match for ct87 `vocab_size`) |
| BOS token | `<s>` (ID 1) |
| EOS token | `</s>` (ID 2) |

Loaded via `AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")`.
Called with `add_special_tokens=False` per document — EOS is inserted
manually between documents during concatenation.

## Dataset

**Source:** `HuggingFaceFW/fineweb-edu-score-2` — high-quality educational
web text curated by HuggingFace. Available via HuggingFace `datasets`
streaming, so no full download is required upfront.

**Two-stage plan:**

| Stage | Token count | Purpose | Time estimate (4090) |
|---|---|---|---|
| Proof-of-concept | ~100M tokens | Validate pipeline + architecture trains | Hours |
| Full training | ~5-10B tokens | Chinchilla-optimal for 0.5B model | Days |

Phase 1a builds the pipeline. Phase 1c runs the training.

## Preprocessing Pipeline

New file: `training/ct87/prepare_data.py`

### Flow

1. Stream FineWeb-Edu from HuggingFace (no full download).
2. Tokenize each document: `tokenizer.encode(text, add_special_tokens=False)`.
3. Concatenate all token sequences with EOS (ID 2) inserted between documents.
4. Chunk the concatenated stream into fixed-length sequences of `seq_len`
   tokens. Discard the final partial chunk.
5. Split chunks into train (99%) and validation (1%) sets.
6. Save both splits as HuggingFace Arrow datasets on disk.
7. Print progress every 10K documents and final statistics.

### Concatenate-and-Chunk Strategy

Documents are glued together into one long token stream with EOS separators:

```
[doc1_tok1, doc1_tok2, ..., doc1_tokN, EOS, doc2_tok1, ..., doc2_tokM, EOS, ...]
```

This stream is sliced into fixed-length chunks of `seq_len` tokens. Every
token is used (no padding waste). EOS tokens teach the model document
boundaries naturally.

### CLI

```
python3 -m ct87.prepare_data \
  --output data/fineweb-edu-poc \
  --seq-len 2048 \
  --max-tokens 100_000_000 \
  --val-fraction 0.01
```

| Arg | Default | Description |
|---|---|---|
| `--output` | (required) | Output directory for train/ and val/ splits |
| `--seq-len` | 2048 | Tokens per training sequence |
| `--max-tokens` | None (all) | Stop after this many tokens (for proof-of-concept) |
| `--val-fraction` | 0.01 | Fraction of chunks reserved for validation |

### Output Format

```
data/
  fineweb-edu-poc/
    train/          # HuggingFace Arrow dataset, column: input_ids (List[int])
    val/            # Same format
```

Compatible with the existing `make_hf_dataloader()` in `train.py` — no
changes needed to the data loading path.

### Key Detail

The `data/` directory at the repo root is gitignored. Tokenized datasets are
large binary files that must never be committed. Each `prepare_data.py` run
produces a self-contained output directory, so multiple prepared datasets
can coexist side-by-side.

## Training Loop Updates

Minimal changes to `train.py`:

### Validation Loss

New `--val-data` CLI arg pointing to the validation split directory. At
every checkpoint boundary (`--save-every` steps), the training loop:

1. Switches to evaluation mode + `torch.no_grad()`.
2. Iterates over a fixed number of validation batches (10).
3. Computes average cross-entropy loss across those batches.
4. Prints `step {N}: train_loss={X:.4f}, val_loss={Y:.4f}`.
5. Switches back to `model.train()`.

This provides the validation loss curve needed to distinguish learning from
memorization. If `--val-data` is omitted, validation is skipped (backwards
compatible with existing usage).

### Validation Dataloader

Reuses the existing `make_hf_dataloader()` to load the validation split.
Same format, same code path, different directory.

### What Does NOT Change

- Optimizer (Muon + AdamW hybrid)
- LR schedule (WSD)
- Loss function (cross-entropy)
- Synthetic data path (still works for testing)
- Checkpoint save/load

## Dependencies

Add to `training/pyproject.toml`:

```toml
dependencies = [
    "torch>=2.2",
    "safetensors>=0.4",
    "datasets>=2.16",
    "gguf>=0.6",
    "transformers>=4.38",   # NEW: Mistral tokenizer
]
```

The `transformers` library is used only for tokenizer loading in the
preprocessing script. The training loop does not import it.

## Files Changed

| File | Action | ~Lines |
|---|---|---|
| `training/ct87/prepare_data.py` | Create | ~120 |
| `training/ct87/train.py` | Modify | +30 |
| `training/pyproject.toml` | Modify | +1 |
| `.gitignore` | Modify | +1 |
| `training/tests/test_prepare_data.py` | Create | ~80 |

## Testing

### Unit Tests (`training/tests/test_prepare_data.py`)

- **Chunking logic:** Given a list of token sequences and an EOS token,
  verify concatenate-and-chunk produces correct-length chunks with EOS at
  document boundaries.
- **Train/val split:** Given N chunks and a val_fraction, verify the split
  ratio is correct and there is no overlap between splits.
- **End-to-end smoke test:** Run `prepare_data` with `--max-tokens 10000`
  on a tiny FineWeb-Edu slice, verify the output loads via
  `make_hf_dataloader()` and produces batches with the correct shape.

### Existing Tests

All existing tests in `training/tests/` remain unchanged and must continue
to pass. The synthetic data path is unaffected.

## Scope Boundary

**In scope:**
- Preprocessing script (stream, tokenize, concatenate-and-chunk, split, save)
- Mistral v0.1 tokenizer integration (via `transformers`)
- `transformers` added to `pyproject.toml`
- Train/val split (99/1 default)
- Validation loss loop in `train.py`
- `data/` gitignored
- Tests for chunking, splitting, and end-to-end smoke test

**Out of scope:**
- Gradient accumulation / mixed precision (Phase 1b)
- Wandb / tensorboard logging (Phase 1b)
- Actually running a training session (Phase 1c)
- Training on the TARGET config (proof-of-concept uses TINY config)
- Multi-GPU / distributed training
- Engram, UQ head, Chronos integration
- Custom tokenizer training
- Data augmentation or filtering
