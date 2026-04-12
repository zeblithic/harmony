# Corpus-Based Engram Table Generation (ZEB-102, Path 1)

## Problem

The synthetic engram table contains random embeddings with no semantic structure. xxhash keys work (-0.49% val loss) only because they're deterministic — same n-gram always retrieves the same random-but-consistent vector. All projection-based keying schemes failed because nearby keys in a random table don't retrieve related information. The bottleneck is table content, not key quality.

## Solution

Build the engram table from corpus statistics: for each table entry, encode the distribution of tokens that follow the n-grams hashing to that entry. Compressed via Johnson-Lindenstrauss random projection from vocab-sized frequency vectors to engram_dim. Drop-in replacement — no training loop or architecture changes.

## Pipeline

1. Load tokenized training data (Arrow dataset from `prepare_data.py`)
2. Scan all tokens, extract bigrams `[t[i], t[i+1]]` and trigrams `[t[i], t[i+1], t[i+2]]`
3. Hash each n-gram to its table index using `_hash_ngram(tokens, seed) % total_entries` (same xxhash + seeds as training lookup)
4. Record the next token after each n-gram: for a bigram at positions [i, i+1] (attributed to position i+1), the next token is at i+2. For a trigram at [i, i+1, i+2] (attributed to i+2), next token is at i+3. This is the token the model is trying to predict at the engram injection point.
5. Build count matrix `[total_entries, vocab_size]` — `counts[table_idx, next_token] += 1`
6. Normalize each row to a probability distribution (L1 normalize)
7. Compress each row `[vocab_size]` -> `[engram_dim]` via a fixed seeded random projection matrix
8. Unit-normalize each row (L2)
9. Save as safetensors in the existing format

## Parameters

- `total_entries`: 10,000 (matches training runs)
- `engram_dim`: 128 (matches tiny config)
- `vocab_size`: 32,000 (Mistral tokenizer)
- `hash_seeds`: [42, 99, 137, 251] (4 heads)
- `projection_seed`: 0 (fixed, for reproducibility)

## Random projection

The projection matrix is `[vocab_size, engram_dim]` drawn from N(0, 1/sqrt(engram_dim)) with a fixed numpy seed. This is a Johnson-Lindenstrauss projection — preserves pairwise distances between the high-dimensional frequency vectors with high probability at 128 dimensions.

The matrix is regenerated from the seed (not saved separately). The projection is deterministic.

## Memory

Count matrix `[10000, 32000]` is ~1.2GB in float32. Projection matrix `[32000, 128]` is ~16MB. Both fit comfortably in memory.

## Hash collision behavior

With 10K entries and ~200M n-grams from a 100M-token corpus, each entry averages ~20K n-gram contributions. Each entry becomes a rich aggregate dominated by its most common n-grams' following-token distributions.

## Edge cases

- Table entries with zero counts (no n-grams hash there): zero vector, same as position-0 behavior in training.
- Entries where all n-grams are followed by the same token: the distribution is a one-hot, projected to a single direction. Unit normalization preserves this.

## Compatibility

Output format is identical to the current synthetic table:
- Same safetensors file with `engram.weight` tensor
- Same f16 dtype
- Same config TOML with shard_size, hash_seeds
- Training loop, eval harness, GGUF export unchanged — drop-in replacement

## CLI

```bash
# Corpus-based table (new)
python -m ct87.generate_engram_table \
    --corpus <path-to-tokenized-train-data> \
    --entries 10000 --dim 128 --output-dir engram_corpus

# Random table (existing behavior, unchanged)
python -m ct87.generate_engram_table \
    --entries 10000 --dim 128 --output-dir engram_random
```

`--corpus` triggers the new path. Without it, existing random generation is unchanged.

## Files to modify

1. **`training/ct87/generate_engram_table.py`** — add `generate_corpus_table()`, `--corpus` CLI flag
2. **`training/tests/test_engram.py`** (or new test file) — tests for corpus table generation

## Success criteria

- KRILE trains with corpus-based table + xxhash keys
- Compare val_loss to random-table baseline (-0.49%)
- If corpus table gives meaningfully better improvement (e.g., -1% or more), the engram approach validates for further investment
- If similar ~0.5%, the ceiling may be low at this scale regardless of table content
