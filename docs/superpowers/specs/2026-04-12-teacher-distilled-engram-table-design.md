# Teacher-Distilled Engram Table Generation (ZEB-102, Step 1b)

## Problem

PR #213 builds engram tables from next-token co-occurrence statistics compressed via random Johnson-Lindenstrauss projection. The random projection discards semantic structure — nearby entries in the compressed space don't correspond to semantically related distributions. A pre-trained model's embedding matrix encodes semantic relationships that random projection cannot.

## Solution

Replace the random projection matrix with the teacher model's token embedding matrix, then apply PCA to reduce dimensionality. Each table entry becomes the probability-weighted centroid of what follows in the teacher's semantic space — semantically rich instead of randomly compressed.

## Pipeline

1. Load tokenized training data (same Arrow dataset as PR #213)
2. Load teacher model's embedding matrix: `[32000, 4096]` for Mistral 7B
3. Same corpus scan as `--corpus` mode: build `[10000, 32000]` next-token frequency counts
4. Normalize to probability distributions (L1)
5. Project through teacher embeddings: `probs @ embed_matrix` → `[10000, 4096]`
   - Each entry is the probability-weighted centroid of next-token embeddings
6. PCA: fit on non-zero rows, reduce `[10000, 4096]` → `[10000, 128]`
   - Captures the 128 most informative directions in teacher space for this corpus
7. Unit-normalize (L2)
8. Save as same safetensors format

## What changes vs PR #213

Only the projection step (step 5-6). The corpus scan, frequency counting, normalization, output format, and CLI structure are all identical. The `--teacher` flag is an optional modifier to `--corpus`:

- `--corpus` alone: random JL projection (existing PR #213 behavior)
- `--corpus --teacher <model>`: teacher embedding projection + PCA (this spec)

## CLI

```bash
python -m ct87.generate_engram_table \
    --corpus <path-to-tokenized-data> \
    --teacher mistralai/Mistral-7B-v0.1 \
    --entries 10000 --dim 128 --output-dir engram_teacher
```

`--teacher` requires `--corpus`. Without `--corpus`, `--teacher` is rejected.

## Embedding loading

Load only what we need from the teacher model:

```python
from transformers import AutoModel
model = AutoModel.from_pretrained(teacher_name, torch_dtype=torch.float16)
embed_matrix = model.get_input_embeddings().weight.detach().float().numpy()
del model  # free memory immediately
```

Peak memory: ~14GB briefly for Mistral 7B model load. The embedding matrix itself is `[32000, 4096]` float32 = ~500MB. KRILE's training machine handles this.

## PCA

`sklearn.decomposition.PCA(n_components=128)` fitted on non-zero projected rows. Trivial computation (<1 second for 10K vectors of dim 4096). Fallback via numpy SVD if sklearn unavailable.

PCA is preferred over random projection because it preserves the most informative directions of the teacher's semantic space for this specific corpus's distribution patterns.

## Compatibility

Output format identical to PR #213 and the original random table:
- Same safetensors file with `engram.weight` tensor
- Same f16 dtype
- Same config TOML
- Training loop, eval harness, GGUF export unchanged — drop-in replacement

## Files to modify

1. **`training/ct87/generate_engram_table.py`** — add `--teacher` CLI flag, teacher embedding loading, PCA projection path in `generate_corpus_table()`
2. **`training/tests/test_generate_engram_table.py`** — test for teacher projection path (with mock/tiny embeddings)

## Success criteria

- KRILE trains with teacher-distilled table + xxhash keys
- Compare to: random table (-0.49%), corpus co-occurrence table (PR #213 results), and baseline
- If teacher distillation gives meaningfully better improvement than random JL, the semantic quality of the projection matters and further investment is warranted
