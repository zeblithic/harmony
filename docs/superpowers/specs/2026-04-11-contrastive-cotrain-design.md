# End-to-End Contrastive Co-Training for Latent Projection (ZEB-85, Approach B)

## Problem

The latent projection head trains cleanly as a side task (contrastive loss 5.27 to 2.92), but the model can't use projection-generated engram keys at inference because it was trained on xxhash-retrieved residuals. The key distribution mismatch is 99.99%. Fine-tuning with frozen projection keys caused the gating to atrophy entirely.

## Solution

Co-train the projection alongside the model from scratch. The projection's own keys are used for actual engram retrieval during training (not xxhash), and contrastive loss keeps the projection meaningful as the model's embeddings evolve. No train/inference mismatch because training and inference use the same key generation path.

## Architecture

### Contrastive loss function

Port of the Rust InfoNCE loss from `latent_projection.rs`, operating on n-gram averaged embeddings (not raw per-token embeddings). N-gram level is the natural fit because these are the representations that become binary keys.

```
contrastive_loss(original, projected, temperature=0.07, k=4) -> scalar
```

- `original`: `[num_ngrams, hidden_dim]` -- bigram/trigram averaged embeddings (detached)
- `projected`: `[num_ngrams, latent_dim]` -- MLP output (post-tanh, with grad)
- Cosine similarity matrices for both spaces
- Top-k soft targets from original space, cross-entropy against projected logits
- Diagonal masked to -inf, processed per-sequence to keep matrix size at O(seq_len^2)

Helper: `compute_ngram_averages(embeddings, seq_len)` extracts bigram/trigram averaged embeddings and their positions without projecting. Operates per-sequence (takes `[1, seq_len, hidden_dim]`, returns `[num_ngrams, hidden_dim]` and positions list). Called in a per-batch-item loop in the training step, matching the per-sequence contrastive loss computation.

### Training loop integration

Follows the established auxiliary head pattern (UQ, MTP, ThoughtNorm):

1. `embeddings = model.embed_tokens(input_ids)` -- shared, with grad
2. `ngram_avgs, positions = compute_ngram_averages(embeddings, seq_len)` -- per-sequence
3. `projected = projection(ngram_avgs.detach())` -- with grad for contrastive loss
4. `binary_keys = projection.to_binary_keys(projected)` -- binarization, no grad
5. `engram_emb = engram_table.lookup_from_keys(binary_keys, positions, batch_size, seq_len)` -- new method, takes pre-computed keys
6. Model forward with `engram_embeddings=engram_emb`
7. `ce_loss = cross_entropy(logits, targets)`
8. `cl_loss = contrastive_loss(ngram_avgs.detach(), projected)` -- detach originals so contrastive loss only trains projection
9. `loss = ce_loss + alpha * cl_loss`

The `detach()` on `ngram_avgs` at step 3 and 8 prevents circular gradients through the embedding layer. The contrastive loss trains only the projection; the CE loss trains the model (including embeddings). They share the embedding layer but have clean gradient boundaries.

### New `EngramTable.lookup_from_keys()` method

Existing `lookup_batch_projected()` re-runs the full projection internally. Co-training needs to split this: compute keys once (with grad for contrastive loss), then pass pre-computed keys to table lookup (no grad). New method:

```python
def lookup_from_keys(
    self,
    binary_keys: list[bytes],
    positions: list[int],
    batch_size: int,
    seq_len: int,
) -> torch.Tensor:  # [batch, seq_len, engram_dim]
```

Takes pre-computed binary keys and positions directly, performs `_xxhash64(key_bytes, seed) % total_entries` and scatter-add aggregation. No projection dependency.

### Projection setup

- **Trainable:** `requires_grad_(True)` (opposite of PR 207's frozen mode)
- **Optimizer:** projection params added to the Adam param group (small 2-layer MLP, not suited for Muon)
- **Grad clipping:** projection params included in `clip_grad_norm_` alongside model/UQ/MTP/ThoughtNorm
- **Init options:**
  - `--latent-projection <path>`: load pre-trained weights (trainable when `--contrastive-loss` is set)
  - `--latent-projection-init`: randomly initialize (for from-scratch co-training)
  - One of the two required when `--contrastive-loss` is set

### CLI flags

- `--contrastive-loss`: enable contrastive co-training (makes projection trainable)
- `--contrastive-loss-weight FLOAT`: alpha for contrastive loss (default: 0.1)
- `--contrastive-temperature FLOAT`: temperature for InfoNCE (default: 0.07)
- `--contrastive-k INT`: number of top neighbors to preserve (default: 4)
- `--latent-projection-init`: randomly initialize projection instead of loading checkpoint

### Checkpoint saving

`latent_projection_step_N.pt` saved alongside model/UQ/MTP/ThoughtNorm at each save point. Same pattern as `_save_uq_head()` / `_save_mtp_head()`. Saved weights are directly usable by the eval harness and by Rust inference after GGUF export.

### CSV logging

New column `cl_loss` in the CSV header:

```
step, loss, uq_loss, mtp_loss, cl_loss, val_loss, lr, grad_norm, num_thoughts, dt_ms
```

Empty string when `--contrastive-loss` is not enabled (same pattern as uq_loss/mtp_loss).

### Validation

`compute_validation_loss()` uses the projection for engram lookup but does NOT compute contrastive loss during validation. val_loss is pure CE, keeping it comparable across runs.

### Mode interactions

- `--contrastive-loss` requires `--engram-table` and one of `--latent-projection` / `--latent-projection-init`
- `--contrastive-loss` requires `--latent-intermediate-dim` and `--latent-dim`
- Compatible with all other auxiliary heads (COCONUT, UQ, MTP, QAT)
- Without `--contrastive-loss`, `--latent-projection` behaves as before (frozen, PR 207 behavior)

## Files to modify

1. **`ct87/latent_projection.py`** -- add `contrastive_loss()`, `compute_ngram_averages()`
2. **`ct87/engram.py`** -- add `EngramTable.lookup_from_keys()`
3. **`ct87/train.py`** -- co-training integration, CLI flags, optimizer/checkpoint/logging changes
4. **`tests/test_latent_projection.py`** -- tests for contrastive loss, ngram averages, lookup_from_keys

## Usage

```bash
# Co-train from scratch (Approach B)
python -m ct87.train \
    --config target --data <training-data> --val-data <val-data> \
    --engram-table <engram.safetensors> \
    --latent-projection-init \
    --latent-intermediate-dim 640 --latent-dim 64 \
    --contrastive-loss --contrastive-loss-weight 0.1 \
    --steps 10000 --lr 3e-4 --save-every 250 \
    --log-file cotrain.csv

# Co-train from pre-trained projection
python -m ct87.train \
    --config target --data <training-data> --val-data <val-data> \
    --engram-table <engram.safetensors> \
    --latent-projection <projection.pt> \
    --latent-intermediate-dim 640 --latent-dim 64 \
    --contrastive-loss --contrastive-loss-weight 0.1 \
    --steps 10000 --lr 3e-4

# Evaluate after co-training
python -m ct87.eval \
    --checkpoint <checkpoint> --config target \
    --data <val-data> --engram-table <engram.safetensors> \
    --latent-projection <latent_projection_step_N.pt> \
    --latent-intermediate-dim 640 --latent-dim 64 \
    --compare --key-overlap --num-batches 100
```

## Success criteria

- Co-trained model with projection keys matches or beats the xxhash-engram val_loss (-0.49%)
- Contrastive loss decreases during training (projection is learning)
- Key overlap analysis shows projection keys are meaningfully different from xxhash
- Engram contribution is non-zero (projection eval != baseline eval)
