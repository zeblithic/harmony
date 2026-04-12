# Contrastive Co-Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable end-to-end co-training of the latent projection alongside the model, using projection keys for actual engram retrieval and InfoNCE contrastive loss at the n-gram level.

**Architecture:** The projection is a trainable auxiliary head (like UQ/MTP). Each training step: embed tokens, compute n-gram averages, project through MLP, binarize to keys, look up engram table, forward model with engram embeddings. Contrastive loss on n-gram averages trains the projection; CE loss trains the model. Detach boundaries prevent circular gradients.

**Tech Stack:** Python, PyTorch, ct87 training framework

**Spec:** `docs/superpowers/specs/2026-04-11-contrastive-cotrain-design.md`

---

### Task 1: `compute_ngram_averages()` function

**Files:**
- Modify: `training/ct87/latent_projection.py`
- Test: `training/tests/test_latent_projection.py`

This extracts the n-gram windowing logic from `project_ngrams()` into a standalone function that returns the pre-projection averages and positions. Needed because co-training computes contrastive loss on the averages separately from the key generation.

- [ ] **Step 1: Write the failing test**

In `training/tests/test_latent_projection.py`, add:

```python
class TestComputeNgramAverages:
    def test_output_shapes(self):
        """4 tokens -> 3 bigrams + 2 trigrams = 5 n-gram averages."""
        from ct87.latent_projection import compute_ngram_averages

        emb = torch.randn(1, 4, HIDDEN_DIM)
        avgs, positions = compute_ngram_averages(emb, 4)
        assert avgs.shape == (5, HIDDEN_DIM)
        assert positions == [1, 2, 3, 2, 3]

    def test_single_token_returns_empty(self):
        from ct87.latent_projection import compute_ngram_averages

        avgs, positions = compute_ngram_averages(torch.randn(1, 1, HIDDEN_DIM), 1)
        assert avgs.shape[0] == 0
        assert positions == []

    def test_bigram_average_is_correct(self):
        """Verify the average of two adjacent embeddings."""
        from ct87.latent_projection import compute_ngram_averages

        emb = torch.zeros(1, 3, HIDDEN_DIM)
        emb[0, 0, :] = 2.0
        emb[0, 1, :] = 4.0
        avgs, _ = compute_ngram_averages(emb, 3)
        # First bigram: avg(emb[0], emb[1]) = avg(2, 4) = 3
        assert torch.allclose(avgs[0], torch.full((HIDDEN_DIM,), 3.0))

    def test_preserves_grad(self):
        """Averages should be differentiable w.r.t. input embeddings."""
        from ct87.latent_projection import compute_ngram_averages

        emb = torch.randn(1, 4, HIDDEN_DIM, requires_grad=True)
        avgs, _ = compute_ngram_averages(emb, 4)
        avgs.sum().backward()
        assert emb.grad is not None
        assert emb.grad.abs().sum() > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/zeblith/work/zeblithic/harmony/training && python3 -m pytest tests/test_latent_projection.py::TestComputeNgramAverages -v`
Expected: FAIL with `ImportError: cannot import name 'compute_ngram_averages'`

- [ ] **Step 3: Write the implementation**

In `training/ct87/latent_projection.py`, add this function after the `LatentProjection` class (before `compute_key_overlap`):

```python
def compute_ngram_averages(
    embeddings: torch.Tensor,
    seq_len: int,
) -> tuple[torch.Tensor, list[int]]:
    """Extract bigram/trigram averaged embeddings without projecting.

    Same n-gram windowing as LatentProjection.project_ngrams() but returns
    the pre-projection averages. Used by contrastive co-training where the
    contrastive loss needs both the raw averages and the projected output.

    Operates per-sequence: takes [1, seq_len, hidden_dim], returns
    [num_ngrams, hidden_dim] and positions list.

    Args:
        embeddings: [1, seq_len, hidden_dim] token embeddings
        seq_len: sequence length

    Returns:
        (ngram_averages, positions) where ngram_averages is
        [num_ngrams, hidden_dim] and positions maps each n-gram
        to its attributed token position. Bigrams come first,
        then trigrams.
    """
    if seq_len < 2:
        return embeddings.new_zeros(0, embeddings.shape[-1]), []

    emb = embeddings.squeeze(0)  # [seq_len, hidden_dim]

    num_bi = seq_len - 1
    bi_avg = (emb[:num_bi] + emb[1 : num_bi + 1]) * 0.5

    num_tri = max(seq_len - 2, 0)
    if num_tri > 0:
        tri_avg = (emb[:num_tri] + emb[1 : num_tri + 1] + emb[2 : num_tri + 2]) / 3.0
        parts = torch.cat([bi_avg, tri_avg], dim=0)
    else:
        parts = bi_avg

    positions: list[int] = []
    for i in range(1, num_bi + 1):
        positions.append(i)
    for i in range(2, 2 + num_tri):
        positions.append(i)

    return parts, positions
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_latent_projection.py::TestComputeNgramAverages -v`
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add training/ct87/latent_projection.py training/tests/test_latent_projection.py
git commit -m "feat: add compute_ngram_averages() for contrastive co-training"
```

---

### Task 2: `contrastive_loss()` function

**Files:**
- Modify: `training/ct87/latent_projection.py`
- Test: `training/tests/test_latent_projection.py`

Python port of the InfoNCE contrastive loss from `crates/harmony-inference/src/latent_projection.rs:154-227`. Preserves nearest-neighbor topology from the original embedding space in the projected latent space.

- [ ] **Step 1: Write the failing tests**

In `training/tests/test_latent_projection.py`, add:

```python
class TestContrastiveLoss:
    def test_scalar_output(self):
        from ct87.latent_projection import contrastive_loss

        original = torch.randn(8, HIDDEN_DIM)
        projected = torch.randn(8, LATENT_DIM)
        loss = contrastive_loss(original, projected, temperature=0.07, k=4)
        assert loss.dim() == 0  # scalar

    def test_finite_nonnegative(self):
        from ct87.latent_projection import contrastive_loss

        original = torch.randn(16, HIDDEN_DIM)
        projected = torch.randn(16, LATENT_DIM)
        loss = contrastive_loss(original, projected, temperature=0.07, k=4)
        assert loss.item() >= 0.0
        assert torch.isfinite(loss)

    def test_aligned_lower_than_random(self):
        """Projection that preserves structure should have lower loss."""
        from ct87.latent_projection import contrastive_loss

        torch.manual_seed(42)
        original = torch.randn(16, 128)
        random_proj = torch.randn(16, 32)
        # Aligned: just truncate original (preserves all neighborhood structure)
        aligned_proj = original[:, :32]

        loss_random = contrastive_loss(original, random_proj, temperature=0.07, k=4)
        loss_aligned = contrastive_loss(original, aligned_proj, temperature=0.07, k=4)
        assert loss_aligned.item() < loss_random.item()

    def test_single_vector_returns_zero(self):
        from ct87.latent_projection import contrastive_loss

        loss = contrastive_loss(torch.randn(1, HIDDEN_DIM), torch.randn(1, LATENT_DIM))
        assert loss.item() == 0.0

    def test_gradient_flows_to_projected(self):
        """Contrastive loss should produce gradients for projected, not original."""
        from ct87.latent_projection import contrastive_loss

        original = torch.randn(8, HIDDEN_DIM)  # no requires_grad
        projected = torch.randn(8, LATENT_DIM, requires_grad=True)
        loss = contrastive_loss(original, projected, temperature=0.07, k=4)
        loss.backward()
        assert projected.grad is not None
        assert projected.grad.abs().sum() > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_latent_projection.py::TestContrastiveLoss -v`
Expected: FAIL with `ImportError: cannot import name 'contrastive_loss'`

- [ ] **Step 3: Write the implementation**

In `training/ct87/latent_projection.py`, add after `compute_ngram_averages`:

```python
def contrastive_loss(
    original: torch.Tensor,
    projected: torch.Tensor,
    temperature: float = 0.07,
    k: int = 4,
) -> torch.Tensor:
    """InfoNCE contrastive loss preserving nearest-neighbor topology.

    Port of crates/harmony-inference/src/latent_projection.rs::contrastive_loss.
    Ensures vectors that are nearest neighbors in the original embedding space
    remain nearest neighbors in the projected latent space.

    Args:
        original: [N, hidden_dim] -- n-gram averaged embeddings (typically detached)
        projected: [N, latent_dim] -- MLP output, post-tanh (with grad)
        temperature: scaling factor for logits (default: 0.07)
        k: number of top neighbors to preserve (default: 4)

    Returns:
        Scalar loss tensor.
    """
    n = original.shape[0]
    if n <= 1:
        return original.new_tensor(0.0)

    k = min(k, n - 1)
    if k == 0:
        return original.new_tensor(0.0)

    # Cosine similarity matrices [N, N]
    def _cosine_sim(x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(dim=1, keepdim=True).clamp(min=1e-8)
        normalized = x / norm
        return normalized @ normalized.t()

    sim_orig = _cosine_sim(original)
    sim_proj = _cosine_sim(projected)

    # Scale projected similarities by temperature, mask diagonal
    logits = sim_proj / temperature
    diag_mask = torch.full((n, n), 0.0, device=original.device)
    diag_mask.fill_diagonal_(-1e9)
    logits = logits + diag_mask

    # Build soft targets from original similarities: keep only top-k per row
    with torch.no_grad():
        orig_data = sim_orig.clone()
        orig_data.fill_diagonal_(float("-inf"))  # exclude self
        _, top_indices = orig_data.topk(k, dim=1)

        target_logits = torch.full((n, n), float("-inf"), device=original.device)
        target_logits.scatter_(1, top_indices, sim_orig.gather(1, top_indices))
        targets = torch.softmax(target_logits, dim=1)

    # Cross-entropy: -sum(targets * log_softmax(logits)) / N
    log_probs = torch.log_softmax(logits, dim=1)
    loss = -(targets * log_probs).sum() / n

    return loss
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_latent_projection.py::TestContrastiveLoss -v`
Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
git add training/ct87/latent_projection.py training/tests/test_latent_projection.py
git commit -m "feat: add contrastive_loss() — InfoNCE for n-gram embeddings"
```

---

### Task 3: `EngramTable.lookup_from_keys()` method

**Files:**
- Modify: `training/ct87/engram.py`
- Test: `training/tests/test_latent_projection.py`

New method that takes pre-computed binary keys and positions directly, without re-running the projection. Needed because co-training computes keys once (with grad for contrastive loss) then passes them to the table lookup (no grad).

- [ ] **Step 1: Write the failing tests**

In `training/tests/test_latent_projection.py`, add:

```python
class TestLookupFromKeys:
    def test_output_shape(self):
        tbl = _make_table()
        # Simulate 3 bigrams + 2 trigrams = 5 keys for a seq_len=4 sequence
        proj = _make_projection()
        emb = torch.randn(1, 4, HIDDEN_DIM)
        keys, positions = proj.project_ngrams(emb, 4)
        out = tbl.lookup_from_keys(keys, positions, batch_size=1, seq_len=4)
        assert out.shape == (1, 4, tbl.engram_dim)

    def test_position_zero_is_zero(self):
        tbl = _make_table()
        proj = _make_projection()
        emb = torch.randn(1, 4, HIDDEN_DIM)
        keys, positions = proj.project_ngrams(emb, 4)
        out = tbl.lookup_from_keys(keys, positions, batch_size=1, seq_len=4)
        assert out[0, 0].abs().max().item() < 1e-6

    def test_matches_lookup_batch_projected(self):
        """lookup_from_keys with projection-generated keys should match
        lookup_batch_projected on the same input."""
        cfg = _tiny_config()
        torch.manual_seed(42)
        model = HarmonyModel(cfg)
        proj = LatentProjection(cfg.hidden_dim, INTERMEDIATE_DIM, LATENT_DIM)
        tbl = _make_table()

        input_ids = torch.randint(0, cfg.vocab_size, (1, 8))
        with torch.no_grad():
            emb = model.embed_tokens(input_ids)

        # lookup_batch_projected computes keys internally
        expected = tbl.lookup_batch_projected(input_ids, emb, proj)

        # lookup_from_keys uses pre-computed keys
        keys, positions = proj.project_ngrams(emb[0:1], 8)
        actual = tbl.lookup_from_keys(keys, positions, batch_size=1, seq_len=8)

        assert torch.allclose(expected, actual, atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_latent_projection.py::TestLookupFromKeys -v`
Expected: FAIL with `AttributeError: 'EngramTable' object has no attribute 'lookup_from_keys'`

- [ ] **Step 3: Write the implementation**

In `training/ct87/engram.py`, add this method to `EngramTable` after `lookup_batch_projected`:

```python
    def lookup_from_keys(
        self,
        binary_keys: list[bytes],
        positions: list[int],
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor:
        """Look up Engram embeddings from pre-computed binary keys.

        Unlike lookup_batch_projected(), this does not run the projection —
        it takes pre-computed keys directly. Used by contrastive co-training
        where keys are computed once (with grad) and reused for both
        contrastive loss and table lookup.

        All keys/positions belong to batch item 0. For multi-item batches,
        call once per batch item and stack, or extend with batch indexing.

        Args:
            binary_keys: list of binary key bytes from to_binary_keys()
            positions: token position each key is attributed to
            batch_size: output batch dimension (typically 1 per call)
            seq_len: output sequence length dimension

        Returns:
            [batch_size, seq_len, engram_dim] embedding tensor
        """
        batch_indices: list[int] = []
        pos_list: list[int] = []
        table_indices: list[int] = []

        for key_bytes, pos in zip(binary_keys, positions):
            for seed in self.hash_seeds:
                idx = _xxhash64(key_bytes, seed) % self.total_entries
                batch_indices.append(0)
                pos_list.append(pos)
                table_indices.append(idx)

        result = torch.zeros(
            batch_size, seq_len, self.engram_dim, dtype=torch.float32,
        )

        if table_indices:
            idx_t = torch.tensor(table_indices, dtype=torch.long)
            embs = self.table[idx_t]

            for i, (b, pos) in enumerate(zip(batch_indices, pos_list)):
                result[b, pos] += embs[i]

        return result.to(self.device)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_latent_projection.py::TestLookupFromKeys -v`
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add training/ct87/engram.py training/tests/test_latent_projection.py
git commit -m "feat: add EngramTable.lookup_from_keys() for pre-computed keys"
```

---

### Task 4: CLI flags and projection setup in train.py

**Files:**
- Modify: `training/ct87/train.py`

Add the contrastive loss CLI flags, the `--latent-projection-init` option, and update the projection setup block to handle trainable mode.

- [ ] **Step 1: Add CLI flags**

In `training/ct87/train.py`, after the existing `--latent-dim` argument (around line 268), add:

```python
    parser.add_argument(
        "--latent-projection-init", action="store_true",
        help="Randomly initialize latent projection (for from-scratch co-training)",
    )
    parser.add_argument(
        "--contrastive-loss", action="store_true",
        help="Enable contrastive co-training (makes projection trainable, uses "
             "projection keys for engram retrieval with InfoNCE auxiliary loss)",
    )
    parser.add_argument(
        "--contrastive-loss-weight", type=float, default=0.1,
        help="Weight for contrastive auxiliary loss (default: 0.1)",
    )
    parser.add_argument(
        "--contrastive-temperature", type=float, default=0.07,
        help="Temperature for InfoNCE contrastive loss (default: 0.07)",
    )
    parser.add_argument(
        "--contrastive-k", type=int, default=4,
        help="Number of top neighbors to preserve in contrastive loss (default: 4)",
    )
```

- [ ] **Step 2: Add validation**

After the existing `--latent-projection` validation block (around line 330), add:

```python
    if args.contrastive_loss:
        if args.latent_projection is None and not args.latent_projection_init:
            print(
                "Error: --contrastive-loss requires --latent-projection or "
                "--latent-projection-init",
                file=sys.stderr,
            )
            sys.exit(1)
        if args.latent_intermediate_dim is None or args.latent_dim is None:
            print(
                "Error: --contrastive-loss requires both "
                "--latent-intermediate-dim and --latent-dim",
                file=sys.stderr,
            )
            sys.exit(1)
        if args.engram_table is None:
            print(
                "Error: --contrastive-loss requires --engram-table",
                file=sys.stderr,
            )
            sys.exit(1)

    if args.latent_projection_init:
        if args.latent_intermediate_dim is None or args.latent_dim is None:
            print(
                "Error: --latent-projection-init requires both "
                "--latent-intermediate-dim and --latent-dim",
                file=sys.stderr,
            )
            sys.exit(1)
        if args.latent_projection is not None:
            print(
                "Error: --latent-projection and --latent-projection-init "
                "are mutually exclusive",
                file=sys.stderr,
            )
            sys.exit(1)
```

- [ ] **Step 3: Update projection setup block**

Replace the existing latent projection setup block (the `if args.latent_projection is not None:` block around line 420) with:

```python
    # Latent projection setup
    latent_projection = None
    if args.latent_projection is not None:
        from ct87.latent_projection import LatentProjection

        latent_projection = LatentProjection.from_checkpoint(
            args.latent_projection,
            hidden_dim=config.hidden_dim,
            intermediate_dim=args.latent_intermediate_dim,
            latent_dim=args.latent_dim,
            device=device,
        )
        if args.contrastive_loss:
            latent_projection.requires_grad_(True)
            print(
                f"Latent projection loaded (trainable): "
                f"{config.hidden_dim}->{args.latent_intermediate_dim}->{args.latent_dim}"
            )
        else:
            latent_projection.requires_grad_(False)
            proj_params = sum(p.numel() for p in latent_projection.parameters())
            print(
                f"Latent projection loaded (frozen): "
                f"{config.hidden_dim}->{args.latent_intermediate_dim}->{args.latent_dim}, "
                f"{proj_params:,} params"
            )
    elif args.latent_projection_init:
        from ct87.latent_projection import LatentProjection

        latent_projection = LatentProjection(
            config.hidden_dim, args.latent_intermediate_dim, args.latent_dim,
        ).to(device)
        proj_params = sum(p.numel() for p in latent_projection.parameters())
        print(
            f"Latent projection initialized (trainable): "
            f"{config.hidden_dim}->{args.latent_intermediate_dim}->{args.latent_dim}, "
            f"{proj_params:,} params"
        )
```

- [ ] **Step 4: Add projection to optimizer and grad clipping**

Update the optimizer setup (around line 476). After the existing `if mtp_head is not None:` block:

```python
    if latent_projection is not None and args.contrastive_loss:
        adam_params.extend(latent_projection.parameters())
```

Update the grad clipping block (around line 665). After `if mtp_head is not None:`:

```python
                if latent_projection is not None and args.contrastive_loss:
                    all_params.extend(latent_projection.parameters())
```

- [ ] **Step 5: Add `_save_latent_projection` helper**

After the existing `_save_mtp_head` function (around line 130):

```python
def _save_latent_projection(
    projection: torch.nn.Module,
    step: int,
    output_dir: str,
) -> None:
    """Save latent projection weights alongside the model checkpoint."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"latent_projection_step_{step}.pt")
    torch.save(projection.state_dict(), path)
```

- [ ] **Step 6: Commit**

```bash
git add training/ct87/train.py
git commit -m "feat: CLI flags and projection setup for contrastive co-training"
```

---

### Task 5: Training loop integration

**Files:**
- Modify: `training/ct87/train.py`

Wire contrastive loss into the training loop: replace the frozen-projection engram lookup with the co-training path, add contrastive loss to total loss, update CSV logging, add checkpoint saving.

- [ ] **Step 1: Update CSV header**

Change the `expected_header` line (around line 499):

```python
        expected_header = ["step", "loss", "uq_loss", "mtp_loss", "cl_loss", "val_loss", "lr", "grad_norm", "num_thoughts", "dt_ms"]
```

- [ ] **Step 2: Add `accum_cl_loss` accumulator**

In the training loop, after `accum_mtp_loss = 0.0` (around line 538):

```python
            accum_cl_loss = 0.0
```

- [ ] **Step 3: Replace engram lookup with co-training path**

Replace the engram lookup block (the `if engram_table is not None and latent_projection is not None:` block, around line 548) with:

```python
                # Compute Engram embeddings if table is loaded
                engram_emb = None
                cl_projected = None
                cl_ngram_avgs = None
                if engram_table is not None and latent_projection is not None and args.contrastive_loss:
                    # Co-training: compute keys with grad for contrastive loss,
                    # use them for engram retrieval
                    from ct87.latent_projection import compute_ngram_averages

                    emb = model.embed_tokens(input_ids)
                    seq_len_actual = input_ids.shape[1]

                    all_keys: list[bytes] = []
                    all_positions: list[int] = []
                    all_ngram_avgs: list[torch.Tensor] = []
                    all_projected: list[torch.Tensor] = []

                    for b_idx in range(input_ids.shape[0]):
                        ngram_avgs, positions = compute_ngram_averages(
                            emb[b_idx : b_idx + 1], seq_len_actual,
                        )
                        projected = latent_projection(ngram_avgs.detach())
                        binary_keys = latent_projection.to_binary_keys(projected)

                        all_ngram_avgs.append(ngram_avgs.detach())
                        all_projected.append(projected)
                        all_keys.extend(binary_keys)
                        all_positions.extend(positions)

                    cl_ngram_avgs = all_ngram_avgs
                    cl_projected = all_projected

                    # Build engram embeddings from pre-computed keys
                    engram_parts = []
                    offset = 0
                    for b_idx in range(input_ids.shape[0]):
                        n_keys = len(all_ngram_avgs[b_idx])
                        part_keys = all_keys[offset : offset + n_keys]
                        part_pos = all_positions[offset : offset + n_keys]
                        offset += n_keys
                        part = engram_table.lookup_from_keys(
                            part_keys, part_pos,
                            batch_size=1, seq_len=seq_len_actual,
                        )
                        engram_parts.append(part)
                    engram_emb = torch.cat(engram_parts, dim=0)
                elif engram_table is not None and latent_projection is not None:
                    with torch.no_grad():
                        emb = model.embed_tokens(input_ids)
                        engram_emb = engram_table.lookup_batch_projected(
                            input_ids, emb, latent_projection,
                        )
                elif engram_table is not None:
                    with torch.no_grad():
                        engram_emb = engram_table.lookup_batch(input_ids)
```

- [ ] **Step 4: Add contrastive loss to total loss**

After the MTP auxiliary loss block (after `accum_mtp_loss += mtp_loss_val.item()`), add:

```python
                    # Contrastive auxiliary loss
                    if args.contrastive_loss and cl_projected is not None:
                        from ct87.latent_projection import contrastive_loss

                        cl_total = original.new_tensor(0.0) if not cl_projected else sum(
                            contrastive_loss(
                                avg, proj,
                                temperature=args.contrastive_temperature,
                                k=args.contrastive_k,
                            )
                            for avg, proj in zip(cl_ngram_avgs, cl_projected)
                        ) / len(cl_projected)
                        loss = loss + args.contrastive_loss_weight * cl_total
                        accum_cl_loss += cl_total.item()
```

```python
                    # Contrastive auxiliary loss
                    if args.contrastive_loss and cl_projected is not None:
                        from ct87.latent_projection import contrastive_loss

                        cl_total = sum(
                            contrastive_loss(
                                avg, proj,
                                temperature=args.contrastive_temperature,
                                k=args.contrastive_k,
                            )
                            for avg, proj in zip(cl_ngram_avgs, cl_projected)
                        ) / len(cl_projected)
                        loss = loss + args.contrastive_loss_weight * cl_total
                        accum_cl_loss += cl_total.item()
```

- [ ] **Step 5: Update logging and CSV output**

In the step logging block (around line 675), after the `mtp_str` section, add:

```python
                cl_str = ""
                if args.contrastive_loss:
                    raw_cl = accum_cl_loss / args.grad_accum_steps
                    cl_str = f"  cl_loss={raw_cl:.4f}"
```

Update the print line to include `cl_str`:

```python
                print(f"step={step:5d}  loss={raw_loss:.4f}  lr={current_lr:.6f}{ct_str}{uq_str}{mtp_str}{cl_str}")
```

In the CSV writing block (around line 715), after `mtp_loss_str`, add:

```python
                cl_loss_str = ""
                if args.contrastive_loss:
                    cl_loss_str = f"{accum_cl_loss / args.grad_accum_steps:.6f}"
```

Update the `csv_writer.writerow` call to include `cl_loss_str` after `mtp_loss_str`:

```python
                csv_writer.writerow([
                    step,
                    f"{raw_loss:.6f}",
                    uq_loss_str,
                    mtp_loss_str,
                    cl_loss_str,
                    val_loss_str,
                    f"{current_lr:.8f}",
                    f"{grad_norm:.6f}" if grad_norm is not None else "",
                    num_thoughts,
                    f"{dt_ms:.1f}",
                ])
```

- [ ] **Step 6: Add checkpoint saving**

At both checkpoint save points (the periodic save around line 694 and the final save around line 735), after the MTP save, add:

```python
                if latent_projection is not None and args.contrastive_loss:
                    _save_latent_projection(latent_projection, step, args.output_dir)
```

(Same line at both save points, with appropriate `step` variable — `step` for periodic, `args.steps` for final.)

- [ ] **Step 7: Commit**

```bash
git add training/ct87/train.py
git commit -m "feat: wire contrastive co-training into training loop"
```

---

### Task 6: Run all tests and verify

**Files:**
- All modified files

- [ ] **Step 1: Run the full test suite**

Run: `cd /Users/zeblith/work/zeblithic/harmony/training && python3 -m pytest tests/test_latent_projection.py tests/test_eval.py tests/test_engram.py -v`
Expected: All tests pass (64 existing + new tests from tasks 1-3)

- [ ] **Step 2: Verify no regressions in train tests**

Run: `python3 -m pytest tests/test_train.py -v -k "not test_too_few_tokens_raises"`
Expected: Pass (excluding the pre-existing `datasets` module failure)

- [ ] **Step 3: Commit final state**

If any fixups were needed, commit them:

```bash
git add -A
git commit -m "fix: test fixups for contrastive co-training"
```

---

### Task 7: Create PR

**Files:** None (git operations only)

- [ ] **Step 1: Push and create PR**

```bash
git push -u origin HEAD
gh pr create --title "feat: contrastive co-training for latent projection (ZEB-85 Approach B)" \
  --body "$(cat <<'EOF'
## Summary

- Add `contrastive_loss()` — InfoNCE loss at the n-gram level preserving embedding topology
- Add `compute_ngram_averages()` — extracts pre-projection n-gram averages for contrastive loss
- Add `EngramTable.lookup_from_keys()` — table lookup from pre-computed binary keys
- Wire co-training into `train.py`: trainable projection, contrastive auxiliary loss, checkpoint saving
- New CLI: `--contrastive-loss`, `--contrastive-loss-weight`, `--contrastive-temperature`, `--contrastive-k`, `--latent-projection-init`

### Context

Approach A (frozen projection from scratch) is running. This PR enables Approach B: end-to-end co-training where the projection adapts to the model's evolving embeddings. No train/inference key distribution mismatch.

### Usage

```bash
python -m ct87.train --config target \
    --data <data> --val-data <val> \
    --engram-table <engram.safetensors> \
    --latent-projection-init \
    --latent-intermediate-dim 640 --latent-dim 64 \
    --contrastive-loss --contrastive-loss-weight 0.1 \
    --steps 10000 --lr 3e-4
```

## Test plan

- [ ] All existing + new tests pass
- [ ] KRILE runs Approach B co-training from scratch
- [ ] Compare results: Approach A vs Approach B vs xxhash-engram vs baseline

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```
