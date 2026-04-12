# Teacher-Distilled Engram Table Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `--teacher` flag to the engram table generator that projects next-token distributions through a pre-trained model's embedding matrix + PCA, replacing random JL projection with semantically meaningful compression.

**Architecture:** Extends `generate_corpus_table()` with an optional `teacher_embed_matrix` parameter. When provided, projects `probs @ embed_matrix` (→ [entries, teacher_dim]) then applies PCA to reduce to `embedding_dim`. Everything else (corpus scan, counting, normalization, output format) stays identical.

**Tech Stack:** Python, NumPy, sklearn (PCA), transformers (AutoModel for embedding loading)

**Spec:** `docs/superpowers/specs/2026-04-12-teacher-distilled-engram-table-design.md`

---

### Task 1: Add teacher projection path to `generate_corpus_table()`

**Files:**
- Modify: `training/ct87/generate_engram_table.py`
- Test: `training/tests/test_generate_engram_table.py`

Add an optional `teacher_embed_matrix` parameter that replaces the random JL projection with teacher embedding projection + PCA.

- [ ] **Step 1: Write the failing tests**

Add to `training/tests/test_generate_engram_table.py`:

```python
class TestTeacherProjection:
    def test_output_shape_with_teacher(self):
        """Teacher projection should produce [entries, dim] float32."""
        from ct87.generate_engram_table import generate_corpus_table

        chunks = [[1, 2, 3, 4, 5, 6, 7, 8]] * 10
        # Fake teacher embeddings: [vocab=32, teacher_dim=64]
        rng = np.random.RandomState(42)
        teacher_matrix = rng.randn(32, 64).astype(np.float32)

        table = generate_corpus_table(
            chunks, total_entries=100, embedding_dim=16,
            vocab_size=32, hash_seeds=[42, 99],
            teacher_embed_matrix=teacher_matrix,
        )
        assert table.shape == (100, 16)
        assert table.dtype == np.float32

    def test_unit_normalized_with_teacher(self):
        """Non-zero rows from teacher projection should be unit-normalized."""
        from ct87.generate_engram_table import generate_corpus_table

        chunks = [[1, 2, 3, 4, 5, 6, 7, 8]] * 10
        rng = np.random.RandomState(42)
        teacher_matrix = rng.randn(32, 64).astype(np.float32)

        table = generate_corpus_table(
            chunks, total_entries=100, embedding_dim=16,
            vocab_size=32, hash_seeds=[42, 99],
            teacher_embed_matrix=teacher_matrix,
        )
        for i in range(100):
            norm = np.linalg.norm(table[i])
            if norm > 0:
                assert abs(norm - 1.0) < 1e-5, f"Row {i} norm={norm}"

    def test_different_from_random_projection(self):
        """Teacher projection should produce different results than random JL."""
        from ct87.generate_engram_table import generate_corpus_table

        chunks = [[1, 2, 3, 4, 5, 6, 7, 8]] * 20
        rng = np.random.RandomState(42)
        teacher_matrix = rng.randn(32, 64).astype(np.float32)

        table_random = generate_corpus_table(
            chunks, total_entries=100, embedding_dim=16,
            vocab_size=32, hash_seeds=[42, 99],
        )
        table_teacher = generate_corpus_table(
            chunks, total_entries=100, embedding_dim=16,
            vocab_size=32, hash_seeds=[42, 99],
            teacher_embed_matrix=teacher_matrix,
        )
        assert not np.allclose(table_random, table_teacher)

    def test_deterministic_with_teacher(self):
        """Same inputs should produce identical results."""
        from ct87.generate_engram_table import generate_corpus_table

        chunks = [[1, 2, 3, 4, 5, 6, 7, 8]] * 10
        rng = np.random.RandomState(42)
        teacher_matrix = rng.randn(32, 64).astype(np.float32)

        t1 = generate_corpus_table(
            chunks, total_entries=100, embedding_dim=16,
            vocab_size=32, hash_seeds=[42, 99],
            teacher_embed_matrix=teacher_matrix,
        )
        t2 = generate_corpus_table(
            chunks, total_entries=100, embedding_dim=16,
            vocab_size=32, hash_seeds=[42, 99],
            teacher_embed_matrix=teacher_matrix,
        )
        np.testing.assert_array_equal(t1, t2)

    def test_teacher_vocab_mismatch_raises(self):
        """Teacher matrix with wrong vocab dimension should raise."""
        from ct87.generate_engram_table import generate_corpus_table

        chunks = [[1, 2, 3, 4, 5, 6, 7, 8]] * 5
        # vocab_size=32 but teacher has 16 rows
        wrong_matrix = np.random.randn(16, 64).astype(np.float32)

        with pytest.raises(ValueError, match="teacher_embed_matrix"):
            generate_corpus_table(
                chunks, total_entries=100, embedding_dim=16,
                vocab_size=32, hash_seeds=[42, 99],
                teacher_embed_matrix=wrong_matrix,
            )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/zeblith/work/zeblithic/harmony/training && python3 -m pytest tests/test_generate_engram_table.py::TestTeacherProjection -v`
Expected: FAIL with `TypeError: generate_corpus_table() got an unexpected keyword argument 'teacher_embed_matrix'`

- [ ] **Step 3: Write the implementation**

In `training/ct87/generate_engram_table.py`, modify `generate_corpus_table()`:

1. Add `teacher_embed_matrix` parameter to the function signature:

```python
def generate_corpus_table(
    chunks: Sequence[list[int]],
    total_entries: int = DEFAULT_ENTRIES,
    embedding_dim: int = DEFAULT_DIM,
    vocab_size: int = 32000,
    hash_seeds: list[int] | None = None,
    projection_seed: int = 0,
    teacher_embed_matrix: np.ndarray | None = None,
) -> np.ndarray:
```

2. Update the docstring to document the new parameter:

```python
        teacher_embed_matrix: Optional [vocab_size, teacher_dim] embedding matrix
            from a pre-trained model. When provided, projects through teacher
            embeddings + PCA instead of random JL projection. Shape[0] must
            equal vocab_size.
```

3. Add validation after the existing `hash_seeds` check:

```python
    if teacher_embed_matrix is not None:
        if teacher_embed_matrix.shape[0] != vocab_size:
            raise ValueError(
                f"teacher_embed_matrix has {teacher_embed_matrix.shape[0]} rows, "
                f"expected vocab_size={vocab_size}"
            )
```

4. Replace the projection block (lines 146-171, from `# Johnson-Lindenstrauss` through `table[has_counts] = projected / norms`) with:

```python
    # Only project rows that have data (avoids overflow warnings on empty rows)
    table = np.zeros((total_entries, embedding_dim), dtype=np.float32)
    if has_counts.any():
        # Normalize active rows to probability distributions (L1)
        active_counts = counts[has_counts]
        active_sums = row_sums[has_counts, np.newaxis]
        probs = active_counts / active_sums

        if teacher_embed_matrix is not None:
            # Teacher projection: probs @ embed_matrix -> [active, teacher_dim]
            # then PCA to reduce to embedding_dim
            from sklearn.decomposition import PCA

            projected_full = (probs @ teacher_embed_matrix.astype(np.float32))
            np.nan_to_num(projected_full, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

            pca = PCA(n_components=embedding_dim)
            projected = pca.fit_transform(projected_full).astype(np.float32)
        else:
            # Random Johnson-Lindenstrauss projection
            rng = np.random.RandomState(projection_seed)
            proj_matrix = rng.randn(vocab_size, embedding_dim).astype(np.float32)
            proj_matrix /= np.sqrt(embedding_dim)

            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                projected = probs @ proj_matrix

        # Clamp any NaN/inf from float32 overflow before normalizing
        np.nan_to_num(projected, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # Unit-normalize (L2)
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        table[has_counts] = projected / norms
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_generate_engram_table.py -v`
Expected: 12 PASSED (7 existing + 5 new)

- [ ] **Step 5: Verify no regressions**

Run: `python3 -m pytest tests/test_latent_projection.py tests/test_eval.py tests/test_engram.py -q`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add training/ct87/generate_engram_table.py training/tests/test_generate_engram_table.py
git commit -m "feat: teacher embedding projection + PCA in generate_corpus_table()"
```

---

### Task 2: Wire `--teacher` CLI flag and embedding loader

**Files:**
- Modify: `training/ct87/generate_engram_table.py`

Add the `--teacher` CLI flag, load the embedding matrix, and pass it through.

- [ ] **Step 1: Add CLI flag**

After the `--projection-seed` argument in `main()`, add:

```python
    parser.add_argument(
        "--teacher", type=str, default=None,
        help="HuggingFace model name for teacher embedding projection "
             "(e.g., mistralai/Mistral-7B-v0.1). Requires --corpus.",
    )
```

- [ ] **Step 2: Add validation**

After the existing CLI validations (before the `output_dir` setup), add:

```python
    if args.teacher is not None and args.corpus is None:
        parser.error("--teacher requires --corpus")
```

- [ ] **Step 3: Add embedding loader and update corpus path**

In the `if args.corpus is not None:` branch, before the call to `generate_and_save_corpus_table()`, add the teacher loading logic. Replace the entire `if args.corpus is not None:` block with:

```python
    if args.corpus is not None:
        teacher_embed_matrix = None
        if args.teacher is not None:
            import torch
            from transformers import AutoModel

            print(f"Loading teacher embeddings from {args.teacher}...")
            teacher_model = AutoModel.from_pretrained(
                args.teacher, torch_dtype=torch.float16,
            )
            teacher_embed_matrix = (
                teacher_model.get_input_embeddings()
                .weight.detach().float().numpy()
            )
            del teacher_model
            print(
                f"Teacher embeddings loaded: {teacher_embed_matrix.shape} "
                f"({teacher_embed_matrix.nbytes / 1024 / 1024:.0f} MB)"
            )

        generate_and_save_corpus_table(
            data_path=args.corpus,
            total_entries=args.entries,
            embedding_dim=args.dim,
            output_dir=args.output_dir,
            shard_size=args.shard_size,
            vocab_size=args.vocab_size,
            projection_seed=args.projection_seed,
            teacher_embed_matrix=teacher_embed_matrix,
        )
```

- [ ] **Step 4: Update `generate_and_save_corpus_table()` to pass through teacher matrix**

Add the parameter to the function signature:

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
    teacher_embed_matrix: np.ndarray | None = None,
) -> Path:
```

Update the docstring to add:
```
        teacher_embed_matrix: Optional teacher embedding matrix for projection.
```

And pass it through to `generate_corpus_table()`:

```python
    table = generate_corpus_table(
        chunks,
        total_entries=total_entries,
        embedding_dim=embedding_dim,
        vocab_size=vocab_size,
        hash_seeds=hash_seeds,
        projection_seed=projection_seed,
        teacher_embed_matrix=teacher_embed_matrix,
    )
```

- [ ] **Step 5: Run all tests**

Run: `python3 -m pytest tests/test_generate_engram_table.py tests/test_latent_projection.py tests/test_eval.py tests/test_engram.py -q`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add training/ct87/generate_engram_table.py
git commit -m "feat: --teacher CLI flag for teacher-distilled engram table"
```

---

### Task 3: Push and create PR

**Files:** None (git operations only)

- [ ] **Step 1: Push and create PR**

```bash
git push -u origin HEAD
gh pr create --title "feat: teacher-distilled engram table via pre-trained embeddings + PCA (ZEB-102)" \
  --body "$(cat <<'EOF'
## Summary

- Add `--teacher <model>` flag to `generate_engram_table.py`
- Projects next-token distributions through pre-trained embedding matrix + PCA instead of random JL
- Each table entry becomes the probability-weighted centroid of next-tokens in the teacher's semantic space
- 5 new tests, all existing pass

### How it works

Same corpus scan as PR #213, but the projection step changes:
1. `probs @ embed_matrix` → `[entries, 4096]` (centroid in Mistral space)
2. PCA → `[entries, 128]` (most informative 128 directions)
3. Unit normalize, save

### Usage

```bash
python -m ct87.generate_engram_table \
    --corpus <tokenized-data> \
    --teacher mistralai/Mistral-7B-v0.1 \
    --entries 10000 --dim 128 --output-dir engram_teacher

# Train with it (same as always)
python -m ct87.train --config tiny \
    --data <data> --val-data <val> \
    --engram-table engram_teacher/engram_table.safetensors
```

### Context

Step 1b of the engram table quality roadmap (ZEB-102). Tests whether semantically rich projection improves over random JL projection (PR #213) and the random table baseline (-0.49%).

## Test plan

- [ ] All new + existing tests pass
- [ ] KRILE generates teacher table and trains with it
- [ ] Compare: teacher table vs corpus table vs random table vs baseline

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```
