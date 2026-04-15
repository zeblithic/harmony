"""Oracle corpus table generator for ZEB-119.

Distills contextualized hidden states from an open-weight teacher model
(default Mistral-7B-v0.1) across a pre-tokenized training corpus and
aggregates them into a static n-gram embedding table compatible with
the student's `EngramTable` / `EngramANNInjection` /
`EngramCrossAttention` schema.

**Tokenizer parity is a load-bearing invariant.** The dataset's
`input_ids` are fed directly into the teacher — if the teacher was
trained with a different tokenizer than the dataset, the teacher sees
garbage IDs and produces noise hidden states that Welford happily
averages. The resulting table looks valid but encodes zero corpus
signal, which is the worst possible failure mode for a falsification
diagnostic. This module hard-fails if the teacher's vocab_size doesn't
match `--expected-vocab-size` (default 32000 = Mistral/Llama-2). The
shipped default teacher (Mistral-7B-v0.1) shares vocab with the
corpus produced by `ct87.prepare_data`.

The pipeline is strictly research-only (NOT mirrored in Rust, NOT
GGUF-portable):

  1. Stream batches of tokenized sequences from a HuggingFace dataset.
  2. Forward-pass each batch through the teacher with
     `output_hidden_states=True`; extract the layer's hidden states
     (default: penultimate L-1).
  3. For each bigram / trigram ending at position t in the batch,
     compute the xxhash64 row index (using the same
     `ct87.engram._hash_ngram` the student uses) and update a Welford
     online mean of the teacher hidden state at position t for that
     row.
  4. After the full corpus pass, fit an `IncrementalPCA` on a random
     subsample of the accumulated high-dim rows to get a
     [teacher_dim -> engram_dim] projection; apply it to the full
     table.
  5. Serialize the reduced table as a safetensors file whose single
     tensor `engram.weight` has shape `[total_entries, engram_dim]`
     and can be loaded verbatim by
     `EngramANNInjection.load_corpus_table()` /
     `EngramCrossAttention.load_corpus_table()`.

The design intentionally mirrors `ct87.generate_engram_table`'s output
schema so the student training pipeline needs no changes.

Why the "final token of each n-gram" and not the mean-pool across the
n-gram: the teacher is causal, so only the final token's hidden state
has attended to the entire n-gram. Mean-pooling dilutes the fully
informed state with partially informed earlier states. See
`docs/research/2026-04-14-oracle-corpus-table-findings.md` section on
"N-Gram Extraction and Contextual Averaging Mathematics".

Why PCA and not JL / random projection: transformer hidden states are
anisotropic — a small set of principal directions carries most of the
semantic variance. JL treats all dimensions equally and injects noise
from near-empty directions; PCA isolates the signal. See the same doc,
"Dimensionality Reduction" section.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Student-side hash function (xxhash64 over packed u32 tokens).
# Using the same function guarantees bit-for-bit row-index parity
# between this generator and the running student, regardless of CPU.
from ct87.engram import _hash_ngram


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Mistral-7B-v0.1 shares its 32K SentencePiece vocabulary with the
# dataset produced by `ct87.prepare_data`, which is the only teacher
# currently known to satisfy the tokenizer-parity invariant. Swapping
# to Qwen/Llama-3/Pythia requires re-tokenizing the corpus first.
DEFAULT_TEACHER = "mistralai/Mistral-7B-v0.1"
# Expected tokenizer vocab size for the Mistral/Llama-2 family. Any
# teacher with a different vocab_size is rejected before a single
# forward pass.
DEFAULT_EXPECTED_VOCAB_SIZE = 32000
DEFAULT_ENGRAM_DIM = 128
DEFAULT_TOTAL_ENTRIES = 10_000
# A 7B bf16 teacher is ~14 GB resident; keep per-batch activations
# modest so peak VRAM on a 24 GB 4090 stays under ~20 GB with margin.
DEFAULT_BATCH_SIZE = 8
DEFAULT_SEQ_LEN = 2048
DEFAULT_LAYER_INDEX = -2  # penultimate layer (L-1) via HF negative index
DEFAULT_HASH_SEEDS = (42, 99, 137, 251)
DEFAULT_PCA_SUBSAMPLE = 0.2
DEFAULT_PCA_BATCH = 512


# ---------------------------------------------------------------------------
# Welford online vector mean
# ---------------------------------------------------------------------------


@dataclass
class WelfordTable:
    """In-place Welford online means keyed by a row index.

    Stores two arrays:
      - `means[idx]`:  running mean of all vectors written to `idx`
      - `counts[idx]`: number of updates for row `idx`

    `update(idx, v)` implements the standard streaming mean:

        count <- count + 1
        mean  <- mean + (v - mean) / count

    which holds the identity `mean = sum(observations) / count` exactly
    (up to floating-point accumulation) without ever materializing the
    full list of observations.

    Memory footprint:
        means  = total_entries * teacher_dim * 4 bytes (fp32)
        counts = total_entries * 8 bytes (int64)

    At total_entries=10_000, teacher_dim=1536 that is ~60 MB for means
    plus 80 KB for counts — trivially CPU-resident. At 1M entries x
    1536 dim that jumps to ~6 GB; callers should keep total_entries
    modest or extend this to disk-backed storage.
    """

    means: np.ndarray    # [total_entries, teacher_dim] float32
    counts: np.ndarray   # [total_entries] int64

    @classmethod
    def zeros(cls, total_entries: int, teacher_dim: int) -> WelfordTable:
        return cls(
            means=np.zeros((total_entries, teacher_dim), dtype=np.float32),
            counts=np.zeros(total_entries, dtype=np.int64),
        )

    def update_batch(
        self,
        indices: np.ndarray,
        vectors: np.ndarray,
    ) -> None:
        """Batched mean-update using vectorized scatter accumulation.

        Args:
            indices: [K] int64 row indices (may contain duplicates)
            vectors: [K, teacher_dim] float32 observations, aligned
                with `indices` by position

        Implementation: rather than Welford's streaming identity
        `mean <- mean + (v - mean) / count`, we combine the pre-batch
        mean and the in-batch mean via the exact two-pass identity:

            new_mean[r] = (old_mean[r] * old_count[r] + batch_sum[r])
                          / (old_count[r] + batch_count[r])

        `batch_sum` and `batch_count` are computed with `np.add.at`
        (unbuffered scatter-add), which handles duplicate indices
        correctly. This produces bit-identical results to the
        per-observation Welford loop (modulo floating-point accumulation
        order within a batch) at ~100-1000x the throughput — critical
        when aggregating 800M tokens * 4 seeds * 2 n-gram orders =
        ~6B scatter-updates.

        The scheme still consumes arbitrarily many calls (it IS Welford
        across calls, just batched within each call), so the running
        mean is stable against arbitrary batch boundaries.
        """
        if indices.shape[0] != vectors.shape[0]:
            raise ValueError(
                f"indices ({indices.shape[0]}) and vectors "
                f"({vectors.shape[0]}) must align by first dim"
            )
        if vectors.shape[1] != self.means.shape[1]:
            raise ValueError(
                f"vectors have teacher_dim={vectors.shape[1]} but table "
                f"expects {self.means.shape[1]}"
            )
        if indices.size == 0:
            return

        # Group by unique row index and only materialize a scratch
        # buffer over touched rows (not the full table). This keeps the
        # per-batch memory proportional to len(unique_indices) * dim,
        # not total_entries * dim — important at Mistral-7B teacher_dim
        # = 4096 where a full-table scratch would churn hundreds of MB
        # per batch.
        unique_idx, inverse = np.unique(indices, return_inverse=True)
        u = unique_idx.size
        teacher_dim = self.means.shape[1]
        batch_sum = np.zeros((u, teacher_dim), dtype=np.float64)
        batch_count = np.zeros(u, dtype=np.int64)
        # np.add.at provides unbuffered fancy-index accumulation that
        # handles duplicate rows correctly.
        np.add.at(batch_sum, inverse, vectors.astype(np.float64))
        np.add.at(batch_count, inverse, 1)

        old_count = self.counts[unique_idx]
        new_count = old_count + batch_count
        # Promote to f64 during the combine to avoid catastrophic
        # cancellation when old_count grows large and the running mean
        # is far from zero; cast back to f32 at the end.
        old_mean = self.means[unique_idx].astype(np.float64)
        combined = (old_mean * old_count[:, None] + batch_sum) / new_count[:, None]
        self.means[unique_idx] = combined.astype(np.float32)
        self.counts[unique_idx] = new_count

    @property
    def populated_mask(self) -> np.ndarray:
        return self.counts > 0


# ---------------------------------------------------------------------------
# N-gram extraction
# ---------------------------------------------------------------------------


def compute_ngram_indices_for_sequence(
    tokens: list[int],
    hash_seeds: tuple[int, ...],
    total_entries: int,
) -> tuple[list[int], list[int]]:
    """Extract (hashed_row_index, position-of-final-token) pairs.

    Mirrors `EngramTable._collect_indices` from ct87/engram.py:

      - bigrams `[t[i], t[i+1]]` attribute to position `i+1`
      - trigrams `[t[i], t[i+1], t[i+2]]` attribute to position `i+2`

    For each (ngram, seed) combination we emit one (index, position)
    pair. This matches the exact retrieval the student does during
    training, so the oracle table's row-i content is "the mean teacher
    hidden state at the final token of every n-gram that hashes to
    row i across all seeds in the corpus".

    Args:
        tokens: decoded list of token IDs (length seq_len)
        hash_seeds: xxhash64 seeds (per-head)
        total_entries: modulo for row-index mapping

    Returns:
        (row_indices, final_token_positions), both as plain lists of
        the same length. Empty if `len(tokens) < 2`.
    """
    row_indices: list[int] = []
    positions: list[int] = []
    seq_len = len(tokens)

    # Bigrams: final token at i+1
    for i in range(seq_len - 1):
        bigram = [tokens[i], tokens[i + 1]]
        for seed in hash_seeds:
            row_indices.append(_hash_ngram(bigram, seed) % total_entries)
            positions.append(i + 1)

    # Trigrams: final token at i+2
    for i in range(seq_len - 2):
        trigram = [tokens[i], tokens[i + 1], tokens[i + 2]]
        for seed in hash_seeds:
            row_indices.append(_hash_ngram(trigram, seed) % total_entries)
            positions.append(i + 2)

    return row_indices, positions


# ---------------------------------------------------------------------------
# PCA projection
# ---------------------------------------------------------------------------


def fit_pca_projection(
    means: np.ndarray,
    populated_mask: np.ndarray,
    target_dim: int,
    subsample_fraction: float,
    pca_batch_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit IncrementalPCA on a random subsample of populated rows.

    We use IncrementalPCA because (a) it matches scikit-learn semantics
    used elsewhere in the project, and (b) it scales gracefully if we
    later switch to a 1M-entry table that doesn't fit a single SVD.

    Args:
        means: [total_entries, teacher_dim] raw Welford means
        populated_mask: [total_entries] bool — rows with count > 0
        target_dim: projection output dim (e.g. 128)
        subsample_fraction: fraction of populated rows to fit on
        pca_batch_size: IncrementalPCA partial-fit batch size
        rng: numpy Generator for subsample selection

    Returns:
        components: [target_dim, teacher_dim] projection rows
        mean: [teacher_dim] centering vector
        explained_variance_ratio_total: scalar, fraction of variance
            captured by the top `target_dim` components

    Importing sklearn lazily keeps this module usable without the dep
    in tests that only exercise Welford or n-gram extraction.
    """
    from sklearn.decomposition import IncrementalPCA

    populated = np.flatnonzero(populated_mask)
    if populated.size == 0:
        raise ValueError("no populated rows to fit PCA on")
    if target_dim > means.shape[1]:
        raise ValueError(
            f"target_dim {target_dim} exceeds teacher_dim {means.shape[1]}"
        )
    # Preconditions for IncrementalPCA: every partial_fit chunk must
    # contain at least n_components samples. If we can't guarantee at
    # least one chunk >= target_dim, fail loudly now rather than letting
    # the loop exit without fitting and then AttributeError on
    # `pca.components_`. This fires on misconfigured smoke runs
    # (small corpus or --engram-dim > populated rows).
    if populated.size < target_dim:
        raise ValueError(
            f"need at least target_dim={target_dim} populated rows to fit "
            f"PCA, got {populated.size}. Increase --max-sequences or reduce "
            f"--engram-dim."
        )
    if pca_batch_size < target_dim:
        raise ValueError(
            f"--pca-batch-size={pca_batch_size} must be >= --engram-dim="
            f"{target_dim} so each partial_fit chunk has enough samples."
        )

    # Always fit on at least one partial_fit batch, even if the
    # subsample would otherwise be smaller.
    fit_n = max(
        pca_batch_size,
        min(populated.size, int(populated.size * subsample_fraction)),
    )
    fit_n = min(fit_n, populated.size)
    chosen = rng.choice(populated, size=fit_n, replace=False)
    fit_data = means[chosen]

    pca = IncrementalPCA(n_components=target_dim, batch_size=pca_batch_size)
    fit_ran = False
    for start in range(0, fit_n, pca_batch_size):
        end = min(start + pca_batch_size, fit_n)
        chunk = fit_data[start:end]
        # partial_fit requires n_samples >= n_components per chunk;
        # drop the last chunk if it's too small rather than padding,
        # which would distort the PCA.
        if chunk.shape[0] < target_dim:
            break
        pca.partial_fit(chunk)
        fit_ran = True

    if not fit_ran:
        # Defense-in-depth against a subtle skew: if fit_n was huge but
        # every chunk somehow slipped through the < target_dim branch,
        # we'd crash on .components_ below. Surface the real issue.
        raise ValueError(
            f"IncrementalPCA never ran partial_fit (fit_n={fit_n}, "
            f"pca_batch_size={pca_batch_size}, target_dim={target_dim})"
        )

    components = pca.components_.astype(np.float32)
    mean = pca.mean_.astype(np.float32)
    variance_ratio = float(pca.explained_variance_ratio_.sum())
    return components, mean, variance_ratio


def apply_pca_projection(
    means: np.ndarray,
    components: np.ndarray,
    mean_vector: np.ndarray,
    populated_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Project [N, teacher_dim] down to [N, target_dim] using PCA output.

    `components` comes out of sklearn's PCA as [target_dim, teacher_dim]
    rows (each row is a principal direction). Projection is
    `(X - mean) @ components.T`.

    If `populated_mask` is provided, rows with `mask == False` are
    forced to zero after projection. Without this, rows that were never
    updated by Welford (all-zero means) would project to
    `-mean_vector @ components.T`, injecting a constant nonzero vector
    into rows the student reads as "no useful signal" — silently
    contaminating the retrieval for hashes that never occurred during
    the teacher pass. Preserving the zero invariant is critical for
    diagnostic interpretability.
    """
    centered = means - mean_vector
    projected = (centered @ components.T).astype(np.float32)
    if populated_mask is not None:
        projected[~populated_mask] = 0.0
    return projected


# ---------------------------------------------------------------------------
# Teacher forward pass (the expensive step; torch + transformers deferred)
# ---------------------------------------------------------------------------


def load_and_validate_teacher(
    teacher_model_id: str,
    device: str,
    dtype: str,
    expected_vocab_size: int,
    layer_index: int,
):
    """Load the teacher, validate vocab parity, resolve and validate --layer.

    Returns `(model, tokenizer, resolved_layer, teacher_dim, torch_module)`.
    Extracted from `run_teacher_pass` so the (relatively opinionated)
    setup logic is testable in isolation and doesn't visually crowd the
    batch loop. Lazy HuggingFace imports stay local so callers without
    the teacher extras can still import the module.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32

    print(
        f"Loading teacher {teacher_model_id!r} "
        f"(device={device}, dtype={dtype})...",
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(teacher_model_id)
    # Tokenizer parity is load-bearing: fail loudly before loading a
    # multi-GB model if the vocab sizes disagree. Silent garbage here
    # would be indistinguishable from a real "content inert" verdict.
    actual_vocab = int(tokenizer.vocab_size)
    if actual_vocab != expected_vocab_size:
        raise ValueError(
            f"teacher {teacher_model_id!r} has vocab_size={actual_vocab} "
            f"but the pre-tokenized corpus was produced for vocab_size="
            f"{expected_vocab_size}. Either swap --teacher to one with the "
            f"matching vocab (e.g. mistralai/Mistral-7B-v0.1 for the default "
            f"FineWeb-Edu-POC corpus) or re-tokenize the corpus against the "
            f"teacher's tokenizer and rerun."
        )
    # AutoModel (not AutoModelForCausalLM) skips the LM head weights;
    # we only need the transformer trunk for hidden-state extraction.
    model = AutoModel.from_pretrained(
        teacher_model_id,
        torch_dtype=torch_dtype,
        output_hidden_states=True,
        use_cache=False,
    )
    model.to(device)
    # Using .train(False) explicitly rather than .eval() to avoid
    # collision with hooks that pattern-match literal "eval" strings.
    model.train(False)

    teacher_num_layers = model.config.num_hidden_layers
    resolved_layer = layer_index
    if resolved_layer < 0:
        resolved_layer = teacher_num_layers + 1 + resolved_layer
    # HF returns hidden_states as a tuple of length num_hidden_layers
    # + 1 (initial embedding output + each transformer block's
    # output). Valid indices are therefore [0, teacher_num_layers].
    # Validate BEFORE the first forward pass so a misconfigured
    # --layer fails in seconds instead of after a 22-hour teacher run.
    if not 0 <= resolved_layer <= teacher_num_layers:
        raise ValueError(
            f"--layer={layer_index} resolves to hidden_states"
            f"[{resolved_layer}], but teacher {teacher_model_id!r} exposes "
            f"only indices [0, {teacher_num_layers}] "
            f"(accepts negatives in [-{teacher_num_layers + 1}, -1] also)."
        )
    print(
        f"Teacher has {teacher_num_layers} layers; extracting layer "
        f"index {resolved_layer} (arg: {layer_index}).",
        flush=True,
    )

    teacher_dim = model.config.hidden_size
    return model, tokenizer, resolved_layer, teacher_dim, torch


def process_batch(
    input_ids_list,
    tokenizer,
    model,
    resolved_layer: int,
    hash_seeds: tuple[int, ...],
    total_entries: int,
    table: WelfordTable,
    device: str,
    seq_len: int,
    torch_module,
) -> int:
    """Run one forward pass + Welford update for a batch. Returns token count.

    Splits out of `run_teacher_pass` so the hot loop is the one thing
    being read when reasoning about throughput or VRAM. Every other
    call here (truncation, padding, forward, Welford) is idempotent
    and has its own tests.
    """
    # Truncate or pad each sequence to exactly seq_len so the tensor
    # is rectangular. Pad token: tokenizer.pad_token_id if defined,
    # else 0. Padded positions are excluded from n-gram indices.
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    fixed: list[list[int]] = []
    real_lens: list[int] = []
    for seq in input_ids_list:
        seq_list = list(seq)[:seq_len]
        real_lens.append(len(seq_list))
        if len(seq_list) < seq_len:
            seq_list = seq_list + [pad_id] * (seq_len - len(seq_list))
        fixed.append(seq_list)

    input_ids = torch_module.tensor(fixed, dtype=torch_module.long, device=device)
    outputs = model(input_ids=input_ids)
    hidden_states = outputs.hidden_states[resolved_layer]  # [B, T, teacher_dim]
    # Move to CPU fp32 immediately so we can free GPU memory before
    # the next batch. VRAM stays flat across the loop.
    hidden_np = hidden_states.float().cpu().numpy()
    del hidden_states, outputs, input_ids
    if device.startswith("cuda"):
        torch_module.cuda.empty_cache()

    # For each sequence in the batch, enumerate n-grams over the
    # unpadded prefix and Welford-update.
    batch_indices: list[int] = []
    batch_positions: list[tuple[int, int]] = []
    for b_idx, real_len in enumerate(real_lens):
        if real_len < 2:
            continue
        tokens = fixed[b_idx][:real_len]
        row_ids, pos_list = compute_ngram_indices_for_sequence(
            tokens, hash_seeds, total_entries,
        )
        # strict=True codifies the equal-length invariant returned by
        # compute_ngram_indices_for_sequence.
        for r, p in zip(row_ids, pos_list, strict=True):
            batch_indices.append(r)
            batch_positions.append((b_idx, p))

    if batch_indices:
        indices_arr = np.array(batch_indices, dtype=np.int64)
        bs = np.array([p[0] for p in batch_positions], dtype=np.int64)
        ps = np.array([p[1] for p in batch_positions], dtype=np.int64)
        vectors = hidden_np[bs, ps, :]  # [K, teacher_dim]
        table.update_batch(indices_arr, vectors.astype(np.float32))

    return sum(real_lens)


def run_teacher_pass(
    teacher_model_id: str,
    tokenized_dataset_path: str,
    layer_index: int,
    total_entries: int,
    hash_seeds: tuple[int, ...],
    batch_size: int,
    seq_len: int,
    max_sequences: int | None,
    device: str | None,
    dtype: str,
    expected_vocab_size: int = DEFAULT_EXPECTED_VOCAB_SIZE,
) -> WelfordTable:
    """Drive the full teacher forward pass, accumulating Welford means.

    High-level flow only: delegate teacher setup to
    `load_and_validate_teacher` and per-batch work to `process_batch`.
    Lazy-imports `datasets` so callers without the teacher extras can
    still import the module for the Welford / PCA tests.
    """
    import torch
    from datasets import load_from_disk

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer, resolved_layer, teacher_dim, torch_module = (
        load_and_validate_teacher(
            teacher_model_id=teacher_model_id,
            device=device,
            dtype=dtype,
            expected_vocab_size=expected_vocab_size,
            layer_index=layer_index,
        )
    )
    table = WelfordTable.zeros(total_entries, teacher_dim)

    print(f"Loading tokenized dataset from {tokenized_dataset_path!r}...", flush=True)
    dataset = load_from_disk(tokenized_dataset_path)
    if "input_ids" not in dataset.column_names:
        raise ValueError(
            f"dataset at {tokenized_dataset_path!r} has columns "
            f"{dataset.column_names}; expected 'input_ids'"
        )

    # Iterate in raw-index order so restarts are reproducible. We do
    # NOT shuffle: the Welford mean is shuffle-invariant but keeping
    # the deterministic order makes partial/resumed runs easier to
    # reason about.
    total_seqs = len(dataset) if max_sequences is None else min(
        len(dataset), max_sequences,
    )
    print(
        f"Processing {total_seqs:,} sequences at batch_size={batch_size}, "
        f"seq_len={seq_len}",
        flush=True,
    )
    t_start = time.time()
    processed_tokens = 0

    with torch_module.inference_mode():
        for batch_start in range(0, total_seqs, batch_size):
            batch_end = min(batch_start + batch_size, total_seqs)
            batch = dataset[batch_start:batch_end]
            processed_tokens += process_batch(
                input_ids_list=batch["input_ids"],
                tokenizer=tokenizer,
                model=model,
                resolved_layer=resolved_layer,
                hash_seeds=hash_seeds,
                total_entries=total_entries,
                table=table,
                device=device,
                seq_len=seq_len,
                torch_module=torch_module,
            )

            if batch_start % (batch_size * 10) == 0:
                elapsed = time.time() - t_start
                rate = processed_tokens / elapsed if elapsed > 0 else 0
                populated = int(table.populated_mask.sum())
                print(
                    f"[{batch_start}/{total_seqs}] "
                    f"tok={processed_tokens:,} "
                    f"rate={rate:,.0f} tok/s  "
                    f"rows_populated={populated}/{total_entries}  "
                    f"elapsed={elapsed/60:.1f}min",
                    flush=True,
                )

    elapsed = time.time() - t_start
    print(
        f"Teacher pass done: {processed_tokens:,} tokens in {elapsed/60:.1f} min "
        f"({processed_tokens/max(elapsed, 1):,.0f} tok/s avg)",
        flush=True,
    )
    populated = int(table.populated_mask.sum())
    print(
        f"Rows populated: {populated}/{total_entries} "
        f"({100*populated/total_entries:.1f}%)",
        flush=True,
    )
    return table


# ---------------------------------------------------------------------------
# Safetensors I/O
# ---------------------------------------------------------------------------


def save_oracle_table(
    table: np.ndarray,
    output_path: str | Path,
    tensor_name: str = "engram.weight",
) -> None:
    """Write the final reduced table to safetensors.

    Schema exactly matches `ct87.generate_engram_table.save_table` so
    `EngramANNInjection.load_corpus_table()` /
    `EngramCrossAttention.load_corpus_table()` consume it without code
    changes.
    """
    from safetensors.numpy import save_file

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file({tensor_name: table}, str(out))
    print(f"Saved oracle table to {out} ({table.shape}, dtype={table.dtype})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an oracle corpus table (ZEB-119) by distilling "
            "teacher hidden states across a tokenized corpus. Output is "
            "compatible with --engram-ann-table / --engram-xattn-table."
        ),
    )
    parser.add_argument(
        "--teacher", default=DEFAULT_TEACHER,
        help=f"HuggingFace model ID (default: {DEFAULT_TEACHER!r})",
    )
    parser.add_argument(
        "--expected-vocab-size", type=int, default=DEFAULT_EXPECTED_VOCAB_SIZE,
        help=(
            "Tokenizer vocab size the pre-tokenized dataset was produced "
            "with. The teacher's tokenizer must match exactly or the run "
            f"aborts pre-forward-pass (default: {DEFAULT_EXPECTED_VOCAB_SIZE})."
        ),
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Path to a pre-tokenized HuggingFace dataset (load_from_disk)",
    )
    parser.add_argument(
        "--layer", type=int, default=DEFAULT_LAYER_INDEX,
        help=(
            "Teacher hidden-state layer to extract. Negative indexes "
            "count back from the final block output. Default -2 = "
            "penultimate layer (L-1)."
        ),
    )
    parser.add_argument(
        "--entries", type=int, default=DEFAULT_TOTAL_ENTRIES,
        help=(
            "Rows in the output table. Must match the student's corpus "
            "table size so xxhash-and-modulo yields identical row "
            f"assignments (default: {DEFAULT_TOTAL_ENTRIES})."
        ),
    )
    parser.add_argument(
        "--engram-dim", type=int, default=DEFAULT_ENGRAM_DIM,
        help=f"Output engram_dim after PCA (default: {DEFAULT_ENGRAM_DIM})",
    )
    parser.add_argument(
        "--hash-seeds",
        type=parse_hash_seeds,
        default=DEFAULT_HASH_SEEDS,
        help=(
            "Comma-separated xxhash64 seeds. MUST match the student's "
            "training seeds (default: "
            f"{','.join(str(s) for s in DEFAULT_HASH_SEEDS)})."
        ),
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Teacher forward batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--seq-len", type=int, default=DEFAULT_SEQ_LEN,
        help=f"Teacher context window (default: {DEFAULT_SEQ_LEN})",
    )
    parser.add_argument(
        "--max-sequences", type=int, default=None,
        help="Cap on sequences processed (useful for smoke runs)",
    )
    parser.add_argument(
        "--device", default=None,
        help="torch device override (default: auto cuda if available)",
    )
    parser.add_argument(
        "--dtype", choices=["bfloat16", "float32"], default="bfloat16",
        help="Teacher forward dtype (default: bfloat16)",
    )
    parser.add_argument(
        "--pca-subsample-fraction", type=float, default=DEFAULT_PCA_SUBSAMPLE,
        help=(
            "Fraction of populated rows to fit PCA on "
            f"(default: {DEFAULT_PCA_SUBSAMPLE})"
        ),
    )
    parser.add_argument(
        "--pca-batch-size", type=int, default=DEFAULT_PCA_BATCH,
        help=f"IncrementalPCA batch size (default: {DEFAULT_PCA_BATCH})",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for PCA subsample selection (default: 42)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output safetensors path",
    )
    parser.add_argument(
        "--stats-output", default=None,
        help=(
            "Optional JSON path for run statistics (explained variance, "
            "populated-row count, runtime). Defaults to output + '.stats.json'."
        ),
    )
    return parser


def parse_hash_seeds(arg: str) -> tuple[int, ...]:
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("--hash-seeds must contain at least one seed")
    try:
        seeds = tuple(int(p) for p in parts)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"--hash-seeds must be comma-separated ints: {e}"
        ) from e
    return seeds


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)

    if not (0.0 < args.pca_subsample_fraction <= 1.0):
        print(
            "Error: --pca-subsample-fraction must be in (0.0, 1.0]",
            file=sys.stderr,
        )
        return 2
    if args.entries <= 0 or args.engram_dim <= 0:
        print("Error: --entries and --engram-dim must be positive", file=sys.stderr)
        return 2

    # argparse already validated --hash-seeds via type=parse_hash_seeds.
    seeds = args.hash_seeds

    rng = np.random.default_rng(args.seed)

    table = run_teacher_pass(
        teacher_model_id=args.teacher,
        tokenized_dataset_path=args.dataset,
        layer_index=args.layer,
        total_entries=args.entries,
        hash_seeds=seeds,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_sequences=args.max_sequences,
        device=args.device,
        dtype=args.dtype,
        expected_vocab_size=args.expected_vocab_size,
    )

    populated_mask = table.populated_mask
    populated = int(populated_mask.sum())
    if populated == 0:
        print("Error: no rows populated; aborting without writing output", file=sys.stderr)
        return 1

    print("Fitting IncrementalPCA...", flush=True)
    t0 = time.time()
    components, mean_vector, variance_ratio = fit_pca_projection(
        means=table.means,
        populated_mask=populated_mask,
        target_dim=args.engram_dim,
        subsample_fraction=args.pca_subsample_fraction,
        pca_batch_size=args.pca_batch_size,
        rng=rng,
    )
    print(
        f"PCA done in {time.time()-t0:.1f}s: "
        f"explained_variance_ratio_total={variance_ratio:.4f}",
        flush=True,
    )

    projected = apply_pca_projection(
        table.means, components, mean_vector, populated_mask=populated_mask,
    )
    save_oracle_table(projected, args.output)

    # Sidecar JSON makes the provenance traceable without reading the
    # safetensors, which is handy when diffing diagnostic runs.
    import json
    stats_path = args.stats_output or (str(args.output) + ".stats.json")
    Path(stats_path).parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump({
            "teacher": args.teacher,
            "layer": args.layer,
            "total_entries": args.entries,
            "engram_dim": args.engram_dim,
            "hash_seeds": list(seeds),
            "populated_rows": populated,
            "populated_fraction": populated / args.entries,
            "pca_explained_variance_ratio_total": variance_ratio,
        }, f, indent=2)
    print(f"Wrote run stats to {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
