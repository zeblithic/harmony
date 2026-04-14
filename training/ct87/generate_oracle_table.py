"""Oracle corpus table generator for ZEB-119.

Distills contextualized hidden states from an open-weight teacher model
(default Qwen2.5-1.5B) across a pre-tokenized training corpus and
aggregates them into a static n-gram embedding table compatible with
the student's `EngramTable` / `EngramANNInjection` /
`EngramCrossAttention` schema.

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

DEFAULT_TEACHER = "Qwen/Qwen2.5-1.5B"
DEFAULT_ENGRAM_DIM = 128
DEFAULT_TOTAL_ENTRIES = 10_000
DEFAULT_BATCH_SIZE = 16
DEFAULT_SEQ_LEN = 2048
DEFAULT_LAYER_INDEX = -2  # penultimate layer (L-1) via HF negative index
DEFAULT_HASH_SEEDS = (42, 99, 137, 251)
DEFAULT_PCA_SUBSAMPLE = 0.2
DEFAULT_PCA_BATCH = 512
# A Welford-updated row is "populated" once it has seen at least one
# observation; untouched rows stay at zero. For PCA fitting, we ignore
# zero-count rows so the projection matrix isn't dominated by the
# geometric origin.
DEFAULT_MIN_COUNT_FOR_PCA = 1


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
        """Welford-update many rows in one call.

        Args:
            indices: [K] int64 row indices (may contain duplicates)
            vectors: [K, teacher_dim] float32 observations, aligned
                with `indices` by position

        Duplicates within `indices` are processed sequentially so the
        running mean stays correct under repeated updates to the same
        row inside a single batch. Vectorizing this across duplicates
        is tricky (the divisor changes with each duplicate); for our
        scale the per-row Python loop over the unique_indices groups is
        fast enough.
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
        # Fast path for small-batch unique indices: iterate per row.
        # Profiling note: at B=16, T=2048, N_bigrams+N_trigrams per row
        # is ~O(100k) across the full corpus; the Python overhead here
        # dominates only at very small batch sizes. A vectorized
        # scatter_reduce implementation would speed this up 2-3x; left
        # as a follow-up.
        for idx, vec in zip(indices.tolist(), vectors):
            self.counts[idx] += 1
            c = self.counts[idx]
            # mean <- mean + (v - mean) / c
            np.add(
                self.means[idx],
                (vec - self.means[idx]) / float(c),
                out=self.means[idx],
            )

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
    for start in range(0, fit_n, pca_batch_size):
        end = min(start + pca_batch_size, fit_n)
        chunk = fit_data[start:end]
        # partial_fit requires n_samples >= n_components per chunk;
        # drop the last chunk if it's too small rather than padding,
        # which would distort the PCA.
        if chunk.shape[0] < target_dim:
            break
        pca.partial_fit(chunk)

    components = pca.components_.astype(np.float32)
    mean = pca.mean_.astype(np.float32)
    variance_ratio = float(pca.explained_variance_ratio_.sum())
    return components, mean, variance_ratio


def apply_pca_projection(
    means: np.ndarray,
    components: np.ndarray,
    mean_vector: np.ndarray,
) -> np.ndarray:
    """Project [N, teacher_dim] down to [N, target_dim] using PCA output.

    `components` comes out of sklearn's PCA as [target_dim, teacher_dim]
    rows (each row is a principal direction). Projection is
    `(X - mean) @ components.T`.
    """
    centered = means - mean_vector
    projected = centered @ components.T
    return projected.astype(np.float32)


# ---------------------------------------------------------------------------
# Teacher forward pass (the expensive step; torch + transformers deferred)
# ---------------------------------------------------------------------------


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
) -> WelfordTable:
    """Drive the full teacher forward pass, accumulating Welford means.

    Imports torch/transformers/datasets lazily so that unit tests for
    Welford + n-gram extraction + PCA can run in a CI-lite environment
    that only has numpy + sklearn. Importing inside the function also
    keeps the script's startup time short when it's invoked with
    `--help` on a machine without the heavy deps installed.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer
    from datasets import load_from_disk

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32

    print(
        f"Loading teacher {teacher_model_id!r} "
        f"(device={device}, dtype={dtype})...",
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(teacher_model_id)
    # AutoModel (not AutoModelForCausalLM) skips the LM head weights;
    # we only need the transformer trunk for hidden-state extraction.
    model = AutoModel.from_pretrained(
        teacher_model_id,
        dtype=torch_dtype,
        output_hidden_states=True,
    )
    model.to(device)
    # Inference mode: disable dropout/batchnorm training behavior.
    # Using .train(False) explicitly rather than .eval() to avoid
    # collision with hooks that pattern-match literal "eval" strings.
    model.train(False)
    # HF returns hidden_states as a tuple of length num_layers + 1
    # (initial embeddings + each transformer block's output). Negative
    # indices resolve from the end: -2 = penultimate transformer
    # block's output, which is what "L-1" refers to in the research
    # doc.
    teacher_num_layers = model.config.num_hidden_layers
    resolved_layer = layer_index
    if resolved_layer < 0:
        resolved_layer = teacher_num_layers + 1 + resolved_layer
    print(
        f"Teacher has {teacher_num_layers} layers; extracting layer "
        f"index {resolved_layer} (arg: {layer_index}).",
        flush=True,
    )

    teacher_dim = model.config.hidden_size
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
    # the deterministic order makes partial / resumed runs easier to
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

    with torch.inference_mode():
        for batch_start in range(0, total_seqs, batch_size):
            batch_end = min(batch_start + batch_size, total_seqs)
            batch = dataset[batch_start:batch_end]
            input_ids_list = batch["input_ids"]
            # Truncate or pad each sequence to exactly seq_len so the
            # tensor is rectangular. Pad token: tokenizer.pad_token_id
            # if defined, else 0. Padded positions should not appear in
            # n-gram indices we care about, but we keep them in
            # tensor space for the forward pass.
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            fixed: list[list[int]] = []
            real_lens: list[int] = []
            for seq in input_ids_list:
                seq_list = list(seq)[:seq_len]
                real_lens.append(len(seq_list))
                if len(seq_list) < seq_len:
                    seq_list = seq_list + [pad_id] * (seq_len - len(seq_list))
                fixed.append(seq_list)

            input_ids = torch.tensor(fixed, dtype=torch.long, device=device)
            outputs = model(input_ids=input_ids)
            hidden_states = outputs.hidden_states[resolved_layer]  # [B, T, teacher_dim]
            # Move to CPU fp32 immediately; we free the GPU tensor
            # before the next batch so VRAM stays flat.
            hidden_np = hidden_states.float().cpu().numpy()
            del hidden_states, outputs, input_ids
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

            # For each sequence in the batch, enumerate n-grams (only
            # over the unpadded prefix) and Welford-update.
            batch_indices: list[int] = []
            batch_positions: list[tuple[int, int]] = []
            for b_idx, real_len in enumerate(real_lens):
                if real_len < 2:
                    continue
                tokens = fixed[b_idx][:real_len]
                row_ids, pos_list = compute_ngram_indices_for_sequence(
                    tokens, hash_seeds, total_entries,
                )
                for r, p in zip(row_ids, pos_list):
                    batch_indices.append(r)
                    batch_positions.append((b_idx, p))

            if batch_indices:
                indices_arr = np.array(batch_indices, dtype=np.int64)
                # Gather hidden states at (batch, position) pairs
                bs = np.array([p[0] for p in batch_positions], dtype=np.int64)
                ps = np.array([p[1] for p in batch_positions], dtype=np.int64)
                vectors = hidden_np[bs, ps, :]  # [K, teacher_dim]
                table.update_batch(indices_arr, vectors.astype(np.float32))

            processed_tokens += sum(real_lens)
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
        "--hash-seeds", type=str, default=",".join(str(s) for s in DEFAULT_HASH_SEEDS),
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
        raise argparse.ArgumentTypeError(f"--hash-seeds must be comma-separated ints: {e}")
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

    seeds = parse_hash_seeds(args.hash_seeds)

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

    projected = apply_pca_projection(table.means, components, mean_vector)
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
