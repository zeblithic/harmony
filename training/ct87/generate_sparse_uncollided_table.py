"""Sparse corpus table generator for ZEB-119 (collision-reduced baseline).

Companion to `generate_oracle_table.py`. Where the primary oracle
averages every hashed n-gram in the corpus into a 10K-row table
(~80,000 distinct n-grams per row at 800M tokens, the dominant source
of semantic collapse), this generator populates the table with
sentence-transformer embeddings of only the top-N most frequent
UNIQUE n-gram TEXT STRINGS.

**Important naming caveat:** the historical module name says
"uncollided" but the emitted table is NOT fully collision-free. We use
the student's xxhash-and-modulo to write each of the top-N n-gram
embeddings into the 10K-row table (option (b) below), so at N=50K and
entries=10K each row still averages ~5 top-N n-gram vectors per seed
× 4 seeds = ~20 writes per row. This is a ~4000x reduction in per-row
collision contamination vs the primary oracle (20 vs 80,000), but it
is a reduction, not elimination. If you want fully-uncollided
semantics you need option (a), which requires a student-side code
path we have not yet implemented (flagged as a follow-up).

If the primary oracle fails but this collision-reduced baseline
succeeds, it's strong but not airtight evidence that hash topology is
the dominant failure mode. A cleaner conclusion would come from
option (a), but option (b) gives us a fast answer with zero
student-side risk.

Options:

  (a) Replace the student's hash-and-modulo with a dictionary lookup
      `ngram_tokens -> row_idx`, with rows that miss the top-N falling
      back to a zero vector (or a sentinel "unknown" row).
  (b) Map the sparse table into a dense 10K-row table via hash (same
      schema as the primary oracle), only writing top-N n-gram
      vectors. Unwritten rows stay at zero. **This is what the
      generator currently produces.**

Option (b) requires no student-side code changes and lets us A/B test
the exact same mechanism; that's what this generator produces.
Option (a) is cleaner from a research-design standpoint but requires
a minor student-side code path; flagged as a follow-up.

Schema: identical to the primary oracle's output
(`[total_entries, engram_dim]` safetensors under `engram.weight`), so
the student can load it verbatim via the same `--engram-ann-table` /
`--engram-xattn-table` flags.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ct87.engram import _hash_ngram
from ct87.generate_oracle_table import parse_hash_seeds, save_oracle_table


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# sentence-transformers/all-MiniLM-L6-v2 is the de-facto standard
# compact text embedder (384-dim, ~22M params, fast on CPU). Apache
# 2.0 licensed, no domain mismatch constraints for this diagnostic.
DEFAULT_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TOP_NGRAMS = 50_000
DEFAULT_TOTAL_ENTRIES = 10_000
DEFAULT_ENGRAM_DIM = 128
DEFAULT_HASH_SEEDS = (42, 99, 137, 251)
# Top-k n-gram frequencies stabilize well before the full corpus — at
# 100K sequences × ~2K tokens ≈ 200M tokens, the top 50K most-frequent
# bigrams/trigrams have already converged against later long-tail
# additions. Capping by default prevents accidental full-corpus scans
# that would balloon Counter memory into the tens of GB. Override with
# --max-sequences 0 to explicitly request a full pass.
DEFAULT_MAX_SEQUENCES = 100_000
DEFAULT_COUNT_BATCH_SIZE = 256


# ---------------------------------------------------------------------------
# N-gram frequency counting
# ---------------------------------------------------------------------------


@dataclass
class NgramFrequencies:
    """Counts of each distinct (tokens, order) n-gram.

    Keyed by `(n, tuple(tokens))` so bigrams and trigrams share a
    single counter namespace. The `n` field lets the later code path
    distinguish them for text reconstruction and hashing.
    """

    counter: Counter  # {(n, (t0, t1, ...)): count}

    def top_k(self, k: int) -> list[tuple[int, tuple[int, ...], int]]:
        """Return [(n, tokens, count)] sorted by count descending."""
        items = []
        for (n, tokens), count in self.counter.most_common(k):
            items.append((n, tokens, count))
        return items


def count_ngrams(
    tokenized_dataset_path: str,
    max_sequences: int | None,
    batch_size: int = DEFAULT_COUNT_BATCH_SIZE,
) -> NgramFrequencies:
    """Single-pass bigram/trigram frequency count over the corpus.

    Skips sequences shorter than 3 tokens. Counts are NOT seed-scaled;
    one n-gram occurrence contributes one increment, regardless of how
    many hash seeds the student uses at training time (the seed loop
    is applied later at row-hash time).

    Throughput notes:
      - Iterates in batches of `batch_size` sequences via dataset
        slicing. Single-item indexing triggers a Python round-trip per
        sequence, which dominates cost at 100K+ sequences; batched
        slicing pulls arrow chunks in one call.
      - Uses `Counter.update(iter(...))` so the increment path is a
        tight C loop per batch instead of a Python `counter[k] += 1`
        loop.
      - For the full default corpus cap (100K sequences) this finishes
        in under 5 minutes on a laptop CPU.

    Memory: Counter keyspace grows with distinct n-grams. At 200M
    tokens we observe ~20–40M distinct n-grams; at ~200 B per Counter
    entry that's ~4–8 GB. If you need a larger scan, use a streaming
    heavy-hitters structure (SpaceSaving / Misra–Gries) instead.
    """
    from datasets import load_from_disk

    dataset = load_from_disk(tokenized_dataset_path)
    if "input_ids" not in dataset.column_names:
        raise ValueError(
            f"dataset at {tokenized_dataset_path!r} has columns "
            f"{dataset.column_names}; expected 'input_ids'"
        )

    counter: Counter = Counter()
    # max_sequences == 0 is an explicit opt-in for full-corpus scan.
    if max_sequences is None or max_sequences == 0:
        total = len(dataset)
    else:
        total = min(len(dataset), max_sequences)
    t_start = time.time()
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch = dataset[batch_start:batch_end]["input_ids"]
        for seq in batch:
            seq_len = len(seq)
            if seq_len < 2:
                continue
            # Local aliases eliminate per-iteration attribute lookups.
            tokens = seq
            # Generator expressions here keep the increment path inside
            # Counter.update's C fast-path; materializing a list would
            # double memory for no gain.
            counter.update(
                (2, (tokens[i], tokens[i + 1]))
                for i in range(seq_len - 1)
            )
            if seq_len >= 3:
                counter.update(
                    (3, (tokens[i], tokens[i + 1], tokens[i + 2]))
                    for i in range(seq_len - 2)
                )
        if batch_start % (batch_size * 8) == 0 and batch_start > 0:
            elapsed = time.time() - t_start
            print(
                f"[{batch_start}/{total}] unique_ngrams={len(counter):,}  "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
    return NgramFrequencies(counter=counter)


# ---------------------------------------------------------------------------
# Text embedding (sentence-transformer)
# ---------------------------------------------------------------------------


def embed_ngram_texts(
    ngram_texts: list[str],
    embedder_model_id: str,
    engram_dim: int,
    device: str | None,
    batch_size: int = 128,
) -> np.ndarray:
    """Encode a list of n-gram text strings to [N, engram_dim] float32.

    If the embedder's native output dim exceeds `engram_dim`, a PCA is
    fit on the full set of embeddings and applied — same rationale as
    the primary oracle's anisotropic-preserving reduction. If native
    dim is less than engram_dim, zero-pads on the right (no
    interpolation so downstream PCA on this table would see real
    structure in the low dims and zeros elsewhere).

    For the default MiniLM-L6 embedder with 384-dim output and
    engram_dim=128, this runs a single PCA fit on all ~50K vectors
    (fast: <1s CPU).
    """
    import torch
    from sentence_transformers import SentenceTransformer

    resolved_device = device
    if resolved_device is None:
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"

    print(
        f"Loading embedder {embedder_model_id!r} (device={resolved_device})...",
        flush=True,
    )
    embedder = SentenceTransformer(embedder_model_id, device=resolved_device)
    native_dim = embedder.get_sentence_embedding_dimension()
    print(
        f"Embedder native dim = {native_dim}, target engram_dim = {engram_dim}",
        flush=True,
    )

    t0 = time.time()
    raw = embedder.encode(
        ngram_texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=False,
    ).astype(np.float32)
    print(f"Embedded {len(ngram_texts):,} n-grams in {time.time()-t0:.1f}s", flush=True)

    if native_dim == engram_dim:
        return raw
    if native_dim < engram_dim:
        padded = np.zeros((raw.shape[0], engram_dim), dtype=np.float32)
        padded[:, :native_dim] = raw
        print(
            f"Padded embeddings right with zeros: {native_dim} -> {engram_dim}",
            flush=True,
        )
        return padded
    # native_dim > engram_dim: PCA-reduce
    from sklearn.decomposition import PCA

    # sklearn's PCA requires n_samples >= n_components; small smoke
    # runs (tiny --max-sequences or --top-ngrams) easily violate this
    # and would raise a confusing ValueError deep in sklearn. Fail
    # loudly with a message that points at the CLI flags to adjust.
    n_samples = raw.shape[0]
    if n_samples < engram_dim:
        raise ValueError(
            f"need at least {engram_dim} n-gram embeddings to PCA-reduce "
            f"into {engram_dim} dimensions, got {n_samples}. Raise "
            f"--top-ngrams or --max-sequences, or lower --engram-dim."
        )

    print(f"PCA-reducing {native_dim} -> {engram_dim}...", flush=True)
    pca = PCA(n_components=engram_dim)
    reduced = pca.fit_transform(raw).astype(np.float32)
    variance = float(pca.explained_variance_ratio_.sum())
    print(f"PCA done: explained_variance_ratio_total={variance:.4f}", flush=True)
    return reduced


# ---------------------------------------------------------------------------
# Hash-bucket assignment (option b: dense table, sparse population)
# ---------------------------------------------------------------------------


def build_sparse_table(
    top_ngrams: list[tuple[int, tuple[int, ...], int]],
    ngram_vectors: np.ndarray,
    hash_seeds: tuple[int, ...],
    total_entries: int,
    engram_dim: int,
) -> tuple[np.ndarray, int, int]:
    """Place top-N n-gram vectors into a dense table via hashes (collisions expected, averaged via Welford).

    For each top-N n-gram we compute its row index under every hash
    seed (matching the student's multi-seed lookup). Write collisions
    still happen — at N=50K and entries=10K the expected occupancy is
    ~5 top-N n-grams per (seed, row) pair — but this is ~4000× lower
    than the primary oracle's full-corpus collision rate. Collided
    writes are combined via Welford's running mean, which is identical
    to averaging and, at this occupancy level, produces vectors that
    are still dominated by the top-N signal rather than diluted into
    noise.

    Rows with zero populations are left at numerical zero, which the
    student's retrieval reads as "no useful signal here" — useful as a
    sentinel when a retrieved key doesn't land in a populated row.

    For a fully collision-free variant, switch to option (a) in the
    module docstring (explicit `ngram -> row` dictionary with a
    student-side dispatch); not yet implemented.

    Returns:
        (table, populated_rows, total_writes):
          table: [total_entries, engram_dim] float32
          populated_rows: count of rows with >= 1 write (<= total_entries)
          total_writes: total number of write operations performed
    """
    table = np.zeros((total_entries, engram_dim), dtype=np.float32)
    counts = np.zeros(total_entries, dtype=np.int64)

    total_writes = 0
    for row_idx, (_n, tokens, _freq) in enumerate(top_ngrams):
        vec = ngram_vectors[row_idx]
        for seed in hash_seeds:
            h = _hash_ngram(list(tokens), seed) % total_entries
            counts[h] += 1
            c = counts[h]
            # Welford online mean for write collisions at the same (h)
            # over multiple seeds or multiple n-grams.
            table[h] = table[h] + (vec - table[h]) / float(c)
            total_writes += 1

    populated = int((counts > 0).sum())
    return table, populated, total_writes


# ---------------------------------------------------------------------------
# Text reconstruction (detokenize n-gram tokens for the embedder)
# ---------------------------------------------------------------------------


def detokenize_ngrams(
    top_ngrams: list[tuple[int, tuple[int, ...], int]],
    tokenizer_model_id: str,
) -> list[str]:
    """Decode each n-gram's tokens back to a text string.

    Uses the SAME tokenizer the student was trained with (inferred
    from the teacher or passed explicitly). Decoded strings feed the
    sentence-transformer embedder, so decoding quality directly
    affects diagnostic signal — a tokenizer mismatch would produce
    garbled inputs and void the diagnostic.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_id)
    texts = []
    for _n, tokens, _freq in top_ngrams:
        # skip_special_tokens keeps BOS/EOS/PAD out of the text;
        # clean_up_tokenization_spaces normalizes BPE whitespace so
        # the embedder sees readable n-grams.
        text = tokenizer.decode(
            list(tokens),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()
        texts.append(text if text else "<empty>")
    return texts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a sparse uncollided corpus table (ZEB-119) as a "
            "fallback baseline when the primary Qwen-oracle table is "
            "suspected of hash-collision collapse. Output format is "
            "compatible with --engram-ann-table / --engram-xattn-table."
        ),
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Path to a pre-tokenized HuggingFace dataset (load_from_disk)",
    )
    parser.add_argument(
        "--tokenizer", required=True,
        help=(
            "HuggingFace tokenizer ID used to produce the dataset. Must "
            "match the student's tokenizer so decoded n-gram text "
            "aligns with the token-id-based hash keys."
        ),
    )
    parser.add_argument(
        "--embedder", default=DEFAULT_EMBEDDER,
        help=f"sentence-transformers model ID (default: {DEFAULT_EMBEDDER!r})",
    )
    parser.add_argument(
        "--top-ngrams", type=int, default=DEFAULT_TOP_NGRAMS,
        help=(
            "Number of most-frequent unique n-grams to embed "
            f"(default: {DEFAULT_TOP_NGRAMS})"
        ),
    )
    parser.add_argument(
        "--entries", type=int, default=DEFAULT_TOTAL_ENTRIES,
        help=(
            "Target dense-table rows. Must match student training "
            f"corpus-table size (default: {DEFAULT_TOTAL_ENTRIES})."
        ),
    )
    parser.add_argument(
        "--engram-dim", type=int, default=DEFAULT_ENGRAM_DIM,
        help=f"Output engram_dim (default: {DEFAULT_ENGRAM_DIM})",
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
        "--max-sequences", type=int, default=DEFAULT_MAX_SEQUENCES,
        help=(
            "Cap on sequences scanned for frequency counting. Defaults "
            f"to {DEFAULT_MAX_SEQUENCES} to bound Counter memory. Pass 0 "
            "for a full-corpus scan (expect tens of GB of RAM)."
        ),
    )
    parser.add_argument(
        "--device", default=None,
        help="torch device override for the embedder (default: auto)",
    )
    parser.add_argument(
        "--embed-batch-size", type=int, default=128,
        help="sentence-transformer encode batch size (default: 128)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output safetensors path",
    )
    parser.add_argument(
        "--stats-output", default=None,
        help="Optional JSON stats path (default: output + '.stats.json')",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)

    if args.top_ngrams <= 0 or args.entries <= 0 or args.engram_dim <= 0:
        print(
            "Error: --top-ngrams, --entries, and --engram-dim must be positive",
            file=sys.stderr,
        )
        return 2

    # argparse validated --hash-seeds via type=parse_hash_seeds.
    seeds = args.hash_seeds

    print("Phase 1: counting n-gram frequencies...", flush=True)
    freqs = count_ngrams(args.dataset, args.max_sequences)
    print(f"  found {len(freqs.counter):,} unique n-grams", flush=True)

    print(f"Phase 2: selecting top {args.top_ngrams:,} most-frequent n-grams...", flush=True)
    top = freqs.top_k(args.top_ngrams)
    if not top:
        print(
            "Error: no bigrams/trigrams found in the scanned dataset. "
            "Raise --max-sequences or verify the dataset has sequences of "
            "length >= 2.",
            file=sys.stderr,
        )
        return 1
    print(f"  top frequency: {top[0][2]:,}", flush=True)
    print(f"  bottom-of-top frequency: {top[-1][2]:,}", flush=True)

    print("Phase 3: detokenizing n-grams to text...", flush=True)
    texts = detokenize_ngrams(top, args.tokenizer)
    print(f"  sample text[0]: {texts[0]!r}", flush=True)

    print("Phase 4: embedding n-gram texts...", flush=True)
    vectors = embed_ngram_texts(
        texts,
        embedder_model_id=args.embedder,
        engram_dim=args.engram_dim,
        device=args.device,
        batch_size=args.embed_batch_size,
    )
    print(f"  embeddings shape: {vectors.shape}", flush=True)

    print("Phase 5: placing vectors into sparse dense table via hash...", flush=True)
    table, populated, total_writes = build_sparse_table(
        top_ngrams=top,
        ngram_vectors=vectors,
        hash_seeds=seeds,
        total_entries=args.entries,
        engram_dim=args.engram_dim,
    )
    print(
        f"  populated {populated:,}/{args.entries:,} rows "
        f"({100*populated/args.entries:.1f}%) "
        f"via {total_writes:,} writes",
        flush=True,
    )

    save_oracle_table(table, args.output)

    import json
    stats_path = args.stats_output or (str(args.output) + ".stats.json")
    Path(stats_path).parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump({
            "kind": "sparse_uncollided",
            "embedder": args.embedder,
            "tokenizer": args.tokenizer,
            "top_ngrams_requested": args.top_ngrams,
            "unique_ngrams_seen": len(freqs.counter),
            "total_entries": args.entries,
            "engram_dim": args.engram_dim,
            "hash_seeds": list(seeds),
            "populated_rows": populated,
            "populated_fraction": populated / args.entries,
            "total_writes": total_writes,
        }, f, indent=2)
    print(f"Wrote run stats to {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
