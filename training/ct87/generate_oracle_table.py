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

# Sidecar safetensors written when --save-teacher-logits is set. ZEB-139 KL-
# retrofit training reads this file in parallel with the main oracle table to
# compute KL(P_router || P_teacher). The tensor name is fixed so the trainer
# doesn't need to know the producer's CLI flags.
TEACHER_LOGITS_SIDECAR_KEY = "teacher_logits.weight"
TEACHER_LOGITS_SIDECAR_SUFFIX = "_teacher_logits.safetensors"
# Chunk size for the per-batch logits Welford update. The hidden-state path
# materializes a [K, teacher_dim] scratch tensor (~2 GB at K=130k * 4096-dim
# Mistral teacher) which has been stable in production for ZEB-119/134/136.
# The logits path is 8x wider (vocab=32000 vs teacher_dim≤4096), so the same
# unchunked materialization would be ~17 GB at default batch=8/seq_len=2048
# — release-blocking OOM on most boxes. Chunking K caps the scratch at
# `LOGITS_UPDATE_CHUNK_K * vocab * 4 bytes` (~64 MB at vocab=32000) and the
# downstream `update_batch` float64 buffer at the same row count (~128 MB).
LOGITS_UPDATE_CHUNK_K = 512


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


_HARMONY_TEACHER_URI_PREFIX = "harmony:"
# Tokenizers known to share the Mistral-v0.1 SentencePiece vocab (32K).
# Any of these can serve as the tokenizer for a Harmony teacher load — we
# prefer whichever is already in the local HF cache to avoid network/auth.
_HARMONY_COMPATIBLE_TOKENIZERS = (
    "mistralai/Mistral-7B-v0.1",
    "TinyLlama/TinyLlama_v1.1",
)
# State-dict prefixes we silently drop when loading a Harmony teacher
# checkpoint into a stock backbone. Anything matching these is a research
# engram module attached at training time, irrelevant to teacher hidden-
# state extraction.
_TEACHER_IRRELEVANT_PREFIXES = (
    "engram_injections.",
    "engram_skip_router.",
    "engram_xattn.",
    "engram_ann.",
)


def load_and_validate_teacher(
    teacher_model_id: str,
    device: str,
    dtype: str,
    expected_vocab_size: int,
    layer_index: int,
    save_teacher_logits: bool = False,
):
    """Load the teacher, validate vocab parity, resolve and validate --layer.

    Returns `(model, tokenizer, resolved_layer, teacher_dim, torch_module)`.
    Extracted from `run_teacher_pass` so the (relatively opinionated)
    setup logic is testable in isolation and doesn't visually crowd the
    batch loop. Lazy HuggingFace imports stay local so callers without
    the teacher extras can still import the module.

    When `save_teacher_logits=True`, the HF branch loads
    `AutoModelForCausalLM` (which includes the LM head) so callers can
    read `outputs.logits` for the ZEB-139 KL-retrofit sidecar. The
    Harmony branch rejects the combo upfront — the adapter only captures
    hidden states, and ZEB-139 only needs TinyLlama anyway (spec §10).

    A `harmony:<path>` URI dispatches to the Harmony-architecture loader
    (`_load_harmony_teacher`) instead of the HuggingFace AutoModel path.
    """
    if teacher_model_id.startswith(_HARMONY_TEACHER_URI_PREFIX):
        if save_teacher_logits:
            raise ValueError(
                "--save-teacher-logits is not supported with the harmony: "
                "teacher URI. The Harmony adapter captures hidden states "
                "only; logits would require an additional lm_head pass that "
                "is out of scope for ZEB-139 (TinyLlama-only). See spec §10."
            )
        # Strip the prefix and reject empty / whitespace-only paths up-
        # front. Without this, "--teacher harmony:" or "--teacher
        # harmony:   " would fall through to `torch.load("")` and
        # surface as an opaque OSError ("[Errno 2] No such file or
        # directory: ''") that doesn't hint at the URI being malformed.
        ckpt_path = teacher_model_id[len(_HARMONY_TEACHER_URI_PREFIX):].strip()
        if not ckpt_path:
            raise ValueError(
                "--teacher expects 'harmony:<path>' with a non-empty checkpoint "
                f"path; got {teacher_model_id!r}."
            )
        return _load_harmony_teacher(
            ckpt_path=ckpt_path,
            device=device,
            dtype=dtype,
            expected_vocab_size=expected_vocab_size,
            layer_index=layer_index,
        )

    import torch
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

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
    # When --save-teacher-logits is set we upgrade to ForCausalLM so the
    # forward pass also produces `outputs.logits` for the KL-retrofit
    # sidecar (ZEB-139). For Mistral-7B the lm_head adds ~250MB at bf16;
    # for TinyLlama (tied embeddings) the cost is ~zero.
    model_cls = AutoModelForCausalLM if save_teacher_logits else AutoModel
    model = model_cls.from_pretrained(
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


def _load_harmony_teacher(
    ckpt_path: str,
    device: str,
    dtype: str,
    expected_vocab_size: int,
    layer_index: int,
):
    """Load a Harmony-architecture checkpoint as a teacher (ZEB-138 prep).

    Handles the integration gap between `transformers.AutoModel` (used for
    Mistral / TinyLlama / etc.) and `ct87.model.HarmonyModel` (used for
    same-architecture teacher experiments). Returns the exact same tuple
    shape as the HF path so `process_batch` is unchanged: the model is
    wrapped in `_HarmonyTeacherAdapter`, which intercepts the call and
    exposes a HuggingFace-style `outputs.hidden_states[resolved_layer]`
    accessor by attaching a forward hook on the target block.

    Checkpoint format requirements: this loader expects the bundled
    payload that `train.py:save_resumable_checkpoint` writes when
    `--checkpoint-interval > 0` — a torch-pickled dict with at least
    `model_state_dict` and `config` keys (see `ct87/train.py:145-190`).
    The output filename is `<output-dir>/checkpoint.pt`.

    train.py also writes per-step `model_step_<step>.safetensors` +
    `optimizer_step_<step>.pt` via `ct87/train.py:save_checkpoint` (lines
    83-98). Those bare safetensors files are NOT accepted here because
    they hold weights only — without the persisted `config` we cannot
    reconstruct `HarmonyModel(config)`'s architecture (num_layers,
    hidden_dim, engram flags, ...). A future revision could add a
    `<file>.safetensors + <file>.config.json` sidecar branch when KRILE's
    Harmony-474M handoff format is known; until then the bundled `.pt`
    is the only supported input and the pre-flight checklist in the
    findings doc covers verifying it.
    """
    import dataclasses

    import torch

    from ct87.model import HarmonyModel, HarmonyModelConfig

    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32

    print(
        f"Loading Harmony teacher checkpoint {ckpt_path!r} "
        f"(device={device}, dtype={dtype})...",
        flush=True,
    )

    # `weights_only=True` rejects arbitrary pickle globals — necessary
    # because --teacher comes from CLI input and a malicious checkpoint
    # would otherwise execute arbitrary code on torch.load. The safe-
    # globals allowlist mirrors `scripts/forensic_eta_b_capgap.py`'s
    # `_TORCH_LOAD_SAFE_GLOBALS` (kept inline rather than imported to
    # avoid the sys.path manipulation that scripts/ requires); the list
    # is the minimum needed to round-trip a train.py-saved checkpoint
    # (HarmonyModelConfig + the numpy bits used by the saved RNG state).
    safe_globals = _build_torch_load_safe_globals(HarmonyModelConfig)
    with torch.serialization.safe_globals(safe_globals):
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "model_state_dict" not in payload or "config" not in payload:
        raise ValueError(
            f"{ckpt_path}: expected a resumable Harmony checkpoint with "
            "'model_state_dict' and 'config' keys (saved by train.py with "
            "--checkpoint-interval > 0). Got "
            f"{type(payload).__name__} with keys "
            f"{sorted(payload.keys()) if isinstance(payload, dict) else 'n/a'}."
        )
    # Type-check the values, not just the keys. With our restricted
    # safe_globals allowlist + weights_only=True the unpickle would
    # already reject most divergent payloads, but an explicit guard
    # turns "AttributeError: 'dict' object has no attribute
    # 'vocab_size'" / "TypeError: load_state_dict expected a Mapping"
    # into the same actionable error pattern as the keys check above.
    model_state_dict = payload["model_state_dict"]
    config = payload["config"]
    if not isinstance(model_state_dict, dict):
        raise ValueError(
            f"{ckpt_path}: expected payload['model_state_dict'] to be a "
            f"dict, got {type(model_state_dict).__name__}."
        )
    if not isinstance(config, HarmonyModelConfig):
        raise ValueError(
            f"{ckpt_path}: expected payload['config'] to be a "
            f"HarmonyModelConfig, got {type(config).__name__}."
        )
    if config.vocab_size != expected_vocab_size:
        raise ValueError(
            f"Harmony teacher config has vocab_size={config.vocab_size} but "
            f"the corpus expects vocab_size={expected_vocab_size}. The student "
            "and teacher must share a tokenizer for n-gram hash parity."
        )

    # Clear research-engram declarations so HarmonyModel.forward()'s misuse
    # guard (which raises if engram_inject_layers is set but no injection
    # modules were attached) doesn't fire. We only want the bare backbone
    # for hidden-state extraction; the engram path is irrelevant to teacher
    # inference. The state_dict's engram_residual.* entries still load
    # cleanly into the always-built engram_residual module, but it stays
    # inert because we never pass `engram_embeddings` to forward().
    teacher_config = dataclasses.replace(
        config,
        engram_inject_layers=(),
        engram_vcontrast_enabled=False,
        engram_qdiv_enabled=False,
        use_ann_engram=False,
        use_xattn_engram=False,
    )
    model = HarmonyModel(teacher_config)
    missing, unexpected = model.load_state_dict(model_state_dict, strict=False)
    # Filter known-OK absences/extras:
    # - `*.table` / `*.table_normalized` are non-persistent buffers on
    #   EngramCrossAttention modules that aren't in any saved state_dict.
    # - `lm_head.weight` is missing when tie_embeddings=True (it shares with
    #   embed_tokens.weight via assignment after construction).
    # - Engram-attached training modules are irrelevant to teacher inference.
    real_missing = [
        k for k in missing
        if ".table" not in k
        and not (teacher_config.tie_embeddings and k == "lm_head.weight")
    ]
    real_unexpected = [
        k for k in unexpected
        if ".table" not in k
        and not any(k.startswith(p) for p in _TEACHER_IRRELEVANT_PREFIXES)
    ]
    if real_missing or real_unexpected:
        def _summarize(keys):
            return f"{keys[:5]}{f' (+{len(keys) - 5} more)' if len(keys) > 5 else ''}"
        raise RuntimeError(
            f"{ckpt_path}: state_dict load incompatible with stock HarmonyModel(config). "
            f"missing={_summarize(real_missing)} unexpected={_summarize(real_unexpected)}"
        )

    # Disable autograd machinery for inference. Matches what HF's `train(False)`
    # implicitly enables but with explicit grad detachment for memory.
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(device=device, dtype=torch_dtype)
    model.train(False)

    # Resolve --layer against Harmony's num_layers, mirroring the HF semantics
    # used elsewhere in this module: hidden_states[0] = embedding output;
    # hidden_states[i+1] = output of layers[i] for i in [0, num_layers).
    teacher_num_layers = config.num_layers
    resolved_layer = layer_index
    if resolved_layer < 0:
        resolved_layer = teacher_num_layers + 1 + resolved_layer
    if not 0 <= resolved_layer <= teacher_num_layers:
        raise ValueError(
            f"--layer={layer_index} resolves to hidden_states[{resolved_layer}], "
            f"but Harmony teacher exposes only indices [0, {teacher_num_layers}] "
            f"(accepts negatives in [-{teacher_num_layers + 1}, -1] also)."
        )
    print(
        f"Harmony teacher has {teacher_num_layers} layers; extracting "
        f"hidden_states[{resolved_layer}] (arg: {layer_index}).",
        flush=True,
    )

    teacher_dim = config.hidden_dim
    adapter = _HarmonyTeacherAdapter(model, resolved_layer)

    # Resolve the tokenizer last — only after every local checkpoint
    # check has passed (payload shape, value types, vocab match,
    # state-dict load, layer-index resolution). This way an HF
    # cache/network failure can never mask a more-specific, more-
    # actionable checkpoint error on an offline box.
    tokenizer = _load_harmony_compatible_tokenizer(expected_vocab_size)

    return adapter, tokenizer, resolved_layer, teacher_dim, torch


def _build_torch_load_safe_globals(config_cls: type) -> list:
    """Return the safe-globals allowlist for loading a Harmony checkpoint.

    Mirrors `scripts/forensic_eta_b_capgap.py:_TORCH_LOAD_SAFE_GLOBALS`.
    The minimum set needed to round-trip a `train.py`-saved checkpoint
    payload under `weights_only=True`:

      - `HarmonyModelConfig` dataclass (passed in by the caller because
        importing it at module top would force a heavy `ct87.model`
        import for callers that only want the Welford / PCA tests).
      - `numpy.core.multiarray._reconstruct` (numpy 1.x) /
        `numpy._core.multiarray._reconstruct` (numpy 2.x), used by the
        saved numpy RNG state. The submodule moved between major
        versions; try-import covers both.
      - `numpy.ndarray` and `numpy.dtype` for the RNG state arrays.
      - Common numpy 2.x scalar dtype classes (`UInt32DType`, etc.) that
        appear in any pickled RNG payload. `getattr` guards keep the
        list valid on numpy 1.x where `numpy.dtypes` doesn't exist.
    """
    import numpy as np

    try:
        from numpy._core.multiarray import _reconstruct as _np_reconstruct
    except ImportError:  # numpy 1.x
        from numpy.core.multiarray import _reconstruct as _np_reconstruct

    dtype_class_names = (
        "BoolDType",
        "Int8DType", "Int16DType", "Int32DType", "Int64DType",
        "UInt8DType", "UInt16DType", "UInt32DType", "UInt64DType",
        "Float16DType", "Float32DType", "Float64DType",
    )
    dtype_classes: list = []
    dtypes_mod = getattr(np, "dtypes", None)
    if dtypes_mod is not None:
        for name in dtype_class_names:
            cls = getattr(dtypes_mod, name, None)
            if isinstance(cls, type):
                dtype_classes.append(cls)

    return [config_cls, _np_reconstruct, np.ndarray, np.dtype, *dtype_classes]


def _load_harmony_compatible_tokenizer(expected_vocab_size: int):
    """Return a Mistral-vocab-compatible tokenizer for the Harmony teacher path.

    The corpus is pre-tokenized, so the tokenizer is only consulted for
    `vocab_size` (validation) and `pad_token_id` (per-batch padding). Any
    Mistral-v0.1 SentencePiece tokenizer suffices.

    Loading strategy: cache-first across all candidates, then network as
    a one-time fallback. Cache-first avoids the HF-auth detour on the
    gated `mistralai/Mistral-7B-v0.1` repo when a non-gated cache hit
    (e.g. TinyLlama) already satisfies the vocab requirement. Network
    fallback keeps the script working out-of-the-box on machines without
    a pre-warmed cache.
    """
    from transformers import AutoTokenizer

    cache_only_errors: list[tuple[str, Exception]] = []
    network_errors: list[tuple[str, Exception]] = []

    # Pass 1: cache-only. Fast and never reaches the network/auth flow.
    for candidate in _HARMONY_COMPATIBLE_TOKENIZERS:
        try:
            tok = AutoTokenizer.from_pretrained(candidate, local_files_only=True)
        except Exception as e:  # noqa: BLE001 — try next candidate
            cache_only_errors.append((candidate, e))
            continue
        if int(tok.vocab_size) != expected_vocab_size:
            cache_only_errors.append((candidate, ValueError(
                f"vocab_size={tok.vocab_size} != {expected_vocab_size}"
            )))
            continue
        return tok

    # Pass 2: network-allowed. Reached only when nothing was cached.
    for candidate in _HARMONY_COMPATIBLE_TOKENIZERS:
        try:
            tok = AutoTokenizer.from_pretrained(candidate)
        except Exception as e:  # noqa: BLE001 — try next candidate
            network_errors.append((candidate, e))
            continue
        if int(tok.vocab_size) != expected_vocab_size:
            network_errors.append((candidate, ValueError(
                f"vocab_size={tok.vocab_size} != {expected_vocab_size}"
            )))
            continue
        return tok

    def _format(errors: list[tuple[str, Exception]]) -> str:
        return ", ".join(f"{name}: {type(err).__name__}: {err}" for name, err in errors) or "(none tried)"

    raise RuntimeError(
        f"Could not load any Mistral-vocab-compatible tokenizer for the "
        f"Harmony teacher path. Tried {list(_HARMONY_COMPATIBLE_TOKENIZERS)}. "
        f"Cache-only attempts: [{_format(cache_only_errors)}]. "
        f"Network-allowed attempts: [{_format(network_errors)}]. "
        "Pre-cache one of these via `huggingface-cli download <id>` and rerun."
    )


class _HarmonyTeacherAdapter:
    """HuggingFace-style call interface around a `HarmonyModel`.

    `process_batch` calls `model(input_ids=...)` and reads
    `outputs.hidden_states[resolved_layer]`. The HF AutoModel path returns
    a tuple of every layer's hidden state; HarmonyModel returns logits
    only. We bridge that gap by registering a single forward hook on the
    target module (embedding for resolved_layer=0, transformer block
    `resolved_layer-1` otherwise) and exposing the captured tensor
    through a one-element view.
    """

    def __init__(self, model, resolved_layer: int):
        self._model = model
        self._resolved_layer = resolved_layer
        if resolved_layer == 0:
            self._hook_module = model.embed_tokens
        else:
            self._hook_module = model.layers[resolved_layer - 1]
        self._captured = None
        self._hook_module.register_forward_hook(self._capture_hook)

    def _capture_hook(self, _module, _inputs, output):
        # Both nn.Embedding and TransformerLayer.forward currently return a
        # plain tensor. Future-proofing: collapse a tuple to its first elem.
        self._captured = output[0] if isinstance(output, tuple) else output

    def __call__(self, input_ids):
        import torch

        self._captured = None
        with torch.no_grad():
            self._model(input_ids=input_ids)
        # Rebind to a local and clear the adapter's reference BEFORE
        # returning so `process_batch`'s post-batch `del + empty_cache()`
        # can actually reclaim the GPU tensor. Otherwise the adapter
        # holds a second reference until the next __call__, defeating the
        # caller's batch-by-batch VRAM contract that the HF teacher path
        # naturally satisfies.
        captured = self._captured
        if captured is None:
            raise RuntimeError(
                "Harmony teacher hook did not fire during forward — the "
                "registered module may have been replaced or skipped. "
                f"Hook target: {type(self._hook_module).__name__} at "
                f"resolved_layer={self._resolved_layer}."
            )
        self._captured = None
        return _HarmonyTeacherOutput(
            hidden_states=_HookedHiddenStatesView(captured, self._resolved_layer)
        )


class _HarmonyTeacherOutput:
    """Mimics the `BaseModelOutput.hidden_states` access pattern."""

    def __init__(self, hidden_states):
        self.hidden_states = hidden_states


class _HookedHiddenStatesView:
    """Single-layer stand-in for `outputs.hidden_states` (a tuple in HF land).

    Only the index that the adapter hooked is queryable — any other index
    raises, which is the desired loud-failure mode if `process_batch` ever
    asks for a layer we didn't capture.
    """

    def __init__(self, tensor, layer_idx: int):
        self._tensor = tensor
        self._layer_idx = layer_idx

    def __getitem__(self, idx: int):
        if idx != self._layer_idx:
            raise IndexError(
                f"Harmony teacher adapter only captured layer {self._layer_idx}; "
                f"requested {idx}. Re-create the adapter with the correct "
                "resolved_layer."
            )
        return self._tensor


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
    logits_table: WelfordTable | None = None,
) -> int:
    """Run one forward pass + Welford update for a batch. Returns token count.

    Splits out of `run_teacher_pass` so the hot loop is the one thing
    being read when reasoning about throughput or VRAM. Every other
    call here (truncation, padding, forward, Welford) is idempotent
    and has its own tests.

    If `logits_table` is provided, also extracts `outputs.logits` from
    the same forward pass and Welford-updates the logits table at the
    same n-gram row indices used for the hidden-state table. Caller
    must have loaded the model via `AutoModelForCausalLM` (or any model
    whose forward output exposes `.logits`) — otherwise this raises
    AttributeError on the first batch.
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
    # Move hidden state to CPU fp32 immediately so we can free GPU memory
    # before the next batch. VRAM stays flat across the loop. The
    # hidden tensor is small (teacher_dim ≤ 4096) so a full CPU staging
    # copy is cheap (~2 GB at default batch/seq for Mistral; ~1 GB for
    # TinyLlama).
    hidden_np = hidden_states.float().cpu().numpy()
    # Capture a reference to the GPU-side logits tensor (when needed)
    # WITHOUT staging to CPU: the logits path is 8x wider than the
    # hidden path (vocab=32000 vs teacher_dim≤4096), so the analogous
    # full CPU copy would be ~2 GB at default settings. Round-1 (PR
    # #255) chunked the K-loop downstream of `logits_np` but left this
    # full-tensor staging in place; CodeRabbit round-2 caught it. We
    # now per-chunk-pull from `logits_gpu` inside the K loop below,
    # bounding peak host RAM at chunk*vocab*4 bytes (~64 MB).
    logits_gpu = outputs.logits if logits_table is not None else None
    del hidden_states, outputs, input_ids
    if device.startswith("cuda"):
        # NB: empty_cache() doesn't free `logits_gpu` because we still
        # hold a reference to it. The K-loop's `del logits_gpu` +
        # second empty_cache() below releases it after the chunked
        # transfers finish.
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
        if logits_table is not None:
            # Same row indices, same (b, p) positions — guarantees the
            # logits sidecar's row r corresponds to the hidden table's
            # row r for the KL-retrofit lookup. Only the value vectors
            # differ (vocab-wide logits instead of teacher_dim-wide
            # hidden state).
            #
            # Chunked along K AND pulled per-chunk from the GPU tensor
            # (no full-batch CPU staging). At default batch=8/seq=2048/
            # vocab=32000 the unchunked `logits_gpu[bs, ps, :]` would
            # peak at ~17 GB host RAM; per-chunk transfer caps it at
            # `LOGITS_UPDATE_CHUNK_K * vocab * 4 bytes` (~64 MB at
            # vocab=32000) plus the downstream Welford float64 buffer
            # (~128 MB).
            #
            # Welford correctness is unaffected — `update_batch` already
            # commutes across calls (combines pre-batch mean with each
            # batch's contribution exactly), so K processed in N chunks
            # vs 1 chunk produces bit-equivalent means modulo float
            # accumulation order.
            #
            # Move bs/ps to the same device as logits_gpu once so the
            # per-chunk fancy index doesn't trigger a fresh H2D copy
            # of small index tensors on every iteration.
            bs_t = torch_module.as_tensor(
                bs, device=logits_gpu.device, dtype=torch_module.long,
            )
            ps_t = torch_module.as_tensor(
                ps, device=logits_gpu.device, dtype=torch_module.long,
            )
            for start in range(0, indices_arr.shape[0], LOGITS_UPDATE_CHUNK_K):
                end = start + LOGITS_UPDATE_CHUNK_K
                idx_chunk = indices_arr[start:end]
                # GPU fancy-index pulls only the [chunk, vocab] slice,
                # then `.cpu().numpy()` materializes that small slice on
                # host. detach() avoids retaining autograd state if the
                # caller ever drops the inference_mode wrapper.
                chunk_gpu = logits_gpu[bs_t[start:end], ps_t[start:end], :]
                chunk_np = chunk_gpu.detach().float().cpu().numpy()
                logits_table.update_batch(
                    idx_chunk, chunk_np.astype(np.float32, copy=False),
                )
                del chunk_gpu
            del logits_gpu, bs_t, ps_t
            if device.startswith("cuda"):
                torch_module.cuda.empty_cache()

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
    save_teacher_logits: bool = False,
) -> tuple[WelfordTable, WelfordTable | None]:
    """Drive the full teacher forward pass, accumulating Welford means.

    High-level flow only: delegate teacher setup to
    `load_and_validate_teacher` and per-batch work to `process_batch`.
    Lazy-imports `datasets` so callers without the teacher extras can
    still import the module for the Welford / PCA tests.

    Returns `(hidden_table, logits_table_or_None)`. The logits table is
    `None` when `save_teacher_logits=False` (preserves the no-flag
    memory profile — no extra `[total_entries, vocab]` allocation). The
    second-element shape is `[total_entries, expected_vocab_size]`
    float32 when populated (~1.28 GB at 10K rows × 32K vocab; bf16-cast
    happens at the sidecar save site).
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
            save_teacher_logits=save_teacher_logits,
        )
    )
    table = WelfordTable.zeros(total_entries, teacher_dim)
    logits_table = (
        WelfordTable.zeros(total_entries, expected_vocab_size)
        if save_teacher_logits
        else None
    )

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
                logits_table=logits_table,
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
    return table, logits_table


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


def default_teacher_logits_sidecar_path(output_path: str | Path) -> Path:
    """Derive the default sidecar path from the main `--output` value.

    Strategy: replace the `.safetensors` suffix with
    `_teacher_logits.safetensors` so `oracle.safetensors` becomes
    `oracle_teacher_logits.safetensors`. If the suffix is missing, we
    append the suffix verbatim — keeps the rule simple and predictable
    for unusual output paths.
    """
    out = Path(output_path)
    if out.suffix == ".safetensors":
        return out.with_name(out.stem + TEACHER_LOGITS_SIDECAR_SUFFIX)
    return out.with_name(out.name + TEACHER_LOGITS_SIDECAR_SUFFIX)


def save_teacher_logits_sidecar(
    logits_means: np.ndarray,
    output_path: str | Path,
) -> None:
    """Write the Welford-mean teacher logits to a bf16 safetensors sidecar.

    Output schema is fixed: a single tensor named
    `TEACHER_LOGITS_SIDECAR_KEY` with shape `[total_entries, vocab]`,
    dtype `bfloat16`. ZEB-139 KL-retrofit training reads this file
    keyed by xxhash row index identical to the main oracle table, so
    `train.py` can compute `KL(P_router || P_teacher)` without any
    additional metadata exchange.

    bf16 is chosen over fp16 because the logit means span both very
    large positive (in-context tokens) and very negative (suppressed
    tokens) values — bf16's wider exponent prevents underflow on the
    suppressed-token tail that fp16 would zero out.
    """
    import torch
    from safetensors.torch import save_file

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # numpy has no native bf16 — round-trip through torch to get a true
    # bf16 tensor that safetensors persists losslessly.
    tensor_bf16 = torch.from_numpy(logits_means).to(torch.bfloat16)
    save_file({TEACHER_LOGITS_SIDECAR_KEY: tensor_bf16}, str(out))
    print(
        f"Saved teacher-logits sidecar to {out} "
        f"({tuple(tensor_bf16.shape)}, dtype=bfloat16)"
    )


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
        help=(
            f"Teacher model identifier (default: {DEFAULT_TEACHER!r}). "
            "Two forms accepted: a HuggingFace model ID (e.g. "
            "'mistralai/Mistral-7B-v0.1', 'TinyLlama/TinyLlama_v1.1') "
            "loads via transformers.AutoModel; a 'harmony:<path>' URI "
            "(e.g. 'harmony:checkpoints/harmony474m/checkpoint.pt') "
            "loads a Harmony-architecture checkpoint via "
            "_load_harmony_teacher."
        ),
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
    parser.add_argument(
        "--save-teacher-logits", action="store_true",
        help=(
            "Also Welford-mean the teacher's full LM-head outputs and "
            "save as a bf16 sidecar safetensors at "
            "<output-stem>_teacher_logits.safetensors (override with "
            "--teacher-logits-output). Required by ZEB-139 KL-retrofit "
            "training. Forces the HF teacher to load via "
            "AutoModelForCausalLM (vs AutoModel) so lm_head is available; "
            "adds ~1.28GB CPU-resident accumulator at 10K rows × 32K "
            "vocab. Not supported with --teacher harmony:."
        ),
    )
    parser.add_argument(
        "--teacher-logits-output", default=None,
        help=(
            "Override path for the teacher-logits sidecar. Only consulted "
            "when --save-teacher-logits is set. Default derives from "
            "--output via default_teacher_logits_sidecar_path()."
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

    table, logits_table = run_teacher_pass(
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
        save_teacher_logits=args.save_teacher_logits,
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

    # Validate logits/hidden parity BEFORE writing any artifacts. If the
    # check fails, exiting with a half-written `oracle.safetensors` on
    # disk would let a downstream consumer (e.g. someone re-running
    # training without --save-teacher-logits) pick up the orphaned
    # output and not realize the producer aborted. Fail clean instead:
    # PCA-project, validate, then write everything together or nothing.
    teacher_logits_sidecar_path: str | None = None
    if logits_table is not None:
        # The logits table is xxhash-keyed identically to the hidden
        # table, so its populated_mask must match. Hard-fail if it
        # doesn't — that'd indicate a per-batch bug where the two
        # update_batch calls saw different indices.
        logits_populated_mask = logits_table.populated_mask
        if not np.array_equal(logits_populated_mask, populated_mask):
            print(
                "Error: teacher-logits sidecar populated_mask diverged from "
                "hidden-state oracle populated_mask. They must match exactly "
                "(same xxhash row indices, same forward-pass batches). "
                f"hidden_populated={populated}, "
                f"logits_populated={int(logits_populated_mask.sum())}.",
                file=sys.stderr,
            )
            return 1
        teacher_logits_sidecar_path = args.teacher_logits_output or str(
            default_teacher_logits_sidecar_path(args.output),
        )

    projected = apply_pca_projection(
        table.means, components, mean_vector, populated_mask=populated_mask,
    )
    save_oracle_table(projected, args.output)
    if teacher_logits_sidecar_path is not None:
        save_teacher_logits_sidecar(
            logits_table.means, teacher_logits_sidecar_path,
        )

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
            "teacher_logits_sidecar": teacher_logits_sidecar_path,
        }, f, indent=2)
    print(f"Wrote run stats to {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
