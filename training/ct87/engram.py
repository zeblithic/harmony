"""Engram conditional memory -- gated residual injection and table lookup.

Mirrors crates/harmony-inference/src/engram_residual.rs (EngramGatedResidual)
and crates/harmony-engram/src/hash.rs (xxhash64 N-gram lookups).

The EngramGatedResidual module takes hidden states and Engram embeddings,
applies gated projection + causal depthwise conv1d + SiLU, and returns a
residual tensor for injection into the transformer hidden state.

The EngramTable class loads a safetensors embedding table and performs
N-gram hash lookups to produce per-position Engram embeddings.
"""

from __future__ import annotations

import math
import struct
import warnings
from pathlib import Path

import torch
import torch.nn as nn

try:
    import xxhash
    _fast_hash = xxhash.xxh64_intdigest
except ImportError:
    _fast_hash = None
import torch.nn.functional as F

from ct87.model import HarmonyModelConfig, RMSNorm

# Default depthwise conv1d kernel size (matches Rust default)
CONV_KERNEL_SIZE = 3


def compute_qdiv_aux(
    topk_idx: torch.Tensor,
    attn: torch.Tensor,
    table_size: int,
) -> torch.Tensor:
    """MoE-style load-balancing auxiliary loss over retrieval row usage.

    Minimized when Q spreads retrieval uniformly over table rows (loss -> 1);
    maximized under full concentration on a single row (loss -> table_size).

    Only P (soft attention-weighted mass) carries gradient — f (hard top-k
    selection frequency) is non-differentiable and serves as a frequency
    weight. Gradient flows through attn into the softmax, then into q_proj,
    k_proj, and retrieval_query_proj (via the retrieval_bias_weight * topk_sims
    term that also participates in the pre-softmax scores).

    Args:
        topk_idx: [B, L, k] int64 — top-k row indices selected per query.
        attn:     [B, L, H, k] float — softmaxed attention weights per head.
        table_size: N — full corpus table size.

    Returns:
        Scalar loss. Under uniform row usage, this equals 1.0. Under full
        concentration on one row, it equals table_size. Under uniform over
        S<N rows, it equals table_size/S.
    """
    B, L, k = topk_idx.shape
    H = attn.shape[2]

    f = torch.bincount(
        topk_idx.reshape(-1), minlength=table_size,
    ).to(attn.dtype) / (B * L * k)

    idx = topk_idx.unsqueeze(2).expand(B, L, H, k).reshape(-1)
    P = torch.zeros(table_size, device=attn.device, dtype=attn.dtype)
    P.scatter_add_(0, idx, attn.reshape(-1))
    P = P / (B * L * H)

    return table_size * (f * P).sum()


class EngramGatedResidual(nn.Module):
    """Gated residual injection for Engram embeddings.

    Implements the DeepSeek-inspired gating pattern:
    1. Project engram into key/value space
    2. Gate via normalized dot product: sigmoid(dot(norm(h), norm(k)) / sqrt(d))
    3. Apply gate to value
    4. Causal depthwise conv1d with left-padding
    5. SiLU activation
    6. Return residual (caller adds to hidden state)

    Weight shapes match Rust EngramGatedResidual::from_tensors() exactly:
    - key_proj:   [hidden_dim, engram_dim] (no bias)
    - value_proj: [hidden_dim, engram_dim] (no bias)
    - gate_norm:  [hidden_dim]
    - key_norm:   [hidden_dim]
    - conv1d:     [hidden_dim, 1, kernel_size] (depthwise, no bias)
    """

    def __init__(
        self,
        config: HarmonyModelConfig,
        conv_kernel_size: int = CONV_KERNEL_SIZE,
    ):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.engram_dim = config.engram_dim
        self.conv_kernel_size = conv_kernel_size

        self.key_proj = nn.Linear(config.engram_dim, config.hidden_dim, bias=False)
        self.value_proj = nn.Linear(config.engram_dim, config.hidden_dim, bias=False)
        self.gate_norm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.key_norm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)

        # Depthwise causal conv1d: groups=hidden_dim, no bias
        self.conv1d = nn.Conv1d(
            in_channels=config.hidden_dim,
            out_channels=config.hidden_dim,
            kernel_size=conv_kernel_size,
            groups=config.hidden_dim,
            bias=False,
            padding=0,  # We handle causal padding manually
        )

    def forward(
        self, hidden_state: torch.Tensor, engram_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Compute gated residual for Engram injection.

        Args:
            hidden_state:    [batch, seq_len, hidden_dim]
            engram_embedding: [batch, seq_len, engram_dim]

        Returns:
            Residual tensor: [batch, seq_len, hidden_dim]
            Caller adds: hidden_state = hidden_state + self.forward(...)
        """
        # Project engram into key/value space: [b, l, hidden_dim]
        key = self.key_proj(engram_embedding)
        value = self.value_proj(engram_embedding)

        # Normalize for stable gating
        h_norm = self.gate_norm(hidden_state)
        k_norm = self.key_norm(key)

        # Dot product gate: [b, l, 1]
        dot = (h_norm * k_norm).sum(dim=-1, keepdim=True)
        gate = torch.sigmoid(dot / (self.hidden_dim ** 0.5))

        # Apply gate to value: [b, l, hidden_dim]
        gated_value = gate * value

        # Causal depthwise conv1d:
        # NLC → NCL for conv1d
        ncl = gated_value.transpose(1, 2)

        # Left-pad with (kernel_size - 1) zeros for causal masking
        padded = F.pad(ncl, (self.conv_kernel_size - 1, 0))

        # Apply depthwise conv1d
        conv_out = self.conv1d(padded)

        # NCL → NLC
        nlc = conv_out.transpose(1, 2)

        # SiLU activation
        return F.silu(nlc)


# ---------------------------------------------------------------------------
# Engram table lookup (training-time, in-memory)
# ---------------------------------------------------------------------------

def _xxhash64(data: bytes, seed: int) -> int:
    """Pure-Python xxhash64 implementation for N-gram hashing.

    Must produce identical results to xxhash_rust::xxh64::xxh64 used in
    crates/harmony-engram/src/hash.rs. The algorithm is fixed - any
    divergence breaks Engram table compatibility.
    """
    PRIME1 = 11400714785074694791
    PRIME2 = 14029467366897019727
    PRIME3 = 1609587929392839161
    PRIME4 = 9650029242287828579
    PRIME5 = 2870177450012600261
    M64 = 0xFFFFFFFFFFFFFFFF

    def _round(v: int, inp: int) -> int:
        v = (v + inp * PRIME2) & M64
        v = ((v << 31) | (v >> 33)) & M64
        return (v * PRIME1) & M64

    length = len(data)
    idx = 0

    if length >= 32:
        v1 = (seed + PRIME1 + PRIME2) & M64
        v2 = (seed + PRIME2) & M64
        v3 = seed & M64
        v4 = (seed - PRIME1) & M64

        limit = length - 32
        while idx <= limit:
            v1 = _round(v1, int.from_bytes(data[idx:idx+8], 'little'))
            idx += 8
            v2 = _round(v2, int.from_bytes(data[idx:idx+8], 'little'))
            idx += 8
            v3 = _round(v3, int.from_bytes(data[idx:idx+8], 'little'))
            idx += 8
            v4 = _round(v4, int.from_bytes(data[idx:idx+8], 'little'))
            idx += 8

        h64 = (((v1 << 1) | (v1 >> 63)) & M64)
        h64 = (h64 + (((v2 << 7) | (v2 >> 57)) & M64)) & M64
        h64 = (h64 + (((v3 << 12) | (v3 >> 52)) & M64)) & M64
        h64 = (h64 + (((v4 << 18) | (v4 >> 46)) & M64)) & M64

        def _merge_round(acc: int, val: int) -> int:
            val = (val * PRIME2) & M64
            val = ((val << 31) | (val >> 33)) & M64
            val = (val * PRIME1) & M64
            acc = (acc ^ val) & M64
            acc = (acc * PRIME1 + PRIME4) & M64
            return acc

        h64 = _merge_round(h64, v1)
        h64 = _merge_round(h64, v2)
        h64 = _merge_round(h64, v3)
        h64 = _merge_round(h64, v4)
    else:
        h64 = (seed + PRIME5) & M64

    h64 = (h64 + length) & M64

    # Remaining bytes
    while idx <= length - 8:
        k1 = int.from_bytes(data[idx:idx+8], 'little')
        k1 = (k1 * PRIME2) & M64
        k1 = ((k1 << 31) | (k1 >> 33)) & M64
        k1 = (k1 * PRIME1) & M64
        h64 = (h64 ^ k1) & M64
        h64 = (((h64 << 27) | (h64 >> 37)) * PRIME1 + PRIME4) & M64
        idx += 8

    while idx <= length - 4:
        k1 = int.from_bytes(data[idx:idx+4], 'little')
        h64 = (h64 ^ (k1 * PRIME1)) & M64
        h64 = (((h64 << 23) | (h64 >> 41)) * PRIME2 + PRIME3) & M64
        idx += 4

    while idx < length:
        h64 = (h64 ^ (data[idx] * PRIME5)) & M64
        h64 = (((h64 << 11) | (h64 >> 53)) * PRIME1) & M64
        idx += 1

    # Final avalanche
    h64 = (h64 ^ (h64 >> 33)) & M64
    h64 = (h64 * PRIME2) & M64
    h64 = (h64 ^ (h64 >> 29)) & M64
    h64 = (h64 * PRIME3) & M64
    h64 = (h64 ^ (h64 >> 32)) & M64

    return h64


def _hash_ngram(tokens: list[int], seed: int) -> int:
    """Hash an N-gram's tokens with a single xxhash64 seed.

    Tokens are encoded as contiguous little-endian u32 bytes, matching
    crates/harmony-engram/src/hash.rs::hash_ngram().
    """
    data = b"".join(struct.pack("<I", t) for t in tokens)
    return _xxhash64(data, seed)


class EngramTable:
    """In-memory Engram embedding table for training.

    Loads a safetensors file containing a flat [total_entries, engram_dim]
    embedding table, then performs xxhash64 N-gram lookups to produce
    per-position embeddings matching the Rust harmony-engram pipeline.

    For training, the table is NOT a trainable parameter - it provides
    external knowledge that the model learns to gate and use via
    EngramGatedResidual.
    """

    def __init__(
        self,
        table: torch.Tensor,
        hash_seeds: list[int],
        device: torch.device | str = "cpu",
    ):
        """Construct from an existing embedding tensor.

        Args:
            table: [total_entries, engram_dim] embedding tensor
            hash_seeds: per-head xxhash64 seeds (length = num_heads)
            device: device to place embeddings on
        """
        # Keep table on CPU for hash-and-index phase to avoid per-row CUDA
        # kernel launches. Only the aggregated result is moved to device.
        self.table = table.cpu().float()
        self.total_entries = table.shape[0]
        self.engram_dim = table.shape[1]
        self.hash_seeds = hash_seeds
        self.num_heads = len(hash_seeds)
        self.device = device if isinstance(device, torch.device) else torch.device(device)

    @staticmethod
    def from_safetensors(
        path: str | Path,
        hash_seeds: list[int],
        tensor_name: str = "engram.weight",
        device: torch.device | str = "cpu",
    ) -> EngramTable:
        """Load from a safetensors file."""
        from safetensors.torch import load_file
        tensors = load_file(str(path))
        if tensor_name not in tensors:
            available = list(tensors.keys())
            raise KeyError(
                f"Tensor '{tensor_name}' not found in {path}; "
                f"available: {available}"
            )
        return EngramTable(tensors[tensor_name], hash_seeds, device)

    def lookup_batch(
        self, input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Engram embeddings for a batch of token sequences.

        Extracts bigrams and trigrams, hashes to table indices, and
        aggregates embeddings per position. Position 0 always gets zero
        embeddings (no N-gram coverage).

        All table lookups are collected into a single batched index_select
        on the CPU table, then scatter-added to the output and moved to
        the target device - avoiding per-row CUDA kernel launches.

        Args:
            input_ids: [batch, seq_len] token IDs

        Returns:
            [batch, seq_len, engram_dim] embedding tensor
        """
        batch_size, seq_len = input_ids.shape
        # Collect all (batch, position, table_index) triples
        batch_indices: list[int] = []
        positions: list[int] = []
        table_indices: list[int] = []

        for b in range(batch_size):
            tokens = input_ids[b].tolist()
            self._collect_indices(tokens, b, batch_indices, positions, table_indices)

        # Build result on CPU, then move to device once
        result = torch.zeros(
            batch_size, seq_len, self.engram_dim, dtype=torch.float32,
        )

        if table_indices:
            # Single batched lookup on CPU table
            idx_t = torch.tensor(table_indices, dtype=torch.long)
            embs = self.table[idx_t]  # [n, engram_dim]

            # Scatter-add into result
            for i, (b, pos) in enumerate(zip(batch_indices, positions)):
                result[b, pos] += embs[i]

        return result.to(self.device)

    def lookup_batch_projected(
        self,
        input_ids: torch.Tensor,
        embeddings: torch.Tensor,
        projection: nn.Module,
    ) -> torch.Tensor:
        """Compute Engram embeddings using projection-generated binary keys.

        Same as lookup_batch() but replaces xxhash-on-token-bytes with
        projection→binarize→xxhash-on-binary-key-bytes, matching the Rust
        inference path in compute_lookup_from_bytes().

        Args:
            input_ids: [batch, seq_len] token IDs (for shape only)
            embeddings: [batch, seq_len, hidden_dim] from model.embed_tokens
            projection: LatentProjection module with project_ngrams()

        Returns:
            [batch, seq_len, engram_dim] embedding tensor
        """
        batch_size, seq_len = input_ids.shape
        batch_indices: list[int] = []
        positions: list[int] = []
        table_indices: list[int] = []

        total_entries = self.total_entries
        seeds = self.hash_seeds

        for b in range(batch_size):
            emb = embeddings[b : b + 1]  # [1, seq_len, hidden_dim]
            proj_keys, proj_positions = projection.project_ngrams(emb, seq_len)

            for key_bytes, pos in zip(proj_keys, proj_positions):
                for seed in seeds:
                    if _fast_hash is not None:
                        idx = _fast_hash(key_bytes, seed=seed) % total_entries
                    else:
                        idx = _xxhash64(key_bytes, seed) % total_entries
                    batch_indices.append(b)
                    positions.append(pos)
                    table_indices.append(idx)

        result = torch.zeros(
            batch_size, seq_len, self.engram_dim, dtype=torch.float32,
        )

        if table_indices:
            idx_t = torch.tensor(table_indices, dtype=torch.long)
            embs = self.table[idx_t]  # [n, engram_dim]

            flat = result.view(-1, self.engram_dim)
            b_idx = torch.tensor(batch_indices, dtype=torch.long)
            p_idx = torch.tensor(positions, dtype=torch.long)
            linear_idx = b_idx * seq_len + p_idx
            flat.index_add_(0, linear_idx, embs)

        return result.to(self.device)

    def lookup_from_keys(
        self,
        binary_keys: list[bytes],
        positions: list[int],
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor:
        """Look up Engram embeddings from pre-computed binary keys.

        Unlike lookup_batch_projected(), this does not run the projection -
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
        if len(binary_keys) != len(positions):
            raise ValueError(
                f"binary_keys and positions must have the same length, "
                f"got {len(binary_keys)} and {len(positions)}"
            )

        batch_indices: list[int] = []
        pos_list: list[int] = []
        table_indices: list[int] = []

        total_entries = self.total_entries
        seeds = self.hash_seeds

        for key_bytes, pos in zip(binary_keys, positions):
            for seed in seeds:
                if _fast_hash is not None:
                    idx = _fast_hash(key_bytes, seed=seed) % total_entries
                else:
                    idx = _xxhash64(key_bytes, seed) % total_entries
                batch_indices.append(0)
                pos_list.append(pos)
                table_indices.append(idx)

        result = torch.zeros(
            batch_size, seq_len, self.engram_dim, dtype=torch.float32,
        )

        if table_indices:
            idx_t = torch.tensor(table_indices, dtype=torch.long)
            embs = self.table[idx_t]

            flat = result.view(-1, self.engram_dim)
            b_idx = torch.tensor(batch_indices, dtype=torch.long)
            p_idx = torch.tensor(pos_list, dtype=torch.long)
            linear_idx = b_idx * seq_len + p_idx
            flat.index_add_(0, linear_idx, embs)

        return result.to(self.device)

    def _collect_indices(
        self,
        tokens: list[int],
        batch_idx: int,
        batch_indices: list[int],
        positions: list[int],
        table_indices: list[int],
    ) -> None:
        """Collect all N-gram table indices for a single sequence."""
        seq_len = len(tokens)

        # Bigrams: [t[i], t[i+1]] attributed to position i+1
        for i in range(seq_len - 1):
            bigram = [tokens[i], tokens[i + 1]]
            for seed in self.hash_seeds:
                idx = _hash_ngram(bigram, seed) % self.total_entries
                batch_indices.append(batch_idx)
                positions.append(i + 1)
                table_indices.append(idx)

        # Trigrams: [t[i], t[i+1], t[i+2]] attributed to position i+2
        for i in range(seq_len - 2):
            trigram = [tokens[i], tokens[i + 1], tokens[i + 2]]
            for seed in self.hash_seeds:
                idx = _hash_ngram(trigram, seed) % self.total_entries
                batch_indices.append(batch_idx)
                positions.append(i + 2)
                table_indices.append(idx)


# ---------------------------------------------------------------------------
# ZEB-117 Model gamma: ANN retrieval + gated residual + anti-collapse
# ---------------------------------------------------------------------------
#
# Research-only: NOT mirrored in Rust (crates/harmony-inference). Configs
# that use this are not GGUF-portable. Implements the recommendations in
# docs/research/2026-04-14-engram-injection-mechanism-findings.md:
#
#   - brute-force k=1 cosine retrieval against a fixed corpus table (s1)
#   - layer-norm on retrieved embeddings pre-projection (s8.1 mitigation)
#   - hard gate clamp g_min during warmup steps (s3.3.3, s7.3 phase 1)
#   - gate-probability side output for the training loop to apply entropy
#     regularization (s3.3.1) and optional auxiliary reconstruction loss
#     (s3.3.2)


# Warning threshold for the full-table softmax retrieval
# (batch * seq_len * table_entries). Retrieval scales linearly in this
# product; at or above this threshold we emit a one-time RuntimeWarning
# so the user knows to consider chunked/top-k retrieval before scaling
# further.
_ANN_RETRIEVAL_SIZE_WARN_THRESHOLD = 16 * 1024 * 1024
_ann_retrieval_warned = False


class EngramANNInjection(nn.Module):
    """Gated residual injection with internal ANN retrieval and anti-collapse.

    Differences from `EngramGatedResidual`:
      * Retrieval is internal: projects hidden state to engram space and
        does brute-force k=1 cosine nearest-neighbor lookup against a
        fixed corpus table.
      * LayerNorm on retrieved embeddings before key/value projection.
      * Forward returns (residual, gate_probs) so the training loop can
        apply entropy regularization and optional reconstruction loss.
      * Supports a hard gate clamp for the first `clamp_until_step` steps
        to force gradient flow through the memory path during the
        chaotic-alignment phase of pretraining.

    The table is a non-trainable buffer - retrieved embeddings flow
    through learnable projections but the table itself is frozen, matching
    the Memorizing-Transformer and RETRO paradigm.
    """

    def __init__(
        self,
        config: HarmonyModelConfig,
        table: torch.Tensor,
        conv_kernel_size: int = CONV_KERNEL_SIZE,
        clamp_until_step: int = 800,
        clamp_min: float = 0.5,
        retrieval_temperature: float | None = None,
        use_head_gates: bool = False,
    ):
        super().__init__()
        if table.dim() != 2:
            raise ValueError(
                f"table must be 2-D [total_entries, engram_dim], got {table.shape}"
            )
        if table.shape[1] != config.engram_dim:
            raise ValueError(
                f"table engram_dim {table.shape[1]} != config.engram_dim "
                f"{config.engram_dim}"
            )

        self.hidden_dim = config.hidden_dim
        self.engram_dim = config.engram_dim
        self.conv_kernel_size = conv_kernel_size
        self.clamp_until_step = int(clamp_until_step)
        self.clamp_min = float(clamp_min)
        # Temperature for softmax attention over the table. Defaults to
        # the standard scaled-dot-product value so the softmax is sharp
        # enough to approximate k=1 retrieval while remaining
        # differentiable. At inference, hard argmax can replace softmax
        # as a speed optimization with minimal accuracy loss once the
        # query projection is trained.
        self.retrieval_temperature = (
            float(retrieval_temperature)
            if retrieval_temperature is not None
            else 1.0 / (config.engram_dim ** 0.5)
        )
        if (
            not math.isfinite(self.retrieval_temperature)
            or self.retrieval_temperature <= 0
        ):
            raise ValueError(
                f"retrieval_temperature must be a positive finite float, "
                f"got {self.retrieval_temperature!r}"
            )
        if not (0.0 <= self.clamp_min <= 1.0):
            raise ValueError(
                f"clamp_min must be in [0.0, 1.0], got {self.clamp_min!r}"
            )
        if self.clamp_until_step < 0:
            raise ValueError(
                f"clamp_until_step must be >= 0, got {self.clamp_until_step!r}"
            )

        # Corpus table: non-persistent buffer. Checkpoints do NOT include
        # the table (keeps artifacts small and forces the table to be
        # re-loaded from --engram-ann-table on resume, keeping train and
        # eval configurations explicit).
        self.register_buffer("table", table.float(), persistent=False)
        self.register_buffer(
            "table_normalized",
            F.normalize(table.float(), dim=-1, eps=1e-8),
            persistent=False,
        )

        self.query_proj = nn.Linear(config.hidden_dim, config.engram_dim, bias=False)
        self.retrieval_norm = nn.LayerNorm(config.engram_dim, eps=config.rms_norm_eps)
        self.key_proj = nn.Linear(config.engram_dim, config.hidden_dim, bias=False)
        self.value_proj = nn.Linear(config.engram_dim, config.hidden_dim, bias=False)
        self.gate_norm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.key_norm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.conv1d = nn.Conv1d(
            in_channels=config.hidden_dim,
            out_channels=config.hidden_dim,
            kernel_size=conv_kernel_size,
            groups=config.hidden_dim,
            bias=False,
            padding=0,
        )
        # Match EngramGatedResidual's zero-init on the causal depthwise
        # conv (production path has it in HarmonyModel._init_weights()
        # but this module is attached AFTER _init_weights runs, so we
        # must zero conv1d here). Without this, the hard gate clamp
        # during warmup injects random conv-filtered noise into the
        # residual stream from step 0, undermining the anti-collapse
        # strategy that depends on a quiet start.
        nn.init.zeros_(self.conv1d.weight)

        self.register_buffer(
            "current_step", torch.zeros((), dtype=torch.long), persistent=False,
        )
        # Cache the current step as a Python int to avoid per-forward
        # GPU->CPU sync via .item().
        self._current_step_int: int = 0

        self.use_head_gates = use_head_gates
        if use_head_gates:
            if config.hidden_dim % config.num_query_heads != 0:
                raise ValueError(
                    f"hidden_dim {config.hidden_dim} must be divisible by "
                    f"num_query_heads {config.num_query_heads} when "
                    "use_head_gates=True"
                )
            self.head_gates = nn.Parameter(torch.zeros(config.num_query_heads))
            self._num_heads = config.num_query_heads
            self._head_dim = config.hidden_dim // config.num_query_heads

    def set_step(self, step: int) -> None:
        """Update the current training step for gate-clamp scheduling."""
        step_int = int(step)
        self.current_step.fill_(step_int)
        self._current_step_int = step_int

    def _refresh_table_normalized(self) -> None:
        """Recompute the cached normalized table from `self.table`.

        Call this whenever `self.table` is replaced (e.g., after
        `load_state_dict` replaces the buffer, or if a caller mutates
        the table in place). The normalized cache is non-persistent and
        must stay in sync for retrieval to be correct.
        """
        self.table_normalized = F.normalize(self.table, dim=-1, eps=1e-8)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata,
        strict, missing_keys, unexpected_keys, error_msgs,
    ):
        """Refresh the normalized-table cache after state-dict loading.

        The base table is non-persistent so it usually isn't in the
        state_dict, but if a caller attaches a module and then loads a
        checkpoint that DOES contain the table (legacy checkpoints), we
        must re-derive `table_normalized` to keep retrieval consistent.
        """
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata,
            strict, missing_keys, unexpected_keys, error_msgs,
        )
        # Always refresh: cheap for research-sized tables and keeps
        # invariants explicit.
        self._refresh_table_normalized()

    @property
    def total_entries(self) -> int:
        return int(self.table.shape[0])

    def retrieve(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Brute-force differentiable retrieval over the corpus table.

        Uses softmax attention (temperature = `retrieval_temperature`) over
        cosine similarities. A sharp temperature (default `1/sqrt(engram_dim)`)
        makes the attention approximate k=1 retrieval while preserving
        gradient flow through `query_proj` - essential for the projection
        to actually learn. At inference the softmax can be replaced by
        hard argmax with minimal accuracy loss once the projection has
        converged.

        Args:
            hidden_state: [batch, seq_len, hidden_dim]

        Returns:
            [batch, seq_len, engram_dim] - softmax-attended blend of
            table rows (raw, not layer-normed; caller applies
            `retrieval_norm`).
        """
        q = self.query_proj(hidden_state)
        q_norm = F.normalize(q, dim=-1, eps=1e-8)
        batch, seq_len = q.shape[0], q.shape[1]
        retrieval_product = batch * seq_len * self.total_entries
        global _ann_retrieval_warned
        if (
            retrieval_product >= _ANN_RETRIEVAL_SIZE_WARN_THRESHOLD
            and not _ann_retrieval_warned
        ):
            warnings.warn(
                f"EngramANNInjection full-table softmax retrieval allocates "
                f"[batch, seq_len, total_entries] = [{batch}, {seq_len}, "
                f"{self.total_entries}] = {retrieval_product:,} elements "
                f"per forward pass. This scales linearly in table size. "
                f"Consider chunked top-k retrieval before growing further.",
                RuntimeWarning,
                stacklevel=2,
            )
            _ann_retrieval_warned = True
        # Cosine similarity [batch, seq_len, total_entries]
        sims = torch.einsum("ble,te->blt", q_norm, self.table_normalized)
        weights = F.softmax(sims / self.retrieval_temperature, dim=-1)
        # Weighted blend of table rows [batch, seq_len, engram_dim]
        return weights @ self.table

    def retrieve_argmax(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Inference-only hard k=1 retrieval (non-differentiable).

        Included for future evaluation of how much accuracy is lost when
        softmax is replaced by argmax after training. Not used in the
        standard forward path.
        """
        q = self.query_proj(hidden_state)
        q_norm = F.normalize(q, dim=-1, eps=1e-8)
        sims = torch.einsum("ble,te->blt", q_norm, self.table_normalized)
        idx = sims.argmax(dim=-1)
        return self.table[idx]

    def forward(
        self, hidden_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute gated residual with ANN retrieval.

        Returns:
            (residual, gate):
              residual: [batch, seq_len, hidden_dim] - caller adds to h.
              gate:     [batch, seq_len, 1] - post-clamp probabilities,
                        used by the training loop for entropy and
                        reconstruction losses.
        """
        retrieved = self.retrieve(hidden_state)
        retrieved = self.retrieval_norm(retrieved)

        key = self.key_proj(retrieved)
        value = self.value_proj(retrieved)

        h_norm = self.gate_norm(hidden_state)
        k_norm = self.key_norm(key)

        dot = (h_norm * k_norm).sum(dim=-1, keepdim=True)
        raw_gate = torch.sigmoid(dot / (self.hidden_dim ** 0.5))

        if self._current_step_int < self.clamp_until_step:
            gate = torch.clamp(raw_gate, min=self.clamp_min)
        else:
            gate = raw_gate

        gated_value = gate * value

        if self.use_head_gates:
            B, L, _ = gated_value.shape
            H, D = self._num_heads, self._head_dim
            gated_value = gated_value.view(B, L, H, D)
            gate_weights = torch.sigmoid(self.head_gates).view(1, 1, H, 1)
            gated_value = (gated_value * gate_weights).view(B, L, H * D)

        ncl = gated_value.transpose(1, 2)
        padded = F.pad(ncl, (self.conv_kernel_size - 1, 0))
        conv_out = self.conv1d(padded)
        nlc = conv_out.transpose(1, 2)

        return F.silu(nlc), gate

    @staticmethod
    def load_corpus_table(
        path: str | Path,
        tensor_name: str = "engram.weight",
    ) -> torch.Tensor:
        """Load the corpus table from a safetensors file as a CPU tensor."""
        from safetensors.torch import load_file
        tensors = load_file(str(path))
        if tensor_name not in tensors:
            raise KeyError(
                f"Tensor '{tensor_name}' not found in {path}; "
                f"available: {list(tensors.keys())}"
            )
        return tensors[tensor_name].float()


def compute_gate_entropy_loss(
    gate: torch.Tensor, eps: float = 1e-6,
) -> torch.Tensor:
    """Bernoulli-entropy loss to prevent gate collapse.

    Minimizing `-H(p)` where `H(p) = -[p log p + (1-p) log(1-p)]` is
    equivalent to maximizing gate-probability entropy, keeping the gate
    indecisive and preserving gradient flow through the memory path. See
    s3.3.1 of the research report.
    """
    p = gate.clamp(eps, 1.0 - eps)
    bernoulli_entropy = -(p * p.log() + (1.0 - p) * (1.0 - p).log())
    return -bernoulli_entropy.mean()


# ---------------------------------------------------------------------------
# ZEB-117 Model delta: cross-attention to external memory with top-k retrieval
# ---------------------------------------------------------------------------
#
# Research-only: NOT mirrored in Rust (crates/harmony-inference). Configs
# that use this are not GGUF-portable.
#
# Design rationale:
#   - Memorizing-Transformer-style injection: dedicated cross-attention
#     block with independent W_q/W_k/W_v/W_o projections, attached at the
#     configured injection layer, residual-adds to the hidden state.
#   - Top-k retrieval (k>1): each position attends over its own k nearest
#     corpus neighbors, letting the softmax pick the useful ones rather
#     than being forced into a pre-blended k=1 summary as in gamma.
#   - W_o zero-initialized: the residual add is a no-op at step 0, so
#     training starts from the alpha baseline and ramps injection in
#     smoothly as W_o learns. This replaces gamma's gate-clamp + entropy
#     regularization with a structurally simpler anti-collapse mechanism.
#   - Differentiable retrieval bias: top-k gather is non-differentiable,
#     so the retrieval projection would not learn from the main loss if
#     the cross-attention only saw Q*K scores. We add the raw top-k
#     cosine similarities as an additive bias to the attention logits,
#     preserving gradient flow into `retrieval_query_proj`. Bias weight
#     is configurable.


class EngramCrossAttention(nn.Module):
    """Cross-attention injection to a fixed corpus memory table.

    Attached to `HarmonyModel` at `config.engram_injection_layer` via
    `HarmonyModel.attach_engram_xattn()`. Produces a residual the caller
    adds to the hidden state:

        h = h + engram_xattn(h)

    Unlike `EngramANNInjection` this module has no gate, no clamp, and
    no entropy side-channel. Anti-collapse is structural: `o_proj.weight`
    starts at zero, so the residual is identically zero at step 0 and
    the model trains as if unattached until `o_proj` learns a nontrivial
    direction.

    The corpus table is a non-persistent buffer (checkpoints exclude it;
    caller must re-load via `load_corpus_table()` on resume), mirroring
    the gamma contract so both models share a single corpus-table file.
    """

    def __init__(
        self,
        config: HarmonyModelConfig,
        table: torch.Tensor,
        num_heads: int | None = None,
        k_retrieved: int = 8,
        retrieval_temperature: float | None = None,
        retrieval_bias_weight: float = 1.0,
        use_head_gates: bool = False,
    ):
        super().__init__()
        if table.dim() != 2:
            raise ValueError(
                f"table must be 2-D [total_entries, engram_dim], got {table.shape}"
            )
        if table.shape[1] != config.engram_dim:
            raise ValueError(
                f"table engram_dim {table.shape[1]} != config.engram_dim "
                f"{config.engram_dim}"
            )

        resolved_heads = config.num_query_heads if num_heads is None else int(num_heads)
        if resolved_heads <= 0:
            raise ValueError(f"num_heads must be > 0, got {resolved_heads}")
        if config.hidden_dim % resolved_heads != 0:
            raise ValueError(
                f"hidden_dim {config.hidden_dim} must be divisible by "
                f"num_heads {resolved_heads}"
            )
        if k_retrieved <= 0:
            raise ValueError(f"k_retrieved must be > 0, got {k_retrieved}")
        if k_retrieved > table.shape[0]:
            raise ValueError(
                f"k_retrieved {k_retrieved} exceeds table size {table.shape[0]}"
            )
        if not math.isfinite(retrieval_bias_weight):
            raise ValueError(
                f"retrieval_bias_weight must be finite, got {retrieval_bias_weight!r}"
            )

        self.hidden_dim = config.hidden_dim
        self.engram_dim = config.engram_dim
        self.num_heads = resolved_heads
        self.head_dim = config.hidden_dim // resolved_heads
        self.k_retrieved = int(k_retrieved)
        self.retrieval_bias_weight = float(retrieval_bias_weight)
        self.retrieval_temperature = (
            float(retrieval_temperature)
            if retrieval_temperature is not None
            else 1.0 / (config.engram_dim ** 0.5)
        )
        if (
            not math.isfinite(self.retrieval_temperature)
            or self.retrieval_temperature <= 0
        ):
            raise ValueError(
                f"retrieval_temperature must be a positive finite float, "
                f"got {self.retrieval_temperature!r}"
            )

        # Shared corpus table buffers (non-persistent, excluded from
        # checkpoints). Kept in float32 on CPU/GPU alongside module
        # parameters via `.to(device)`.
        self.register_buffer("table", table.float(), persistent=False)
        self.register_buffer(
            "table_normalized",
            F.normalize(table.float(), dim=-1, eps=1e-8),
            persistent=False,
        )

        self.retrieval_query_proj = nn.Linear(
            config.hidden_dim, config.engram_dim, bias=False,
        )
        self.retrieval_norm = nn.LayerNorm(
            config.engram_dim, eps=config.rms_norm_eps,
        )

        # Independent Q/K/V/O projections. Attention inner dim is
        # hidden_dim (head_dim * num_heads); K/V project directly from
        # engram_dim rather than promoting retrieved embeddings to
        # hidden_dim first - keeps the parameter count close to the
        # Model-beta FFN-expansion control.
        self.q_proj = nn.Linear(
            config.hidden_dim, config.hidden_dim, bias=False,
        )
        self.k_proj = nn.Linear(
            config.engram_dim, config.hidden_dim, bias=False,
        )
        self.v_proj = nn.Linear(
            config.engram_dim, config.hidden_dim, bias=False,
        )
        self.o_proj = nn.Linear(
            config.hidden_dim, config.hidden_dim, bias=False,
        )
        # Per-head RMSNorm matching the main attention block's QK-norm
        # pattern. Stabilizes the dot product when retrieval magnitudes
        # drift from hidden-state magnitudes.
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Residual-zero: o_proj starts at zero so the module is a no-op
        # at step 0. HarmonyModel._init_weights() runs BEFORE this module
        # is attached, so we must zero-init here explicitly (same pattern
        # as EngramANNInjection.conv1d).
        nn.init.zeros_(self.o_proj.weight)

        self.use_head_gates = use_head_gates
        if use_head_gates:
            self.head_gates = nn.Parameter(torch.zeros(self.num_heads))

    def _refresh_table_normalized(self) -> None:
        """Recompute the normalized-table cache after `self.table` changes."""
        self.table_normalized = F.normalize(self.table, dim=-1, eps=1e-8)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata,
        strict, missing_keys, unexpected_keys, error_msgs,
    ):
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata,
            strict, missing_keys, unexpected_keys, error_msgs,
        )
        self._refresh_table_normalized()

    @property
    def total_entries(self) -> int:
        return int(self.table.shape[0])

    def retrieve_topk(
        self,
        hidden_state: torch.Tensor,
        return_indices: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieve top-k nearest corpus embeddings per position.

        The top-k gather is non-differentiable, but `topk_sims` is a
        differentiable function of `retrieval_query_proj.weight`. Adding
        `topk_sims` as an attention-logit bias in `forward()` preserves
        gradient flow into the retrieval projection.

        Args:
            hidden_state: [batch, seq_len, hidden_dim]
            return_indices: if True, also return the gather indices so callers
                (e.g. the V-contrastive branch) can reuse them without
                recomputing the similarity matmul and top-k.

        Returns:
            retrieved:  [batch, seq_len, k_retrieved, engram_dim]
            topk_sims:  [batch, seq_len, k_retrieved] (raw cosine sims)
            topk_idx:   [batch, seq_len, k_retrieved] (only when return_indices)
        """
        q = self.retrieval_query_proj(hidden_state)
        q_norm = F.normalize(q, dim=-1, eps=1e-8)
        sims = torch.einsum("ble,te->blt", q_norm, self.table_normalized)
        topk_sims, topk_idx = sims.topk(self.k_retrieved, dim=-1)
        retrieved = self.table[topk_idx]
        if return_indices:
            return retrieved, topk_sims, topk_idx
        return retrieved, topk_sims

    def _attention_block(
        self,
        hidden_state: torch.Tensor,
        retrieved: torch.Tensor,
        topk_sims: torch.Tensor,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run the post-retrieval attention pipeline on caller-supplied retrievals.

        Exposed so V-contrastive variants (θ-V-contrast, ZEB-130) can run
        the same attention/o_proj path against a shuffled-table retrieval
        without duplicating the Q/K/V/o_proj logic. Logic must stay
        bit-identical to what was previously inlined in `forward`.

        Args:
            hidden_state: [batch, seq_len, hidden_dim]
            retrieved:    [batch, seq_len, k_retrieved, engram_dim] — raw
                          (pre-`retrieval_norm`) retrieved rows.
            topk_sims:    [batch, seq_len, k_retrieved] — cosine sims to add
                          as the differentiable retrieval bias.
            return_attn:  if True, also return the [B, L, H, k] softmax
                          attention weights (for Q-div load-balancing aux).

        Returns:
            out:  [batch, seq_len, hidden_dim] residual (pre-gate).
            attn: [batch, seq_len, H, k] softmax weights (only when
                  return_attn=True).
        """
        B, L, _ = hidden_state.shape
        H, D, k = self.num_heads, self.head_dim, self.k_retrieved

        retrieved = self.retrieval_norm(retrieved)

        q = self.q_proj(hidden_state).view(B, L, H, D)
        q = self.q_norm(q)
        k_tensor = self.k_proj(retrieved).view(B, L, k, H, D)
        k_tensor = self.k_norm(k_tensor)
        v_tensor = self.v_proj(retrieved).view(B, L, k, H, D)

        scores = torch.einsum("blhd,blkhd->blhk", q, k_tensor) / (D ** 0.5)
        # Broadcast the retrieval-similarity bias across heads. This is the
        # only path through which `retrieval_query_proj` receives gradient
        # from the main loss (topk gather blocks the rest).
        scores = scores + self.retrieval_bias_weight * topk_sims.unsqueeze(2)
        attn = F.softmax(scores, dim=-1)

        out = torch.einsum("blhk,blkhd->blhd", attn, v_tensor)

        if self.use_head_gates:
            gate_weights = torch.sigmoid(self.head_gates).view(1, 1, H, 1)
            out = out * gate_weights

        out = self.o_proj(out.reshape(B, L, H * D))

        if return_attn:
            return out, attn
        return out

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Compute cross-attention residual for the injection layer.

        Args:
            hidden_state: [batch, seq_len, hidden_dim]

        Returns:
            [batch, seq_len, hidden_dim] residual. Zero at step 0 due to
            `o_proj` zero-init; caller adds to hidden state.
        """
        retrieved, topk_sims = self.retrieve_topk(hidden_state)
        return self._attention_block(hidden_state, retrieved, topk_sims)

    @staticmethod
    def load_corpus_table(
        path: str | Path,
        tensor_name: str = "engram.weight",
    ) -> torch.Tensor:
        """Load the corpus table from a safetensors file as a CPU tensor.

        Alias for `EngramANNInjection.load_corpus_table`; duplicated here
        so callers don't need to know which research-only engram module
        they're using to load the shared table.
        """
        return EngramANNInjection.load_corpus_table(path, tensor_name)


class EngramConsolidationDecoder(nn.Module):
    """Lightweight decoder for engram consolidation (ZEB-128).

    Predicts the engram module's residual output from the model's own
    hidden state. Used as an auxiliary MSE training target to force
    internalization of the engram signal into parametric knowledge.
    Discarded after training — never saved or shipped.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        nn.init.xavier_uniform_(self.net[0].weight)
        nn.init.zeros_(self.net[0].bias)
        nn.init.xavier_uniform_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_state)


class GatedEngramInjection(nn.Module):
    """Gated engram cross-attention injection with optional training-only
    auxiliary losses (η-B / ZEB-130).

    Wraps an ``EngramCrossAttention`` and applies a learnable scalar ``alpha``
    through ``tanh`` to the xattn output. When ``alpha_init=0`` the gate
    outputs zero on the first forward pass, so a freshly attached
    ``GatedEngramInjection`` produces no perturbation until the optimizer
    opens the gate — the capacity-gap experiment relies on this to preserve
    a frozen pretrained baseline's step-0 behavior.

    Forward: ``h_out = tanh(alpha) * xattn(h)``.

    The model's forward loop adds this to the residual stream via
    ``h = h + engram_inject_mult * wrapper(h)`` so that the global
    ``engram_inject_mult`` (used by ``--zero-injection-eval``) still zeroes
    out the injection regardless of gate state.

    Note: the wrapper takes over the step-0 no-op job from the wrapped
    ``EngramCrossAttention``, so ``__init__`` re-initializes
    ``engram_xattn.o_proj`` from the default zero-init to ``xavier_uniform``.
    Without that, both ``tanh(alpha_init)`` and ``o_proj`` starting at zero
    creates a gradient dead zone (both partial derivatives vanish) and the
    injection can never learn.

    Two independent aux-loss hooks can be attached via caller-supplied sink
    lists (each of which the training loop drains once per optimizer step and
    scales by its own lambda x warmup schedule):

    - vcontrast_sink: V-contrastive aux (PR #250, ZEB-130 theta). On every
      training forward, runs a second xattn with a per-step random row-
      permuted value branch; appends (cos(inj_real, inj_shuf) ** 2).mean().
    - qdiv_sink: Q-div aux (ZEB-130 iota). Captures the softmax attention
      weights and top-k indices from the main forward and computes the
      MoE load-balancing loss N * sum_i f[i] * P[i]; appends the scalar.

    When both sinks are provided, both run independently; their losses are
    summed into the training objective with separate lambdas. Passing None
    (or any aux sink but calling model.eval()) disables the corresponding
    aux path entirely (no extra compute beyond the baseline forward).

    shuffle_generator is an optional dedicated RNG for V-contrast's per-step
    row permutation (reproducibility debugging only); used only when
    vcontrast_sink is not None.
    """

    def __init__(
        self,
        engram_xattn: EngramCrossAttention,
        alpha_init: float = 0.0,
        *,
        vcontrast_sink: list[torch.Tensor] | None = None,
        qdiv_sink: list[torch.Tensor] | None = None,
        shuffle_generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        self.engram_xattn = engram_xattn
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        # Held by reference: the training script owns each list (so it can
        # clear it between optimizer steps and stack the per-layer scalars)
        # and the wrappers append to them.
        self._vcontrast_sink: list[torch.Tensor] | None = vcontrast_sink
        self._qdiv_sink: list[torch.Tensor] | None = qdiv_sink
        self._shuffle_generator: torch.Generator | None = shuffle_generator

        # Break the dual-zero-init gradient dead zone. EngramCrossAttention
        # zero-inits o_proj.weight as its own anti-collapse mechanism (used by
        # the legacy δ path, where there is no gate). When we wrap it with a
        # gate whose tanh(alpha_init)=0 (the default), the gate itself becomes
        # the step-0 no-op guarantee, so the wrapped o_proj=0 contract is
        # redundant — and actively harmful. With both the gate AND o_proj at
        # zero, forward = tanh(0) * o_proj @ attn_out = 0, AND both partial
        # derivatives vanish:
        #   ∂/∂alpha   = sech²(0) * (o_proj @ attn_out) = 1 * 0 = 0
        #   ∂/∂o_proj  = tanh(0)  * attn_out           = 0 * anything = 0
        # So neither param ever receives gradient and the injection stays
        # permanently closed. Re-init o_proj to xavier_uniform so the gate is
        # the *only* zero: gradients can then flow to alpha (via the non-zero
        # xattn_out) and once alpha opens, to o_proj itself.
        nn.init.xavier_uniform_(self.engram_xattn.o_proj.weight)
        if self.engram_xattn.o_proj.bias is not None:
            nn.init.zeros_(self.engram_xattn.o_proj.bias)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        xattn = self.engram_xattn
        need_vcontrast = self.training and self._vcontrast_sink is not None
        need_qdiv = self.training and self._qdiv_sink is not None

        if need_vcontrast or need_qdiv:
            # One retrieval covers both aux paths when both are enabled.
            retrieved_real, topk_sims_real, topk_idx_real = xattn.retrieve_topk(
                hidden_state, return_indices=True,
            )
        else:
            retrieved_real, topk_sims_real = xattn.retrieve_topk(hidden_state)

        # Main injection forward — conditionally capture attn weights for Q-div.
        if need_qdiv:
            inj_real, attn_weights = xattn._attention_block(
                hidden_state, retrieved_real, topk_sims_real, return_attn=True,
            )
            self._qdiv_sink.append(
                compute_qdiv_aux(topk_idx_real, attn_weights, xattn.table.shape[0])
            )
        else:
            inj_real = xattn._attention_block(
                hidden_state, retrieved_real, topk_sims_real,
            )

        # V-contrast aux — shuffled-value second forward (training only).
        if need_vcontrast:
            N = xattn.table.shape[0]
            gen = self._shuffle_generator
            if gen is not None:
                perm = torch.randperm(N, generator=gen, device=gen.device)
                if perm.device != xattn.table.device:
                    perm = perm.to(xattn.table.device)
            else:
                perm = torch.randperm(N, device=xattn.table.device)
            shuf_idx = perm[topk_idx_real]
            retrieved_shuf = xattn.table[shuf_idx]
            inj_shuf = xattn._attention_block(
                hidden_state, retrieved_shuf, topk_sims_real,
            )
            # Mean-squared cosine across [B, L]. Smooth, bounded [0, 1],
            # natural attractor at cos=0. Avoids the anti-alignment
            # pathology of signed cosine minimization.
            cos = F.cosine_similarity(inj_real, inj_shuf, dim=-1)
            self._vcontrast_sink.append((cos ** 2).mean())

        # Under bf16 autocast, inj_real is bf16 but self.alpha stays fp32;
        # a plain multiply would promote the product to fp32 and defeat AMP
        # for the rest of the injected layer. Cast the gate into inj_real's
        # dtype so the residual stays in low precision.
        gate = torch.tanh(self.alpha).to(dtype=inj_real.dtype)
        return gate * inj_real


class ContrastiveGatedEngramInjection(GatedEngramInjection):
    """V-contrastive engram injection (θ-V-contrast, ZEB-130).

    Subclasses ``GatedEngramInjection`` to add a training-only auxiliary
    loss path. On every forward in training mode, runs a second xattn
    pipeline against a per-step random row-permutation of the primary
    table, then appends ``(cos(inj_real, inj_shuf)**2).mean()`` to the
    caller-supplied ``aux_loss_sink`` list. The shuffled branch never
    contributes to the residual — only the parent's primary-branch
    output is gated and returned.

    The aux loss measures alignment at the post-``o_proj`` pre-gate level,
    so a shrinking gate (``tanh(alpha) -> 0``) cannot be used to minimize
    it — the gate's role is firing rate, the aux loss's role is
    response-shape. They must remain independent.

    Cheap permutation: ``F.normalize`` is row-permute-equivariant, so we
    use the cached ``self.engram_xattn.table_normalized`` indexed by the
    fresh ``perm`` instead of re-normalizing per step.
    """

    def __init__(
        self,
        engram_xattn: EngramCrossAttention,
        alpha_init: float = 0.0,
        aux_loss_sink: list[torch.Tensor] | None = None,
        shuffle_generator: torch.Generator | None = None,
    ) -> None:
        super().__init__(engram_xattn, alpha_init=alpha_init)
        # Held by reference: the training script owns the list (so it can
        # clear it between optimizer steps and stack the per-layer scalars)
        # and the wrappers append to it. A None sink disables the aux
        # branch — used so HarmonyModel can construct the contrastive
        # variant before the sink is wired up if needed.
        self._aux_sink: list[torch.Tensor] | None = aux_loss_sink
        # Optional dedicated RNG for the per-step row permutation. When None,
        # the global PyTorch RNG is used (preferred for production). A seeded
        # generator is for reproducibility debugging only.
        self._shuffle_generator: torch.Generator | None = shuffle_generator

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        xattn = self.engram_xattn

        if self.training and self._aux_sink is not None:
            # Reuse the real-branch top-k indices for the shuffled branch so
            # we don't re-run the similarity matmul and top-k gather twice.
            retrieved_real, topk_sims_real, topk_idx_real = xattn.retrieve_topk(
                hidden_state, return_indices=True,
            )
            inj_real = xattn._attention_block(hidden_state, retrieved_real, topk_sims_real)

            # Per-step random row-permutation of the primary table (value
            # shuffle). We keep the retrieval *keys* fixed (same top-k indices
            # as the real branch) but supply content from a permuted table so
            # the shuffled branch sees genuinely different rows at those
            # positions. A full key+value permutation would be a semantic no-op:
            # top-k selection is permutation-equivariant, so the same logical
            # rows would win and the injections would be identical.
            #
            # Index-only shuffle: apply the permutation to the gather indices
            # (`perm[topk_idx_real]`) instead of materializing a full [N, D]
            # `table_shuf` and then re-gathering k rows from it. Same result,
            # O(B·L·k) rather than O(N·D) bandwidth per forward.
            N = xattn.table.shape[0]
            gen = self._shuffle_generator
            if gen is not None:
                perm = torch.randperm(N, generator=gen, device=gen.device)
                if perm.device != xattn.table.device:
                    perm = perm.to(xattn.table.device)
            else:
                perm = torch.randperm(N, device=xattn.table.device)
            shuf_idx = perm[topk_idx_real]
            retrieved_shuf = xattn.table[shuf_idx]
            inj_shuf = xattn._attention_block(hidden_state, retrieved_shuf, topk_sims_real)

            # Mean-squared cosine across [B, L]. Smooth, bounded [0, 1],
            # natural attractor at cos=0. Avoids the anti-alignment
            # pathology of signed cosine minimization.
            cos = F.cosine_similarity(inj_real, inj_shuf, dim=-1)
            aux_loss = (cos ** 2).mean()
            self._aux_sink.append(aux_loss)
        else:
            retrieved_real, topk_sims_real = xattn.retrieve_topk(hidden_state)
            inj_real = xattn._attention_block(hidden_state, retrieved_real, topk_sims_real)

        gate = torch.tanh(self.alpha).to(dtype=inj_real.dtype)
        return gate * inj_real
