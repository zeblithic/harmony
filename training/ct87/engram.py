"""Engram conditional memory — gated residual injection and table lookup.

Mirrors crates/harmony-inference/src/engram_residual.rs (EngramGatedResidual)
and crates/harmony-engram/src/hash.rs (xxhash64 N-gram lookups).

The EngramGatedResidual module takes hidden states and Engram embeddings,
applies gated projection + causal depthwise conv1d + SiLU, and returns a
residual tensor for injection into the transformer hidden state.

The EngramTable class loads a safetensors embedding table and performs
N-gram hash lookups to produce per-position Engram embeddings.
"""

from __future__ import annotations

import struct
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ct87.model import HarmonyModelConfig, RMSNorm

# Default depthwise conv1d kernel size (matches Rust default)
CONV_KERNEL_SIZE = 3


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
    crates/harmony-engram/src/hash.rs. The algorithm is fixed — any
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

    For training, the table is NOT a trainable parameter — it provides
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
        the target device — avoiding per-row CUDA kernel launches.

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

        for b in range(batch_size):
            emb = embeddings[b : b + 1]  # [1, seq_len, hidden_dim]
            proj_keys, proj_positions = projection.project_ngrams(emb, seq_len)

            for key_bytes, pos in zip(proj_keys, proj_positions):
                for seed in self.hash_seeds:
                    idx = _xxhash64(key_bytes, seed) % self.total_entries
                    batch_indices.append(b)
                    positions.append(pos)
                    table_indices.append(idx)

        result = torch.zeros(
            batch_size, seq_len, self.engram_dim, dtype=torch.float32,
        )

        if table_indices:
            idx_t = torch.tensor(table_indices, dtype=torch.long)
            embs = self.table[idx_t]  # [n, engram_dim]

            for i, (b, pos) in enumerate(zip(batch_indices, positions)):
                result[b, pos] += embs[i]

        return result.to(self.device)

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
