"""Latent projection for semantic Engram key generation.

Python port of crates/harmony-inference/src/latent_projection.rs.

Projects token embeddings through a 2-layer MLP (SiLU + tanh) to produce
compact latent codes, then binarizes via sign bits for locality-sensitive
hashing.  Replaces xxhash64 token-byte hashing when projection weights are
available.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from ct87.engram import EngramTable


def clean_projection_state_dict(
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Strip common prefixes and filter to layer1/layer2 keys.

    Handles state dicts with prefixes like 'module.', 'latent_projection.',
    'projection.' (including nested combinations), and filters out unrelated
    keys from full-model checkpoints.

    Used by both LatentProjection.from_checkpoint() and export_gguf.py.
    """
    cleaned: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        clean_key = k
        while True:
            stripped = False
            for prefix in ("module.", "latent_projection.", "projection."):
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix):]
                    stripped = True
                    break
            if not stripped:
                break
        if clean_key.startswith(("layer1.", "layer2.")):
            cleaned[clean_key] = v
    return cleaned


class LatentProjection(nn.Module):
    """2-layer MLP that projects token embeddings to a compact latent space.

    Architecture (matches Rust LatentProjection exactly):
        Linear(hidden_dim → intermediate_dim)
        → SiLU
        → Linear(intermediate_dim → latent_dim)
        → Tanh

    The tanh output is binarized via sign bits to produce binary keys
    for Engram table lookup.
    """

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        latent_dim: int,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.latent_dim = latent_dim

        self.layer1 = nn.Linear(hidden_dim, intermediate_dim)
        self.layer2 = nn.Linear(intermediate_dim, latent_dim)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Project embeddings to latent space.

        Args:
            embeddings: [..., hidden_dim]

        Returns:
            Latent codes: [..., latent_dim], values in [-1, 1] (tanh).
        """
        h = F.silu(self.layer1(embeddings))
        return torch.tanh(self.layer2(h))

    # Bit-position weights for vectorized LSB-first packing: [1, 2, 4, ..., 128]
    _BIT_WEIGHTS = torch.tensor([1 << i for i in range(8)], dtype=torch.uint8)

    def to_binary_keys(self, latent: torch.Tensor) -> list[bytes]:
        """Binarize latent codes via sign bits.

        Matches Rust LatentProjection::to_binary_keys():
        - Bit i is set if latent[i] >= 0.0
        - Bits are packed LSB-first within each byte (bit 0 = LSB)
        - Output is ceil(latent_dim / 8) bytes per vector

        Args:
            latent: [num_vectors, latent_dim]

        Returns:
            List of binary key bytes, one per vector.
        """
        key_bytes = (self.latent_dim + 7) // 8

        # Vectorized: sign bits -> bool -> pad to multiple of 8 -> pack with bit weights
        bits = (latent.detach().cpu() >= 0).to(torch.uint8)  # [N, latent_dim]
        n = bits.shape[0]

        # Pad to multiple of 8 if needed
        pad = key_bytes * 8 - self.latent_dim
        if pad > 0:
            bits = torch.nn.functional.pad(bits, (0, pad))

        # Reshape to [N, key_bytes, 8], multiply by bit weights, sum -> packed bytes
        packed = (bits.reshape(n, key_bytes, 8) * self._BIT_WEIGHTS).sum(dim=2).to(torch.uint8)
        packed_np = packed.numpy()

        return [row.tobytes() for row in packed_np]

    def project_ngrams(
        self,
        embeddings: torch.Tensor,
        seq_len: int,
    ) -> tuple[list[bytes], list[int]]:
        """Extract N-gram windows, average embeddings, project, and binarize.

        Mirrors Rust LatentProjection::project_ngrams():
        - Bigrams at positions 1..seq_len (each attributed to last token)
        - Trigrams at positions 2..seq_len (each attributed to last token)
        - Each N-gram's embeddings are averaged before projection

        Args:
            embeddings: [1, seq_len, hidden_dim] token embeddings
            seq_len: sequence length

        Returns:
            (binary_keys, positions) — binary_keys is a list of bytes objects,
            positions is the token position each key is attributed to.
        """
        parts, positions = compute_ngram_averages(embeddings, seq_len)
        if len(positions) == 0:
            return [], []

        # Project all N-grams in one batch
        with torch.no_grad():
            latent = self.forward(parts)  # [num_ngrams, latent_dim]
        keys = self.to_binary_keys(latent)

        return keys, positions

    @staticmethod
    def from_checkpoint(
        path: str | Path,
        hidden_dim: int,
        intermediate_dim: int,
        latent_dim: int,
        device: torch.device | str = "cpu",
    ) -> LatentProjection:
        """Load projection weights from a saved checkpoint.

        Supports both full state_dict .pt files and safetensors files.
        Looks for keys prefixed with 'latent_projection.' or bare
        'layer1.'/'layer2.' names.
        """
        path = Path(path)
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file

            state = load_file(str(path))
        else:
            state = torch.load(str(path), map_location="cpu", weights_only=True)

        cleaned = clean_projection_state_dict(state)

        proj = LatentProjection(hidden_dim, intermediate_dim, latent_dim)
        proj.load_state_dict(cleaned)
        return proj.to(device)


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

    # original is detached — compute its similarities without autograd overhead
    with torch.no_grad():
        sim_orig = _cosine_sim(original)
    sim_proj = _cosine_sim(projected)

    # Scale projected similarities by temperature, mask diagonal
    logits = sim_proj / temperature
    diag_mask = torch.full((n, n), 0.0, device=projected.device)
    diag_mask.fill_diagonal_(-1e9)
    logits = logits + diag_mask

    # Build soft targets from original similarities: keep only top-k per row
    with torch.no_grad():
        orig_data = sim_orig.clone()
        orig_data.fill_diagonal_(float("-inf"))  # exclude self
        _, top_indices = orig_data.topk(k, dim=1)

        target_logits = torch.full((n, n), float("-inf"), device=projected.device)
        target_logits.scatter_(1, top_indices, sim_orig.gather(1, top_indices))
        targets = torch.softmax(target_logits, dim=1)

    # Cross-entropy: -sum(targets * log_softmax(logits)) / N
    log_probs = torch.log_softmax(logits, dim=1)
    loss = -(targets * log_probs).sum() / n

    return loss


def compute_key_overlap(
    model: nn.Module,
    projection: LatentProjection,
    engram_table: EngramTable,
    input_ids: torch.Tensor,
) -> dict[str, float | int]:
    """Compare xxhash vs projection key distributions for a batch.

    For each (n-gram, head) combination, computes table indices via both
    xxhash (token bytes) and projection (binary keys), then measures how
    often they agree.

    Args:
        model: HarmonyModel (needs embed_tokens for embedding lookup)
        projection: trained LatentProjection module
        engram_table: EngramTable with hash_seeds and total_entries
        input_ids: [batch, seq_len] token IDs

    Returns:
        Dict with keys:
        - total_lookups: total (n-gram, head) pairs compared
        - matching_indices: how many produced the same table index
        - overlap_pct: matching / total as a percentage
        - unique_xxhash: number of unique table indices from xxhash
        - unique_projection: number of unique table indices from projection
    """
    from ct87.engram import _hash_ngram, _xxhash64

    batch_size, seq_len = input_ids.shape
    total_entries = engram_table.total_entries
    seeds = engram_table.hash_seeds

    total_lookups = 0
    matching = 0
    xxhash_indices: set[int] = set()
    proj_indices: set[int] = set()

    proj_device = next(projection.parameters()).device

    with torch.no_grad():
        embeddings = model.embed_tokens(input_ids)  # [batch, seq_len, hidden_dim]

    for b in range(batch_size):
        tokens = input_ids[b].tolist()
        emb = embeddings[b : b + 1].to(proj_device)  # [1, seq_len, hidden_dim]

        # Get projection keys + positions
        proj_keys, proj_positions = projection.project_ngrams(emb, seq_len)

        # Build a map: position → list of projection binary keys at that position.
        # project_ngrams() returns all bigram keys first, then all trigram keys,
        # so pos_to_proj_keys[pos][0] = bigram key, [1] = trigram key (if exists).
        pos_to_proj_keys: dict[int, list[bytes]] = {}
        for key, pos in zip(proj_keys, proj_positions):
            pos_to_proj_keys.setdefault(pos, []).append(key)

        # Compare xxhash vs projection for each n-gram
        # Bigrams: tokens[i:i+2] attributed to position i+1
        for i in range(seq_len - 1):
            pos = i + 1
            bigram = [tokens[i], tokens[i + 1]]
            proj_keys_at_pos = pos_to_proj_keys.get(pos, [])
            if not proj_keys_at_pos:
                continue
            proj_key = proj_keys_at_pos[0]  # bigram key

            for seed in seeds:
                xx_idx = _hash_ngram(bigram, seed) % total_entries
                pr_idx = _xxhash64(proj_key, seed) % total_entries

                xxhash_indices.add(xx_idx)
                proj_indices.add(pr_idx)
                total_lookups += 1
                if xx_idx == pr_idx:
                    matching += 1

        # Trigrams: tokens[i:i+3] attributed to position i+2
        for i in range(seq_len - 2):
            pos = i + 2
            trigram = [tokens[i], tokens[i + 1], tokens[i + 2]]
            proj_keys_at_pos = pos_to_proj_keys.get(pos, [])
            if len(proj_keys_at_pos) < 2:
                continue
            proj_key = proj_keys_at_pos[1]  # trigram key

            for seed in seeds:
                xx_idx = _hash_ngram(trigram, seed) % total_entries
                pr_idx = _xxhash64(proj_key, seed) % total_entries

                xxhash_indices.add(xx_idx)
                proj_indices.add(pr_idx)
                total_lookups += 1
                if xx_idx == pr_idx:
                    matching += 1

    overlap_pct = (matching / total_lookups * 100) if total_lookups > 0 else 0.0

    return {
        "total_lookups": total_lookups,
        "matching_indices": matching,
        "overlap_pct": overlap_pct,
        "unique_xxhash": len(xxhash_indices),
        "unique_projection": len(proj_indices),
    }
