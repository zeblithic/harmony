"""COCONUT continuous thought training for ct87.

Implements the training-side scaffolding for Phase 4b continuous thought:
hidden states are fed back as input embeddings at designated think positions,
letting the model reason in continuous latent space rather than projecting
through the vocabulary at every step.

The Rust inference side (Phase 4a) handles runtime routing via the UQ head
in crates/harmony-inference/src/continuous_thought.rs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ct87.model import HarmonyModel, RMSNorm


class ThoughtNorm(nn.Module):
    """Normalize hidden states for feedback as input embeddings.

    Applies RMSNorm + learnable scalar gate. The gate starts at sigmoid(-2)
    ~ 0.12 so early training mostly ignores thought feedback, providing
    stability while the model learns to use continuous thought.
    """

    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        super().__init__()
        self.norm = RMSNorm(hidden_dim, eps=eps)
        self.gate_bias = nn.Parameter(torch.tensor(-2.0))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_bias)
        return gate * self.norm(h)


class CurriculumSchedule:
    """Step-function curriculum: gradually increase thought steps.

    Divides training into (max_steps + 1) equal stages. Stage 0 has 0
    thoughts, stage 1 has 1, etc., up to max_steps.
    """

    def __init__(self, max_steps: int, total_train_steps: int):
        self.max_steps = max_steps
        self.total_train_steps = total_train_steps
        if max_steps > 0:
            self.stage_length = max(1, total_train_steps // (max_steps + 1))
        else:
            self.stage_length = total_train_steps

    def num_thoughts(self, train_step: int) -> int:
        return min(train_step // self.stage_length, self.max_steps)


def insert_think_tokens(
    input_ids: torch.Tensor,
    think_token_id: int,
    num_thoughts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepend think tokens to a sequence.

    Args:
        input_ids: [batch, seq_len] token IDs
        think_token_id: token ID for <think>
        num_thoughts: number of think tokens to prepend

    Returns:
        augmented_ids: [batch, seq_len + num_thoughts]
        think_mask: [batch, seq_len + num_thoughts] bool, True at think positions
    """
    if num_thoughts == 0:
        return input_ids, torch.zeros(input_ids.shape, dtype=torch.bool, device=input_ids.device)

    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    think_prefix = torch.full(
        (batch_size, num_thoughts), think_token_id,
        dtype=input_ids.dtype, device=device,
    )
    augmented_ids = torch.cat([think_prefix, input_ids], dim=1)

    think_mask = torch.zeros(batch_size, seq_len + num_thoughts, dtype=torch.bool, device=device)
    think_mask[:, :num_thoughts] = True

    return augmented_ids, think_mask


def coconut_forward(
    model: HarmonyModel,
    thought_norm: ThoughtNorm,
    input_ids: torch.Tensor,
    think_token_id: int,
    num_thoughts: int,
    engram_embeddings: torch.Tensor | None = None,
    return_hidden_states: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """COCONUT 2-pass forward with continuous thought.

    Pass 1 (bootstrap): standard forward to get hidden states.
    Pass 2 (COCONUT): think positions receive ThoughtNorm(hidden_states)
    as input embeddings; causal attention creates the chain effect.

    Args:
        model: HarmonyModel instance
        thought_norm: ThoughtNorm module for hidden→embedding normalization
        input_ids: [batch, seq_len] token IDs (without think tokens)
        think_token_id: token ID for <think>
        num_thoughts: number of think tokens to prepend (0 = standard forward)
        engram_embeddings: optional [batch, seq_len, engram_dim] Engram
            embeddings (aligned to original input_ids, NOT augmented)
        return_hidden_states: if True, also return the pre-final_norm hidden
            states from Pass 2 (for MTP head training).

    Returns:
        logits: [batch, augmented_seq_len, vocab_size]
        think_mask: [batch, augmented_seq_len] bool
        hidden_states: (only if return_hidden_states=True)
            [batch, augmented_seq_len, hidden_dim]
    """
    if num_thoughts == 0:
        logits = model(input_ids=input_ids, engram_embeddings=engram_embeddings)
        think_mask = torch.zeros(input_ids.shape, dtype=torch.bool, device=input_ids.device)
        return logits, think_mask

    augmented_len = input_ids.shape[1] + num_thoughts
    if augmented_len > model.config.max_seq_len:
        raise ValueError(
            f"augmented sequence length ({augmented_len}) exceeds "
            f"model max_seq_len ({model.config.max_seq_len})"
        )

    augmented_ids, think_mask = insert_think_tokens(input_ids, think_token_id, num_thoughts)

    # Align engram embeddings: zero-pad at front for think positions
    padded_engram = None
    if engram_embeddings is not None:
        batch_size = engram_embeddings.shape[0]
        engram_dim = engram_embeddings.shape[2]
        pad = torch.zeros(batch_size, num_thoughts, engram_dim,
                          dtype=engram_embeddings.dtype, device=engram_embeddings.device)
        padded_engram = torch.cat([pad, engram_embeddings], dim=1)

    # Pass 1: bootstrap forward to get hidden states (discard logits to free VRAM)
    bootstrap_logits, hidden = model(
        input_ids=augmented_ids,
        engram_embeddings=padded_engram,
        return_hidden_states=True,
    )
    del bootstrap_logits
    # Detach hidden: gradients only flow through Pass 2 via ThoughtNorm
    hidden = hidden.detach()

    # Build input embeddings: ThoughtNorm(hidden) at think positions,
    # token embeddings at real positions
    normed_hidden = thought_norm(hidden[:, :num_thoughts, :])
    embeds_suffix = model.embed_tokens(augmented_ids[:, num_thoughts:])
    embeds = torch.cat([normed_hidden, embeds_suffix], dim=1)

    # Pass 2: COCONUT forward with continuous thought embeddings
    if return_hidden_states:
        logits, hidden = model(
            input_embeds=embeds,
            engram_embeddings=padded_engram,
            return_hidden_states=True,
        )
        return logits, think_mask, hidden

    logits = model(
        input_embeds=embeds,
        engram_embeddings=padded_engram,
    )
    return logits, think_mask


def coconut_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    think_mask: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy loss excluding think positions.

    Args:
        logits: [batch, augmented_seq_len, vocab_size]
        targets: [batch, augmented_seq_len] (values at think positions ignored)
        think_mask: [batch, augmented_seq_len] bool, True at think positions

    Returns:
        Scalar loss averaged over non-think positions.
    """
    # Set targets at think positions to -100 (PyTorch's ignore_index)
    masked_targets = targets.clone()
    masked_targets[think_mask] = -100

    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        masked_targets.reshape(-1),
        ignore_index=-100,
    )
