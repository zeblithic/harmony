"""Multi-Token Prediction head — shared-weight recursive design.

A single MLP block applied recursively to predict K future tokens from
the base model's hidden states.  Shares embed_tokens and lm_head with
the main model (no extra unembedding matrix).

Training (teacher-forced):
    For draft step k = 0 .. K-1:
        embed_k = embed_tokens(targets[:, k : k+S])
        h_{k+1} = mlp(input_norm(h_k[:, :S] + embed_k))
        logits_k = lm_head(output_norm(h_{k+1}))
        loss_k  = CE(logits_k, targets[:, k+1 : k+1+S])
    mtp_loss = mean(loss_0 .. loss_{K-1})

    where S = seq_len - K (truncated so all targets exist).

Inference (autoregressive drafting — deferred to ZEB-60):
    Same recursive structure but sampling from logits_k to get embed_{k+1}.

References:
    Gloeckle et al., "Better & Faster LLMs via Multi-token Prediction"
    FastMTP (arXiv 2509.18362)
    Nemotron 3 Super technical report
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ct87.model import HarmonyModelConfig, RMSNorm


class MtpHead(nn.Module):
    """Shared-weight recursive MTP head.

    Uses a SwiGLU MLP block (same architecture as the base model's FFN)
    with its own input/output RMSNorms.  Parameter overhead: ~2.8% of the
    base 474M model.

    The head does NOT own embed_tokens or lm_head — the caller passes them
    in as callables so that tied weights are shared automatically.
    """

    def __init__(self, config: HarmonyModelConfig, depth: int = 4):
        super().__init__()
        if depth < 1:
            raise ValueError(f"MTP depth must be >= 1, got {depth}")
        self.depth = depth
        self.hidden_dim = config.hidden_dim
        self.ffn_dim = config.ffn_dim

        self.input_norm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.output_norm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.gate_proj = nn.Linear(config.hidden_dim, config.ffn_dim, bias=False)
        self.up_proj = nn.Linear(config.hidden_dim, config.ffn_dim, bias=False)
        self.down_proj = nn.Linear(config.ffn_dim, config.hidden_dim, bias=False)

    def _mlp(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    def forward(
        self,
        hidden_states: torch.Tensor,
        targets: torch.Tensor,
        embed_fn: nn.Module,
        lm_head_fn: nn.Module,
        think_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute teacher-forced MTP loss.

        Args:
            hidden_states: [batch, seq_len, hidden_dim] — pre-final_norm
                output from the base model's last transformer layer.
            targets: [batch, seq_len] — target token IDs (same as the main
                LM loss targets: input shifted by 1).
            embed_fn: model.embed_tokens (nn.Embedding).
            lm_head_fn: model.lm_head (nn.Linear) — shared unembedding.
            think_mask: optional [batch, seq_len] bool mask, True at COCONUT
                think positions.  MTP loss is zeroed at these positions.

        Returns:
            Scalar MTP loss averaged across draft depths and valid positions.
        """
        batch, seq_len, _ = hidden_states.shape
        K = self.depth

        # Truncate so all K draft targets exist
        S = seq_len - K
        if S <= 0:
            return torch.tensor(0.0, device=hidden_states.device, requires_grad=True)

        h = hidden_states[:, :S, :]
        total_loss = torch.tensor(0.0, device=hidden_states.device)

        for k in range(K):
            # Token embedding at offset k (the "known" token for this draft step)
            tok_embed = embed_fn(targets[:, k : k + S])
            # Recursive MLP: h_{k+1} = mlp(norm(h_k + embed_k))
            h = self._mlp(self.input_norm(h + tok_embed))
            # Predict next token at offset k+1
            logits = lm_head_fn(self.output_norm(h))

            draft_targets = targets[:, k + 1 : k + 1 + S]

            if think_mask is not None:
                # Mask out think positions — use -100 (CE ignore_index)
                mask_slice = think_mask[:, k + 1 : k + 1 + S]
                draft_targets = draft_targets.clone()
                draft_targets[mask_slice] = -100

            step_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                draft_targets.reshape(-1),
                ignore_index=-100,
            )
            total_loss = total_loss + step_loss

        return total_loss / K
