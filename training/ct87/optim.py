"""Muon optimizer and WSD learning rate schedule for ct87 training.

Muon applies Newton-Schulz orthogonalization to the momentum buffer before
updating matrix-shaped parameters (Linear weights). Non-matrix parameters
(embeddings, norms, BlockAttnRes queries) use standard AdamW.

WSD (Warmup-Stable-Decay) is a piecewise linear LR schedule: linear warmup,
constant stable phase, linear decay.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn


def newton_schulz_orthogonalize(X: torch.Tensor, num_iters: int = 5) -> torch.Tensor:
    """Orthogonalize a matrix via Newton-Schulz iteration.

    Approximates the polar decomposition. 5 iterations is sufficient.
    For tall matrices (rows > cols), transposes before iterating to avoid
    rank-deficient intermediates, then transposes back.

    Args:
        X: matrix to orthogonalize [m, n]
        num_iters: number of iterations (default 5)

    Returns:
        Orthogonalized matrix with same shape.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    # Transpose tall matrices so NS operates on wide-or-square input.
    # X @ X.T must be full-rank for convergence; tall matrices produce
    # rank-deficient intermediates without this.
    transposed = False
    if X.shape[0] > X.shape[1]:
        X = X.T
        transposed = True
    # Normalize so spectral norm < 1 (required for convergence)
    X = X / (X.norm() + 1e-7)
    for _ in range(num_iters):
        A = X @ X.T
        X = a * X + b * (A @ X) + c * (A @ (A @ X))
    if transposed:
        X = X.T
    return X


def partition_params(model: nn.Module) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Partition model parameters into Muon (2D matrices) and Adam (everything else).

    Muon group: 2D weight matrices from Linear layers (q/k/v/o_proj, gate/up/down_proj).
    Adam group: embeddings, RMSNorm weights, BlockAttnRes queries, lm_head if tied.

    Returns:
        (muon_params, adam_params)
    """
    muon_params = []
    adam_params = []
    seen: set[int] = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in seen:
            continue  # Skip tied params (lm_head.weight == embed_tokens.weight)
        seen.add(id(param))

        if param.dim() == 2 and "embed" not in name and "norm" not in name:
            muon_params.append(param)
        else:
            adam_params.append(param)

    return muon_params, adam_params


class Muon(torch.optim.Optimizer):
    """Muon optimizer with AdamW fallback for non-matrix parameters.

    Args:
        muon_params: list of 2D parameters for Muon update rule
        adam_params: list of non-matrix parameters for AdamW
        lr: learning rate for Muon parameters
        momentum: Muon momentum coefficient (default: 0.95)
        adam_lr: learning rate for Adam parameters
        adam_betas: Adam beta coefficients (default: (0.9, 0.95))
        adam_eps: Adam epsilon (default: 1e-8)
        adam_wd: Adam weight decay (default: 0.0)
    """

    def __init__(
        self,
        muon_params: list[torch.Tensor],
        adam_params: list[torch.Tensor],
        lr: float = 3e-4,
        momentum: float = 0.95,
        adam_lr: float = 3e-4,
        adam_betas: tuple[float, float] = (0.9, 0.95),
        adam_eps: float = 1e-8,
        adam_wd: float = 0.0,
    ):
        defaults: dict[str, Any] = dict(lr=lr, momentum=momentum)
        params = []

        if muon_params:
            params.append({"params": muon_params, "lr": lr, "momentum": momentum, "is_muon": True})

        if adam_params:
            params.append({
                "params": adam_params,
                "lr": adam_lr,
                "betas": adam_betas,
                "eps": adam_eps,
                "weight_decay": adam_wd,
                "is_muon": False,
            })

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any = None) -> None:
        for group in self.param_groups:
            if group.get("is_muon", False):
                self._muon_step(group)
            else:
                self._adam_step(group)

    def _muon_step(self, group: dict) -> None:
        lr = group["lr"]
        momentum = group["momentum"]

        for p in group["params"]:
            if p.grad is None:
                continue

            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(p.data)

            buf = state["momentum_buffer"]
            buf.mul_(momentum).add_(p.grad)

            update = newton_schulz_orthogonalize(buf)

            # Aspect-ratio scaling: tall matrices (e.g. gate_proj [ffn_dim, hidden_dim])
            # need proportionally larger updates. Without this, MLP layers undertrain.
            # Reference: "Muon is Scalable" (arXiv:2502.16982).
            aspect = max(1, p.shape[0] / p.shape[1]) ** 0.5
            p.data.add_(update, alpha=-lr * aspect)

    def _adam_step(self, group: dict) -> None:
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        wd = group["weight_decay"]

        for p in group["params"]:
            if p.grad is None:
                continue

            state = self.state[p]
            if "step" not in state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p.data)
                state["exp_avg_sq"] = torch.zeros_like(p.data)

            state["step"] += 1
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]

            if wd != 0:
                p.data.mul_(1.0 - lr * wd)

            exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)

            bc1 = 1 - beta1 ** state["step"]
            bc2 = 1 - beta2 ** state["step"]
            step_size = lr / bc1

            denom = (exp_avg_sq.sqrt() / math.sqrt(bc2)).add_(eps)
            p.data.addcdiv_(exp_avg, denom, value=-step_size)


class WSDSchedule:
    """Warmup-Stable-Decay learning rate schedule.

    Three phases:
    1. Warmup: linear from 0 to max LR over warmup_steps
    2. Stable: constant at max LR
    3. Decay: linear from max LR to min_lr_ratio * max_lr over last
       decay_fraction * total_steps steps

    Args:
        warmup_steps: number of warmup steps
        total_steps: total number of training steps
        decay_fraction: fraction of total_steps for decay phase (default: 0.1)
        min_lr_ratio: minimum LR as fraction of max LR (default: 0.0)
    """

    def __init__(
        self,
        warmup_steps: int,
        total_steps: int,
        decay_fraction: float = 0.1,
        min_lr_ratio: float = 0.0,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.decay_steps = int(total_steps * decay_fraction)
        self.decay_start = total_steps - self.decay_steps
        self.min_lr_ratio = min_lr_ratio

    def get_lr_multiplier(self, step: int) -> float:
        """Get the learning rate multiplier for a given step.

        Returns a value in [min_lr_ratio, 1.0].
        """
        if step < self.warmup_steps:
            return step / self.warmup_steps if self.warmup_steps > 0 else 1.0
        elif step < self.decay_start:
            return 1.0
        else:
            progress = (step - self.decay_start) / self.decay_steps if self.decay_steps > 0 else 1.0
            progress = min(progress, 1.0)
            return 1.0 - (1.0 - self.min_lr_ratio) * progress
