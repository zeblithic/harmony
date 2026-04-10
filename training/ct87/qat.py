"""Quantization-Aware Training (QAT) at q8_0 precision.

Simulates q8_0 quantization noise during training so the model learns
to route critical information away from quantization-sensitive parameters.
Uses Straight-Through Estimator (STE) for gradient flow through the
non-differentiable rounding step.

q8_0 format: blocks of 32 values, per-block absmax scaling to int8 [-127, 127].

Only applied to the base model's nn.Linear layers. Auxiliary modules
(UQ head, ThoughtNorm, MTP head) are separate nn.Module instances and
must NOT be passed to enable_qat().
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

Q8_0_BLOCK_SIZE = 32
Q8_0_MAX_VAL = 127


class _FakeQuantQ8_0(torch.autograd.Function):
    """Straight-through estimator for q8_0 fake quantization.

    Forward: simulate q8_0 round-trip (scale -> round -> clamp -> dequantize).
    Backward: pass gradients through unchanged (identity).
    """

    @staticmethod
    def forward(ctx, weight: torch.Tensor) -> torch.Tensor:
        return _q8_0_roundtrip(weight)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor]:
        return (grad_output,)


def _q8_0_roundtrip(weight: torch.Tensor) -> torch.Tensor:
    """Simulate q8_0 quantize -> dequantize round-trip.

    Groups weights into blocks of 32, computes per-block absmax scale,
    rounds to int8, clamps to [-127, 127], and dequantizes back to float.

    For weights whose total element count isn't divisible by 32,
    the last partial block is zero-padded, processed, then trimmed.
    """
    original_shape = weight.shape
    flat = weight.reshape(-1)
    n = flat.numel()

    # Pad to multiple of block size
    remainder = n % Q8_0_BLOCK_SIZE
    if remainder != 0:
        pad_len = Q8_0_BLOCK_SIZE - remainder
        flat = F.pad(flat, (0, pad_len))

    blocks = flat.reshape(-1, Q8_0_BLOCK_SIZE)

    # Per-block absmax scale: scale = absmax / 127
    absmax = blocks.abs().amax(dim=1, keepdim=True)
    scale = absmax / Q8_0_MAX_VAL
    # Avoid division by zero for all-zero blocks
    scale = scale.clamp(min=1e-12)

    # Quantize: round and clamp to int8 range
    quantized = (blocks / scale).round().clamp(-Q8_0_MAX_VAL, Q8_0_MAX_VAL)

    # Dequantize back to float
    dequantized = quantized * scale

    # Trim padding and reshape to original
    return dequantized.reshape(-1)[:n].reshape(original_shape)


def fake_quantize_q8_0(weight: torch.Tensor) -> torch.Tensor:
    """Apply q8_0 fake quantization with STE gradient pass-through."""
    return _FakeQuantQ8_0.apply(weight)


def enable_qat(model: nn.Module) -> None:
    """Enable QAT on all nn.Linear layers in the model.

    Replaces each Linear's forward method to apply fake q8_0 quantization
    to weights before the linear operation. Original forward is saved
    for restoration via disable_qat().

    Only pass the base model -- auxiliary modules (UQ head, ThoughtNorm,
    MTP head) should NOT be instrumented.
    """
    for module in model.modules():
        if isinstance(module, nn.Linear):
            _patch_linear(module)


def disable_qat(model: nn.Module) -> None:
    """Remove QAT from all nn.Linear layers, restoring original forward."""
    for module in model.modules():
        if isinstance(module, nn.Linear) and hasattr(module, "_qat_original_forward"):
            module.forward = module._qat_original_forward
            del module._qat_original_forward


def _patch_linear(linear: nn.Linear) -> None:
    """Patch a single Linear module to use fake-quantized weights."""
    if hasattr(linear, "_qat_original_forward"):
        return  # Already patched

    linear._qat_original_forward = linear.forward

    def qat_forward(x: torch.Tensor) -> torch.Tensor:
        fq_weight = fake_quantize_q8_0(linear.weight)
        return F.linear(x, fq_weight, linear.bias)

    linear.forward = qat_forward
