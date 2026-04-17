"""Tests for ZEB-134 Skip-to-Logit engram router.

The router maps engram output directly into logit space, bypassing
the frozen L6/L7 + final_norm pipeline. Tests cover the safe-init
contract, trainability of alpha, shape / tied-weight invariants,
numerical stability after a handful of steps, and CLI validation.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from ct87.engram import SkipToLogitEngramRouter


def _make_lm_head_weight(vocab: int, d_model: int) -> nn.Parameter:
    """Build a freestanding parameter the way HarmonyModel's tied
    lm_head weight looks from the router's perspective."""
    w = nn.Parameter(torch.randn(vocab, d_model))
    return w


def test_zero_init_W_align_produces_zero_output() -> None:
    """With W_align zeroed, the router must produce all-zero output
    for ANY input. This is the safe-init contract: attaching the
    router to a trained model should not perturb step-0 behavior
    until the optimizer opens alpha."""
    torch.manual_seed(0)
    d_model, vocab = 32, 100
    lm_head = _make_lm_head_weight(vocab, d_model)
    router = SkipToLogitEngramRouter(d_model, lm_head, alpha_init=0.5)
    x = torch.randn(3, 7, d_model)
    out = router(x)
    assert torch.all(out == 0), (
        "W_align zero-init must yield exactly zero output. Got "
        f"max |out| = {out.abs().max().item():.3e}"
    )


def test_log_alpha_is_trainable_parameter() -> None:
    """log_alpha must appear in .parameters() and require grad."""
    d_model, vocab = 16, 50
    router = SkipToLogitEngramRouter(
        d_model, _make_lm_head_weight(vocab, d_model), alpha_init=0.1,
    )
    assert isinstance(router.log_alpha, nn.Parameter)
    assert router.log_alpha.requires_grad
    # log_alpha should be present in router.parameters()
    param_ids = {id(p) for p in router.parameters()}
    assert id(router.log_alpha) in param_ids


def test_forward_shape_matches_lm_head() -> None:
    """Output last dim must match vocab; other dims preserved."""
    d_model, vocab = 24, 777
    router = SkipToLogitEngramRouter(
        d_model, _make_lm_head_weight(vocab, d_model), alpha_init=0.2,
    )
    x = torch.randn(2, 5, d_model)
    out = router(x)
    assert out.shape == (2, 5, vocab), (
        f"Expected (2, 5, {vocab}); got {tuple(out.shape)}"
    )


def test_lm_head_weight_is_not_owned_as_router_parameter() -> None:
    """The router must reference lm_head by identity, not own it.

    If the router owned lm_head as a Parameter, it would double-count
    the gradient (once via the main model, once via the router) and
    break tied-embedding invariants in HarmonyModel.
    """
    d_model, vocab = 16, 50
    lm_head = _make_lm_head_weight(vocab, d_model)
    router = SkipToLogitEngramRouter(d_model, lm_head, alpha_init=0.1)
    param_ids = {id(p) for p in router.parameters()}
    assert id(lm_head) not in param_ids, (
        "Router must not register lm_head_weight as its own parameter."
    )
    # And the reference must be identity-preserving.
    assert router._lm_head_weight_ref is lm_head


def test_lm_head_weight_not_mutated_by_forward() -> None:
    """Forward must not mutate the shared LM head weights in-place."""
    d_model, vocab = 16, 50
    lm_head = _make_lm_head_weight(vocab, d_model)
    original = lm_head.detach().clone()
    router = SkipToLogitEngramRouter(d_model, lm_head, alpha_init=0.5)
    _ = router(torch.randn(2, 3, d_model))
    # In-place mutation would show up either as a storage change OR a
    # data_ptr change; both checks are conservative.
    assert torch.equal(lm_head, original), (
        "Router must not mutate the shared LM head weight."
    )


def test_alpha_stays_bounded_over_a_handful_of_training_steps() -> None:
    """With alpha_init=0.1 and 10 optimizer steps on synthetic data,
    torch.exp(log_alpha) must remain below 2.0 — guards against the
    alpha-exploding pathology where log_alpha blows up at step 1."""
    torch.manual_seed(0)
    d_model, vocab = 16, 50
    lm_head = _make_lm_head_weight(vocab, d_model)
    router = SkipToLogitEngramRouter(d_model, lm_head, alpha_init=0.1)
    # Also need to break the zero-init on W_align so gradient flows;
    # otherwise log_alpha never moves and the test is vacuous.
    nn.init.normal_(router.W_align.weight, std=0.1)
    opt = torch.optim.SGD(router.parameters(), lr=1e-2)

    for _ in range(10):
        x = torch.randn(2, 4, d_model)
        target = torch.randint(0, vocab, (2, 4))
        logits = router(x)
        loss = nn.functional.cross_entropy(logits.reshape(-1, vocab), target.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    alpha = router.alpha.item()
    assert alpha < 2.0 and math.isfinite(alpha), (
        f"alpha drifted out of bounds after 10 SGD steps: got {alpha}"
    )


def test_alpha_init_rejects_invalid_values() -> None:
    """Ctor must reject non-finite or non-positive alpha_init."""
    d_model, vocab = 16, 50
    lm_head = _make_lm_head_weight(vocab, d_model)
    for bad in [0.0, -0.1, float("nan"), float("inf"), float("-inf")]:
        with pytest.raises(ValueError):
            SkipToLogitEngramRouter(d_model, lm_head, alpha_init=bad)


def test_hidden_dim_mismatch_raises() -> None:
    """Input with mismatched hidden_dim must fail loudly instead of
    silently producing garbage via shape broadcasting."""
    d_model, vocab = 16, 50
    router = SkipToLogitEngramRouter(
        d_model, _make_lm_head_weight(vocab, d_model), alpha_init=0.1,
    )
    with pytest.raises(ValueError):
        _ = router(torch.randn(2, 3, d_model + 1))


def test_gradient_flows_via_alpha_when_W_align_starts_zero() -> None:
    """With W_align = 0, the router's output is 0 so the loss is
    input-independent — but the GRADIENT path should still flow to
    log_alpha via the nonzero chain d(loss)/d(alpha) = 0 * engram_out.

    Actually with zero output, cross-entropy against a fixed target
    gives a uniform softmax and d(loss)/d(logits) = softmax - onehot,
    which is non-zero. Backprop then sends gradient through
    alpha * (W_align(engram_out) @ W.T). Because W_align = 0, the
    d/dalpha path is STILL zero (alpha multiplies zero). So at step
    0, only W_align receives non-zero gradient (via d/dW_align, which
    does NOT depend on W_align's current value). log_alpha gradient
    is zero until W_align has non-zero weights.

    This test documents the expected gradient structure at step 0:
      log_alpha.grad == 0    (zero-init W_align kills this path)
      W_align.weight.grad != 0 (flows via alpha * (x · W.T))
    """
    torch.manual_seed(0)
    d_model, vocab = 16, 50
    lm_head = _make_lm_head_weight(vocab, d_model)
    router = SkipToLogitEngramRouter(d_model, lm_head, alpha_init=0.1)
    x = torch.randn(2, 3, d_model)
    target = torch.randint(0, vocab, (2, 3))
    loss = nn.functional.cross_entropy(
        router(x).reshape(-1, vocab), target.reshape(-1),
    )
    loss.backward()
    assert router.log_alpha.grad is not None
    assert router.W_align.weight.grad is not None
    # W_align grad must be non-zero so the first step opens the
    # W_align path; log_alpha grad will be zero at this step
    # because alpha multiplies W_align(x) which is zero.
    assert router.W_align.weight.grad.abs().max().item() > 0, (
        "W_align gradient must flow at step 0 so the skip path opens."
    )
    assert router.log_alpha.grad.abs().item() == pytest.approx(0.0, abs=1e-8), (
        f"At step 0 with W_align=0, log_alpha grad must be zero; got "
        f"{router.log_alpha.grad.abs().item():.3e}"
    )


def test_alpha_property_matches_exp_log_alpha() -> None:
    """router.alpha should expose torch.exp(log_alpha) exactly."""
    d_model, vocab = 16, 50
    lm_head = _make_lm_head_weight(vocab, d_model)
    router = SkipToLogitEngramRouter(d_model, lm_head, alpha_init=0.1)
    # Mutate log_alpha and verify the property follows.
    with torch.no_grad():
        router.log_alpha.fill_(math.log(0.25))
    assert router.alpha.item() == pytest.approx(0.25, rel=1e-5)
