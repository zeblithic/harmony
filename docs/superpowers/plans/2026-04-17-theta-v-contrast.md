# Experiment θ-V-contrast Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement V-contrastive auxiliary loss for the η-B engram cross-attention path so V-projection's output direction becomes content-sensitive to which corpus rows retrieval returns.

**Architecture:** A new `ContrastiveGatedEngramInjection` subclass of `GatedEngramInjection` runs a second xattn forward against a per-step random row-permutation of the primary table. An auxiliary `(cos(inj_real, inj_shuf))²` loss measured at the post-`o_proj` pre-gate level is added to the LM loss with a linear-warmup λ schedule. The shuffled branch is training-only and never enters the residual stream.

**Tech Stack:** Python 3.10+, PyTorch, existing ct87 training infrastructure.

**Spec:** `docs/superpowers/specs/2026-04-17-theta-v-contrast-engram-design.md`

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `training/ct87/model.py` | Modify | Add 3 `engram_vcontrast_*` config fields + post_init validation; add `tiny_engram_xattn_capgap_vcontrast` preset; add `_contrastive_aux_losses` model attribute; clear-list-on-forward |
| `training/ct87/engram.py` | Modify | Refactor `EngramCrossAttention.forward` to expose `_attention_block(h, retrieved, topk_sims)` helper; add `ContrastiveGatedEngramInjection(GatedEngramInjection)` subclass |
| `training/ct87/train.py` | Modify | Add 4 CLI flags; instantiate contrastive variant when configured; wire aux-loss accumulation + λ schedule into the training step; add CSV columns + console print + end-of-run summary |
| `training/tests/test_v_contrast.py` | Create | Unit tests for ContrastiveGatedEngramInjection, lambda_schedule, model integration, train.py 1-step integration |
| `training/scripts/forensic_eta_b_capgap.py` | Modify (separate PR, Task 14) | Add `analyze_cross_table` + `--alt-shuffle-seed` CLI |

---

## Branching note

This plan implements two PRs in sequence:

1. **θ-V-contrast feature** (Tasks 1–13): everything except the cross-table forensic.
2. **Cross-table forensic precursor** (Task 14): branch off `main`, push first as a small standalone PR so it can be reviewed and merged ahead of the longer training-time PR.

Within this worktree (`zeblith/zeb-130-theta-vcontrast`), implement Tasks 1–13 sequentially. Task 14 instructs the implementer to create a fresh worktree off `origin/main` for the precursor PR.

---

### Task 1: Config fields for V-contrast

**Files:**
- Modify: `training/ct87/model.py:54-79` (HarmonyModelConfig dataclass body)
- Modify: `training/ct87/model.py:81-143` (`__post_init__`)
- Test: `training/tests/test_v_contrast.py` (new file)

- [ ] **Step 1: Write the failing tests**

Create `training/tests/test_v_contrast.py`:

```python
"""Tests for θ-V-contrast V-contrastive engram injection (ZEB-130)."""
from __future__ import annotations

import pytest
import torch

from ct87.engram import (
    ContrastiveGatedEngramInjection,
    EngramCrossAttention,
    GatedEngramInjection,
)
from ct87.model import HarmonyModel, HarmonyModelConfig


def _tiny_config() -> HarmonyModelConfig:
    """Minimal config for V-contrast tests (matches test_capacity_gap pattern)."""
    c = HarmonyModelConfig(
        num_layers=4, hidden_dim=64, num_query_heads=2, num_kv_heads=2,
        head_dim=32, ffn_dim=128, vocab_size=128, max_seq_len=64,
        rope_theta=10000.0, rms_norm_eps=1e-6, layers_per_block=2,
        engram_injection_layer=1, engram_dim=32, tie_embeddings=True,
    )
    return c


class TestVContrastConfig:

    def test_default_disabled(self):
        c = _tiny_config()
        assert c.engram_vcontrast_enabled is False
        assert c.engram_vcontrast_lambda == 1.0
        assert c.engram_vcontrast_warmup_steps == 200

    def test_enabled_passes_post_init(self):
        c = _tiny_config()
        c.engram_inject_layers = (1,)
        c.engram_vcontrast_enabled = True
        c.engram_vcontrast_lambda = 0.5
        c.engram_vcontrast_warmup_steps = 100
        c.__post_init__()  # should not raise

    def test_negative_lambda_rejected(self):
        c = _tiny_config()
        c.engram_inject_layers = (1,)
        c.engram_vcontrast_enabled = True
        c.engram_vcontrast_lambda = -1.0
        with pytest.raises(ValueError, match="engram_vcontrast_lambda"):
            c.__post_init__()

    def test_negative_warmup_rejected(self):
        c = _tiny_config()
        c.engram_inject_layers = (1,)
        c.engram_vcontrast_enabled = True
        c.engram_vcontrast_warmup_steps = -1
        with pytest.raises(ValueError, match="engram_vcontrast_warmup_steps"):
            c.__post_init__()

    def test_enabled_without_inject_layers_rejected(self):
        """V-contrast lives on top of the multi-layer gated injection path."""
        c = _tiny_config()
        c.engram_vcontrast_enabled = True  # but engram_inject_layers stays empty
        with pytest.raises(ValueError, match="engram_inject_layers"):
            c.__post_init__()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd training && pytest tests/test_v_contrast.py::TestVContrastConfig -v`
Expected: FAIL — `engram_vcontrast_*` attributes do not exist on `HarmonyModelConfig`.

- [ ] **Step 3: Add the three config fields**

In `training/ct87/model.py`, locate the `engram_gate_init: float = 0.0` line near the end of the `HarmonyModelConfig` field list (around line 79) and append three new fields immediately after it:

```python
    engram_gate_init: float = 0.0
    # θ-V-contrast (ZEB-130): V-contrastive auxiliary loss. Adds a second
    # xattn forward against a per-step row-permutation of the primary table
    # and penalizes cosine alignment between real-branch and shuffled-branch
    # post-o_proj outputs. Only meaningful when engram_inject_layers is set
    # (rides on top of the multi-layer gated injection path).
    engram_vcontrast_enabled: bool = False
    engram_vcontrast_lambda: float = 1.0
    engram_vcontrast_warmup_steps: int = 200
```

- [ ] **Step 4: Add post_init validation**

In `__post_init__`, locate the `engram_inject_layers` block (around line 95) and append the V-contrast validation after the existing `for layer_idx in self.engram_inject_layers:` range-check loop, **before** the `if self.ffn_dim_overrides is None` early return. Insert this block:

```python
        if self.engram_vcontrast_enabled:
            if not self.engram_inject_layers:
                raise ValueError(
                    "engram_vcontrast_enabled=True requires "
                    "engram_inject_layers to be non-empty (V-contrast lives "
                    "on top of the multi-layer gated injection path)."
                )
            if self.engram_vcontrast_lambda < 0.0:
                raise ValueError(
                    "engram_vcontrast_lambda must be >= 0.0, got "
                    f"{self.engram_vcontrast_lambda!r}"
                )
            if self.engram_vcontrast_warmup_steps < 0:
                raise ValueError(
                    "engram_vcontrast_warmup_steps must be >= 0, got "
                    f"{self.engram_vcontrast_warmup_steps!r}"
                )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd training && pytest tests/test_v_contrast.py::TestVContrastConfig -v`
Expected: PASS (5 tests).

- [ ] **Step 6: Run full capacity-gap suite to verify no regression**

Run: `cd training && pytest tests/test_capacity_gap.py -v`
Expected: PASS (all existing tests).

- [ ] **Step 7: Commit**

```bash
git add training/ct87/model.py training/tests/test_v_contrast.py
git commit -m "feat(ct87): config fields for θ-V-contrast (ZEB-130)"
```

---

### Task 2: V-contrast preset

**Files:**
- Modify: `training/ct87/model.py:306-324` (after `tiny_engram_xattn_capgap`)
- Test: `training/tests/test_v_contrast.py`

- [ ] **Step 1: Write the failing tests**

Append to `training/tests/test_v_contrast.py`:

```python
class TestVContrastPreset:

    def test_preset_extends_capgap(self):
        c = HarmonyModelConfig.tiny_engram_xattn_capgap_vcontrast()
        # capgap-inherited fields:
        assert c.engram_inject_layers == (2, 5)
        assert c.engram_gate_init == 0.0
        # V-contrast-specific fields:
        assert c.engram_vcontrast_enabled is True
        assert c.engram_vcontrast_lambda == 1.0
        assert c.engram_vcontrast_warmup_steps == 200

    def test_preset_passes_post_init(self):
        # Re-validates after __post_init__ runs (preset calls it explicitly).
        HarmonyModelConfig.tiny_engram_xattn_capgap_vcontrast()
```

- [ ] **Step 2: Run to verify failure**

Run: `cd training && pytest tests/test_v_contrast.py::TestVContrastPreset -v`
Expected: FAIL — `tiny_engram_xattn_capgap_vcontrast` not defined.

- [ ] **Step 3: Add the preset method**

In `training/ct87/model.py`, immediately after the closing `return base` of `tiny_engram_xattn_capgap` (around line 324), add:

```python
    @staticmethod
    def tiny_engram_xattn_capgap_vcontrast() -> HarmonyModelConfig:
        """θ-V-contrast: η-B capgap + V-contrastive auxiliary loss (ZEB-130).

        Extends `tiny_engram_xattn_capgap` with a per-step shuffled-table
        contrastive auxiliary loss on the post-o_proj pre-gate outputs of
        every injection layer. Used to address the (D*) DISTRIBUTIONAL
        ALIGNMENT verdict from the 2026-04-17 (W)/(A) forensic — V's
        per-token output directions are diverse within a run but align
        across runs despite different retrievals; the aux loss pressures
        V toward content-sensitivity.
        """
        base = HarmonyModelConfig.tiny_engram_xattn_capgap()
        base.engram_vcontrast_enabled = True
        base.engram_vcontrast_lambda = 1.0
        base.engram_vcontrast_warmup_steps = 200
        base.__post_init__()  # re-validate after mutation
        return base
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd training && pytest tests/test_v_contrast.py::TestVContrastPreset -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add training/ct87/model.py training/tests/test_v_contrast.py
git commit -m "feat(ct87): tiny_engram_xattn_capgap_vcontrast preset"
```

---

### Task 3: lambda_schedule helper

**Files:**
- Modify: `training/ct87/train.py` (add helper near the other small utility functions; recommend just above `freeze_backbone_for_capgap` at line 321)
- Test: `training/tests/test_v_contrast.py`

- [ ] **Step 1: Write the failing tests**

Append to `training/tests/test_v_contrast.py`:

```python
class TestLambdaSchedule:

    def test_step_zero_returns_zero(self):
        from ct87.train import lambda_schedule
        assert lambda_schedule(0, warmup=200, target=1.0) == 0.0

    def test_warmup_linear(self):
        from ct87.train import lambda_schedule
        # 100/200 = 0.5 of target
        assert lambda_schedule(100, warmup=200, target=1.0) == pytest.approx(0.5)
        # 50/200 = 0.25
        assert lambda_schedule(50, warmup=200, target=2.0) == pytest.approx(0.5)

    def test_at_warmup_boundary(self):
        from ct87.train import lambda_schedule
        # Spec: returns target AT and past the warmup boundary.
        assert lambda_schedule(200, warmup=200, target=1.0) == 1.0

    def test_past_warmup_returns_target(self):
        from ct87.train import lambda_schedule
        assert lambda_schedule(500, warmup=200, target=1.5) == 1.5
        assert lambda_schedule(10000, warmup=200, target=0.3) == 0.3

    def test_zero_warmup_returns_target_immediately(self):
        from ct87.train import lambda_schedule
        # Edge case: warmup=0 should never enter the linear ramp branch
        # (avoids division-by-zero and matches "no warmup" semantics).
        assert lambda_schedule(0, warmup=0, target=1.0) == 1.0
        assert lambda_schedule(100, warmup=0, target=1.0) == 1.0
```

- [ ] **Step 2: Run to verify failure**

Run: `cd training && pytest tests/test_v_contrast.py::TestLambdaSchedule -v`
Expected: FAIL — `lambda_schedule` not importable from `ct87.train`.

- [ ] **Step 3: Add `lambda_schedule` to train.py**

In `training/ct87/train.py`, add immediately above `def freeze_backbone_for_capgap` (around line 321):

```python
def lambda_schedule(step: int, warmup: int, target: float) -> float:
    """Linear warmup from 0 to `target` over `warmup` steps; constant `target` after.

    θ-V-contrast aux-loss schedule (ZEB-130). At step 0 we return 0.0 even
    when warmup=0 is requested; that case immediately returns `target`,
    matching the "no warmup" semantics callers expect.
    """
    if warmup <= 0:
        return target
    if step >= warmup:
        return target
    return target * step / warmup
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd training && pytest tests/test_v_contrast.py::TestLambdaSchedule -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add training/ct87/train.py training/tests/test_v_contrast.py
git commit -m "feat(ct87): lambda_schedule helper for V-contrast warmup"
```

---

### Task 4: Refactor EngramCrossAttention to expose `_attention_block`

**Files:**
- Modify: `training/ct87/engram.py:1003-1040` (`EngramCrossAttention.forward`)
- Test: `training/tests/test_engram.py` (existing) + `training/tests/test_v_contrast.py`

This is a no-op refactor that exposes a hook the contrastive subclass needs without changing any observable behavior of the existing `forward()`.

- [ ] **Step 1: Write the failing test**

Append to `training/tests/test_v_contrast.py`:

```python
class TestAttentionBlockRefactor:
    """The post-retrieval attention pipeline must be callable on caller-supplied
    (retrieved, topk_sims) so the V-contrast subclass can run it against a
    shuffled table without a second `retrieve_topk` call duplicating the
    matmul against `self.table_normalized`."""

    def test_attention_block_matches_forward(self):
        torch.manual_seed(0)
        c = _tiny_config()
        c.use_xattn_engram = True
        table = torch.randn(16, c.engram_dim)
        xattn = EngramCrossAttention(c, table, num_heads=2, k_retrieved=4)
        # Init o_proj non-zero so the residual is non-trivial (it's
        # zero-init'd by default for the legacy delta path; we override
        # here so the test catches a refactor that drops a step).
        torch.nn.init.xavier_uniform_(xattn.o_proj.weight)
        xattn.eval()

        h = torch.randn(2, 5, c.hidden_dim)
        retrieved, topk_sims = xattn.retrieve_topk(h)

        out_via_forward = xattn(h)
        out_via_block = xattn._attention_block(h, retrieved, topk_sims)
        assert torch.allclose(out_via_forward, out_via_block, atol=1e-6), (
            "_attention_block must reproduce forward() bit-for-bit when "
            "fed the same retrieved tensors."
        )
```

- [ ] **Step 2: Run to verify failure**

Run: `cd training && pytest tests/test_v_contrast.py::TestAttentionBlockRefactor -v`
Expected: FAIL — `_attention_block` is not defined.

- [ ] **Step 3: Refactor `EngramCrossAttention.forward`**

In `training/ct87/engram.py`, replace the body of `EngramCrossAttention.forward` (currently lines 1003-1040) with a thin wrapper around a new `_attention_block` helper. The replacement is:

```python
    def _attention_block(
        self,
        hidden_state: torch.Tensor,
        retrieved: torch.Tensor,
        topk_sims: torch.Tensor,
    ) -> torch.Tensor:
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

        Returns:
            [batch, seq_len, hidden_dim] residual (pre-gate).
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
        scores = scores + self.retrieval_bias_weight * topk_sims.unsqueeze(2)
        attn = F.softmax(scores, dim=-1)

        out = torch.einsum("blhk,blkhd->blhd", attn, v_tensor)

        if self.use_head_gates:
            gate_weights = torch.sigmoid(self.head_gates).view(1, 1, H, 1)
            out = out * gate_weights

        return self.o_proj(out.reshape(B, L, H * D))

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
```

Make sure to delete the old in-line body of `forward` so only one definition of `forward` exists.

- [ ] **Step 4: Run the new test plus the existing engram + capacity-gap suites**

Run: `cd training && pytest tests/test_v_contrast.py::TestAttentionBlockRefactor tests/test_engram.py tests/test_capacity_gap.py -v`
Expected: PASS (no regressions; the refactor is behavior-preserving).

- [ ] **Step 5: Commit**

```bash
git add training/ct87/engram.py training/tests/test_v_contrast.py
git commit -m "refactor(ct87): expose EngramCrossAttention._attention_block"
```

---

### Task 5: ContrastiveGatedEngramInjection module

**Files:**
- Modify: `training/ct87/engram.py` (append after `GatedEngramInjection`, line 1141)
- Test: `training/tests/test_v_contrast.py`

- [ ] **Step 1: Write the failing tests**

Append to `training/tests/test_v_contrast.py`:

```python
class TestContrastiveGatedEngramInjection:

    def _make_xattn(self) -> EngramCrossAttention:
        c = _tiny_config()
        c.use_xattn_engram = True
        table = torch.randn(16, c.engram_dim)
        return EngramCrossAttention(c, table, num_heads=2, k_retrieved=4)

    def test_residual_matches_parent_when_eval(self):
        """In eval mode, the contrastive subclass must produce the same residual
        as the parent — the shuffled branch is training-only."""
        torch.manual_seed(0)
        xattn = self._make_xattn()
        sink: list = []
        wrapper = ContrastiveGatedEngramInjection(
            xattn, alpha_init=0.5, aux_loss_sink=sink,
        )
        wrapper.eval()
        h = torch.randn(2, 5, 64)

        out_contrastive = wrapper(h)
        # Compare to the parent's forward against the same xattn module by
        # re-running just the parent path:
        gate = torch.tanh(wrapper.alpha).to(dtype=out_contrastive.dtype)
        expected = gate * xattn(h)
        assert torch.allclose(out_contrastive, expected, atol=1e-6)
        assert sink == [], "Sink must be empty in eval mode"

    def test_training_appends_one_aux_loss(self):
        torch.manual_seed(0)
        xattn = self._make_xattn()
        sink: list = []
        wrapper = ContrastiveGatedEngramInjection(
            xattn, alpha_init=0.5, aux_loss_sink=sink,
        )
        wrapper.train(True)
        h = torch.randn(2, 5, 64)
        _ = wrapper(h)
        assert len(sink) == 1
        aux = sink[0]
        assert aux.dim() == 0, "aux loss must be a scalar"
        assert 0.0 <= aux.item() <= 1.0, "(cos)^2 is bounded in [0, 1]"

    def test_training_with_no_sink_skips_aux_compute(self):
        """If aux_loss_sink is None, training mode must not allocate the
        shuffled branch — used as a guard for accidentally double-paying
        for the aux compute when V-contrast is disabled."""
        torch.manual_seed(0)
        xattn = self._make_xattn()
        wrapper = ContrastiveGatedEngramInjection(
            xattn, alpha_init=0.5, aux_loss_sink=None,
        )
        wrapper.train(True)
        h = torch.randn(2, 5, 64)
        # Should not raise and should produce a valid residual.
        out = wrapper(h)
        assert out.shape == h.shape

    def test_per_step_permutation_differs(self):
        """Two consecutive forwards in training mode must produce different
        aux-loss values almost-surely (different perms → different shuf branches).
        We assert different aux scalars rather than reach into the perm tensor
        directly so the test stays robust to internal refactors."""
        torch.manual_seed(0)
        xattn = self._make_xattn()
        sink: list = []
        wrapper = ContrastiveGatedEngramInjection(
            xattn, alpha_init=0.5, aux_loss_sink=sink,
        )
        wrapper.train(True)
        h = torch.randn(2, 5, 64)
        _ = wrapper(h)
        _ = wrapper(h)
        assert len(sink) == 2
        assert sink[0].item() != sink[1].item(), (
            "Per-step re-permutation must produce two distinct aux losses; "
            "if these match exactly, the perm is being cached or reused."
        )

    def test_gradient_flows_to_v_proj_through_aux(self):
        """The aux loss must produce non-zero gradient on v_proj — that's
        the load-bearing pressure on V toward content-sensitivity."""
        torch.manual_seed(0)
        xattn = self._make_xattn()
        sink: list = []
        wrapper = ContrastiveGatedEngramInjection(
            xattn, alpha_init=0.0, aux_loss_sink=sink,
        )
        wrapper.train(True)
        h = torch.randn(2, 5, 64)
        _ = wrapper(h)
        assert len(sink) == 1
        sink[0].backward()
        v_grad = wrapper.engram_xattn.v_proj.weight.grad
        assert v_grad is not None
        assert v_grad.abs().sum().item() > 1e-8, (
            "Aux loss must flow gradient to v_proj — that's the whole point."
        )

    def test_residual_does_not_depend_on_shuf_branch(self):
        """The shuffled branch must NOT perturb the residual. Two training-mode
        forwards on the same input (with sink reset between) must produce the
        same residual, modulo differences in the shuffle's effect on the residual
        — which must be zero. We verify by setting torch.manual_seed before each
        forward AND ensuring the residual depends only on the unshuffled path."""
        torch.manual_seed(0)
        xattn = self._make_xattn()
        sink: list = []
        wrapper = ContrastiveGatedEngramInjection(
            xattn, alpha_init=0.5, aux_loss_sink=sink,
        )
        wrapper.train(True)
        h = torch.randn(2, 5, 64)

        # Two forwards: the residual must be identical even though the
        # internal randperm produces different shuf branches.
        torch.manual_seed(123)
        out1 = wrapper(h).clone()
        torch.manual_seed(456)
        out2 = wrapper(h).clone()
        assert torch.allclose(out1, out2, atol=1e-6), (
            "Residual changed between forwards even though only the shuffled "
            "branch's RNG should differ — the shuf branch is leaking into "
            "the residual."
        )
```

- [ ] **Step 2: Run to verify failure**

Run: `cd training && pytest tests/test_v_contrast.py::TestContrastiveGatedEngramInjection -v`
Expected: FAIL — `ContrastiveGatedEngramInjection` not defined.

- [ ] **Step 3: Implement the subclass**

In `training/ct87/engram.py`, append at the end of the file (after `GatedEngramInjection`'s closing brace, around line 1141):

```python
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

    Value-only shuffle: we keep the real branch's top-k indices and substitute
    content from a permuted table at those same positions. A full key+value
    permutation would be a semantic no-op: top-k selection is permutation-
    equivariant, so the same logical rows win and retrieved_shuf ≡ retrieved_real,
    yielding zero contrastive signal.
    """

    def __init__(
        self,
        engram_xattn: EngramCrossAttention,
        alpha_init: float = 0.0,
        aux_loss_sink: list[torch.Tensor] | None = None,
    ) -> None:
        super().__init__(engram_xattn, alpha_init=alpha_init)
        # Held by reference: the training script owns the list (so it can
        # clear it between optimizer steps and stack the per-layer scalars)
        # and the wrappers append to it. A None sink disables the aux
        # branch — used so HarmonyModel can construct the contrastive
        # variant before the sink is wired up if needed.
        self._aux_sink: list[torch.Tensor] | None = aux_loss_sink

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        xattn = self.engram_xattn

        # Primary branch (residual-contributing):
        retrieved_real, topk_sims_real = xattn.retrieve_topk(hidden_state)
        inj_real = xattn._attention_block(hidden_state, retrieved_real, topk_sims_real)

        if self.training and self._aux_sink is not None:
            # Per-step random row-permutation of the primary table (value
            # shuffle). We keep the retrieval *keys* fixed (same top-k indices
            # as the real branch) but supply content from a permuted table so
            # the shuffled branch sees genuinely different rows at those
            # positions. A full key+value permutation would be a semantic no-op:
            # top-k selection is permutation-equivariant, so the same logical
            # rows would win and the injections would be identical.
            N = xattn.table.shape[0]
            perm = torch.randperm(N, device=xattn.table.device)
            table_shuf = xattn.table[perm]

            # Re-derive the top-k index tensor (not exposed by retrieve_topk)
            # using the already-computed retrieval projection. One q-proj
            # forward + einsum + topk — no _attention_block overhead.
            q = xattn.retrieval_query_proj(hidden_state)
            q_norm = F.normalize(q, dim=-1, eps=1e-8)
            sims_real = torch.einsum("ble,te->blt", q_norm, xattn.table_normalized)
            topk_sims_real2, topk_idx_real = sims_real.topk(xattn.k_retrieved, dim=-1)
            retrieved_shuf = table_shuf[topk_idx_real]
            inj_shuf = xattn._attention_block(hidden_state, retrieved_shuf, topk_sims_real2)

            # Mean-squared cosine across [B, L]. Smooth, bounded [0, 1],
            # natural attractor at cos=0. Avoids the anti-alignment
            # pathology of signed cosine minimization.
            cos = F.cosine_similarity(inj_real, inj_shuf, dim=-1)
            aux_loss = (cos ** 2).mean()
            self._aux_sink.append(aux_loss)

        gate = torch.tanh(self.alpha).to(dtype=inj_real.dtype)
        return gate * inj_real
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd training && pytest tests/test_v_contrast.py::TestContrastiveGatedEngramInjection -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add training/ct87/engram.py training/tests/test_v_contrast.py
git commit -m "feat(ct87): ContrastiveGatedEngramInjection (θ-V-contrast)"
```

---

### Task 6: HarmonyModel `_contrastive_aux_losses` side-channel

**Files:**
- Modify: `training/ct87/model.py:559-562` (HarmonyModel.__init__ side-channel block)
- Modify: `training/ct87/model.py:760-764` (forward, side-channel reset block)
- Test: `training/tests/test_v_contrast.py`

Adds the model-owned aux-loss list and the per-forward clear so the wrappers always see a fresh empty list.

- [ ] **Step 1: Write the failing tests**

Append to `training/tests/test_v_contrast.py`:

```python
class TestModelAuxLossSink:

    def _make_capgap_vcontrast_model(self):
        """Build a tiny model + ContrastiveGatedEngramInjection wrappers
        without going through train.py — keeps the test self-contained."""
        c = _tiny_config()
        c.engram_inject_layers = (1,)
        c.engram_vcontrast_enabled = True
        c.__post_init__()
        model = HarmonyModel(c)
        table = torch.randn(16, c.engram_dim)
        sink: list = []
        injections = {}
        for layer_idx in c.engram_inject_layers:
            xattn = EngramCrossAttention(c, table, num_heads=2, k_retrieved=4)
            injections[layer_idx] = ContrastiveGatedEngramInjection(
                xattn, alpha_init=0.0, aux_loss_sink=sink,
            )
        model.attach_gated_engram_injections(injections)
        model._contrastive_aux_losses = sink
        return model, sink

    def test_default_attribute_is_none(self):
        c = _tiny_config()
        model = HarmonyModel(c)
        assert model._contrastive_aux_losses is None

    def test_forward_clears_list_in_training(self):
        model, sink = self._make_capgap_vcontrast_model()
        # Pre-load the sink with a stale value to verify forward clears it
        # before the wrappers append.
        sink.append(torch.tensor(99.0))
        model.train(True)
        input_ids = torch.randint(0, model.config.vocab_size, (2, 5))
        _ = model(input_ids)
        # After forward, the sink should contain ONLY the per-layer entries
        # appended during this forward (one per injection layer), not the
        # stale 99.0 we put in.
        assert len(sink) == len(model.config.engram_inject_layers)
        for entry in sink:
            assert entry.item() != 99.0

    def test_forward_does_not_clear_in_eval(self):
        """In eval mode the wrappers don't append, but model.forward must
        also not clear — so a caller's pre-loaded list survives. (Edge case
        for resumable forensic flows that may want to inspect the sink.)"""
        model, sink = self._make_capgap_vcontrast_model()
        sink.append(torch.tensor(42.0))
        model.eval()
        input_ids = torch.randint(0, model.config.vocab_size, (2, 5))
        _ = model(input_ids)
        assert len(sink) == 1
        assert sink[0].item() == 42.0
```

- [ ] **Step 2: Run to verify failure**

Run: `cd training && pytest tests/test_v_contrast.py::TestModelAuxLossSink -v`
Expected: FAIL — `_contrastive_aux_losses` attribute does not exist.

- [ ] **Step 3: Add the attribute to `HarmonyModel.__init__`**

In `training/ct87/model.py`, locate the side-channel block in `HarmonyModel.__init__` (around line 559-562):

```python
        self._last_ann_gate: torch.Tensor | None = None
        self._last_xattn_output: torch.Tensor | None = None
        self._last_pre_injection_hidden: torch.Tensor | None = None
        self.engram_inject_mult: float = 1.0
```

Add a fourth side-channel field directly after the three `_last_*` fields, before `self.engram_inject_mult`:

```python
        self._last_ann_gate: torch.Tensor | None = None
        self._last_xattn_output: torch.Tensor | None = None
        self._last_pre_injection_hidden: torch.Tensor | None = None
        # θ-V-contrast (ZEB-130): aux-loss sink populated by
        # ContrastiveGatedEngramInjection wrappers during the forward pass.
        # Owned by the training script (which assigns the same list reference
        # into both this attribute and each wrapper's `_aux_sink`); cleared
        # at the start of every training-mode forward.
        self._contrastive_aux_losses: list[torch.Tensor] | None = None
        self.engram_inject_mult: float = 1.0
```

- [ ] **Step 4: Add the per-forward clear**

In `training/ct87/model.py`, locate the side-channel reset block in `HarmonyModel.forward` (around line 760-763):

```python
        # Reset gate side-channel at the start of every forward
        self._last_ann_gate = None
        self._last_xattn_output = None
        self._last_pre_injection_hidden = None
```

Add the V-contrast clear immediately after, **only in training mode** so the eval-time forensic inspection path (above test) is unaffected:

```python
        # Reset gate side-channel at the start of every forward
        self._last_ann_gate = None
        self._last_xattn_output = None
        self._last_pre_injection_hidden = None
        # V-contrast aux-loss sink: clear only in training mode so eval-time
        # callers (e.g. forensic) can inspect a pre-loaded list without it
        # being wiped out.
        if self._contrastive_aux_losses is not None and self.training:
            self._contrastive_aux_losses.clear()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd training && pytest tests/test_v_contrast.py::TestModelAuxLossSink -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Run capacity-gap suite to confirm no regression**

Run: `cd training && pytest tests/test_capacity_gap.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add training/ct87/model.py training/tests/test_v_contrast.py
git commit -m "feat(ct87): HarmonyModel V-contrast aux-loss sink"
```

---

### Task 7: CLI flags

**Files:**
- Modify: `training/ct87/train.py:362-410` area (parser flag definitions; add a small block adjacent to `--engram-xattn-*` flags around line 493)

- [ ] **Step 1: Add the four CLI arguments**

In `training/ct87/train.py`, locate the `--engram-xattn-num-heads` argument (around line 510) — right after that, add a new V-contrast block. The four flags are:

```python
    parser.add_argument(
        "--engram-vcontrast", action="store_true",
        help=(
            "Enable θ-V-contrast V-contrastive auxiliary loss (ZEB-130). "
            "Requires --config=tiny_engram_xattn_capgap_vcontrast (or another "
            "preset with engram_inject_layers set). Adds a per-layer aux loss "
            "penalizing cosine alignment between real-table and shuffled-table "
            "post-o_proj outputs."
        ),
    )
    parser.add_argument(
        "--engram-vcontrast-lambda", type=float, default=None,
        help=(
            "Override the config's engram_vcontrast_lambda (default 1.0). "
            "Aux loss is added as lambda * sum_layers(aux_loss_l)."
        ),
    )
    parser.add_argument(
        "--engram-vcontrast-warmup-steps", type=int, default=None,
        help=(
            "Override the config's engram_vcontrast_warmup_steps "
            "(default 200). Linear warmup from 0 to lambda."
        ),
    )
    parser.add_argument(
        "--engram-vcontrast-shuffle-seed", type=int, default=None,
        help=(
            "Optional seed for the per-step shuffle generator. Default: "
            "use the global PyTorch RNG (preferred for production runs; "
            "the seed is for reproducibility debugging only)."
        ),
    )
```

- [ ] **Step 2: Wire CLI overrides into the config**

In `training/ct87/train.py`, locate the config-construction block ending around line 776 (`config = HarmonyModelConfig.target()`). After the config is fully selected and BEFORE the validation/setup blocks downstream, add the V-contrast preset switch and CLI override block. Add a new `elif` for the new preset alongside the existing `elif args.config == "tiny_engram_xattn_capgap":`:

```python
    elif args.config == "tiny_engram_xattn_capgap":
        config = HarmonyModelConfig.tiny_engram_xattn_capgap()
    elif args.config == "tiny_engram_xattn_capgap_vcontrast":
        config = HarmonyModelConfig.tiny_engram_xattn_capgap_vcontrast()
```

Then add the V-contrast preset to the `--config` choices list (around line 365-372). The full updated `choices=` list should include `"tiny_engram_xattn_capgap_vcontrast"`:

```python
    parser.add_argument(
        "--config",
        default="tiny",
        choices=[
            "tiny", "tiny_ffn_expanded",
            "tiny_engram_ann", "tiny_engram_ann_routed",
            "tiny_engram_xattn", "tiny_engram_xattn_routed",
            "tiny_engram_xattn_consol_online",
            "tiny_engram_xattn_consol_phased",
            "tiny_engram_xattn_ctrl",
            "tiny_engram_xattn_capgap",
            "tiny_engram_xattn_capgap_vcontrast",
            "target",
        ],
        ...  # keep existing help string
    )
```

Then add CLI overrides + `--engram-vcontrast` consistency check immediately after the config has been selected and validated, but BEFORE the engram-specific table-loading block (a good location is right after the existing `seq_len = args.seq_len or ...` line, around line 777):

```python
    # θ-V-contrast (ZEB-130): apply CLI overrides on top of the preset's
    # defaults, then re-validate. --engram-vcontrast must agree with the
    # preset; mismatch is a configuration error (silently ignoring either
    # flag leads to runs that look complete but are actually misconfigured).
    if args.engram_vcontrast and not config.engram_vcontrast_enabled:
        config.engram_vcontrast_enabled = True
    elif config.engram_vcontrast_enabled and not args.engram_vcontrast:
        print(
            f"Error: --config={args.config} enables engram_vcontrast but "
            "--engram-vcontrast was not passed. Pass --engram-vcontrast "
            "explicitly to confirm intent (V-contrast doubles engram-forward "
            "compute and changes the loss surface).",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.engram_vcontrast_lambda is not None:
        config.engram_vcontrast_lambda = args.engram_vcontrast_lambda
    if args.engram_vcontrast_warmup_steps is not None:
        config.engram_vcontrast_warmup_steps = args.engram_vcontrast_warmup_steps
    if config.engram_vcontrast_enabled:
        # Re-run validation after CLI mutation, mirroring tiny_ffn_expanded /
        # tiny_engram_xattn_capgap pattern.
        config.__post_init__()
```

- [ ] **Step 3: Smoke-test with `--help`**

Run: `cd training && python -m ct87.train --help 2>&1 | grep -A2 -- '--engram-vcontrast'`
Expected: Four flag descriptions print with no argparse errors.

- [ ] **Step 4: Commit**

```bash
git add training/ct87/train.py
git commit -m "feat(ct87): CLI flags for θ-V-contrast"
```

---

### Task 8: Wire ContrastiveGatedEngramInjection into train.py setup

**Files:**
- Modify: `training/ct87/train.py:997-1042` (the `if config.engram_inject_layers:` capgap injection-construction block)

- [ ] **Step 1: Replace the injection-construction loop**

In `training/ct87/train.py`, locate the existing capgap injection-construction block (lines ~997-1042 in current state). The original block (paraphrased) is:

```python
    if config.engram_inject_layers:
        from ct87.engram import EngramCrossAttention, GatedEngramInjection
        if args.engram_xattn_table is not None:
            capgap_table = EngramCrossAttention.load_corpus_table(args.engram_xattn_table)
        elif args.synthetic:
            ...
            capgap_table = torch.randn(config.vocab_size, config.engram_dim)
        else:
            print(...); sys.exit(1)
        capgap_injections: dict[int, GatedEngramInjection] = {}
        for layer_idx in config.engram_inject_layers:
            xattn_mod = EngramCrossAttention(
                config, capgap_table, ..., k_retrieved=args.engram_xattn_k_retrieved,
                retrieval_bias_weight=args.engram_xattn_retrieval_bias_weight,
            )
            capgap_injections[layer_idx] = GatedEngramInjection(
                xattn_mod, alpha_init=config.engram_gate_init,
            )
        model.attach_gated_engram_injections(capgap_injections)
```

Update this block so that, when `config.engram_vcontrast_enabled` is True, `ContrastiveGatedEngramInjection` is constructed instead of `GatedEngramInjection`, and the shared aux-loss list is wired into both the wrappers and the model. The replacement is:

```python
    if config.engram_inject_layers:
        from ct87.engram import (
            ContrastiveGatedEngramInjection,
            EngramCrossAttention,
            GatedEngramInjection,
        )

        if args.engram_xattn_table is not None:
            capgap_table = EngramCrossAttention.load_corpus_table(args.engram_xattn_table)
        elif args.synthetic:
            print(
                "[capgap] --synthetic with no --engram-xattn-table; using random "
                f"table (vocab_size={config.vocab_size}, "
                f"engram_dim={config.engram_dim})"
            )
            capgap_table = torch.randn(config.vocab_size, config.engram_dim)
        else:
            print(
                "Error: --config=tiny_engram_xattn_capgap requires "
                "--engram-xattn-table for non-synthetic runs (a random "
                "table would invalidate the capacity-gap interpretation).",
                file=sys.stderr,
            )
            sys.exit(1)

        # θ-V-contrast: when enabled, construct ContrastiveGatedEngramInjection
        # wrappers sharing a single aux-loss list. The training step pulls aux
        # losses out of that list, sums them, and adds lambda * sum to LM loss.
        capgap_aux_sink: list[torch.Tensor] | None = (
            [] if config.engram_vcontrast_enabled else None
        )
        capgap_injections: dict[int, GatedEngramInjection] = {}
        for layer_idx in config.engram_inject_layers:
            xattn_mod = EngramCrossAttention(
                config,
                capgap_table,
                num_heads=config.num_query_heads,
                k_retrieved=args.engram_xattn_k_retrieved,
                retrieval_bias_weight=args.engram_xattn_retrieval_bias_weight,
            )
            if config.engram_vcontrast_enabled:
                capgap_injections[layer_idx] = ContrastiveGatedEngramInjection(
                    xattn_mod,
                    alpha_init=config.engram_gate_init,
                    aux_loss_sink=capgap_aux_sink,
                )
            else:
                capgap_injections[layer_idx] = GatedEngramInjection(
                    xattn_mod,
                    alpha_init=config.engram_gate_init,
                )
        model.attach_gated_engram_injections(capgap_injections)
        if config.engram_vcontrast_enabled:
            # Both the model and the wrappers reference the same list — the
            # model clears it at the start of each training-mode forward and
            # the wrappers append per-layer scalars.
            model._contrastive_aux_losses = capgap_aux_sink

        model.engram_injections.to(device)
        print(
            f"[capgap] Attached "
            f"{'ContrastiveGatedEngramInjection' if config.engram_vcontrast_enabled else 'GatedEngramInjection'}"
            f" at layers {list(config.engram_inject_layers)} with alpha_init="
            f"{config.engram_gate_init}"
            + (
                f"; vcontrast lambda={config.engram_vcontrast_lambda} "
                f"warmup={config.engram_vcontrast_warmup_steps}"
                if config.engram_vcontrast_enabled else ""
            )
        )
```

(Preserve the existing `model.engram_injections.to(device)` call and the print statement that follow — replace them as shown above so they still execute exactly once, with the V-contrast suffix in the print.)

- [ ] **Step 2: Smoke-test that the setup runs**

Run: `cd training && python -m ct87.train --config tiny_engram_xattn_capgap_vcontrast --engram-vcontrast --synthetic --steps 0 --data /tmp/dummy --output-dir /tmp/vctest 2>&1 | tail -10`

Expected: prints `[capgap] Attached ContrastiveGatedEngramInjection at layers [2, 5] with alpha_init=0.0; vcontrast lambda=1.0 warmup=200` and exits cleanly without entering the training loop.

(If `--data /tmp/dummy` causes an error before reaching this print, also pass `--no-validate-data` if it exists, or substitute a real data path. Goal of this smoke step is just to confirm the setup block runs.)

- [ ] **Step 3: Commit**

```bash
git add training/ct87/train.py
git commit -m "feat(ct87): wire ContrastiveGatedEngramInjection in train setup"
```

---

### Task 9: Aux-loss accumulation + λ schedule in training step

**Files:**
- Modify: `training/ct87/train.py:1520-1530` area (per-step accumulator init)
- Modify: `training/ct87/train.py:1755-1758` area (just before `accum_loss += loss.item()` at the end of the inner accumulation loop)

- [ ] **Step 1: Initialize V-contrast accumulators per outer step**

In `training/ct87/train.py`, locate the per-step accumulator init block around line 1520-1530 (the `accum_loss = 0.0`, `accum_uq_loss = 0.0`, etc.). Append two V-contrast accumulators after `accum_mse_loss = 0.0`:

```python
            accum_loss = 0.0
            accum_uq_loss = 0.0
            accum_mtp_loss = 0.0
            accum_cl_loss = 0.0
            accum_ann_ent_loss = 0.0
            accum_ann_gate_mean = 0.0
            accum_mse_loss = 0.0
            # θ-V-contrast (ZEB-130): aux-loss accumulators. Sum is captured
            # as a scalar (no grad), per-layer values are retained for CSV.
            accum_vcontrast_aux = 0.0
            accum_vcontrast_per_layer: dict[str, float] = {}
```

- [ ] **Step 2: Add V-contrast loss-composition step**

In the inner accumulation loop, locate the consolidation MSE loss block (around lines 1744-1756, ending with `accum_mse_loss += mse_loss.item()`). Append the V-contrast block immediately AFTER consolidation MSE and BEFORE `accum_loss += loss.item()` (which is around line 1758):

```python
                    # ZEB-128: Consolidation MSE loss
                    if (
                        consol_decoder is not None
                        ...  # existing block ends with accum_mse_loss += mse_loss.item()
                    ):
                        ...
                        accum_mse_loss += mse_loss.item()

                    # θ-V-contrast (ZEB-130): aggregate per-layer V-contrastive
                    # aux losses appended by ContrastiveGatedEngramInjection
                    # forwards. The sink is owned by the model and cleared
                    # at the start of every training-mode forward.
                    if (
                        config.engram_vcontrast_enabled
                        and model._contrastive_aux_losses
                    ):
                        aux_per_layer = model._contrastive_aux_losses
                        # Stack to a single tensor so the .sum() is a single
                        # CUDA op rather than a Python-loop over scalars.
                        aux_total = torch.stack(aux_per_layer).sum()
                        lam = lambda_schedule(
                            step,
                            config.engram_vcontrast_warmup_steps,
                            config.engram_vcontrast_lambda,
                        )
                        loss = loss + lam * aux_total
                        # Detach for logging — these are consumed by CSV /
                        # console emit and must not retain the graph.
                        accum_vcontrast_aux += aux_total.detach().item()
                        # Per-layer accumulator: keys are str(layer_idx),
                        # matching the ModuleDict key convention.
                        for layer_key, layer_loss in zip(
                            (str(i) for i in config.engram_inject_layers),
                            aux_per_layer,
                        ):
                            accum_vcontrast_per_layer[layer_key] = (
                                accum_vcontrast_per_layer.get(layer_key, 0.0)
                                + layer_loss.detach().item()
                            )

                accum_loss += loss.item()
                (loss / args.grad_accum_steps).backward()
```

- [ ] **Step 3: Verify train.py loads without syntax errors**

Run: `cd training && python -c "import ct87.train"`
Expected: no output, no import error.

- [ ] **Step 4: Commit**

```bash
git add training/ct87/train.py
git commit -m "feat(ct87): V-contrast aux-loss accumulation + lambda schedule"
```

---

### Task 10: CSV columns + console + end-of-run summary

**Files:**
- Modify: `training/ct87/train.py:1416-1423` (`expected_header`)
- Modify: `training/ct87/train.py:1810-1870` (console print block)
- Modify: `training/ct87/train.py:1909-1968` (CSV row block)
- Modify: `training/ct87/train.py:1973-2000` (end-of-run summary block, after the final `save_checkpoint`)

- [ ] **Step 1: Append V-contrast columns to `expected_header`**

In `training/ct87/train.py`, locate `expected_header` (around line 1416-1423). Append four V-contrast columns at the end of the list:

```python
        expected_header = [
            "step", "loss", "uq_loss", "mtp_loss", "cl_loss",
            "ann_ent_loss", "ann_gate_mean", "ann_lambda_ent",
            "val_loss", "lr", "grad_norm", "num_thoughts", "dt_ms",
            "hg_0", "hg_1", "hg_2", "hg_3", "hg_4", "hg_5", "hg_6", "hg_7",
            "hg_std", "hg_min", "hg_max",
            "mse_loss", "consol_phase", "inject_mult",
            # θ-V-contrast (ZEB-130):
            "vcontrast_aux_loss", "vcontrast_aux_l2", "vcontrast_aux_l5",
            "vcontrast_lambda",
        ]
```

The `n_hg_slots = len(expected_header) - 16` calculation around line 1926 needs an updated arithmetic constant. The relationship is:

- Before this change: `len(expected_header) = 26`, `n_hg_slots = 26 - 16 = 10` (gives 8 head slots + std/min/max — wait, `13 base + 8 hg + 3 stat + 3 consol = 27`; the original constant is intentional offset accounting). **The current constant `len(expected_header) - 16` was tuned for `len(expected_header) = 26` giving `n_hg_slots = 10` (= 8 head slots + std/min/max — only 3 trailing stat slots, so 8 + 3 = 11; the actual hg_cols block is 11 elements).** Re-derive carefully:

The hg_cols block in current code is `[hg_0..hg_7, hg_std, hg_min, hg_max]` = 11 entries. The math: `13 (base columns before hg) + 11 (hg columns) + 3 (mse, consol_phase, inject_mult) = 27`. But the constant is `- 16`, which would give `27 - 16 = 11`. Reconcile: it's `13 base + 3 trailing → 16 non-hg columns`, so `n_hg_slots = total - 16 = 11`. That matches. Good.

After adding 4 V-contrast columns, `len(expected_header) = 31`. We need `n_hg_slots = 11` to remain, so the constant becomes `len(expected_header) - 20` (i.e., `31 - 20 = 11`).

Update the calculation around line 1926 from:

```python
                n_hg_slots = len(expected_header) - 16  # 13 base columns before hg_*, 3 consol columns after
```

to:

```python
                n_hg_slots = len(expected_header) - 20  # 13 base + 3 consol + 4 vcontrast = 20 non-hg columns
```

- [ ] **Step 2: Wire V-contrast values into the CSV row write**

In the CSV row block (around line 1949-1967), the row currently ends with `inject_mult_str` as the last column. Append four V-contrast values to the end of the `csv_writer.writerow([...])` call:

```python
                # θ-V-contrast (ZEB-130): per-layer aux losses + total + lambda.
                vcontrast_aux_str = ""
                vcontrast_l2_str = ""
                vcontrast_l5_str = ""
                vcontrast_lambda_str = ""
                if config.engram_vcontrast_enabled:
                    raw_aux = accum_vcontrast_aux / args.grad_accum_steps
                    vcontrast_aux_str = f"{raw_aux:.6f}"
                    vcontrast_l2_str = (
                        f"{accum_vcontrast_per_layer.get('2', 0.0) / args.grad_accum_steps:.6f}"
                        if "2" in accum_vcontrast_per_layer else ""
                    )
                    vcontrast_l5_str = (
                        f"{accum_vcontrast_per_layer.get('5', 0.0) / args.grad_accum_steps:.6f}"
                        if "5" in accum_vcontrast_per_layer else ""
                    )
                    current_lam = lambda_schedule(
                        step,
                        config.engram_vcontrast_warmup_steps,
                        config.engram_vcontrast_lambda,
                    )
                    vcontrast_lambda_str = f"{current_lam:.6f}"
                csv_writer.writerow([
                    step,
                    f"{raw_loss:.6f}",
                    uq_loss_str,
                    mtp_loss_str,
                    cl_loss_str,
                    ann_ent_str,
                    ann_gate_str,
                    ann_lambda_str,
                    val_loss_str,
                    f"{current_lr:.8f}",
                    f"{grad_norm:.6f}" if grad_norm is not None else "",
                    num_thoughts,
                    f"{dt_ms:.1f}",
                    *hg_cols,
                    mse_loss_str,
                    consol_phase_str,
                    inject_mult_str,
                    vcontrast_aux_str,
                    vcontrast_l2_str,
                    vcontrast_l5_str,
                    vcontrast_lambda_str,
                ])
                csv_file.flush()
```

(Replace the existing `csv_writer.writerow([...])` call in full so the row matches `expected_header` length.)

- [ ] **Step 3: Add V-contrast to the per-step console line**

In the console-print block (around line 1810-1870, just before the `print(...)` call that emits the step line), add a V-contrast snippet built the same way as `capgap_str`. Insert immediately after the `capgap_str` block:

```python
                # θ-V-contrast (ZEB-130): aux loss + lambda + per-layer.
                vcontrast_str = ""
                if config.engram_vcontrast_enabled:
                    raw_aux = accum_vcontrast_aux / args.grad_accum_steps
                    raw_lam = lambda_schedule(
                        step,
                        config.engram_vcontrast_warmup_steps,
                        config.engram_vcontrast_lambda,
                    )
                    parts = [f"aux={raw_aux:.4f}"]
                    for lk in (str(i) for i in config.engram_inject_layers):
                        if lk in accum_vcontrast_per_layer:
                            v = accum_vcontrast_per_layer[lk] / args.grad_accum_steps
                            parts.append(f"aux_L{lk}={v:.4f}")
                    parts.append(f"λ={raw_lam:.3f}")
                    vcontrast_str = "  " + "  ".join(parts)
```

Then add `{vcontrast_str}` to the `print(...)` format string right before `{capgap_str}`:

```python
                print(
                    f"step={step:5d}  loss={raw_loss:.4f}  lr={current_lr:.6f}"
                    f"{ct_str}{uq_str}{mtp_str}{cl_str}{ann_str}{hg_str}{consol_str}{vcontrast_str}{capgap_str}"
                )
```

- [ ] **Step 4: Add end-of-run V-contrast summary**

In `training/ct87/train.py`, locate the end-of-run block (around line 1973-2000, where the final checkpoint is saved and `Final val_loss=...` is printed). After `print(f"Training complete. Final checkpoint at step {args.steps}")`, add:

```python
            print(f"Training complete. Final checkpoint at step {args.steps}")

            # θ-V-contrast (ZEB-130): final summary block.
            if config.engram_vcontrast_enabled:
                with torch.no_grad():
                    print(f"[vcontrast] Final step={args.steps}")
                    if model._contrastive_aux_losses:
                        # The list survives one final forward — but reliably the
                        # most recent training-step accumulator is what we want.
                        print(
                            f"    last_aux_total = {accum_vcontrast_aux / max(args.grad_accum_steps, 1):.6f}"
                        )
                    if final_val_loss is not None:
                        print(f"    val_loss (with inj)    = {final_val_loss:.6f}")
                    if model.engram_injections is not None:
                        for layer_key, injection in model.engram_injections.items():
                            alpha_val = injection.alpha.detach().item()
                            print(
                                f"    alpha_L{layer_key} = {alpha_val:+.6f}, "
                                f"tanh(alpha_L{layer_key}) = "
                                f"{math.tanh(alpha_val):+.6f}"
                            )
```

(`math` is already imported at the top of `train.py` — line 11. `final_val_loss` is defined a few lines above in the same block.)

- [ ] **Step 5: Verify train.py still imports cleanly**

Run: `cd training && python -c "import ct87.train"`
Expected: no output, no import error.

- [ ] **Step 6: Commit**

```bash
git add training/ct87/train.py
git commit -m "feat(ct87): V-contrast CSV columns + console + end-of-run summary"
```

---

### Task 11: Integration test — 1-step training with V-contrast

**Files:**
- Modify: `training/tests/test_v_contrast.py`

This is the load-bearing integration test: build a tiny model + ContrastiveGatedEngramInjection, run one optimizer step, verify aux loss appeared in the sink and the parameters updated.

- [ ] **Step 1: Add the integration test**

Append to `training/tests/test_v_contrast.py`:

```python
class TestOneStepIntegration:
    """One forward + backward + optimizer step exercises the full V-contrast wiring
    end-to-end. Catches integration bugs that the unit tests miss (e.g. sink not
    cleared between steps, lambda not applied, gradient not flowing)."""

    def test_one_step_with_vcontrast(self):
        torch.manual_seed(0)
        c = _tiny_config()
        c.engram_inject_layers = (1,)
        c.engram_vcontrast_enabled = True
        c.engram_vcontrast_lambda = 1.0
        c.engram_vcontrast_warmup_steps = 1  # use full lambda from step 0
        c.__post_init__()

        model = HarmonyModel(c)
        table = torch.randn(16, c.engram_dim)
        sink: list = []
        injections = {}
        for layer_idx in c.engram_inject_layers:
            xattn = EngramCrossAttention(c, table, num_heads=2, k_retrieved=4)
            injections[layer_idx] = ContrastiveGatedEngramInjection(
                xattn, alpha_init=0.0, aux_loss_sink=sink,
            )
        model.attach_gated_engram_injections(injections)
        model._contrastive_aux_losses = sink

        # Force the optimizer to see only the engram_injections params, mirroring
        # the --freeze-backbone path so this test is sensitive to V-contrast
        # gradient flow rather than to backbone gradient noise.
        engram_params = list(model.engram_injections.parameters())
        opt = torch.optim.SGD(engram_params, lr=1e-2)

        model.train(True)
        input_ids = torch.randint(0, c.vocab_size, (2, 5))
        targets = torch.randint(0, c.vocab_size, (2, 5))

        # Snapshot v_proj before the step.
        v_before = (
            model.engram_injections["1"]
            .engram_xattn.v_proj.weight.detach().clone()
        )

        logits = model(input_ids)
        lm_loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, c.vocab_size), targets.reshape(-1),
        )
        # At alpha_init=0, the residual is identically zero so LM loss has
        # NO gradient path to engram params via the residual. The aux loss
        # is therefore the *only* source of engram-param gradient — perfect
        # for verifying V-contrast wiring in isolation.
        assert len(sink) == 1, "Aux sink must hold exactly one entry per inject layer"
        aux = sink[0]
        # Lambda is 1.0 with warmup=1, so step=0 schedule returns 0.0 at step 0;
        # but step 1 returns full lambda. Exercise that path explicitly.
        from ct87.train import lambda_schedule
        lam_step1 = lambda_schedule(1, 1, 1.0)
        assert lam_step1 == 1.0
        total_loss = lm_loss + lam_step1 * aux

        opt.zero_grad()
        total_loss.backward()
        opt.step()

        v_after = (
            model.engram_injections["1"]
            .engram_xattn.v_proj.weight.detach().clone()
        )
        delta = (v_after - v_before).abs().sum().item()
        assert delta > 1e-8, (
            "v_proj weight unchanged after a V-contrast aux-loss step — the "
            "aux gradient is not reaching V."
        )
```

- [ ] **Step 2: Run the integration test**

Run: `cd training && pytest tests/test_v_contrast.py::TestOneStepIntegration -v`
Expected: PASS.

- [ ] **Step 3: Run the full V-contrast test file**

Run: `cd training && pytest tests/test_v_contrast.py -v`
Expected: PASS (all classes from Tasks 1, 2, 3, 4, 5, 6, 11 — roughly 23 tests total).

- [ ] **Step 4: Run the broader regression suite**

Run: `cd training && pytest tests/test_capacity_gap.py tests/test_engram.py tests/test_model.py -v`
Expected: PASS (no regressions in existing engram/model/capacity-gap behavior).

- [ ] **Step 5: Commit**

```bash
git add training/tests/test_v_contrast.py
git commit -m "test(ct87): one-step integration test for θ-V-contrast"
```

---

### Task 12: Integration smoke test — `train.py` actually trains a step

**Files:**
- Modify: `training/tests/test_v_contrast.py` (or split into a new file `test_v_contrast_train.py` if the existing file feels overgrown)

End-to-end test that actually invokes `train.py`'s setup + a few steps using `--synthetic` so it has no external dependencies.

- [ ] **Step 1: Look up how `--synthetic` is wired**

Run: `cd training && grep -n 'synthetic' ct87/train.py | head -20`

Note the data path the existing `--synthetic` flag takes (you'll need this in the test invocation). If `--synthetic` is not a recognized flag, this task can be skipped and replaced by exercising train.py via `subprocess.run` with a real tiny dataset OR by deleting Task 12 in favor of Task 11's coverage (which already proves the wiring end-to-end at the model level).

- [ ] **Step 2: Add the subprocess smoke test (only if `--synthetic` is supported)**

Append to `training/tests/test_v_contrast.py` (or a new file `test_v_contrast_train.py`):

```python
import subprocess
import sys
from pathlib import Path


class TestTrainPyVContrastSmoke:

    def test_train_steps_with_vcontrast(self, tmp_path):
        """Run train.py for 2 steps with --synthetic + --engram-vcontrast and
        verify the CSV log has the new vcontrast_* columns populated."""
        log_file = tmp_path / "train.csv"
        output_dir = tmp_path / "output"
        result = subprocess.run(
            [
                sys.executable, "-m", "ct87.train",
                "--config", "tiny_engram_xattn_capgap_vcontrast",
                "--engram-vcontrast",
                "--synthetic",
                "--steps", "2",
                "--save-every", "0",
                "--log-file", str(log_file),
                "--output-dir", str(output_dir),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=Path(__file__).parent.parent,  # training/
        )
        assert result.returncode == 0, (
            f"train.py exited {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )
        # Confirm V-contrast banner printed during setup
        assert "ContrastiveGatedEngramInjection" in result.stdout
        # Confirm CSV header includes the V-contrast columns
        lines = log_file.read_text().splitlines()
        header = lines[0].split(",")
        assert "vcontrast_aux_loss" in header
        assert "vcontrast_lambda" in header
```

- [ ] **Step 3: Run the smoke test**

Run: `cd training && pytest tests/test_v_contrast.py::TestTrainPyVContrastSmoke -v`
Expected: PASS. If FAIL because `--synthetic` is not a recognized flag, omit this task and proceed (Task 11 covers the integration at a more local level).

- [ ] **Step 4: Commit (only if Task 12 added a passing test)**

```bash
git add training/tests/test_v_contrast.py
git commit -m "test(ct87): subprocess smoke test for train.py V-contrast"
```

---

### Task 13: Documentation (CLAUDE.md / experiment runbook)

**Files:**
- Modify: any user-facing experiment runbook in the repo if one exists for ZEB-130; if not, skip.

- [ ] **Step 1: Check if a runbook exists**

Run: `cd /Users/zeblith/work/zeblithic/harmony/.worktrees/zeb-130-theta-vcontrast && grep -rln "ZEB-130\|capgap\|tiny_engram_xattn_capgap" docs/ 2>/dev/null | head -10`

If a runbook is found that currently describes how to run η-B but does not mention V-contrast, append a "θ-V-contrast" subsection mirroring the spec's "Run 1 / Run 2" command examples (spec lines 180-204).

If no runbook exists, **skip this task**. The spec itself documents the run protocol; no new doc is required by the design.

- [ ] **Step 2: Commit (only if a runbook was updated)**

```bash
git add docs/...
git commit -m "docs: add θ-V-contrast runbook section"
```

---

### Task 14: Cross-table forensic probe (separate worktree, separate PR)

> **HISTORICAL (superseded 2026-04-17):** Task 14 as originally written used `alt_table = primary_table[perm]` (a row permutation) and a `--alt-shuffle-seed` CLI flag. The row permutation is tautological on cosine top-k retrieval — it always produces `|cos| = 1.0`, not a content-sensitivity signal. The corrected design uses a random-gaussian alt table with matched per-dim statistics, sampled from each model's own training table, via a `--alt-table-seed` flag. The step-by-step instructions below are preserved for historical context; do NOT follow them verbatim. See `training/scripts/forensic_eta_b_capgap.py` and its `analyze_cross_table` / `_sample_matched_gaussian_alt` functions for the authoritative implementation.

**Files:**
- Create new worktree from `origin/main`
- Modify: `training/scripts/forensic_eta_b_capgap.py`
- Test: `training/tests/test_forensic_cross_table.py` (new file) — only if existing forensic has a test file pattern; otherwise hand-verify

This task ships the cross-table within-run forensic probe as its own PR ahead of the V-contrast training-time work. It's completely independent of Tasks 1-13.

- [ ] **Step 1: Set up a fresh worktree off `origin/main`**

(Per CLAUDE.md hard rule "Pull before work", the cross-table forensic must be based on the latest `origin/main`, NOT on the V-contrast worktree's branch.)

```bash
cd /Users/zeblith/work/zeblithic/harmony
git fetch origin
git worktree add .worktrees/zeb-130-cross-table-forensic -b zeblith/zeb-130-cross-table-forensic origin/main
cd .worktrees/zeb-130-cross-table-forensic
```

- [ ] **Step 2: Add `--alt-shuffle-seed` CLI argument**

In `training/scripts/forensic_eta_b_capgap.py`, locate the `parse_args` function (around line 88-130). Add a new argument:

```python
    p.add_argument(
        "--alt-shuffle-seed", type=int, default=42,
        help=(
            "Seed for the held-out alt shuffle used by the cross-table "
            "within-run probe. Default 42 (explicitly different from any "
            "training-time shuffle seed). The probe forwards a single "
            "trained model against its training-primary table AND a fresh "
            "torch.randperm(N) of that table seeded by this argument; "
            "comparing matched-position injection outputs is the direct "
            "test of V content-sensitivity."
        ),
    )
```

- [ ] **Step 3: Implement `analyze_cross_table` function**

In `training/scripts/forensic_eta_b_capgap.py`, append after the existing `analyze_injection` function (around line 430):

```python
@torch.no_grad()
def analyze_cross_table(
    model: HarmonyModel,
    primary_table: torch.Tensor,
    alt_table: torch.Tensor,
    val_batches: list[torch.Tensor],
    layers: list[int],
) -> dict[int, dict[str, float]]:
    """Cross-table within-run probe: same model, same tokens, different table.

    For each injection layer, runs the model twice on each batch — once with
    the primary table installed on the layer's xattn, once with the alt
    table installed — and computes cosine alignment between matched-position
    injection outputs. A content-sensitive V should produce different
    outputs when the table changes (cross-table cos near 0). A V that has
    found the trivial-orthogonality shortcut produces near-zero cross-table
    cos as a side effect of randomness; the secondary "random-baseline"
    probe (cos between two random tokens' V outputs in the same forward)
    distinguishes the two cases.

    Args:
        model: Trained HarmonyModel with `engram_injections` attached.
        primary_table: The table the model was trained against.
        alt_table: A held-out random row-permutation of `primary_table`.
        val_batches: Validation token batches (each [batch, seq_len]).
        layers: Injection-layer indices (typically [2, 5]).

    Returns:
        Per-layer dict with keys:
            'cross_table_cos_signed', 'cross_table_cos_abs',
            'within_run_random_pair_cos_abs' (the trivial-orthogonality floor).
    """
    if primary_table.shape != alt_table.shape:
        raise ValueError(
            f"primary_table {primary_table.shape} vs alt_table "
            f"{alt_table.shape} shape mismatch"
        )
    device = next(model.parameters()).device
    primary_table = primary_table.to(device)
    alt_table = alt_table.to(device)
    primary_table_normalized = F.normalize(primary_table, dim=-1, eps=1e-8)
    alt_table_normalized = F.normalize(alt_table, dim=-1, eps=1e-8)

    # Hook to capture the input hidden state at each injection layer.
    probes = {layer: InjectionPreHook() for layer in layers}
    handles = []
    for layer_idx in layers:
        h = model.engram_injections[str(layer_idx)].register_forward_pre_hook(
            probes[layer_idx],
        )
        handles.append(h)

    per_layer: dict[int, list[dict[str, float]]] = {i: [] for i in layers}

    try:
        for batch in val_batches:
            for probe in probes.values():
                probe.reset()

            input_ids = batch.to(device)
            _ = model(input_ids)

            for layer_idx in layers:
                hidden_state = probes[layer_idx].captured
                if hidden_state is None:
                    raise RuntimeError(
                        f"Pre-hook at layer {layer_idx} did not fire — check "
                        f"engram_injections are attached and in the forward path."
                    )
                wrapper = model.engram_injections[str(layer_idx)]
                xattn = wrapper.engram_xattn

                # Primary-table injection (using the in-place buffers as
                # the trained model would see them).
                primary_inj = _injection_with_table(
                    xattn, hidden_state, primary_table, primary_table_normalized,
                )
                alt_inj = _injection_with_table(
                    xattn, hidden_state, alt_table, alt_table_normalized,
                )

                cos = F.cosine_similarity(primary_inj, alt_inj, dim=-1)
                signed = cos.mean().item()
                abs_mean = cos.abs().mean().item()

                # Random-baseline within-run pair: shuffle the primary
                # injection along the (B*L) flattened axis, then compute
                # |cos| against the unshuffled — this is what V-output
                # alignment would look like under "random in high-dim".
                flat = primary_inj.reshape(-1, primary_inj.shape[-1])
                perm = torch.randperm(flat.shape[0], device=flat.device)
                random_pair_cos = F.cosine_similarity(
                    flat, flat[perm], dim=-1,
                ).abs().mean().item()

                per_layer[layer_idx].append({
                    "cross_table_cos_signed": signed,
                    "cross_table_cos_abs": abs_mean,
                    "within_run_random_pair_cos_abs": random_pair_cos,
                })
    finally:
        for h in handles:
            h.remove()

    # Average per-layer scalars across batches.
    out: dict[int, dict[str, float]] = {}
    for layer_idx in layers:
        rows = per_layer[layer_idx]
        n = max(len(rows), 1)
        out[layer_idx] = {
            "cross_table_cos_signed": sum(r["cross_table_cos_signed"] for r in rows) / n,
            "cross_table_cos_abs": sum(r["cross_table_cos_abs"] for r in rows) / n,
            "within_run_random_pair_cos_abs": sum(
                r["within_run_random_pair_cos_abs"] for r in rows
            ) / n,
        }
    return out


def _injection_with_table(
    xattn: "EngramCrossAttention",
    hidden_state: torch.Tensor,
    table: torch.Tensor,
    table_normalized: torch.Tensor,
) -> torch.Tensor:
    """Compute injection output using a caller-supplied table (no in-place buffer mutation).

    Mirrors `EngramCrossAttention.forward` but uses the supplied table rather
    than `xattn.table` / `xattn.table_normalized`. We do NOT swap the buffers
    in-place because that would race with concurrent forwards if any exist
    and would also leave the model in a wrong state if an exception fires.
    """
    q = xattn.retrieval_query_proj(hidden_state)
    q_norm = F.normalize(q, dim=-1, eps=1e-8)
    sims = torch.einsum("ble,te->blt", q_norm, table_normalized)
    topk_sims, topk_idx = sims.topk(xattn.k_retrieved, dim=-1)
    retrieved = table[topk_idx]
    # If `_attention_block` is exposed (post-θ-V-contrast PR), prefer it.
    # Otherwise reproduce inline. This branch is here so the cross-table PR
    # can land BEFORE the θ-V-contrast PR without depending on it.
    if hasattr(xattn, "_attention_block"):
        return xattn._attention_block(hidden_state, retrieved, topk_sims)
    # Inline reproduction for pre-refactor xattn (matches the original
    # forward at commit b2c3a96):
    B, L, _ = hidden_state.shape
    H, D, k = xattn.num_heads, xattn.head_dim, xattn.k_retrieved
    retrieved_normed = xattn.retrieval_norm(retrieved)
    q_attn = xattn.q_norm(xattn.q_proj(hidden_state).view(B, L, H, D))
    k_attn = xattn.k_norm(xattn.k_proj(retrieved_normed).view(B, L, k, H, D))
    v_attn = xattn.v_proj(retrieved_normed).view(B, L, k, H, D)
    scores = torch.einsum("blhd,blkhd->blhk", q_attn, k_attn) / (D ** 0.5)
    scores = scores + xattn.retrieval_bias_weight * topk_sims.unsqueeze(2)
    attn = F.softmax(scores, dim=-1)
    out = torch.einsum("blhk,blkhd->blhd", attn, v_attn)
    if xattn.use_head_gates:
        gate_heads = torch.sigmoid(xattn.head_gates).view(1, 1, H, 1)
        out = out * gate_heads
    return xattn.o_proj(out.reshape(B, L, H * D))
```

- [ ] **Step 4: Wire `analyze_cross_table` into `run_forensic`**

In the same file, locate `run_forensic` (around line 435). After the existing per-layer cross-run printout but before `print_verdict_criteria(args.k_retrieved)`, add a cross-table block. Before printing, generate the alt table from the real-checkpoint's primary table:

```python
    print("\n--- (X) cross-table within-run probe ---")
    print(f"Alt-shuffle seed: {args.alt_shuffle_seed}")

    # Re-load the real-oracle table to use as the primary; derive the alt
    # table by held-out random permutation seeded by --alt-shuffle-seed.
    primary_table = EngramCrossAttention.load_corpus_table(str(args.real_table))
    g = torch.Generator(device="cpu")
    g.manual_seed(args.alt_shuffle_seed)
    perm = torch.randperm(primary_table.shape[0], generator=g)
    alt_table = primary_table[perm]

    # Re-collect val batches for the cross-table probe (val_loader is
    # one-shot in some implementations; safest to reuse the same dataloader
    # constructor with the same seed).
    val_loader_xtable = make_hf_dataloader(
        args.val_data, args.seq_len, args.batch_size, args.seed,
    )
    val_batches: list[torch.Tensor] = []
    for _ in range(args.num_batches):
        val_batches.append(next(val_loader_xtable)[:, :-1])

    cross_table_real = analyze_cross_table(
        real_model, primary_table, alt_table, val_batches, layers,
    )
    print("\n[real-oracle model]")
    for layer_idx in layers:
        stats = cross_table_real[layer_idx]
        print(f"  L{layer_idx}:")
        print(f"    signed (primary vs alt)           {stats['cross_table_cos_signed']:+.4f}")
        print(f"    |abs|                             {stats['cross_table_cos_abs']:.4f}")
        print(f"    random-pair |cos| (orth. floor)   {stats['within_run_random_pair_cos_abs']:.4f}")

    cross_table_shuf = analyze_cross_table(
        shuf_model, primary_table, alt_table, val_batches, layers,
    )
    print("\n[shuffled-oracle model]")
    for layer_idx in layers:
        stats = cross_table_shuf[layer_idx]
        print(f"  L{layer_idx}:")
        print(f"    signed (primary vs alt)           {stats['cross_table_cos_signed']:+.4f}")
        print(f"    |abs|                             {stats['cross_table_cos_abs']:.4f}")
        print(f"    random-pair |cos| (orth. floor)   {stats['within_run_random_pair_cos_abs']:.4f}")
```

(Insert this block AFTER `print_cross_run(cross_run_per_layer, layers)` and BEFORE `print_verdict_criteria(args.k_retrieved)`.)

- [ ] **Step 5: Hand-test on a synthetic checkpoint**

The forensic script has no existing tests, so verify by hand:

```bash
cd /Users/zeblith/work/zeblithic/harmony/.worktrees/zeb-130-cross-table-forensic
cd training
python -c "
import torch
from ct87.engram import EngramCrossAttention
from scripts.forensic_eta_b_capgap import _injection_with_table

# Build a small xattn, compute injection two different ways, verify shapes match.
from ct87.model import HarmonyModelConfig
c = HarmonyModelConfig(
    num_layers=4, hidden_dim=64, num_query_heads=2, num_kv_heads=2,
    head_dim=32, ffn_dim=128, vocab_size=128, max_seq_len=64,
    rope_theta=10000.0, rms_norm_eps=1e-6, layers_per_block=2,
    engram_injection_layer=1, engram_dim=32, tie_embeddings=True,
)
table = torch.randn(16, c.engram_dim)
xattn = EngramCrossAttention(c, table, num_heads=2, k_retrieved=4)
torch.nn.init.xavier_uniform_(xattn.o_proj.weight)
xattn.eval()
h = torch.randn(2, 5, 64)
import torch.nn.functional as F
table_norm = F.normalize(table, dim=-1, eps=1e-8)
out = _injection_with_table(xattn, h, table, table_norm)
print('shape:', out.shape)
assert out.shape == (2, 5, 64)
# And: must match xattn(h)
import torch
assert torch.allclose(out, xattn(h), atol=1e-6), 'inline reproduction mismatch'
print('OK')
"
```
Expected output: `shape: torch.Size([2, 5, 64])` then `OK`.

- [ ] **Step 6: Commit + push + open PR**

```bash
cd /Users/zeblith/work/zeblithic/harmony/.worktrees/zeb-130-cross-table-forensic
git add training/scripts/forensic_eta_b_capgap.py
git commit -m "feat(training): cross-table within-run forensic probe (ZEB-130)

Adds analyze_cross_table + --alt-shuffle-seed to the η-B forensic. The
cross-table within-run probe is the direct test of V content-sensitivity
needed to interpret θ-V-contrast results: same model weights, same tokens,
different table → must give different injection outputs if V is genuinely
content-sensitive. Includes a 'random-pair |cos|' baseline that gives the
trivial-orthogonality floor so we can distinguish 'V is content-sensitive'
from 'V emits random high-dim outputs'."
git push -u origin zeblith/zeb-130-cross-table-forensic
gh pr create --title "Cross-table within-run forensic probe (ZEB-130)" \
  --body "$(cat <<'EOF'
## Summary
- Adds `analyze_cross_table` to `forensic_eta_b_capgap.py`: forwards a single trained model against its training-primary table AND a held-out random shuffle of that table, computes cos between matched-position injection outputs.
- Adds `--alt-shuffle-seed` CLI flag (default 42) for the held-out alt table.
- Adds a "random-pair |cos|" baseline within the same forward to give the trivial-orthogonality floor.

## Why
The (D*) DISTRIBUTIONAL ALIGNMENT verdict from PR #249's (W)/(A) probes localized engram content-invariance to V's response-distribution shape. The next experiment (θ-V-contrast, separate PR) trains V with an aux loss that pressures content-sensitivity. To know whether the aux loss worked, we need a probe that directly answers "if I keep V's weights but change the table, does V's output change?" — that's this probe.

This probe ships first as a separate small PR so it lands ahead of the V-contrast training PR and can be reviewed independently of the larger training-time changes.

## Test plan
- [x] `_injection_with_table` matches `EngramCrossAttention.forward` bit-for-bit on a synthetic xattn (hand-tested in a `python -c` snippet).
- [ ] Smoke-tested by KRILE against the existing η-B real + shuffled checkpoints once merged.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 7: After cross-table PR merges, return to the V-contrast worktree**

```bash
cd /Users/zeblith/work/zeblithic/harmony/.worktrees/zeb-130-theta-vcontrast
git fetch origin
# (no rebase needed — Tasks 1-13 don't touch forensic_eta_b_capgap.py)
```

---

## Self-Review

Done as part of writing this plan; brief notes:

**Spec coverage check:**
- [x] Forward pass design (spec lines 36-56) → Tasks 4 + 5
- [x] Mean squared cosine, post-o_proj pre-gate, per-step shuffle → Task 5 + integration test in Task 11
- [x] λ schedule with warmup → Task 3 + Task 9
- [x] Total loss composition → Task 9
- [x] 3 config fields + preset → Tasks 1 + 2
- [x] CLI flags (4) → Task 7
- [x] CSV columns (4 vcontrast columns) + console + end-of-run → Task 10
- [x] ContrastiveGatedEngramInjection subclass → Task 5
- [x] Model integration (`_contrastive_aux_losses` sink + clear in forward) → Task 6
- [x] Training-loop integration (aux accum + λ schedule applied) → Task 9
- [x] Cross-table within-run probe (separate PR) → Task 14
- [x] Unit tests (forward populates sink, eval skips, residual unchanged, per-step perm differs, lambda_schedule correctness) → Tasks 1, 3, 5, 6
- [x] Integration test (1-step training records aux loss) → Tasks 11 + 12
- [x] Forward compatibility (η-B checkpoint loads into vcontrast config) → covered indirectly by the no-regression runs in Tasks 1, 4, 6 plus Task 11's structural test

**Type / signature consistency:**
- `lambda_schedule(step, warmup, target)` signature is consistent across Tasks 3, 9, 10, 11.
- `aux_loss_sink: list[torch.Tensor] | None` is consistent across Tasks 5, 6, 8, 11.
- `_contrastive_aux_losses` attribute name is consistent across Tasks 6, 8, 9, 11, plus the existing spec.
- `ContrastiveGatedEngramInjection.__init__(engram_xattn, alpha_init, aux_loss_sink)` signature is consistent across Tasks 5, 8, 11.
- CSV column names `vcontrast_aux_loss`, `vcontrast_aux_l2`, `vcontrast_aux_l5`, `vcontrast_lambda` are consistent across Task 10 and the spec lines 290-300.

**No-placeholder check:** No `TODO` / `TBD` / "fill in details" / "similar to" / "add appropriate" / "handle edge cases" patterns remain.

**Scope check:** This is a single feature with a clean interface boundary. Task 14 is intentionally split off as a separate PR per the spec (line 93). No need to decompose further.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-17-theta-v-contrast.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
