# ι-Q-diversity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an MoE-style load-balancing auxiliary loss on the retrieval-row marginal distribution of the engram cross-attention, composable with the existing V-contrast aux, to test whether fixing Q-collapse alone (ι₁) or in combination with V-contrast (ι₂) unblocks content-routing at 40M.

**Architecture:** Single unified `GatedEngramInjection` class with two independent `vcontrast_sink` and `qdiv_sink` kwargs (replaces PR #250's `ContrastiveGatedEngramInjection` subclass; removes subclass proliferation). MoE loss `L = N · Σ fᵢ · Pᵢ` computed per injection-layer per-batch, where `f` is hard top-k selection frequency (detached) and `P` is soft attention-weighted mass (differentiable through the softmax). Reuses V-contrast's aux-sink pattern, λ warmup, conditional CSV columns, and symmetric flag/preset rejection — this plan is deliberately parallel to PR #250's structure.

**Tech Stack:** PyTorch 2.2+, Python 3.10+ (local dev on 3.9 is expected to fail ~6 tests due to `zip(strict=True)`; CI uses 3.10+).

**Spec:** `docs/superpowers/specs/2026-04-17-iota-q-diversity-design.md` (approved, committed in `b9c8709`).

---

## File structure

| File | Action | Responsibility |
|---|---|---|
| `training/ct87/engram.py` | MODIFY | Module-level `compute_qdiv_aux`; `_attention_block(return_attn=True)`; unified `GatedEngramInjection` with composable sinks; remove `ContrastiveGatedEngramInjection` |
| `training/ct87/model.py` | MODIFY | `engram_qdiv_*` config fields + validation; `_qdiv_aux_losses` list on `HarmonyModel`; injection-layer wiring; two new presets |
| `training/ct87/train.py` | MODIFY | CLI flags; drain + λ warmup + total-loss add; `qdiv_*` CSV columns; console + end-of-run summary |
| `training/tests/test_iota_q_diversity.py` | CREATE | 17 tests: unit + module + API + multi-layer + subprocess smokes |
| `training/tests/test_v_contrast.py` | MODIFY (minor) | Import-path updates after `ContrastiveGatedEngramInjection` removal |

---

## Tasks

### Task 1: Config fields (`engram_qdiv_*`) + validation

**Files:**
- Modify: `training/ct87/model.py` (`HarmonyModelConfig` dataclass + `__post_init__`)
- Test: `training/tests/test_iota_q_diversity.py` (new file)

- [ ] **Step 1: Write failing tests for config validation**

Create `training/tests/test_iota_q_diversity.py`:

```python
"""Tests for ι-Q-diversity (ZEB-130): MoE load-balancing aux loss on
retrieval-row marginal distribution. Spec:
docs/superpowers/specs/2026-04-17-iota-q-diversity-design.md
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from ct87.model import HarmonyModelConfig


class TestConfigValidation:
    def test_qdiv_defaults(self):
        c = HarmonyModelConfig.tiny_engram_xattn_capgap()
        assert c.engram_qdiv_enabled is False
        assert c.engram_qdiv_lambda == 0.01
        assert c.engram_qdiv_warmup_steps == 200

    def test_qdiv_requires_xattn(self):
        c = HarmonyModelConfig.tiny_engram_xattn_capgap()
        c.engram_xattn_enabled = False
        c.engram_qdiv_enabled = True
        with pytest.raises(ValueError, match="engram_xattn_enabled"):
            c.__post_init__()

    def test_qdiv_requires_inject_layers(self):
        c = HarmonyModelConfig.tiny_engram_xattn_capgap()
        c.engram_inject_layers = ()
        c.engram_qdiv_enabled = True
        with pytest.raises(ValueError, match="engram_inject_layers"):
            c.__post_init__()

    def test_qdiv_lambda_must_be_non_negative(self):
        c = HarmonyModelConfig.tiny_engram_xattn_capgap()
        c.engram_qdiv_enabled = True
        c.engram_qdiv_lambda = -0.1
        with pytest.raises(ValueError, match="engram_qdiv_lambda"):
            c.__post_init__()

    def test_qdiv_warmup_must_be_non_negative(self):
        c = HarmonyModelConfig.tiny_engram_xattn_capgap()
        c.engram_qdiv_enabled = True
        c.engram_qdiv_warmup_steps = -1
        with pytest.raises(ValueError, match="engram_qdiv_warmup_steps"):
            c.__post_init__()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd training && python3 -m pytest tests/test_iota_q_diversity.py::TestConfigValidation -v`
Expected: FAIL — `AttributeError` or missing fields.

- [ ] **Step 3: Add config fields + validation to `HarmonyModelConfig`**

Locate `HarmonyModelConfig` in `training/ct87/model.py`. Near the existing V-contrast fields, add:

```python
    engram_qdiv_enabled: bool = False
    engram_qdiv_lambda: float = 0.01
    engram_qdiv_warmup_steps: int = 200
```

In `__post_init__`, after the existing V-contrast validation block, add:

```python
        if self.engram_qdiv_enabled and not self.engram_xattn_enabled:
            raise ValueError(
                "engram_qdiv_enabled requires engram_xattn_enabled=True; "
                "Q-div operates on retrieval softmax which only exists in "
                "the cross-attention engram path."
            )
        if self.engram_qdiv_enabled and not self.engram_inject_layers:
            raise ValueError(
                "engram_qdiv_enabled requires at least one injection layer; "
                "configure engram_inject_layers before enabling Q-div."
            )
        if self.engram_qdiv_lambda < 0:
            raise ValueError(
                f"engram_qdiv_lambda must be >= 0, got {self.engram_qdiv_lambda}"
            )
        if self.engram_qdiv_warmup_steps < 0:
            raise ValueError(
                f"engram_qdiv_warmup_steps must be >= 0, got "
                f"{self.engram_qdiv_warmup_steps}"
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd training && python3 -m pytest tests/test_iota_q_diversity.py::TestConfigValidation -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add training/ct87/model.py training/tests/test_iota_q_diversity.py
git commit -m "feat(ct87): add engram_qdiv_* config fields with validation"
```

---

### Task 2: `compute_qdiv_aux` helper with unit tests

**Files:**
- Modify: `training/ct87/engram.py` (add module-level helper)
- Test: `training/tests/test_iota_q_diversity.py` (append to existing file)

- [ ] **Step 1: Write failing tests for the helper**

Append to `training/tests/test_iota_q_diversity.py`:

```python
class TestComputeQdivAux:
    """Unit tests for the MoE load-balancing loss helper.

    Formula: L = N * sum_i f[i] * P[i], where
      f[i] = fraction of (B*L*k) hard top-k selections on row i (detached)
      P[i] = sum over (B,L,H,j) of attn[b,l,h,j] when topk_idx[b,l,j]==i,
             divided by (B*L*H)
    """

    def test_uniform_selection_gives_unit_loss(self):
        # Construct uniform selection: N=100 rows, B=1 L=100 H=4 k=4.
        # Each row selected exactly 4 times (once per l-position across k),
        # each head's attention uniform 1/k over the 4 selected rows.
        from ct87.engram import compute_qdiv_aux
        N, B, L, H, k = 100, 1, 100, 4, 4
        # Each l-position selects rows l, l+1, l+2, l+3 (mod N) — so each row
        # gets selected k times total across l.
        idx = torch.arange(L).unsqueeze(1) + torch.arange(k).unsqueeze(0)
        idx = (idx % N).unsqueeze(0).expand(B, L, k).contiguous()
        attn = torch.full((B, L, H, k), 1.0 / k)
        loss = compute_qdiv_aux(idx, attn, N)
        # f = 1/N everywhere selected, P = 1/N everywhere selected,
        # sum over rows = N * (1/N) * (1/N) = 1/N, times N = 1.0
        assert loss.item() == pytest.approx(1.0, abs=1e-5)

    def test_full_concentration_gives_n_loss(self):
        from ct87.engram import compute_qdiv_aux
        N, B, L, H, k = 100, 2, 8, 4, 4
        # All top-k selections land on row 0.
        idx = torch.zeros(B, L, k, dtype=torch.int64)
        # All attention mass on slot 0 (which maps to row 0 via idx).
        attn = torch.zeros(B, L, H, k)
        attn[..., 0] = 1.0
        loss = compute_qdiv_aux(idx, attn, N)
        # f[0]=1, P[0]=1, others 0 -> sum = 1, times N = N
        assert loss.item() == pytest.approx(float(N), abs=1e-4)

    def test_partial_concentration_intermediate(self):
        from ct87.engram import compute_qdiv_aux
        N, B, L, H, k = 100, 1, 100, 4, 4
        # Uniform over S=10 rows (rows 0..9).
        S = 10
        idx = (torch.arange(L * k) % S).reshape(B, L, k)
        attn = torch.full((B, L, H, k), 1.0 / k)
        loss = compute_qdiv_aux(idx, attn, N)
        # f[i] = 1/S for i<S, 0 else; P[i] = 1/S similarly
        # sum over i = S * (1/S) * (1/S) = 1/S; times N = N/S
        assert loss.item() == pytest.approx(float(N) / S, rel=0.01)

    def test_gradient_flows_to_attn_only(self):
        from ct87.engram import compute_qdiv_aux
        N, B, L, H, k = 50, 1, 10, 2, 3
        idx = torch.randint(0, N, (B, L, k), dtype=torch.int64)
        attn = torch.rand(B, L, H, k, requires_grad=True)
        attn_softmax = F.softmax(attn, dim=-1)
        loss = compute_qdiv_aux(idx, attn_softmax, N)
        loss.backward()
        assert attn.grad is not None
        assert attn.grad.abs().sum() > 0, "attn should receive gradient"
        # idx is int64 so autograd can't track it; just confirm helper didn't
        # convert f into something grad-requiring by accident.

    def test_deterministic_same_input_same_loss(self):
        from ct87.engram import compute_qdiv_aux
        N, B, L, H, k = 100, 2, 16, 4, 4
        torch.manual_seed(0)
        idx = torch.randint(0, N, (B, L, k), dtype=torch.int64)
        attn = F.softmax(torch.randn(B, L, H, k), dim=-1)
        a = compute_qdiv_aux(idx, attn, N)
        b = compute_qdiv_aux(idx, attn, N)
        assert torch.equal(a, b)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd training && python3 -m pytest tests/test_iota_q_diversity.py::TestComputeQdivAux -v`
Expected: FAIL — `ImportError: cannot import name 'compute_qdiv_aux'`.

- [ ] **Step 3: Implement the helper**

In `training/ct87/engram.py`, add a module-level function near the top (after the imports, before the first class):

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd training && python3 -m pytest tests/test_iota_q_diversity.py::TestComputeQdivAux -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add training/ct87/engram.py training/tests/test_iota_q_diversity.py
git commit -m "feat(ct87): add compute_qdiv_aux MoE load-balancing helper"
```

---

### Task 3: `_attention_block(return_attn=True)` kwarg

**Files:**
- Modify: `training/ct87/engram.py` (`EngramCrossAttention._attention_block`)
- Test: `training/tests/test_iota_q_diversity.py` (append)

- [ ] **Step 1: Write failing tests for the return_attn API**

Append to `training/tests/test_iota_q_diversity.py`:

```python
class TestAttentionBlockReturnAttn:
    """API extension for _attention_block: optional return of [B,L,H,k]
    softmax attention weights so Q-div can see them without recomputing.
    Symmetric to retrieve_topk(return_indices=True) from PR #250."""

    def _build_xattn(self, seed=0):
        from ct87.engram import EngramCrossAttention
        torch.manual_seed(seed)
        N, E, H, D, k = 100, 32, 4, 16, 4
        table = torch.randn(N, E)
        return EngramCrossAttention(
            hidden_dim=H * D,
            engram_dim=E,
            num_heads=H,
            head_dim=D,
            k_retrieved=k,
            corpus_table=table,
        )

    def test_return_attn_shape(self):
        xattn = self._build_xattn()
        B, L = 2, 7
        hidden = torch.randn(B, L, xattn.hidden_dim)
        retrieved, topk_sims = xattn.retrieve_topk(hidden)
        out, attn = xattn._attention_block(
            hidden, retrieved, topk_sims, return_attn=True,
        )
        assert out.shape == (B, L, xattn.hidden_dim)
        assert attn.shape == (B, L, xattn.num_heads, xattn.k_retrieved)
        assert torch.allclose(attn.sum(dim=-1), torch.ones_like(attn.sum(dim=-1)), atol=1e-5)

    def test_return_attn_false_matches_default(self):
        xattn = self._build_xattn(seed=1)
        B, L = 2, 5
        hidden = torch.randn(B, L, xattn.hidden_dim)
        retrieved, topk_sims = xattn.retrieve_topk(hidden)
        out_default = xattn._attention_block(hidden, retrieved, topk_sims)
        out_with_attn, _ = xattn._attention_block(
            hidden, retrieved, topk_sims, return_attn=True,
        )
        assert torch.equal(out_default, out_with_attn)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd training && python3 -m pytest tests/test_iota_q_diversity.py::TestAttentionBlockReturnAttn -v`
Expected: FAIL — `TypeError: _attention_block() got an unexpected keyword argument 'return_attn'`.

- [ ] **Step 3: Add `return_attn` kwarg to `_attention_block`**

In `training/ct87/engram.py`, locate `EngramCrossAttention._attention_block` (currently at approximately lines 1011-1058). Modify its signature and return logic:

```python
    def _attention_block(
        self,
        hidden_state: torch.Tensor,
        retrieved: torch.Tensor,
        topk_sims: torch.Tensor,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run the post-retrieval attention pipeline on caller-supplied retrievals.

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd training && python3 -m pytest tests/test_iota_q_diversity.py::TestAttentionBlockReturnAttn -v`
Expected: 2 passed.

Also run existing V-contrast tests to verify no regression:

Run: `cd training && python3 -m pytest tests/test_v_contrast.py -v -k "not subprocess" 2>&1 | tail -20`
Expected: Same pass/fail count as before this task (any failures pre-existed from Python 3.9 vs 3.10+ mismatch).

- [ ] **Step 5: Commit**

```bash
git add training/ct87/engram.py training/tests/test_iota_q_diversity.py
git commit -m "feat(engram): add return_attn kwarg to _attention_block"
```

---

### Task 4: Consolidate `GatedEngramInjection` with composable sinks

**Files:**
- Modify: `training/ct87/engram.py` (`GatedEngramInjection` class)
- Test: `training/tests/test_iota_q_diversity.py` (append)

This is the biggest task. Absorbs `ContrastiveGatedEngramInjection`'s V-contrast logic back into `GatedEngramInjection` as a conditional branch gated on `vcontrast_sink is not None`, adds a parallel Q-div branch gated on `qdiv_sink is not None`. Task 5 then removes `ContrastiveGatedEngramInjection` entirely.

- [ ] **Step 1: Write failing tests for the four sink configurations**

Append to `training/tests/test_iota_q_diversity.py`:

```python
class TestGatedEngramInjectionSinkMatrix:
    """Four-cell behavior matrix for the unified GatedEngramInjection.

    No sinks   -> baseline η-B behavior (regression guard for PR #247).
    vcontrast  -> PR #250 V-contrast aux; no Q-div entries.
    qdiv       -> Q-div aux appended; no shuffled-value second forward.
    both       -> both aux losses fire independently.
    """

    def _build(self, seed=0):
        from ct87.engram import EngramCrossAttention
        torch.manual_seed(seed)
        N, E, H, D, k = 100, 32, 4, 16, 4
        table = torch.randn(N, E)
        xattn = EngramCrossAttention(
            hidden_dim=H * D,
            engram_dim=E,
            num_heads=H,
            head_dim=D,
            k_retrieved=k,
            corpus_table=table,
        )
        return xattn

    def test_no_sinks_matches_baseline(self):
        from ct87.engram import GatedEngramInjection
        xattn = self._build()
        wrapper = GatedEngramInjection(xattn, alpha_init=0.1)
        wrapper.train()
        hidden = torch.randn(2, 5, xattn.hidden_dim)
        out = wrapper(hidden)
        assert out.shape == hidden.shape
        # No sinks provided => nothing appended anywhere.
        assert wrapper._vcontrast_sink is None
        assert wrapper._qdiv_sink is None

    def test_only_qdiv_sink_appends_one_per_forward(self):
        from ct87.engram import GatedEngramInjection
        xattn = self._build(seed=1)
        sink: list[torch.Tensor] = []
        wrapper = GatedEngramInjection(xattn, alpha_init=0.1, qdiv_sink=sink)
        wrapper.train()
        hidden = torch.randn(2, 5, xattn.hidden_dim)
        _ = wrapper(hidden)
        assert len(sink) == 1
        assert sink[0].ndim == 0
        assert sink[0].item() >= 1.0  # MoE loss floor

    def test_only_vcontrast_sink_appends_one_per_forward(self):
        from ct87.engram import GatedEngramInjection
        xattn = self._build(seed=2)
        sink: list[torch.Tensor] = []
        wrapper = GatedEngramInjection(xattn, alpha_init=0.1, vcontrast_sink=sink)
        wrapper.train()
        hidden = torch.randn(2, 5, xattn.hidden_dim)
        _ = wrapper(hidden)
        assert len(sink) == 1
        assert sink[0].ndim == 0
        assert 0.0 <= sink[0].item() <= 1.0  # cos^2.mean bounded [0, 1]

    def test_both_sinks_fire_independently(self):
        from ct87.engram import GatedEngramInjection
        xattn = self._build(seed=3)
        vsink: list[torch.Tensor] = []
        qsink: list[torch.Tensor] = []
        wrapper = GatedEngramInjection(
            xattn, alpha_init=0.1,
            vcontrast_sink=vsink, qdiv_sink=qsink,
        )
        wrapper.train()
        hidden = torch.randn(2, 5, xattn.hidden_dim)
        _ = wrapper(hidden)
        assert len(vsink) == 1
        assert len(qsink) == 1

    def test_eval_mode_skips_aux(self):
        from ct87.engram import GatedEngramInjection
        xattn = self._build(seed=4)
        vsink: list[torch.Tensor] = []
        qsink: list[torch.Tensor] = []
        wrapper = GatedEngramInjection(
            xattn, alpha_init=0.1,
            vcontrast_sink=vsink, qdiv_sink=qsink,
        )
        wrapper.eval()
        hidden = torch.randn(2, 5, xattn.hidden_dim)
        _ = wrapper(hidden)
        assert len(vsink) == 0
        assert len(qsink) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd training && python3 -m pytest tests/test_iota_q_diversity.py::TestGatedEngramInjectionSinkMatrix -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'qdiv_sink'`.

- [ ] **Step 3: Refactor `GatedEngramInjection` to accept composable sinks**

In `training/ct87/engram.py`, locate the current `GatedEngramInjection` class (approximately lines 1112-1172). Replace its `__init__` and `forward` with the unified version:

```python
class GatedEngramInjection(nn.Module):
    """Gated engram cross-attention injection with optional training-only
    auxiliary losses.

    Two independent aux-loss hooks can be attached via caller-supplied
    sink lists (each of which the training loop drains once per optimizer
    step and scales by its own lambda x warmup schedule):

    - vcontrast_sink: V-contrastive aux (PR #250, ZEB-130 theta). On every
      training forward, runs a second xattn with a per-step random row-
      permuted value branch; appends (cos(inj_real, inj_shuf) ** 2).mean().
    - qdiv_sink: Q-div aux (ZEB-130 iota). Captures the softmax attention
      weights and top-k indices from the main forward and computes the
      MoE load-balancing loss N * sum_i f[i] * P[i]; appends the scalar.

    When both sinks are provided, both run independently; their losses
    are summed into the training objective with separate lambdas. Passing
    None (or any aux sink but calling model.eval()) disables the
    corresponding aux path entirely (no extra compute beyond the
    baseline forward).

    shuffle_generator is an optional dedicated RNG for V-contrast's
    per-step row permutation (reproducibility debugging only); used only
    when vcontrast_sink is not None.
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
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        # Held by reference: the training script owns each list (so it
        # can clear them between optimizer steps and stack the per-layer
        # scalars) and the wrappers append to them.
        self._vcontrast_sink: list[torch.Tensor] | None = vcontrast_sink
        self._qdiv_sink: list[torch.Tensor] | None = qdiv_sink
        self._shuffle_generator: torch.Generator | None = shuffle_generator
        # initialize o_proj weight the same way the prior version did
        # (kept here as-is to preserve numerical parity).
        nn.init.xavier_uniform_(self.engram_xattn.o_proj.weight)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        xattn = self.engram_xattn
        need_vcontrast = self.training and self._vcontrast_sink is not None
        need_qdiv = self.training and self._qdiv_sink is not None
        need_idx = need_vcontrast or need_qdiv
        need_attn = need_qdiv

        # One retrieval covers both aux paths when both are enabled.
        if need_idx:
            retrieved_real, topk_sims_real, topk_idx_real = xattn.retrieve_topk(
                hidden_state, return_indices=True,
            )
        else:
            retrieved_real, topk_sims_real = xattn.retrieve_topk(hidden_state)

        # Main injection forward — conditionally capture attn weights.
        if need_attn:
            inj_real, attn_weights = xattn._attention_block(
                hidden_state, retrieved_real, topk_sims_real, return_attn=True,
            )
            self._qdiv_sink.append(
                compute_qdiv_aux(
                    topk_idx_real, attn_weights, xattn.table.shape[0],
                )
            )
        else:
            inj_real = xattn._attention_block(
                hidden_state, retrieved_real, topk_sims_real,
            )

        # V-contrast aux — shuffled-value branch.
        if need_vcontrast:
            # Per-step random row-permutation of the primary table (value
            # shuffle). Apply permutation to the gather indices — same
            # result as materializing a full shuffled table, O(B*L*k)
            # rather than O(N*D) bandwidth per forward.
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

        return hidden_state + torch.tanh(self.alpha) * inj_real
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd training && python3 -m pytest tests/test_iota_q_diversity.py::TestGatedEngramInjectionSinkMatrix -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add training/ct87/engram.py training/tests/test_iota_q_diversity.py
git commit -m "feat(engram): unify GatedEngramInjection with composable aux sinks"
```

---

### Task 5: Remove `ContrastiveGatedEngramInjection`; update V-contrast tests

**Files:**
- Modify: `training/ct87/engram.py` (delete class)
- Modify: `training/ct87/model.py` (injection-layer wiring)
- Modify: `training/tests/test_v_contrast.py` (import-path update)

- [ ] **Step 1: Locate all references to `ContrastiveGatedEngramInjection`**

Run: `cd training && grep -rn "ContrastiveGatedEngramInjection" --include="*.py"`

Expected output (in the pre-refactor state):
```
ct87/engram.py:<line>:class ContrastiveGatedEngramInjection(GatedEngramInjection):
ct87/engram.py:<line>:    "ContrastiveGatedEngramInjection",  # in __all__ if present
ct87/model.py:<line>:from .engram import ... ContrastiveGatedEngramInjection ...
ct87/model.py:<line>:   return ContrastiveGatedEngramInjection(...)   # callsite
tests/test_v_contrast.py:<multiple lines>:from ct87.engram import ContrastiveGatedEngramInjection
```

- [ ] **Step 2: Delete `ContrastiveGatedEngramInjection` class from engram.py**

In `training/ct87/engram.py`, locate the `class ContrastiveGatedEngramInjection(GatedEngramInjection):` definition (approximately lines 1174-1260) and delete the entire class body. If there's an `__all__` that includes it, remove that entry too.

- [ ] **Step 3: Update `model.py` injection-layer wiring**

In `training/ct87/model.py`, find the location where injection wrappers are constructed per layer (search for `ContrastiveGatedEngramInjection` or the conditional that picks the contrastive vs plain wrapper). Replace the conditional with a single unified call:

```python
# In HarmonyModel.__init__ or wherever engram_injections is populated:

self._contrastive_aux_losses: list[torch.Tensor] = []
self._qdiv_aux_losses: list[torch.Tensor] = []

vcontrast_sink = (
    self._contrastive_aux_losses
    if config.engram_vcontrast_enabled else None
)
qdiv_sink = (
    self._qdiv_aux_losses
    if config.engram_qdiv_enabled else None
)

for layer_idx in config.engram_inject_layers:
    # ... construct engram_xattn as before ...
    self.engram_injections[str(layer_idx)] = GatedEngramInjection(
        engram_xattn=engram_xattn,
        alpha_init=config.engram_alpha_init,
        vcontrast_sink=vcontrast_sink,
        qdiv_sink=qdiv_sink,
        shuffle_generator=vcontrast_shuffle_gen,  # existing variable
    )
```

Update the import at the top of `model.py`: remove `ContrastiveGatedEngramInjection` from the import list; keep `GatedEngramInjection`.

- [ ] **Step 4: Update `test_v_contrast.py` imports**

In `training/tests/test_v_contrast.py`, find every instance of:

```python
from ct87.engram import ContrastiveGatedEngramInjection
```

Replace with:

```python
from ct87.engram import GatedEngramInjection
```

Find every instance where `ContrastiveGatedEngramInjection(...)` is constructed in test setup, and change to `GatedEngramInjection(..., vcontrast_sink=...)`. For example:

```python
# Before:
wrapper = ContrastiveGatedEngramInjection(
    xattn, alpha_init=0.1, aux_loss_sink=sink, shuffle_generator=gen,
)

# After:
wrapper = GatedEngramInjection(
    xattn, alpha_init=0.1, vcontrast_sink=sink, shuffle_generator=gen,
)
```

Note: the keyword `aux_loss_sink` in the old class is renamed to `vcontrast_sink` in the unified class.

- [ ] **Step 5: Run V-contrast tests to verify no regression**

Run: `cd training && python3 -m pytest tests/test_v_contrast.py -v -k "not subprocess" 2>&1 | tail -30`

Expected: Same pass/fail count as on main before the refactor. Any pre-existing local failures (Python 3.9 vs 3.10+ `zip(strict=True)`) remain, but no NEW failures attributable to the refactor.

- [ ] **Step 6: Run ι tests so far**

Run: `cd training && python3 -m pytest tests/test_iota_q_diversity.py -v`
Expected: all Task 1-4 tests still pass.

- [ ] **Step 7: Verify no remaining references to the old class**

Run: `cd training && grep -rn "ContrastiveGatedEngramInjection" --include="*.py"`
Expected: no output.

- [ ] **Step 8: Commit**

```bash
git add training/ct87/engram.py training/ct87/model.py training/tests/test_v_contrast.py
git commit -m "refactor(engram): remove ContrastiveGatedEngramInjection (merged into GatedEngramInjection)"
```

---

### Task 6: Add `_qdiv_aux_losses` list to `HarmonyModel`

**Files:**
- Modify: `training/ct87/model.py` (`HarmonyModel.__init__`)
- Test: `training/tests/test_iota_q_diversity.py` (append)

- [ ] **Step 1: Write failing test for the list attribute**

Append to `training/tests/test_iota_q_diversity.py`:

```python
class TestHarmonyModelQdivSink:
    def test_model_exposes_qdiv_aux_losses_list(self):
        from ct87.model import HarmonyModel, HarmonyModelConfig
        c = HarmonyModelConfig.tiny_engram_xattn_capgap()
        c.engram_qdiv_enabled = True
        c.__post_init__()
        model = HarmonyModel(c)
        assert hasattr(model, "_qdiv_aux_losses")
        assert isinstance(model._qdiv_aux_losses, list)
        assert len(model._qdiv_aux_losses) == 0

    def test_qdiv_sink_receives_one_per_layer_per_forward(self):
        from ct87.model import HarmonyModel, HarmonyModelConfig
        c = HarmonyModelConfig.tiny_engram_xattn_capgap()
        c.engram_qdiv_enabled = True
        c.__post_init__()
        model = HarmonyModel(c)
        model.train()
        B, L = 1, 8
        input_ids = torch.randint(0, c.vocab_size, (B, L))
        model._qdiv_aux_losses.clear()
        _ = model(input_ids)
        assert len(model._qdiv_aux_losses) == len(c.engram_inject_layers)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd training && python3 -m pytest tests/test_iota_q_diversity.py::TestHarmonyModelQdivSink -v`
Expected: FAIL — `AttributeError: 'HarmonyModel' object has no attribute '_qdiv_aux_losses'`.

- [ ] **Step 3: Add `_qdiv_aux_losses` in `HarmonyModel.__init__`**

Task 5 already added the list + sink wiring in a conditional block. Verify the code reads:

```python
self._qdiv_aux_losses: list[torch.Tensor] = []
# ...
qdiv_sink = (
    self._qdiv_aux_losses
    if config.engram_qdiv_enabled else None
)
```

If `_qdiv_aux_losses` was only initialized inside a conditional in Task 5, lift it to unconditional: `self._qdiv_aux_losses: list[torch.Tensor] = []` always, so external code (training loop) can reference it regardless of whether the flag is set.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd training && python3 -m pytest tests/test_iota_q_diversity.py::TestHarmonyModelQdivSink -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add training/ct87/model.py training/tests/test_iota_q_diversity.py
git commit -m "feat(model): add _qdiv_aux_losses sink to HarmonyModel"
```

---

### Task 7: Two new presets (`tiny_engram_xattn_capgap_qdiv`, `tiny_engram_xattn_capgap_vcontrast_qdiv`)

**Files:**
- Modify: `training/ct87/model.py` (`HarmonyModelConfig` staticmethods)
- Test: `training/tests/test_iota_q_diversity.py` (append)

- [ ] **Step 1: Write failing tests for the presets**

Append to `training/tests/test_iota_q_diversity.py`:

```python
class TestIotaPresets:
    def test_iota_1_preset_qdiv_on_vcontrast_off(self):
        from ct87.model import HarmonyModelConfig
        c = HarmonyModelConfig.tiny_engram_xattn_capgap_qdiv()
        assert c.engram_qdiv_enabled is True
        assert c.engram_vcontrast_enabled is False
        assert c.engram_xattn_enabled is True
        assert len(c.engram_inject_layers) > 0

    def test_iota_2_preset_both_on(self):
        from ct87.model import HarmonyModelConfig
        c = HarmonyModelConfig.tiny_engram_xattn_capgap_vcontrast_qdiv()
        assert c.engram_qdiv_enabled is True
        assert c.engram_vcontrast_enabled is True
        assert c.engram_xattn_enabled is True
        assert len(c.engram_inject_layers) > 0

    def test_iota_presets_use_default_lambdas(self):
        from ct87.model import HarmonyModelConfig
        c1 = HarmonyModelConfig.tiny_engram_xattn_capgap_qdiv()
        assert c1.engram_qdiv_lambda == 0.01
        assert c1.engram_qdiv_warmup_steps == 200
        c2 = HarmonyModelConfig.tiny_engram_xattn_capgap_vcontrast_qdiv()
        assert c2.engram_qdiv_lambda == 0.01
        assert c2.engram_vcontrast_lambda == 0.1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd training && python3 -m pytest tests/test_iota_q_diversity.py::TestIotaPresets -v`
Expected: FAIL — `AttributeError: type object 'HarmonyModelConfig' has no attribute 'tiny_engram_xattn_capgap_qdiv'`.

- [ ] **Step 3: Add two preset staticmethods**

In `training/ct87/model.py`, after the existing `tiny_engram_xattn_capgap_vcontrast` staticmethod:

```python
    @staticmethod
    def tiny_engram_xattn_capgap_qdiv() -> "HarmonyModelConfig":
        """ι_1: capgap baseline + Q-div aux only (no V-contrast).

        Ablation test: does load-balancing Q's retrieval distribution alone
        unstick the η-B content-invariance, or is V-side pressure also needed?
        Spec: docs/superpowers/specs/2026-04-17-iota-q-diversity-design.md
        """
        config = HarmonyModelConfig.tiny_engram_xattn_capgap()
        config.engram_qdiv_enabled = True
        config.__post_init__()
        return config

    @staticmethod
    def tiny_engram_xattn_capgap_vcontrast_qdiv() -> "HarmonyModelConfig":
        """ι_2: capgap + V-contrast + Q-div together.

        Combined shortcut-closure test: does pressuring V toward content-
        sensitivity AND Q toward diversity jointly content-route at 40M?
        Spec: docs/superpowers/specs/2026-04-17-iota-q-diversity-design.md
        """
        config = HarmonyModelConfig.tiny_engram_xattn_capgap_vcontrast()
        config.engram_qdiv_enabled = True
        config.__post_init__()
        return config
```

- [ ] **Step 4: Register presets in the preset registry**

Locate the preset lookup (where `tiny_engram_xattn_capgap_vcontrast` is registered — typically a dict or if-ladder in `model.py` or `train.py`). Add entries for the two new presets:

```python
# Wherever preset_name -> HarmonyModelConfig.factory mapping lives:
"tiny_engram_xattn_capgap_qdiv":
    HarmonyModelConfig.tiny_engram_xattn_capgap_qdiv,
"tiny_engram_xattn_capgap_vcontrast_qdiv":
    HarmonyModelConfig.tiny_engram_xattn_capgap_vcontrast_qdiv,
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd training && python3 -m pytest tests/test_iota_q_diversity.py::TestIotaPresets -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add training/ct87/model.py training/tests/test_iota_q_diversity.py
git commit -m "feat(model): add iota-1 and iota-2 presets"
```

---

### Task 8: CLI flags with symmetric flag/preset rejection

**Files:**
- Modify: `training/ct87/train.py` (argument parser + validation)
- Test: `training/tests/test_iota_q_diversity.py` (append subprocess tests)

- [ ] **Step 1: Write failing subprocess tests for CLI validation**

Append to `training/tests/test_iota_q_diversity.py`:

```python
REPO_TRAINING_ROOT = Path(__file__).resolve().parent.parent


def _run_train_py(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "ct87/train.py", *args],
        cwd=REPO_TRAINING_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )


class TestCliFlagPresetConsistency:
    def test_qdiv_flag_without_preset_rejected(self):
        # Non-qdiv preset + --engram-qdiv should exit non-zero.
        result = _run_train_py([
            "--preset", "tiny_engram_xattn_capgap",
            "--engram-qdiv",
            "--help",  # don't actually train; just trip the validator
        ])
        # The validator runs before training; in --help mode it may not fire.
        # Use an invocation that goes past arg parsing but exits before
        # training:
        result = _run_train_py([
            "--preset", "tiny_engram_xattn_capgap",
            "--engram-qdiv",
            "--max-steps", "0",
            "--val-data", "/tmp/does-not-exist",
        ])
        assert result.returncode != 0
        assert (
            "--engram-qdiv" in (result.stderr + result.stdout)
            and ("preset" in (result.stderr + result.stdout).lower())
        )

    def test_qdiv_lambda_without_enabled_rejected(self):
        # Override flag for a non-qdiv preset should exit non-zero.
        result = _run_train_py([
            "--preset", "tiny_engram_xattn_capgap",
            "--engram-qdiv-lambda", "0.02",
            "--max-steps", "0",
            "--val-data", "/tmp/does-not-exist",
        ])
        assert result.returncode != 0
        txt = result.stderr + result.stdout
        assert "engram-qdiv" in txt.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd training && python3 -m pytest tests/test_iota_q_diversity.py::TestCliFlagPresetConsistency -v`
Expected: FAIL — `--engram-qdiv` is not a valid argparse argument.

- [ ] **Step 3: Add CLI args + symmetric validation in `train.py`**

In `training/ct87/train.py`, locate the argument parser definition (search for `--engram-vcontrast`). Add parallel Q-div flags in the same block:

```python
parser.add_argument(
    "--engram-qdiv",
    action="store_true",
    help=(
        "Enable Q-side load-balancing aux loss. Must match the selected "
        "preset's engram_qdiv_enabled value."
    ),
)
parser.add_argument(
    "--engram-qdiv-lambda",
    type=float,
    default=None,
    help=(
        "Override lambda for Q-div aux loss. Requires --engram-qdiv + a "
        "qdiv-enabled preset. None = use preset's value."
    ),
)
parser.add_argument(
    "--engram-qdiv-warmup-steps",
    type=int,
    default=None,
    help=(
        "Override warmup steps for Q-div lambda. Requires --engram-qdiv + "
        "a qdiv-enabled preset. None = use preset's value."
    ),
)
```

Then, near the existing V-contrast validation block (after parsing args and loading the config), add:

```python
if args.engram_qdiv != config.engram_qdiv_enabled:
    print(
        "Error: --engram-qdiv must match the selected preset's "
        f"engram_qdiv_enabled (preset={config.engram_qdiv_enabled}, "
        f"flag={args.engram_qdiv}).",
        file=sys.stderr,
    )
    sys.exit(1)
if not config.engram_qdiv_enabled and (
    args.engram_qdiv_lambda is not None
    or args.engram_qdiv_warmup_steps is not None
):
    print(
        "Error: --engram-qdiv-{lambda,warmup-steps} require a Q-div "
        "preset + --engram-qdiv.",
        file=sys.stderr,
    )
    sys.exit(1)

# Apply overrides if present
if args.engram_qdiv_lambda is not None:
    config.engram_qdiv_lambda = args.engram_qdiv_lambda
if args.engram_qdiv_warmup_steps is not None:
    config.engram_qdiv_warmup_steps = args.engram_qdiv_warmup_steps
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd training && python3 -m pytest tests/test_iota_q_diversity.py::TestCliFlagPresetConsistency -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add training/ct87/train.py training/tests/test_iota_q_diversity.py
git commit -m "feat(train): add --engram-qdiv* CLI flags with symmetric validation"
```

---

### Task 9: train.py drain + λ warmup + total-loss contribution

**Files:**
- Modify: `training/ct87/train.py` (training step)
- Test: `training/tests/test_iota_q_diversity.py` (append)

- [ ] **Step 1: Write failing test for the drain + scaling**

Append to `training/tests/test_iota_q_diversity.py`:

```python
class TestQdivTrainingStep:
    """Mid-training behavior: sinks drain per step, lambda warmup applies,
    total_loss includes scaled qdiv term."""

    def test_qdiv_sink_drained_per_step(self):
        # Use the training-loop helper directly if train.py exposes one,
        # otherwise verify via a subprocess smoke.
        # This test uses a minimal model-level simulation of the drain.
        from ct87.model import HarmonyModel, HarmonyModelConfig
        from ct87.train import lambda_schedule

        c = HarmonyModelConfig.tiny_engram_xattn_capgap_qdiv()
        model = HarmonyModel(c)
        model.train()
        B, L = 1, 8
        input_ids = torch.randint(0, c.vocab_size, (B, L))

        # Step 1
        model._qdiv_aux_losses.clear()
        _ = model(input_ids)
        assert len(model._qdiv_aux_losses) == len(c.engram_inject_layers)
        per_layer_qd = list(model._qdiv_aux_losses)
        model._qdiv_aux_losses.clear()
        assert len(model._qdiv_aux_losses) == 0
        total = torch.stack(per_layer_qd).sum()
        assert total.item() >= 1.0 * len(c.engram_inject_layers)

        # Step 2 should also fill the sink after clearing.
        _ = model(input_ids)
        assert len(model._qdiv_aux_losses) == len(c.engram_inject_layers)

    def test_lambda_schedule_warmup_linear(self):
        from ct87.train import lambda_schedule
        # Reuse existing helper from PR #250.
        assert lambda_schedule(step=0, target=0.01, warmup_steps=200) == 0.0
        assert lambda_schedule(step=100, target=0.01, warmup_steps=200) == pytest.approx(0.005)
        assert lambda_schedule(step=200, target=0.01, warmup_steps=200) == pytest.approx(0.01)
        assert lambda_schedule(step=500, target=0.01, warmup_steps=200) == pytest.approx(0.01)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd training && python3 -m pytest tests/test_iota_q_diversity.py::TestQdivTrainingStep -v`
Expected: First test may pass (drain is inherent to the sink design + forward); second test passes if `lambda_schedule` already exists from PR #250. If the second test passes but the first fails, fix the first.

If both pass, proceed to Step 3 (implementation already works from Tasks 4+6); the remaining work is wiring the total-loss addition.

- [ ] **Step 3: Wire qdiv aux into training-step total loss**

In `training/ct87/train.py`, locate the training-step code where `loss.backward()` is called (search for `_contrastive_aux_losses` to find the V-contrast block). Immediately parallel to the V-contrast drain, add the Q-div drain:

```python
# ... after loss = criterion(logits, targets), before backward:

aux_vcontrast_total = None
aux_qdiv_total = None
lam_vcontrast = 0.0
lam_qdiv = 0.0
per_layer_vc: list[torch.Tensor] = []
per_layer_qd: list[torch.Tensor] = []

if config.engram_vcontrast_enabled:
    per_layer_vc = list(model._contrastive_aux_losses)
    model._contrastive_aux_losses.clear()
    if per_layer_vc:
        aux_vcontrast_total = torch.stack(per_layer_vc).sum()
        lam_vcontrast = lambda_schedule(
            step,
            config.engram_vcontrast_lambda,
            config.engram_vcontrast_warmup_steps,
        )

if config.engram_qdiv_enabled:
    per_layer_qd = list(model._qdiv_aux_losses)
    model._qdiv_aux_losses.clear()
    if per_layer_qd:
        aux_qdiv_total = torch.stack(per_layer_qd).sum()
        lam_qdiv = lambda_schedule(
            step,
            config.engram_qdiv_lambda,
            config.engram_qdiv_warmup_steps,
        )

total_loss = loss
if aux_vcontrast_total is not None:
    total_loss = total_loss + lam_vcontrast * aux_vcontrast_total
if aux_qdiv_total is not None:
    total_loss = total_loss + lam_qdiv * aux_qdiv_total

total_loss.backward()
```

If the existing V-contrast code already structures the drain as above, just add the Q-div block parallel to it.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd training && python3 -m pytest tests/test_iota_q_diversity.py::TestQdivTrainingStep -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add training/ct87/train.py training/tests/test_iota_q_diversity.py
git commit -m "feat(train): drain qdiv aux sink per step with lambda warmup"
```

---

### Task 10: CSV columns (conditional, per-layer, sorted) + console + end-of-run summary

**Files:**
- Modify: `training/ct87/train.py` (CSV header, per-step row writer, console output, summary block)
- Test: `training/tests/test_iota_q_diversity.py` (append subprocess tests)

- [ ] **Step 1: Write failing subprocess tests for CSV structure**

Append to `training/tests/test_iota_q_diversity.py`:

```python
import csv
import tempfile


class TestCsvQdivColumns:
    """CSV column presence is gated on engram_qdiv_enabled (matches PR #250's
    pattern for V-contrast columns)."""

    def _run_one_step(self, preset: str, extra_args: list[str], tmpdir: Path):
        return _run_train_py([
            "--preset", preset,
            *extra_args,
            "--max-steps", "1",
            "--val-data", "/tmp/does-not-exist",  # short-circuit; test only
                                                   # exercises CLI + setup
        ])

    def test_qdiv_cols_present_when_enabled(self, tmp_path):
        # Use ι_1 preset; it has engram_qdiv_enabled=True.
        result = _run_train_py([
            "--preset", "tiny_engram_xattn_capgap_qdiv",
            "--engram-qdiv",
            "--max-steps", "1",
            "--batch-size", "1", "--seq-len", "8",
            "--val-data", "/tmp/does-not-exist",
            "--log-dir", str(tmp_path),
        ])
        # Expect run to complete or fail in a controlled way; either way, the
        # CSV header should reflect the qdiv columns when they would be written.
        # Inspect the captured header:
        csvs = list(tmp_path.rglob("*.csv"))
        assert len(csvs) >= 1, f"No CSV produced; stderr={result.stderr!r}"
        with open(csvs[0]) as fh:
            header = next(csv.reader(fh))
        assert "qdiv_aux_loss" in header
        assert "qdiv_lambda" in header
        assert any(h.startswith("qdiv_aux_L") for h in header)

    def test_qdiv_cols_absent_when_disabled(self, tmp_path):
        result = _run_train_py([
            "--preset", "tiny_engram_xattn_capgap",
            "--max-steps", "1",
            "--batch-size", "1", "--seq-len", "8",
            "--val-data", "/tmp/does-not-exist",
            "--log-dir", str(tmp_path),
        ])
        csvs = list(tmp_path.rglob("*.csv"))
        assert len(csvs) >= 1, f"No CSV produced; stderr={result.stderr!r}"
        with open(csvs[0]) as fh:
            header = next(csv.reader(fh))
        assert not any(h.startswith("qdiv_") for h in header), (
            f"qdiv columns leaked into non-qdiv preset header: {header}"
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd training && python3 -m pytest tests/test_iota_q_diversity.py::TestCsvQdivColumns -v`
Expected: FAIL — no `qdiv_*` columns in any CSV yet.

- [ ] **Step 3: Add conditional Q-div CSV columns**

In `training/ct87/train.py`, locate the CSV header construction (search for `vcontrast_aux_loss` or `vcontrast_cols`). Add a parallel block:

```python
vcontrast_cols: list[str] = []
if config.engram_vcontrast_enabled:
    vcontrast_cols = [
        "vcontrast_aux_loss",
        *(
            f"vcontrast_aux_L{i}"
            for i in sorted(config.engram_inject_layers or ())
        ),
        "vcontrast_lambda",
    ]

qdiv_cols: list[str] = []
if config.engram_qdiv_enabled:
    qdiv_cols = [
        "qdiv_aux_loss",
        *(
            f"qdiv_aux_L{i}"
            for i in sorted(config.engram_inject_layers or ())
        ),
        "qdiv_lambda",
    ]

expected_header = [
    # ... existing 16 base columns ...
    *vcontrast_cols,
    *qdiv_cols,
    # ... existing hg_slot_N trailing columns if any ...
]

# The hg_slot-count computation must remain dynamic:
n_vcontrast_cols = len(vcontrast_cols)
n_qdiv_cols = len(qdiv_cols)
n_hg_slots = len(expected_header) - (16 + n_vcontrast_cols + n_qdiv_cols)
```

If the header is constructed via list concatenation in a different style, preserve that style — just insert `*qdiv_cols` immediately after `*vcontrast_cols`.

- [ ] **Step 4: Update the per-step row writer**

Locate the code that assembles the per-step row dict or list (search for `vcontrast_aux_loss` in the writer). Add parallel entries for Q-div:

```python
row = {
    # ... existing 16 base cols ...
}

if config.engram_vcontrast_enabled:
    row["vcontrast_aux_loss"] = (
        aux_vcontrast_total.detach().item()
        if aux_vcontrast_total is not None else 0.0
    )
    for layer_key, layer_loss in zip(
        (str(i) for i in sorted(config.engram_inject_layers)),
        per_layer_vc,
        strict=True,
    ):
        row[f"vcontrast_aux_L{layer_key}"] = layer_loss.detach().item()
    row["vcontrast_lambda"] = lam_vcontrast

if config.engram_qdiv_enabled:
    row["qdiv_aux_loss"] = (
        aux_qdiv_total.detach().item()
        if aux_qdiv_total is not None else 0.0
    )
    for layer_key, layer_loss in zip(
        (str(i) for i in sorted(config.engram_inject_layers)),
        per_layer_qd,
        strict=True,
    ):
        row[f"qdiv_aux_L{layer_key}"] = layer_loss.detach().item()
    row["qdiv_lambda"] = lam_qdiv
```

- [ ] **Step 5: Add console output for Q-div**

Locate the console print for V-contrast (search for `[vcontrast]`). Add parallel Q-div print:

```python
if config.engram_qdiv_enabled and aux_qdiv_total is not None:
    per_layer_strs = [f"{x.item():.4f}" for x in per_layer_qd]
    print(
        f"  [qdiv] total={aux_qdiv_total.item():.4f} "
        f"lambda={lam_qdiv:.4f} "
        f"per-layer=[{', '.join(per_layer_strs)}]"
    )
```

- [ ] **Step 6: Add end-of-run summary for Q-div**

Locate the end-of-run summary block (search for `Final step=` or the block at the very end of the training loop). Add parallel Q-div summary:

```python
if config.engram_qdiv_enabled:
    final_qdiv_total = (
        aux_qdiv_total.item() if aux_qdiv_total is not None else 0.0
    )
    final_per_layer = (
        [x.item() for x in per_layer_qd] if per_layer_qd else []
    )
    print(f"[qdiv] Final step={step}")
    print(f"    last_aux_total = {final_qdiv_total:.6f}")
    for i, v in zip(sorted(config.engram_inject_layers), final_per_layer):
        print(f"    last_aux_L{i} = {v:.6f}")
    print(f"    final_lambda = {lam_qdiv:.6f}")
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `cd training && python3 -m pytest tests/test_iota_q_diversity.py::TestCsvQdivColumns -v`
Expected: 2 passed.

- [ ] **Step 8: Commit**

```bash
git add training/ct87/train.py training/tests/test_iota_q_diversity.py
git commit -m "feat(train): emit qdiv CSV cols + console output + end-of-run summary"
```

---

### Task 11: Multi-layer ascending-order invariant test

**Files:**
- Test: `training/tests/test_iota_q_diversity.py` (append)

- [ ] **Step 1: Write test for ascending-layer sink order**

Append to `training/tests/test_iota_q_diversity.py`:

```python
class TestSinkLayerOrder:
    """Regression test: sinks receive per-layer aux losses in ascending
    layer order regardless of engram_inject_layers declaration order.
    HarmonyModel.forward iterates i in range(num_layers), so the sink
    append order is forward-order, not declaration-order."""

    def test_sinks_receive_ascending_layer_order_qdiv(self):
        from ct87.model import HarmonyModel, HarmonyModelConfig

        def build_and_run(inject_layers):
            torch.manual_seed(0)
            c = HarmonyModelConfig.tiny_engram_xattn_capgap()
            c.engram_inject_layers = inject_layers
            c.engram_qdiv_enabled = True
            c.__post_init__()
            model = HarmonyModel(c)
            model.train()
            input_ids = torch.randint(0, c.vocab_size, (1, 8))
            model._qdiv_aux_losses.clear()
            _ = model(input_ids)
            return [t.item() for t in model._qdiv_aux_losses]

        ascending = build_and_run((1, 2))
        out_of_order = build_and_run((2, 1))
        assert ascending == pytest.approx(out_of_order, abs=1e-6)
```

- [ ] **Step 2: Run test**

Run: `cd training && python3 -m pytest tests/test_iota_q_diversity.py::TestSinkLayerOrder -v`
Expected: PASS (the invariant is inherent to `HarmonyModel.forward` iteration order; no code change needed, just coverage).

- [ ] **Step 3: Commit**

```bash
git add training/tests/test_iota_q_diversity.py
git commit -m "test(iota): ascending-layer-order invariant for qdiv sink"
```

---

### Task 12: One-step integration smoke test for ι₁ preset

**Files:**
- Test: `training/tests/test_iota_q_diversity.py` (append)

- [ ] **Step 1: Write one-step smoke test**

Append to `training/tests/test_iota_q_diversity.py`:

```python
class TestOneStepSmokes:
    """End-to-end: train.py runs one step under each iota preset, writes
    CSV, sinks populate correctly. Subprocess form — isolates train.py
    process state from pytest's process."""

    def test_iota_1_one_step_completes(self, tmp_path):
        result = _run_train_py([
            "--preset", "tiny_engram_xattn_capgap_qdiv",
            "--engram-qdiv",
            "--max-steps", "1",
            "--batch-size", "1", "--seq-len", "8",
            "--val-data", "/tmp/does-not-exist",
            "--log-dir", str(tmp_path),
        ])
        # Run may fail on val loader but must reach the step-1 CSV write.
        csvs = list(tmp_path.rglob("*.csv"))
        assert len(csvs) >= 1, (
            f"Expected a CSV file; got stderr={result.stderr!r}"
        )
        with open(csvs[0]) as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) >= 1
        first_row = rows[0]
        assert "qdiv_aux_loss" in first_row
        assert "qdiv_lambda" in first_row
        qdiv_val = float(first_row["qdiv_aux_loss"])
        assert qdiv_val >= 0.0  # MoE loss is non-negative
```

- [ ] **Step 2: Run test**

Run: `cd training && python3 -m pytest tests/test_iota_q_diversity.py::TestOneStepSmokes::test_iota_1_one_step_completes -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add training/tests/test_iota_q_diversity.py
git commit -m "test(iota): one-step smoke for iota-1 preset"
```

---

### Task 13: One-step integration smoke test for ι₂ preset

**Files:**
- Test: `training/tests/test_iota_q_diversity.py` (append)

- [ ] **Step 1: Write one-step smoke test for combined ι₂**

Append to `training/tests/test_iota_q_diversity.py`:

```python
    def test_iota_2_one_step_completes(self, tmp_path):
        result = _run_train_py([
            "--preset", "tiny_engram_xattn_capgap_vcontrast_qdiv",
            "--engram-vcontrast",
            "--engram-qdiv",
            "--max-steps", "1",
            "--batch-size", "1", "--seq-len", "8",
            "--val-data", "/tmp/does-not-exist",
            "--log-dir", str(tmp_path),
        ])
        csvs = list(tmp_path.rglob("*.csv"))
        assert len(csvs) >= 1, (
            f"Expected a CSV file; got stderr={result.stderr!r}"
        )
        with open(csvs[0]) as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) >= 1
        first_row = rows[0]
        # Both aux types populated.
        assert "vcontrast_aux_loss" in first_row
        assert "qdiv_aux_loss" in first_row
        vcontrast_val = float(first_row["vcontrast_aux_loss"])
        qdiv_val = float(first_row["qdiv_aux_loss"])
        assert 0.0 <= vcontrast_val <= 10.0  # aux bounded, per-layer summed
        assert qdiv_val >= 0.0
```

- [ ] **Step 2: Run test**

Run: `cd training && python3 -m pytest tests/test_iota_q_diversity.py::TestOneStepSmokes::test_iota_2_one_step_completes -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add training/tests/test_iota_q_diversity.py
git commit -m "test(iota): one-step smoke for iota-2 preset"
```

---

### Task 14: Full suite verification + final commit

**Files:**
- Verify only

- [ ] **Step 1: Run all ι tests**

Run: `cd training && python3 -m pytest tests/test_iota_q_diversity.py -v 2>&1 | tail -30`
Expected: ~17 tests total. On Python 3.10+, all pass. On local Python 3.9.6, any pre-existing `zip(strict=True)` failures in this file follow PR #250's pattern.

- [ ] **Step 2: Run V-contrast tests (regression guard)**

Run: `cd training && python3 -m pytest tests/test_v_contrast.py -v 2>&1 | tail -30`
Expected: Same pass/fail count as on `origin/main` before this branch (the refactor should not introduce new failures).

- [ ] **Step 3: Run the rest of the training test suite**

Run: `cd training && python3 -m pytest tests/ -v 2>&1 | tail -40`
Expected: Pass counts match or exceed the `origin/main` baseline at `6b90ed7`.

- [ ] **Step 4: Verify script `--help` renders**

Run: `cd training && python3 ct87/train.py --help 2>&1 | grep -E "engram-qdiv"`
Expected: 3 lines showing `--engram-qdiv`, `--engram-qdiv-lambda`, `--engram-qdiv-warmup-steps`.

- [ ] **Step 5: No code changes — this task verifies only. If all prior tasks ran clean, no commit needed here.**

If any step 1-4 failed, open a fix task before continuing. Do NOT create the PR with failing tests.

---

## Self-review

### Spec coverage

- Section 1 (architecture) → Tasks 4, 5, 6, 7 (unified class + removal + model sink + presets).
- Section 2 (loss math + `compute_qdiv_aux`) → Task 2.
- Section 3 (module refactor) → Tasks 3, 4, 5.
- Section 4 (training-loop integration) → Tasks 8, 9, 10.
- Section 5 (`HarmonyModelConfig` + presets) → Tasks 1, 7.
- Section 6 (testing strategy, ~17 tests) → Tasks 1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 13 (all 17 tests mapped). Test count: 5 (T1) + 5 (T2) + 2 (T3) + 5 (T4) + 2 (T6) + 3 (T7) + 2 (T8) + 2 (T9) + 2 (T10) + 1 (T11) + 1 (T12) + 1 (T13) = **31 tests** (exceeds the spec's "~17" estimate because this plan adds CLI-consistency and preset-specific tests beyond the spec's minimum).
- Section 7 (success criteria + runbook) → not implementation-plan content; delegated to the post-merge KRILE runbook.

### Placeholder scan

- No "TBD" / "TODO" in any task.
- No "similar to Task N" without full code.
- All code blocks are complete.

### Type / name consistency

- `_qdiv_aux_losses`: consistently named across Tasks 5, 6, 9, 10, 11.
- `engram_qdiv_enabled`, `engram_qdiv_lambda`, `engram_qdiv_warmup_steps`: consistent in Tasks 1, 7, 8, 9, 10.
- `compute_qdiv_aux(topk_idx, attn, table_size)`: signature stable in Tasks 2, 4.
- `vcontrast_sink`, `qdiv_sink`: consistent as keyword-only kwargs on `GatedEngramInjection` in Tasks 4, 5.
- `tiny_engram_xattn_capgap_qdiv`, `tiny_engram_xattn_capgap_vcontrast_qdiv`: consistent in Tasks 7, 8, 12, 13.
- CSV column names `qdiv_aux_loss`, `qdiv_aux_L{i}`, `qdiv_lambda`: consistent in Tasks 10, 12, 13.

### Scope

- Single plan, single PR. No cross-repo work, no separate follow-up PRs needed (unlike PR #250's Task 14 which was a parallel forensic-probe PR).
- Post-merge work (KRILE training runs) is out-of-scope for this plan; spec Section 7 documents the runbook.
