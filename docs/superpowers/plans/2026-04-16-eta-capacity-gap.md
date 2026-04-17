# Experiment η-B: Capacity-Gap Pretraining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Attach a zero-init tanh-gated cross-attention engram injection at two layers of a **frozen** pretrained dense LM, train only the injection params for 2K steps, and measure whether val_loss drops below the frozen baseline — isolating the utilization question from the co-training-interference question.

**Architecture:** Extends the existing Model-δ (cross-attention engram) path with (a) multi-layer injection via `nn.ModuleDict`, (b) a `GatedEngramInjection` wrapper that applies a learnable `tanh(α)` scalar per injection point with `α` initialized at 0 (so the gate outputs zero at init, preserving the frozen baseline's behavior on step 0), and (c) a `--freeze-backbone` + `--init-from` training mode that loads a pretrained checkpoint, freezes all backbone params, and trains only engram injection params.

**Tech Stack:** PyTorch, Muon optimizer, existing `EngramCrossAttention` module, FineWeb-Edu-POC corpus, seq_len=2048.

**Hypothesis tested (H2 from the η experimental program):** if a 40M dense model *can* use a well-gated engram, it should be able to do so when the backbone is already well-trained and the injection is introduced with a zero-init gate (no step-0 perturbation). If val_loss drops, the bottleneck was co-training interference. If val_loss does not drop, the bottleneck is capacity itself.

**Success criterion:** `val_loss_with_injection < frozen_baseline_val_loss - 0.005 nats` AND `delta_removal > 0.01 nats` on the same held-out val batches (via `--zero-injection-eval`).

---

## Context and Design Constraints

The ζ runs (PR #246) demonstrated three failures that η-B is designed to avoid:

1. **Bypass valve via auxiliary module.** ζ used a 2-layer MLP decoder to predict the engram signal, with MSE pressure applied to the decoder. Because the decoder had its own full-capacity weights, it absorbed the MSE gradient without propagating utilization pressure to the transformer. η-B has **no auxiliary decoder** — the only pathway to reduce loss is for the injection itself to produce a useful additive signal to the frozen residual stream.

2. **Step-0 perturbation.** The existing xattn path injects `h + xattn_out * inject_mult` where `inject_mult=1.0` from step 0. Even with xattn's Xavier-initialized projections, the injected signal is nonzero at step 0 and perturbs the forward pass. When the backbone is also learning from scratch, it learns to route around this perturbation (the bypass-valve failure mode). η-B initializes the gate to `tanh(0)=0` so the model is **bit-exactly the frozen baseline** at step 0 and the optimizer can only open the gate if the signal helps.

3. **Anneal confound.** ζ-B annealed `inject_mult` from 1 → 0, which the ζ-2048 interpretation identified as the model "learning to route around adversarial input." η-B has **no anneal** — `inject_mult` stays at 1.0 throughout; the gate scalar is the only modulator and it's learned.

DeepSeek Engram (arXiv:2601.07372) and the retrieval-augmentation survey both support **multi-layer injection** (U-shaped scaling law: early layer for static entity binding + mid-late layer for semantic refinement). η-B injects at layers 2 and 5 of the 8-layer tiny model (matching the 25%/62% depth ratio from DeepSeek's 30-layer design).

---

## File Structure

- **Modify:** `training/ct87/engram.py` — add `GatedEngramInjection` class at end (near `EngramConsolidationDecoder`)
- **Modify:** `training/ct87/model.py`:
  - Add fields to `HarmonyModelConfig`: `engram_inject_layers: tuple[int, ...]`, `engram_gate_init: float`
  - Add config preset `tiny_engram_xattn_capgap()`
  - Add to `HarmonyModel`: `engram_injections: nn.ModuleDict | None`
  - Add method `attach_gated_engram_injections()`
  - Modify `forward()` to dispatch to multi-layer path when `engram_injections` is populated
- **Modify:** `training/ct87/train.py`:
  - Add CLI flags: `--init-from`, `--freeze-backbone`, `--engram-inject-layers`, `--engram-gate-init`
  - Add partial-load logic for `--init-from` (model weights only, fresh optimizer/step/RNG)
  - Add backbone-freeze logic after model construction
  - Wire the `tiny_engram_xattn_capgap` config preset
  - Construct `GatedEngramInjection` instances for each layer in `engram_inject_layers`
- **Create:** `training/tests/test_capacity_gap.py` — all new tests for this feature

## Key Architecture Decisions

1. **Backward compatibility.** The existing single-injection path (`engram_xattn`, `engram_injection_layer`) is **preserved unchanged** so ζ-class configs keep working. The new `engram_injections` ModuleDict path is additive; dispatch in `forward()` chooses between them based on which field is non-`None`. This mirrors how the codebase already handles ann/xattn/production paths (mutually exclusive by construction).

2. **Config is the single source of truth.** `engram_inject_layers` being non-empty means "use multi-layer gated path." Validation rejects mixing `use_xattn_engram=True` (single-point legacy) with `engram_inject_layers` non-empty.

3. **`--init-from` vs `--resume-from`.** Existing `--resume-from` restores model + optimizer + step + RNG (training continuation). New `--init-from` restores only model weights via `strict=False`, then starts fresh (step=0, fresh optimizer, fresh RNG). Missing keys (engram injection params not in source checkpoint) and unexpected keys are logged but not errors.

4. **Freeze mechanism.** After model construction and `--init-from` loading, iterate `model.named_parameters()` and set `requires_grad=False` for any param whose name does **not** start with `engram_injections.`. Optimizer is then built from `[p for p in model.parameters() if p.requires_grad]`. This gives zero optimizer memory overhead for frozen weights.

5. **Gate scalar is a `nn.Parameter`.** Initialized via `torch.full((), engram_gate_init)`. Stored inside each `GatedEngramInjection` so it moves with `.to(device)` and is discovered by the optimizer.

6. **`engram_inject_mult` stays.** The existing float attribute is reused for `--zero-injection-eval` (set to 0.0 to zero out all injection paths). In the multi-layer path, the forward applies it as `h + engram_inject_mult * tanh(α) * xattn_out` per injection point.

7. **Gradient checkpointing compatibility.** The engram injections happen **outside** `_gradient_checkpoint(layer, h)` calls, so no interaction. Existing code pattern at model.py:636-639.

8. **PyTorch module mode.** Tests use `model.train(False)` (equivalent to `.eval()`) to set inference mode — avoids accidental dropout/training-mode side effects and keeps test code compatible with lint rules that restrict Python's `eval` builtin.

---

## Tasks

### Task 1: `GatedEngramInjection` module

**Files:**
- Modify: `training/ct87/engram.py` (add class at end of file)
- Test: `training/tests/test_capacity_gap.py` (create)

**Rationale:** This is the new atomic unit — an `EngramCrossAttention` module wrapped with a learnable scalar gate applied via `tanh`. Zero-init (`α=0`) means the gate outputs 0 at step 0, so an attached `GatedEngramInjection` produces no perturbation on the first forward pass.

- [ ] **Step 1: Write the failing test**

Create `training/tests/test_capacity_gap.py`:

```python
"""Tests for η-B capacity-gap pretraining feature (ZEB-130)."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from ct87.engram import EngramCrossAttention, GatedEngramInjection
from ct87.model import HarmonyModel, HarmonyModelConfig


class TestGatedEngramInjection:
    """GatedEngramInjection wraps EngramCrossAttention with a learnable tanh gate."""

    def _make_xattn(self) -> EngramCrossAttention:
        return EngramCrossAttention(
            hidden_dim=64, engram_dim=32, num_heads=2,
            num_entries=16, top_k=4,
        )

    def test_forward_zero_at_init(self):
        """With alpha=0, tanh(alpha)=0, so output must equal input exactly."""
        torch.manual_seed(0)
        wrapper = GatedEngramInjection(self._make_xattn(), alpha_init=0.0)
        wrapper.train(False)
        h = torch.randn(2, 5, 64)
        out = wrapper(h)
        assert torch.allclose(out, torch.zeros_like(h), atol=1e-6), (
            "With alpha_init=0, gate output must be zero (tanh(0)=0)"
        )

    def test_alpha_is_learnable_parameter(self):
        """Alpha is registered as a parameter and is discoverable by optimizers."""
        wrapper = GatedEngramInjection(self._make_xattn(), alpha_init=0.5)
        param_names = dict(wrapper.named_parameters())
        assert "alpha" in param_names
        assert param_names["alpha"].requires_grad
        assert torch.allclose(
            param_names["alpha"].detach(), torch.tensor(0.5), atol=1e-6
        )

    def test_gradient_flows_to_alpha(self):
        """A loss depending on the wrapper output must produce a non-zero grad on alpha."""
        torch.manual_seed(0)
        wrapper = GatedEngramInjection(self._make_xattn(), alpha_init=0.1)
        wrapper.train(True)
        h = torch.randn(2, 5, 64, requires_grad=False)
        out = wrapper(h)
        loss = out.pow(2).mean()
        loss.backward()
        assert wrapper.alpha.grad is not None
        assert wrapper.alpha.grad.abs().item() > 1e-8

    def test_forward_nonzero_when_alpha_nonzero(self):
        """With alpha != 0, output must be non-zero (injection is active)."""
        torch.manual_seed(0)
        wrapper = GatedEngramInjection(self._make_xattn(), alpha_init=1.0)
        wrapper.train(False)
        h = torch.randn(2, 5, 64)
        out = wrapper(h)
        assert out.abs().max().item() > 1e-4

    def test_alpha_init_default_is_zero(self):
        """Default alpha_init is 0.0 (the safe zero-perturbation value)."""
        wrapper = GatedEngramInjection(self._make_xattn())
        assert wrapper.alpha.detach().item() == 0.0
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd training && python3 -m pytest tests/test_capacity_gap.py::TestGatedEngramInjection -v`

Expected: ImportError for `GatedEngramInjection` (not yet defined).

- [ ] **Step 3: Implement `GatedEngramInjection`**

Append to `training/ct87/engram.py`:

```python
class GatedEngramInjection(nn.Module):
    """Cross-attention engram injection gated by a learnable scalar (η-B / ZEB-130).

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
    """

    def __init__(
        self,
        engram_xattn: EngramCrossAttention,
        alpha_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.engram_xattn = engram_xattn
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        xattn_out = self.engram_xattn(hidden_state)
        return torch.tanh(self.alpha) * xattn_out
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cd training && python3 -m pytest tests/test_capacity_gap.py::TestGatedEngramInjection -v`

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add training/ct87/engram.py training/tests/test_capacity_gap.py
git commit -m "feat(training): add GatedEngramInjection for η-B capacity-gap (ZEB-130)"
```

---

### Task 2: Config fields + preset for multi-layer injection

**Files:**
- Modify: `training/ct87/model.py` (HarmonyModelConfig dataclass + presets)
- Test: `training/tests/test_capacity_gap.py`

- [ ] **Step 1: Write failing tests**

Add to `training/tests/test_capacity_gap.py`:

```python
class TestCapacityGapConfig:
    """Config fields and preset for η-B."""

    def test_config_has_inject_layers_default_empty(self):
        """Default config has empty engram_inject_layers (legacy behavior preserved)."""
        c = HarmonyModelConfig.tiny()
        assert c.engram_inject_layers == ()

    def test_config_has_gate_init_default_zero(self):
        """Default config has engram_gate_init=0.0."""
        c = HarmonyModelConfig.tiny()
        assert c.engram_gate_init == 0.0

    def test_capgap_preset_sets_inject_layers(self):
        """tiny_engram_xattn_capgap uses multi-layer injection at layers 2 and 5."""
        c = HarmonyModelConfig.tiny_engram_xattn_capgap()
        assert c.engram_inject_layers == (2, 5)
        assert c.engram_gate_init == 0.0

    def test_capgap_preset_does_not_set_legacy_xattn_flag(self):
        """capgap preset uses the new path, not the legacy single-point path."""
        c = HarmonyModelConfig.tiny_engram_xattn_capgap()
        assert c.use_xattn_engram is False
        assert c.use_ann_engram is False

    def test_config_rejects_mixing_legacy_and_multi_layer(self):
        """Config must not set use_xattn_engram=True alongside non-empty inject_layers."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            HarmonyModelConfig(
                num_layers=8, hidden_dim=512, num_query_heads=8, num_kv_heads=4,
                head_dim=64, ffn_dim=1365, vocab_size=32000, max_seq_len=4096,
                rope_theta=1e6, rms_norm_eps=1e-6, layers_per_block=2,
                engram_injection_layer=2, engram_dim=128, tie_embeddings=True,
                use_xattn_engram=True,
                engram_inject_layers=(2, 5),
            )

    def test_config_rejects_out_of_range_inject_layer(self):
        """engram_inject_layers must reference valid layer indices."""
        with pytest.raises(ValueError, match="outside"):
            HarmonyModelConfig(
                num_layers=8, hidden_dim=512, num_query_heads=8, num_kv_heads=4,
                head_dim=64, ffn_dim=1365, vocab_size=32000, max_seq_len=4096,
                rope_theta=1e6, rms_norm_eps=1e-6, layers_per_block=2,
                engram_injection_layer=2, engram_dim=128, tie_embeddings=True,
                engram_inject_layers=(2, 99),
            )
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd training && python3 -m pytest tests/test_capacity_gap.py::TestCapacityGapConfig -v`

Expected: all fail — fields don't exist yet.

- [ ] **Step 3: Add config fields and validation**

Modify `training/ct87/model.py`. Add fields to the `HarmonyModelConfig` dataclass (find the block around line 67-70 where `use_ann_engram`, `use_xattn_engram`, `use_head_gates`, `ffn_dim_overrides` are declared; add directly after `ffn_dim_overrides`):

```python
    use_ann_engram: bool = False
    use_xattn_engram: bool = False
    use_head_gates: bool = False
    ffn_dim_overrides: dict[int, int] | None = None
    # η-B / ZEB-130: multi-layer gated cross-attention injection.
    # Non-empty means use the GatedEngramInjection ModuleDict path
    # (mutually exclusive with use_xattn_engram / use_ann_engram).
    engram_inject_layers: tuple[int, ...] = ()
    engram_gate_init: float = 0.0
```

Extend `__post_init__` (around line 72) — add these checks at the top, before the `ffn_dim_overrides` block:

```python
    def __post_init__(self) -> None:
        if self.use_ann_engram and self.use_xattn_engram:
            raise ValueError(
                "use_ann_engram and use_xattn_engram are mutually "
                "exclusive - only one research-only engram injection "
                "module can be attached at a time."
            )
        # η-B multi-layer injection is mutually exclusive with the legacy
        # single-point paths (model delta xattn, model gamma ANN).
        if self.engram_inject_layers:
            if self.use_xattn_engram or self.use_ann_engram:
                raise ValueError(
                    "engram_inject_layers (multi-layer gated injection) is "
                    "mutually exclusive with use_xattn_engram / use_ann_engram "
                    "(single-point legacy paths)."
                )
            for layer_idx in self.engram_inject_layers:
                if not (0 <= layer_idx < self.num_layers):
                    raise ValueError(
                        f"engram_inject_layers has layer_idx={layer_idx} "
                        f"outside [0, {self.num_layers}) - would be silently "
                        "ignored at model construction."
                    )
        if self.ffn_dim_overrides is None:
            return
        # ... existing ffn_dim_overrides validation stays unchanged
```

Add new preset after `tiny_engram_xattn_ctrl()` (around line 258):

```python
    @staticmethod
    def tiny_engram_xattn_capgap() -> HarmonyModelConfig:
        """η-B capacity-gap: multi-layer zero-init gated xattn injection (ZEB-130).

        Attaches gated cross-attention injection at layers 2 and 5 (early +
        mid-late, matching DeepSeek Engram's U-shaped scaling). Gate alpha
        initialized to 0 so tanh(0)=0 produces no perturbation at step 0 —
        preserves the frozen pretrained baseline's behavior until the
        optimizer learns to open the gate.

        Used with --freeze-backbone and --init-from <beta_checkpoint> to test
        H2: can a well-trained backbone leverage a gated engram signal that
        the from-scratch co-training setup could not?
        """
        base = HarmonyModelConfig.tiny()
        base.engram_inject_layers = (2, 5)
        base.engram_gate_init = 0.0
        return base
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cd training && python3 -m pytest tests/test_capacity_gap.py::TestCapacityGapConfig -v`

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add training/ct87/model.py training/tests/test_capacity_gap.py
git commit -m "feat(training): add engram_inject_layers config + tiny_engram_xattn_capgap preset (ZEB-130)"
```

---

### Task 3: Multi-layer injection in HarmonyModel

**Files:**
- Modify: `training/ct87/model.py` (HarmonyModel)
- Test: `training/tests/test_capacity_gap.py`

- [ ] **Step 1: Write failing tests**

Add to `training/tests/test_capacity_gap.py`:

```python
class TestHarmonyModelMultiLayerInjection:
    """HarmonyModel supports GatedEngramInjection at multiple layers via ModuleDict."""

    def _build_model(self, device: str = "cpu") -> HarmonyModel:
        config = HarmonyModelConfig.tiny_engram_xattn_capgap()
        model = HarmonyModel(config)
        model.to(device)
        return model

    def _build_injections(
        self, config: HarmonyModelConfig
    ) -> dict[int, GatedEngramInjection]:
        from ct87.engram import EngramCrossAttention, GatedEngramInjection
        out: dict[int, GatedEngramInjection] = {}
        for layer_idx in config.engram_inject_layers:
            xattn = EngramCrossAttention(
                hidden_dim=config.hidden_dim,
                engram_dim=config.engram_dim,
                num_heads=2, num_entries=16, top_k=4,
            )
            out[layer_idx] = GatedEngramInjection(
                xattn, alpha_init=config.engram_gate_init
            )
        return out

    def test_engram_injections_default_none(self):
        """Model with empty inject_layers has engram_injections=None."""
        config = HarmonyModelConfig.tiny()  # empty inject_layers
        model = HarmonyModel(config)
        assert model.engram_injections is None

    def test_attach_gated_engram_injections(self):
        """Attaching populates engram_injections as a ModuleDict keyed by layer index."""
        model = self._build_model()
        injections = self._build_injections(model.config)
        model.attach_gated_engram_injections(injections)
        assert isinstance(model.engram_injections, nn.ModuleDict)
        assert set(model.engram_injections.keys()) == {"2", "5"}

    def test_attach_rejects_wrong_layer_indices(self):
        """Attach rejects layer indices not declared in config.engram_inject_layers."""
        model = self._build_model()
        from ct87.engram import EngramCrossAttention, GatedEngramInjection
        wrong = {
            7: GatedEngramInjection(EngramCrossAttention(
                hidden_dim=model.config.hidden_dim,
                engram_dim=model.config.engram_dim,
                num_heads=2, num_entries=16, top_k=4,
            )),
        }
        with pytest.raises(ValueError, match="layer_idx"):
            model.attach_gated_engram_injections(wrong)

    def test_attach_rejects_when_config_flag_absent(self):
        """Attach fails if config.engram_inject_layers is empty."""
        config = HarmonyModelConfig.tiny()
        model = HarmonyModel(config)
        with pytest.raises(ValueError, match="engram_inject_layers"):
            model.attach_gated_engram_injections({})

    def test_forward_zero_perturbation_at_init(self):
        """With alpha_init=0 on all injections, forward must match a no-injection baseline."""
        torch.manual_seed(123)
        config = HarmonyModelConfig.tiny_engram_xattn_capgap()
        model_capgap = HarmonyModel(config)
        injections = self._build_injections(config)
        model_capgap.attach_gated_engram_injections(injections)
        model_capgap.train(False)

        torch.manual_seed(123)
        config_plain = HarmonyModelConfig.tiny()
        model_plain = HarmonyModel(config_plain)
        model_plain.train(False)

        ids = torch.randint(0, config.vocab_size, (2, 16))
        with torch.no_grad():
            out_capgap = model_capgap(ids)
            out_plain = model_plain(ids)
        # Zero-init gate means the injection adds exactly zero to the residual;
        # outputs must be identical up to floating-point noise.
        assert torch.allclose(out_capgap, out_plain, atol=1e-5)

    def test_forward_diverges_when_gate_opened(self):
        """Opening the gate must cause outputs to diverge from the no-injection baseline."""
        torch.manual_seed(123)
        config = HarmonyModelConfig.tiny_engram_xattn_capgap()
        model = HarmonyModel(config)
        injections = self._build_injections(config)
        model.attach_gated_engram_injections(injections)
        model.train(False)

        ids = torch.randint(0, config.vocab_size, (2, 16))
        with torch.no_grad():
            out_closed = model(ids)
            for injection in model.engram_injections.values():
                injection.alpha.data = torch.tensor(1.0)
            out_open = model(ids)
        assert not torch.allclose(out_closed, out_open, atol=1e-3)

    def test_inject_mult_zero_disables_all_layers(self):
        """Setting engram_inject_mult=0 must zero all multi-layer injections."""
        torch.manual_seed(123)
        config = HarmonyModelConfig.tiny_engram_xattn_capgap()
        model = HarmonyModel(config)
        injections = self._build_injections(config)
        model.attach_gated_engram_injections(injections)
        for injection in model.engram_injections.values():
            injection.alpha.data = torch.tensor(1.0)
        model.train(False)

        torch.manual_seed(123)
        config_plain = HarmonyModelConfig.tiny()
        model_plain = HarmonyModel(config_plain)
        model_plain.train(False)

        ids = torch.randint(0, config.vocab_size, (2, 16))
        with torch.no_grad():
            model.engram_inject_mult = 0.0
            out_zeroed = model(ids)
            out_plain = model_plain(ids)
        assert torch.allclose(out_zeroed, out_plain, atol=1e-5)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd training && python3 -m pytest tests/test_capacity_gap.py::TestHarmonyModelMultiLayerInjection -v`

Expected: all fail — `engram_injections`, `attach_gated_engram_injections` don't exist yet.

- [ ] **Step 3: Add multi-layer injection support to HarmonyModel**

Modify `training/ct87/model.py`. In `HarmonyModel.__init__` (around line 484), add after `self.engram_xattn: EngramCrossAttention | None = None`:

```python
        self.engram_xattn: EngramCrossAttention | None = None
        # η-B multi-layer injection path (ZEB-130).
        # Populated by attach_gated_engram_injections(). ModuleDict keys are
        # str(layer_idx). Mutually exclusive with engram_ann / engram_xattn.
        self.engram_injections: nn.ModuleDict | None = None
```

Add method after `attach_engram_xattn()` (around line 552):

```python
    def attach_gated_engram_injections(
        self, injections_by_layer: dict[int, "GatedEngramInjection"]
    ) -> None:
        """Attach the η-B multi-layer gated injection modules (ZEB-130).

        Called by the training script after constructing the model.
        `injections_by_layer` must cover exactly the layer indices listed
        in `config.engram_inject_layers`.

        Registers the injections as submodules (keyed by str(layer_idx))
        so their parameters are discovered by the optimizer and their
        buffers move with `.to(device)`.
        """
        from ct87.engram import GatedEngramInjection  # avoid top-level circular

        if not self.config.engram_inject_layers:
            raise ValueError(
                "attach_gated_engram_injections() called but "
                "config.engram_inject_layers is empty - use "
                "HarmonyModelConfig.tiny_engram_xattn_capgap() (or set the "
                "field directly) before attaching."
            )
        expected = set(self.config.engram_inject_layers)
        got = set(injections_by_layer.keys())
        if got != expected:
            raise ValueError(
                f"attach_gated_engram_injections() got layer_idx keys "
                f"{sorted(got)} but config declares {sorted(expected)}."
            )
        for inj in injections_by_layer.values():
            if not isinstance(inj, GatedEngramInjection):
                raise TypeError(
                    "attach_gated_engram_injections() values must be "
                    "GatedEngramInjection instances."
                )
        self.engram_injections = nn.ModuleDict({
            str(layer_idx): injections_by_layer[layer_idx]
            for layer_idx in self.config.engram_inject_layers
        })
```

Modify the forward loop in `HarmonyModel.forward()` (around line 641-655) to dispatch to the multi-layer path:

```python
            # Engram injection at configured layer(s).
            # Precedence (mutually exclusive by construction):
            #   η-B multi-layer > Model delta xattn > Model gamma ANN > production
            if self.engram_injections is not None:
                key = str(i)
                if key in self.engram_injections:
                    injection_out = self.engram_injections[key](h)
                    h = h + self.engram_inject_mult * injection_out
            elif i == self.config.engram_injection_layer:
                if self.engram_xattn is not None:
                    self._last_pre_injection_hidden = h
                    xattn_out = self.engram_xattn(h)
                    self._last_xattn_output = xattn_out
                    h = h + xattn_out * self.engram_inject_mult
                elif self.engram_ann is not None:
                    residual, gate = self.engram_ann(h)
                    h = h + residual
                    self._last_ann_gate = gate
                elif engram_embeddings is not None:
                    h = h + self.engram_residual(h, engram_embeddings)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cd training && python3 -m pytest tests/test_capacity_gap.py::TestHarmonyModelMultiLayerInjection -v`

Expected: 7 passed.

Also run the pre-existing test suite to confirm no regressions in the legacy single-point path:

Run: `cd training && python3 -m pytest tests/test_consolidation.py tests/test_model.py -v --tb=short 2>&1 | tail -30`

Expected: all consolidation/model tests still pass.

- [ ] **Step 5: Commit**

```bash
git add training/ct87/model.py training/tests/test_capacity_gap.py
git commit -m "feat(training): multi-layer GatedEngramInjection in HarmonyModel (ZEB-130)"
```

---

### Task 4: `--init-from` CLI flag (partial weight restore, fresh training state)

**Files:**
- Modify: `training/ct87/train.py`
- Test: `training/tests/test_capacity_gap.py`

**Rationale:** `--resume-from` restores model + optimizer + step + RNG (training continuation). η-B needs a **distinct** mode: load weights only from a β-class checkpoint, then start fresh (step=0, fresh optimizer, fresh RNG). Missing keys (the engram injections aren't in the source checkpoint) must be permitted via `strict=False`.

- [ ] **Step 1: Write failing test**

Add to `training/tests/test_capacity_gap.py`:

```python
class TestInitFromFlag:
    """--init-from loads model weights but not training state."""

    def test_init_from_loads_backbone_weights(self, tmp_path):
        """After --init-from, a new model can load a source checkpoint via strict=False."""
        # Build a source checkpoint: tiny baseline (no engram injections).
        src_ckpt = tmp_path / "src" / "checkpoint.pt"
        src_ckpt.parent.mkdir()
        config = HarmonyModelConfig.tiny()
        model = HarmonyModel(config)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": {},
                "step": 500,
                "rng_state": torch.get_rng_state(),
                "config": config,
            },
            src_ckpt,
        )

        # Load into a capgap-configured target (which has extra engram_injections keys).
        ckpt = torch.load(src_ckpt, map_location="cpu", weights_only=False)
        new_config = HarmonyModelConfig.tiny_engram_xattn_capgap()
        new_model = HarmonyModel(new_config)
        missing, unexpected = new_model.load_state_dict(
            ckpt["model_state_dict"], strict=False
        )
        # No unexpected keys (source is strict subset of target)
        assert unexpected == []
        # Missing, if any, must only be engram_injections keys.
        for k in missing:
            assert k.startswith("engram_injections"), (
                f"Unexpected missing key outside engram_injections: {k}"
            )

    def test_train_rejects_both_init_from_and_resume_from(self):
        """CLI validation rejects using both flags simultaneously."""
        import subprocess
        import sys
        result = subprocess.run(
            [
                sys.executable, "-m", "ct87.train",
                "--init-from", "/tmp/nonexistent.pt",
                "--resume-from", "/tmp/nonexistent.pt",
                "--config", "tiny_engram_xattn_capgap",
                "--dataset", "synthetic",
                "--output-dir", "/tmp/capgap_test",
                "--steps", "1",
            ],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode != 0
        assert (
            "--init-from" in result.stderr
            and "--resume-from" in result.stderr
            and "mutually exclusive" in result.stderr.lower()
        )
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd training && python3 -m pytest tests/test_capacity_gap.py::TestInitFromFlag -v`

Expected: `test_train_rejects_both_init_from_and_resume_from` fails (CLI flag not defined).

- [ ] **Step 3: Add `--init-from` CLI flag and handling**

Modify `training/ct87/train.py`. In the argument parser (find block around line 360 with `--resume-from`), add:

```python
    parser.add_argument(
        "--resume-from", type=str, default=None,
        help=(
            "Path to a resumable checkpoint .pt file. Restores model "
            "weights, optimizer state, step, and RNG — use for training "
            "continuation."
        ),
    )
    parser.add_argument(
        "--init-from", type=str, default=None,
        help=(
            "Path to a checkpoint .pt file for weight initialization only "
            "(η-B capacity-gap / ZEB-130). Loads model weights via "
            "strict=False (missing keys for new engram injections are "
            "allowed), then starts training fresh (step=0, fresh optimizer, "
            "fresh RNG). Mutually exclusive with --resume-from."
        ),
    )
```

Add validation near the top of `main()` (right after args are parsed, and before any heavy setup). Find a suitable location — the existing validation for consolidation args is a good neighbor:

```python
    if args.init_from is not None and args.resume_from is not None:
        print(
            "Error: --init-from and --resume-from are mutually exclusive. "
            "Use --resume-from to continue training from a checkpoint; use "
            "--init-from to load weights only and start a fresh run.",
            file=sys.stderr,
        )
        sys.exit(2)
    if args.init_from is not None and not args.init_from.endswith(".pt"):
        print(
            "Error: --init-from must point to a resumable checkpoint .pt "
            "file (weights + config), not a safetensors file.",
            file=sys.stderr,
        )
        sys.exit(2)
```

Add the model-load logic. The cleanest approach is to add an `elif args.init_from is not None:` branch after the existing `if args.resume_from is not None and args.resume_from.endswith(".pt"):` block (around line 888-912). The new branch must load only weights (strict=False), log missing/unexpected keys, and NOT touch the optimizer state:

```python
    elif args.init_from is not None:
        print(f"Initializing model weights from {args.init_from}")
        ckpt = torch.load(args.init_from, map_location="cpu", weights_only=False)
        if "model_state_dict" not in ckpt:
            print(
                f"Error: {args.init_from} is not a resumable checkpoint "
                "(no model_state_dict).",
                file=sys.stderr,
            )
            sys.exit(2)
        missing, unexpected = model.load_state_dict(
            ckpt["model_state_dict"], strict=False
        )
        if unexpected:
            print(
                f"Warning: --init-from has {len(unexpected)} unexpected "
                f"keys (will be ignored): {unexpected[:5]}"
                + ("..." if len(unexpected) > 5 else "")
            )
        if missing:
            non_engram_missing = [
                k for k in missing if not k.startswith("engram_injections")
            ]
            if non_engram_missing:
                print(
                    f"Warning: --init-from missing {len(non_engram_missing)} "
                    f"non-engram keys: {non_engram_missing[:5]}"
                    + ("..." if len(non_engram_missing) > 5 else "")
                )
            else:
                print(
                    f"--init-from loaded cleanly (missing only "
                    f"{len(missing)} engram_injections keys as expected)."
                )
        # Explicitly DO NOT touch optimizer, step, or RNG: this is a fresh run.
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cd training && python3 -m pytest tests/test_capacity_gap.py::TestInitFromFlag -v`

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add training/ct87/train.py training/tests/test_capacity_gap.py
git commit -m "feat(training): add --init-from for weight-only partial restore (ZEB-130)"
```

---

### Task 5: `--freeze-backbone` CLI flag + optimizer scoping

**Files:**
- Modify: `training/ct87/train.py`
- Test: `training/tests/test_capacity_gap.py`

**Rationale:** With a frozen backbone, the optimizer sees only `engram_injections.*` parameters. This isolates utilization pressure to the injection path and keeps optimizer memory overhead near zero for the frozen 40M weights.

- [ ] **Step 1: Write failing test**

Add to `training/tests/test_capacity_gap.py`:

```python
class TestFreezeBackbone:
    """--freeze-backbone disables grad on all non-engram params."""

    def _build_and_attach(self) -> HarmonyModel:
        config = HarmonyModelConfig.tiny_engram_xattn_capgap()
        model = HarmonyModel(config)
        from ct87.engram import EngramCrossAttention, GatedEngramInjection
        injections = {
            layer_idx: GatedEngramInjection(
                EngramCrossAttention(
                    hidden_dim=config.hidden_dim,
                    engram_dim=config.engram_dim,
                    num_heads=2, num_entries=16, top_k=4,
                ),
                alpha_init=0.0,
            )
            for layer_idx in config.engram_inject_layers
        }
        model.attach_gated_engram_injections(injections)
        return model

    def test_freeze_sets_requires_grad_correctly(self):
        """After freezing, only engram_injections params have requires_grad=True."""
        model = self._build_and_attach()
        from ct87.train import freeze_backbone_for_capgap
        freeze_backbone_for_capgap(model)
        for name, p in model.named_parameters():
            if name.startswith("engram_injections"):
                assert p.requires_grad, f"{name} should be trainable"
            else:
                assert not p.requires_grad, f"{name} should be frozen"

    def test_trainable_param_count_is_small(self):
        """Frozen capgap model has O(engram_injections) trainable params, not O(backbone)."""
        model = self._build_and_attach()
        from ct87.train import freeze_backbone_for_capgap
        freeze_backbone_for_capgap(model)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        # Trainable should be << 1% of frozen (injections are small relative to 40M backbone).
        assert trainable * 100 < frozen, (
            f"Trainable ({trainable}) should be under 1% of frozen ({frozen})"
        )
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd training && python3 -m pytest tests/test_capacity_gap.py::TestFreezeBackbone -v`

Expected: fails — `freeze_backbone_for_capgap` not defined.

- [ ] **Step 3: Add freeze helper + CLI flag + optimizer wiring**

Modify `training/ct87/train.py`. Add helper near the other model-setup helpers (search for `attach_engram_xattn` or similar in train.py for neighborhood):

```python
def freeze_backbone_for_capgap(model: HarmonyModel) -> None:
    """Freeze all parameters except engram_injections.* (η-B / ZEB-130).

    Sets requires_grad=False on every parameter whose name does not start
    with "engram_injections." — the optimizer then only sees the
    GatedEngramInjection params (xattn projections + alpha scalars).
    """
    frozen_count = 0
    trainable_count = 0
    for name, param in model.named_parameters():
        if name.startswith("engram_injections"):
            param.requires_grad = True
            trainable_count += param.numel()
        else:
            param.requires_grad = False
            frozen_count += param.numel()
    print(
        f"[capgap] Frozen {frozen_count:,} backbone params; "
        f"{trainable_count:,} engram params remain trainable."
    )
```

Add CLI flag alongside `--init-from`:

```python
    parser.add_argument(
        "--freeze-backbone", action="store_true",
        help=(
            "Freeze all non-engram parameters (η-B capacity-gap / ZEB-130). "
            "Only engram_injections.* params receive gradients and are "
            "added to the optimizer. Typically combined with --init-from "
            "and the tiny_engram_xattn_capgap config."
        ),
    )
```

Apply the freeze **after** `--init-from` weight loading and **before** optimizer construction. Find the optimizer construction site (search for `Muon(` or `optimizer = `):

```python
    # Apply capgap freezing BEFORE optimizer construction so only trainable
    # params are added to the optimizer (zero memory for frozen weights).
    if args.freeze_backbone:
        freeze_backbone_for_capgap(model)

    # ... existing optimizer construction ...
    # Ensure the optimizer only iterates over params with requires_grad=True.
    # Existing code likely has `list(model.parameters())` — change to:
    #   [p for p in model.parameters() if p.requires_grad]
```

If the optimizer construction uses `model.parameters()` directly, change to a filtered list comprehension. Inspect the surrounding code to confirm; the Muon optimizer may have special param-group handling (e.g., 2D matrix params vs. 1D) that needs to propagate through the filter.

- [ ] **Step 4: Run tests to verify pass**

Run: `cd training && python3 -m pytest tests/test_capacity_gap.py::TestFreezeBackbone -v`

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add training/ct87/train.py training/tests/test_capacity_gap.py
git commit -m "feat(training): add --freeze-backbone for η-B capgap mode (ZEB-130)"
```

---

### Task 6: Wire capgap config through training loop

**Files:**
- Modify: `training/ct87/train.py`
- Test: `training/tests/test_capacity_gap.py`

**Rationale:** When the selected config has non-empty `engram_inject_layers`, the training loop must construct a `GatedEngramInjection` per layer and call `attach_gated_engram_injections()` before loading the corpus table / dataloader setup.

- [ ] **Step 1: Write failing test**

Add to `training/tests/test_capacity_gap.py`:

```python
class TestCapgapEndToEndWiring:
    """Smoke-level integration: capgap preset is registered in train.py."""

    def test_capgap_preset_in_cli_list(self):
        """The capgap preset name is registered in the config dispatch."""
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "-m", "ct87.train", "--config", "invalid_name"],
            capture_output=True, text=True, timeout=30,
        )
        # Assert the capgap preset appears in the valid-config list error output.
        assert "tiny_engram_xattn_capgap" in (result.stderr + result.stdout)
```

- [ ] **Step 2: Run test to verify failure**

Run: `cd training && python3 -m pytest tests/test_capacity_gap.py::TestCapgapEndToEndWiring -v`

Expected: fails — capgap preset not registered in train.py's config dispatch.

- [ ] **Step 3: Wire capgap preset in train.py config dispatch**

Modify `training/ct87/train.py`. Find the config preset dispatch (search for `tiny_engram_xattn_consol_online` or similar). Add the new preset:

```python
    elif args.config == "tiny_engram_xattn_capgap":
        config = HarmonyModelConfig.tiny_engram_xattn_capgap()
```

Add the gated-engram injection construction + attach. Find where the existing xattn attach happens (search for `attach_engram_xattn`) and add the new branch right next to it:

```python
    # η-B multi-layer gated injection (ZEB-130).
    if config.engram_inject_layers:
        from ct87.engram import EngramCrossAttention, GatedEngramInjection
        injections: dict[int, GatedEngramInjection] = {}
        for layer_idx in config.engram_inject_layers:
            xattn_module = EngramCrossAttention(
                hidden_dim=config.hidden_dim,
                engram_dim=config.engram_dim,
                num_heads=config.num_query_heads,
                num_entries=corpus_table.num_entries,
                top_k=args.xattn_top_k if hasattr(args, "xattn_top_k") else 4,
            )
            injections[layer_idx] = GatedEngramInjection(
                xattn_module, alpha_init=config.engram_gate_init,
            )
        model.attach_gated_engram_injections(injections)
        # Move attached modules to device (parent .to() won't pick them up if
        # attach happens after the top-level .to()).
        model.engram_injections.to(device)
        print(
            f"[capgap] Attached GatedEngramInjection at layers "
            f"{list(config.engram_inject_layers)} with alpha_init="
            f"{config.engram_gate_init}"
        )
```

**Check** whether the corpus table / xattn construction depends on CLI args already used by the legacy xattn preset (e.g., `--xattn-top-k`, corpus table arg). If so, reuse them. If there's a gap (e.g., no top-k CLI), add a defaulted arg:

```python
    parser.add_argument(
        "--xattn-top-k", type=int, default=4,
        help="top-k for cross-attention retrieval (shared by δ and η-B paths).",
    )
```

- [ ] **Step 4: Run test to verify pass**

Run: `cd training && python3 -m pytest tests/test_capacity_gap.py::TestCapgapEndToEndWiring -v`

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add training/ct87/train.py training/tests/test_capacity_gap.py
git commit -m "feat(training): wire capgap preset + multi-layer attach in train.py (ZEB-130)"
```

---

### Task 7: End-to-end smoke test + zero-injection-eval compatibility

**Files:**
- Test: `training/tests/test_capacity_gap.py`
- Possible touch-up: `training/ct87/train.py` (zero-injection-eval path)

**Rationale:** Final integration gate. Tests that:
1. A minimal run with synthetic data actually trains (loss moves)
2. Backbone params don't change during training
3. Engram (gate + xattn) params do change
4. `--zero-injection-eval` works on a capgap checkpoint

- [ ] **Step 1: Write the integration test**

Add to `training/tests/test_capacity_gap.py`:

```python
class TestCapgapSmokeIntegration:
    """End-to-end: init from source checkpoint, train N steps frozen, eval."""

    def test_frozen_backbone_params_unchanged_after_training(self, tmp_path):
        """After N training steps with --freeze-backbone, backbone weights must be unchanged."""
        import copy
        import subprocess
        import sys

        # Step A: build a source checkpoint with random weights (substitute for β).
        src_ckpt = tmp_path / "src" / "checkpoint.pt"
        src_ckpt.parent.mkdir()
        src_config = HarmonyModelConfig.tiny()
        torch.manual_seed(42)
        src_model = HarmonyModel(src_config)
        torch.save(
            {
                "model_state_dict": src_model.state_dict(),
                "optimizer_state_dict": {},
                "step": 1000,
                "rng_state": torch.get_rng_state(),
                "config": src_config,
            },
            src_ckpt,
        )

        # Snapshot the source state dict for later comparison.
        src_state = copy.deepcopy(src_model.state_dict())

        # Step B: run train.py with --init-from + --freeze-backbone + capgap preset.
        out_dir = tmp_path / "run"
        result = subprocess.run(
            [
                sys.executable, "-m", "ct87.train",
                "--init-from", str(src_ckpt),
                "--freeze-backbone",
                "--config", "tiny_engram_xattn_capgap",
                "--dataset", "synthetic",
                "--output-dir", str(out_dir),
                "--steps", "3",
                "--batch-size", "2",
                "--seq-len", "32",
            ],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, (
            f"train.py failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

        # Step C: load the output checkpoint and confirm backbone is bitwise identical.
        out_ckpt_path = out_dir / "checkpoint.pt"
        assert out_ckpt_path.exists(), "checkpoint.pt not written after training"
        out_ckpt = torch.load(out_ckpt_path, map_location="cpu", weights_only=False)
        out_state = out_ckpt["model_state_dict"]
        for name, src_tensor in src_state.items():
            out_tensor = out_state[name]
            assert torch.equal(src_tensor, out_tensor), (
                f"Backbone param '{name}' changed despite --freeze-backbone"
            )

        # Step D: confirm engram_injections params DO exist in the output.
        injection_param_names = [
            n for n in out_state.keys() if n.startswith("engram_injections")
        ]
        assert len(injection_param_names) > 0, (
            "No engram_injections params saved — check attach wiring"
        )

    def test_zero_injection_eval_works_on_capgap_checkpoint(self, tmp_path):
        """After a capgap run, --zero-injection-eval must produce a sensible delta."""
        import subprocess
        import sys

        src_ckpt = tmp_path / "src" / "checkpoint.pt"
        src_ckpt.parent.mkdir()
        src_config = HarmonyModelConfig.tiny()
        torch.manual_seed(7)
        src_model = HarmonyModel(src_config)
        torch.save(
            {
                "model_state_dict": src_model.state_dict(),
                "optimizer_state_dict": {},
                "step": 100,
                "rng_state": torch.get_rng_state(),
                "config": src_config,
            },
            src_ckpt,
        )
        run_dir = tmp_path / "run"
        r = subprocess.run(
            [
                sys.executable, "-m", "ct87.train",
                "--init-from", str(src_ckpt),
                "--freeze-backbone",
                "--config", "tiny_engram_xattn_capgap",
                "--dataset", "synthetic",
                "--output-dir", str(run_dir),
                "--steps", "5",
                "--batch-size", "2",
                "--seq-len", "32",
            ],
            capture_output=True, text=True, timeout=180,
        )
        assert r.returncode == 0, f"train failed: {r.stderr}"

        eval_dir = tmp_path / "eval"
        r2 = subprocess.run(
            [
                sys.executable, "-m", "ct87.train",
                "--resume-from", str(run_dir / "checkpoint.pt"),
                "--zero-injection-eval",
                "--config", "tiny_engram_xattn_capgap",
                "--dataset", "synthetic",
                "--output-dir", str(eval_dir),
                "--batch-size", "2",
                "--seq-len", "32",
            ],
            capture_output=True, text=True, timeout=120,
        )
        assert r2.returncode == 0, f"eval failed: {r2.stderr}"
        assert "delta_removal" in r2.stdout.lower() or "delta" in r2.stdout.lower(), (
            f"--zero-injection-eval did not report delta:\n{r2.stdout}"
        )
```

- [ ] **Step 2: Run the integration test**

Run: `cd training && python3 -m pytest tests/test_capacity_gap.py::TestCapgapSmokeIntegration -v --tb=long`

Expected: first run likely reveals small integration bugs (e.g., `--zero-injection-eval` may not handle the capgap multi-layer path; if so, the fix is to confirm that setting `model.engram_inject_mult = 0.0` correctly zeroes all multi-layer injections — already handled by Task 3's forward changes).

- [ ] **Step 3: Fix any integration bugs surfaced**

Common issues to check:
- `--zero-injection-eval` early-exit must accept the capgap config (the existing path should work since it only zeros `engram_inject_mult`; verify no hidden check for `engram_xattn is not None`).
- Optimizer construction with all-frozen-plus-tiny-engram params must not produce empty param groups (Muon's 2D/1D separation may need to gracefully handle an empty 1D group or an empty 2D group).
- CSV header generalization from PR #246 already handles arbitrary trailing columns, but verify no capgap-specific CSV fields are introduced here.

Iterate: edit train.py, re-run test until it passes.

- [ ] **Step 4: Re-run the full test suite to confirm no regressions**

Run: `cd training && python3 -m pytest tests/ --tb=short -q 2>&1 | tail -30`

Expected: all tests pass (except the pre-existing Python-3.9 `zip(strict=True)` failure in `test_checkpoint.py::test_model_state_round_trip`, which is unrelated).

- [ ] **Step 5: Commit**

```bash
git add training/tests/test_capacity_gap.py training/ct87/train.py
git commit -m "test(training): end-to-end capgap smoke + zero-injection-eval (ZEB-130)"
```

---

## Post-Implementation: KRILE Run Instructions

Once all 7 tasks are complete and tests pass, KRILE runs the η-B experiment in two phases.

### Phase 1: Build a frozen-baseline checkpoint at seq_len=2048

Reuse the ζ-ctrl-2048 checkpoint (val_loss=4.5423 with injection, 4.5586 without injection). That's the "β-class" seq_len=2048 checkpoint we init from. No additional training needed for this phase.

Source checkpoint path on KRILE: `checkpoints/zeta_ctrl_2048/checkpoint.pt`.

### Phase 2: η-B capgap run

```bash
python -m ct87.train \
    --init-from checkpoints/zeta_ctrl_2048/checkpoint.pt \
    --freeze-backbone \
    --config tiny_engram_xattn_capgap \
    --data <fineweb-edu-poc-path> \
    --engram-xattn-table <oracle-table-path> \
    --output-dir checkpoints/eta_b_capgap \
    --steps 2000 \
    --batch-size <matching ζ-2048 batch> \
    --seq-len 2048 \
    --lr <matching or slightly higher than ζ-2048, since fewer params> \
    --checkpoint-interval 200 \
    --save-every 200
```

CLI flag notes:
- `--data` (not `--dataset`) points at the training data root
- `--engram-xattn-table` provides the oracle corpus table (shared with the ζ / δ path); without it, capgap falls back to a random placeholder (only meant for `--synthetic` smoke tests)
- `--save-every` controls both checkpointing and validation-loss computation (there is no separate `--eval-every`)

Wall time estimate: ~2-3 GPU-hours on the 4090 (200× fewer trainable params than a full ζ run; forward-pass cost dominates).

### Phase 3: Post-removal measurement

```bash
python -m ct87.train \
    --resume-from checkpoints/eta_b_capgap/checkpoint.pt \
    --zero-injection-eval \
    --config tiny_engram_xattn_capgap \
    --data <fineweb-edu-poc-path> \
    --engram-xattn-table <oracle-table-path> \
    --output-dir /tmp/eta_b_eval \
    --seq-len 2048
```

### Decision criteria

| Outcome | delta_removal | val_loss_with_injection | Interpretation | Next step |
|---|---|---|---|---|
| **H2 supported** | > +0.010 | < 4.538 (ζ-ctrl-2048 with-injection) | Capacity-gap hypothesis confirmed — well-gated injection on a pretrained backbone DOES help. The from-scratch co-training interference was the dominant failure mode. | Run η-A (add LMLM-style aux loss) to see if joint training with proper gating improves further. |
| **Null** | < +0.005 | ≈ ζ-ctrl-2048 baseline | 2603.11513 confirmed at 40M even with zero-init multi-layer gated injection. Capacity itself is the bottleneck. | Pivot to Option 2 (phase-transition study 40M→200M) or Option 3 (Gemma 4 base). |
| **Avoidance** | negative | > ζ-ctrl-2048 with-injection | Same failure mode as ζ-B: gate stayed near zero (tanh(α)≈0), injection learned nothing useful, and adding it at post-removal measurement time hurts. | Diagnose: check gate α values over training. If they stayed near 0, the signal never had enough gradient pressure — likely needs LMLM-style auxiliary utilization loss. |

### Primary diagnostic during training

Watch the `g2=...` and `g5=...` values appended to each step log line (one per injection layer, showing the current `tanh(α)` gate value in `[-1, +1]`). Typical trajectories:

- **Gates stay near 0 (g2, g5 ∈ [-0.02, +0.02] throughout):** strongest evidence the injection has nothing useful to contribute. Either retrieval is bad or the model can't use it at this scale.
- **Gates open to positive values (g2, g5 → 0.3 to 0.8+):** retrieval IS being used. If val_loss still doesn't drop, that points to a different bottleneck (e.g., signal is used but adds noise equivalent to what the backbone would have learned anyway).
- **Gates oscillate or stay negative:** diagnostic unclear; could be optimizer instability, bad initialization interaction, or adversarial signal. Reduce learning rate and re-run.

---

## Self-Review Checklist

After implementation, before calling the plan done:

- [ ] All 7 tasks committed separately, each with passing tests.
- [ ] Full test suite passes (modulo the pre-existing Python 3.9 `zip(strict=True)` incompatibility).
- [ ] Legacy ζ/ε/δ configs still work (regression check on `test_consolidation.py`, `test_model.py`).
- [ ] `--zero-injection-eval` works on both legacy xattn and new capgap checkpoints.
- [ ] No new `--flag` overlaps with existing ones.
- [ ] The multi-layer injection forward path preserves gradient-checkpointing compatibility (injections happen outside `_gradient_checkpoint(layer, h)` calls — verified by inspection of the forward loop).
- [ ] `.worktrees/zeb-130-eta-capgap/` is a fresh worktree off `origin/main` (verified at start).
