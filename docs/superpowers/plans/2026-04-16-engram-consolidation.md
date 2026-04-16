# ZEB-128 Engram Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add consolidation training modes (online and phased) that teach the model to predict the engram module's output from its own hidden states, with a zero-injection evaluation mode to measure table dependency.

**Architecture:** A lightweight 2-layer MLP decoder (`EngramConsolidationDecoder`) predicts the engram cross-attention residual from the model's hidden state at the injection layer. An MSE loss between the decoder's prediction and the detached engram output drives internalization. The injection multiplier (phased mode only) linearly anneals the engram signal from 1.0 to 0.0 during the consolidation phase.

**Tech Stack:** Python 3.10+, PyTorch, existing ct87 training infrastructure

**Spec:** `docs/superpowers/specs/2026-04-16-engram-consolidation-design.md`

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `training/ct87/engram.py` | Modify (append) | Add `EngramConsolidationDecoder` class |
| `training/ct87/model.py` | Modify | Add side-channel for xattn output capture; add 3 config presets; add `_last_xattn_output` field |
| `training/ct87/train.py` | Modify | Add consolidation CLI args, MSE loss computation, injection multiplier, `--zero-injection-eval` mode, CSV columns |
| `training/tests/test_consolidation.py` | Create | Tests for decoder, MSE loss, injection multiplier, zero-injection eval |

---

### Task 1: EngramConsolidationDecoder Module

**Files:**
- Test: `training/tests/test_consolidation.py`
- Modify: `training/ct87/engram.py`

- [ ] **Step 1: Write the failing tests**

Create `training/tests/test_consolidation.py`:

```python
"""Tests for ZEB-128 engram consolidation."""
import torch
import pytest

from ct87.engram import EngramConsolidationDecoder


class TestEngramConsolidationDecoder:

    def test_output_shape_matches_input(self):
        decoder = EngramConsolidationDecoder(hidden_dim=32)
        x = torch.randn(2, 16, 32)
        out = decoder(x)
        assert out.shape == (2, 16, 32)

    def test_xavier_init_weights_are_nonzero(self):
        decoder = EngramConsolidationDecoder(hidden_dim=32)
        assert decoder.net[0].weight.abs().sum() > 0
        assert decoder.net[2].weight.abs().sum() > 0

    def test_bias_initialized_to_zero(self):
        decoder = EngramConsolidationDecoder(hidden_dim=32)
        assert torch.allclose(decoder.net[0].bias, torch.zeros(32))
        assert torch.allclose(decoder.net[2].bias, torch.zeros(32))

    def test_gradient_flows_to_input(self):
        decoder = EngramConsolidationDecoder(hidden_dim=32)
        x = torch.randn(1, 8, 32, requires_grad=True)
        out = decoder(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_nonlinear_mapping(self):
        decoder = EngramConsolidationDecoder(hidden_dim=32)
        x1 = torch.randn(1, 4, 32)
        x2 = x1 * 2.0
        out1 = decoder(x1)
        out2 = decoder(x2)
        # GELU makes this nonlinear — outputs should NOT be 2x
        assert not torch.allclose(out2, out1 * 2.0, atol=1e-4)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd training && python3 -m pytest tests/test_consolidation.py -v`
Expected: FAIL with `ImportError: cannot import name 'EngramConsolidationDecoder'`

- [ ] **Step 3: Implement EngramConsolidationDecoder**

Add to the end of `training/ct87/engram.py` (after the `EngramCrossAttention` class, around line 1054):

```python
class EngramConsolidationDecoder(nn.Module):
    """Lightweight decoder for engram consolidation (ZEB-128).

    Predicts the engram module's residual output from the model's own
    hidden state. Used as an auxiliary MSE training target to force
    internalization of the engram signal into parametric knowledge.
    Discarded after training — never saved or shipped.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        nn.init.xavier_uniform_(self.net[0].weight)
        nn.init.zeros_(self.net[0].bias)
        nn.init.xavier_uniform_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_state)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd training && python3 -m pytest tests/test_consolidation.py -v`
Expected: All 5 PASS

- [ ] **Step 5: Commit**

```bash
git add training/ct87/engram.py training/tests/test_consolidation.py
git commit -m "feat: EngramConsolidationDecoder for ZEB-128 consolidation"
```

---

### Task 2: Cross-Attention Output Side-Channel

The model's forward pass currently calls `h = h + self.engram_xattn(h)` and discards the raw engram output. We need to capture it for the MSE target, similar to how `_last_ann_gate` captures the ANN gate.

**Files:**
- Test: `training/tests/test_consolidation.py`
- Modify: `training/ct87/model.py`

- [ ] **Step 1: Write the failing test**

Append to `training/tests/test_consolidation.py`:

```python
from ct87.model import HarmonyModel, HarmonyModelConfig
from ct87.engram import EngramCrossAttention


def _tiny_config() -> HarmonyModelConfig:
    """Shared tiny config for consolidation tests."""
    return HarmonyModelConfig(
        num_layers=4, hidden_dim=32, num_query_heads=4, num_kv_heads=2,
        head_dim=8, ffn_dim=64, vocab_size=128, max_seq_len=64,
        rope_theta=10000.0, rms_norm_eps=1e-6, layers_per_block=2,
        engram_injection_layer=1, engram_dim=16, tie_embeddings=True,
    )


def _fake_table(total_entries: int, engram_dim: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    return torch.randn(total_entries, engram_dim, generator=g)


class TestXattnOutputCapture:

    def test_last_xattn_output_captured_after_forward(self):
        config = _tiny_config()
        config.use_xattn_engram = True
        model = HarmonyModel(config)
        table = _fake_table(100, config.engram_dim)
        xattn = EngramCrossAttention(config, table)
        model.attach_engram_xattn(xattn)

        x = torch.randint(0, config.vocab_size, (1, 16))
        model(x)

        assert model._last_xattn_output is not None
        assert model._last_xattn_output.shape == (1, 16, config.hidden_dim)

    def test_last_xattn_output_is_none_without_xattn(self):
        config = _tiny_config()
        model = HarmonyModel(config)
        x = torch.randint(0, config.vocab_size, (1, 16))
        model(x)
        assert model._last_xattn_output is None

    def test_last_xattn_output_reset_each_forward(self):
        config = _tiny_config()
        config.use_xattn_engram = True
        model = HarmonyModel(config)
        table = _fake_table(100, config.engram_dim)
        xattn = EngramCrossAttention(config, table)
        model.attach_engram_xattn(xattn)

        x = torch.randint(0, config.vocab_size, (1, 16))
        model(x)
        first_output = model._last_xattn_output.clone()

        x2 = torch.randint(0, config.vocab_size, (1, 8))
        model(x2)
        # Shape should change with different input length
        assert model._last_xattn_output.shape == (1, 8, config.hidden_dim)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd training && python3 -m pytest tests/test_consolidation.py::TestXattnOutputCapture -v`
Expected: FAIL with `AttributeError: 'HarmonyModel' object has no attribute '_last_xattn_output'`

- [ ] **Step 3: Add _last_xattn_output side-channel to HarmonyModel**

In `training/ct87/model.py`, add two fields in `__init__` (after `_last_ann_gate` on line 468):

```python
        self._last_xattn_output: torch.Tensor | None = None
        self._last_pre_injection_hidden: torch.Tensor | None = None
```

In the `forward()` method, reset both at the top (after `self._last_ann_gate = None` on line 601):

```python
        self._last_xattn_output = None
        self._last_pre_injection_hidden = None
```

Change the engram xattn injection (line 619-620) from:

```python
                if self.engram_xattn is not None:
                    h = h + self.engram_xattn(h)
```

to:

```python
                if self.engram_xattn is not None:
                    self._last_pre_injection_hidden = h.detach()
                    xattn_out = self.engram_xattn(h)
                    self._last_xattn_output = xattn_out
                    h = h + xattn_out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd training && python3 -m pytest tests/test_consolidation.py -v`
Expected: All 8 PASS

- [ ] **Step 5: Run existing tests to check for regressions**

Run: `cd training && python3 -m pytest tests/test_engram.py tests/test_checkpoint.py -v`
Expected: All pass (except the known Python 3.9 `zip(strict=True)` failure)

- [ ] **Step 6: Commit**

```bash
git add training/ct87/model.py training/tests/test_consolidation.py
git commit -m "feat: capture xattn output in side-channel for consolidation MSE target"
```

---

### Task 3: Injection Multiplier

The phased consolidation mode (zeta-B) needs to linearly anneal the injection strength from 1.0 to 0.0 during the consolidation phase. This is applied outside the engram module, in the model's forward pass.

**Files:**
- Test: `training/tests/test_consolidation.py`
- Modify: `training/ct87/model.py`

- [ ] **Step 1: Write the failing test**

Append to `training/tests/test_consolidation.py`:

```python
class TestInjectionMultiplier:

    def test_multiplier_scales_xattn_output(self):
        config = _tiny_config()
        config.use_xattn_engram = True
        model = HarmonyModel(config)
        table = _fake_table(100, config.engram_dim)
        xattn = EngramCrossAttention(config, table)
        model.attach_engram_xattn(xattn)

        x = torch.randint(0, config.vocab_size, (1, 16))

        # Full injection
        model.engram_inject_mult = 1.0
        model(x)
        full_output = model._last_xattn_output.clone()

        # Half injection
        model.engram_inject_mult = 0.5
        model(x)
        half_output = model._last_xattn_output.clone()

        # The captured output should be the same (pre-scaling)
        # but the logits should differ because injection is scaled
        assert torch.allclose(full_output, half_output, atol=1e-5)

    def test_zero_multiplier_zeroes_injection(self):
        config = _tiny_config()
        config.use_xattn_engram = True
        model = HarmonyModel(config)
        table = _fake_table(100, config.engram_dim)
        xattn = EngramCrossAttention(config, table)
        model.attach_engram_xattn(xattn)

        x = torch.randint(0, config.vocab_size, (1, 16))

        model.engram_inject_mult = 0.0
        logits_zero = model(x)

        # Compare to no-xattn model
        config2 = _tiny_config()
        model2 = HarmonyModel(config2)
        model2.load_state_dict(model.state_dict(), strict=False)
        logits_no_engram = model2(x)

        assert torch.allclose(logits_zero, logits_no_engram, atol=1e-5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd training && python3 -m pytest tests/test_consolidation.py::TestInjectionMultiplier -v`
Expected: FAIL with `AttributeError: 'HarmonyModel' has no attribute 'engram_inject_mult'`

- [ ] **Step 3: Add engram_inject_mult to HarmonyModel**

In `training/ct87/model.py`, add the field in `__init__` (after `_last_xattn_output`):

```python
        self.engram_inject_mult: float = 1.0
```

Change the engram xattn injection in `forward()` from (the version modified in Task 2):

```python
                if self.engram_xattn is not None:
                    self._last_pre_injection_hidden = h.detach()
                    xattn_out = self.engram_xattn(h)
                    self._last_xattn_output = xattn_out
                    h = h + xattn_out
```

to:

```python
                if self.engram_xattn is not None:
                    self._last_pre_injection_hidden = h.detach()
                    xattn_out = self.engram_xattn(h)
                    self._last_xattn_output = xattn_out
                    h = h + xattn_out * self.engram_inject_mult
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd training && python3 -m pytest tests/test_consolidation.py -v`
Expected: All 11 PASS

- [ ] **Step 5: Run existing tests to check for regressions**

Run: `cd training && python3 -m pytest tests/test_engram.py -v`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add training/ct87/model.py training/tests/test_consolidation.py
git commit -m "feat: injection multiplier for phased consolidation gate annealing"
```

---

### Task 4: Config Presets

**Files:**
- Test: `training/tests/test_consolidation.py`
- Modify: `training/ct87/model.py`

- [ ] **Step 1: Write the failing tests**

Append to `training/tests/test_consolidation.py`:

```python
class TestConsolidationConfigPresets:

    def test_consol_online_preset(self):
        config = HarmonyModelConfig.tiny_engram_xattn_consol_online()
        assert config.use_xattn_engram is True
        assert config.use_head_gates is False
        assert config.use_ann_engram is False

    def test_consol_phased_preset(self):
        config = HarmonyModelConfig.tiny_engram_xattn_consol_phased()
        assert config.use_xattn_engram is True
        assert config.use_head_gates is False
        assert config.use_ann_engram is False

    def test_ctrl_preset(self):
        config = HarmonyModelConfig.tiny_engram_xattn_ctrl()
        assert config.use_xattn_engram is True
        assert config.use_head_gates is False
        assert config.use_ann_engram is False

    def test_all_three_presets_equivalent(self):
        """All three zeta presets have the same model architecture.

        The difference is in CLI args (consolidation mode), not model config.
        """
        a = HarmonyModelConfig.tiny_engram_xattn_consol_online()
        b = HarmonyModelConfig.tiny_engram_xattn_consol_phased()
        c = HarmonyModelConfig.tiny_engram_xattn_ctrl()
        for field in ["num_layers", "hidden_dim", "num_query_heads",
                       "head_dim", "ffn_dim", "vocab_size", "engram_dim",
                       "engram_injection_layer", "use_xattn_engram",
                       "use_head_gates", "use_ann_engram"]:
            assert getattr(a, field) == getattr(b, field) == getattr(c, field), \
                f"Mismatch in {field}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd training && python3 -m pytest tests/test_consolidation.py::TestConsolidationConfigPresets -v`
Expected: FAIL with `AttributeError: type object 'HarmonyModelConfig' has no attribute 'tiny_engram_xattn_consol_online'`

- [ ] **Step 3: Add config presets**

In `training/ct87/model.py`, after `tiny_engram_ann_routed()` (after line 237), add:

```python
    @staticmethod
    def tiny_engram_xattn_consol_online() -> HarmonyModelConfig:
        """Zeta-A: cross-attention + online consolidation (ZEB-128)."""
        base = HarmonyModelConfig.tiny()
        base.use_xattn_engram = True
        return base

    @staticmethod
    def tiny_engram_xattn_consol_phased() -> HarmonyModelConfig:
        """Zeta-B: cross-attention + phased consolidation (ZEB-128)."""
        base = HarmonyModelConfig.tiny()
        base.use_xattn_engram = True
        return base

    @staticmethod
    def tiny_engram_xattn_ctrl() -> HarmonyModelConfig:
        """Zeta-ctrl: cross-attention control, no consolidation (ZEB-128)."""
        base = HarmonyModelConfig.tiny()
        base.use_xattn_engram = True
        return base
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd training && python3 -m pytest tests/test_consolidation.py -v`
Expected: All 15 PASS

- [ ] **Step 5: Commit**

```bash
git add training/ct87/model.py training/tests/test_consolidation.py
git commit -m "feat: zeta config presets for consolidation experiments"
```

---

### Task 5: CLI Arguments

**Files:**
- Modify: `training/ct87/train.py`

- [ ] **Step 1: Add consolidation CLI arguments**

In `training/ct87/train.py`, after the `--qat-start-pct` argument (line 499), before `args = parser.parse_args()` (line 500), add:

```python
    # ---- ZEB-128: Engram consolidation ----
    parser.add_argument(
        "--consolidation-mode", choices=["none", "online", "phased"],
        default="none",
        help="Consolidation strategy: 'none' (control), 'online' (MSE loss "
             "throughout), 'phased' (MSE loss after --consolidation-start-step)",
    )
    parser.add_argument(
        "--consolidation-lambda", type=float, default=0.1,
        help="Weight for consolidation MSE loss (default: 0.1)",
    )
    parser.add_argument(
        "--consolidation-start-step", type=int, default=0,
        help="Step at which consolidation MSE loss activates (default: 0 for "
             "online, typically 7000 for phased)",
    )
    parser.add_argument(
        "--consolidation-anneal", action="store_true",
        help="Linearly anneal injection multiplier from 1.0 to 0.0 during "
             "consolidation phase (only meaningful with --consolidation-mode=phased)",
    )
    parser.add_argument(
        "--zero-injection-eval", action="store_true",
        help="Load checkpoint, zero engram injection, run validation, and exit. "
             "Used for post-removal measurement in consolidation experiments.",
    )
```

- [ ] **Step 2: Add config preset wiring**

In the config selection block (around lines 596-609), add the three new presets. Change:

```python
    elif args.config == "tiny_engram_xattn_routed":
        config = HarmonyModelConfig.tiny_engram_xattn_routed()
    else:
        config = HarmonyModelConfig.target()
```

to:

```python
    elif args.config == "tiny_engram_xattn_routed":
        config = HarmonyModelConfig.tiny_engram_xattn_routed()
    elif args.config == "tiny_engram_xattn_consol_online":
        config = HarmonyModelConfig.tiny_engram_xattn_consol_online()
    elif args.config == "tiny_engram_xattn_consol_phased":
        config = HarmonyModelConfig.tiny_engram_xattn_consol_phased()
    elif args.config == "tiny_engram_xattn_ctrl":
        config = HarmonyModelConfig.tiny_engram_xattn_ctrl()
    else:
        config = HarmonyModelConfig.target()
```

Also update the `--config` choices list (line 319-322) to include the new presets:

```python
        choices=[
            "tiny", "target", "tiny_ffn_expanded",
            "tiny_engram_ann", "tiny_engram_ann_routed",
            "tiny_engram_xattn", "tiny_engram_xattn_routed",
            "tiny_engram_xattn_consol_online",
            "tiny_engram_xattn_consol_phased",
            "tiny_engram_xattn_ctrl",
        ],
```

- [ ] **Step 3: Add consolidation arg validation**

After the existing delta arg validation block (after line 669), add:

```python
    # ZEB-128 consolidation arg validation
    if args.consolidation_mode != "none":
        if not config.use_xattn_engram:
            print(
                "Error: --consolidation-mode requires a cross-attention engram "
                "config (e.g. tiny_engram_xattn_consol_online)",
                file=sys.stderr,
            )
            sys.exit(1)
        if args.consolidation_lambda <= 0 or not math.isfinite(args.consolidation_lambda):
            print(
                "Error: --consolidation-lambda must be finite and > 0",
                file=sys.stderr,
            )
            sys.exit(1)
        if args.consolidation_start_step < 0:
            print(
                "Error: --consolidation-start-step must be >= 0",
                file=sys.stderr,
            )
            sys.exit(1)
    if args.consolidation_anneal and args.consolidation_mode != "phased":
        print(
            "Error: --consolidation-anneal only applies to --consolidation-mode=phased",
            file=sys.stderr,
        )
        sys.exit(1)
```

- [ ] **Step 4: Verify CLI parsing works**

Run: `cd training && python3 -m ct87.train --help 2>&1 | grep -A2 consolidation`
Expected: Should show the new consolidation arguments

- [ ] **Step 5: Commit**

```bash
git add training/ct87/train.py
git commit -m "feat: consolidation CLI arguments and config wiring"
```

---

### Task 6: Zero-Injection Eval Mode

**Files:**
- Test: `training/tests/test_consolidation.py`
- Modify: `training/ct87/train.py`

- [ ] **Step 1: Write the failing test**

Append to `training/tests/test_consolidation.py`:

```python
import os
import tempfile

from ct87.train import save_resumable_checkpoint


class TestZeroInjectionEval:

    def test_zero_injection_produces_different_loss(self):
        """Verify that zeroing injection changes the model's output."""
        config = _tiny_config()
        config.use_xattn_engram = True
        model = HarmonyModel(config)
        table = _fake_table(100, config.engram_dim)
        xattn = EngramCrossAttention(config, table)
        model.attach_engram_xattn(xattn)

        x = torch.randint(0, config.vocab_size, (1, 16))

        # Full injection
        model.engram_inject_mult = 1.0
        logits_full = model(x).detach()

        # Zero injection
        model.engram_inject_mult = 0.0
        logits_zero = model(x).detach()

        # They should differ (engram contributes something)
        # At initialization the xattn o_proj is zero so they'll match;
        # train for a step to break symmetry
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model.engram_inject_mult = 1.0
        logits = model(x)
        loss = logits.sum()
        loss.backward()
        optimizer.step()

        model.engram_inject_mult = 1.0
        logits_full2 = model(x).detach()
        model.engram_inject_mult = 0.0
        logits_zero2 = model(x).detach()

        # After training, the outputs should differ
        assert not torch.allclose(logits_full2, logits_zero2, atol=1e-5)
```

- [ ] **Step 2: Run test to verify it passes**

Run: `cd training && python3 -m pytest tests/test_consolidation.py::TestZeroInjectionEval -v`
Expected: PASS (this test validates existing injection multiplier behavior)

- [ ] **Step 3: Implement --zero-injection-eval mode**

In `training/ct87/train.py`, in the `main()` function, after the model is fully constructed and the engram xattn module is attached (after line 794), but before the training loop setup, add the zero-injection eval early-exit path:

```python
    # ZEB-128: Zero-injection evaluation mode
    if args.zero_injection_eval:
        if args.resume_from is None:
            print("Error: --zero-injection-eval requires --resume-from", file=sys.stderr)
            sys.exit(1)
        if val_loader is None:
            print("Error: --zero-injection-eval requires --val-data", file=sys.stderr)
            sys.exit(1)
        # Load checkpoint
        ckpt = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        required_keys = {"model_state_dict", "step"}
        if not required_keys.issubset(ckpt.keys()):
            print(
                f"Error: checkpoint missing required keys: "
                f"{required_keys - ckpt.keys()}",
                file=sys.stderr,
            )
            sys.exit(1)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        step = ckpt["step"]
        # Measure with injection active
        val_pre = compute_validation_loss(
            model, val_loader, config.vocab_size, device,
            amp_dtype=amp_dtype, engram_table=engram_table,
            latent_projection=latent_projection,
        )
        # Measure with injection zeroed
        model.engram_inject_mult = 0.0
        val_post = compute_validation_loss(
            model, val_loader, config.vocab_size, device,
            amp_dtype=amp_dtype, engram_table=engram_table,
            latent_projection=latent_projection,
        )
        delta_removal = val_post - val_pre
        print(f"Step: {step}")
        print(f"Val loss (with injection):    {val_pre:.6f}")
        print(f"Val loss (without injection): {val_post:.6f}")
        print(f"Delta (removal cost):         {delta_removal:+.6f}")
        return
```

Note: This block must go after `val_loader` is set up (around line 722) and after the engram xattn module is attached (around line 786). Find the exact insertion point between engram attachment and the resume-from checkpoint loading block.

- [ ] **Step 4: Verify the help text shows the new flag**

Run: `cd training && python3 -m ct87.train --help 2>&1 | grep zero-injection`
Expected: Shows `--zero-injection-eval`

- [ ] **Step 5: Commit**

```bash
git add training/ct87/train.py training/tests/test_consolidation.py
git commit -m "feat: --zero-injection-eval mode for post-removal measurement"
```

---

### Task 7: Consolidation Training Loop Integration

This is the core task — adding the MSE loss computation, injection multiplier annealing, and decoder construction to the training loop.

**Files:**
- Modify: `training/ct87/train.py`

- [ ] **Step 1: Construct the consolidation decoder**

In `training/ct87/train.py`, after the engram xattn module is attached (after line 786), add decoder construction:

```python
    # ZEB-128: Consolidation decoder
    consol_decoder = None
    if args.consolidation_mode != "none" and model.engram_xattn is not None:
        from ct87.engram import EngramConsolidationDecoder
        consol_decoder = EngramConsolidationDecoder(config.hidden_dim).to(device)
        print(
            f"Consolidation decoder attached: mode={args.consolidation_mode}, "
            f"lambda={args.consolidation_lambda}, "
            f"start_step={args.consolidation_start_step}"
        )
```

- [ ] **Step 2: Include decoder params in optimizer and grad clipping**

The consolidation decoder's parameters need to be included in the optimizer. Find where the optimizer is constructed (search for `partition_params` or `Muon` construction) and add the decoder's parameters.

After the optimizer is created, add:

```python
    if consol_decoder is not None:
        # Add decoder params to optimizer as a standard AdamW group
        optimizer.add_param_group({
            "params": list(consol_decoder.parameters()),
            "lr": args.lr,
        })
```

In the gradient clipping section (around line 1286-1297), add the decoder:

```python
                if consol_decoder is not None:
                    all_params.extend(consol_decoder.parameters())
```

Add this after the `latent_projection` params line (line 1294).

- [ ] **Step 3: Add MSE loss computation inside the training loop**

After the gate-entropy regularization block (after line 1279), add the consolidation MSE loss. The pre-injection hidden state and xattn output are already captured in model side-channels (set up in Task 2):

```python
                    # ZEB-128: Consolidation MSE loss
                    if (
                        consol_decoder is not None
                        and model._last_xattn_output is not None
                        and model._last_pre_injection_hidden is not None
                        and step >= args.consolidation_start_step
                    ):
                        consol_target = model._last_xattn_output.detach()
                        consol_input = model._last_pre_injection_hidden
                        consol_pred = consol_decoder(consol_input)
                        mse_loss = F.mse_loss(consol_pred, consol_target)
                        loss = loss + args.consolidation_lambda * mse_loss
                        accum_mse_loss += mse_loss.item()
```

- [ ] **Step 4: Add injection multiplier annealing**

In the training loop, before the forward pass (around where `num_thoughts` is set, line 1080), add:

```python
            # ZEB-128: Update injection multiplier for phased annealing
            inject_mult = 1.0
            if args.consolidation_anneal and step >= args.consolidation_start_step:
                progress = (step - args.consolidation_start_step) / max(
                    1, args.steps - args.consolidation_start_step
                )
                inject_mult = max(0.0, 1.0 - progress)
            if model.engram_xattn is not None:
                model.engram_inject_mult = inject_mult
```

- [ ] **Step 5: Add accum_mse_loss accumulator**

In the per-step accumulator section (around lines 1074-1080), add:

```python
            accum_mse_loss = 0.0
```

- [ ] **Step 6: Verify it runs with --synthetic**

Run: `cd training && python3 -m ct87.train --config tiny_engram_xattn_consol_online --engram-xattn-table /dev/null --synthetic --steps 5 --consolidation-mode online --consolidation-lambda 0.1 2>&1 | head -20`

Note: This will fail because `/dev/null` is not a valid safetensors file. The point is to verify CLI parsing works. For a real test, you'll need a small test table.

- [ ] **Step 7: Commit**

```bash
git add training/ct87/model.py training/ct87/train.py
git commit -m "feat: consolidation MSE loss and injection multiplier annealing"
```

---

### Task 8: CSV Logging for Consolidation Metrics

**Files:**
- Modify: `training/ct87/train.py`

- [ ] **Step 1: Add new CSV columns to expected header**

In the CSV header definition (around line 982-988), add three new columns after the existing `hg_max` column:

```python
    expected_header = [
        "step", "loss", "uq_loss", "mtp_loss", "cl_loss",
        "ann_ent_loss", "ann_gate_mean", "ann_lambda_ent",
        "val_loss", "lr", "grad_norm", "num_thoughts", "dt_ms",
        "hg_0", "hg_1", "hg_2", "hg_3", "hg_4", "hg_5", "hg_6", "hg_7",
        "hg_std", "hg_min", "hg_max",
        "mse_loss", "consol_phase", "inject_mult",
    ]
```

- [ ] **Step 2: Add console logging for consolidation metrics**

In the per-step console logging block (around lines 1331-1371), add after the `hg_str` section:

```python
                consol_str = ""
                if consol_decoder is not None:
                    raw_mse = accum_mse_loss / args.grad_accum_steps
                    consol_active = 1 if step >= args.consolidation_start_step else 0
                    consol_str = (
                        f"  mse={raw_mse:.4f}"
                        f"  phase={consol_active}"
                        f"  inject={inject_mult:.3f}"
                    )
```

Add `{consol_str}` to the print statement on line 1369-1371.

- [ ] **Step 3: Add CSV row values for consolidation columns**

In the CSV writing block (around lines 1410-1459), after the `hg_cols` section, add:

```python
                mse_loss_str = ""
                consol_phase_str = ""
                inject_mult_str = ""
                if consol_decoder is not None:
                    mse_loss_str = f"{accum_mse_loss / args.grad_accum_steps:.6f}"
                    consol_phase_str = "1" if step >= args.consolidation_start_step else "0"
                    inject_mult_str = f"{inject_mult:.6f}"
```

Add these three values to the `csv_writer.writerow()` call:

```python
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
                ])
```

- [ ] **Step 4: Commit**

```bash
git add training/ct87/train.py
git commit -m "feat: CSV and console logging for consolidation metrics"
```

---

### Task 9: Integration Test with Synthetic Data

**Files:**
- Test: `training/tests/test_consolidation.py`

- [ ] **Step 1: Write integration tests**

Append to `training/tests/test_consolidation.py`:

```python
import torch.nn.functional as F


class TestConsolidationMSELoss:

    def test_mse_loss_computes_correctly(self):
        """Verify the MSE loss path from decoder to target."""
        decoder = EngramConsolidationDecoder(hidden_dim=32)
        # Simulate pre-injection hidden state
        h_pre = torch.randn(2, 8, 32)
        # Simulate xattn output (the target)
        xattn_out = torch.randn(2, 8, 32)

        prediction = decoder(h_pre)
        mse = F.mse_loss(prediction, xattn_out.detach())

        assert mse.item() > 0
        assert mse.requires_grad  # Gradient flows through decoder

    def test_mse_gradient_does_not_flow_to_target(self):
        """The MSE target must be detached — no gradient through engram module."""
        decoder = EngramConsolidationDecoder(hidden_dim=32)
        h_pre = torch.randn(2, 8, 32)
        xattn_out = torch.randn(2, 8, 32, requires_grad=True)

        prediction = decoder(h_pre)
        mse = F.mse_loss(prediction, xattn_out.detach())
        mse.backward()

        # xattn_out should have no gradient (was detached)
        assert xattn_out.grad is None


class TestInjectionMultiplierAnnealing:

    def test_linear_annealing_schedule(self):
        """Verify the injection multiplier follows a linear schedule."""
        start_step = 7000
        total_steps = 10000

        for step, expected in [(0, 1.0), (7000, 1.0), (8500, 0.5), (10000, 0.0)]:
            if step >= start_step:
                progress = (step - start_step) / max(1, total_steps - start_step)
                mult = max(0.0, 1.0 - progress)
            else:
                mult = 1.0
            assert abs(mult - expected) < 1e-6, f"Step {step}: {mult} != {expected}"
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cd training && python3 -m pytest tests/test_consolidation.py -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add training/tests/test_consolidation.py
git commit -m "test: integration tests for consolidation MSE loss and annealing"
```

---

### Task 10: End-to-End Smoke Test

Verify the full pipeline works with synthetic data for all three modes.

**Files:**
- Test: `training/tests/test_consolidation.py`

- [ ] **Step 1: Write end-to-end smoke test**

Append to `training/tests/test_consolidation.py`:

```python
import subprocess
import sys


class TestConsolidationSmoke:

    @pytest.fixture
    def fake_table_path(self, tmp_path):
        """Create a minimal safetensors table for testing."""
        from safetensors.torch import save_file
        table = torch.randn(100, 16)  # 100 entries, engram_dim=16
        path = str(tmp_path / "test_table.safetensors")
        save_file({"engram.weight": table}, path)
        return path

    def test_online_mode_runs(self, tmp_path, fake_table_path):
        result = subprocess.run(
            [
                sys.executable, "-m", "ct87.train",
                "--config", "tiny_engram_xattn_consol_online",
                "--engram-xattn-table", fake_table_path,
                "--synthetic", "--steps", "10",
                "--consolidation-mode", "online",
                "--consolidation-lambda", "0.1",
                "--output-dir", str(tmp_path / "online"),
            ],
            capture_output=True, text=True, cwd=str(tmp_path.parent),
            timeout=120,
        )
        assert result.returncode == 0, f"STDERR:\n{result.stderr}"

    def test_phased_mode_runs(self, tmp_path, fake_table_path):
        result = subprocess.run(
            [
                sys.executable, "-m", "ct87.train",
                "--config", "tiny_engram_xattn_consol_phased",
                "--engram-xattn-table", fake_table_path,
                "--synthetic", "--steps", "10",
                "--consolidation-mode", "phased",
                "--consolidation-start-step", "5",
                "--consolidation-anneal",
                "--consolidation-lambda", "0.1",
                "--output-dir", str(tmp_path / "phased"),
            ],
            capture_output=True, text=True, cwd=str(tmp_path.parent),
            timeout=120,
        )
        assert result.returncode == 0, f"STDERR:\n{result.stderr}"

    def test_ctrl_mode_runs(self, tmp_path, fake_table_path):
        result = subprocess.run(
            [
                sys.executable, "-m", "ct87.train",
                "--config", "tiny_engram_xattn_ctrl",
                "--engram-xattn-table", fake_table_path,
                "--synthetic", "--steps", "10",
                "--consolidation-mode", "none",
                "--output-dir", str(tmp_path / "ctrl"),
            ],
            capture_output=True, text=True, cwd=str(tmp_path.parent),
            timeout=120,
        )
        assert result.returncode == 0, f"STDERR:\n{result.stderr}"
```

- [ ] **Step 2: Run smoke tests**

Run: `cd training && python3 -m pytest tests/test_consolidation.py::TestConsolidationSmoke -v --timeout=120`
Expected: All 3 PASS

- [ ] **Step 3: Run full test suite**

Run: `cd training && python3 -m pytest tests/ -v`
Expected: All pass (except known Python 3.9 issue)

- [ ] **Step 4: Commit**

```bash
git add training/tests/test_consolidation.py
git commit -m "test: end-to-end smoke tests for all consolidation modes"
```

---

## Notes for Implementers

### Key Architecture Decision: Pre-Injection Hidden State

The MSE decoder needs the hidden state *before* engram injection as input. This is captured via `model._last_pre_injection_hidden` (set with `.detach()` in Task 2's forward pass changes). The `.detach()` is intentional — it means the MSE loss's gradient path is: `loss → decoder params` only. The MSE loss does NOT directly push gradients into the model's transformer layers. The model learns to internalize the engram signal indirectly: the LM loss is the only gradient path into the model, and as the injection multiplier anneals to 0, the model must produce good logits without the engram — which requires it to have internalized the signal parametrically.

If we later want the MSE loss to directly backprop into the transformer, remove the `.detach()` on `_last_pre_injection_hidden`. But that changes the experiment semantics.

### Existing Model Side-Channel Pattern

This follows the same pattern as `model._last_ann_gate` — a non-persistent field set during forward() and read by the training loop. The field is reset to None at the start of each forward call.

### CSV Header Migration

The three new columns (`mse_loss`, `consol_phase`, `inject_mult`) are appended after the existing `hg_max` column. The existing CSV migration logic in train.py handles row padding for legacy formats. New columns will be empty strings for non-consolidation runs, consistent with how `hg_*` columns are empty for non-head-gate runs.
