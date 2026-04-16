# Experiment Zeta: Engram Consolidation (ZEB-128)

**Date:** 2026-04-16
**Issue:** ZEB-128 (consolidation)
**Parent:** ZEB-102 (Engram table quality)
**Depends on:** ZEB-119 (oracle table), PR #244 (epsilon infrastructure)
**Status:** Design approved, awaiting epsilon completion

## Motivation

Five experiments (alpha/beta/gamma/delta/epsilon) all land in a 0.009-nat inert band around beta=3.648. Systematic elimination has ruled out:

- **Content quality (ZEB-129):** Oracle table (Mistral-7B teacher hidden states) barely beats hash content (+0.0005 nats). Content is not the bottleneck.
- **Per-head routing (ZEB-127):** Epsilon's 8 per-head gates moved in lockstep to ~0.68 with no differentiation. The signal's utility doesn't vary by head.
- **Cross-attention vs gated-residual:** Delta (3.6391) marginally beats gamma (3.6398). Mechanism matters slightly but isn't the breakthrough lever.

The remaining untested hypothesis is **consolidation** (ZEB-128): can the model internalize the engram signal into parametric knowledge, or is the injection fundamentally a forward-pass crutch?

Evidence supporting this hypothesis:
- The "diminishing marginal value" pattern: across all experiments, the engram gate surges early (0.61 in ZEB-119) then settles to equilibrium (0.50-0.52 or 0.68). The model finds the signal useful early but can't sustain the benefit.
- Bio-inspired analogy: the hippocampal-cortical axis uses offline replay (sharp-wave ripples) to consolidate episodic memory into parametric cortex. Our system has no analogue — the engram signal is injected but never internalized.

## Strategy: A/B Comparison of Consolidation Timing

Rather than picking one consolidation approach, we run a head-to-head comparison of two strategies that differ only in *when* consolidation happens:

- **zeta-A (online):** Concurrent MSE loss throughout training
- **zeta-B (phased):** Distinct "awake" and "sleep" phases

Plus a no-consolidation control (zeta-ctrl) to measure baseline table dependency.

## Architecture

### Consolidation Decoder

A lightweight two-layer MLP that learns to predict the engram module's residual contribution from the model's own hidden states:

```python
class EngramConsolidationDecoder(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Xavier uniform init on both layers
        nn.init.xavier_uniform_(self.net[0].weight)
        nn.init.zeros_(self.net[0].bias)
        nn.init.xavier_uniform_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_state)
```

**Input:** Hidden state `h` at `engram_injection_layer` (shape `[B, L, 512]`)

**Target:** The engram cross-attention module's output — `EngramCrossAttention.forward(h)` — detached. Gradients from MSE flow into the model's hidden states but NOT back through the engram module.

**Parameter cost:** ~525K params (1.3% of 39.6M base). Negligible memory/compute impact.

**Loss:** `loss_total = loss_lm + lambda_consol * MSE(decoder(h), engram_output.detach())`

**lambda_consol:** 0.1 (tunable via CLI). Large enough for meaningful gradient, small enough not to overwhelm LM objective.

### Injection Multiplier (zeta-B only)

During the phased consolidation's "sleep" phase, the engram module's output is multiplied by a linearly decaying factor before injection:

```python
if consolidation_anneal and step >= consolidation_start_step:
    progress = (step - consolidation_start_step) / (total_steps - consolidation_start_step)
    inject_mult = max(0.0, 1.0 - progress)
    engram_residual = engram_residual * inject_mult
```

The module still computes its full output (used as the MSE target), but the actual injection strength decays from 1.0 to 0.0 over the consolidation phase. The model is gradually weaned off the table while being trained to replicate its contribution.

### Table Removal Test

At step 10K, zero the injection (multiply engram output by 0) and run validation. The model architecture doesn't change — we just cut the signal. This isolates exactly one variable.

Implemented as a `--zero-injection-eval` CLI flag: loads checkpoint, zeros injection, runs validation, prints val_loss, exits.

## Run Configurations

### zeta-A: Online Consolidation

```
Steps 0-10K:  LM loss + lambda_consol * MSE loss
              Engram injection active throughout
              Decoder active throughout
Step 10K:     Zero injection, measure val_loss
```

Config preset: `tiny_engram_xattn_consol_online`

### zeta-B: Phased Consolidation

```
Phase 1 (steps 0-7K):    LM loss only, engram injection active
                          No MSE loss, decoder inactive

Phase 2 (steps 7K-10K):  LM loss + lambda_consol * MSE loss
                          Decoder activated
                          Injection multiplier: linear 1.0 -> 0.0

Step 10K:                 Injection already at 0, measure val_loss
```

The decoder is constructed at model init for both zeta-A and zeta-B, but in zeta-B the MSE loss term is gated on `step >= consolidation_start_step`. Before that step, the decoder exists but receives no gradients and has no effect on training.

Config preset: `tiny_engram_xattn_consol_phased`

### zeta-ctrl: No-Consolidation Control

```
Steps 0-10K:  LM loss only, engram injection active
              No MSE loss, no decoder
Step 10K:     Zero injection, measure val_loss
```

Config preset: `tiny_engram_xattn_ctrl`

### Common Configuration

All three runs share:
- Base: delta cross-attention path (EngramCrossAttention) with oracle table
- NO per-head gates (epsilon showed they don't differentiate; dropping reduces confounds)
- Tiny config: 8-layer, 512-hidden, 8 heads, 39.6M params
- FineWeb-Edu-POC dataset, seq_len=2048, bf16, Muon optimizer
- 10K total steps, checkpoint every 1000
- Same oracle table: `oracle_mistral7b_10k.safetensors`

## CLI Arguments

New arguments for consolidation:

| Argument | Type | Default | Description |
|---|---|---|---|
| `--consolidation-mode` | `none`/`online`/`phased` | `none` | Consolidation strategy |
| `--consolidation-lambda` | float | 0.1 | MSE loss weight |
| `--consolidation-start-step` | int | 0 | When MSE loss activates (auto-set by mode) |
| `--consolidation-anneal` | flag | false | Enable linear gate annealing |
| `--zero-injection-eval` | flag | false | Load checkpoint, zero injection, run val, exit |

## Metrics and Diagnostics

### CSV Columns (new)

| Column | Description |
|---|---|
| `mse_loss` | Raw MSE between decoder prediction and engram residual |
| `consol_phase` | 0 during normal training, 1 during consolidation |
| `inject_mult` | Injection multiplier (1.0 normally, annealing in zeta-B) |

### Key Measurements

At step 10K for each run, record:

| Metric | Description |
|---|---|
| `val_pre` | Val loss with injection active |
| `val_post` | Val loss with injection zeroed |
| `delta_removal` | `val_post - val_pre` (how much model loses without table) |
| `delta_vs_beta` | `val_post - 3.648` (does consolidated model beat dense baseline?) |

### Diagnostic Signals

**zeta-A:** Watch `mse_loss` convergence over 10K steps. Steady decline = internalization happening. Plateau = model can't learn the mapping.

**zeta-B:** Watch `mse_loss` during Phase 2. Watch `val_loss` as `inject_mult` decays — flat val_loss during annealing = consolidation working in real time. Val_loss degrading linearly with inject_mult = model losing signal without internalizing.

**zeta-ctrl:** Only `delta_removal` matters. Baseline for "how much does the table matter."

## Decision Tree

### If zeta-ctrl delta_removal ~ 0:

The model barely notices table removal. The engram injection was never providing lasting value. Consolidation is moot — fundamental architectural rethink needed.

### If zeta-ctrl delta_removal > 0 AND zeta-A or zeta-B delta_removal < zeta-ctrl:

Consolidation works. The model retained some engram benefit after table removal.

| Outcome | Interpretation | Next Step |
|---|---|---|
| A better than B | Online consolidation sufficient | Simpler approach wins; scale up |
| B better than A | Phased approach justified | Biological intuition validated; refine sleep schedule |
| A ~ B | Consolidation robust to timing | Use simpler A; scale up |

### If zeta-A and zeta-B delta_removal ~ zeta-ctrl:

Consolidation didn't help. The model couldn't internalize even with explicit MSE pressure. Options:
- Deeper decoder (current is 2-layer MLP, try 3-4 layers)
- Multi-layer injection (current is single layer 2)
- The engram signal is fundamentally non-internalizable at this scale

### If any post-removal val_loss < beta (3.648):

Strongest possible result. The engram table acted as a training-time scaffold that permanently improved the model. The model beats dense baseline even without the table.

### Success Threshold

Given the inert band reality:
- **Promising:** delta_removal(zeta-A or B) < delta_removal(zeta-ctrl) by >= 0.005 nats
- **Strong:** Post-removal val_loss < beta (3.648)

## Run Commands

```bash
# zeta-A: Online consolidation
python -m ct87.train --config tiny_engram_xattn_consol_online \
  --engram-xattn-table <oracle-table> \
  --data <dataset> --val-data <val-dataset> \
  --steps 10000 --consolidation-mode online \
  --consolidation-lambda 0.1 \
  --checkpoint-interval 1000 \
  --output-dir <output-zeta-a>

# zeta-B: Phased consolidation
python -m ct87.train --config tiny_engram_xattn_consol_phased \
  --engram-xattn-table <oracle-table> \
  --data <dataset> --val-data <val-dataset> \
  --steps 10000 --consolidation-mode phased \
  --consolidation-start-step 7000 --consolidation-anneal \
  --consolidation-lambda 0.1 \
  --checkpoint-interval 1000 \
  --output-dir <output-zeta-b>

# zeta-ctrl: Control (no consolidation)
python -m ct87.train --config tiny_engram_xattn_ctrl \
  --engram-xattn-table <oracle-table> \
  --data <dataset> --val-data <val-dataset> \
  --steps 10000 --consolidation-mode none \
  --checkpoint-interval 1000 \
  --output-dir <output-zeta-ctrl>

# Post-removal measurement (run for each of zeta-A, zeta-B, zeta-ctrl)
python -m ct87.train --config <same-config> \
  --engram-xattn-table <oracle-table> \
  --val-data <val-dataset> \
  --resume-from <output-dir>/checkpoint.pt \
  --zero-injection-eval
```

## Timeline

```
Phase 0: Code changes (~2-3 hours)
  +-- EngramConsolidationDecoder in engram.py
  +-- Consolidation CLI args + training loop changes in train.py
  +-- Config presets in model.py
  +-- --zero-injection-eval mode
  +-- Tests

Phase 1: Run zeta-ctrl on KRILE (~10 GPU-h)
  +-- Prerequisite: epsilon complete, oracle table available
  +-- May already have usable data from epsilon run

Phase 2: Run zeta-A on KRILE (~10 GPU-h)

Phase 3: Run zeta-B on KRILE (~10 GPU-h)
  +-- Or run zeta-A on KRILE + zeta-B on AVALON in parallel

Phase 4: Post-removal eval for all three (~minutes each)

Phase 5: Analysis and decision (~1h)
```

Total: ~30 GPU-hours sequential, ~20 GPU-hours if parallelized across KRILE + AVALON.

## Hardware

| Machine | GPU | Role |
|---|---|---|
| KRILE | RTX 4090 | Primary: sequential runs or zeta-A |
| AVALON | RTX 5080 16GB | Parallel: zeta-B (needs oracle table copied via scp) |

## Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| MSE loss overwhelms LM objective | Poor language modeling | lambda_consol = 0.1 conservative; monitor val_loss for degradation |
| Decoder too weak to learn mapping | MSE plateaus high | 2-layer MLP with 512 hidden should suffice; can increase if needed |
| Phase transition destabilizes training (zeta-B) | Loss spike at step 7K | Checkpoint at 7K for fallback; MSE ramp-in if needed |
| zeta-ctrl already exists as epsilon data | Wasted GPU time | Check epsilon post-removal before running separate ctrl |
| Gate annealing too aggressive | Model can't keep up | Linear over 3K steps is conservative; can slow to 5K if needed |

## References

- Complementary Learning Systems (McClelland et al.): dual-rate hippocampal-cortical learning
- Sleep and memory consolidation (Diekelmann & Born 2010): offline replay transfers episodic to semantic memory
- Knowledge distillation (Hinton et al. 2015): student learns to mimic teacher's outputs
- ZEB-119 oracle diagnostic: Mistral-7B distilled table, 88% PCA variance
- ZEB-127 epsilon results: per-head gates lockstep at ~0.68, no differentiation
- Alpha/beta/gamma/delta bake-off: 0.009-nat inert band
