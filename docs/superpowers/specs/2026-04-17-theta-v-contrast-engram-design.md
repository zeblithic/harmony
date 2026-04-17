# Experiment θ-V-contrast: V-contrastive engram injection (ZEB-130)

**Date:** 2026-04-17
**Issue:** ZEB-130 (engram capacity-gap investigation)
**Parent:** ZEB-102 (Engram table quality)
**Depends on:** PR #247 (η-B capgap infrastructure), PR #249 (forensic (W)/(A) probes)
**Status:** Design approved, awaiting implementation plan

## Motivation

The 40M engram cross-attention architecture is **content-invariant**: training metrics (val loss, Δ-removal) are identical when the oracle table is row-shuffled (η-B 2026-04-16: real +0.0125 vs shuffled +0.0124; ζ-ctrl 2026-04-17: real +0.0163 vs shuffled +0.0158). Two independent architectures (xxhash/conv1d from Phase 0 and cross-attention from η-B / ζ-ctrl) both show the same failure mode.

The (W)/(A) forensic (PR #249, run against η-B real + shuffled checkpoints on 2026-04-17) localized the mechanism to verdict **(D\*) DISTRIBUTIONAL ALIGNMENT**:

| Metric | L2 | L5 | Implication |
|---|---|---|---|
| Within-run RMS_cos (axis concentration) | 0.23 | 0.20 | V is token-sensitive; per-token injection directions are diverse |
| Within-run R (signed concentration) | 0.29 | 0.32 | No fixed per-run direction |
| \|cos(inj, hidden)\| | 0.08 | 0.06 | Injection is orthogonal to residual stream — not a residual-axis amplifier |
| Cross-run cos (real vs shuffled models, matched tokens) | +0.34 | +0.63 | Different models trained on different tables produce correlated injection directions despite different retrieved content |

Verdict summary: V's per-token output directions are **diverse within a run** but **align across runs** despite different retrievals. Content-invariance lives in V's response-distribution shape — V's outputs are hidden-state-determined with near-zero dependence on which rows retrieval returns.

This rules out the earlier "hard retrieval-relevance gate" direction (which would have addressed a fixed-axis amplifier that the forensic showed doesn't exist). The remaining productive direction is **architectural pressure on V to make its output content-sensitive**.

## Strategy: V-contrastive auxiliary loss

At training time, the model runs a **second xattn forward** per injection layer against a per-step random permutation of the primary table. An auxiliary loss penalizes cosine alignment between the real-branch and shuffled-branch post-o_proj outputs at matched positions. The shuffled branch does NOT contribute to the residual stream — it exists only to shape gradient pressure on V/K/Q/o_proj.

Because both branches share parameters, any V that learns content-sensitivity on the shuffled branch becomes content-sensitive everywhere.

## Architecture

### Forward pass (per injection layer)

```python
# Primary branch (identical to η-B, minus the gate — we need inj_real pre-gate):
retrieved_real, topk_sims_real = retrieve_topk(h, table_real, table_real_normalized)
inj_real = xattn_pipeline(h, retrieved_real, topk_sims_real)   # [B, L, hidden_dim]

# Shuffled branch (training only, aux-loss only).
# Value-only shuffle: we keep the real branch's top-k *indices* and substitute
# content from a permuted table at those same positions. A full key+value
# permutation would be a semantic no-op — top-k selection is permutation-
# equivariant, so the same logical rows win and retrieved_shuf ≡ retrieved_real,
# yielding zero contrastive signal.
if self.training:
    perm = torch.randperm(N, device=table_real.device)
    table_shuf = table_real[perm]                          # [N, engram_dim]
    # Re-derive top-k indices on the REAL table (retrieve_topk does not
    # expose them). This uses the already-trained retrieval_query_proj so
    # gradient still flows into it via the shuf branch's attention-bias.
    q = retrieval_query_proj(h)
    q_norm = F.normalize(q, dim=-1, eps=1e-8)
    sims_real = torch.einsum("ble,te->blt", q_norm, table_real_normalized)
    topk_sims_real, topk_idx_real = sims_real.topk(k_retrieved, dim=-1)
    # Substitute shuffled-table content at the same retrieval positions.
    # `topk_sims_real` is passed as the retrieval bias so attention scoring
    # sees the same "rank" signal as the real branch — isolating V/K as the
    # projections whose content-sensitivity is being pressured.
    retrieved_shuf = table_shuf[topk_idx_real]
    inj_shuf = xattn_pipeline(h, retrieved_shuf, topk_sims_real)

    cos = F.cosine_similarity(inj_real, inj_shuf, dim=-1)   # [B, L]
    aux_loss_this_layer = (cos ** 2).mean()
    model._contrastive_aux_losses.append(aux_loss_this_layer)

# Residual (unchanged from η-B):
gate = torch.tanh(alpha).to(dtype=inj_real.dtype)
return gate * inj_real
```

### Design decisions

**Aux-loss target: post-o_proj pre-gate.** Measuring pre-gate closes the loophole where the model could minimize aux loss by closing the gate (tanh(alpha) → 0) instead of making V content-sensitive. The gate's role is firing-rate; the aux loss's role is response-shape. Keeping them independent.

**Loss shape: mean squared cosine.** Per-position `cos(inj_real, inj_shuf)²`, averaged over `[B, L]`. Smooth everywhere, natural attractor at `cos=0`, bounded `[0, 1]`. Avoids the anti-alignment pathology of signed cosine minimization (where V could converge to `V(shuf) = -V(real)`, still fixed).

**Shuffle strategy: per-step value-only permutation of the primary table.** Fresh `torch.randperm(N)` each forward prevents V from learning features specific to one fixed shuffle. Value-only means we keep the real branch's top-k indices (`topk_idx_real`) and substitute content at those positions from a permuted table (`table_shuf = table_real[perm]`, `retrieved_shuf = table_shuf[topk_idx_real] = table_real[perm[topk_idx_real]]`). This delivers genuinely different content at the same retrieval positions. A full key+value permutation would be a semantic no-op: top-k selection is permutation-equivariant, so the same logical rows win regardless of row ordering, and `retrieved_shuf ≡ retrieved_real`, yielding zero contrastive signal. Runtime cost: one extra `retrieval_query_proj` + einsum + topk per layer (the primary `retrieve_topk` does not expose its `topk_idx`, so the shuf branch re-derives them).

**Training protocol: capgap with frozen backbone.** Mirrors η-B exactly: load β-baseline weights, freeze backbone, train only engram params. Clean isolation — any content-sensitivity gained is attributable to aux pressure on the engram module, not backbone co-adaptation. If θ-V-contrast succeeds at capgap, end-to-end (ζ-style) extension is a follow-up experiment.

**λ schedule: linear warmup 0 → 1.0 over first 200 steps, constant thereafter.** λ=0 at step 0 avoids pushing V based on random-init cosines before LM loss has shaped useful representations. 200 steps is 10% of η-B's 2000-step budget. A small sweep (λ ∈ {0.3, 1.0, 3.0}) is a fallback if the initial run shows aux dominating LM or being ignored.

**Gate interaction: unchanged.** `tanh(alpha)`, `alpha_init=0.0`, step-0 no-op guarantee, `xavier_uniform` o_proj re-init — all inherited from `GatedEngramInjection`. Alpha learns only from LM loss through the residual path; aux loss never touches alpha's gradient.

### Total loss

```python
def lambda_schedule(step: int, warmup: int = 200, target: float = 1.0) -> float:
    if step < warmup:
        return target * step / warmup
    return target

aux_loss_total = sum(model._contrastive_aux_losses)   # sum over injection layers (2 + 5)
loss = lm_loss + lambda_schedule(step) * aux_loss_total
```

Both injection layers (2, 5) contribute aux loss with equal weight. D\* appeared at both layers with comparable magnitude — no principled reason for per-layer weighting in the initial design.

## Implementation surface

| File | Change |
|---|---|
| `training/ct87/engram.py` | Add `ContrastiveGatedEngramInjection(GatedEngramInjection)` subclass |
| `training/ct87/model.py` | Add 3 config fields, `tiny_engram_xattn_capgap_vcontrast` preset, `_contrastive_aux_losses: list` model attribute, update `attach_gated_engram_injections` to attach the contrastive variant when the config flag is set |
| `training/ct87/train.py` | CLI args, setup, training-loop λ-scheduling and aux-loss accumulation + logging columns |
| `training/scripts/forensic_eta_b_capgap.py` | Add `analyze_cross_table` function + `--alt-shuffle-seed` CLI arg. Ships as a separate small PR ahead of θ-V-contrast training |

All changes are additive. η-B, ζ-ctrl, δ xattn paths behave identically to today when the new flag is off.

### New config fields (HarmonyModelConfig)

```python
engram_vcontrast_enabled: bool = False
engram_vcontrast_lambda: float = 1.0
engram_vcontrast_warmup_steps: int = 200
```

### New CLI flags (train.py)

```
--engram-vcontrast                    # enable aux loss (default off)
--engram-vcontrast-lambda FLOAT       # default 1.0
--engram-vcontrast-warmup-steps INT   # default 200
--engram-vcontrast-shuffle-seed INT   # optional, for reproducibility debugging only
```

### New config preset

```python
@staticmethod
def tiny_engram_xattn_capgap_vcontrast() -> HarmonyModelConfig:
    """θ-V-contrast: η-B capgap + V-contrastive aux loss (ZEB-130)."""
    base = HarmonyModelConfig.tiny_engram_xattn_capgap()
    base.engram_vcontrast_enabled = True
    base.engram_vcontrast_lambda = 1.0
    base.engram_vcontrast_warmup_steps = 200
    base.__post_init__()
    return base
```

### Module design (subclass, not flag)

`ContrastiveGatedEngramInjection` subclasses `GatedEngramInjection`. Stores a reference to a model-level `aux_loss_sink: list[Tensor]`. Overrides `forward` to:

1. Compute `inj_real` via the parent's xattn (unchanged primary branch).
2. If `self.training` and `aux_loss_sink is not None`: compute `inj_shuf` via the shuffled branch (fresh permutation of the xattn table), compute `aux_loss_this_layer`, append to sink.
3. Apply `tanh(alpha)` gate to `inj_real` and return (identical to parent).

Subclass over flag because it keeps the η-B module surface untouched; the contrastive variant is a new operational mode, not a parameter of the existing one. Old checkpoints load into the parent class unchanged.

### Model-level integration

```python
# HarmonyModel.__init__
self._contrastive_aux_losses: list[torch.Tensor] | None = None    # enabled when attaching

# HarmonyModel.attach_gated_engram_injections(injections):
# If config.engram_vcontrast_enabled, attach ContrastiveGatedEngramInjection
# instances with self._contrastive_aux_losses as the sink. Else attach plain
# GatedEngramInjection. Initialize the list iff vcontrast is enabled.

# HarmonyModel.forward (at start of forward):
if self._contrastive_aux_losses is not None and self.training:
    self._contrastive_aux_losses.clear()
# Injection modules append during the subsequent layer-by-layer forward.
```

### Training-loop integration

```python
# After forward + lm_loss compute, inside the training step:
if config.engram_vcontrast_enabled and model.training:
    aux_losses = model._contrastive_aux_losses
    if aux_losses:
        aux = torch.stack(aux_losses).sum()
        lam = lambda_schedule(
            step,
            config.engram_vcontrast_warmup_steps,
            config.engram_vcontrast_lambda,
        )
        loss = loss + lam * aux
        # Logging accumulators:
        accum_vcontrast_aux += aux.detach()
        accum_vcontrast_lambda = lam
```

## Training protocol

### Base config: `tiny_engram_xattn_capgap_vcontrast`

Extends `tiny_engram_xattn_capgap` (η-B config) with aux-loss fields. Keeps `engram_inject_layers=(2, 5)`, `engram_gate_init=0.0`.

### Run 1 — real-primary

```bash
python train.py \
  --config tiny_engram_xattn_capgap_vcontrast \
  --init-from checkpoints/beta_baseline/ \
  --engram-xattn-table artifacts/oracle_mistral7b_10k.safetensors \
  --engram-vcontrast \
  --freeze-backbone \
  --max-steps 2000 \
  --save-dir checkpoints/theta_vcontrast_real/
```

### Run 2 — shuffled-primary

```bash
python train.py \
  --config tiny_engram_xattn_capgap_vcontrast \
  --init-from checkpoints/beta_baseline/ \
  --engram-xattn-table artifacts/oracle_mistral7b_10k_shuffled_seed0.safetensors \
  --engram-vcontrast \
  --freeze-backbone \
  --max-steps 2000 \
  --save-dir checkpoints/theta_vcontrast_shuffled/
```

Both runs enable aux loss. "Primary" means the table used for the residual-contributing branch (and the first half of the aux-loss target). Each run's shuffled branch is per-step permutations of its own primary table. The between-run comparison is the downstream content-sensitivity test.

### Optimizer, LR, steps

- AdamW, lr=9e-4, 2000 steps, cosine decay — same as η-B
- grad_accum_steps as per existing η-B protocol
- No LR sweep in the initial design. If empirics surface differently (aux interacts with optimal LR), a small LR sweep at iteration 2.

### Compute budget

- Engram forward: ~2× (retrieval done twice per injection layer).
- Overall training time estimate: 1.3–1.8× η-B's ~1 hr = ~1.5–2 hr per run on KRILE.
- Two runs = ~3–4 hr total KRILE time.

## Validation protocol

### After each training run lands

1. **Δ-removal measurement** (same protocol as η-B). Eval with injection and with `--zero-injection-eval`, compute diff.
2. **Forensic with existing (W)/(A) probes** (PR #249 content). Cross-run comparison between the two trained checkpoints.
3. **Cross-table within-run probe** (new forensic extension, separate small PR ahead of training).

### Cross-table within-run probe

For a single trained model, forward the same tokens through its engram against TWO different tables — its training primary, and a fresh held-out random shuffle (never seen during training). Compute cos between matched-position injection outputs.

```python
# New function in scripts/forensic_eta_b_capgap.py:
@torch.no_grad()
def analyze_cross_table(
    model: HarmonyModel,
    primary_table: torch.Tensor,
    alt_table: torch.Tensor,              # held-out random shuffle
    val_batches: Iterable[torch.Tensor],
    layers: list[int],
) -> dict[int, dict[str, float]]:
    """Forward the same tokens through `model` against primary vs alt tables;
    compute cos between matched-position injection outputs per layer."""
```

Held-out alt table is a fresh `torch.randperm(N)` seeded by a new CLI arg `--alt-shuffle-seed` (default 42) so it's explicitly different from any training-time shuffle.

Output (new forensic rows per layer):

```
(X) cross-table within-run cos
    signed (primary vs alt)           <X.XXXX>
    |abs|                             <X.XXXX>
```

This is the direct test of "is V content-sensitive?" — same weights, same tokens, different table → must give different outputs if V is content-sensitive.

### Secondary "random baseline" probe (trivial-orthogonality guard)

Also compute cos between two random tokens' V outputs within the same forward (no table swap). If `cross_table_|cos|` is as low as this "random within-run token-pair cos", V found a trivial-orthogonality shortcut without genuine content-awareness — need to pivot.

## Success criteria

Order-of-magnitude targets, not tight thresholds:

| Metric | η-B baseline | θ-V-contrast target | If missed |
|---|---|---|---|
| Cross-run cos (L2) | +0.34 | < 0.15 | V's cross-model alignment still high — different V weights still producing correlated responses |
| Cross-run cos (L5) | +0.63 | < 0.15 | (same) |
| Within-run RMS_cos | ~0.20 | ~0.20 (unchanged) | Aux loss doesn't target within-run axis concentration; unchanged is fine |
| Within-run cross-table \|cos\| (NEW) | not measured | < 0.2 | Direct test of V content-sensitivity |
| Δ-removal (real-primary) | +0.0125 | any (monitor) | Value less important than the differentiation below |
| Δ-removal (shuffled-primary) | +0.0124 | any (monitor) | (same) |
| **Δ-removal diff (real − shuffled)** | **+0.0001** | **> 0.002** | **Decisive downstream metric** |

**Verdict matrix:**

| Outcome | Diagnosis | Next step |
|---|---|---|
| All 4 primary targets hit | α.1 works; architecture was the bottleneck | ζ-style extension; scale to 90M |
| Cross-table cos target hit, Δ-removal diff flat | V content-sensitive but signal doesn't help LM at 40M on fineweb | Long-tail / synthetic KV probe to test "right signal, wrong task" hypothesis |
| Cross-run cos target hit only | Two runs converge similarly but neither individually content-sensitive against held-out shuffle | Increase shuffle diversity during training; else α.2 (adversarial V) |
| Trivial-orthogonality collapse (cross-run cos low, cross-table cos NOT low relative to random baseline) | V found shortcut around aux loss | α.2 adversarial V |
| Nothing hits | α.1 theory-of-action wrong for this failure mode | α.3 (contrastive V-pretraining) or more fundamental redesign |
| LM loss materially worse than η-B | λ too aggressive | Lower λ to 0.3, retry once |
| Alpha never opens past step 500 | Injection mechanism dead under capgap regardless of V's content-sensitivity | Fundamental issue — reconsider capgap vs ζ-style |

## Logging & monitoring

### New training-time CSV columns (conditional on `--engram-vcontrast`)

| Column | What it tells us |
|---|---|
| `vcontrast_aux_loss` | Total aux loss summed across layers; trends down if V is learning to separate |
| `vcontrast_aux_l2` | Per-layer aux loss at injection layer 2 |
| `vcontrast_aux_l5` | Per-layer aux loss at injection layer 5 |
| `vcontrast_lambda` | Current λ (rising during warmup, then constant) |
| `engram_gate_l2` | `tanh(alpha_l2)` current value |
| `engram_gate_l5` | `tanh(alpha_l5)` current value |

### Per-step console print

```
step= 1500  loss=4.5601  lr=0.000450  aux=0.1234  aux_L2=0.0623  aux_L5=0.0611  λ=1.000  g2=-0.187  g5=-0.194
```

### End-of-run summary

```
[vcontrast] Final step=2000
    aux_loss = <X.XXXX>
    val_loss (with inj)    = <X.XXXX>
    val_loss (without inj) = <X.XXXX>
    Δ-removal = <X.XXXX>
    alpha_L2 = <X.XXXX>, tanh(alpha_L2) = <X.XXXX>
    alpha_L5 = <X.XXXX>, tanh(alpha_L5) = <X.XXXX>
```

### Convergence signals (for the training summary)

- Aux loss trending down from ~0.5 at step 200 toward ~0.1 by step 2000 → V is learning to separate.
- Aux loss does NOT collapse to near-zero by step 500 (suggests a trivial orthogonality shortcut).
- `tanh(alpha)` opens to material magnitude (~±0.2 like η-B). If it stays at 0, injection isn't useful to LM regardless of V's content-sensitivity.
- LM loss stays competitive with η-B's 4.5461 with-inj. Material degradation indicates aux dominating LM gradient.

## Risks and failure modes

### Risk A — Trivial-orthogonality shortcut

Aux loss could be minimized by V emitting constant-magnitude random-direction outputs regardless of content. `cos(inj_real, inj_shuf) ≈ 0` from random-in-high-dim-space alone, without content-awareness.

**Detection:** the cross-table within-run probe catches this — V's output for the same token should change when we swap tables. The "random-baseline" secondary probe (cos between two random tokens' V outputs within a single forward, same table) gives the orthogonality-floor V could achieve without content-sensitivity.

**If detected:** pivot to α.2 (adversarial V) or α.3 (contrastive V-pretraining).

### Risk B — Per-step shuffle overfitting

V could learn something specific about `torch.randperm`-generated shuffles and produce content-blind outputs anyway. Unlikely in practice.

**Detection:** the cross-table forensic probe uses a held-out `torch.randperm` with a different seed. Genuine content-sensitivity generalizes.

### Risk C — LM-loss degradation

Aggressive λ causes V to optimize for orthogonality at the expense of representation quality. LM loss stalls above η-B's 4.5461.

**Detection:** training-time LM loss watch. Material worse-than-η-B by step 1000 is the trigger.

**Mitigation:** reduce λ to 0.3 and retry. The λ warmup already helps; the fallback is a cheap one-config-line change.

### Risk D — V content-sensitive but LM doesn't benefit

V becomes content-sensitive (cross-table cos drops) but Δ-removal stays flat / shuffle-invariant. Content-sensitivity exists but doesn't translate to LM improvement at 40M on fineweb.

**Interpretation:** V CAN be content-sensitive when pressured; 40M on fineweb just doesn't create gradient pressure for using that content. This is the "right signal, wrong task" hypothesis.

**If detected:** long-tail / synthetic KV probe task to check if V's content-sensitivity helps on a task that provably requires retrieval content.

### Risk E — Alpha never opens

Even after aux loss + warmup, `tanh(alpha)` stays near 0. LM gradient through the gated residual isn't strong enough to crack the gate.

**Detection:** trivial — `engram_gate_l2` / `engram_gate_l5` stay near 0 past step 500.

**Not mitigation-worthy on first run.** Alpha not opening means LM isn't finding ANY injection useful (content-sensitive or not). That's a more fundamental failure indicating the injection mechanism itself is dead under a trained-backbone capgap regardless of content-awareness. Flag as a separate finding.

## Cost estimate

- 2× training runs on KRILE: ~3–4 hr GPU time
- Forensic (with cross-table probe extension): ~5 min
- Total wall clock on KRILE: ~3–4 hr + iteration time if we need a λ retry

Falsifies or confirms α.1 cleanly. Pivot decisions afterward (α.2/α.3/scale/long-tail probe) are informed by concrete data rather than guesses.

## Testing expectations

Unit tests (plan-level; listed here so the scope is visible):

- `ContrastiveGatedEngramInjection.forward` in `self.training=True` populates the model's aux-loss sink with one scalar tensor per forward.
- Same forward in `self.training=False` leaves the sink untouched (no aux-loss compute, no per-step permutation).
- Residual returned by the contrastive forward is bit-identical to `GatedEngramInjection.forward` output given the same inputs (the shuffled branch is aux-loss-only and must not perturb the residual).
- Per-step permutation actually differs across two consecutive forwards (not a stale or cached permutation).
- `lambda_schedule` returns 0.0 at step 0, `target * step/warmup` during warmup, and exactly `target` at and past the warmup boundary.

Integration tests:

- 1-step training on a tiny 2-layer model with `engram_vcontrast_enabled=True` runs without error and records a non-None aux loss in the CSV log.
- Eval-only run (with `--zero-injection-eval`) on a vcontrast-trained checkpoint produces the same "with-inj" and "without-inj" val_loss format as η-B.
- Loading an η-B checkpoint (produced WITHOUT vcontrast) into a vcontrast-enabled config attaches `ContrastiveGatedEngramInjection` modules and loads parent weights cleanly (forward compatibility).

Forensic tests (in the separate cross-table PR):

- `analyze_cross_table` produces well-formed output for a single trained checkpoint + two tables.
- Held-out alt-shuffle seed is explicitly different from training-time random-shuffle seeds (CLI arg defaults to 42, overridable).

## Out of scope

- End-to-end (ζ-style) training with aux loss. Deferred to follow-up if capgap version succeeds; if it fails, the fallback is α.2 or α.3, not more training-regime variations.
- Scale-up to 90M. Same as above — contingent on capgap outcome.
- Changes to retrieval mechanism (top-k cosine), attention mechanism (Q·K + sim-bias), or gate mechanism (`tanh(alpha)`). All stay as-is; θ-V-contrast only adds an aux-loss branch.
- wandb / metrics-server integration. CSV + stdout matches existing training infrastructure.

## References

- ZEB-130 forensic memory: `project_zeb130_shuffle_kill.md` (updated 2026-04-17)
- KRILE engram memory: `project_krile_engram_training.md` (updated 2026-04-17)
- η-B capgap implementation: PR #247 (merged 2026-04-16)
- Forensic script: PR #248 (merged 2026-04-16) + PR #249 (open, (W)/(A) probes)

### α.1 / α.2 / α.3 redesign taxonomy

From the KRILE 2026-04-17 analysis after the forensic v2 revealed the (D\*) verdict. All three attack V's response-distribution alignment rather than firing decisions (which the now-deprecated "option (c) hard retrieval-relevance gate" would have attacked):

- **α.1 V-contrastive auxiliary loss** — penalize V-output alignment between real and shuffled retrievals during training. **This spec.**
- **α.2 V-discriminator (adversarial)** — train a small discriminator to predict V's output distribution from hidden state alone; train V to defeat it. Forces V to encode information not derivable from hidden state. Fallback if α.1's contrastive pressure proves insufficient.
- **α.3 Contrastive V-pretraining** — pre-train V on a contrastive task (true vs shuffled row retrievals) before mixing into the LM objective. Clean phase separation; fallback if α.1 and α.2 both fail to shift V.

Original hard-gate design ("option (c)") was deprecated on 2026-04-17 after the (W)/(A) forensic showed V is not a fixed-axis amplifier — the problem lives in response-distribution shape, not firing decisions.
