# ZEB-139 — KL-Retrofit Objective-Axis Experiment (Design Spec)

> **STATUS: DRAFT — blocked on PR #254 merge + teacher-logits extension.** This spec is written during the ZEB-137/138 wait so the experiment can launch immediately once the prereq PRs land on main. AVALON can execute end-to-end in ~4-6h once unblocked.

**Linear:** [ZEB-139](https://linear.app/zeblith/issue/ZEB-139/kl-retrofit-experiment-objective-axis-diagnostic-for-engram-attractor)
**Parent:** [ZEB-102](https://linear.app/zeblith/issue/ZEB-102)
**Prior art:** [ZEB-134](https://linear.app/zeblith/issue/ZEB-134), [ZEB-136](https://linear.app/zeblith/issue/ZEB-136), [ZEB-138](https://linear.app/zeblith/issue/ZEB-138)
**Reference memory:** `reference_router_entropy_attractor.md` (emergent label smoothing framing)
**Reference research:** `docs/research/2026-04-18-post-zeb138-path-research-findings.md` on `zeblith/zeb-138-same-arch-teacher`
**Base:** `origin/main` at c96f3e7

---

## 1. Goal

Determine whether the router-entropy attractor observed in ZEB-134/136 is **training-objective-dependent** (CE-only training fails, KL+CE training escapes) or **signal-informativeness-dependent** regardless of objective. This runs orthogonally to ZEB-138 (teacher-architecture axis) and together they form a 2×2 matrix that cleanly disambiguates which axis dominates.

---

## 2. Hypothesis

Per the Gemini findings §6.2 and Memory Decoder (Cao et al., NeurIPS 2025, arXiv:2508.09874): adding `L_KL = KL(P_router || P_teacher)` to the training objective pre-aligns the router to the teacher's output distribution, providing dense gradient signal that escapes the label-smoothing attractor. Memory Decoder validates this at sub-500M scales on frozen backbones.

**Prediction:** if the attractor is purely an artifact of CE-only optimization under uninformative side-signal, adding KL pressure should break it even on the cross-arch TinyLlama setup where ZEB-136 observed attractor. If it doesn't break, the attractor is deeper than objective choice and the Gemini steelman gains weight.

---

## 3. The 2×2 design

|  | CE-only training | CE + KL training |
|---|---|---|
| **cross-arch teacher** (TinyLlama-1.1B) | ZEB-136 (done; attractor observed) | **this experiment** |
| **same-arch teacher** (Harmony-474M) | ZEB-138 (running) | follow-up if this + ZEB-138 both yield signal |

Together with ZEB-138, the four outcomes disambiguate the axis structure (see verdict matrix §11).

---

## 4. Training formulation

### 4.1 Loss function

```text
L_total = (1 − λ) · L_ce + λ · L_kl
L_kl    = KL(P_router || P_teacher)
P_router = softmax(engram_logits)       # shape [batch, seq, 32000]
P_teacher = softmax(teacher_logits)     # looked up from oracle, same shape
```

### 4.2 Hyperparameters

- **λ = 0.5** for initial run (Memory Decoder's starting value)
- **λ ablation:** if initial run produces attractor-break, sweep {0.1, 0.3, 0.5, 0.7, 0.9} to find the λ at which break first appears. If no break at 0.5, try 0.9 once before concluding.
- **KL direction:** forward KL `KL(router || teacher)` (mean-seeking, matches Memory Decoder). Backward KL is not in primary scope but can be added as sensitivity if warranted.
- **Base LR, batch, aux losses:** unchanged from ZEB-136 — ι₂ (Q-div + V-contrast) still active.

### 4.3 Implementation

Add to `training/ct87/train.py` training loop:

```python
# Where engram_logits is already computed per ZEB-134 skip-to-logit router
if args.kl_lambda > 0:
    # P_teacher looked up from oracle at the same row indices used for engram retrieval
    teacher_logits = oracle_teacher_logits[row_indices]  # [batch, seq, vocab]
    log_p_router  = F.log_softmax(engram_logits, dim=-1)
    p_teacher     = F.softmax(teacher_logits, dim=-1)
    # Per-token reduction matches CE (which uses reduction="mean" =
    # divide by batch*seq). F.kl_div's "batchmean" only divides by
    # batch, so at seq=2048 it would be ~2048x larger per token than
    # the CE term — making `λ` implicitly control loss normalization
    # rather than the actual KL/CE blend. If a padding/attention mask
    # is in scope, replace `.mean()` with `(kl * mask).sum()/mask.sum()`.
    kl_per_token = F.kl_div(log_p_router, p_teacher, reduction="none").sum(-1)
    kl_loss      = kl_per_token.mean()
    loss         = (1.0 - args.kl_lambda) * ce_loss + args.kl_lambda * kl_loss
```

The `oracle_teacher_logits` tensor is a new input to the training loop (shape `[10000, 32000]` bf16 ≈ 640MB) loaded alongside the existing oracle table. Lookup uses the same xxhash → row-index the engram retrieval already computes.

---

## 5. Prerequisite: teacher-logits extension to `generate_oracle_table.py`

### 5.1 Scope

Add optional `--save-teacher-logits` flag to `generate_oracle_table.py`. When enabled:

1. During the teacher forward pass, also extract the teacher's LM-head output (after the final norm, shape `[batch, seq, vocab]`).
2. Welford-mean the per-position logit distributions keyed by the same xxhash row index used for hidden-state aggregation.
3. Save as a separate sidecar file `<output>_teacher_logits.safetensors` with tensor `teacher_logits.weight` shape `[total_entries, vocab]`, dtype bf16.

### 5.2 Storage budget

10,000 rows × 32,000 vocab × 2 bytes (bf16) = **640MB per oracle shard**. Tractable on AVALON SSD; no compression needed.

### 5.3 Dependency

- **Blocker:** PR #254 merge (Harmony-teacher URI path touches same file). Wait for merge before opening this extension PR to avoid conflicts.
- **Wait order:** PR #254 lands → open extension PR → merge → extract TinyLlama oracle with new flag → run KL-retrofit training.

---

## 6. Experimental matrix

4-cell structure identical to ZEB-134/136, with KL term added only to router-on cells:

| Cell | Router | Oracle | KL term | Purpose |
|------|--------|--------|---------|---------|
| 1 | off | real | off | baseline val_loss (should match ZEB-136 cell 1 exactly — sanity check) |
| 2 | off | shuf | off | baseline val_loss shuffled (should match ZEB-136 cell 2 — sanity check) |
| 3 | on | real | **on** | **KEY CELL** — tests KL-retrofit attractor-break |
| 4 | on | shuf | **on** | shuffled control — KL term against shuffled-teacher logits (should NOT break attractor — validates the KL signal is content-dependent) |

Cell 4 is a critical sanity check: if attractor breaks on cell 4 (shuffled teacher logits) as strongly as on cell 3 (real), then the KL term is generically forcing non-uniform output regardless of content — not evidence of actual content-routing.

---

## 7. Diagnostic fingerprint

Full ZEB-138 §3/§7 attractor-fingerprint suite:

1. `engram_logit_entropy` < log(32000) − 0.1 — router emits non-uniform distributions
2. `α` outside [0.14, 0.20] — no longer the dataset-optimal label-smoothing factor
3. Cross-run cosine < 0.7 — content-driven variance across seeds
4. Δ-diff ≥ +0.001 nats (real − shuffled val_loss) — LM-measurable effect
5. `W_align` Frobenius drift > 2× init — gradients flowed and were absorbed

### 7.1 KL-specific additions

- **KL divergence trajectory:** log KL loss every training step; expected to drop steadily if the router is learning to match teacher distributions
- **Teacher-match cosine:** per-step cosine between `P_router` and `P_teacher` averaged over batch — direct measurement of whether the router is tracking teacher
- **Attention to teacher vs. to backbone:** if attractor breaks, did it break *because* KL forced sharp output (cheap win) or *because* the router learned content-specific routing (real win)? Discriminator: compare cell-3 vs. cell-4 final-cell fingerprints. If cell 4 has broken attractor too, it's the cheap win.

---

## 8. Hardware + wall time

- **Oracle re-extraction (TinyLlama, with --save-teacher-logits flag):** ~1-2h AVALON (recomputes what ZEB-136 already did, just with extra logits logged)
- **4-cell training matrix:** ~45 min AVALON (matches ZEB-136's observed time)
- **Forensics + findings doc:** ~1h
- **Total:** **~4-6h AVALON end-to-end** once PR #254 and the extension PR have merged.

---

## 9. Dependencies

- **Hard blocker:** PR #254 merge (Harmony-teacher URI path)
- **Hard blocker:** teacher-logits extension PR (filed after #254 merges)
- **Soft coupling to ZEB-138:** interpretation is strongest with both sets of results in hand, but runs are independent. Can execute while KRILE is on ZEB-137 and before ZEB-138 cell A runs.

---

## 10. Scope boundaries

### In scope

- Teacher-logits extension to oracle generator
- KL+CE training-loss implementation
- 4-cell TinyLlama matrix with the new loss
- Full attractor-fingerprint forensics + findings doc

### Out of scope

- **Same-arch teacher + KL:** the 2×2's fourth cell is a follow-up contingent on results here + ZEB-138
- **Mistral-7B teacher:** 7B teacher's logit extraction is expensive; 640MB × 32k = fine for storage, but forward pass requires re-loading the 7B model. If results here are interesting on TinyLlama, re-running on Mistral can be scoped separately
- **Backbone unfreezing:** separate Gemini Path-A recommendation; would confound the objective-axis test
- **Multi-layer non-linear W_align:** Gemini Path-C recommendation; separate issue scope

---

## 11. Verdict matrix — this experiment × ZEB-138

| This (KL-retrofit) | ZEB-138 | Combined interpretation | Recommended next step |
|---|---|---|---|
| Attractor breaks | Attractor breaks | **Objective shift alone is sufficient** — both teachers work under KL+CE | Lead with Memory-Decoder-style KL training; same-arch teacher is a multiplier not a dependency |
| Attractor breaks | Attractor holds | **Objective dominates, teacher-arch is secondary** — KL escapes attractor even on cross-arch setup that ZEB-138's CE-only can't | Pivot to KL+CE as primary training; teacher choice deprioritized |
| Attractor holds | Attractor breaks | **Teacher-arch dominates, objective insufficient** — KL can't overcome cross-arch modality gap at 40M | Same-arch teacher is the critical axis; pursue ZEB-138 cell-A-break as substrate |
| Attractor holds | Attractor holds | **Structural ceiling confirmed** (per Gemini §7 steelman) | Multi-layer non-linear W_align OR abandon frozen-backbone constraint. 40M linear pipeline is a dead end |
| Attractor partially breaks (one or two thresholds) | any | **Soft signal** — extend training, λ-sweep, investigate which threshold broke first | Inconclusive; needs more data before committing to a path |

### Additional discriminator — cell 3 vs. cell 4 within this experiment

| Cell 3 attractor-broken? | Cell 4 attractor-broken? | What it means |
|---|---|---|
| yes | yes | KL signal is content-independent — router just produces sharp output because KL says so. Not evidence of content-routing. |
| yes | no | KL signal is content-dependent — cell 3 sees real teacher signal, cell 4 sees shuffled. **This is the clean positive result.** |
| no | no | KL term didn't break attractor. Default interpretation: label-smoothing attractor is mechanism-resistant at 40M. |
| no | yes | Anomaly — worth investigating (could indicate a bug in oracle lookup indexing). |

---

## 12. Open questions

1. **KL direction:** forward vs. backward — primary design is forward (mean-seeking). Running both in parallel as a sensitivity ablation is small-cost (same runtime, different loss term) and would make the result more robust. **Recommend including both if AVALON time allows.**
2. **Temperature on P_router:** some distillation literature recommends temperature > 1 on both sides of the KL to smooth gradients. ZEB-139 starts without it (T=1). Revisit if initial run shows pathological training dynamics.
3. **Batch size / teacher-logit load:** 640MB resident is fine on 14GB 5080, but at higher batch sizes the per-batch KL logit tensor can become large. Might need gradient accumulation if memory becomes an issue.
4. **Cold-start vs. warm-start:** should the router W_align start from ZEB-136's trained weights (warm) or from scratch (cold)? Warm-start is faster but biases away from baseline comparison. **Recommend cold-start for cleanest signal against ZEB-136.**

---

## 13. Self-review

- **Placeholder scan:** §4.3 pseudo-code is illustrative, not final — actual implementation should match the training-loop patterns already established in `training/ct87/train.py`. This is a known handoff point to whoever executes the issue. Similarly §5.1 is scope-only; implementer fills in details.
- **Internal consistency:** §3 2×2 design, §11 verdict matrix, and the discriminator within this experiment (§11 cell 3 vs. cell 4) all line up.
- **Scope:** single experiment, testable independently of ZEB-138, with a clear prerequisite chain (PR #254 → extension PR → this).
- **Ambiguity:** §7.1's "teacher-match cosine" metric isn't in the existing forensic suite; implementer may need to add it. Small addition. No downstream consequences.
- **Cell 4 sanity check:** the single most important design element. Without cell 4, we can't distinguish "KL broke the attractor because content routing works" from "KL broke the attractor because KL term forces sharp output." If cell 4 also breaks, the result is not a positive signal — it's a confound.
