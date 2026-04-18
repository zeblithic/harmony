# ZEB-138 — Same-Architecture Teacher Engram Experiment (Design Spec)

> **STATUS: DRAFT — blocked on ZEB-137 checkpoint.** This spec is written during the ZEB-137 wait so that oracle extraction and matrix training can start at T+0s after the first usable Harmony-474M checkpoint lands. Open questions are flagged inline; they resolve once we know the checkpoint's step count, val-loss trajectory, and where `generate_oracle_table.py` extraction lands.

**Linear:** [ZEB-138](https://linear.app/zeblith/issue/ZEB-138/same-architecture-teacher-engram-experiment-scale-teacher-matrix)
**Parent:** [ZEB-102](https://linear.app/zeblith/issue/ZEB-102) (Engram content-routing research)
**Prior art:** [ZEB-134](https://linear.app/zeblith/issue/ZEB-134) (Mistral skip-to-logit router), [ZEB-136](https://linear.app/zeblith/issue/ZEB-136) (TinyLlama cross-arch replication)
**Reference memory:** `reference_router_entropy_attractor.md` (hypothesis + diagnostic thresholds)
**Base:** `origin/main` at c96f3e7

---

## 1. Goal

Decisively disambiguate **cross-architecture modality gap** from **fundamental 40M-capacity ceiling** as the driver of engram content-routing failure by introducing a same-architecture teacher (Harmony-474M from ZEB-137) to the skip-to-logit router setup established in ZEB-134.

**Decision criterion:** whether the router-entropy attractor (observed identically under Mistral-7B and TinyLlama-1.1B teachers) breaks under same-arch teacher. Attractor-break outcome and the remediation path it implies are spelled out in §8.

---

## 2. Context — the attractor observation

ZEB-134 (Mistral-7B, KRILE) and ZEB-136 (TinyLlama-1.1B, AVALON) produced the same router fingerprint within measurement noise:

| Metric                    | ZEB-134 (Mistral-7B) | ZEB-136 (TinyLlama-1.1B) | Spread |
|---------------------------|----------------------|---------------------------|--------|
| `engram_logit_entropy`    | log(32000) = 10.3735 | log(32000) = 10.3735      | 0.0%   |
| `α` (mixing weight)       | 0.17                 | 0.164                     | 3.5%   |
| `‖W_align‖_F`             | 1.96                 | 1.91                      | 2.5%   |
| Δ-diff (real − shuf)      | ~0.0002 nats         | +0.00022 / +0.00024       | noise  |

Two teacher scales (175× vs 28× capacity gap), two architectures in the same LLaMA lineage — **identical stable attractor**. The router emits exact max-entropy uniform and α settles at ~0.17 by KL balance against the frozen LM logits.

The attractor is a degenerate optimum: for any router that cannot produce content-sensitive distributions beating uniform cross-entropy, gradient descent finds "emit uniform, let α balance noise" regardless of teacher. Same-arch teacher is the one untested axis; if the attractor holds under Harmony-474M → Harmony-40M, the 40M capacity ceiling is empirically confirmed.

---

## 3. Primary diagnostic — attractor break

**Δ-diff alone is insufficient** (too noisy at +0.0002 to distinguish small signal). The attractor-fingerprint gives four orthogonal views; any one threshold crossed is positive signal. All four must hold within tight bands for a clean null.

| # | Threshold                                                 | Crossed meaning                                                    |
|---|-----------------------------------------------------------|--------------------------------------------------------------------|
| 1 | `engram_logit_entropy < log(vocab) − 0.1`                 | Router emits content-sensitive (non-uniform) distributions         |
| 2 | `α` outside [0.14, 0.20]                                  | No longer KL-balancing against uniform                             |
| 3 | Cross-run cosine of engram-logit outputs < 0.5            | Outputs are content-driven across RNG seeds, not shared baseline   |
| 4 | Δ-diff ≥ +0.001 nats                                      | Content-routing produces LM-measurable effect                      |

**Decision rules:**
- All four inside attractor band → attractor holds → capacity ceiling (see §8 Outcome A)
- One threshold crossed → soft signal → extend training / investigate which broke first (§8 Outcome B)
- Two or more crossed → attractor broken → modality gap is the bottleneck (§8 Outcome C)
- Δ-diff ≥ +0.005 → strong result → expand to scaling study (§8 Outcome D)

---

## 4. Experiment matrix

### Cell A (decisive, primary, MUST run)
| | |
|--|--|
| Student | Harmony-40M (frozen, same checkpoint used in ZEB-134/136) |
| Teacher | Harmony-474M (ZEB-137 checkpoint) |
| Layer (teacher) | TBD — recommend L18 of 24 (mid-to-late, matches relative position of Mistral L-2) |
| Hardware | AVALON (5080, 14GB) — 40M student fits cleanly, ZEB-136 infrastructure reusable |
| Sub-matrix | 4-cell: router {off, on} × oracle {real, shuffled} (same as ZEB-134/136) |

### Cell B (reference, REUSE from ZEB-134 — no new training)
Harmony-40M × Mistral-7B. Already ran in ZEB-134 with full attractor fingerprint. Reference that run's numbers in the ZEB-138 findings doc; no re-training needed.

### Cell C (contingent on cell-A outcome)
Harmony-474M student × Harmony-474M_v2 teacher (requires second pretraining run or a mid-training checkpoint from ZEB-137). Purpose: locate capacity ceiling at scaled student. Gate: only run if cell A attractor holds (Outcome A path).

### Cell D (contingent, lower priority)
Harmony-474M student × Mistral-7B teacher. Purpose: test whether scaled student overcomes cross-arch gap. Gate: only run if cell A attractor breaks (Outcome C path) and cell C has run.

**This spec commits to cell A + the reuse of cell B numbers.** Cells C and D are scoped separately as follow-up spec drafts once cell A produces a verdict.

---

## 5. Oracle extraction protocol

### 5.1 Teacher checkpoint selection
- **From ZEB-137:** first checkpoint that meets ZEB-137's health criteria (val-loss curve clean, step ≥ some TBD threshold — probably 5000+ steps so we have meaningful knowledge in the teacher).
- **Deliverable from ZEB-137:** the pinned checkpoint file (safetensors + optimizer pair), plus train.csv log, transferred to AVALON via USB SSD.

**Open question:** should we take the final step-7800 checkpoint, or an earlier checkpoint to give ZEB-138 earlier access? Tradeoff is teacher quality (later = better) vs wall-time savings (earlier = faster spinup).

### 5.2 Tokenizer parity invariant
`generate_oracle_table.py` hard-fails if teacher vocab ≠ 32000. Harmony-474M was pretrained with the Mistral v0.1 SentencePiece tokenizer (32000 vocab, confirmed in `ct87.prepare_data`), so parity holds. No corpus re-tokenization needed.

### 5.3 Layer selection
- Mistral (32 layers) used L-2 = L30 in ZEB-134
- TinyLlama (22 layers) used L-2 = L20 in ZEB-136
- Harmony-474M (24 layers): propose L22 (= L-2 by same convention)

**Open question:** should we also run L18 or L12 (earlier-layer) to see if earlier-layer content-sensitivity differs? Probably one cell first, revisit if attractor breaks.

### 5.4 Corpus
Reuse AVALON's 99M-token corpus from ZEB-136 (already tokenized, already on disk). This matches the scale at which TinyLlama's attractor was observed, keeps cell-A comparison apples-to-apples.

### 5.5 Output
- 10,000 rows × 128 dim (engram_dim default)
- `.safetensors` with tensor `engram.weight` shape `[10000, 128]`
- Saved to `/home/zebli/work/LOCAL/zeb138/oracles/harmony474_v1.safetensors` (or whatever AVALON conventions follow)

### 5.6 Health check thresholds
Pre-training gate: if any threshold is violated, pause and investigate before committing to the 4-cell matrix.

| Metric                                        | Threshold                              | Rationale                                       |
|-----------------------------------------------|----------------------------------------|-------------------------------------------------|
| `populated_fraction`                          | = 1.0                                  | Fewer populated rows = corpus too small         |
| `pca_explained_variance_ratio_total`          | ≥ 0.80                                 | Teacher hidden states should be PCA-compressible|
| Median pairwise `|cos|`                       | ≤ 0.25                                 | High cos = teacher's rows collapse to low-rank  |
| p90 pairwise `|cos|`                          | ≤ 0.50                                 | Tail-row redundancy check                       |

ZEB-136's TinyLlama numbers (0.934 PCA, 0.152 median |cos|) are the gold-standard bar. Harmony-474M may not clear it (much smaller teacher, less diverse representations) — acceptable as long as the table is above the minimums.

**Open question:** if Harmony-474M oracle health is significantly worse than Mistral/TinyLlama (e.g., PCA < 0.80), does that invalidate the cell-A comparison? Plausible answer: "it's still a valid test of same-arch vs cross-arch" but weakens the decisive-result claim. Worth a judgment call once we see the numbers.

---

## 6. Training protocol (4-cell sub-matrix)

Identical to ZEB-134/136's ι₂ + skip-router setup. Fields marked [same as ZEB-136] are verbatim.

### 6.1 Training config
- Student: 40M Harmony frozen backbone (layer-freeze mask [same as ZEB-136])
- Injection sites: L2, L5 [same as ZEB-136]
- Router: skip-to-logit with Tuned-Lens-style W_align [same as ZEB-134]
- Aux losses: ι₂ = Q-div + V-contrast [same as ZEB-136]
- Optimizer: AdamW (not Muon — engram module is the only thing training, small params)
- Steps: [same as ZEB-136] — whatever step count produced ZEB-136's clean numbers
- Seeds: 1 seed per sub-matrix cell for initial run; revisit if on-the-edge result

### 6.2 Shuffled oracle construction
`--alt-shuffle-seed 99` (different from training seed 42). Row-permutation is training-valid but not forensic-valid (see `project_zeb130_shuffle_kill.md` gotcha).

### 6.3 Logging
- Full train.csv log per cell
- Router fingerprint snapshot at end of training per cell
- All forensic probes ((Q-overlap), (V-rank), (X), Q-div, V-contrast) enabled [c96f3e7 has these]

---

## 7. Forensic suite

Same probe suite as ZEB-134/136, report per cell:

**Attractor fingerprint (primary):**
- `engram_logit_entropy` (final-step + trajectory)
- `log_alpha`, `α` (final-step)
- `‖W_align‖_F` (final-step)
- Cross-run engram-logit cosine (across cells 3 and 4 = router-on real and router-on shuf)

**Supporting forensics:**
- Within-run R (content-sensitivity per token) — expect ~0.09-0.15 per ZEB-136
- (X) primary-vs-alt |cos| — expect ≫ 0.07 floor (V is content-sensitive)
- Cross-run cos of V outputs

**LM measurements:**
- val_loss per cell, Δ-diff = cell 1 val_loss − cell 2 val_loss and cell 3 − cell 4
- val_loss trajectory (final 500 steps) per cell

---

## 8. Verdict matrix

| Outcome | Definition | Conclusion | Next step |
|---------|------------|------------|-----------|
| **A (attractor holds)** | All four thresholds inside attractor band | 40M student is capacity-bound; teacher architecture does not matter | File cell C (474M student × same-arch teacher) spec; locate capacity ceiling |
| **B (one threshold)** | Exactly one of `entropy`, `α`, `cross-run cos`, `Δ-diff` crosses | Soft signal — teacher-match partially helps | Extend training to 2× steps, diagnose which broke first; may reveal slow-converging routing |
| **C (two+ thresholds)** | Two or more threshold crossings, Δ-diff < 0.005 | Cross-architecture modality gap is the bottleneck | Pursue same-arch teacher as primary substrate; scope follow-up for architectural work |
| **D (Δ-diff ≥ 0.005)** | Strong LM-measurable signal | Major result — content-routing is viable at 40M with same-arch teacher | Expand to scaling + ablation study; publish |

**Auxiliary observation:** regardless of outcome, cell-A results get compared directly against cell-B (ZEB-134 Mistral) and ZEB-136 TinyLlama numbers. The three-teacher table is the publication-ready artifact.

---

## 9. Hardware + wall-time budget

### 9.1 Oracle extraction
- Teacher: Harmony-474M loaded on GPU (~1GB bf16 resident)
- Corpus: 99M tokens
- Est. wall time: **2-4 hours AVALON** (Harmony-474M is 14× smaller than Mistral-7B, extraction should be proportionally faster than ZEB-136's 5.5h)

### 9.2 4-cell training matrix
- Est. wall time: **40-60 min AVALON** (matches ZEB-136's observed 40 min)
- Plus ~10 min forensics, ~5 min findings doc

### 9.3 Total AVALON budget for cell A
**~3-5 hours end-to-end** once checkpoint is on disk.

### 9.4 KRILE allocation
Cell C (if it runs) is the 474M-student cell, requires 22GB for student + oracle + forensics. KRILE is the only machine with the headroom, but KRILE is pinned to ZEB-137 for ~8 days. Cell C gates on either (a) ZEB-137 completion, or (b) a ZEB-137 checkpoint early-enough + ZEB-138 delaying cell C until then.

---

## 10. Dependencies

- **Hard blocker:** ZEB-137 produces usable Harmony-474M checkpoint. [Pending, on KRILE]
- **Hard blocker:** `generate_oracle_table.py` validated end-to-end with a Harmony teacher. [Handoff to AVALON, see `AVALON-handoff-ZEB-138-oracle-validation.md` in this branch]
- **Soft prereq:** variance-floor measurement on ZEB-136 baseline (confirms Δ-diff ambiguity threshold). [Optional, can run in parallel on AVALON if cycles available]
- **Code on main at c96f3e7:** forensic (Q-overlap) + (V-rank) probes (#253); engram Q-div (#252); skip-to-logit + Tuned-Lens router (#244 from ZEB-134); 4-cell matrix runner (whatever script AVALON used in ZEB-136).

---

## 11. Open questions (resolve before kickoff)

1. **Teacher checkpoint step:** which ZEB-137 checkpoint to extract from? (final step-7800 vs earliest viable, e.g., step-5000). Tradeoff: teacher quality vs spinup time.
2. **Teacher layer index:** default is L22 (= L-2); consider also running L18 to probe earlier-layer content-sensitivity. Answer: single layer first, expand only if attractor breaks.
3. **Oracle corpus:** reuse ZEB-136's 99M-token corpus (recommended for apples-to-apples) or re-extract from a larger split. Recommend: reuse.
4. **Seed count:** 1 seed per cell for initial run, 3 seeds if on-the-edge result. Confirmed as standard.
5. **Cell B re-run:** reference ZEB-134 numbers only, or re-run on AVALON to control for hardware/environment drift? Recommend: reference-only unless numbers look suspicious.
6. **Cell C/D gating threshold:** what specifically triggers "cell C must run"? Attractor-holds is ambiguous — do we always run cell C after a confirmed capacity-ceiling, or only if we want to quantify the ceiling?
7. **Findings artifacts:** follow ZEB-136's precedent (single findings doc + branch push)? Confirmed.

---

## 12. Not in scope

- **Router architecture changes:** skip-to-logit with Tuned-Lens W_align is held fixed. Alternative routers (gated cross-attention, content-addressable) are separate issue trees (ZEB-117, ZEB-129).
- **Unfreezing the backbone:** if attractor holds, unfreezing is the obvious next step, but it's a major separate experiment with its own scoping.
- **Distillation-style pretraining:** if attractor breaks, scaling up with same-arch distillation is the natural path, but requires its own spec.
- **Harmony-474M student cells (C, D):** deferred until cell A produces a verdict; scope separately.

---

## 13. Self-review notes

- **Placeholder scan:** open questions are flagged in §11, all other sections are concrete. No TBDs outside explicit "Open question" blocks.
- **Internal consistency:** §3 thresholds align with §8 verdict matrix. §4 cell table matches §10 dependency table.
- **Scope:** single implementation (cell A + oracle extraction), testable independently, gates downstream cells. Correct spec scope.
- **Ambiguity:** §3 threshold #3 ("Cross-run cosine < 0.5") — ZEB-136 showed cross-run cos = 0.798, which is well inside the attractor band. The 0.5 threshold may be too tight; could revise after cell A if needed. Flagged as soft threshold.
