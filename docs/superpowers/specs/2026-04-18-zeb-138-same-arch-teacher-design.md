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

**Mechanism (2026-04-18):** The Gemini Deep Research report (`docs/research/2026-04-18-post-zeb138-path-research-findings.md`) establishes the attractor as **gradient-descent autonomously inventing label smoothing**. When the router signal is uninformative, driving `P_router → uniform` minimizes expected CE-loss contribution, and the mixing equation structurally becomes `P_final = (1-α)·P_LM + α·uniform` — the exact formula for label smoothing. α = 0.17 is the empirically optimal smoothing factor for FineWeb-Edu under our base LM's calibration; it is corpus + base-LM dependent, not teacher dependent. This matches the observed teacher-invariance.

Under this mechanism, the ZEB-138 question is precisely: **does same-arch teacher make the router signal informative to the frozen 40M student?** If yes, any non-uniform output beats uniform CE, and gradient descent will vacate the attractor. If no (for any reason), uniform remains optimal and the attractor holds.

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

**Δ-diff alone is insufficient** (too noisy at +0.0002 to distinguish small signal). The attractor-fingerprint gives five orthogonal views; any one threshold crossed is positive signal. All five must hold within tight bands for a clean null.

| # | Threshold                                                 | Crossed meaning                                                    |
|---|-----------------------------------------------------------|--------------------------------------------------------------------|
| 1 | `engram_logit_entropy < log(vocab) − 0.1`                 | Router emits content-sensitive (non-uniform) distributions         |
| 2 | `α` outside [0.14, 0.20]                                  | No longer the dataset-optimal label-smoothing factor              |
| 3 | Cross-run cosine of engram-logit outputs < 0.7            | Outputs are content-driven across RNG seeds, not shared baseline   |
| 4 | Δ-diff ≥ +0.001 nats                                      | Content-routing produces LM-measurable effect                      |
| 5 | `W_align` Frobenius drift > 2× init                       | W_align received and used non-trivial gradients (structural vs. capacity discriminator — see §7.2 and §8) |

**Decision rules:**
- All five inside attractor band → attractor holds → further disambiguation needed (see §8 Outcomes A, C)
- One threshold crossed → soft signal → extend training / investigate which broke first (§8 Outcome B)
- Two or more crossed → attractor broken → modality gap is the bottleneck (§8 Outcome B-strong)
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
- **From ZEB-137:** the **final step-7800 checkpoint** (full pretraining run). Earlier checkpoints are not used — the wall-time savings don't justify the loss of teacher quality, and high-confidence teacher representation is load-bearing for the cell-A decisive verdict.
- **Deliverable from ZEB-137:** the final `model_step_7800.safetensors` (+ its optimizer pair, though optimizer state isn't used here), plus train.csv log for teacher-health context, transferred to AVALON via USB SSD.

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

Same probe suite as ZEB-134/136, with one addition in §7.2.

### 7.1 Attractor fingerprint (primary)
- `engram_logit_entropy` (final-step + full trajectory, logged every N steps)
- `log_alpha`, `α` (final-step + trajectory)
- `‖W_align‖_F` (final-step + trajectory — see §7.2)
- Cross-run engram-logit cosine (across cells 3 and 4 = router-on real and router-on shuf)

### 7.2 W_align trajectory logging (NEW, 2026-04-18)

**Why:** under attractor-holds, we cannot distinguish "40M capacity ceiling" (Outcome A) from "linear W_align is structurally too narrow" (Outcome C) without seeing whether W_align actually absorbed gradient signal during training. This is the A-vs-C discriminator.

**Log per training step (or every 50 steps, whichever is cheaper):**
- `‖W_align‖_F` (Frobenius norm)
- `‖W_align‖_F / ‖W_align_init‖_F` (drift ratio)
- `‖∇W_align‖_F` (gradient norm on the alignment matrix)
- `rank_effective(W_align)` via SVD tail ratio (computed every 500 steps is fine — more expensive)

**Expected patterns:**
- **Outcome A (capacity ceiling):** `W_align` drift > 2× init, gradients flow throughout training, rank stable. Signal reached the alignment matrix but couldn't be used.
- **Outcome C (structural ceiling):** `W_align` drift ≈ 1× init, gradients suppressed to near-zero within first 1000 steps, rank collapses. The linear form itself is the ceiling — no signal got absorbed.

**Supporting forensics:**
- Within-run R (content-sensitivity per token) — expect ~0.09-0.15 per ZEB-136
- (X) primary-vs-alt |cos| — expect ≫ 0.07 floor (V is content-sensitive)
- Cross-run cos of V outputs

**LM measurements:**
- val_loss per cell, Δ-diff = cell 1 val_loss − cell 2 val_loss and cell 3 − cell 4
- val_loss trajectory (final 500 steps) per cell

---

## 8. Verdict matrix

Restructured 2026-04-18 to distinguish capacity-ceiling (A) from structural-linear-insufficiency (C) under the attractor-holds branch, per Gemini steelman. The key discriminator is `W_align` Frobenius drift (§7.2).

| Outcome | Definition | Conclusion | Next step |
|---------|------------|------------|-----------|
| **A (attractor holds + W_align drift > 2× init)** | All five thresholds inside band; W_align absorbed gradients but student couldn't use them | 40M student is capacity-bound; teacher architecture does not matter; W_align was not the ceiling | Unfreeze top 2 layers (Gemini Path-A #1) or apply LoRA to v_proj/o_proj (Gemini Path-A #2); or scale student to 474M |
| **B (attractor breaks, one threshold)** | Exactly one threshold crosses | Soft signal — teacher-match partially helps | Extend training to 2× steps, diagnose which broke first |
| **B-strong (attractor breaks, two+ thresholds)** | Two or more threshold crossings, Δ-diff < 0.005 | Cross-architecture modality gap is the bottleneck | Pursue same-arch teacher as primary substrate; switch training objective to Memory-Decoder-style KL + CE (Gemini Path-B #1) |
| **C (attractor holds + W_align drift ≈ init)** | All five thresholds inside band; W_align near-init, gradients suppressed | **Linear W_align is structurally insufficient** regardless of student capacity; PCA-128d → linear → 32k-vocab pipeline is mathematically too narrow | Replace W_align with multi-layer non-linear module (Memory-Decoder-style auxiliary decoder); unfreezing / scaling student won't help |
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

1. ~~**Teacher checkpoint step:** which ZEB-137 checkpoint to extract from?~~ **Resolved 2026-04-18:** final step-7800 checkpoint. Teacher quality prioritized over spinup time. See §5.1.
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
- **Threshold calibration:** §3 threshold #3 was tightened from the initial 0.5 to **0.7** (2026-04-18) after observing that ZEB-136's cross-run cos = 0.798 sits comfortably in the attractor band. 0.7 is conservative enough that crossing it represents real content-driven variance, not noise. A looser 0.5 would have been easy to rationalize post-hoc, which is exactly the failure mode to avoid.
- **Verdict matrix restructuring (2026-04-18):** the original A/B/C/D structure conflated "capacity ceiling" with "structural W_align ceiling." Post-Gemini, the verdict matrix now separates capacity-A (W_align drift > 2× init) from structural-C (W_align near-init), using the `W_align` Frobenius trajectory logged under §7.2. This lets a single ZEB-138 run resolve both failure modes instead of requiring a follow-up experiment to disambiguate.
- **Attractor framing (2026-04-18):** §2 now frames the attractor as emergent label smoothing (mechanism documented in Gemini findings doc). This sharpens the ZEB-138 question from "does teacher-arch matter?" to "does same-arch teacher make the router signal *informative*?" — a precise mathematical criterion rather than a phenomenological observation.
