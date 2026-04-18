# Post-ZEB-138 Decision-Path Research — Gemini Deep Research Prompt

**Date:** 2026-04-18
**Purpose:** Pre-position architectural and methodological options for both outcome branches of ZEB-138 (attractor-break = modality-gap confirmed → distillation-style path; attractor-holds = capacity-ceiling confirmed → scaling path), so that when ZEB-138 returns a verdict in ~2 weeks, the follow-up work is already scoped and prior-art-grounded.
**Parent issue:** [ZEB-102](https://linear.app/zeblith/issue/ZEB-102) (Engram content-routing research)
**Companion issue:** [ZEB-138](https://linear.app/zeblith/issue/ZEB-138) — the decisive same-arch-teacher experiment this research anticipates
**Prior Gemini report:** `2026-04-17-engram-lm-blindness-cross-layer-research-prompt.md` (the ι₂-era LM-blindness prompt that informed ZEB-133/134/135). This current prompt is the sequel: those three experiments returned null, confirming capacity-boundedness at 40M × cross-arch teacher; this prompt targets what comes after ZEB-138 resolves the same-arch confound.

---

## Research prompt (copy-paste ready for Gemini Deep Research)

I am running a pretraining-research program investigating whether a small transformer can route retrieved content from an external oracle memory into its language-modeling output. After a batch of seven architectural iterations produced forensically-clean injection but LM-invariant output, we ran three orthogonal diagnostic experiments (layer-wise content decay, skip-to-logit router with Tuned-Lens alignment, EMA baseline subtraction). All three produced Δ-diff within noise (+0.0003 nats). We then replicated the skip-to-logit router across a second teacher (TinyLlama-1.1B, 28× capacity gap vs Mistral-7B's 175×). The two teacher configurations produced **identical router fingerprints within measurement noise**, including an exact max-entropy attractor (`engram_logit_entropy = log(vocab)` to machine precision) and α = 0.17 ± 0.005.

The final untested confound is **teacher-student architecture match**. We are now pretraining a 474M Harmony-architecture model (same architecture as the 40M student, 12× capacity gap) to serve as a same-arch teacher. The ZEB-138 experiment runs the same skip-to-logit router with this same-arch teacher; outcome is expected in ~2 weeks. The question this research prompt anticipates: **what follow-up work becomes viable depending on which outcome lands?**

This report should surface prior art and concrete methodological recommendations for both branches, so implementation can begin within a day of the ZEB-138 result.

### System under study

- **Student:** 40M-parameter decoder-only transformer (8 layers, 512 hidden, 8 heads, RoPE, RMSNorm, GQA 8/4). Pretrained on FineWeb-Edu-POC for ~800M tokens. **Frozen during engram experiments.**
- **Teacher options tested so far:** Mistral-7B-v0.1, TinyLlama-1.1B. Next: Harmony-474M (same-arch, from ZEB-137 pretraining).
- **Router architecture:** skip-to-logit from layer 5 post-injection, through a learned Tuned-Lens-style `W_align` matrix, producing `engram_logits` of shape `[batch, seq, vocab=32000]`. These are mixed with frozen LM logits via a learned `α` gate.
- **Oracle:** 10,000-row table keyed by xxhash of input-context bigrams/trigrams, populated with teacher hidden states from layer L-2, PCA-reduced to engram_dim=128.
- **Training:** ι₂ aux losses (Q-diversity MoE load-balancing + V-contrast InfoNCE) on top of LM cross-entropy. Only the engram-adjacent modules (Q/K/V projections, W_align, α) train.
- **Shuffle control:** every experiment pairs real-oracle training with a row-permutation-shuffled oracle (training-valid but retrieval-value-preserved — a separate finding we documented). Shuffled-oracle training produces Δ-diff ≈ 0 from real-oracle training.

### The attractor observation (central datum)

| Fingerprint                      | ZEB-134 Mistral-7B | ZEB-136 TinyLlama-1.1B | Agreement |
|----------------------------------|---------------------|-------------------------|-----------|
| `engram_logit_entropy`           | log(32000) = 10.373 | log(32000) = 10.373     | exact     |
| `α` (mixing weight)              | 0.17                | 0.164                   | 3.5%      |
| `‖W_align‖_F`                    | 1.96                | 1.91                    | 2.5%      |
| Δ-diff (real − shuf)             | ~0.0002             | +0.00022 / +0.00024     | noise     |
| Cross-run cos (two seeds)        | ~0.80 (KRILE)       | 0.798 (AVALON)          | exact     |

The router converges to **max-entropy uniform output**. Gradient descent finds "emit uniform, let α balance noise against LM logits" because it cannot produce content-sensitive distributions that beat uniform cross-entropy. Two teachers, two scales, same lineage (both LLaMA-family): identical attractor. ZEB-138 tests whether a same-arch teacher (Harmony → Harmony) breaks this.

### Two outcome paths for ZEB-138

**Path A — attractor holds under same-arch teacher.** The 40M frozen backbone's capacity is the real bottleneck. Teacher choice is immaterial. To make progress, we must either scale the student, unfreeze the backbone, or both.

**Path B — attractor breaks under same-arch teacher.** The cross-architecture modality gap was the real bottleneck at 40M. Same-architecture oracle features are decodable by the 40M student's LM head. This opens distillation-style pretraining and same-arch retrieval as viable pathways, but we need to understand what scale of distillation work succeeds at what student sizes.

### What I want from this research report

#### 1. Same-architecture distillation literature (for Path B)

If the attractor breaks with same-arch teacher at 40M, the natural next experiments involve same-architecture knowledge distillation. Please survey:

- **MiniLM, DistilBERT, DistilGPT family:** what are the consistent methodological choices across these works? Same-architecture teacher/student (what exact relationship?), layer-alignment losses, attention-transfer losses, hidden-state-matching losses. Which of these scale down to our 40M / 474M setup?
- **Student-scale curves in distillation:** where is the empirical inflection point at which same-arch distillation starts to work? Is there a minimum student scale below which even same-arch teacher signal cannot be absorbed?
- **Recent distillation scaling laws (2024-2025):** has anyone published Chinchilla-style scaling curves for distillation — i.e., tokens-per-parameter laws where the teacher is a same-family larger model? The core question: given a same-arch teacher at 10× the student's scale, how many distillation tokens does the student need for loss-improvement to cross X nats?
- **Retrieval-based distillation specifically:** is there prior work where the distillation signal is delivered via retrieved embeddings rather than direct logit/hidden matching? The distinction matters because our setup is retrieval-mediated, not continuous-signal distillation.

For each finding: (a) empirical results at scales closest to our 40M / 474M, (b) what the distillation objective looked like concretely (formula, not prose), (c) was the student's backbone frozen or trainable during distillation.

#### 2. Frozen-backbone capacity ceilings (for Path A)

If the attractor holds even with same-arch teacher, we're capacity-bound at 40M regardless of teacher quality. The key question becomes: **what student scale is necessary** for content-sensitive routing to become LM-measurable, assuming the routing mechanism is valid?

- **Frozen-backbone + auxiliary-module training literature:** recent work on LoRA-style, adapter-style, or side-tuning approaches where the main backbone is frozen and a small trainable module attempts to inject new behavior. What scales work? Where does it fail?
- **kNN-LM / Memory-Decoder / RETRO scale curves:** at what backbone size does output-side retrieval start producing measurable LM gains on natural text? I recall the original Memory Decoder paper reported useful gains at ~1B scale; is there anything below that?
- **Attention-head and MLP width scaling for "absorbing" external signal:** mechanistic interpretability has documented that specific circuits (induction heads, copy-heads, etc.) appear at specific scale thresholds. Is there work on what scale is needed for "retrieval-routing" circuits specifically?
- **The narrower question:** if I want a student that can demonstrably route a same-arch teacher's retrieved features into its LM output, and I'm limited to ≤$1000 cloud compute and commodity GPUs (one 4090 available), what's the smallest student scale that prior work suggests would work?

#### 3. Retrieval-augmented pretraining vs retrieval-augmented fine-tuning

Our current setup is retrieval-augmented **training** (retrieval signal present throughout pretraining, not bolted on). Most published retrieval-LM work integrates retrieval at fine-tune or inference time on an already-pretrained base. Please surface:

- Prior work that actually pretrains the student with retrieval in the loop from step 0 (Memory Decoder does something like this; what else?).
- Evidence for or against the hypothesis that retrieval-from-step-0 pretraining is *easier* than retrieval-bolted-onto-pretrained, because the student's representation learns to be retrieval-compatible.
- If the literature suggests bolted-on retrieval is hopeless below some scale but retrieval-from-step-0 succeeds at smaller scale, that's actionable — we could restart with a same-arch oracle from step 0 at the 474M scale.

#### 4. The attractor phenomenon itself

The exact max-entropy attractor observation (`engram_logit_entropy = log(vocab)` to machine precision, α = 0.17 stable across teachers) is striking. I haven't found literature that documents this specific failure mode. Please search:

- **Router / gated-fusion literature:** any prior work reporting a learned-router converging to "emit uniform + balance alpha" as a degenerate optimum? This would be the generic form of our finding.
- **Mixture-of-experts degeneracy:** MoE literature has documented router collapse (all routing to one expert), but the *opposite* failure — router output becoming uniform across experts/vocab — is less commonly discussed. Any prior art on uniform-output attractors in gated architectures?
- **Output-space distillation failures:** when a distilled student's output converges to uniform-over-vocab rather than matching teacher distribution, what is the diagnosed cause in the literature?
- **Theoretical framing:** is there a principled reason (from optimization theory or information geometry) why a mixing-gate α and a uniform-output router would form a stable attractor when the input signal is uninformative? The answer would help us understand whether this attractor is a universal property of skip-to-logit routers with any uninformative signal, or specific to our setup.

#### 5. Concrete recommendations contingent on outcome

Given ZEB-138 will resolve in ~2 weeks, I want a decision tree:

**If attractor holds (Path A):**
- What are the top 2-3 experiments to run with a 474M student, same-arch teacher, frozen 474M backbone?
- What's the minimum viable retrainable parameter budget (LoRA rank, adapter width) that prior work suggests would work at 474M?
- What's the strongest prior-art case for "unfreeze the backbone" at our scale budget?

**If attractor breaks (Path B):**
- Top 2-3 distillation approaches appropriate for same-arch 40M student + same-arch 474M teacher, given we've already proven the retrieval-routing path works (by hypothesis).
- Is there an argument for abandoning retrieval entirely and moving to direct distillation (logit-match, hidden-match)? What would prior work suggest gives better compute-efficiency?
- If we scale the student to 474M and retain retrieval, what scale of teacher becomes necessary for the capacity-gap-to-matter-again threshold?

Rank-order 3-5 concrete experiments across both paths. For each recommendation: (a) what the modification looks like as a PyTorch module diff or training-loop change, (b) what the decisive forensic reading would be, (c) estimated compute cost on one RTX 4090.

#### 6. Steelman against the research thread

What's the strongest argument that this entire line of research (same-arch teacher, 40M or 474M student, retrieval-based routing) is a dead end worth abandoning? What alternative would stronger prior art recommend instead? The specific failure modes that would make this dead-on-arrival:

- If same-arch distillation at small scale has been repeatedly shown to fail.
- If retrieval-augmented pretraining has consistently been shown to need teacher-student scale ratios we can't afford.
- If the attractor phenomenon is documented as universal and unsolvable under any choice of gate architecture.

---

## Output format requested

1. **Executive summary (under 500 words):**
   - What's the top-1 recommended follow-up if attractor holds (Path A)?
   - What's the top-1 recommended follow-up if attractor breaks (Path B)?
   - What's the single strongest counter-argument to the whole line of work?

2. **Literature survey by section**, per research question above, with primary-source citations (not secondary surveys).

3. **Scale-curves table** if available: student-size × teacher-size × distillation-success-threshold for documented works. Columns: student params, teacher params, training tokens, method, reported effect size, citation. Sparse is fine — just what the literature actually contains.

4. **Attractor-phenomenon analysis**: §4 deserves its own short section. If no direct prior art, explicitly say so and provide theoretical framing (optimization theory / information-geometric reasoning about why max-entropy + fixed α would form a stable fixed point under uninformative input).

5. **Ranked recommendations** — 3-5 experiments total across Paths A and B, ordered by expected information gain per compute-dollar.

6. **Counter-argument section** — the steelman (§6).

7. **Bibliography with arXiv / primary links.** Prefer 2023+ sources unless the foundational work is older.

### Prioritization under token limits

If Gemini Deep Research needs to trim for token / time reasons, the section priority order is:

1. §1 (distillation lit for Path B) — most likely to be actionable
2. §5 (concrete recommendations) — decision-tree is what we'll act on
3. §4 (attractor phenomenon) — if this has a named counterpart in the literature, that's load-bearing for interpretation
4. §2 (scale ceilings for Path A)
5. §3 (retrieval-augmented pretraining)
6. §6 (steelman)

---

## How I'll use this report

Once ZEB-138 returns a verdict (cell A outcome), I'll read the corresponding path's section (A or B), cross-reference against the attractor analysis, and the 3-5 ranked recommendations become candidate issues. Expectation is ~1 issue per top-ranked recommendation filed the same week as ZEB-138 closes.

If Path B wins, this report will also inform the brainstorming for "retrieval-augmented same-arch distillation pretraining at 474M+" which would be a multi-week project I'd want to scope carefully before starting.

If Path A wins, this report helps quantify the gap between our budget (≤$1000 cloud compute, commodity GPUs, 474M ceiling absent much more data) and what the literature suggests is minimum viable. If that gap is large, this report's steelman section (§6) becomes load-bearing for "do we pivot axes entirely or scale down the ambition."
