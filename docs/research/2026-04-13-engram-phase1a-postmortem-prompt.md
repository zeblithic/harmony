# Engram Phase 1a Post-Mortem: Should We Proceed with Teacher Distillation?

## Context for the researcher

We're building **Harmony**, a decentralized mesh computing platform with an on-device language model (474M parameter, Qwen3-derived). The model includes an **Engram** system — an external conditional memory that injects context-dependent embeddings into the transformer hidden state via a gated residual mechanism.

This is a follow-up to our previous research session (2026-04-12). We've now completed Phase 1a experiments and need to reevaluate our strategy before committing GPU time to Phase 1b.

### Architecture recap

- **Engram table:** Flat embedding table `[10000, 128]` stored on disk/mesh
- **Key generation:** N-gram tokens (bigrams/trigrams) hashed via xxhash64 with 4 seeds (multi-head), producing table indices
- **Injection:** Retrieved embeddings pass through a gated residual module (key/value projection, sigmoid dot-product gate, causal depthwise conv1d, SiLU) and are added to the hidden state
- **Training config:** config=tiny (8-layer, 512-hidden, 39.6M params), FineWeb-Edu-POC dataset, seq_len=2048, bf16, Muon optimizer, 10k steps on RTX 4090

### Complete experiment history

**Round 1 — Projection keying experiments (6 experiments, all failed):**

All 6 experiments attempted to replace xxhash keys with learned latent projection keys. Every approach failed — the model zeroed out the engram gate rather than adapt to projection-based retrieval. Root cause identified: the table contains random embeddings, so projection keys that retrieve "semantically nearby" entries still get random noise.

**Round 2 — Table content experiments (Phase 1a, just completed):**

Based on your previous research recommendations, we tested whether real corpus content in the table improves over random:

| Experiment | Table content | Key mechanism | Val loss delta | Result |
|-----------|--------------|---------------|---------------|--------|
| Random table (baseline) | Random synthetic embeddings | xxhash | -0.49% | Works |
| Corpus co-occurrence | Next-token frequency distributions, JL-projected | xxhash (same) | -0.49% | **No improvement** |

**The corpus table tracked identically to the random table through 6k steps.** Not slightly worse, not slightly better — identical curves.

### What the corpus table contained

For each of the 10,000 table entries:
1. Scanned training corpus for all bigrams/trigrams that hash to that entry's index
2. Built a `[vocab_size]` frequency distribution of the next token following those n-grams
3. Compressed the distribution from `[32000]` to `[128]` via Johnson-Lindenstrauss random projection
4. L2-normalized

Each entry averages ~20K n-gram contributions from a 100M-token corpus. The frequency distributions are real signal — they encode what tokens tend to follow specific n-gram patterns.

### What was supposed to happen next

Our previous roadmap (based on your recommendations) said:

- **Phase 1b:** Teacher distillation table — replace the JL random projection with Mistral 7B's embedding matrix + PCA. Same corpus frequency distributions, but projected through a semantically structured space instead of random.
- **Decision gate:** If either 1a or 1b shows >1% val loss improvement → Phase 2 (USearch ANN retrieval). If both show ~0.5% → consider Phase 3 (advanced architectures).

Phase 1a showed 0% improvement over random, which was not one of the anticipated outcomes in the decision gate.

## The question

**Should we still run Phase 1b (teacher distillation), or does Phase 1a's failure change the strategic picture?**

## Evidence that needs interpretation

### 1. The -0.49% mystery

The random table gives a consistent -0.49% val loss improvement. The corpus table gives the same -0.49%. Two hypotheses:

**Hypothesis A — Regularization artifact:** The gated residual module adds a learned perturbation to the hidden state. The table's actual content doesn't matter because the model is using the gate as a regularizer (dropout-like effect) or as additional parameters (the key/value projections learn something useful regardless of what they project). The -0.49% is a parameter-count effect, not a retrieval effect.

**Hypothesis B — Content doesn't matter yet at this scale:** The 39.6M-param model doesn't have enough capacity to learn a useful retrieval strategy. The gated residual adds value (the -0.49%), but the model can't discriminate between useful and useless table content at this parameter count. Teacher distillation might still work at the full 474M scale.

**Hypothesis C — The compression destroyed the signal:** The JL projection from `[32000]` to `[128]` may have destroyed the useful structure in the frequency distributions. The corpus table IS better than random in the high-dimensional space, but after random projection to 128 dims, the difference is lost. Teacher distillation (which uses PCA instead of random projection) would preserve the most informative directions.

**Hypothesis D — xxhash lookup can't exploit structure:** Even if the table has structure, xxhash maps n-grams to entries pseudo-randomly. Two semantically related n-grams won't hash to nearby entries. The table has structure, but the lookup mechanism can't exploit it. This would mean teacher distillation also fails (same xxhash keys), and the path forward requires changing the retrieval mechanism first.

### 2. Identical curves, not just similar endpoints

The corpus and random tables didn't just converge to the same final loss — they tracked identically from the start. This seems to rule out "the corpus table is better but the model needs longer to learn to use it." The model sees no difference from step 1.

### 3. The gating learned something in both cases

In both random and corpus experiments, the gate did NOT zero out (unlike the projection experiments). The sigmoid gate learned non-trivial values and the -0.49% improvement was real and consistent. The engram module IS contributing — just apparently content-independently.

## Research questions

### 1. Which hypothesis is most likely?

Given the identical training curves (not just endpoints), which of hypotheses A-D best explains the data? Are there other hypotheses we're not considering?

If the answer is "we can't tell from this data alone," what's the cheapest diagnostic experiment to disambiguate?

### 2. Diagnostic experiments before committing to Phase 1b

Before spending a day of GPU time on teacher distillation + training, are there cheap experiments (minutes, not hours) that would tell us whether it's worth trying?

Ideas we're considering:
- **Gate ablation:** Replace the learned gate with a constant (e.g., 0.01) and train. If val loss improvement is similar, the model is using the gate as extra parameters, not as a retrieval mechanism.
- **Table ablation:** Train with the engram module's weights but feed it zeros instead of table lookups. If similar improvement, the table is irrelevant — the module's internal projections are doing the work.
- **Per-layer gate logging:** Log the mean gate activation magnitude per layer during training. If the gate is large and content-insensitive, it's a parameter effect. If it's small and variable, the model may be doing real retrieval but the signal is weak.
- **Shuffle test:** Randomly permute the corpus table's rows after generation (breaking the hash→content correspondence) and train. If same result as ordered corpus table, the content-to-key mapping doesn't matter — just having non-random row vectors doesn't help.

Which of these would be most informative? Are there better diagnostics?

### 3. Does the corpus table failure change the probability for teacher distillation?

Your previous assessment ranked teacher distillation + USearch as "highest probability of success." Given that corpus statistics (real signal from the training data) showed zero improvement over random:

- Does this change your confidence in teacher distillation working?
- The teacher distillation spec changes two things simultaneously: (a) the table content (probability-weighted centroids in teacher embedding space) and (b) the compression method (PCA preserving top-128 variance directions instead of random JL projection). Could (b) alone explain a potential difference?
- Is it worth isolating these variables — e.g., test PCA-compressed corpus statistics before adding teacher embeddings?

### 4. Is the retrieval mechanism the actual bottleneck?

The previous research session focused on table content because the projection keying experiments all failed. But maybe we drew the wrong lesson. Consider:

- xxhash is deterministic but semantically arbitrary — related n-grams don't hash to related entries
- The corpus table has structure, but xxhash can't exploit it (Hypothesis D)
- Maybe the path forward is: (1) fix the retrieval mechanism first (similarity-based lookup), THEN (2) populate the table with structured content

If this is the right order, what's the minimal experiment to test it? Could we:
- Use a tiny USearch index (even 1000 entries) with teacher embeddings
- Replace xxhash with approximate nearest neighbor lookup on the n-gram embedding
- Test whether *any* improvement appears over the hash-based approach

### 5. Scale effects — is 39.6M too small?

The config=tiny model is 39.6M parameters. The production target is 474M (12x larger). At what parameter count does the literature suggest retrieval-augmented models start showing retrieval benefits?

- kNN-LM showed benefits at 247M+
- RETRO showed benefits at 172M+
- Memorizing Transformers at 8.6M+

Is 39.6M below the threshold where the model can learn to USE external memory effectively? Should we test at config=small or config=medium before concluding the approach doesn't work?

### 6. Alternative framings we might be missing

Given everything we now know (8 experiments, all showing the same -0.49% regardless of approach):

- Is -0.49% actually a reasonable ceiling for external memory at this scale, and we should declare success and move on to scaling?
- Are there approaches to external memory for small LMs (<100M) that we haven't considered?
- Could the engram module architecture itself be the bottleneck? (e.g., injection point, gating mechanism, number of heads)
- Is there a way to make the gated residual content-sensitive without changing the retrieval mechanism — e.g., by training the gate to attend to table content quality?

## What we're looking for

1. **Diagnosis:** Which hypothesis best explains the identical curves? Confidence level?
2. **Cheapest next experiment:** What single experiment gives us the most information about whether to continue this line of research?
3. **Updated strategy ranking:** Given Phase 1a results, re-rank the approaches from your previous report. Has teacher distillation dropped in probability? Has something else moved up?
4. **Scale guidance:** Should we be testing at a larger model size before drawing conclusions about the engram mechanism?
5. **Kill criteria:** At what point should we accept that the gated residual engram at this architecture isn't going to work for knowledge retrieval and pivot to a fundamentally different approach?
