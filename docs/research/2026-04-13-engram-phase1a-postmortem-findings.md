# Engram Phase 1a Strategic Evaluation: Dimensionality, Retrieval Bottlenecks, and the Path to Phase 1b

**Source:** Gemini research report, 2026-04-13
**Prompt:** `docs/research/2026-04-13-engram-phase1a-postmortem-prompt.md`

## Summary

Phase 1a's identical validation loss curves (corpus table = random table = -0.49%) are the predictable result of:

1. **JL compression destroyed the signal (Hypothesis C):** Projecting 32K-dim frequency vectors to 128 dims via random projection violates the JL lower bound, obliterating semantic structure.
2. **xxhash can't exploit structure (Hypothesis D):** Deterministic hashing shatters semantic topology — related n-grams don't retrieve related entries.
3. **The -0.49% is a regularization artifact (Hypothesis A):** The Engram module's internal parameters (key/value projections, convolution, gate) act as a stochastic regularizer. The table content is irrelevant.
4. **Scale is NOT the bottleneck (Hypothesis B rejected):** Memorizing Transformer showed benefits at 40M params. 39.6M is sufficient.

## Key conclusions

- **Do NOT run Phase 1b (teacher distillation) under current xxhash architecture** — high-fidelity teacher embeddings will be retrieved pseudo-randomly, yielding another identical -0.49% curve.
- **Fix retrieval before fixing content** — the strategic order must be: prove semantic retrieval works, THEN optimize table content.
- **The PCA vs JL compression change alone may be significant** — PCA preserves top-128 variance directions; JL destroys them indiscriminately.

## Revised roadmap

### Phase 0: Diagnostic disambiguation

Confirm the regularization artifact hypothesis with cheap ablations:

1. **Null table test (highest priority):** Zero out all table entries, train from scratch. If -0.49% holds, table content is irrelevant.
2. **Correspondence test:** Shuffle corpus table rows randomly (breaking hash→content mapping). If -0.49% holds, the model doesn't care which vector it retrieves.
3. **Gate freeze test:** Replace learned sigmoid gate with a static constant. If -0.49% holds, no context-sensitive retrieval is occurring.

**Decision gate:** If ablations confirm artifact → proceed to Phase 1.

### Phase 0.5: Compression isolation (our addition)

**PCA corpus table + xxhash** — Same corpus frequency distributions, PCA-compressed instead of JL. Isolates compression variable from retrieval variable. If this beats -0.49%, the signal survived PCA and even xxhash can exploit structured content.

### Phase 1: Semantic retrieval integration

Replace xxhash with ANN (USearch/HNSW). Linear projection of hidden state → cosine similarity query → k=1 nearest neighbor. Populate with PCA-compressed corpus statistics.

**Decision gate:** If val loss > 0.7% improvement → proceed to Phase 2. If still at -0.49% → evaluate kill criteria.

### Phase 2: Teacher distillation injection

Populate validated ANN index with Mistral 7B embeddings compressed via PCA.

**Decision gate:** If val loss > 1.0% improvement → fully validated. Proceed to scaling.

### Phase 3: Scaling and mesh deployment

Scale to 474M params, expand index, integrate with mesh CAS storage.

## Kill criteria

1. **ANN failure:** If Phase 1 (semantic retrieval) fails to exceed -0.49% after hyperparameter tuning → the gated residual injection mechanism is fundamentally incompatible with external memory.
2. **Attention pivot:** If gated residual fails, pivot to cross-attention integration (concatenate retrieved vectors to K/V cache instead of adding to residual stream).

## Full report

The complete Gemini analysis with mathematical derivations, literature citations, and detailed experimental protocols is included below.

---

## Introduction: The External Memory Paradigm and the Harmony Architecture

The integration of external, non-parametric memory into large language models represents a critical frontier in artificial intelligence, particularly for decentralized and edge-computing applications. The Harmony platform, designed around an on-device language model (474M parameters, Qwen3-derived), seeks to leverage this paradigm through the Engram system — an external conditional memory mechanism designed to inject context-dependent embeddings directly into the transformer's hidden state via a gated residual connection.

The recent conclusion of Phase 1a experiments on the config=tiny 39.6M parameter variant has yielded an empirical anomaly: the corpus co-occurrence table and the random table produced identical validation loss trajectories, both at -0.49% improvement. The curves tracked identically from step 1, and the gating mechanism did not collapse to zero in either condition.

## Part I: Exhaustive Diagnosis

The identical performance is explained by a superposition of extreme dimensionality collapse (Hypothesis C), the disruption of semantic topology by deterministic hashing (Hypothesis D), and the network's capacity to repurpose arbitrary architectural parameters for regularization (Hypothesis A).

### The Mathematical Failure of JL Compression (Hypothesis C)

The JL lemma requires target dimensionality k proportional to O(log(n)/epsilon^2). Projecting 32,000-dimensional vocabulary distributions to 128 dimensions severely violates this bound. Unlike PCA (which preserves maximum-variance directions), random projection indiscriminately scatters data, obliterating semantic clusters.

The resulting 128-dimensional vectors were mathematically indistinguishable from random noise. The signal was eliminated before the model ever saw it.

### Topological Incompatibility of xxhash (Hypothesis D)

xxhash enforces a strict avalanche criterion — semantically identical n-grams ("the cat" vs "a cat") produce completely divergent indices. This prevents gradient-based optimization from learning a generalized retrieval strategy. The model is forced to memorize rigid, arbitrary n-gram→index mappings without generalization.

### Gated Residual as Stochastic Regularizer (Hypothesis A)

The Engram module's internal parameters (key/value projections, depthwise convolution, sigmoid gate) learn to inject managed, low-magnitude noise into the residual stream. This smooths the optimization landscape, yielding a consistent -0.49% improvement decoupled from table content.

### Scale Is Not the Bottleneck (Hypothesis B — Rejected)

The Memorizing Transformer demonstrated definitive perplexity improvements at 40M parameters. Sub-100M parameter models benefit from retrieval as a direct substitute for limited parametric capacity. Testing at 39.6M is valid; scaling up before fixing retrieval would obscure architectural flaws.

## Part II: Diagnostic Ablations

### Null Table Test

Zero out all table entries. If -0.49% holds → table content is entirely irrelevant, gains are purely parametric.

### Correspondence Test

Shuffle corpus table rows randomly. If -0.49% holds → hash-to-content mapping is unexploited.

### Gate Freeze Test

Replace learned gate with static constant (e.g., 0.01). If -0.49% holds → no context-sensitive retrieval is occurring.

## Part III: The ANN Transition

Replace xxhash with similarity-based ANN lookup (USearch/HNSW). Generate continuous query vectors via linear projection of hidden state. For 10K entries, HNSW introduces negligible latency overhead on edge devices.

Minimal viability test: linear projection of hidden state → 128-dim query → k=1 USearch lookup over 1K-5K entries with PCA-compressed centroids. Target: break the -0.49% ceiling.

## Part IV: Teacher Distillation Reassessment

Phase 1a's failure does NOT invalidate teacher distillation. The PCA compression change alone could explain a massive difference (preserving top-128 principal components vs random projection). Teacher embeddings encode deep polysemous contextual relationships from trillions of tokens.

However, teacher distillation will fail under xxhash retrieval (Hypothesis D). The retrieval bottleneck must be resolved first.

## Part V: Scaling Dynamics

The 39.6M parameter model possesses sufficient capacity to learn retrieval. The high parameter-to-signal ratio at this scale amplifies the regularization artifact, but does not prevent genuine retrieval learning if the signal is preserved and addressable.

## References

- Memorizing Transformer (Wu et al. 2022): kNN memory benefits at 40M-800M scale
- kNN-LM (Khandelwal et al.): Token-level datastore retrieval
- RETRO (Borgeaud et al. 2022): Chunked cross-attention, retrieval-augmented
- HASH-RAG: Deep hashing bridging semantic retrieval and binary efficiency
- MiniCPM: Sub-100M "wind tunnel" models predicting scaling laws
- Multi-Sense Embeddings (ACL 2025): Teacher distillation preserving polysemous representations
