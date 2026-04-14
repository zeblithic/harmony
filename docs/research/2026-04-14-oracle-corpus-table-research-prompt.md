# Oracle Corpus Table Diagnostic — Gemini Deep Research Prompt

**Date:** 2026-04-14
**Context:** Drafted mid-ZEB-117 bake-off to inform the next research project regardless of outcome.
**Parent:** `2026-04-14-engram-injection-mechanism-findings.md`, `2026-04-14-model-delta-cross-attention-scaffold.md`

---

## Research prompt (copy-paste ready for Gemini Deep Research)

I am running a controlled experiment to determine whether retrieval from an
external corpus memory table can transmit useful signal to a small
transformer during pretraining. My architecture, results so far, and the
diagnostic I want designed are below. Please produce a thorough research
report covering methodology, literature precedents, failure modes, and
concrete engineering recommendations.

### System under study

- Architecture: 40M-parameter decoder-only transformer (8 layers, 512
  hidden, 8 heads, FFN ratio 4.0, GQA 8/4, RoPE, RMSNorm, BlockAttnRes
  for depth-wise context mixing). Trained on FineWeb-Edu-POC,
  Chinchilla-optimal token budget (~800M tokens at bs=128, seq=512 for
  20k steps).
- Retrieval target: a fixed corpus table of embeddings indexed by
  bigram/trigram xxhash64 (research-only; not mirrored in Rust/GGUF).
  Table is [total_entries, engram_dim=128], attached at
  `config.engram_injection_layer=2`, non-trainable.
- Injection mechanisms tested in a 20k-step bake-off (in progress):
  - **alpha**: no retrieval (baseline)
  - **beta**: alpha + FFN expansion at the injection layer to match
    delta's parameter overhead (+786K params, params-matched dense
    control)
  - **gamma**: alpha + gated-residual injection with differentiable
    softmax retrieval (temperature = 1/sqrt(engram_dim)), RMS-norm
    gate, hard gate clamp + Bernoulli-entropy regularization with
    dynamic lambda scaling for anti-collapse
  - **delta**: alpha + per-position cross-attention to top-k retrieved
    neighbors, independent W_q/W_k/W_v/W_o, W_o zero-init as
    anti-collapse, differentiable retrieval-similarity bias to
    preserve gradient through the top-k gather
- Decision gates (pre-committed before the 20k runs):
  - gamma or delta beats beta by >= 0.05 nats -> retrieval transmits
    signal with THIS mechanism
  - gamma and delta both match beta (+/- 0.05 nats) -> injection
    architecture cannot transmit signal from this retrieval path
  - gamma or delta worse than beta by > 0.05 nats -> anti-collapse
    machinery actively hurts training

### The question the bake-off CANNOT answer

gamma/delta vs beta tells us whether mechanism M can transmit signal
from the CURRENT corpus table, but it conflates two independent
questions:

1. **Is the mechanism inert?** (i.e. the injection pathway cannot use
   retrieved embeddings even if they carried perfect signal)
2. **Is the corpus-table CONTENT inert?** (i.e. n-gram hashed into a
   random-ish bag of embeddings doesn't carry signal our model can
   exploit, regardless of injection mechanism)

Our Phase 0 ablations (ZEB-102) confirmed the ORIGINAL production
engram table (xxhash -> fixed table -> JL-projection) transmits zero
signal: null-table, shuffle, and random-table all give val loss
indistinguishable from the corpus-derived table. But we never
constructed a corpus table that is KNOWN to carry strong semantic
signal. If gamma/delta match beta in the 20k bake-off, we can't tell
whether:

- (a) the mechanism is the failure mode, or
- (b) the content is the failure mode

This distinction determines the next research direction completely:

- (a) -> research alternative mechanisms (product keys, hash layers,
  adapter-style retrieval, chunked RETRO-style cross-attention)
- (b) -> research content quality (teacher-distilled embeddings,
  contrastive table training, semantic key spaces) and keep the
  winning mechanism

### The diagnostic I want designed

**Build an "oracle" corpus table from a teacher model** and re-run the
winning ZEB-117 mechanism against it. Specifically:

1. Pick a teacher model known to perform well on FineWeb-Edu-class
   data at a scale 10-100x ours (candidates: Qwen3-1.7B, Qwen2.5-1.5B,
   Llama-3.2-1B, Pythia-1.4B, or similar open-weight <= 3B model with
   a permissive license).
2. Run the teacher over the FineWeb-Edu-POC training corpus once,
   extracting hidden states at some layer L (which? middle layer
   typically; literature guidance needed). For each token position,
   record the hidden state.
3. Aggregate hidden states into bigram/trigram n-gram embeddings
   matching our existing `EngramTable` schema: same total_entries,
   same engram_dim=128 (projected down from the teacher's hidden_dim
   via some scheme - learned? random projection? mean-pool + linear?).
   The key property: each row must represent the teacher's "view" of
   what that n-gram means in context.
4. Feed this oracle table to the winning ZEB-117 mechanism (gamma or
   delta) for a 20k-step training run. Compare val loss to:
   - alpha (baseline)
   - beta (params-matched dense control)
   - The same mechanism on the ORIGINAL random/corpus table (the
     ZEB-117 result)

### Interpretation matrix for the diagnostic

| ZEB-117 result | Oracle diagnostic result | Interpretation |
|----------------|--------------------------|----------------|
| gamma or delta beats beta | Oracle much stronger | Corpus quality is the primary lever. Invest in teacher distillation pipeline (ZEB-102 Phase 2). |
| gamma or delta beats beta | Oracle similar | Mechanism ceiling hit at 40M scale. Next: scale study (80M/160M) with the winning mechanism. |
| Both match beta | Oracle beats beta cleanly | **Mechanism is the failure mode**, not content. Pivot to alternative mechanisms: product keys, hash layers, adapter-retrieval, chunked RETRO. |
| Both match beta | Oracle also matches beta | External memory at 40M scale doesn't transmit signal regardless of content OR mechanism. Engram dead for Harmony v1. Pivot to COCONUT / latent reasoning / just scale the baseline. |

### What I want the research report to cover

#### 1. Teacher model selection

- Which 1-3B open-weight models are known to produce hidden states
  that transfer well to smaller models via distillation? Specifically
  for the mid-layer feature-extraction use case (NOT full
  logit-distillation).
- Should I use the embed_tokens output, a specific transformer layer's
  output, or some pooled/averaged representation? What does the
  literature say about WHICH layer of a pretrained model carries the
  richest semantic signal for transfer to a smaller student?
- License considerations for FineWeb-Edu-based distillation from Qwen,
  Llama, Pythia, etc. — any gotchas?

#### 2. N-gram representation extraction

- Given teacher hidden states at some layer, how do I aggregate them
  into n-gram embeddings that match our `EngramTable` schema?
  Candidate schemes:
  - Mean-pool hidden states of the constituent tokens in the n-gram
  - Use the final token's hidden state (since it has seen the full
    n-gram via causal attention)
  - Concatenate-then-project (learnable or random projection)
  - Contextualized: extract hidden state at position i for the n-gram
    ending at position i, across all occurrences of that n-gram, then
    average
- Which scheme has theoretical/empirical support? Any prior work
  specifically on "distill n-gram features from a teacher"?

#### 3. Dimensionality reduction

- Teacher hidden_dim is typically 1536-4096; my engram_dim is 128.
  Schemes for projecting down:
  - Random Gaussian projection (fixed, JL-like)
  - Learned linear projection trained with a reconstruction or
    contrastive objective on held-out data
  - PCA / SVD on the distribution of teacher hidden states
  - Quantization-aware projection (product quantization, LSH)
- Which preserves the retrieval-relevant structure best for
  cross-attention / gated-residual retrieval? Any literature on the
  effective dimensionality of transformer hidden states for
  retrieval-adjacent tasks?

#### 4. Expected signal strength

- What's the literature estimate for how much a teacher-distilled
  retrieval feature SHOULD improve a small student on language
  modeling perplexity? I.e. if the oracle table is genuinely good,
  roughly how many nats of val-loss improvement should I expect at
  40M / 20k steps / Chinchilla-optimal tokens?
- Is there a known ceiling from distillation literature (DistilBERT,
  TinyBERT, MiniLM, etc.) that tells me "if you see less than X nats
  improvement even with perfect retrieval, the 40M scale genuinely
  cannot use this signal"?

#### 5. Failure modes specific to the oracle diagnostic

- The teacher's hidden states are contextualized — they encode
  information the student hasn't seen at position i yet (the teacher
  processed the full sequence). Does this constitute label leakage?
  Does it matter that the retrieval happens at training time (where
  the loss is next-token prediction and the retrieval feature is
  conditional on only the past tokens)?
- If n-grams are hashed via xxhash and the hash collides, different
  n-grams map to the same row. This averaging contamination exists
  for both the original and oracle tables; does it affect the oracle
  diagnostic's interpretability?
- The teacher is a 1-3B model trained on different data distribution
  (typically). How much does teacher-pretraining-data mismatch with
  FineWeb-Edu-POC matter for the quality of extracted features?

#### 6. Engineering pipeline

- Concrete design for a `generate_oracle_table.py` tool that:
  - Takes a HuggingFace model ID + tokenized dataset path
  - Runs forward passes with output_hidden_states=True
  - Extracts the chosen layer's hidden states
  - Aggregates into the n-gram embedding schema
  - Projects to engram_dim
  - Writes a safetensors file compatible with our existing
    `--engram-ann-table` / `--engram-xattn-table` flags
- Memory / compute estimates: how long does it take to extract
  features from ~800M tokens of FineWeb-Edu-POC using a 1.5-3B
  teacher on a single RTX 4090? Is this a weekend job or a week job?
- Storage: a table with N entries x 128 dims x f32 = 512N bytes. For
  our typical table size (let's say 1M entries) that's 512MB. Any
  scaling concerns?

#### 7. Comparative baselines

- Is there a cheaper diagnostic that separates content from mechanism
  without requiring teacher distillation?
- Specifically: what if I use the STUDENT MODEL's own hidden states
  from a partially-trained checkpoint of alpha as the oracle table?
  Pros: free (already training alpha as part of bake-off). Cons: the
  hidden states encode the same information the student already has,
  so retrieval may be tautological.
- What if I use a larger student (80M or 160M) as a "teacher" for the
  40M model? Smaller gap than 1-3B distillation but cheaper.
- What if I use a SPARSE / INFORMATIVE table: one row per UNIQUE
  n-gram in the corpus (no hashing collisions), embeddings from a
  sentence-transformer model applied to the n-gram text directly?
  This isolates hash-collision contamination from content quality.

#### 8. Related work I should know about

- RETRO (Borgeaud et al. 2022) — retrieved-neighbor cross-attention,
  but with BERT-embedded retrieval keys
- Memorizing Transformer (Wu et al. 2022) — cached KV retrieval
- REALM (Guu et al. 2020) — retrieval-augmented pretraining with
  learned retriever
- kNN-LM (Khandelwal et al. 2020) — post-hoc retrieval at inference
- Atlas, RAG, FiD, RECOMP — retrieval-augmented generation variants
- Any direct prior work on "oracle retrieval table as a diagnostic
  for architectural capacity"? I suspect this is an unexplored
  methodological tool.

#### 9. Concrete research-project proposal

Based on the analysis, propose:

- A ranked list of 2-3 oracle-table construction approaches (teacher
  model + layer + aggregation + projection scheme), from most likely
  to produce strong signal to least.
- A go/no-go criterion: what val-loss improvement over alpha would
  justify investing in a production teacher-distillation pipeline
  (ZEB-102 Phase 2)?
- A runtime budget: total GPU-hours for the diagnostic (oracle-table
  generation + a 20k student run per approach).
- A contingency for the worst-case outcome (oracle also matches
  alpha): what does it tell us about Harmony v1 roadmap?

### Known context the report should assume

- The training pipeline is mature: BF16, gradient checkpointing, Muon
  optimizer, AdamW fallback, CSV logging with per-step metrics, ANN
  and cross-attention modules are implemented and tested.
- GGUF / Rust portability is NOT a constraint for this diagnostic;
  oracle-derived tables can stay research-only. Rust parity is needed
  only if the mechanism + content combo is promoted to production.
- Compute budget: 1x RTX 4090 (KRILE) + 1x RTX 5080 16GB (AVALON).
  Weekend-scale experiments OK; week-scale experiments require
  justification.
- Data: FineWeb-Edu-POC tokenized, ~800M training tokens available
  locally at `/data/fineweb-edu-poc/{train,val}`.
- Corpus table infrastructure: `EngramTable.from_safetensors()` loads
  a table; `EngramANNInjection.load_corpus_table()` /
  `EngramCrossAttention.load_corpus_table()` consume it.
  `training/ct87/generate_engram_table.py` currently generates the
  random+hashed corpus table; a new
  `generate_oracle_table.py` would live alongside it.

### Output format I want

1. Executive summary (< 500 words): top recommendation (which oracle
   construction approach to try first, why).
2. Literature review (2-4 pages): teacher selection, feature
   extraction, n-gram aggregation, dim reduction, with citations.
3. Methodology section: detailed design of the oracle-table
   construction pipeline.
4. Expected-signal analysis: quantitative estimate of what val-loss
   delta the oracle should produce IF retrieval works and the content
   is good.
5. Failure-mode analysis: what could go wrong with the diagnostic
   itself, and how to distinguish diagnostic failure from
   architectural failure.
6. Engineering plan: step-by-step implementation guidance, runtime
   budget, storage, tooling.
7. Interpretation playbook: how to read the diagnostic's result in
   each cell of the 2x4 matrix (ZEB-117 outcome x oracle outcome),
   with specific go/no-go thresholds.
8. Bibliography.

Produce the report. Be opinionated — if one teacher model or one
aggregation scheme is obviously better than the others, say so and
defend the choice rather than listing options neutrally. If you
encounter a question where the literature is divided, say so and
recommend the approach best suited for a 40M-parameter student with
Chinchilla-optimal token budget and a single-GPU compute envelope.
