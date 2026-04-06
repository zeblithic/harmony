# Harmony Custom Edge Model — Architecture Specification

**Epic:** harmony-ct87
**Status:** Design approved, pending implementation planning
**Date:** 2026-04-06

## Goal

Design and train a custom ~0.5B parameter language model built from the ground up
around Harmony's architectural innovations — Block Attention Residuals, Engram
conditional memory, TurboQuant KV compression, uncertainty quantification, and
Chronos temporal decay — rather than bolting them onto an existing pretrained model.

**Core thesis:** A model designed natively with Engram conditional memory can be
dramatically smaller than conventional models because it only needs reasoning
capacity — static facts are looked up, not memorized. This is analogous to how
human cognition works: we don't memorize encyclopedias, we know what's worth
looking up and how to use what we find.

## Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Model type | Dense (no MoE) | Isolate innovation variables; MoE adds routing instability as a 6th research variable |
| Base architecture | Qwen3-derived | GQA, SwiGLU, RoPE, RMSNorm already optimal at sub-1B; existing candle implementation to reference |
| Training infra | PyTorch training + candle inference validation | Mature training ecosystem; candle validates GGUF export at every checkpoint |
| Engram injection | Layer 2 only | Research: early injection liberates attention heads from pattern reconstruction; architecture supports multi-depth for v2 |
| UQ head coupling | Parallel monitor (independent of gate) | Three mechanisms compose independently: Chronos attenuates, gate filters, UQ monitors |
| Deployment (v1) | Single-machine, local Engram tables | Proves the synergy without distributed systems complexity; mesh is a natural extension |
| Compute strategy | Local experiments (tiny model on 4090) then cloud burst (8xA100) | Validate architecture cheaply before committing cloud spend |

## System Architecture

The three metacognitive mechanisms operate at different points in the pipeline and
compose independently:

- **Chronos** acts *before* the gate — attenuates stale embeddings
- **EngramGatedResidual** acts *during* the forward pass — filters injection by semantic relevance
- **UQ head** acts *after* the forward pass — decides whether to trust output or trigger retrieval/escalation

```
INPUT TOKENS
     |
     v
  Tokenizer (Qwen3-derived, ~32K vocab)
     |
     v
  ENGRAM BRIDGE
  tokens -> N-gram hashes -> shard lookup -> embeddings
  (async: starts immediately, results ready by Layer 2)
     |
     v
  CHRONOS temporal decay (attenuate stale embeddings)
     |
     v
  TRANSFORMER BACKBONE (0.5B dense, 24 layers, 8 blocks)
     |
     +-- Block 0 (Layers 0-2)
     |     Layer 2: + EngramGatedResidual <-- attenuated embeddings
     |   [Block AttnRes boundary]
     +-- Block 1 (Layers 3-5)
     |   [Block AttnRes boundary]
     +-- ...
     +-- Block 7 (Layers 21-23)
     |     Final: RMSNorm -> LM Head
     |
     |   KV Cache: TurboQuant (PolarQuant 3-bit + QJL 1-bit)
     |
     v
  LOGITS [vocab_size]
     |
     v
  UQ HEAD (parallel monitor)
  Inputs: hidden state norms, attention lookback, logit entropy
  Outputs: uncertainty type + confidence score
     |
     v
  ROUTING: Confident -> emit | HighVolume -> Engram lookup | SpectralCollapse -> escalate
```

## 1. Transformer Backbone

Qwen3-derived decoder-only transformer with specific modifications for our scale.

### Architecture Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Layers | 24 | Deep-and-thin; 8 blocks x 3 layers for clean AttnRes boundaries |
| Hidden dim | 1280 | Wide enough for Engram gate dot-product; hits ~471M total params |
| Query heads | 16 | Standard for this hidden dim |
| KV heads | 8 | GQA-8: 75% KV cache reduction vs MHA, ~98% quality retention |
| Head dim | 80 | 1280 / 16 (same for query and KV heads; GQA uses fewer KV heads, not bigger ones) |
| FFN dim | 3413 | ~8:3 ratio with SwiGLU |
| Vocab size | ~32K | Trimmed from Qwen3's 151K for parameter savings |
| Embeddings | Tied input/output | ~25% parameter savings |
| Position encoding | RoPE | Rotary, same as Qwen3 |
| Normalization | RMSNorm (eps=1e-6) | Pre-norm, same as Qwen3 |
| Activation | SwiGLU | In FFN blocks |

### Parameter Count

```
Embeddings:     32K x 1280                        = ~41M (tied, counted once)
Per layer:
  Q proj:       1280 x (16 x 80)                 = ~1.6M
  K proj:       1280 x (8 x 80)                  = ~0.8M
  V proj:       1280 x (8 x 80)                  = ~0.8M
  Output proj:  1280 x 1280                       = ~1.6M
  SwiGLU FFN:   1280 x 3413 x 3                  = ~13.1M
  RMSNorm x 2:  1280 x 2                         = ~0.003M
  Layer total:                                     = ~17.9M
24 layers:      17.9M x 24                        = ~430M
                                            TOTAL = ~471M active params
```

### Tiny Model Configuration (for local experiments)

| Parameter | Tiny | Target |
|-----------|------|--------|
| Layers | 8 | 24 |
| Hidden dim | 512 | 1280 |
| Query heads | 8 | 16 |
| KV heads | 4 | 8 |
| FFN dim | 1365 | 3413 |
| AttnRes blocks | 4 (4x2) | 8 (8x3) |
| Total params | ~50M | ~509M |
| Engram dim | 128 | 256 |
| Training hardware | Single 4090 | 8xA100 |

## 2. Block Attention Residuals

Replaces standard additive residual connections with a learned depth-wise attention
mechanism at block boundaries, solving the PreNorm dilution problem.

### The Problem

In a 24-layer PreNorm transformer, the residual stream accumulates all layer outputs
via unweighted addition. By layer 20, the contribution of layer 1 is a tiny fraction
of the total signal. Critical information extracted in early layers (including Engram
injections at Layer 2) gets diluted by intermediate processing.

### The Mechanism

```
Block 0: Layers 0, 1, 2    |
Block 1: Layers 3, 4, 5    | Within block: standard additive residuals
Block 2: Layers 6, 7, 8    |
...                         |
Block 7: Layers 21, 22, 23  |

At each block boundary: depth-wise attention over block summaries

  Input to block k = sum_{j=0..k-1} alpha_j * BlockSummary_j + alpha_partial * PartialSum_k
  where alpha = softmax(q . [BlockSummary_0, ..., BlockSummary_{k-1}, PartialSum_k])
  q = learned pseudo-query vector per boundary
```

### Key Properties

- **Block summary:** Accumulated residual at end of each block (bookkeeping, not extra compute)
- **Pseudo-query:** One learned vector per boundary. 7 boundaries x 1280 dims = 8,960 params total.
- **Memory cost:** O(N x d) = 8 x 1280 = ~10KB per token. Negligible vs KV cache.
- **I/O cost:** ~5.5d per layer. For d=1280, ~7KB/layer — far below mHC's 34d.

### Interaction with Engram

Engram injects at Layer 2 (inside Block 0). The Block 0 summary captures the
Engram-enriched hidden state. All subsequent blocks can attend directly to this
summary without dilution. AttnRes preserves the Engram signal across depth.

### Implementation

Candle: `block_attnres.rs` — stores block summary tensors, computes boundary
attention every 3 layers. PyTorch: `block_attnres.py` — mirror implementation
for training.

## 3. TurboQuant KV Cache

Upgrades the current 3-bit uniform quantization to PolarQuant + QJL for better
quality at the same bit rate while eliminating per-vector metadata overhead.

### Current System (kv_compress.rs)

- Per-vector min/scale quantization (8 bytes overhead per vector)
- 3-bit packed indices (48 bytes for 128-dim)
- Total: 56 bytes per vector vs 256 bytes at f16
- ~4.6x compression, but per-vector metadata costs ~14% of compressed size

### TurboQuant Replacement

**Step 1: Random Orthogonal Preconditioning (once at model load)**
Multiply KV vectors by Walsh-Hadamard Transform matrix. Smooths channel-wise
outliers so all dimensions have similar distributions.

**Step 2: PolarQuant (per-vector)**
Transform to polar coordinates (radius + angles). Quantize angles using a
GLOBAL pre-computed Lloyd-Max codebook. No per-vector min/scale needed.
Cost: 2 bytes (f16 radius) + 30 bytes (3-bit angles for 80-dim) = 32 bytes.

**Step 3: QJL Error Correction (1-bit residual)**
Project quantization residual via random Johnson-Lindenstrauss matrix.
Quantize to sign bits. Provides UNBIASED inner product estimation.
Cost: 10 bytes (1 bit per dimension for 80-dim).

**Combined:** 42 bytes per vector at head_dim=80.
vs f16: 160 bytes per vector. ~3.8x compression.
Effective ~4.2 bits/value with better quality than current 3.5 bits.

### Memory Budget (24 layers, GQA-8, head_dim=80)

| Context | f16 | Current 3-bit | TurboQuant |
|---------|-----|---------------|------------|
| 4K tokens | 240 MB | 53 MB | 63 MB |
| 32K tokens | 1.9 GB | 420 MB | 504 MB |

TurboQuant at 32K context fits comfortably in RPi5 headroom. f16 would consume
nearly all available memory.

### Implementation

Replace internals of `kv_compress.rs`. External API unchanged:
`compress()`, `decompress()`, `serialize_compressed()`, `deserialize_compressed()`.
WHT matrix and Lloyd-Max codebook generated once at model load, stored as constants.

## 4. Engram Injection

External knowledge enters the model via gated residual injection at Layer 2.
Most machinery already exists in `harmony-engram` and `harmony-inference`.

### Pipeline

```
Tokens -> N-gram extraction (bigrams + trigrams)
       -> Hash (deterministic, multi-head)
       -> Shard lookup (local for v1, mesh for future)
       -> Resolve f16 embeddings, aggregate per position
       -> Chronos temporal decay (attenuate stale entries)
       -> EngramGatedResidual at Layer 2
```

### Gated Residual Module (exists: engram_residual.rs)

```
k = key_proj(engram_emb)       [1, seq, 1280]
v = value_proj(engram_emb)     [1, seq, 1280]
h_norm = RMSNorm(hidden_state)
k_norm = RMSNorm(k)
gate = sigmoid(dot(h_norm, k_norm) / sqrt(1280))   in [0, 1]
gated = gate * v
conv_out = CausalConv1D(gated, kernel=4)
residual = SiLU(conv_out)
hidden_state = hidden_state + residual
```

### Configuration Changes from Current

| Aspect | Current | Custom Model |
|--------|---------|-------------|
| engram_dim | configurable | 256 |
| hidden_dim | matches model | 1280 |
| conv_kernel_size | 3 | 4 |
| Injection point | configurable | Layer 2 |
| Weight source | random/GGUF | Trained in PyTorch, exported to GGUF |

### Engram Table Sizing

U-shaped scaling law: ~20-25% of sparse capacity to Engram.
For our 471M active-param model: ~100-130M params of external storage.

```
100M params / 256 dims = ~390K entries x 512 bytes = ~200 MB
130M params / 256 dims = ~508K entries x 512 bytes = ~260 MB
```

Fits on all target hardware. Loaded at startup, queried via hash — no GPU memory needed.

### What Already Exists (unchanged)

- `engram_bridge.rs` — Sans-I/O two-phase bridge (prepare_engram_request, resolve_engram_embeddings)
- `EngramGatedResidual` — Module forward pass (from_tensors handles arbitrary dims)
- `EngramClient` in harmony-engram — Hash, manifest, resolve

### What's New

- **Engram table builder** — Ingest knowledge sources, produce sharded f16 tables (harmony-1kha)
- **Vocabulary normalization** — Normalize N-gram tokens for better hash hits (harmony-0aa4)
- **Training-time simulation** — PyTorch training loop performs real Engram lookups against frozen tables

## 5. Uncertainty Quantification Head

Parallel metacognitive monitor. Observes model internals after the forward pass
and decides whether to trust output, trigger retrieval, or escalate.

### Architecture

A deliberately simple MLP on hand-crafted features for v1. Interpretable,
cheap to train, negligible overhead, and upgradeable to a transformer encoder later.

**Feature extraction (8 floats):**
- f1-f4: Hidden state L2 norms at layers 0, 8, 16, 23 (one per network quadrant)
- f5: L2 norm trajectory slope (growing, stable, or collapsing?)
- f6: Logit entropy (Shannon)
- f7: Top-k probability mass (distribution concentration)
- f8: Attention lookback ratio (prompt attention vs generated-token attention)

**Classification head:**
- Linear(8 -> 32) -> ReLU -> Linear(32 -> 4) -> softmax
- Classes: confident, high_volume, spectral_collapse, uncertain
- ~300 parameters total

**Confidence scalar:**
- Linear(8 -> 1) -> sigmoid -> [0, 1]

### Dual-Trigger Geometry

**High-volume uncertainty:**
- Norms: stable, normal magnitude
- Entropy: HIGH (flat distribution)
- Top-k mass: low concentration
- Meaning: many candidates, model knows the category but not the answer
- Action: trigger Engram lookup with hidden state as semantic query

**Spectral collapse (gray vector):**
- Norms: collapsing toward zero across depth
- Entropy: may be LOW (false confidence on random token)
- Norm slope: steep negative
- Meaning: no signal, model has nothing
- Action: abort generation, flag as unknowable, escalate

**Critical insight:** Spectral collapse can produce LOW entropy (confident wrong answer).
Hidden state norm monitoring catches this; logit entropy alone would miss it.

### Routing Decisions

```
confident (score > tau_high)    -> emit token, continue
high_volume                     -> Engram lookup, re-run with injection
spectral_collapse               -> abort, escalate
uncertain (ambiguous)           -> conservative: treat as high_volume
```

### Training

During Phase 3 of staged curriculum (after base model + Engram are trained):
1. Run trained model on diverse question set
2. Compare outputs against ground truth
3. Label each step: correct / hallucinated / refused-correctly / missed-retrieval
4. Extract 8 features at each step
5. Train MLP classifier (standard supervised classification)

### Future: SConU Integration (v1.1)

Selective Conformal Uncertainty wraps UQ output in conformal prediction p-values
for mathematically guaranteed abstention. Requires calibration dataset. The UQ
head's interface (features -> class + confidence) is SConU-ready.

## 6. Chronos Temporal Decay

Attenuates Engram embeddings based on knowledge freshness. Stale facts weaken
toward zero — the model treats expired knowledge like missing knowledge.

### Per-Entry Metadata

```
hash_key:     u64
embedding:    [f16; 256]     (the knowledge)
timestamp:    u32            (Unix epoch)
tier:         u8             (1-5)
ttl_seconds:  u32            (tier-derived)
```

### Decay Function

```
age = now() - entry.timestamp

if tier == 1 (eternal):
  decay = 1.0

else:
  if age <= ttl:
    decay = 1.0              // still fresh
  else:
    staleness = (age - ttl) / ttl
    decay = exp(-staleness^2 / 2)   // Gaussian decay past TTL

attenuated_embedding = embedding * decay
```

### Frequency Tiers

| Tier | Name | TTL | Examples |
|------|------|-----|----------|
| 1 | Eternal | infinity | Speed of light, mathematical axioms |
| 2 | Near-eternal | 10 years | Historical facts, geography |
| 3 | Episodic | 1 year | Political leaders, tech standards |
| 4 | Regular | 30 days | Ages, populations |
| 5 | Ephemeral | 0 | Stock prices, live sensor data |

### Decay Curve

At age = TTL: decay = 1.0 (just expired, still fully trusted)
At age = 2x TTL: decay = 0.61
At age = 3x TTL: decay = 0.14
At age = 4x TTL: decay = 0.01 (effectively zero)

Knowledge gracefully dies over ~3-4x its TTL.

### Connection to Gray Vector

When decay -> 0, attenuated embedding -> zero vector. EngramGatedResidual
produces zero residual for zero input (tested: zero_embedding_returns_zero_residual).
Model proceeds as if no Engram entry exists. UQ head detects resulting uncertainty.

### Implementation

Pure function in harmony-engram:
```rust
pub fn temporal_decay(entries: &[EngramMetadata], now: u32) -> Vec<f32>
```

Caller multiplies resolved embedding tensor by decay vector before passing
to EngramGatedResidual. Tier classification uses heuristic rules for v1
(Wikipedia current events -> tier 3/4, science -> tier 1/2).

## 7. Training Pipeline & Staged Curriculum

PyTorch training with candle validation, following the research-backed
three-phase curriculum.

### Infrastructure

- **Optimizer:** Muon + WSD (Warmup-Stable-Decay) schedule
- **Data loading:** HuggingFace datasets + custom loaders
- **Distributed:** torch.distributed for multi-GPU
- **Checkpointing:** Every N steps -> PyTorch + GGUF export
- **Candle validation:** Every M steps, load GGUF, assert logit divergence < epsilon

### Staged Curriculum

**Phase 1: Foundation (pure parametric)**
- Train base transformer + Block AttnRes
- Data: reasoning-dense synthetic (math, code, logic, textbook-style). NOT encyclopedic.
- Engram gates initialized to zero (module present but invisible)
- UQ head not attached
- Scale: tiny on 4090, then 0.5B on cloud
- Tokens: ~200-300B curated data (overtraining regime)

**Phase 2: Memory Integration (learn to use Engram)**
- Unfreeze Engram gating weights, freeze base model
- Data: synthetic Q&A requiring factual lookup
- Mix: ~40% answerable-from-Engram, ~30% unknown, ~30% stale
- Chronos active — some entries deliberately expired
- Scale: consumer hardware (only gating params update)
- Tokens: ~10-50B focused dataset

**Phase 3: Metacognitive Alignment (learn uncertainty)**
- Train UQ head on labeled uncertainty data
- Generate outputs from Phase 2 model, label correctness
- Extract 8 UQ features, train MLP classifier
- Standard supervised classification — very cheap

**Phase 4: Fine-Tuning (optional stretch)**
- DPO/RLHF on full system with very low LR
- Reward correct Engram usage, penalize hallucination

### Data Strategy

**Parametric training (Phase 1):**
- Synthetic textbook data generated by larger models (Phi-style)
- Open math/code datasets (OpenMathInstruct, CodeContests, APPS)
- Logic puzzles, symbolic reasoning traces

**Engram tables:**
- Wikipedia entity knowledge (tier 2-3)
- Wikidata structured facts (tier 2-4)
- Domain-specific knowledge bases (configurable)

**Engram training data (Phase 2):**
- Synthetic Q&A where answers exist in Engram tables
- "Known unknowns" — questions about facts not in table
- "Stale facts" — questions with deliberately expired entries

## 8. Evaluation Strategy

### Benchmark Suite

| Category | Benchmark | Measures | Target Innovation |
|----------|-----------|---------|-------------------|
| Reasoning | ARC-Challenge | Grade-school science | Block AttnRes |
| Reasoning | HellaSwag | Commonsense reasoning | Backbone |
| Reasoning | WinoGrande | Pronoun resolution | Backbone |
| Reasoning | GSM8K | Multi-step math | Block AttnRes |
| Factual | TriviaQA | Entity knowledge | Engram |
| Factual | Natural Questions | Factual Q&A | Engram |
| Factual | MMLU (subset) | Broad knowledge | Engram + Chronos |
| Temporal | Custom: StaleFacts | Stale fact detection | Chronos |
| Temporal | Custom: FreshVsStale | Fresh vs expired comparison | Chronos + UQ |
| Uncertainty | Custom: HallucinationRate | Wrong fact rate | UQ head |
| Uncertainty | Custom: EscalationAccuracy | Correct "I don't know" rate | UQ head |
| Uncertainty | Custom: RetrievalPrecision | Necessary vs unnecessary lookups | UQ + gate |
| Efficiency | KV memory @ 4K/32K | Cache size | TurboQuant |
| Efficiency | Tokens/sec (4090, RPi5) | Throughput | All |

### Success Criteria

The thesis is proven if:
- Full system 0.5B on TriviaQA: ~55-65% (vs vanilla 0.5B ~30%)
- Reasoning benchmarks (ARC, GSM8K): within 5% of vanilla baseline
- Hallucination rate: measurably lower with UQ head active
- KV memory at 32K: under 550MB (TurboQuant)

### Ablation Studies (Phase 4)

Disable each innovation one at a time and re-evaluate. Each delta shows
the contribution of that innovation. If any delta is near zero, simplify.

## 9. Implementation Phases

### Crate Structure

```
crates/
  harmony-engram/           (EXISTS)
    + chronos.rs            NEW: temporal decay
    + metadata.rs           NEW: per-entry timestamp/tier/ttl types

  harmony-inference/        (EXISTS)
    engine.rs               UNCHANGED (QwenEngine for Qwen3)
    engram_bridge.rs        UNCHANGED
    engram_residual.rs      UNCHANGED
    kv_compress.rs          REPLACE internals with TurboQuant
    + harmony_model.rs      NEW: custom model forward pass
    + block_attnres.rs      NEW: Block AttnRes module
    + uq_head.rs            NEW: uncertainty head

  harmony-model-tools/      NEW CRATE
    gguf_export.rs          PyTorch -> GGUF converter
    engram_builder.rs       Knowledge source -> Engram table builder
    eval.rs                 Evaluation harness

training/                   NEW TOP-LEVEL DIR (PyTorch)
  harmony_model/
    config.py               Model config (tiny + target)
    transformer.py          Qwen3-derived backbone
    block_attnres.py        Block AttnRes (PyTorch mirror)
    engram_gate.py          EngramGatedResidual (PyTorch mirror)
    uq_head.py              UQ head
    chronos.py              Temporal decay
  train.py                  Main training loop
  curriculum/               Phase 1/2/3 training configs
  export/to_gguf.py         Export to GGUF
  eval/                     Benchmarks + custom evals
```

### Phase 0: Foundation Modules

Build all candle modules + PyTorch mirrors + tiny model validation.

Sub-projects (0a-0d can run in parallel):
- 0a. Block AttnRes module (candle + PyTorch, unit tested)
- 0b. TurboQuant KV (replace kv_compress.rs internals)
- 0c. Chronos temporal decay (in harmony-engram)
- 0d. UQ head (candle + PyTorch, unit tested)
- 0e. Custom model forward pass (assembles all modules)
- 0f. PyTorch training scaffold
- 0g. GGUF export/import (roundtrip validated)
- 0h. Candle validation loop

Deliverable: Tiny model trains on 4090, exports to GGUF, runs in candle
with all modules exercised.
Hardware: KRILE (4090 24GB)

### Phase 1: Tiny-Scale Integration

All innovations running together at tiny scale.

- 1a. Build test Engram tables (~10K entries)
- 1b. Run staged curriculum at tiny scale
- 1c. Evaluate: does tiny model use Engram correctly?
- 1d. Iterate on parameters

Deliverable: Tiny model demonstrating all innovations working together.
Hardware: KRILE (4090 24GB)

### Phase 2: Vanilla 0.5B Baseline

Train standard Qwen3-style 0.5B (no innovations) for benchmark comparison.

- 2a. Curate training data (~200-300B tokens)
- 2b. Train vanilla baseline
- 2c. Evaluate on full benchmark suite

Deliverable: Baseline benchmark scores.
Hardware: Cloud 8xA100
Note: Can run in PARALLEL with Phase 1.

### Phase 3: Full 0.5B With All Innovations

The real v1.

- 3a. Build production Engram tables (~400K entries, ~200MB)
- 3b. Train with staged curriculum
- 3c. Evaluate on full benchmark suite
- 3d. Compare against vanilla baseline

Deliverable: The v1 model running single-machine with all innovations.
Hardware: Cloud 8xA100

### Phase 4: Ablation & Polish

Measure each innovation's contribution.

- 4a. Ablation studies
- 4b. Error analysis
- 4c. Tune thresholds
- 4d. Package for deployment

Deliverable: Published results, packaged model.
Hardware: Consumer GPUs

## Related Beads

| Bead | Area | Relationship |
|------|------|-------------|
| harmony-ct87 | Epic | This spec |
| harmony-0lge | Block AttnRes | Phase 0a |
| harmony-iwsh | PolarQuant | Phase 0b (TurboQuant) |
| harmony-3570 | TurboQuant fidelity | Phase 0b |
| harmony-qldu | Chronos | Phase 0c |
| harmony-iyot | Uncertainty topology | Phase 0d |
| harmony-1kha | Engram ingest | Phase 1a / 3a |
| harmony-0aa4 | Engram vocab normalization | Phase 1a |
| harmony-uujk | PoC vs baseline | Phase 1 / Phase 2 |
| harmony-ixiq | Training infrastructure | Phase 0f |
| harmony-qvtr | Training data curation | Phase 2a |
| harmony-ynge | Base model survey | Resolved: Qwen3-derived |
| harmony-5nwj | Generic model architecture trait | Phase 0e |
| harmony-8hjt | Memory budget | Phase 0b |
| harmony-xcha | Engram benchmarks | Phase 1c |
| harmony-hrpg | AttnRes + Engram interaction | Phase 1d (v2: multi-depth) |
| harmony-mv4y | mHC -> AttnRes decision | Resolved: AttnRes chosen |
| harmony-g689 | Hardware benchmarks | Phase 4 |
| harmony-pzlx | Speculative decoding | Post-v1 |
| harmony-cu6s | Adaptive Engram gating | Post-v1 |
| harmony-hf1y | Speculative Engram prefetch | Post-v1 |
| harmony-qs9a | Chunked periodic injection | Post-v1 |
| harmony-ibf6 | MoE + Engram | Post-v1 |
