# Gemini Deep Research Prompt: Latent-Space Reasoning for Harmony's Custom Edge Model

## Background

We are building **HARMONY**, a decentralized mesh computing platform that runs inference on constrained edge devices. We are designing and training a custom ~0.5B parameter language model (codename **ct87**) built from the ground up around five architectural innovations. The model is a dense Qwen3-derived decoder-only transformer (24 layers, 1280 hidden dim, GQA-8, SwiGLU, RoPE, RMSNorm).

### Current Architecture (Token-Based Autoregressive)

The model currently uses standard next-token prediction: at each step, the full transformer forward pass produces logits over a ~32K token vocabulary, a token is sampled, embedded, and fed back as the next input. All "reasoning" happens implicitly within a single forward pass — the model has no mechanism for iterative internal deliberation without producing tokens.

### Five Existing Innovations

1. **Engram Conditional Memory** — External hash-indexed embedding tables decouple static factual knowledge from model weights. N-gram hashes perform O(1) lookups into f16 embedding tables (~200-260 MB, ~400K entries). Results are injected at Layer 2 via a gated residual module (`EngramGatedResidual`):
   ```
   gate = sigmoid(dot(RMSNorm(hidden), RMSNorm(key_proj(engram))) / sqrt(d))
   residual = SiLU(CausalConv1D(gate * value_proj(engram)))
   hidden = hidden + residual
   ```
   The model only needs reasoning capacity — facts are looked up, not memorized.

2. **Chronos Temporal Decay** — Attenuates Engram embeddings based on knowledge freshness using a Gaussian decay curve. Five frequency tiers (Eternal → Ephemeral). Stale knowledge gracefully decays to zero over 3-4× its TTL, at which point the model proceeds as if no Engram entry exists.

3. **Block Attention Residuals (BlockAttnRes)** — Replaces standard additive residuals with learned depth-wise attention at block boundaries (8 blocks × 3 layers). Solves PreNorm dilution — preserves Engram-injected signals from Layer 2 across full network depth. Only ~9K additional parameters.

4. **TurboQuant KV Cache** — PolarQuant (3-bit angles via global Lloyd-Max codebook) + QJL (1-bit error correction) = 42 bytes/vector vs 160 bytes f16. 3.8× compression enabling 32K context in <550 MB.

5. **Uncertainty Quantification (UQ) Head** — Parallel metacognitive monitor: MLP on 8 hand-crafted features (hidden state L2 norms at 4 depth points, norm trajectory slope, logit entropy, top-k mass, attention lookback ratio). Classifies each generation step as Confident / HighVolume / SpectralCollapse / Uncertain. Routes: emit token, trigger Engram lookup, or abort/escalate.

### The Limitation We Want to Address

In the current architecture, the model's "thinking" is constrained to a single forward pass per token. If the model needs to reason through a complex problem, it must do so across multiple generated tokens — each token requiring a full forward pass, vocabulary projection, sampling, and re-embedding. This is analogous to forcing a human to vocalize every thought through Broca's area rather than thinking in concepts/images/feelings and only converting to speech when communicating.

We want to explore whether the model can **reason natively in embedding space** — performing multiple deliberation steps in the continuous latent space without projecting to tokens — and only invoke text generation when it needs to produce output for a user or file.

## Key Prior Art We've Identified

### VL-JEPA (Meta, arxiv:2512.10942, Dec 2025)

Vision-language model that predicts **continuous embeddings** of target text instead of autoregressively generating tokens. Key properties:
- Learns in abstract representation space, focusing on task-relevant semantics while abstracting away surface-level linguistic variability
- Achieves stronger performance with **50% fewer trainable parameters** vs standard token-space VLMs
- A lightweight text decoder is invoked **only when needed** to translate predicted embeddings into text
- Supports **selective decoding** that reduces decoding operations by 2.85× while maintaining performance

### COCONUT — Chain of Continuous Thought (Meta, arxiv:2412.06769, Dec 2024)

Directly addresses latent reasoning in language models:
- Uses the last hidden state as a "continuous thought" — feeds it back to the model as the next input embedding directly in continuous space
- Reasoning in unrestricted latent space instead of language space
- Continuous thoughts can encode **multiple alternative next steps**, enabling breadth-first search (BFS) rather than committing to a single CoT path
- Outperforms chain-of-thought on logical reasoning tasks requiring planning
- Better accuracy-efficiency tradeoff: fewer tokens during inference
- Open-source implementation: github.com/facebookresearch/coconut

### Related Methods

- **CODI** — MLP mapping hidden states into token embedding space via teacher-student distillation
- **PCCOT** — Jacobian iteration improving CODI's training and inference efficiency
- **CoLaR** — Latent head predicting compressed embedding distributions, maintaining alignment between latent and token spaces
- **Chain-of-Embedding (CoE)** — Output-free self-evaluation via progressive hidden states as latent thinking paths (ICLR 2025)
- **LaSER** — Latent reasoning for dense retrieval, eliminating text generation during inference
- **Pause Tokens** (Goyal et al. 2024) — Adding compute steps in embedding space without generating text

## Research Questions

### Q1: What is the precise mechanism by which VL-JEPA and COCONUT achieve latent-space reasoning, and how do they differ?

We understand the high-level concept but need to understand the exact forward-pass mechanics:

**For VL-JEPA:**
- How does the "embedding predictor" work? What is its architecture — is it a separate transformer, an MLP, or something else?
- When VL-JEPA predicts the embedding of target text, what loss function is used? Is it MSE in embedding space, cosine similarity, or something more sophisticated (e.g., VICReg, Barlow Twins)?
- How does the "lightweight text decoder" work at inference? Is it a separate small autoregressive model that takes predicted embeddings as input? Or a single-pass projection from embedding space to token space?
- What prevents representation collapse (all embeddings converging to the same vector)? VL-JEPA must have specific architectural or loss-function provisions for this.

**For COCONUT:**
- When the "continuous thought" (last hidden state) is fed back as the next input embedding, does it bypass the token embedding table entirely? Or is it projected through a learned mapping layer?
- How many continuous thought steps are typical before the model produces a token? Is this fixed, adaptive, or learned?
- During training, how do you supervise the continuous thoughts? The paper uses a curriculum that alternates between token-based CoT and latent thought — what exactly is the training signal for the latent steps?
- How does the BFS-like property emerge? If a continuous thought encodes multiple alternative paths, how does the model "choose" which path to follow when it eventually commits to a token?

### Q2: How would COCONUT-style continuous thought integrate with our existing architecture?

Our model has several modules that interact with the hidden state at specific points. We need to understand how adding a latent reasoning loop would affect each:

**Engram interaction:**
- Engram injection happens at Layer 2. If the model performs multiple continuous thought steps (each step = full forward pass), does Engram get queried once at the start, or re-queried at each thought step? Re-querying could be powerful — the model's evolving hidden state might trigger different Engram lookups as it reasons.
- The `EngramGatedResidual` uses `sigmoid(dot(hidden, key))` gating. Would the gate learn different gating patterns for "thinking" steps vs "output" steps?

**Block AttnRes interaction:**
- BlockAttnRes maintains block summaries across depth. In COCONUT's loop, do block summaries accumulate across thought steps, or reset each step?
- If they accumulate, the model could build up a richer representation of the problem across reasoning steps — similar to how human working memory accumulates context. Is this beneficial or does it cause gradient issues during training?

**UQ Head interaction:**
- The UQ head currently decides per-token: confident/uncertain/collapse. With continuous thoughts, should the UQ head fire after each thought step, or only when the model transitions from thinking to output?
- Could the UQ head be the mechanism that *decides when to stop thinking and start outputting*? (i.e., "confident enough" = exit the thinking loop and produce a token)

**KV Cache:**
- Each continuous thought step presumably runs the full transformer and contributes to the KV cache. For N thought steps before each output token, the KV cache grows N× faster. How does this interact with TurboQuant's memory budget? Is there a way to compress or discard "thinking" KV entries after the thought loop completes?

### Q3: What is the training procedure for adding latent reasoning to an existing autoregressive model?

We plan a staged curriculum:
- **Phase 1:** Train base transformer as standard next-token-prediction (no latent reasoning)
- **Phase 2:** Train Engram integration
- **Phase 3:** Train UQ head

Where does latent reasoning training fit in this curriculum?

- **Option A:** Train latent reasoning from scratch (Phase 1 uses latent loss instead of/alongside token loss). This gives the model native latent reasoning ability but changes the entire training paradigm.
- **Option B:** Add latent reasoning as a post-training phase (Phase 4). Take the fully-trained autoregressive model and teach it to also reason in continuous space. COCONUT uses this approach with curriculum training — is this reliable? Does the model's learned token-space reasoning transfer to latent space?
- **Option C:** Hybrid from the start. Interleave standard token prediction with latent reasoning objectives during Phase 1.

Which approach produces the strongest latent reasoning? Which is most computationally practical? Are there ablation results comparing these approaches?

### Q4: What is the relationship between VL-JEPA's embedding-prediction training objective and our Engram architecture?

This is a speculative but potentially high-value question. VL-JEPA trains by predicting embeddings rather than tokens. Our Engram system stores and retrieves factual knowledge as embeddings. Is there a synthesis here?

- Could we train Engram tables using a JEPA-style objective? Instead of the current plan (supervised from Wikipedia/Wikidata), train the Engram table embeddings such that the model can predict the embedding of an answer given a question — in embedding space, not token space.
- Would a JEPA-style Engram training objective produce higher-quality knowledge representations than supervised extraction?
- VL-JEPA uses a predictor module to predict target embeddings from context. Could harmony's EngramGatedResidual be viewed as a kind of "knowledge predictor" that predicts what knowledge is relevant given the current hidden state?
- If we train with a JEPA-style loss on Engram embeddings, does the Chronos temporal decay mechanism still work the same way? (It should — decay is applied to the retrieved embedding before it reaches the gate.)

### Q5: What are the computational costs and tradeoffs?

For a 0.5B model on edge devices (4090, RPi5):

- **Latency:** If the model performs N continuous thought steps per output token, latency increases ~N×. What is a practical N? COCONUT results suggest N=1-5 thought steps are useful. At what N does latent reasoning stop helping?
- **Memory:** Each thought step adds to the KV cache. At 42 bytes/vector (TurboQuant), how much additional memory does continuous thought add? Is there a way to use "ephemeral" KV entries for thought steps that don't persist after the reasoning loop?
- **Quality vs efficiency:** VL-JEPA achieves comparable quality with 2.85× fewer decoding operations. COCONUT achieves better accuracy with fewer tokens. But both were studied at much larger scale (>1B params). **Do latent reasoning benefits survive at 0.5B scale?** Small models may not have enough capacity for effective latent reasoning.
- **Selective decoding:** VL-JEPA only invokes the text decoder when needed. Could we use the UQ head to decide when the model needs to "think more" in embedding space vs when a single forward pass suffices? This would make the computational cost adaptive rather than fixed.

### Q6: What are the alternatives to full COCONUT/JEPA integration?

If full latent reasoning is too complex for v1, what intermediate steps could capture some of the benefit?

- **Pause tokens:** Add a learned `<think>` token that the model can emit to get extra computation without producing output. Simpler than COCONUT (still operates in token space) but provides some of the "extra thinking time" benefit.
- **Multi-pass self-refinement in embedding space:** Run the forward pass, take the final hidden state, project it back to the input space, and run again — without committing to tokens. A simplified version of COCONUT without the curriculum training.
- **Latent planning head:** Add a small predictor module (like VL-JEPA's) that operates in parallel with the main transformer and produces "plan embeddings" that are injected back via a mechanism similar to Engram. The model plans in latent space while generating in token space.
- **Depth-scaling:** Instead of adding reasoning steps at the sequence level, add adaptive depth — route uncertain tokens through additional transformer layers (similar to early exit but in reverse). This is "thinking harder" per token rather than "thinking longer."
- **Mixture-of-Depths:** Allow certain positions to skip layers while others get full depth computation. Thinking positions get all 24 layers; output positions may use fewer.

### Q7: How does latent reasoning interact with the "Broca's area" problem for our specific model?

The user framing: "Current LLMs run all thinking through a Broca's area-like section — it would be better if the model could think in embeddings directly."

This is the core question. In neuroscience terms:
- **Broca's area** handles speech production — converting thoughts to language
- **Wernicke's area** handles comprehension — converting language to thoughts
- Most cognition happens in neither — it's distributed across cortical regions operating on neural representations (not words)

Current LLMs conflate all three: understanding, reasoning, and output all happen through the same token-prediction mechanism. VL-JEPA and COCONUT begin to decouple these.

For harmony specifically:
- **Input processing (Wernicke's analog):** Already partially decoupled — Engram provides sub-symbolic knowledge injection that doesn't go through tokenization.
- **Output generation (Broca's analog):** Currently every reasoning step produces tokens. COCONUT/VL-JEPA would decouple this.
- **Internal reasoning:** Currently implicit within a single forward pass. Latent reasoning would make it explicit and iterative.

**Question:** Is there research on the neuroscience-inspired decomposition of transformer architectures into perception/reasoning/production modules? Specifically:
- Has anyone built a model where the "reasoner" is architecturally distinct from the "text decoder"?
- Does this decomposition improve performance, interpretability, or both?
- How does this relate to the "world model" concept in LeCun's JEPA framework (the world model predicts in representation space, the actor produces actions)?

### Q8: What changes to the GGUF format and candle inference engine would be needed?

We export trained PyTorch models to GGUF and run inference in Rust using the `candle` framework.

- Does the COCONUT continuous thought loop require new tensor operations not available in candle?
- How would continuous thought steps be represented in the GGUF model file? (Additional projection layers? A separate "thought mode" flag?)
- For selective decoding (VL-JEPA style), what metadata is needed to decide when to invoke the text decoder vs continue in embedding space?
- Are there existing GGUF models or candle implementations that support any form of latent reasoning?

## Context: What We've Already Built (In Rust/Candle)

```
crates/
  harmony-engram/
    chronos.rs         — Temporal decay (fully implemented)
    hash.rs            — N-gram hashing
    resolve.rs         — Shard lookup & embedding resolution

  harmony-inference/
    harmony_model.rs   — ct87 forward pass with BlockAttnRes, Engram callback, UQ features
    block_attnres.rs   — Block Attention Residuals (fully implemented)
    engram_residual.rs — EngramGatedResidual gated injection (fully implemented)
    engram_bridge.rs   — Sans-I/O async bridge
    kv_compress/       — TurboQuant (PolarQuant + QJL, fully implemented)
    uq_head.rs         — UQ head classifier (fully implemented)
    uq_features.rs     — Feature extraction

training/              — PyTorch training scaffold (newly added)
  ct87/model.py        — PyTorch mirror of harmony_model
  ct87/train.py        — Training loop with Muon + WSD schedule
  ct87/export_gguf.py  — PyTorch → GGUF export
```

## What We're Looking For

1. **Architectural clarity:** Exact mechanisms of VL-JEPA and COCONUT, with enough detail to implement in our system.
2. **Integration strategy:** How latent reasoning interacts with our five existing innovations — especially Engram and UQ.
3. **Training plan:** Where latent reasoning fits in our staged curriculum and what data/compute it requires.
4. **Scale viability:** Whether latent reasoning benefits exist at 0.5B scale on edge hardware.
5. **Incremental path:** If full integration is too ambitious for v1, what intermediate steps give us the most benefit?
6. **Corrections:** If our understanding of VL-JEPA or COCONUT is wrong in any way, please correct us. We'd rather have an accurate picture than confirm our assumptions.

## Desired Output Format

For each research question, provide:
- A direct answer with citations to specific papers/results
- Concrete architectural recommendations for harmony's ct87 model
- Risk assessment (what could go wrong, what's unproven)
- Estimated compute/engineering effort relative to our current Phase 0-3 plan
