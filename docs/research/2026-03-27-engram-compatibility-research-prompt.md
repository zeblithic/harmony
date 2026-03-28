# Gemini Deep Research Prompt: Engram Embedding Compatibility Across Model Families

## Background

We are building HARMONY, a decentralized mesh computing platform that runs inference on extremely constrained edge devices (512MB RAM, 880MHz MIPS). We have two key technologies:

1. **TurboQuant** — compresses the KV cache during inference to 3 bits/channel, reducing dynamic memory usage during generation.

2. **Engram** (from DeepSeek) — a "conditional memory" architecture that decouples static factual knowledge from model weights by storing facts in external hash-indexed embedding tables. Instead of the model memorizing facts in its feed-forward network (FFN) layers, it does deterministic O(1) hash lookups into these external tables to retrieve pre-computed knowledge embeddings.

We have already implemented:
- A `harmony-engram` Rust crate that computes N-gram → xxhash64 → shard lookups and aggregates multi-head f16 embeddings into f32 vectors
- Content-addressed storage (CAS) for distributing Engram table shards across the mesh network
- A GGUF model loader and inference engine for Qwen3-0.6B (using the `candle` Rust ML framework)

The fundamental question: **Can we use Engram-style knowledge tables with Qwen3 (a non-DeepSeek model)?**

## Research Questions

### Q1: How does DeepSeek's Engram conditional memory actually work in the forward pass?

Please provide a detailed technical explanation of:
- Where exactly in the transformer block do Engram embeddings get injected? Is it:
  - Before the attention mechanism (augmenting the input)?
  - After the FFN layers (as an additive residual)?
  - As a replacement for specific FFN layers?
  - As a separate "memory attention" head that attends to retrieved embeddings?
- What is the architecture of the "conditional memory" module? Is it a separate sub-network, a gating mechanism, or a simple addition?
- How does the model learn when to use Engram retrieval vs. its own FFN computation? Is there an explicit routing/gating signal?
- What are the input/output dimensions? Specifically: what is the relationship between the Engram embedding dimension and the transformer's hidden state dimension?

### Q2: Are Engram embeddings model-specific or transferable?

- The Engram tables are trained jointly with the model — the embeddings live in the model's learned representation space. Does this mean the tables are inherently tied to DeepSeek's weight space?
- Has anyone published results on using Engram tables trained on one model family with a different model? (e.g., DeepSeek tables with Llama, Qwen, or Mistral?)
- What is the theoretical basis for transferability? Models trained on similar data should develop similar internal representations (the "convergent representation" hypothesis / platonic representation) — does this apply to Engram embedding spaces?
- Is there a quantitative measure of cross-model compatibility? (e.g., CKA similarity, SVCCA, or centered kernel alignment between hidden states of different models)

### Q3: If not directly compatible, what alignment techniques exist?

- **Procrustes alignment**: Can we learn a simple orthogonal transformation W such that DeepSeek_embedding × W ≈ Qwen3_hidden_state?
- **Linear probing**: Can a single learned linear layer project Engram embeddings into Qwen3's space?
- **Adapter networks**: Are there lightweight adapter approaches (e.g., LoRA-style) that can bridge embedding spaces with minimal parameters?
- What training data/compute would each approach require?
- Has this been done for cross-model embedding alignment in other contexts? (e.g., multilingual embedding alignment, model stitching)

### Q4: Alternative — training Engram tables specifically for Qwen3

If cross-model transfer isn't feasible:
- What is the training procedure for Engram tables? Is it:
  - A post-training step applied to an already-trained model? (preferred — we don't want to retrain Qwen3)
  - Part of the original pre-training? (would mean we can't use Qwen3 at all with Engram)
  - A fine-tuning step that can be applied to any model?
- What compute resources are required? (GPU-hours, data requirements)
- Are there open-source implementations of Engram table training?
- Has anyone published a "bolt-on" Engram-style knowledge extraction procedure for existing models?

### Q5: Simpler alternatives that achieve similar goals

If full Engram integration is too model-specific, what simpler approaches could achieve the "distributed knowledge" vision on edge devices?

- **Retrieval-augmented generation (RAG) with hash-indexed knowledge**: Use the same N-gram → hash → CAS lookup pipeline, but instead of injecting embeddings into the forward pass, prepend retrieved text/facts to the prompt as context. Simpler but uses context window budget.
- **Mixture-of-Experts with remote experts**: Some FFN layers run locally, others are "remote experts" fetched from mesh nodes. The model routes tokens to local or remote experts based on content.
- **Knowledge distillation into embedding tables without Engram's specific architecture**: Train a simple key-value store where keys are token N-grams and values are "knowledge tokens" that get prepended/injected. Less sophisticated than Engram but more model-agnostic.
- **Contextual document retrieval via vector similarity**: Skip N-gram hashing entirely, use lightweight embedding models to find relevant knowledge chunks from CAS, inject as context.

### Q6: The "conditional memory" architecture — is it open?

- Is DeepSeek's Engram architecture described in a published paper with sufficient detail to reproduce?
- Are there open-source implementations beyond DeepSeek's own? (If Engram is proprietary, we need alternatives.)
- Are there similar architectures from other research groups? (e.g., RETRO from DeepMind, kNN-LM from Facebook, REALM from Google, Memorizing Transformers)
- Which of these alternatives best maps to our constraints: deterministic O(1) hash lookup, f16 embedding tables distributable via CAS, and a sub-1B parameter local model?

### Q7: Practical integration with candle (Rust ML framework)

Our inference engine uses `candle` (Hugging Face's Rust ML framework) with the `candle-transformers` Qwen3 implementation.

- Can Engram-style embedding injection be implemented as a candle module? (i.e., a `Module` that sits in the forward pass and optionally retrieves/injects external embeddings)
- Does candle's tensor operation set support the needed operations? (embedding lookup, gating, residual addition)
- Are there examples of RAG-style or retrieval-augmented candle inference pipelines?
- Would we need to modify the Qwen3 model definition, or can this be done as a wrapper/adapter around the existing model?

## Context for Answering

### Our constraints:
- **512MB RAM** on edge devices (OpenWRT routers with MIPS MT7621)
- **Qwen3-0.6B** is our target model (Q4_K_M or Q3_K_S quantization)
- Model weights are **memory-mapped from flash** (not fully loaded into RAM)
- The Engram table is **distributed across the mesh** via CAS — edge nodes fetch only the shards they need
- We have a working Engram client in Rust that handles the hash → shard → embedding pipeline
- The local model should handle **reasoning**, while Engram handles **factual recall**
- We're targeting **1-5 tokens/second** on edge (not real-time conversation)

### What we need from this research:
1. A clear recommendation: **full Engram integration** vs. **simpler RAG-style approach** vs. **custom knowledge table training** for Qwen3
2. The technical details needed to implement whichever approach is recommended
3. An honest assessment of what's possible today vs. what requires novel research
4. Specific papers, codebases, or tools we should look at

### Papers to check (starting points):
- DeepSeek's Engram paper (if published — check arxiv for "DeepSeek Engram" or "conditional memory")
- RETRO: "Improving language models by retrieving from trillions of tokens" (Borgeaud et al., 2022)
- kNN-LM: "Generalization through Memorization: Nearest Neighbor Language Models" (Khandelwal et al., 2020)
- REALM: "REALM: Retrieval-Augmented Language Model Pre-Training" (Guu et al., 2020)
- Memorizing Transformers: "Memorizing Transformers" (Wu et al., 2022)
- Model Stitching: "Model soups" and "Git Re-Basin" for cross-model alignment
- Platonic Representation Hypothesis: "The Platonic Representation Hypothesis" (Huh et al., 2024)
