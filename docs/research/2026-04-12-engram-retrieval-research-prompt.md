# Engram Retrieval Research Prompt

## Context for the researcher

We're building **Harmony**, a decentralized mesh computing platform with an on-device language model (474M parameter, Qwen3-derived). The model includes an **Engram** system — an external conditional memory that injects context-dependent embeddings into the transformer hidden state via a gated residual mechanism (inspired by DeepSeek's gating patterns).

### Current architecture

- **Engram table:** A flat embedding table `[total_entries, engram_dim]` stored on disk/mesh, distributed across nodes via content-addressed shards
- **Key generation:** N-gram tokens (bigrams/trigrams) are hashed via xxhash64 with multiple seeds (multi-head), producing table indices
- **Injection:** Retrieved embeddings pass through a gated residual module (key/value projection, sigmoid dot-product gate, causal depthwise conv1d, SiLU) and are added to the hidden state at a configured transformer layer
- **Target deployment:** Edge devices (RPi5 8GB, mobile, mesh routers). Memory and compute are constrained.

### What we've tried and learned

We ran 6 experiments attempting to replace xxhash keys with learned latent projection keys:

1. **xxhash engram (baseline):** Works. -0.49% val loss. The model learns to use the consistent (but random) embeddings that xxhash deterministically retrieves.

2. **Latent projection as auxiliary task:** Train a 2-layer MLP (Linear→SiLU→Linear→Tanh) with InfoNCE contrastive loss alongside the model. Contrastive loss converges (5.27→2.92), but using the projection's keys at inference retrieves from 99.99% different table indices than xxhash. Model ignores the unfamiliar engram embeddings.

3. **Fine-tuning with projection keys (2k steps):** The gating atrophied entirely — model learned to zero out the engram gate rather than adapt to the new retrieval distribution.

4. **Training from scratch with frozen projection keys:** Model ignored engram entirely.

5. **End-to-end co-training (contrastive loss + projection keys for retrieval):** Same result — model ignored engram. Contrastive loss converged (8.13→2.21), projection learned topology, but the model still zeroed out the gate.

**Root cause:** The engram table contains random synthetic embeddings. xxhash works because it's *deterministic* — same tokens always produce the same (random but consistent) embedding, creating learnable patterns the model can memorize. Projection keys fail because nearby keys into a random table don't retrieve semantically related embeddings. The contrastive loss preserves embedding topology in key space, but that topology has no correspondence to table content.

### Key insight

The bottleneck is **table content**, not key quality. For learned keys to outperform xxhash, the table itself must have semantic structure where nearby keys retrieve related information.

## Research questions

### 1. Structured engram table construction

How should we build an engram table with semantic structure?

- **RETRO-style:** Chunk the training corpus, embed chunks with a pre-trained model, store as the engram table. N-gram keys retrieve contextually relevant chunk embeddings.
- **Learned table:** Make the engram table trainable (essentially an additional embedding layer). But this loses the "external memory" property — it's just more parameters.
- **Knowledge distillation:** Distill knowledge from a larger model into the engram table structure, so retrieved embeddings carry genuine information.
- **Corpus statistics:** Build table entries from actual n-gram co-occurrence statistics or contextual embeddings observed during pre-training.

Which approaches are tractable at 474M scale? What table sizes and embedding dimensions make sense?

### 2. Retrieval mechanism for edge deployment

If we move beyond hash-based lookup to similarity-based retrieval, what's feasible on edge devices?

- **Approximate nearest neighbor (ANN):** FAISS, ScaNN, or custom implementations. What's the latency overhead for ANN search in a table of 100K-1M entries on an RPi5?
- **Product quantization (PQ):** Compress embeddings, search in compressed space. How much quality loss at what compression ratio?
- **Locality-sensitive hashing (LSH):** Our binary key approach IS LSH — but the table needs to be built to respect the LSH structure. How to construct a table where the LSH neighborhoods are semantically coherent?
- **Learned hashing with structured table:** Train the hash function and table together so that the hash naturally maps to semantically appropriate entries. Is this just a reformulation of what we tried, or is there a fundamentally different approach?

### 3. RETRO and retrieval-augmented generation at small scale

RETRO (Borgeaud et al. 2022) uses a frozen retrieval database with chunked cross-attention. At 474M parameters and edge deployment:

- Can RETRO's chunked cross-attention be simplified to our gated residual injection?
- How does RETRO handle the table construction? What do the retrieved chunks look like?
- Is there a simplified version of RETRO that works at sub-1B scale?
- How does kNN-LM (Khandelwal et al.) compare — it uses the full hidden state as a query into a datastore of (context, next-token) pairs.

### 4. Connection to COCONUT / continuous latent reasoning

We also have COCONUT continuous thought (Hao et al. 2024) implemented. COCONUT keeps reasoning in continuous latent space rather than projecting to tokens.

- How might engram retrieval interact with continuous thought? Could retrieved embeddings serve as "external latent thoughts" that augment the model's internal reasoning?
- Is there a way to build the engram table as a library of latent reasoning patterns rather than linguistic embeddings?
- Does the retrieve-then-reason paradigm (retrieve external context, then reason over it in latent space) have precedent in the literature?

### 5. What we might be missing

- Are there approaches to external memory/retrieval for small language models that we haven't considered?
- The gated residual injection works well with xxhash (deterministic) keys. Is there a way to make it work with *any* retrieval mechanism, or is the gating fundamentally dependent on deterministic lookup?
- Could the solution be simpler than we think — e.g., a larger model that doesn't need external memory, or a different injection point, or a different gating mechanism?
- How do Memorizing Transformers (Wu et al. 2022), SPALM (sparse retrieval), or other memory-augmented approaches handle the table construction problem?

### 6. Practical considerations for mesh deployment

The engram table is distributed across mesh nodes via content-addressed shards. This means:

- Retrieval latency includes network hops (not just compute)
- The table can be much larger than local memory (paged from mesh)
- Shard-level caching matters — frequently accessed regions should be local
- The key generation scheme determines which shards get accessed (access pattern)

How do these constraints affect the choice of retrieval mechanism? Does the distributed nature favor hash-based (predictable access patterns) over similarity-based (unpredictable) retrieval?

## What we're looking for

1. **Concrete approaches** we can try next, ranked by likelihood of working at our scale
2. **Architecture patterns** from the literature that solve the "structured external memory for small LMs" problem
3. **Trade-off analysis** between retrieval quality and edge compute/latency constraints
4. **Anything we're missing** — alternative framings of the problem, approaches we haven't considered

Please cite specific papers and describe their approaches in enough detail that we can evaluate feasibility for our architecture.
