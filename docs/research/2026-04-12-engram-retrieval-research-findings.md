# Engram Retrieval Research Findings

**Source:** Gemini research report, 2026-04-12
**Prompt:** `docs/research/2026-04-12-engram-retrieval-research-prompt.md`

## Summary

The research confirms our experimental findings and provides a clear strategy hierarchy for making the engram system work at edge scale.

## Key findings

### 1. Root cause confirmed

The gating zeroing out was the mechanism working correctly — protecting the model from noise. The synthetic table's random embeddings create an adversarial signal when accessed via semantically-structured keys. The fix is table content, not architecture.

### 2. Gated residual injection is optimal for edge

RETRO's chunked cross-attention requires O(N^2) dense matrix multiplication and a parallel encoder pathway — too expensive for 474M on RPi5. The DeepSeek Engram / Harmony gated residual approach (element-wise multiply + add) is the right mechanism. No architecture change needed.

### 3. Strategy ranking (by feasibility)

1. **Teacher distillation + USearch ANN** — Use a larger model (Mistral 7B or similar) to generate semantically rich embeddings for the corpus. Retrieve via USearch (189x faster than FAISS, 3K SLOC, ARM native). Highest probability of success.

2. **DiskANN for scaling** — SSD-backed Vamana graph for billion-scale tables. <10ms retrieval from NVMe. Decouples memory capacity from RAM constraints.

3. **COCONUT latent procedural library** — Cache successful continuous thought trajectories as engram entries. Converts memory from declarative (facts) to procedural (reasoning patterns). Experimental.

4. **Learned binary hashing** — Joint optimization of hash function + table so Hamming distance correlates with semantic similarity. Preserves O(1) lookup. Requires structured table first.

### 4. Library recommendations

- **USearch** (unum-cloud/usearch): HNSW with minimal deps, 189x faster than FAISS at 100M scale, native int8 support, ARM/WASM portable
- **DiskANN**: Microsoft Research, Vamana graph with alpha-RNG pruning, SSD-backed, <10ms at billion scale
- **SegPQ**: 4.7x codebook compression for edge memory budgets, 3.3% compute overhead

### 5. Mesh deployment considerations

- Semantic clustering creates hot shards (vs uniform xxhash distribution)
- Solution: Head Index (compressed local map of global semantic space) in RAM
- Latency hiding: gate defaults to zero during async network retrieval — model never degrades below parametric capability
- Align with existing content-addressed shard architecture

## Relevant papers

- kNN-LM (Khandelwal et al.): Token-level datastore, too expensive at edge but informs design
- SPALM: Semi-parametric LM with kNN + gating, closest to Harmony architecture
- RETRO (Borgeaud et al. 2022): Chunked cross-attention, too expensive for edge
- Memorizing Transformers (Wu et al. 2022): kNN attention + sigmoid gating
- ExplicitLM: Decoupled knowledge via external memory banks
- DRAG: Distilling RAG from LLM to SLM via evidence graphs
- SegPQ: Learned codebook compression for constrained memory
- LOKA: Knowledge codebook with conflict-free semantic clustering
- COCONUT (Hao et al. 2024): Continuous latent reasoning
