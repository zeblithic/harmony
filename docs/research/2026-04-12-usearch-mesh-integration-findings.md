# USearch Mesh Integration Research Findings

**Source:** Gemini research report, 2026-04-12
**Topic:** Distributed HNSW vector search over 802.11s mesh with CAS

## Critical architectural decisions

### 1. DiskANN is wrong for our mesh

DiskANN's Vamana algorithm uses sequential I/O with strict data dependencies — it can't determine hop k+1 until hop k resolves. On local NVMe, 50 hops = ~5ms. Over 802.11s mesh (50ms per hop), 50 hops = 2.5 seconds. **Catastrophically misaligned with mesh latency.**

**Decision: USearch HNSW + Head Index, not DiskANN.**

### 2. Head Index pattern (from DistributedANN/Bing)

Upper HNSW layers (sparse, ~1% of vectors) replicated in local RAM on every node. Base layer (dense) distributed via CAS across mesh. Query flow:

1. Search Head Index in local RAM (microseconds)
2. Resolve entry points to specific CAS blobs / mesh nodes
3. Single targeted Zenoh query to correct node
4. Process dense neighborhood locally

**Result: Network latency incurred once per query, not per graph hop.**

### 3. CAS integration: Base + Delta pattern

- **Immutable Base Index:** Large USearch index stored as memory-mapped CAS blob. Read-only, served via `Index.view()`.
- **Mutable Delta Index:** Small in-RAM USearch index for recent additions.
- **Compound search:** Query both concurrently, fuse results.
- **Periodic compaction:** Merge Delta into Base via IGTM algorithm (~70% fewer distance computations than naive reinsertion). Serialize, BLAKE3 hash, publish new CID via Zenoh.
- **Cost amortized** over thousands of insertions, not per-vector.

### 4. Speculative prefetching for engram (better than zero-gating)

Use attention weights during LLM prefill phase to predict future engram needs:
1. Pool attention during prefill → semantic focus vector
2. Async Zenoh query to Head Index before decode begins
3. Retrieved vectors arrive just-in-time for subsequent layers
4. If fetch is late or mispredicted, fall back to zero-gating

**Strictly decouples retrieval from the critical path.**

### 5. RPi5 performance projections

| Dataset | Format | Total RAM | Search latency |
|---------|--------|-----------|---------------|
| Engram (100K x 128d) | i8 quantized | ~28 MB | <0.5 ms |
| Semantic (1M x 256-bit) | Binary/Hamming | ~100 MB | <1.0 ms |

Both fit comfortably in 8GB alongside model weights + KV cache. NEON-optimized Hamming via XOR + popcount.

### 6. Hybrid fallback via Rust traits

```rust
pub trait VectorMemory: Send + Sync {
    fn insert(&mut self, key: u64, vector: &[f32]) -> Result<()>;
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u64, f32)>>;
}
```

ExactHashMemory (xxhash) + USearchMemory as runtime-switchable backends. Shadow-traffic validation before promoting USearch to primary.

## Key papers cited

- **SHINE:** Scalable HNSW in disaggregated memory (graph-preserving distribution)
- **DistributedANN:** Bing's Head Index approach for sharded HNSW
- **IGTM/CGTM:** Efficient HNSW graph merge algorithms (70% fewer distance comps)
- **Ada-ef:** Adaptive exploration factor based on similarity distribution
- **AsyncTLS/PCR/SP-MoE:** Speculative prefetching for LLM inference
