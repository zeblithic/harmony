# Edge Memory Budget Analysis: Inference + Vector Search Co-Residency

**Date:** 2026-03-27
**Status:** Draft
**Bead:** harmony-j2rd
**Type:** Research / Analysis (no implementation)

## Purpose

Determine whether Harmony's inference stack (LLM generation, KV cache, content cache, optional embedding search) can co-reside on constrained edge devices. Produce a decision tool: named configurations with exact byte budgets, a trade-off matrix, and per-tier recommendations.

## Hardware Tiers

### Tier 1: 512MB MIPS (MT7621 OpenWRT Router)

| Property | Value |
|---|---|
| RAM | 512 MB DDR3 |
| Address space | 32-bit |
| CPU | 880 MHz dual-core MIPS1004Kc |
| Storage | NOR/NAND flash (~10-20 MB/s sequential) |
| OS | OpenWRT Linux (minimal) |
| FPU | MIPS FPU (not soft-float, but slow) |
| GPU | None |

**Key constraint:** Flash-backed mmap is very slow. Page faults during inference add significant latency. The mmap working set (pages resident in RAM at any given time) must fit comfortably in available page cache, or inference throughput degrades severely.

### Tier 2: 4-8GB ARM64 (Raspberry Pi 5)

| Property | Value |
|---|---|
| RAM | 4 GB or 8 GB LPDDR4X |
| Address space | 64-bit |
| CPU | 2.4 GHz quad-core Cortex-A76 |
| Storage | SD card (~100 MB/s) or NVMe (~500+ MB/s) |
| OS | Raspbian / Ubuntu (heavier kernel) |
| FPU | Full NEON SIMD |
| GPU | VideoCore VII (not useful for LLM inference) |

**Key advantage:** NVMe-backed mmap is nearly transparent. Page cache pressure is rarely a concern. NEON SIMD accelerates quantized matmul significantly vs MIPS.

## Component Inventory

Every memory consumer in the harmony-node runtime, with formulas derived from the codebase.

### 1. Operating System Base

| | Tier 1 (OpenWRT) | Tier 2 (Raspbian) |
|---|---|---|
| Kernel + init | ~20 MB | ~80 MB |
| System services | ~10 MB | ~50 MB |
| Page cache / buffers | ~10 MB reserved | ~50 MB reserved |
| **Total OS base** | **~40 MB** | **~180 MB** |

Sources: OpenWRT memory profiling on MT7621 (typical fresh boot ~35-45 MB), Raspbian lite ~150-200 MB.

### 2. harmony-node Runtime

The harmony-node binary loads these subsystems unconditionally:

| Subsystem | Crate | Estimated Heap |
|---|---|---|
| Reticulum networking | harmony-reticulum | ~5 MB (path table, announce cache) |
| Zenoh pub/sub | harmony-zenoh + zenoh | ~15 MB (session, subscriptions, buffers) |
| Content (CAS + BlobStore) | harmony-content | ~5 MB base (excluding W-TinyLFU cache) |
| Crypto + Identity | harmony-crypto, harmony-identity | ~2 MB (keypairs, HKDF contexts) |
| Compute (WASM runtime) | harmony-compute + wasmtime | ~10 MB (compiler, module cache) |
| Workflow + misc | harmony-workflow, -peers, -contacts, etc. | ~5 MB |
| Binary text segment | Code pages (Rust + wasmtime + candle) | ~20 MB |
| **Total base runtime** | | **~62 MB** |

**Note:** The binary text segment (mapped code pages) is significant because harmony-node links wasmtime and candle — both large Rust crates. Zenoh is the largest heap cost. On Tier 1, a minimal Reticulum-only mode (no Zenoh) could save ~15 MB but loses pub/sub mesh capabilities.

### 3. W-TinyLFU Content Cache

Configurable. Memory formula from `docs/plans/2026-03-03-wtinylfu-cache-design.md`:

```
Cache memory = data_budget + Count-Min Sketch + LRU overhead
CMS = 4 × max_entries bytes (4-bit paired counters, 4 hash rows)
LRU overhead ≈ 48 bytes per entry (slab entry + HashMap bucket + pointers)
```

| Config | Data Budget | CMS | LRU Overhead | Total |
|---|---|---|---|---|
| 16 MB cache, 1K entries | 16 MB | 4 KB | 48 KB | ~16.1 MB |
| 32 MB cache, 4K entries | 32 MB | 16 KB | 192 KB | ~32.2 MB |
| 64 MB cache, 8K entries | 64 MB | 32 KB | 384 KB | ~64.4 MB |
| 128 MB cache, 16K entries | 128 MB | 64 KB | 768 KB | ~128.8 MB |
| 256 MB cache, 32K entries | 256 MB | 128 KB | 1.5 MB | ~257.6 MB |

The CMS and LRU overhead is negligible — the data budget dominates. Cache size is effectively a direct memory knob.

### 4. GGUF Model Weights (mmap'd)

Model weights are memory-mapped from storage, not fully loaded into RAM. The Linux page cache manages residency — only pages touched during the current forward pass are physically resident.

**Qwen3-0.6B GGUF file sizes** (estimated from confirmed Q8_0 = 639 MB):

| Quantization | File Size | Bits/param | Quality Impact |
|---|---|---|---|
| Q8_0 | ~639 MB | ~8.5 | Negligible quality loss |
| Q6_K | ~495 MB | ~6.6 | Minimal quality loss |
| Q5_K_M | ~430 MB | ~5.7 | Slight quality loss |
| Q4_K_M | ~365 MB | ~4.8 | Acceptable for most tasks |
| Q3_K_S | ~270 MB | ~3.5 | Noticeable on reasoning tasks |

**mmap working set estimate:** During a forward pass, the model touches one transformer layer at a time. Each layer's weights (attention projections + MLP + norms) are roughly `file_size / num_layers`. For Qwen3-0.6B with 28 layers:

| Quantization | Per-Layer | Working Set (2-3 layers) |
|---|---|---|
| Q4_K_M | ~13 MB | ~26-39 MB |
| Q3_K_S | ~10 MB | ~20-30 MB |

Plus the embedding table (~50-70 MB, touched on every forward pass) and the output projection (~same as embedding, may be tied). **Realistic mmap working set: ~80-120 MB for Q4_K_M, ~60-90 MB for Q3_K_S.** This is the physical RAM consumed by model weights during active inference.

**Between inference calls,** the kernel may evict model pages under memory pressure. The next forward pass then re-faults them in. On Tier 1 (slow flash), this adds noticeable latency. On Tier 2 (NVMe), it's nearly free.

### 5. KV Cache

Formula from `crates/harmony-inference/src/kv_compress.rs` (confirmed by tests):

```
Uncompressed:  28 layers × 8 KV heads × seq_len × 128 head_dim × 2 (K,V) × 2 bytes (f16)
             = 114,688 × seq_len bytes
             ≈ 112 KB per token

Compressed:    28 layers × 8 KV heads × seq_len × 2 (K,V) × 56 bytes (3-bit quantized)
             = 25,088 × seq_len bytes
             ≈ 24.5 KB per token
```

**Note:** The compressed formula accounts for both K and V tensors. Each 128-dim f16 vector (256 bytes) compresses to 56 bytes (8-byte min/scale header + 48 bytes packed 3-bit indices) = **4.57x reduction**.

| Context Length | Uncompressed | Compressed (3-bit) | Savings |
|---|---|---|---|
| 512 tokens | 56 MB | 12.2 MB | 43.8 MB |
| 1024 tokens | 112 MB | 24.5 MB | 87.5 MB |
| 2048 tokens | 224 MB | 49 MB | 175 MB |
| 4096 tokens | 448 MB | 98 MB | 350 MB |

KV compression is the single most impactful memory knob for inference-capable nodes. At 2048 context, it frees 175 MB — more than the entire Tier 1 OS + runtime combined.

### 6. Tokenizer

The `tokenizers` crate loads the full vocabulary into heap:

| Component | Size |
|---|---|
| Vocabulary (151K tokens for Qwen3) | ~5 MB |
| Merge table + special tokens | ~3 MB |
| **Total** | **~8 MB** |

Shared between the LLM and embedding model if both use the same tokenizer.

### 7. Oluo Embedding Model (Optional)

pplx-embed-v1-0.6B uses the same Qwen3 backbone. Same GGUF sizes as the LLM model, same mmap behavior. If both models run simultaneously, their mmap working sets compete for page cache.

| Component | Memory |
|---|---|
| Embedding model mmap working set | ~60-120 MB (same as LLM) |
| Second KV cache (not needed — embeddings are single-pass) | 0 |
| Embedding output buffer | ~1 MB |
| **Total Oluo model** | **~60-120 MB** |

**Important:** If the LLM and embedding model share the same Qwen3 backbone architecture, a future optimization could share weights (same GGUF file, different output head). This would eliminate the double mmap cost entirely. Not implemented today.

### 8. Oluo Trie Index (Optional)

From `docs/plans/2026-03-08-oluo-design.md`:

```
Per leaf entry: 64 bytes (32-byte CID + 32-byte Tier 3 binary vector)
Per internal node: 106 bytes
Entries per leaf: up to 256 (split threshold)
```

| Personal Index Size | Entries | Leaf Storage | Trie Nodes | Total |
|---|---|---|---|---|
| Small | 1,000 | 64 KB | ~500 B | ~65 KB |
| Medium | 10,000 | 640 KB | ~5 KB | ~645 KB |
| Large | 100,000 | 6.4 MB | ~50 KB | ~6.5 MB |
| Very large | 1,000,000 | 64 MB | ~500 KB | ~64.5 MB |

Personal-scope search with 100K entries is ~6.5 MB — negligible. The embedding model itself is the expensive part, not the index.

## Reference Configurations

### Tier 1: 512MB MIPS

#### T1-Relay: Mesh Relay Only (No Inference)

| Component | Memory |
|---|---|
| OpenWRT base | 40 MB |
| harmony-node runtime | 62 MB |
| W-TinyLFU cache (16 MB) | 16 MB |
| **Total** | **~118 MB** |
| **Headroom** | **~394 MB** |

Verdict: **Comfortable.** Plenty of room for page cache, kernel buffers, and burst allocations. The MT7621 is a natural mesh relay.

#### T1-Micro: Minimal Inference

| Component | Memory |
|---|---|
| OpenWRT base | 40 MB |
| harmony-node runtime | 62 MB |
| W-TinyLFU cache (16 MB) | 16 MB |
| Tokenizer | 8 MB |
| Model mmap working set (Q3_K_S) | ~70 MB |
| KV cache (512 ctx, compressed) | 12 MB |
| **Total** | **~208 MB** |
| **Headroom** | **~304 MB** |

Verdict: **Feasible.** 304 MB headroom absorbs mmap page cache fluctuation. 512 context tokens is short but enough for simple Q&A. The key question is whether MIPS at 880 MHz can produce tokens fast enough to be useful — memory isn't the bottleneck here, compute is.

#### T1-Stretch: Aggressive Inference

| Component | Memory |
|---|---|
| OpenWRT base | 40 MB |
| harmony-node runtime | 62 MB |
| W-TinyLFU cache (16 MB) | 16 MB |
| Tokenizer | 8 MB |
| Model mmap working set (Q3_K_S) | ~70 MB |
| KV cache (1024 ctx, compressed) | 25 MB |
| **Total** | **~221 MB** |
| **Headroom** | **~291 MB** |

Verdict: **Viable.** Only 13 MB more than T1-Micro thanks to KV compression. The 291 MB headroom is comfortable — the real constraint on Tier 1 is CPU speed, not RAM. Even Q4_K_M with 1024 compressed context would fit (~235 MB total, ~277 MB headroom).

**T1 finding: KV compression makes the difference between "inference barely fits" and "inference fits comfortably."** Without compression, 1024 context would consume 112 MB instead of 25 MB, pushing total to ~308 MB with only 204 MB headroom — still fits, but tight under mmap pressure.

### Tier 2: 4-8GB ARM64

#### T2-Standard: Comfortable Inference (4GB)

| Component | Memory |
|---|---|
| Raspbian base | 180 MB |
| harmony-node runtime | 62 MB |
| W-TinyLFU cache (64 MB) | 64 MB |
| Tokenizer | 8 MB |
| Model mmap working set (Q4_K_M) | ~100 MB |
| KV cache (2048 ctx, uncompressed) | 224 MB |
| **Total** | **~638 MB** |
| **Headroom (4GB)** | **~3.4 GB** |

Verdict: **Comfortable.** No need for KV compression at this tier with 2048 context. The 3.4 GB headroom is more than enough for the full model to be resident in page cache (365 MB file), kernel buffers, and any burst allocations.

#### T2-Full: Everything On (4GB)

| Component | Memory |
|---|---|
| Raspbian base | 180 MB |
| harmony-node runtime | 62 MB |
| W-TinyLFU cache (128 MB) | 128 MB |
| Tokenizer (shared) | 8 MB |
| LLM mmap working set (Q4_K_M) | ~100 MB |
| KV cache (2048 ctx, compressed) | 49 MB |
| Oluo embedding model mmap | ~100 MB |
| Oluo trie index (100K entries) | 7 MB |
| **Total** | **~634 MB** |
| **Headroom (4GB)** | **~3.4 GB** |

Verdict: **Comfortable.** Even with Oluo and a 128 MB cache, only 634 MB. KV compression here frees 175 MB which is nice but not necessary — it extends the headroom for page cache. Both full GGUF files (365 MB LLM + 365 MB embeddings = 730 MB) can be fully resident in the remaining 3.5 GB.

#### T2-Extended: Maximum Context (8GB)

| Component | Memory |
|---|---|
| Raspbian base | 180 MB |
| harmony-node runtime | 62 MB |
| W-TinyLFU cache (256 MB) | 256 MB |
| Tokenizer (shared) | 8 MB |
| LLM mmap working set (Q4_K_M) | ~100 MB |
| KV cache (4096 ctx, compressed) | 98 MB |
| Oluo embedding model mmap | ~100 MB |
| Oluo trie index (100K entries) | 7 MB |
| **Total** | **~811 MB** |
| **Headroom (8GB)** | **~7.4 GB** |

Verdict: **Comfortable.** 4096 context with compression is only 98 MB. Without compression it would be 448 MB — still fits on 8GB but eats into page cache headroom. On 4GB, 4096 uncompressed (448 MB KV alone) would push total to ~1.1 GB, still fine but less room for both full model files to be resident.

## Trade-Off Matrix

Five tunable knobs and their impact:

| Knob | Options | Memory Delta | Capability Impact |
|---|---|---|---|
| **Model quantization** | Q4_K_M → Q3_K_S | −95 MB mmap file, −30 MB working set | Noticeable quality loss on reasoning and instruction following. Acceptable for simple generation. |
| **Context length** | 2048 → 1024 → 512 | −112/−56 MB uncomp, −24.5/−12 MB comp | Shorter conversations, can't process long prompts. 512 is minimum useful. |
| **W-TinyLFU cache** | 128 → 64 → 16 MB | Direct 1:1 (you set what you get) | Smaller cache = more mesh fetches, higher content latency. 16 MB is minimum for usable hit rate. |
| **KV compression** | OFF → ON | −175 MB at 2048 ctx, −88 MB at 1024 | 3-bit lossy: small reconstruction error per step, compounds over long generations. Negligible for short outputs. |
| **Oluo** | Full → Delegated | −60-120 MB (model) − 7 MB (index) | Delegated = semantic search via mesh, adds network round-trip. Personal search becomes remote. |

### Priority Order for Freeing Memory

If under pressure, shed in this order (most expendable first):

1. **Oluo** — delegate to mesh heavy-hitters. Loses local semantic search but all content is still accessible.
2. **Cache size** — reduce to 16 MB. Increases mesh traffic but doesn't lose functionality.
3. **KV compression** — enable. Small quality cost, large memory win.
4. **Context length** — reduce. Directly impacts conversation quality.
5. **Model quantization** — downgrade. Last resort — quality impact is most noticeable.

## Verdict Summary

| Config | Total | Target RAM | Headroom | Verdict |
|---|---|---|---|---|
| **T1-Relay** | 118 MB | 512 MB | 394 MB | Comfortable |
| **T1-Micro** (Q3_K_S, 512ctx, compressed) | 208 MB | 512 MB | 304 MB | Feasible |
| **T1-Stretch** (Q3_K_S, 1024ctx, compressed) | 221 MB | 512 MB | 291 MB | Viable |
| **T2-Standard** (Q4_K_M, 2048ctx, no compression) | 638 MB | 4 GB | 3.4 GB | Comfortable |
| **T2-Full** (Q4_K_M, 2048ctx, compressed, Oluo) | 634 MB | 4 GB | 3.4 GB | Comfortable |
| **T2-Extended** (Q4_K_M, 4096ctx, compressed, Oluo) | 811 MB | 8 GB | 7.4 GB | Comfortable |

## Recommendations

### Tier 1 Default: T1-Relay

Most MT7621 nodes should run as mesh relays + content caches, not inference nodes. The memory fits, but the 880 MHz MIPS CPU is the real bottleneck — even at Q3_K_S, expect <1 token/second. Inference is possible but not practical for interactive use. Best use: background summarization, keyword extraction, or other non-latency-sensitive tasks triggered by mesh events.

**If inference is needed on Tier 1:** Use T1-Stretch (Q3_K_S, 1024 context, KV compression ON, 16 MB cache). Memory is not the constraint — CPU is.

### Tier 2 Default: T2-Standard

Q4_K_M, 2048 context, uncompressed KV, 64 MB cache, no Oluo. This is the "it just works" configuration. KV compression is unnecessary at this tier for standard context lengths.

**Dial-up guidance for Tier 2:**
- Have 4GB and want Oluo? → Enable it, switch to T2-Full (still 3.5 GB headroom)
- Have 8GB and want long context? → T2-Extended with 4096 context + compression
- Want even longer context (8K+)? → KV compression becomes essential — 8192 tokens uncompressed = 896 MB KV cache alone

### Cross-Tier: Mesh Delegation Strategy

For a heterogeneous mesh with both tiers:
- **Tier 1 nodes:** Relay, cache, route. Optionally run background inference for low-priority tasks.
- **Tier 2 nodes:** Run inference, Oluo search, and serve as mesh heavy-hitters for semantic queries.
- **Prefill sharing** (harmony-hbf0): A Tier 2 node can prefill a prompt and distribute the compressed KV cache via CAS. Tier 1 nodes fetch the cache and continue generation from the prefill point, skipping the expensive prompt evaluation.

## Open Questions (Future Beads)

1. **Engram co-residency** (harmony-8hjt): The Engram manifest (50M shard CIDs = 1.6 GB) cannot fit on either tier in full. Requires a sparse/streaming manifest design. Separate analysis needed.
2. **Paged KV cache** (harmony-643): Dynamic allocation could reduce peak KV memory by reclaiming pages for completed conversations.
3. **Actual mmap profiling on MT7621:** The working set estimates are theoretical. Profiling on real hardware would validate the page cache assumptions, especially the flash read latency impact.
4. **MIPS inference throughput:** Memory fits, but tokens/second on 880 MHz MIPS is the real feasibility question for Tier 1 inference. Needs benchmarking.
5. **Shared embedding model weights:** If LLM and Oluo use the same Qwen3 backbone, sharing the GGUF file would eliminate ~365 MB of mmap duplication.
6. **Q3_K_S quality validation:** Need perplexity benchmarks on Qwen3-0.6B Q3_K_S vs Q4_K_M to quantify the actual quality delta for Tier 1 use cases.
