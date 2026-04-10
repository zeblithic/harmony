# Memory Budget Analysis — Harmony TARGET Config

> ZEB-59 · 2026-04-09 · Based on commit state post-PR #198 (q8_0 KV cache)

## Model Architecture Summary

| Parameter | Value |
|-----------|-------|
| Total parameters | **474,205,870 (474M)** |
| Layers | 24 |
| Hidden dim | 1280 |
| Query / KV heads | 16 / 8 (GQA 2:1) |
| Head dim | 80 |
| FFN dim | 3413 |
| Vocab | 32,000 |
| Max seq len | 32,768 |
| Engram dim | 256 |

### Parameter Breakdown

| Component | Parameters | Notes |
|-----------|-----------|-------|
| 24× Transformer layers | 432,572,160 | 18M per layer |
| └─ Attention (Q/K/V/O + norms) | 4,915,360/layer | GQA: K/V projections are half Q/O |
| └─ MLP (gate/up/down) | 13,105,920/layer | SwiGLU: 3× hidden→ffn projections |
| └─ Layer norms | 2,560/layer | attn_norm + ffn_norm |
| Embedding (tied to lm_head) | 40,960,000 | 32K vocab × 1280 |
| EngramGatedResidual | 661,760 | key/value proj + norms + conv1d |
| BlockAttnRes | 8,960 | 7 learned query vectors |
| COCONUT ThoughtNorm | 1,281 | RMSNorm + gate bias |
| UQ head | 429 | Classifier + confidence head |
| final_norm | 1,280 | — |

## Model Weight Memory

| Format | Size | Notes |
|--------|------|-------|
| F32 (training) | **1,809 MB** | PyTorch training only |
| F16 (inference) | **905 MB** | Candle native inference |
| GGUF Q8_0 | **481 MB** | ~1.06 bytes/param |
| GGUF Q4_K | **274 MB** | Embeddings at Q8, rest at Q4_K |

## KV Cache Memory (all 24 layers)

KV cache grows dynamically via `Tensor::cat()` — no pre-allocation.
Memory is proportional to **current** sequence length, not max_seq_len.

### Per-format cost per token (all layers)

| Format | Bytes/token | Formula |
|--------|------------|---------|
| F16 | 61,440 | 24 × 2 × 8 heads × 80 dim × 2 bytes |
| Q8_0 | 32,256 | 24 × 2 × 8 heads × (80 dim + 4 scale) |
| TurboQuant | 16,896 | 24 × 2 × 8 heads × (4 + ⌈240/8⌉ + ⌈80/8⌉) |

### Full cache at key sequence lengths

| seq_len | F16 | Q8_0 | TurboQuant | Q8/F16 |
|---------|-----|------|------------|--------|
| 512 | 30 MB | 16 MB | 8 MB | 52.5% |
| 1,024 | 60 MB | 32 MB | 17 MB | 52.5% |
| 2,048 | 120 MB | 63 MB | 33 MB | 52.5% |
| 4,096 | 240 MB | 126 MB | 66 MB | 52.5% |
| 8,192 | 480 MB | 252 MB | 132 MB | 52.5% |
| 16,384 | 960 MB | 504 MB | 264 MB | 52.5% |
| 32,768 | 1,920 MB | 1,008 MB | 528 MB | 52.5% |

### Q8 mode peak memory

In Q8 mode, 23 layers are quantized (INT8) while 1 layer is temporarily
dequantized to F16 for the active forward pass. Peak = 23 × Q8 + 1 × F16:

| seq_len | Peak Q8 mode | Full F16 | Savings |
|---------|-------------|----------|---------|
| 4,096 | 131 MB | 240 MB | 45.5% |
| 8,192 | 262 MB | 480 MB | 45.5% |
| 32,768 | 1,046 MB | 1,920 MB | 45.5% |

### Three KV representations

| Format | Purpose | When used |
|--------|---------|-----------|
| F16 | Active compute | During layer forward pass (1 layer at a time) |
| Q8_0 | Hot inference | Between forward passes (23 idle layers) |
| TurboQuant | Cold storage | Session suspend/resume, hibernation |

## Harmony Subsystem Overhead

All Harmony-specific subsystems combined add **< 0.1 MB**:

| Subsystem | Memory | Notes |
|-----------|--------|-------|
| Engram shard cache | ~8 KB | Bounded by practical N-gram count |
| Engram cached embedding | 512 B | [1, 1, 256] × F16 |
| Engram token buffer | 40 B | 10 tokens × 4 bytes |
| UQ head parameters | 1.7 KB | 429 params × F32 |
| UQ feature buffer | 32 B | 8 × F32 |
| COCONUT ThoughtNorm | 5 KB | 1,281 params × F32 |
| COCONUT thought states | 10 KB | 4 steps × 1,280 × F16 |
| **Total** | **~25 KB** | **0.002% of model weights** |

## Speculative Decode Overhead (Forward-Looking)

Draft token KV entries are negligible relative to the main cache:

| Draft tokens (K) | Additional KV (Q8) | % of 4K context cache |
|-------------------|--------------------|-----------------------|
| 2 | 63 KB | 0.05% |
| 4 | 126 KB | 0.10% |
| 8 | 252 KB | 0.19% |

Speculative decoding is **not memory-constrained**. The MTP head itself
(ZEB-61) adds ~13.1M parameters (~7 MB at Q8, ~14 MB at F16) — the MLP portion
of a transformer layer, still negligible vs. the KV cache at long contexts.

## Deployment Scenarios

### Single Session

The 474M model fits all target devices at full 32K context, even in the
least compressed configuration:

| Config | Weights | KV @ 32K (Q8) | Total | Fits 4GB? | Fits 8GB? |
|--------|---------|---------------|-------|-----------|-----------|
| F16 weights + Q8 KV | 905 MB | 1,046 MB | 1,951 MB | ✓ | ✓ |
| Q8 weights + Q8 KV | 481 MB | 1,046 MB | 1,527 MB | ✓ | ✓ |
| Q4_K weights + Q8 KV | 274 MB | 1,046 MB | 1,320 MB | ✓ | ✓ |

### Concurrent Sessions

Model weights load once; KV cache is per-session. This is the key scaling
dimension for edge deployment with multiple users/conversations.

Per session at Q8: ~131 MB at 4K tokens, ~1,046 MB at 32K tokens.

| Device | Config | Headroom | Sessions @ 4K | Sessions @ 8K | Sessions @ 32K |
|--------|--------|----------|---------------|---------------|----------------|
| RPi5 8GB | Q4_K | 6,368 MB | 48 | 24 | 6 |
| RPi5 8GB | F16 | 5,738 MB | 43 | 21 | 5 |
| Apple Si 16GB | Q4_K | 13,060 MB | 99 | 49 | 12 |
| Apple Si 16GB | F16 | 12,430 MB | 95 | 47 | 11 |
| aarch64 4GB | Q4_K | 2,772 MB | 21 | 10 | 2 |
| aarch64 4GB | F16 | 2,142 MB | 16 | 8 | 2 |

> Device available memory = total RAM − OS overhead (1–3 GB) − runtime (~50 MB).

## Key Findings

### 1. The model is compact — memory is not a constraint for single sessions

At 474M parameters, even the worst-case configuration (F16 weights + F16 KV at
32K context = 2.8 GB) fits on a 4GB device. Weight quantization is a
performance optimization (memory bandwidth), not a memory-capacity requirement.

### 2. KV cache dominates at long contexts

At 4K tokens: weights = 67% of total, KV = 33%.
At 32K tokens: weights = 21% of total, KV = 79%.

Memory optimization efforts should focus on KV cache for long-context
workloads. The Q8_0 quantization (PR #198) delivers a consistent 45.5%
reduction with < 0.02 max absolute error.

### 3. Harmony subsystems are effectively free

Engram + UQ + COCONUT combined = 25 KB. These can be unconditionally enabled
on all deployment targets with zero memory concern.

### 4. Paged KV (ZEB-58) is about fragmentation and concurrency, not capacity

The current `Tensor::cat()` growth pattern copies the entire KV tensor on
every decode step — O(n²) total copies over a session. Paged KV eliminates
this with amortized O(1) per token. Benefits:

- **Reduced fragmentation**: Fixed-size pages vs. growing contiguous tensors
- **Prefix sharing**: Multiple sessions with the same system prompt share pages
- **Graceful eviction**: Under memory pressure, drop oldest pages vs. OOM
- **Predictable allocation**: Pages are fixed-size, reducing allocator stress

Priority: **Medium** for single-session. **High** if serving concurrent sessions.

### 5. Speculative decoding has ample headroom

Draft token KV overhead (K=4-8) is < 0.3 MB — negligible. The MTP head
(ZEB-61) adds ~10 MB of parameters. No memory-based constraints on speculative
decoding depth.

### 6. Q8_0 KV should be the default for all inference

The 45.5% memory savings with < 0.02 max error makes Q8 KV strictly superior
to F16 KV for autoregressive decode. The only reason to use F16 KV is
debugging or quality-sensitive evaluation.

## Recommended Default Configurations

| Device | Weights | KV | Max context | Rationale |
|--------|---------|-----|-------------|-----------|
| RPi5 8GB | Q4_K | Q8 | 32K | Maximize concurrent session headroom |
| Apple Silicon 16GB | F16 | Q8 | 32K | Quality-first; plenty of headroom |
| aarch64 4GB | Q4_K | Q8 | 32K | Fits comfortably even at max context |
| Multi-session server | Q8 | Q8 | 8K default | Balance session count vs. context |
