# Phase 0f: PyTorch Training Scaffold — Design Spec

**Epic:** harmony-ct87
**Phase:** 0f (blocked on 0e — complete)
**Status:** Design approved
**Date:** 2026-04-07

## Goal

Define the ct87 model architecture in PyTorch and implement a minimal training
loop that proves the architecture trains. This is the first Python code in the
repository — a greenfield addition to a Rust project.

The PyTorch model must be a faithful mirror of the candle `HarmonyModel` so that
Phase 0g can export trained weights to GGUF for candle inference.

## Architecture

### Relationship to Existing Code

The training scaffold lives in a new top-level `training/` directory, alongside
`crates/`. It is pure Python with no Rust interop.

```
harmony/
  crates/harmony-inference/    # Rust: candle inference (existing)
  training/                    # Python: PyTorch training (new, Phase 0f)
```

The two codebases define the same model architecture independently. The candle
version is the source of truth for architecture decisions. The PyTorch version
mirrors it exactly so weights are portable via GGUF (Phase 0g).

### Project Layout

```
training/
  pyproject.toml
  ct87/
    __init__.py
    model.py        # HarmonyModelConfig, all nn.Modules, HarmonyModel
    optim.py        # Muon optimizer, WSD learning rate schedule
    train.py        # Training loop, CLI entry point
  tests/
    test_model.py   # Model construction, forward shapes, config parity
    test_optim.py   # Muon step correctness, WSD schedule values
    test_train.py   # End-to-end: overfit tiny model on small batch
```

### Dependencies

Minimal:
- `torch >= 2.2`
- `safetensors` — checkpoint saving (HF standard, cleaner than torch.save)
- `datasets` — HuggingFace datasets for pre-tokenized data loading
- `pytest` — testing

No W&B, tensorboard, accelerate, deepspeed, or lightning.

## Model Architecture (`model.py`)

### HarmonyModelConfig

A `@dataclass` mirroring the Rust struct field-for-field:

```python
@dataclass
class HarmonyModelConfig:
    num_layers: int
    hidden_dim: int
    num_query_heads: int
    num_kv_heads: int
    head_dim: int
    ffn_dim: int
    vocab_size: int
    max_seq_len: int
    rope_theta: float
    rms_norm_eps: float
    layers_per_block: int
    engram_injection_layer: int  # Present for config parity, unused in 0f
    engram_dim: int              # Present for config parity, unused in 0f
    tie_embeddings: bool
```

Convenience constructors:

- `HarmonyModelConfig.target()` — 24 layers, 1280 hidden, 16 query heads,
  8 KV heads, head_dim 80, ffn_dim 3413, vocab 32000, max_seq 32768,
  rope_theta 1e6, rms_norm_eps 1e-6, 3 layers_per_block, engram_layer 2,
  engram_dim 256, tie_embeddings True
- `HarmonyModelConfig.tiny()` — 8 layers, 512 hidden, 8 query heads,
  4 KV heads, head_dim 64, ffn_dim 1365, vocab 32000, max_seq 4096,
  rope_theta 1e6, rms_norm_eps 1e-6, 2 layers_per_block, engram_layer 2,
  engram_dim 128, tie_embeddings True

Derived properties:
- `num_blocks` — `num_layers // layers_per_block`
- `num_kv_groups` — `num_query_heads // num_kv_heads`

### Layer Building Blocks

All linear projections use **no bias** (matching candle/Qwen3 style).

```
RMSNorm:
  weight: [dim]                    # Initialized to ones
  eps: float                       # From config.rms_norm_eps
  Forward: x * rsqrt(mean(x^2) + eps) * weight

RotaryEmbedding:
  cos_cached: [max_seq_len, head_dim/2]   # Precomputed
  sin_cached: [max_seq_len, head_dim/2]
  Forward: apply_rotary(q_or_k, cos, sin)  # Interleaved or split-half

Attention:
  q_proj: Linear    [hidden_dim, num_query_heads * head_dim, bias=False]
  k_proj: Linear    [hidden_dim, num_kv_heads * head_dim, bias=False]
  v_proj: Linear    [hidden_dim, num_kv_heads * head_dim, bias=False]
  o_proj: Linear    [num_query_heads * head_dim, hidden_dim, bias=False]
  q_norm: RMSNorm   [head_dim]     # Per-head norm
  k_norm: RMSNorm   [head_dim]     # Per-head norm
  Forward:
    q = q_proj(h) -> reshape [b, seq, n_q_heads, head_dim] -> q_norm -> RoPE
    k = k_proj(h) -> reshape [b, seq, n_kv_heads, head_dim] -> k_norm -> RoPE
    v = v_proj(h) -> reshape [b, seq, n_kv_heads, head_dim]
    k, v = repeat_kv(k, v, num_kv_groups)  # GQA broadcast
    attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    out = o_proj(attn.reshape(b, seq, -1))

Mlp (SwiGLU):
  gate_proj: Linear  [hidden_dim, ffn_dim, bias=False]
  up_proj: Linear    [hidden_dim, ffn_dim, bias=False]
  down_proj: Linear  [ffn_dim, hidden_dim, bias=False]
  Forward: down_proj(SiLU(gate_proj(x)) * up_proj(x))

TransformerLayer:
  attn_norm: RMSNorm [hidden_dim]
  attn: Attention
  ffn_norm: RMSNorm  [hidden_dim]
  mlp: Mlp
  Forward: h = h + attn(attn_norm(h)); h = h + mlp(ffn_norm(h))
```

### BlockAttnRes

Mirrors the candle `BlockAttnRes`. Learned pseudo-query vectors at block
boundaries enable deep layers to recall early-layer features.

```python
class BlockAttnRes(nn.Module):
    queries: nn.ParameterList   # num_blocks - 1 parameters, each [1, 1, hidden_dim]
```

Two methods matching the Rust API:

`notify_layer_output(layer_idx, hidden_state, state)` — called after each
transformer layer. At block boundaries (last layer of block), stores hidden
state in `state` (a plain `list` created per forward call).

`block_input(block_idx, hidden_state, state)` — called at the start of each
block (block > 0). Computes dot-product attention between the block's
pseudo-query and all preceding summaries + current hidden state, applies
softmax, and returns the weighted sum:

```
candidates = [summary_0, ..., summary_{k-1}, hidden_state]
scores_j = sum(query_k * candidate_j, dim=-1) / sqrt(hidden_dim)
    # → [batch, seq_len, 1] per candidate
stacked = cat(scores, dim=-1)    # → [batch, seq_len, num_candidates]
alpha = softmax(stacked, dim=-1)
output = sum(alpha_j * candidate_j)  # per-position weighted sum
```

State management: a plain Python list, created at the top of each forward call
and passed through the layer loop. Not stored on the module.

Gradient flow: summaries are live tensors from the forward pass, so gradients
flow back through block boundaries into earlier layers. This is intentional.

### HarmonyModel

```python
class HarmonyModel(nn.Module):
    embed_tokens: nn.Embedding     [vocab_size, hidden_dim]
    layers: nn.ModuleList          [num_layers x TransformerLayer]
    final_norm: RMSNorm            [hidden_dim]
    lm_head: nn.Linear             [hidden_dim, vocab_size, bias=False]
    block_attnres: BlockAttnRes
    rotary_emb: RotaryEmbedding    # Shared across all layers
    config: HarmonyModelConfig
```

**Tied embeddings:** When `config.tie_embeddings` is True, `lm_head.weight`
is set to `embed_tokens.weight` (same tensor, shared parameter).

**Embedding scale:** Embeddings scaled by `1 / sqrt(hidden_dim)` in the
forward pass (matching candle).

**Forward pass orchestration:**

```
def forward(input_ids):                       # [batch, seq_len]
    h = embed_tokens(input_ids) * scale       # [batch, seq_len, hidden_dim]
    attnres_state = []                        # Block summaries accumulate here

    for i, layer in enumerate(layers):
        # Block boundary mixing (blocks > 0)
        if i > 0 and i % layers_per_block == 0:
            block_idx = i // layers_per_block
            h = block_attnres.block_input(block_idx, h, attnres_state)

        # Standard transformer layer
        h = layer(h)

        # Record block summary at block end
        if (i + 1) % layers_per_block == 0:
            block_attnres.notify_layer_output(i, h, attnres_state)

    logits = lm_head(final_norm(h))           # [batch, seq_len, vocab_size]
    return logits
```

No Engram injection (out of scope for 0f). No L2 norm collection (UQ is
inference-only).

**No KV cache** — training uses full causal attention via
`F.scaled_dot_product_attention(is_causal=True)`.

### Weight Initialization

Matching candle `HarmonyModel::new()`:
- Linear weights: Kaiming uniform, scale `1 / sqrt(fan_in)`
- RMSNorm weights: initialized to ones
- Embedding: normal distribution, std `1 / sqrt(hidden_dim)`
- BlockAttnRes queries: small normal, std 0.02

## Optimizer (`optim.py`)

### Muon

Momentum-based optimizer that orthogonalizes the momentum buffer via
Newton-Schulz iteration before applying updates. Designed for matrix-shaped
parameters (Linear weights). Falls back to AdamW for everything else.

```python
class Muon(torch.optim.Optimizer):
    def __init__(self, muon_params, adam_params, lr, momentum=0.95,
                 adam_lr, adam_betas=(0.9, 0.95), adam_eps=1e-8):
```

Two parameter groups:
- **Muon group:** All 2D+ weight matrices (q/k/v/o_proj, gate/up/down_proj)
- **Adam group:** Embeddings, RMSNorm weights, BlockAttnRes queries, lm_head
  (if not tied)

**Muon update rule** (per step, for each 2D parameter):
1. Compute gradient
2. Update momentum buffer: `buf = momentum * buf + grad`
3. Orthogonalize via Newton-Schulz (5 iterations):
   - Coefficients: `a=3.4445, b=-4.7750, c=2.0315`
   - Loop: `X = a*X + b*(X @ X^T) @ X + c*((X @ X^T) @ (X @ X^T)) @ X`
4. Apply: `param -= lr * orthogonalized_momentum`

**Adam update rule** (for non-matrix params): Standard AdamW with decoupled
weight decay.

### Parameter Partitioning

```python
def partition_params(model: HarmonyModel) -> tuple[list, list]:
```

Returns `(muon_params, adam_params)` by inspecting parameter dimensionality
and name. 2D parameters from Linear layers go to Muon; everything else to Adam.

### WSD Schedule

Warmup-Stable-Decay: a piecewise linear learning rate schedule.

```python
class WSDSchedule:
    def __init__(self, warmup_steps, total_steps, decay_fraction=0.1,
                 min_lr_ratio=0.0):
```

Three phases:
1. **Warmup:** Linear from 0 to max LR over `warmup_steps`
2. **Stable:** Constant at max LR
3. **Decay:** Linear from max LR to `min_lr_ratio * max_lr` over last
   `decay_fraction * total_steps` steps

`get_lr_multiplier(step) -> float` returns a multiplier in [0, 1].

## Training Loop (`train.py`)

### CLI

Entry point: `python -m ct87.train`

```
--config       tiny | target           (model size, default: tiny)
--data         path to pre-tokenized dataset (HF datasets format)
--seq-len      context window (default: 512 for tiny, 2048 for target)
--batch-size   micro-batch size (default: 4)
--lr           learning rate (default: 3e-4)
--steps        total training steps (default: 1000)
--warmup       warmup steps (default: 100)
--save-every   checkpoint every N steps (default: 250)
--output-dir   checkpoint directory (default: training/checkpoints/)
--seed         reproducibility (default: 42)
--device       cpu | cuda | mps (auto-detected if omitted)
```

### Data Loading

Pre-tokenized HuggingFace dataset. The dataloader:
1. Draws random samples from the dataset
2. Concatenates and chunks into `[seq_len + 1]` windows (standard LM packing)
3. Returns `input_ids[:, :-1]` and `targets[:, 1:]` (shifted by 1)

### Training Loop

```python
for step in range(total_steps):
    lr_mult = schedule.get_lr_multiplier(step)
    set_lr(optimizer, lr_mult, base_lr)

    batch = next(dataloader)
    input_ids = batch[:, :-1]
    targets = batch[:, 1:]

    logits = model(input_ids)
    loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if step % 10 == 0:
        print(f"step={step} loss={loss.item():.4f} lr={lr_mult * base_lr:.6f}")

    if step % save_every == 0:
        save_checkpoint(model, optimizer, step, output_dir)
```

No gradient accumulation, distributed training, or mixed precision.

### Checkpointing

Model weights saved via `safetensors` (portable, Phase 0g reads these for
GGUF export). Optimizer state saved via `torch.save` separately (optimizer
state does not need to be portable).

## File Structure

| File | Responsibility | Approximate size |
|------|----------------|-----------------|
| `pyproject.toml` | Package definition, dependencies, entry points | ~30 lines |
| `ct87/__init__.py` | Package marker, version | ~5 lines |
| `ct87/model.py` | `HarmonyModelConfig`, `RMSNorm`, `RotaryEmbedding`, `Attention`, `Mlp`, `TransformerLayer`, `BlockAttnRes`, `HarmonyModel` | ~350 lines |
| `ct87/optim.py` | `Muon`, `WSDSchedule`, `partition_params` | ~200 lines |
| `ct87/train.py` | CLI, data loading, training loop, checkpointing | ~200 lines |
| `tests/test_model.py` | Model architecture tests | ~150 lines |
| `tests/test_optim.py` | Optimizer and schedule tests | ~100 lines |
| `tests/test_train.py` | End-to-end integration tests | ~80 lines |

Total: ~1,115 lines across 8 files.

## Scope Boundary

**In scope (0f):**
- `HarmonyModelConfig` dataclass with `target()` and `tiny()`
- All layer modules: `RMSNorm`, `RotaryEmbedding`, `Attention` (GQA, per-head
  QK norm, RoPE), `Mlp` (SwiGLU), `TransformerLayer`
- `BlockAttnRes` with `block_input()` and `notify_layer_output()`
- `HarmonyModel` with forward pass orchestrating BlockAttnRes at block boundaries
- Tied embeddings, embedding scaling
- Weight initialization (Kaiming uniform for Linear, ones for RMSNorm, normal
  for embedding, small normal for BlockAttnRes queries)
- `Muon` optimizer with Newton-Schulz orthogonalization + AdamW fallback
- `WSDSchedule` (Warmup-Stable-Decay) LR schedule
- Parameter partitioning (Muon vs Adam groups)
- Training loop with pre-tokenized data loading, loss computation, checkpointing
- CLI entry point via argparse
- Safetensors checkpoint saving
- `pyproject.toml` with minimal dependencies
- Unit tests for model, optimizer, and end-to-end overfit

**Out of scope (deferred):**
- Engram embedding lookup / injection (post-0f)
- UQ feature collection during training (inference concern)
- KV cache (inference concern — candle handles this)
- GGUF export (Phase 0g)
- Distributed training / FSDP / DDP (Phase 1+)
- Mixed precision / gradient accumulation (Phase 1+)
- W&B / tensorboard logging (Phase 1+)
- Eval loop / perplexity tracking (Phase 1+)
- Hyperparameter tuning (Phase 1+)
- Gradient clipping (add if needed, not specced preemptively)

## Testing Strategy

### Per-file tests

| File | Tests |
|------|-------|
| `test_model` | Config target/tiny values match Rust, config derived values (num_blocks, num_kv_groups), forward output shape [batch, seq, vocab], causal masking (future tokens don't affect past), tied embeddings share weight, BlockAttnRes produces different states than single-block model, weight init scale, embedding scale |
| `test_optim` | Muon step decreases loss, Adam fallback for 1D params, Newton-Schulz produces near-orthogonal matrix, WSD warmup linear increase, WSD stable constant, WSD decay linear decrease, WSD boundary values exact |
| `test_train` | Overfit tiny model on 2-sequence dataset (loss drops below 1.0 in 50 steps), checkpoint save/load roundtrip (logits match), parameter partitioning (Linear in Muon, rest in Adam) |

### Success Criterion

`python -m ct87.train --config tiny --data <small-dataset> --steps 200` shows
loss decreasing. This proves the architecture, optimizer, and training loop all
work together.
