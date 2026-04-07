# Phase 0g: GGUF Export/Import Design

> The bridge between PyTorch training (Phase 0f) and candle inference (Phase 0e).
> Defines how ct87 weights move from safetensors checkpoints to GGUF files that
> `HarmonyEngine` can load and run.

**Dependencies:** Phase 0e (HarmonyModel in candle), Phase 0f (training scaffold in PyTorch)

## Architecture Prefix

GGUF metadata uses `harmony` as the architecture prefix (e.g., `harmony.block_count`).
File-level identity uses `general.name` (e.g., `"ct87-0.5B"`) to distinguish model variants.

## Tensor Naming Convention

Deterministic mapping from Python `state_dict` keys to GGUF tensor names.
Standard transformer components follow the llama.cpp `blk.{i}.*` convention (matching
the existing Qwen3 loader in `qwen3_ext.rs`). BlockAttnRes queries use the
`harmony.*` namespace.

| Python state_dict key | GGUF tensor name | Shape |
|---|---|---|
| `embed_tokens.weight` | `token_embd.weight` | `[vocab, hidden]` |
| `layers.{i}.attn_norm.weight` | `blk.{i}.attn_norm.weight` | `[hidden]` |
| `layers.{i}.attn.q_proj.weight` | `blk.{i}.attn_q.weight` | `[q_heads*head_dim, hidden]` |
| `layers.{i}.attn.k_proj.weight` | `blk.{i}.attn_k.weight` | `[kv_heads*head_dim, hidden]` |
| `layers.{i}.attn.v_proj.weight` | `blk.{i}.attn_v.weight` | `[kv_heads*head_dim, hidden]` |
| `layers.{i}.attn.o_proj.weight` | `blk.{i}.attn_output.weight` | `[hidden, q_heads*head_dim]` |
| `layers.{i}.attn.q_norm.weight` | `blk.{i}.attn_q_norm.weight` | `[head_dim]` |
| `layers.{i}.attn.k_norm.weight` | `blk.{i}.attn_k_norm.weight` | `[head_dim]` |
| `layers.{i}.ffn_norm.weight` | `blk.{i}.ffn_norm.weight` | `[hidden]` |
| `layers.{i}.mlp.gate_proj.weight` | `blk.{i}.ffn_gate.weight` | `[ffn, hidden]` |
| `layers.{i}.mlp.up_proj.weight` | `blk.{i}.ffn_up.weight` | `[ffn, hidden]` |
| `layers.{i}.mlp.down_proj.weight` | `blk.{i}.ffn_down.weight` | `[hidden, ffn]` |
| `final_norm.weight` | `output_norm.weight` | `[hidden]` |
| `lm_head.weight` | `output.weight` | `[vocab, hidden]` (omitted when tied) |
| `block_attnres.queries.{i}` | `harmony.block_attnres.query.{i}.weight` | `[1, 1, hidden]` (stored as `[hidden]` in GGUF, reshaped on load) |

## GGUF Metadata

| Key | Type | Example (target) | Source |
|---|---|---|---|
| `general.architecture` | string | `"harmony"` | Hardcoded |
| `general.name` | string | `"ct87-0.5B"` | CLI arg or default |
| `harmony.block_count` | u32 | 24 | `config.num_layers` |
| `harmony.embedding_length` | u32 | 1280 | `config.hidden_dim` |
| `harmony.attention.head_count` | u32 | 16 | `config.num_query_heads` |
| `harmony.attention.head_count_kv` | u32 | 8 | `config.num_kv_heads` |
| `harmony.attention.key_length` | u32 | 80 | `config.head_dim` |
| `harmony.feed_forward_length` | u32 | 3413 | `config.ffn_dim` |
| `harmony.context_length` | u32 | 32768 | `config.max_seq_len` |
| `harmony.rope.freq_base` | f32 | 1000000.0 | `config.rope_theta` |
| `harmony.attention.layer_norm_rms_epsilon` | f32 | 1e-6 | `config.rms_norm_eps` |
| `harmony.vocab_size` | u32 | 32000 | `config.vocab_size` |
| `harmony.layers_per_block` | u32 | 3 | `config.layers_per_block` |
| `harmony.tie_embeddings` | bool | true | `config.tie_embeddings` |

The first 11 keys mirror Qwen3's metadata convention (prefix swapped to `harmony`).
`layers_per_block` and `tie_embeddings` are ct87-specific, needed for BlockAttnRes
construction and tied-embedding fallback.

## Python Exporter

New file: `training/ct87/export_gguf.py`

New dependency: `gguf` package (from llama.cpp) added to `pyproject.toml`.

### Flow

1. Load `HarmonyModelConfig` from CLI arg (`--config target` or `--config tiny`).
2. Load safetensors checkpoint via `safetensors.torch.load_file()`.
3. **Validate completeness**: verify every expected state_dict key exists and every
   state_dict key is accounted for. Fail loudly on mismatch.
4. Create `GGUFWriter` with architecture `"harmony"`.
5. Write all metadata keys per the table above.
6. Iterate the naming map — for each PyTorch key, look it up in the state_dict,
   convert to numpy float32, write as the corresponding GGUF tensor name.
7. Skip `lm_head.weight` when `tie_embeddings=True` (same tensor as `embed_tokens.weight`).
8. Close the writer (flushes to disk).

### CLI Interface

```
python -m ct87.export_gguf \
  --checkpoint path/to/model.safetensors \
  --config target \
  --output path/to/model.gguf \
  --name "ct87-0.5B"   # optional, defaults to "ct87-{config}"
```

### Tensor Format

All tensors are exported as f32. No quantization in Phase 0g — quantization can be
applied to the f32 GGUF later using external tools (e.g., llama.cpp `quantize`).

## Rust Loader

### `HarmonyModel::from_gguf()`

New constructor in `harmony_model.rs`, parallel to the existing `new()` for random init.
Follows the same signature pattern as `ModelWeights::from_gguf()` in `qwen3_ext.rs`.

**Steps:**
1. Accept `gguf_file::Content` + reader + device.
2. Verify `general.architecture == "harmony"`.
3. Read all metadata keys to reconstruct `HarmonyModelConfig`.
4. Build `RotaryEmbedding` from config.
5. Load `token_embd.weight` → dequantize to f32 → `Embedding`.
6. For each layer 0..num_layers: load `blk.{i}.*` tensors via the `Gguf` helper's
   `qmatmul()` and `rms_norm()` methods → construct `Attention`, `Mlp`,
   `TransformerLayer`.
7. Load `output_norm.weight` → `RmsNorm`.
8. Load `output.weight` with fallback to `token_embd.weight` for tied embeddings.
9. Load `harmony.block_attnres.query.{i}.weight` for i in 0..num_blocks-1 →
   dequantize → feed into `BlockAttnRes::from_tensors()`.
10. Assemble and return `HarmonyModel`.

### Structural Changes

The internal layer types (`Attention`, `Mlp`, `TransformerLayer`) currently only have
`new()` constructors with random weights. Each type gets a `from_gguf()` constructor
that accepts a `Gguf` helper and layer prefix, mirroring the pattern in `qwen3_ext.rs`
(`AttentionWeights::new(gg, ...)`, `MlpWeights::new(gg, ...)`).

### `HarmonyEngine::load_gguf()` Update

Replace the "not yet implemented" placeholder:
1. Parse GGUF content from the byte slice.
2. Call `HarmonyModel::from_gguf()`.
3. Store the model. The config is already available from the loaded model.

## Testing

### Python (`training/tests/test_export_gguf.py`)

- **Roundtrip**: Create tiny model → `save_checkpoint()` → export to GGUF → read back
  GGUF → verify every tensor matches original state_dict values within f32 tolerance.
- **Metadata**: Export → verify all metadata keys present with correct values.
- **Tied embeddings**: When `tie_embeddings=True`, `output.weight` absent from GGUF;
  when `False`, `output.weight` present and distinct.
- **Validation**: Exporter rejects state_dict with missing or extra keys.

### Rust (`harmony_model.rs` tests)

- **Metadata parsing**: Construct GGUF with known metadata → load → verify
  reconstructed `HarmonyModelConfig` matches.
- **Full load**: Export tiny model from Python (test fixture) → `from_gguf()` → verify
  correct layer count, head count, etc.
- **Forward pass**: Load from GGUF → forward pass → verify output shape is
  `[1, vocab_size]`.
- **Tied embedding fallback**: GGUF without `output.weight` loads successfully.
- **Bad architecture**: GGUF with wrong `general.architecture` is rejected.

## Scope Boundary

**In scope:**
- Tensor naming convention
- Python GGUF exporter (`training/ct87/export_gguf.py`)
- `gguf` pip dependency
- `HarmonyModel::from_gguf()` + layer-level `from_gguf()` constructors
- `HarmonyEngine::load_gguf()` wired up
- Tests on both sides

**Out of scope:**
- Quantization (Q4_0, Q8_0, etc.) — future work
- Numerical parity validation (Python vs candle identical outputs) — Phase 0h
- Engram weights in GGUF — not trained yet
- UQ head weights in GGUF — not trained yet
- Tokenizer bundling — loaded separately
