# VLM Engine for harmony-inference

## Goal

Extend `harmony-inference` to support Qwen3-VL multimodal inference — text prompt + image input, logits output — so edge nodes can run vision-language models locally.

## Architecture

**Hybrid approach:** the quantized text backbone (existing GGUF `QwenEngine`) is paired with a full-precision (fp16) vision encoder loaded from safetensors. The vision encoder is small (~200-400 MB) and fits on constrained devices. The text model stays Q4_K_M quantized as today.

This avoids the fundamental blocker that candle-transformers has no `quantized_qwen3_vl` module — only full-precision `qwen3_vl`. By splitting the two components, we use what candle provides for vision while keeping our existing quantized text engine.

The `QwenEngine` gains two new capabilities: loading a vision encoder (`load_vision_encoder()`) and running a multimodal forward pass (`forward_with_image()`). Image preprocessing (resize, normalize, patch embedding, placeholder token insertion) happens inside the engine. Candle's `qwen3_vl::Qwen3VLVisionModel` handles the vision encoding.

### Dependencies

| Crate | Purpose |
|-------|---------|
| `harmony-inference` | VLM engine extensions |
| `candle-transformers` | `qwen3_vl::Qwen3VLVisionModel` (vision encoder, fp16) |
| `image` | Resize decoded pixels to model-compatible dimensions |
| `safetensors` | Load vision encoder weights (not GGUF) |

### Why hybrid, not full-precision VLM

Candle's `Qwen3VLModel` loads text+vision jointly from safetensors via VarBuilder. The full text model at fp16 is ~16 GB for 8B params — far too large for edge devices. Our existing quantized text engine (5 GB at Q4_K_M for 8B) works well. The vision encoder alone is small enough for fp16 on constrained hardware.

Future beads track alternatives: harmony-e8h2 (full-precision for high-capability nodes), harmony-fkmv (quantized VLM loader).

## API Surface

### New InferenceEngine trait methods

```rust
/// Load a vision encoder from safetensors weight data.
/// Required for multimodal inference; text-only models return an error.
fn load_vision_encoder(&mut self, weights_data: &[u8]) -> Result<(), InferenceError> {
    let _ = weights_data;
    Err(InferenceError::UnsupportedOperation(
        "vision encoder not supported by this engine".into(),
    ))
}

/// Run a multimodal forward pass: text prompt + decoded image → logits.
///
/// Unlike `forward()` which takes pre-tokenized token IDs, this method
/// takes a raw text prompt because the engine must insert image placeholder
/// tokens (`<|image_pad|>`) at the correct positions. The number of
/// placeholder tokens depends on the image resolution (computed during
/// preprocessing), so tokenization must happen inside the engine.
///
/// `pixels` is raw RGB bytes (decoded from JPEG/PNG by the caller).
/// The engine handles resizing, normalization, patch embedding,
/// placeholder token insertion, and grid metadata computation internally.
///
/// Returns logits at the last position, same as `forward()`.
fn forward_with_image(
    &mut self,
    prompt: &str,
    pixels: &[u8],
    width: u32,
    height: u32,
) -> Result<Vec<f32>, InferenceError> {
    let _ = (prompt, pixels, width, height);
    Err(InferenceError::UnsupportedOperation(
        "multimodal forward not supported by this engine".into(),
    ))
}
```

Default implementations return the new `UnsupportedOperation` error variant, keeping backward compatibility for text-only consumers.

### New InferenceError variant

```rust
/// Operation not supported by this engine variant (e.g. vision on a text-only model).
UnsupportedOperation(String),
```

### QwenEngine extensions

Two new optional fields:

- `vision_encoder: Option<Qwen3VLVisionModel>` — loaded from safetensors. Handles image patch embedding, ViT layers, and projection to the text model's hidden dimension.
- `vision_config: Option<VisionConfig>` — preprocessing parameters (resize bounds, normalization constants, patch size, temporal patch size) from the vision model config.

## Loading Flow

Two-step model loading, reflecting the hybrid architecture:

1. `load_gguf(gguf_data)` — loads the quantized text model (unchanged)
2. `load_tokenizer(tokenizer_json)` — loads the tokenizer (unchanged)
3. `load_vision_encoder(weights_data)` — loads the fp16 vision encoder from safetensors

Steps 1-2 work without step 3 for text-only inference. Step 3 requires step 1 (the vision encoder's output dimension must match the text model's hidden size).

The vision encoder weights are distributed as a separate safetensors file, fetched via CAS like other content. This mirrors how GGUF + tokenizer are already fetched separately.

## Image Preprocessing

When `forward_with_image()` is called, the engine runs this pipeline:

1. **Construct image buffer** — wrap `pixels` (RGB bytes) + `width`/`height` into an image buffer
2. **Dynamic resize** — resize maintaining aspect ratio, round dimensions to nearest multiple of 28 (Qwen convention). Constrain to configurable min/max pixel budget.
3. **Normalize** — scale to `[0.0, 1.0]`, apply model-specific mean/std normalization
4. **Temporal duplication** — duplicate along time axis (time=2) to match the vision encoder's expected input for static images
5. **Patch and convert to Tensor** — shape into `[batch, time, channels, height, width]` for the vision encoder
6. **Compute `image_grid_thw`** — build the grid metadata tensor `[1, 3]` with `[T=2, H=h_patches, W=w_patches]` where patch counts are derived from the resized dimensions and the patch size (14x14)
7. **Vision encode** — run `Qwen3VLVisionModel::forward(patches, grid_thw)` → vision hidden states
8. **Tokenize with placeholders** — tokenize the text prompt, insert `<|image_pad|>` placeholder tokens at the image position. The number of placeholders equals the number of vision tokens output by the encoder.
9. **Build combined input** — merge text token embeddings with vision hidden states at the placeholder positions
10. **Text model forward** — run the quantized text model on the combined input → logits

Normalization constants (Qwen3-VL defaults):
- Mean: `[0.48145466, 0.4578275, 0.40821073]`
- Std: `[0.26862954, 0.26130258, 0.27577711]`
- Spatial patch size: 14x14
- Temporal patch size: 2

## Bridging Vision Encoder and Text Model

The vision encoder outputs fp16 hidden states. The quantized text model expects quantized inputs. The bridge between them:

1. Vision encoder produces `[num_vision_tokens, hidden_size]` in fp16
2. Cast to f32 (candle handles this via `.to_dtype()`)
3. The text model's embedding layer is bypassed for vision token positions — vision hidden states are injected directly at the placeholder positions
4. The text model processes the mixed sequence (text embeddings + vision hidden states) through its quantized attention layers

This bridging is the most architecturally novel part and may require careful study of how candle's quantized models handle mixed-precision input at the attention layer. If the quantized model cannot accept fp32 inputs at arbitrary positions, an alternative is to project the vision hidden states through a small learned linear layer into the quantized space. This projection layer would be part of the vision encoder weights.

## Testing Strategy

### Unit tests (no real model)

- `forward_with_image` returns `UnsupportedOperation` when no vision encoder loaded
- `load_vision_encoder` returns `UnsupportedOperation` when no text model loaded
- Image preprocessing: resize maintains aspect ratio, rounds to multiple of 28
- Image preprocessing: normalization produces expected output for known input pixels
- `image_grid_thw` computation: known image dimensions → correct grid tensor
- `InferenceError::UnsupportedOperation` variant displays correctly

### Integration tests (`#[ignore]`, real models)

- Load Qwen3-VL text GGUF + vision safetensors + tokenizer, call `forward_with_image` with a test image, verify logits are non-empty
- Verify text-only `forward()` still works after vision encoder is loaded
- Verify `forward_with_image` on a simple scene produces coherent token predictions

## Scope Exclusions

- **No NodeRuntime integration** — separate sub-bead (sensor pipeline, harmony-60nv)
- **No camera/sensor input** — engine accepts pre-decoded RGB pixels only
- **No video support** — single-frame images only
- **No streaming output** — returns complete logits like existing `forward()`
- **No full-precision VLM** — hybrid only (tracked as harmony-e8h2)
- **No quantized VLM loader** — tracked as harmony-fkmv
- **No JPEG/PNG decoding** — callers provide decoded RGB bytes
- **No Google Coral TPU** — tracked as harmony-ztgq
