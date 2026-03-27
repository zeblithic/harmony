# VLM Engine for harmony-inference

> **Status: PARKED** — blocked on candle API gap. `quantized_qwen3::ModelWeights::forward()` accepts token IDs only, not embeddings. There is no `forward_from_embeds()` to inject fp16 vision hidden states into the quantized text model. See "Unresolved Blockers" section at end.
>
> **Bead:** harmony-tbbv | **Related:** harmony-e8h2 (full-precision), harmony-fkmv (quantized loader), harmony-ztgq (Coral TPU)

## Goal

Extend `harmony-inference` to support Qwen3-VL multimodal inference — image+text input, logits output — so edge nodes can run vision-language models locally.

## Architecture

The existing `QwenEngine` gains two new capabilities: loading a vision projector and running a multimodal forward pass. Image preprocessing (resize, normalize, tensor conversion) happens inside the engine. Candle's `qwen3_vl` module handles the heavy ML work — Conv3d patch embedding, ViT encoder, multimodal projection, and joint attention with text tokens.

The `InferenceEngine` trait gets two new methods with default implementations that return errors, so existing text-only consumers are unaffected.

### Dependencies

| Crate | Purpose |
|-------|---------|
| `harmony-inference` | VLM engine extensions |
| `candle-transformers` | `qwen3_vl` module (Conv3d, ViT, projection) |
| `image` | Resize decoded pixels to model-compatible dimensions |

## API Surface

### New InferenceEngine trait methods

```rust
/// Load a vision projector (mmproj) from raw bytes.
/// Required for multimodal inference; text-only models return an error.
fn load_vision_projector(&mut self, mmproj_data: &[u8]) -> Result<(), InferenceError> {
    Err(InferenceError::UnsupportedOperation(
        "vision projector not supported by this engine".into(),
    ))
}

/// Run a multimodal forward pass: text tokens + decoded image → logits.
///
/// `pixels` is raw RGB bytes (decoded from JPEG/PNG by the caller).
/// The engine handles resizing, normalization, and patch embedding internally.
/// Returns logits at the last position, same as `forward()`.
fn forward_with_image(
    &mut self,
    tokens: &[u32],
    pixels: &[u8],
    width: u32,
    height: u32,
) -> Result<Vec<f32>, InferenceError> {
    let _ = (tokens, pixels, width, height);
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

- `vision_projector: Option<VisionProjector>` — loaded from mmproj GGUF data. `VisionProjector` wraps the candle `qwen3_vl` vision encoder + projection layers.
- `vision_config: Option<VisionConfig>` — preprocessing parameters (resize bounds, normalization constants) extracted from GGUF metadata or hardcoded to Qwen3-VL defaults.

`load_vision_projector()` parses the mmproj GGUF, constructs the vision encoder, and stores it. Returns an error if no base language model is loaded yet.

`forward_with_image()` returns an error if the projector hasn't been loaded.

## Loading Flow

Two-step model loading, matching how VLM GGUF files are distributed (separate language model + mmproj files):

1. `load_gguf(gguf_data)` — loads the language model weights (unchanged from current behavior)
2. `load_tokenizer(tokenizer_json)` — loads the tokenizer (unchanged)
3. `load_vision_projector(mmproj_data)` — loads the vision encoder and projection layers

Steps 1-2 work without step 3 for text-only inference. Step 3 requires step 1 to have completed (the projector dimensions must match the language model's hidden size).

## Image Preprocessing

When `forward_with_image()` is called, the engine preprocesses raw pixels before the vision encoder:

1. **Construct image buffer** — wrap `pixels` + `width`/`height` into an RGB image
2. **Dynamic resize** — resize maintaining aspect ratio, round dimensions to nearest multiple of 28 (Qwen convention). Constrain to configurable min/max pixel budget to bound compute.
3. **Normalize** — scale pixel values to `[0.0, 1.0]`, then apply model-specific mean/std normalization (values from Qwen3-VL training config)
4. **Temporal duplication** — duplicate along time axis (time=2) to match the model's native video input format for static images
5. **Convert to candle Tensor** — shape into `[batch, time, channels, height, width]` for the qwen3_vl vision encoder

Normalization constants and resize bounds are read from GGUF metadata when available, falling back to Qwen3-VL defaults:
- Mean: `[0.48145466, 0.4578275, 0.40821073]`
- Std: `[0.26862954, 0.26130258, 0.27577711]`
- Patch size: 14x14 spatial
- Min pixels: 256x28 = 7168
- Max pixels: configurable, default 1280x28x28 = 1003520

## Testing Strategy

### Unit tests (no real model)

- `forward_with_image` returns `UnsupportedOperation` on text-only engine (no projector loaded)
- `load_vision_projector` returns `UnsupportedOperation` on text-only engine (no qwen3_vl model loaded)
- Image preprocessing: resize maintains aspect ratio, rounds to multiple of 28
- Image preprocessing: normalization produces expected output for known input pixels
- `InferenceError::UnsupportedOperation` variant displays correctly

### Integration tests (`#[ignore]`, real models)

- Load Qwen3-VL GGUF + mmproj, tokenize prompt, call `forward_with_image` with test image, verify logits are non-empty
- Verify text-only `forward()` still works on a VLM engine

## Scope Exclusions

- **No NodeRuntime integration** — separate sub-bead (sensor pipeline)
- **No camera/sensor input** — engine accepts pre-decoded pixels only
- **No video support** — single-frame images only (temporal duplication is an internal implementation detail)
- **No streaming output** — returns complete logits like existing `forward()`
- **No Qwen 3.5 early-fusion** — target Qwen3-VL first (candle has `qwen3_vl` module). Qwen 3.5 native multimodal is future work once candle adds support.
- **No JPEG/PNG decoding** — callers provide decoded RGB bytes. Format decoding is an I/O concern outside the sans-I/O engine.

## Unresolved Blockers (2026-03-26)

These issues were identified during spec review and block implementation:

1. **`quantized_qwen3::ModelWeights::forward()` takes token IDs, not embeddings.** The hybrid approach (fp16 vision encoder + quantized text model) requires injecting vision hidden states into the text model's attention layers. But `ModelWeights::forward(&input, position)` only accepts a token ID tensor — there is no `forward_from_embeds()` entry point. Without this, vision embeddings cannot reach the text model's attention mechanism.

2. **`Qwen3VLModel::forward()` requires complex metadata.** The full VLM forward pass needs `continuous_img_pad` (placeholder token span ranges), `image_grid_thw` (patch grid dimensions), `seqlens`, and `seqlen_offsets`. These are non-trivial to compute and cannot be hidden behind a simple API without significant internal bookkeeping.

3. **`VisionConfig` source unspecified.** `Qwen3VLVisionModel::new()` requires a deserialized `VisionConfig` (from JSON), not derivable from GGUF metadata.

### Paths Forward

| Approach | Effort | Tradeoff |
|----------|--------|----------|
| Fork candle, add `forward_from_embeds()` to `quantized_qwen3` | Medium | Maintains a fork; may be accepted upstream |
| Full-precision `Qwen3VLModel` for VLM (harmony-e8h2) | Low | 16GB+ RAM required; not edge-viable for 8B models |
| Build `quantized_qwen3_vl` from scratch (harmony-fkmv) | High | Complete solution but significant engineering |
| Google Coral TPU for vision preprocessing (harmony-ztgq) | Research | Offloads vision to dedicated hardware; different architecture |
