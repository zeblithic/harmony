# GGUF Model Loader and Native Inference Engine

## Goal

Load quantized GGUF models (Qwen 3.5) and run token-by-token inference on CPU. A new `harmony-inference` library crate wrapping Hugging Face's candle framework, exposing an `InferenceEngine` trait for composable integration with the Harmony compute stack.

## Architecture

**`harmony-inference`** — new library crate in `crates/harmony-inference/`. Wraps `candle-core`, `candle-nn`, and `candle-transformers` for quantized Qwen3 GGUF inference. Not sans-I/O — performs real tensor computation — but has no async, no network, no filesystem access at runtime. Takes bytes in, returns tokens/logits out.

The crate wraps candle-transformers' `quantized_qwen3::ModelWeights` behind an `InferenceEngine` trait, enabling a future swap to a custom CAS-optimized implementation (harmony-6b0) without breaking consumers.

### Dependencies

| Crate | Purpose |
|-------|---------|
| `candle-core` | Tensor operations, Device abstraction (CPU/CUDA/Metal) |
| `candle-nn` | Neural network primitives (transitive via candle-transformers) |
| `candle-transformers` | Qwen3 quantized model implementation |
| `tokenizers` | Hugging Face BPE tokenizer (Qwen3 vocabulary) |
| `half` | f16 type (already in workspace) |
| `thiserror` | Error types |

### Model Context

Edge nodes run **Qwen 3.5** (0.8B/4B GGUF) for inference, augmented by **DeepSeek Engram tables** (harmony-engram) for O(1) factual lookups from the CAS mesh. This crate handles the Qwen3 inference; Engram integration is future work.

## Core API

### Types

```rust
/// Sampling parameters for token generation.
pub struct SamplingParams {
    pub temperature: f32,    // 0.0 = greedy, 1.0 = default
    pub top_p: f32,          // nucleus sampling threshold
    pub top_k: u32,          // top-k filtering (0 = disabled)
    pub repeat_penalty: f32, // penalize repeated tokens
}
```

### InferenceEngine Trait

```rust
pub trait InferenceEngine {
    /// Load a GGUF model from raw bytes.
    fn load_gguf(&mut self, gguf_data: &[u8]) -> Result<(), InferenceError>;

    /// Load tokenizer from a tokenizer.json file's bytes.
    fn load_tokenizer(&mut self, tokenizer_json: &[u8]) -> Result<(), InferenceError>;

    /// Tokenize text into token IDs.
    fn tokenize(&self, text: &str) -> Result<Vec<u32>, InferenceError>;

    /// Detokenize token IDs back to text.
    fn detokenize(&self, tokens: &[u32]) -> Result<String, InferenceError>;

    /// Run a single forward pass: token IDs → logits.
    /// Manages KV cache internally. First call = prefill, subsequent = decode.
    fn forward(&mut self, tokens: &[u32]) -> Result<Vec<f32>, InferenceError>;

    /// Sample the next token from logits.
    fn sample(&self, logits: &[f32], params: &SamplingParams) -> Result<u32, InferenceError>;

    /// Reset the KV cache (start a new conversation).
    fn reset(&mut self);
}
```

### Usage Pattern

```rust
let mut engine = QwenEngine::new(Device::Cpu);
engine.load_gguf(&gguf_bytes)?;
engine.load_tokenizer(&tokenizer_json)?;

let prompt_tokens = engine.tokenize("What is Harmony?")?;
let logits = engine.forward(&prompt_tokens)?;  // prefill
let mut next = engine.sample(&logits, &params)?;

loop {
    let logits = engine.forward(&[next])?;     // decode one token
    next = engine.sample(&logits, &params)?;
    if next == eos_token { break; }
    print!("{}", engine.detokenize(&[next])?);
}
```

The caller drives the autoregressive loop. This enables streaming output over Zenoh, custom stopping criteria, and future Engram embedding injection between forward passes.

## QwenEngine Implementation

```rust
pub struct QwenEngine {
    model: Option<quantized_qwen3::ModelWeights>,
    tokenizer: Option<tokenizers::Tokenizer>,
    device: candle_core::Device,
}
```

### GGUF Loading

`load_gguf()` takes raw GGUF bytes and uses candle's GGUF parser:

1. Parse GGUF header and tensor metadata via `candle_core::quantized::gguf_file::Content`
2. Load quantized tensors into `candle_core::quantized::QTensor` on the target device
3. Construct `quantized_qwen3::ModelWeights` from the parsed tensors

The KV cache is allocated lazily on the first `forward()` call based on the model's layer count and head dimensions.

### Tokenizer Loading

Qwen3 uses a BPE tokenizer. The `load_tokenizer()` method accepts `tokenizer.json` bytes (Hugging Face format) and constructs a `tokenizers::Tokenizer`. This is separate from `load_gguf()` because:
- Some GGUF files embed tokenizer data, others don't
- The tokenizer.json can be loaded from CAS independently
- Keeps the API explicit — no hidden file I/O

### Forward Pass

`forward()` delegates to `quantized_qwen3::ModelWeights::forward()`:

1. Convert input token IDs to a candle `Tensor`
2. Call `model.forward(&input_tensor, position)` — handles attention, KV cache, RoPE, QK-norm
3. Extract the last token's logits as `Vec<f32>`
4. Advance the internal position counter

The KV cache is managed by candle-transformers internally. `reset()` clears it by reconstructing the cache state.

### Sampling

`sample()` is pure math — no model state needed:

1. Apply temperature scaling: `logits[i] /= temperature`
2. Apply repeat penalty (if enabled): reduce logits for tokens seen in context
3. Apply top-k filtering: keep only the k highest logits, set rest to -inf
4. Apply top-p (nucleus) filtering: sort by probability, keep cumulative mass ≤ top_p
5. Sample from the resulting distribution (softmax → weighted random choice)

For `temperature == 0.0` (greedy): skip all filtering, return argmax.

## Error Handling

```rust
pub enum InferenceError {
    /// No model loaded — call load_gguf() first.
    ModelNotLoaded,
    /// No tokenizer loaded — call load_tokenizer() first.
    TokenizerNotLoaded,
    /// GGUF file is invalid or unsupported.
    InvalidGguf(String),
    /// Tokenizer JSON is invalid.
    TokenizerError(String),
    /// Forward pass failed (tensor operation error).
    ForwardFailed(String),
    /// Sampling failed (empty logits, invalid params).
    SamplingFailed(String),
}
```

Candle errors are mapped into `InferenceError` variants. Callers never see candle types directly.

## Device Selection

CPU-only for the first pass. The `candle-core` `Device` abstraction means adding Metal/CUDA later is a feature flag change:

```toml
[features]
default = []
cuda = ["candle-core/cuda"]
metal = ["candle-core/metal"]
```

The `QwenEngine::new(device: Device)` constructor accepts the device. CPU is the default for edge deployment. GPU features are additive — they don't change the API.

## Testing Strategy

### Unit tests

- **Sampling logic** (no model needed — pure math):
  - Greedy sampling returns argmax
  - Temperature=0.0 equivalent to greedy
  - Top-k=1 equivalent to greedy
  - Top-p filters low-probability tokens
  - Repeat penalty reduces repeated token logits
  - Empty logits returns error
- **Error paths:**
  - `forward()` before `load_gguf()` returns `ModelNotLoaded`
  - `tokenize()` before `load_tokenizer()` returns `TokenizerNotLoaded`
  - Invalid GGUF bytes return `InvalidGguf`

### Integration tests (`#[ignore]`)

Marked `#[ignore]` — require a downloaded GGUF model file (500MB+), not suitable for CI:

- Load real Qwen3 0.8B Q4_K_M GGUF
- Tokenize a prompt, verify token count is reasonable
- Run forward pass, verify logits shape matches vocab size
- Sample a token, verify it's a valid token ID
- Generate 10 tokens, verify output is non-empty text

Developers run these locally: `cargo test -p harmony-inference -- --ignored`

## Scope Exclusions

- **GPU/Metal/CUDA acceleration** — future feature flags, not this bead
- **WASM host function integration** — harmony-e0s
- **Zenoh queryable service** — harmony-yku
- **Engram embedding injection** — future integration point
- **Streaming generation API** — trivial to build on the token-by-token primitives
- **Custom Qwen3 implementation** — harmony-6b0
- **Model downloading** — caller provides GGUF bytes and tokenizer.json
- **MoE models** — Qwen3 MoE (30B-A3B) support is future work; dense models (0.8B/4B/8B) only for now
