# Phase 4a: Continuous Thought Infrastructure — Design Specification

**Epic:** harmony-ct87
**Status:** Design
**Date:** 2026-04-08
**Depends on:** Phase 0 (modules), Phase 3 (trained model)
**Builds toward:** Phase 4b (full COCONUT latent reasoning)

## Goal

Add continuous thought infrastructure to the ct87 model: the ability for the model
to perform extra computation ("thinking") before committing to an output token. Phase
4a uses **pause tokens** (`<think>`) as a stepping stone — structurally simple, zero
changes to the transformer core, but establishes the UQ-driven routing control flow
and async engine changes needed for Phase 4b (full COCONUT continuous thought).

**Core thesis:** The same UQ Head that currently classifies uncertainty can *route*
generation — suppressing token output when the model is uncertain and granting
extra forward passes to deliberate. This transforms UQ from a passive monitor to
an active metacognitive router, the key architectural role it needs for Phase 4b.

## Background: The Latent Reasoning Landscape

Research from Meta (COCONUT, arxiv:2412.06769) and others demonstrates that language
models can reason more effectively in continuous embedding space than in discrete
token space. Key findings relevant to harmony:

- **COCONUT:** Feeds the final hidden state back as the next input embedding, bypassing
  the vocabulary projection. Continuous thought vectors encode multiple reasoning paths
  simultaneously (BFS-like superposition). Outperforms chain-of-thought on planning tasks.
- **VL-JEPA:** Predicts continuous embeddings instead of tokens. Achieves comparable
  quality with 2.85x fewer decoding operations via selective decoding.
- **Latent-Guided Reasoning:** Confirms latent reasoning benefits at 0.5B scale
  (up to 13.9% accuracy gains on out-of-domain logical tasks).
- **Pause Tokens** (Goyal et al.): Extra computation steps in token space. Simplest
  approach — no architecture changes, but still bound to vocabulary bottleneck.

Phase 4a implements pause tokens; Phase 4b replaces them with true continuous thought.

## Architecture Overview

### Phase 4a (This Spec): Pause Token Routing

```
Forward Pass
     |
     v
HarmonyForwardOutput { logits, layer_norms }
     |
     v
UQ Head classify_uncertainty()
     |
     +--> Confident (conf > tau)  --> sample token normally --> emit
     |
     +--> Uncertain / HighVolume  --> force <think> token --> suppress output
     |                                 (still runs full forward, advances KV cache)
     |                                 loop back to Forward Pass (up to max_think_steps)
     |
     +--> SpectralCollapse        --> abort / escalate (existing behavior)
```

### Phase 4b (Future): Full COCONUT

```
Forward Pass (without lm_head projection)
     |
     v
final_hidden_state [1, 1, hidden_dim]
     |
     v
UQ Head evaluates features
     |
     +--> Confident  --> project through lm_head --> sample --> emit token
     |
     +--> Uncertain  --> feed hidden_state directly as next input embedding
     |                   (bypass embed_tokens, bypass vocab projection)
     |                   use ephemeral KV cache (overwrite, don't accumulate)
     |                   re-query Engram with evolved hidden state
     |                   loop back (up to max_latent_steps)
     |
     +--> SpectralCollapse --> abort / escalate
```

## Phase 4a Detailed Design

### 1. Vocabulary Extension: `<think>` Token

Add a single special token `<think>` to the vocabulary.

**Token ID:** `vocab_size` (appended at position 32000 for target model, 32000 for
tiny). The embedding table grows by one row. When `tie_embeddings` is true, the
lm_head weight also grows by one column.

**Training:** During Phase 4a training, `<think>` tokens are inserted into training
data at positions where explicit CoT reasoning tokens were present. The model learns
to associate `<think>` with "more computation needed here." This primes the vocabulary
for Phase 4b, where `<think>` is replaced by unconstrained continuous thought.

**Config change:**

```rust
// HarmonyModelConfig
pub think_token_id: Option<u32>,  // None = continuous thought disabled
```

When `Some(id)`, the engine activates the UQ routing loop.

### 2. UQ Head Routing Upgrade

The UQ Head's architecture and weights are unchanged. The change is in how
`HarmonyEngine` uses the classification output.

**New method on `HarmonyEngine`:**

```rust
/// Routing decision for the generation loop.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ThoughtAction {
    /// Emit the sampled token to the output stream.
    Emit,
    /// Suppress output, force <think> token, continue deliberation.
    Think,
    /// Abort generation (spectral collapse detected).
    Abort,
}

/// Determine the routing action based on UQ classification.
///
/// If no UQ head is set or continuous thought is disabled, always returns Emit.
/// Otherwise, evaluates the UQ output and returns Think when the model is
/// uncertain (up to max_think_steps consecutive thinks).
pub fn route_thought(
    &self,
    output: &HarmonyForwardOutput,
    consecutive_thinks: u32,
) -> Result<ThoughtAction, InferenceError>
```

**Routing logic:**

```
if no UQ head or think_token_id is None:
    return Emit

(class, confidence) = classify_uncertainty(output)

match class:
    Confident if confidence > tau_high:
        return Emit
    SpectralCollapse:
        return Abort
    Uncertain | HighVolume:
        if consecutive_thinks >= max_think_steps:
            return Emit  // safety cap: force output after N thinks
        return Think
    Confident if confidence <= tau_high:
        return Emit  // low-confidence "confident" still emits
```

**Configuration:**

```rust
/// Configuration for continuous thought routing.
#[derive(Debug, Clone)]
pub struct ContinuousThoughtConfig {
    /// Token ID for the <think> pause token. None = disabled.
    pub think_token_id: Option<u32>,
    /// Maximum consecutive think steps before forcing token emission.
    /// Prevents infinite loops. Default: 4.
    pub max_think_steps: u32,
    /// UQ confidence threshold above which the model is considered confident.
    /// Default: 0.85.
    pub confidence_threshold: f32,
}
```

### 3. Inference Loop Changes

The generation loop is caller-managed (not inside `HarmonyEngine`). The engine
provides the building blocks; the caller assembles the loop. The change is
adding a `route_thought` call between `forward_full` and `sample`.

**Current loop (simplified):**

```rust
loop {
    let output = engine.forward_full(&tokens, &mut cache, engram)?;
    let logits = logits_to_vec(&output.logits)?;
    let token = engine.sample(&logits, &params, &history)?;
    history.push(token);
    tokens = vec![token];
    // emit token to output stream
}
```

**Phase 4a loop:**

```rust
let mut consecutive_thinks = 0u32;

loop {
    let output = engine.forward_full(&tokens, &mut cache, engram)?;

    match engine.route_thought(&output, consecutive_thinks)? {
        ThoughtAction::Emit => {
            consecutive_thinks = 0;
            let logits = logits_to_vec(&output.logits)?;
            let token = engine.sample(&logits, &params, &history)?;
            history.push(token);
            tokens = vec![token];
            // emit token to output stream
        }
        ThoughtAction::Think => {
            consecutive_thinks += 1;
            let think_id = engine.think_token_id().unwrap();
            history.push(think_id);
            tokens = vec![think_id];
            // do NOT emit to output stream — thinking is internal
        }
        ThoughtAction::Abort => {
            // emit escalation signal, break
            break;
        }
    }
}
```

**Key properties:**
- The `<think>` token runs a full forward pass (attention, FFN, KV cache update)
  — the model gets extra computation
- RoPE position advances normally (the `<think>` occupies a real sequence position)
- The output token after thinking benefits from attending to the `<think>` KV entries
  via causal attention
- The user never sees `<think>` tokens — they're filtered from the output stream

### 4. KV Cache Behavior (Phase 4a)

In Phase 4a, `<think>` tokens are real tokens that accumulate in the KV cache
normally. This is acceptable because:

- Max 4 think steps per output token = at most 5x KV growth rate
- TurboQuant already compresses to 42 bytes/vector
- For 4K context on the target model: 63 MB baseline, worst case ~315 MB with
  continuous thinking at every token (unrealistic — most tokens are Confident)
- In practice, thinking triggers on hard tokens only (< 10% of generation)

**Phase 4b upgrade:** Replace persistent `<think>` KV entries with ephemeral
overwrites. See Section 7.

### 5. GGUF Metadata Extensions

New metadata keys in the GGUF file. These are informational — the inference engine
reads them to configure the `ContinuousThoughtConfig`:

```python
# In export_gguf.py write_metadata():
writer.add_bool("harmony.continuous_thought.enabled", True)
writer.add_uint32("harmony.continuous_thought.think_token_id", config.think_token_id)
writer.add_uint32("harmony.continuous_thought.max_steps", 4)
writer.add_float32("harmony.continuous_thought.confidence_threshold", 0.85)
```

The candle loader reads these keys and populates `ContinuousThoughtConfig`.
Models without these keys default to `think_token_id = None` (disabled),
maintaining backward compatibility with existing GGUF files.

### 6. PyTorch Training Changes

**Vocabulary:**
- Tokenizer: add `<think>` as special token (ID 32000)
- `embed_tokens`: grow from `[32000, hidden_dim]` to `[32001, hidden_dim]`
- `lm_head`: grow correspondingly (or automatically via tied weights)
- `HarmonyModelConfig.vocab_size` becomes 32001

**Phase 4a training data:**
- Take existing CoT training examples
- Replace the final N reasoning tokens before the answer with `<think>` tokens
- The model learns: "when I see `<think>` in context, I should use the extra
  computation to refine my hidden state before answering"
- Progressive curriculum (same as COCONUT): start replacing 1 token, then 2, etc.

**Loss calculation:**
- `<think>` tokens are included in the causal language modeling loss (the model
  learns to predict when thinking is appropriate)
- But during inference, the model doesn't "choose" to think — the UQ Head forces it
- This asymmetry is intentional: training teaches the model what `<think>` means;
  inference uses UQ to trigger it

### 7. Phase 4b Upgrade Path: Full COCONUT

Phase 4a establishes the control flow. Phase 4b replaces the mechanism:

| Aspect | Phase 4a (Pause Token) | Phase 4b (COCONUT) |
|--------|----------------------|-------------------|
| Thinking mechanism | `<think>` token, full pipeline | Hidden state feedback, bypass lm_head |
| Vocabulary | +1 token (`<think>`) | No extra tokens |
| KV cache | Normal accumulation | Ephemeral overwrite |
| RoPE position | Advances per think step | Static (same position for all think steps) |
| Engram | Single query at seq start | Re-query at each think step |
| BlockAttnRes | Normal block boundaries | Accumulate summaries across think steps |
| Training | CoT-to-`<think>` replacement | Progressive CoT-to-latent distillation |
| Compute per think | Full forward + vocab projection | Full forward, NO vocab projection |

**Phase 4b model changes (preview):**

1. `HarmonyModel::forward()` gains a `skip_lm_head: bool` parameter. When true,
   returns the final hidden state (post-RMSNorm) instead of logits.

2. `HarmonyEngine` gains `forward_latent()` that returns the hidden state tensor
   directly, plus layer norms for UQ evaluation.

3. The hidden state is fed back as the input embedding for the next step, bypassing
   `embed_tokens`. This requires `HarmonyModel::forward()` to accept either a
   token tensor OR an embedding tensor as input.

4. The KV cache gains an ephemeral mode: `cache.set_ephemeral(true)` causes
   writes to overwrite position `cache.position` instead of appending. On
   `set_ephemeral(false)`, `cache.position` advances by 1 (committing the
   final thinking state).

5. RoPE: All thinking steps use the same position index. This is semantically
   correct — the thinking happens "at" the current position, not across a
   sequence. It also avoids consuming RoPE position budget on internal reasoning.

6. Engram: Re-query at each thinking step. The evolving hidden state at Layer 2
   will have different semantic content after each think step, potentially
   triggering different Engram lookups (multi-hop retrieval).

7. BlockAttnRes: Block summaries accumulate across thinking steps within the
   same position, providing cross-step working memory. The softmax routing
   in BlockAttnRes bounds magnitude growth.

### 8. Open Questions for Phase 4b

- **RoPE static vs advancing:** Using the same RoPE position for all think steps
  means attention patterns during thinking are position-invariant. This may be
  fine (thinking is about content, not position) or may lose useful positional
  signal. Needs ablation.

- **BlockAttnRes accumulation stability:** Accumulating block summaries across
  think steps is structurally similar to RNN state. Monitor hidden state norms
  during training for explosion/collapse.

- **Engram re-query overhead:** Each think step with Engram re-query adds an
  O(1) hash lookup + embedding resolution. At 4 think steps, 4x the Engram
  traffic. This is cheap on local tables but may matter for mesh-distributed
  Engram in future.

- **UQ Head feature extraction during thinking:** Layer norms during thinking
  steps come from the feedback hidden state, not token embeddings. The UQ Head
  may need recalibration for thinking-step features.

## Data Model

### ContinuousThoughtConfig

```rust
pub struct ContinuousThoughtConfig {
    pub think_token_id: Option<u32>,
    pub max_think_steps: u32,
    pub confidence_threshold: f32,
}

impl Default for ContinuousThoughtConfig {
    fn default() -> Self {
        Self {
            think_token_id: None,
            max_think_steps: 4,
            confidence_threshold: 0.85,
        }
    }
}
```

### ThoughtAction

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ThoughtAction {
    Emit,
    Think,
    Abort,
}
```

### HarmonyForwardOutput (unchanged in 4a)

```rust
pub struct HarmonyForwardOutput {
    pub logits: Tensor,
    pub layer_norms: Vec<f32>,
}
```

### GGUF Metadata (new keys)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `harmony.continuous_thought.enabled` | bool | false | Activates UQ routing loop |
| `harmony.continuous_thought.think_token_id` | u32 | — | Vocabulary ID for `<think>` |
| `harmony.continuous_thought.max_steps` | u32 | 4 | Safety cap on consecutive thinks |
| `harmony.continuous_thought.confidence_threshold` | f32 | 0.85 | UQ confidence threshold for Emit |

## Testing

### Rust Tests

- `route_thought` returns `Emit` when UQ head is not set
- `route_thought` returns `Emit` when `think_token_id` is `None`
- `route_thought` returns `Think` when UQ classifies `Uncertain` and under max steps
- `route_thought` returns `Emit` when at max consecutive think steps (safety cap)
- `route_thought` returns `Abort` on `SpectralCollapse`
- `route_thought` returns `Emit` when UQ classifies `Confident` above threshold
- `ContinuousThoughtConfig` default values are correct
- `ContinuousThoughtConfig` loads from GGUF metadata (present and absent cases)
- `ThoughtAction` enum round-trips through serialization
- Generation loop with forced `Think` actions produces correct token count
- `<think>` tokens advance KV cache position correctly

### PyTorch Tests

- `HarmonyModelConfig` with `vocab_size=32001` constructs correctly
- `<think>` token embedding is trainable (gradients flow)
- Forward pass with `<think>` token in input produces valid logits
- GGUF export includes continuous thought metadata
- GGUF export handles `think_token_id=None` (metadata keys absent)

## Error Handling

| Scenario | Behavior |
|----------|----------|
| UQ head not set | `route_thought` returns `Emit` always |
| `think_token_id` is `None` | `route_thought` returns `Emit` always |
| Max think steps reached | Force `Emit` (safety cap) |
| SpectralCollapse during thinking | `Abort` immediately |
| GGUF missing continuous thought keys | Default to disabled (backward compat) |
| `think_token_id` >= vocab_size | Error on GGUF load (validated) |

## File Changes Summary

| File | Change |
|------|--------|
| `harmony-inference/src/harmony_engine.rs` | Add `ThoughtAction`, `ContinuousThoughtConfig`, `route_thought()` |
| `harmony-inference/src/harmony_model.rs` | Add `think_token_id` to `HarmonyModelConfig`, load from GGUF |
| `harmony-inference/src/lib.rs` | Re-export `ThoughtAction`, `ContinuousThoughtConfig` |
| `training/ct87/model.py` | Add `think_token_id` to config, adjust vocab_size |
| `training/ct87/export_gguf.py` | Write continuous thought GGUF metadata |
| `training/tests/test_model.py` | Test vocab_size+1 construction |
| `training/tests/test_export_gguf.py` | Test continuous thought metadata export |

## Out of Scope

- Full COCONUT continuous thought (hidden state feedback loop) — Phase 4b
- Ephemeral KV cache — Phase 4b
- Engram re-querying during thinking — Phase 4b
- RoPE position strategies for latent steps — Phase 4b
- JEPA-style Engram training objectives — separate research track
- Seamless context compaction — separate research track (see design note)
- Training the UQ Head for routing — Phase 3 (existing plan)
- `<think>` token in tokenizer — depends on tokenizer selection (Phase 1)

## Related Beads

| Bead | Relationship |
|------|-------------|
| harmony-ct87 | Parent epic |
| Phase 0d (UQ Head) | Prerequisite — UQ Head must be trained before routing |
| Phase 3 (Full 0.5B) | Prerequisite — base model must be trained first |
| Phase 4b (COCONUT) | Direct successor — replaces pause tokens with continuous thought |
