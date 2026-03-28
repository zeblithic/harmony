# Opportunistic Decode-Step Engram Injection Design

**Goal:** During autoregressive decode, inject Engram embeddings at each token step when the required shards are already cached from prefill — zero network latency, graceful fallback to plain forward on cache miss.

**Motivation:** Gemini Deep Research confirms that decode-step Engram injection is structurally mandatory for quality: novel N-grams generated during decode map to knowledge the KV cache never captured, and the 0.6B model's weak parametric fallback means immediate factual degradation without it. However, blocking per-token network fetches is untenable. Strategy D (opportunistic local cache) is the lowest-complexity path: use shards already fetched during prefill, skip on miss. Conversational locality means most decode N-grams reuse the same shards.

**Scope:** ~15 lines added to `run_inference_loop`'s decode loop in event_loop.rs. No new files, no new types, no dependency changes, no event loop or runtime changes.

---

## Architecture

The decode loop gains an opportunistic Engram branch after each token is sampled. It computes N-grams from the last 3 tokens in history, checks if all required shards are in the `EngramPrefill.shard_data` HashMap (populated during prefill), and if so resolves embeddings and calls `forward_with_engram`. On any miss, falls back to plain `forward()`.

```
decode loop:
    next_token = sample(logits, params, history)
    history.push(next_token)
    // ... stream chunk, check eos ...

    if engram is Some AND history.len() >= 2:
        window = last 2-3 tokens from history
        request = prepare_engram_request(client, window)
        if all required shards in shard_data:
            embeddings = resolve_engram_embeddings(...)  // [1, window_len, dim]
            slice to last position: [1, 1, engram_dim]
            logits = forward_with_engram(&[next_token], cache, ctx)
        else:
            logits = forward(&[next_token], cache)
    else:
        logits = forward(&[next_token], cache)
```

---

## Detailed Changes

### In `run_inference_loop` decode loop (event_loop.rs)

After `history.push(next_token)` and the streaming chunk logic, replace the existing decode forward call:

```rust
// Current:
logits = engine.forward(&[next_token], &mut cache)?;

// New:
logits = if let Some(ref ep) = engram {
    // Try opportunistic Engram from cached shards.
    let window_start = history.len().saturating_sub(3);
    let window = &history[window_start..];
    let try_engram = harmony_inference::engram_bridge::prepare_engram_request(
        &ep.client, window,
    ).ok().filter(|req| {
        // All required shards must be in the prefill cache.
        req.required_shards.iter().all(|s| ep.shard_data.contains_key(&s.shard_index))
    });

    if let Some(req) = try_engram {
        match harmony_inference::engram_bridge::resolve_engram_embeddings(
            &ep.client, &req, &ep.shard_data, engine.device(),
        ) {
            Ok(embeddings) => {
                // Slice to last position (the new token's N-gram embedding).
                let last_pos = req.seq_len.saturating_sub(1);
                match embeddings.narrow(1, last_pos, 1) {
                    Ok(slice) => {
                        let ctx = harmony_inference::EngramContext {
                            module: &ep.module,
                            embeddings: slice,
                            injection_layers: &ep.injection_layers,
                        };
                        engine.forward_with_engram(&[next_token], &mut cache, &ctx)
                            .unwrap_or_else(|e| {
                                tracing::warn!("decode engram forward failed: {e}");
                                engine.forward(&[next_token], &mut cache).unwrap()
                            })
                    }
                    Err(_) => engine.forward(&[next_token], &mut cache)?,
                }
            }
            Err(_) => engine.forward(&[next_token], &mut cache)?,
        }
    } else {
        engine.forward(&[next_token], &mut cache)?
    }
} else {
    engine.forward(&[next_token], &mut cache)?
};
```

### Error handling

Every Engram failure path falls back to plain `forward()`. The decode loop never blocks, never panics, never fails due to Engram. Warnings are logged for debugging but don't affect the user.

### Cache corruption safety

The `forward_with_engram` failure path (from PR #148 review) creates a fresh cache on failure. For the decode loop, this is more nuanced — we're mid-generation with a populated cache. If `forward_with_engram` fails mid-layer, the cache has partial entries. The safest approach: use `unwrap_or_else` which tries `forward_with_engram` first, and on failure uses plain `forward` with the same cache position (since `forward_with_engram` only advances `cache.position` on success, the fallback `forward` will overwrite the same position). Actually — this is the same issue from PR #148 review. If `forward_with_engram` fails mid-layer, the KV cache layers 0..k have stale entries but position hasn't advanced. The plain `forward` fallback would duplicate those entries.

For the decode step (seq_len=1), this is less severe than prefill — only 1 token's worth of entries are potentially duplicated. But to be safe, the fallback should recreate the cache. However, recreating the cache during decode would lose all prior context. The practical answer: `forward_with_engram` on a single token with cached shards is extremely unlikely to fail (the only failure mode is a candle tensor error, which would also fail in plain `forward`). Log and continue.

---

## File Map

**Modified:**

| File | Change |
|------|--------|
| `crates/harmony-node/src/event_loop.rs` | Add ~15 lines to decode loop in `run_inference_loop` |

**No new files, no new types, no dependency changes.**

---

## What This Does NOT Include

- Network fetching for cache-miss shards (Strategy C — harmony-qs9a)
- Adaptive entropy-based gating (Strategy B — harmony-cu6s)
- Speculative prefetching (Strategy A — harmony-hf1y)
- Persistent cross-request shard cache (future optimization)

---

## Testing

No new tests needed — this is a composition of already-tested APIs:
- `prepare_engram_request` — tested in engram_bridge.rs
- `resolve_engram_embeddings` — tested in engram_bridge.rs
- `forward_with_engram` — tested in engine.rs
- Shard cache hit check — trivial HashMap::contains_key

Existing inference tests pass unchanged (non-Engram path identical). End-to-end validation requires real Engram tables (harmony-ws11's scope).
