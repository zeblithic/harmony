# DSD Session Timeout and Cleanup

## Overview

Fix a bug where a stale DSD session blocks all inference and speculative queries forever if the target node never replies to a VerifyRequest. Add tick-based timeout eviction following the existing `pending_memo_fetches` pattern.

**Goal:** DSD sessions that receive no VerifyResponse within 10 seconds are automatically cleared, unblocking the engine.

**Scope:** One-file bug fix in runtime.rs. ~20 lines of changes.

## The Bug

When `handle_speculative_query()` starts a DSD session, it sets `self.dsd_session = Some(DsdSession { ... })`. The session is cleared in `handle_verify_response()` on success, error, or completion. But if the target node drops the connection, crashes, or the response is lost in transit, `handle_verify_response` is never called. The session stays active forever, and all subsequent inference requests get "busy: DSD session active" rejections.

## Fix

### New constant

```rust
/// Ticks before a DSD session with no VerifyResponse is evicted.
/// 40 ticks × ~250ms/tick = ~10 seconds.
const DSD_SESSION_TIMEOUT_TICKS: u64 = 40;
```

### New field

```rust
/// Tick when the current DSD session last received activity.
#[cfg(feature = "inference")]
dsd_session_last_activity_tick: u64,
```

### Tick eviction

In `tick()`, after the existing `pending_memo_fetches.retain(...)` eviction:

```rust
#[cfg(feature = "inference")]
if let Some(ref session) = self.dsd_session {
    if self.tick_count.saturating_sub(self.dsd_session_last_activity_tick)
        >= DSD_SESSION_TIMEOUT_TICKS
    {
        let query_id = session.query_id;
        tracing::warn!(query_id, "DSD session timed out");
        let mut payload = vec![0x01];
        payload.extend_from_slice(b"DSD session timed out");
        self.pending_direct_actions
            .push(RuntimeAction::SendReply { query_id, payload });
        self.dsd_session = None;
    }
}
```

### Set timer on session start

Where `dsd_session` is set to `Some(...)` (one location, in `handle_speculative_query`):

```rust
self.dsd_session_last_activity_tick = self.tick_count;
```

### Reset timer on verify response

At the top of `handle_verify_response`, after confirming the session exists:

```rust
self.dsd_session_last_activity_tick = self.tick_count;
```

This ensures multi-round DSD iterations don't time out — only genuinely stale sessions (no response at all) are evicted.

## Testing

1. **Session expires after timeout** — set dsd_session, advance 40 ticks without calling handle_verify_response, verify session cleared + SendReply with error.
2. **Session NOT expired before timeout** — set dsd_session, advance 39 ticks, verify session still active.
3. **Verify response resets timer** — set dsd_session, advance 30 ticks, call handle_verify_response (resets timer), advance 30 more ticks, verify session still active (only 30 ticks since last activity, not 60).

## Files Modified

| File | Change |
|------|--------|
| `crates/harmony-node/src/runtime.rs` | Add constant, field, init, tick eviction, set on session start, reset on verify response |

## What This Does NOT Include

- Configurable timeout — hardcoded at 40 ticks. Can be made configurable later if needed.
- Retry logic — if the target is unresponsive, the session is simply cleared. The next speculative query will try again.
- Target health tracking — no blacklisting of unresponsive targets.
