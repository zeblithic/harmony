# Telemetry Schema and Pub/Sub Namespace

## Goal

Define structured telemetry event types and Zenoh namespace for edge nodes to publish environmental observations and device health to the mesh. Protocol-layer only — no inference or NodeRuntime dependency.

## Architecture

New `harmony-telemetry` crate with a `TelemetryEvent` type, wire encode/decode, and a `TelemetryError` enum. New namespace builders in `harmony-zenoh`. The crate is independently consumable — no dependency on harmony-agent or harmony-inference.

Any node can publish telemetry. Events are tagged by intent (what kind of observation) and carry optional metadata (confidence, source). The intent is a free-form string — common intents are documented as conventions, not enforced by the type system.

### Dependencies

| Crate | Purpose |
|-------|---------|
| `harmony-telemetry` (new) | `TelemetryEvent` type, wire encode/decode |
| `harmony-zenoh` | Namespace builders for telemetry key expressions |

### Workspace Setup

Add to root `Cargo.toml`:
- `"crates/harmony-telemetry"` to `[workspace] members`
- `harmony-telemetry = { path = "crates/harmony-telemetry" }` to `[workspace.dependencies]`

### Crate Manifest

`crates/harmony-telemetry/Cargo.toml`:

```toml
[package]
name = "harmony-telemetry"
version.workspace = true
edition.workspace = true
license.workspace = true
repository.workspace = true
rust-version.workspace = true
description = "Structured telemetry event types for the Harmony mesh network"

[dependencies]
serde = { workspace = true, features = ["derive"] }
serde_json.workspace = true
```

No `no_std` support needed — telemetry publishers are always nodes with a full OS.

### File Layout

| File | Responsibility |
|------|---------------|
| `crates/harmony-telemetry/src/lib.rs` | Module declarations, re-exports |
| `crates/harmony-telemetry/src/types.rs` | `TelemetryEvent` struct |
| `crates/harmony-telemetry/src/wire.rs` | `encode_event`, `decode_event`, `TelemetryError` |

Public API surface (re-exports from `lib.rs`):
```rust
pub use types::TelemetryEvent;
pub use wire::{encode_event, decode_event, TelemetryError};
```

### Intent Categories

Intents are free-form strings. Documented conventions:

**Environmental:**
- `object_detected` — ML model identified an object in a sensor feed
- `scene_changed` — significant change in visual scene
- `anomaly` — unexpected observation requiring attention
- `ocr_result` — text extracted from an image

**Device:**
- `health` — CPU, memory, temperature, disk usage
- `inference_complete` — an inference task finished (timing, tokens/sec)
- `capacity_changed` — node's available compute capacity changed

Custom intents are encouraged — the schema does not restrict them. Intent strings MUST NOT contain Zenoh key expression reserved characters (`/`, `*`, `?`, `#`, `$`, `**`) as they are used directly in key paths.

## TelemetryEvent Type

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryEvent {
    /// Node address (lowercase hex) that produced this event.
    pub node_addr: String,
    /// Intent tag — what kind of observation (e.g. "object_detected", "health").
    pub intent: String,
    /// Monotonic event sequence number (per-node, 0-indexed).
    pub sequence: u64,
    /// Unix epoch seconds when the event was produced.
    pub timestamp: u64,
    /// Intent-specific payload.
    pub payload: serde_json::Value,
    /// Optional confidence score (0.0–1.0) for ML-derived events.
    pub confidence: Option<f32>,
    /// Optional source descriptor (e.g. "camera:0", "cpu_monitor", "qwen3vl-4b").
    pub source: Option<String>,
}
```

- `node_addr` — lowercase hex-encoded Harmony address hash of the publishing node (matches `hex::encode` output). Must be lowercase — Zenoh key expression matching is case-sensitive, so publisher and subscriber must agree on casing.
- `intent` — free-form string matching the `{intent}` segment in the key expression. Must not contain `/`, `*`, or other Zenoh reserved characters. Callers are responsible for validation.
- `sequence` — `u64` because telemetry events accumulate over the node's lifetime (not bounded per-task like `StreamChunk`'s `u32`).
- `timestamp` — Unix epoch seconds, provided by the caller (sans-I/O — no `SystemTime::now()` inside the crate).
- `payload` — opaque JSON, intent-specific. No schema validation in the crate. Any valid `serde_json::Value` is accepted (including `null`, nested objects, arrays).
- `confidence` — optional `f32` in `[0.0, 1.0]`. Present for ML-derived events. Absent for deterministic observations. Not clamped — out-of-range handling is the consumer's responsibility.
- `source` — optional string describing what produced the event.

## Wire Format

Same `[tag][json_bytes]` convention as harmony-agent:

```
[0x00 (JSON_TAG)] [json_bytes]
```

New functions in `harmony-telemetry/src/wire.rs`:

- `encode_event(event: &TelemetryEvent) -> Result<Vec<u8>, serde_json::Error>`
- `decode_event(payload: &[u8]) -> Result<TelemetryEvent, TelemetryError>`

The asymmetry (encode returns `serde_json::Error`, decode returns `TelemetryError`) matches the harmony-agent pattern. There is only one message type (`TelemetryEvent`), so one encode/decode pair suffices (unlike harmony-agent which has per-type pairs for task/result/capacity).

### TelemetryError

Defined in `wire.rs`, co-located with encode/decode functions (same pattern as `AgentError`).

```rust
#[derive(Debug)]
pub enum TelemetryError {
    /// Payload was empty (no tag byte).
    EmptyPayload,
    /// Unknown wire format tag.
    UnsupportedFormat(u8),
    /// JSON deserialization failed.
    Json(serde_json::Error),
}
```

Must implement `Display` and `std::error::Error` (with `source()` returning `Some(e)` for the `Json` variant), mirroring `AgentError`'s trait implementations.

## Key Expressions

| Key Expression | Purpose | Builder |
|---------------|---------|---------|
| `harmony/telemetry/{node_addr}/{intent}` | Per-node, per-intent publish key | `telemetry_key(node_addr, intent)` |
| `harmony/telemetry/{node_addr}/*` | All intents from one node | `telemetry_sub_node(node_addr)` |
| `harmony/telemetry/*/{intent}` | One intent across all nodes | `telemetry_sub_intent(intent)` |

These live in a new `telemetry` module in `harmony-zenoh/src/namespace.rs`, following the pattern of the existing `agent`, `compute`, and `content` modules. The module uses `alloc::{format, string::String}` for `no_std` compatibility (matching all other namespace modules).

```rust
pub mod telemetry {
    use alloc::{format, string::String};

    /// Base prefix: `harmony/telemetry`
    pub const PREFIX: &str = "harmony/telemetry";

    /// Per-node, per-intent telemetry key.
    pub fn telemetry_key(node_addr: &str, intent: &str) -> String {
        format!("{PREFIX}/{node_addr}/{intent}")
    }

    /// Subscribe to all intents from one node.
    pub fn telemetry_sub_node(node_addr: &str) -> String {
        format!("{PREFIX}/{node_addr}/*")
    }

    /// Subscribe to one intent across all nodes.
    pub fn telemetry_sub_intent(intent: &str) -> String {
        format!("{PREFIX}/*/{intent}")
    }
}
```

## Testing Strategy

### Unit tests in `harmony-telemetry`

- `TelemetryEvent` serde roundtrip (all fields populated, verify `node_addr` and `intent` preserved)
- `TelemetryEvent` serde roundtrip (optional fields absent — `confidence: None`, `source: None`)
- `TelemetryEvent` serde roundtrip with varied payloads (`Value::Null`, nested objects, arrays)
- Snake_case key verification (no camelCase in JSON output)
- Wire encode/decode roundtrip (verify `node_addr`, `intent`, `sequence` preserved)
- Wire decode error cases: empty payload, wrong tag, invalid JSON
- `TelemetryError` Display output for each variant

### Namespace tests in `harmony-zenoh`

- `telemetry_key("node01", "anomaly")` → `"harmony/telemetry/node01/anomaly"`
- `telemetry_sub_node("node01")` → `"harmony/telemetry/node01/*"`
- `telemetry_sub_intent("health")` → `"harmony/telemetry/*/health"`

## Scope Exclusions

- **No NodeRuntime integration** — separate bead (harmony-60nv sensor pipeline)
- **No event aggregation or dedup** — consumers handle this
- **No retention or history** — fire-and-forget pub/sub
- **No schema validation of payload** — payload is opaque JSON, intent-specific
- **No confidence range enforcement** — documented as `[0.0, 1.0]` but not clamped; consumer's responsibility
- **No intent string validation** — callers must avoid Zenoh reserved characters; the crate does not enforce this
- **No VLM or inference dependency** — any node can publish telemetry regardless of ML capability
