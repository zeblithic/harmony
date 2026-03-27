# Telemetry Schema Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add structured telemetry event types and Zenoh namespace so edge nodes can publish environmental observations and device health to the mesh.

**Architecture:** New `harmony-telemetry` crate with `TelemetryEvent` type, wire encode/decode, and `TelemetryError`. New `telemetry` module in `harmony-zenoh` namespace. Follows the same patterns as `harmony-agent` (JSON wire format, serde types, separate error enum).

**Tech Stack:** harmony-telemetry (new), harmony-zenoh

**Spec:** `docs/superpowers/specs/2026-03-26-telemetry-schema-design.md`

---

## File Map

| File | Responsibility |
|------|---------------|
| `crates/harmony-telemetry/Cargo.toml` | New crate manifest |
| `crates/harmony-telemetry/src/lib.rs` | Module declarations, re-exports |
| `crates/harmony-telemetry/src/types.rs` | `TelemetryEvent` struct + tests |
| `crates/harmony-telemetry/src/wire.rs` | `encode_event`, `decode_event`, `TelemetryError` + tests |
| `Cargo.toml` | Workspace — add member + dep |
| `crates/harmony-zenoh/src/namespace.rs` | Add `telemetry` module + tests |

---

### Task 1: New Crate — harmony-telemetry with TelemetryEvent

**Files:** Create `crates/harmony-telemetry/` (Cargo.toml, src/lib.rs, src/types.rs). Modify root `Cargo.toml`.

- [ ] **Step 1: Create crate manifest and workspace entries**
- [ ] **Step 2: Create lib.rs with module declarations and re-exports**
- [ ] **Step 3: Create types.rs with TelemetryEvent struct and 4 unit tests**
- [ ] **Step 4: Verify: `cargo test -p harmony-telemetry`**
- [ ] **Step 5: Commit**

### Task 2: Wire Encode/Decode + TelemetryError

**Files:** Create `crates/harmony-telemetry/src/wire.rs`.

- [ ] **Step 1: Create wire.rs with encode_event, decode_event, TelemetryError (Display + Error impls), and 5 unit tests**
- [ ] **Step 2: Verify: `cargo test -p harmony-telemetry`**
- [ ] **Step 3: Commit**

### Task 3: Zenoh Namespace Builders

**Files:** Modify `crates/harmony-zenoh/src/namespace.rs`.

- [ ] **Step 1: Add telemetry module with PREFIX, telemetry_key, telemetry_sub_node, telemetry_sub_intent**
- [ ] **Step 2: Add telemetry_namespace_keys test**
- [ ] **Step 3: Verify: `cargo test -p harmony-zenoh`**
- [ ] **Step 4: Commit**
