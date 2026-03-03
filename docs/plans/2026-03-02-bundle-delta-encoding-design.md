# Bundle Delta-Encoding Design

**Goal:** Efficient bundle updates via COPY/INSERT opcodes so that when a file changes, only the differing CIDs are transferred instead of the full bundle.

**Architecture:** A new `delta.rs` module with three free functions: `compute_delta()` produces a compact opcode stream, `apply_delta()` reconstructs the new bundle, and `encode_update()` picks whichever is smaller (delta or full bundle). CID-aligned granularity — opcodes work in units of 32-byte CIDs, not raw bytes. Linear scan algorithm. Sans-I/O, pure functions.

---

## Wire Format

Two opcodes, CID-aligned:

| Opcode | Format | Size |
|--------|--------|------|
| COPY | `0x00 [cid_offset:u16 BE] [cid_count:u16 BE]` | 5 bytes |
| INSERT | `0x01 [cid_count:u16 BE] [cid_data: count * 32 bytes]` | 3 + N×32 bytes |

A delta is a sequence of these opcodes concatenated. Applying the opcodes in order against the old bundle reconstructs the new bundle.

## API

```rust
/// Compute a CID-aligned delta between two bundles.
pub fn compute_delta(old_bundle: &[u8], new_bundle: &[u8]) -> Result<Vec<u8>, ContentError>

/// Apply a delta to an old bundle, producing the new bundle.
pub fn apply_delta(old_bundle: &[u8], delta: &[u8]) -> Result<Vec<u8>, ContentError>

/// Encode a bundle update: returns either the delta or the full new bundle,
/// whichever is smaller. First byte: 0x00 = full bundle, 0x01 = delta.
pub fn encode_update(old_bundle: &[u8], new_bundle: &[u8]) -> Result<Vec<u8>, ContentError>
```

Both `compute_delta` and `apply_delta` validate inputs are valid bundle lengths (multiples of 32). `encode_update` is the caller-facing convenience for transport code.

## Algorithm — Linear Scan

1. Parse both old and new as CID slices via `parse_bundle`.
2. Walk new CIDs left-to-right with a cursor approach:
   - Maintain `old_cursor` and `new_cursor` starting at 0.
   - If `new[new_cursor] == old[old_cursor]`, extend a COPY run (advance both).
   - Otherwise, extend an INSERT run (advance `new_cursor` only).
   - Flush current run when transitioning between COPY and INSERT.
3. After the forward pass, do a suffix check: scan backward from both ends looking for a matching tail. This catches prepend and middle-insert patterns.

Handles: identical bundles, append, prepend, single CID change, complete replacement.

## Error Handling

One new `ContentError` variant:

- `InvalidDelta { reason: &'static str }` — malformed delta (bad opcode, truncated data, COPY out of bounds)

Reuses existing `InvalidBundleLength` for inputs that aren't multiples of 32 bytes.

## Testing (~8 tests)

1. **Identical bundles** — delta is a single COPY, much smaller than full bundle
2. **Single CID change** — COPY + INSERT(1) + COPY, round-trips correctly
3. **Append** — old 3 CIDs, new 5 (same prefix + 2 new) → COPY + INSERT
4. **Prepend** — new has extra CIDs before old content → INSERT + COPY
5. **Complete replacement** — all different, `encode_update` picks full bundle
6. **Empty old bundle** — entire new bundle is one INSERT
7. **apply_delta round-trip** — `apply_delta(old, compute_delta(old, new)) == new`
8. **Invalid delta rejected** — truncated/malformed → `InvalidDelta` error
