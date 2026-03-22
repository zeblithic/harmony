# Athenaeum CID Format Update Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> **Note:** The `- [ ]` checkboxes below are execution tracking markers for the agentic worker, not persistent TODO items. Task tracking uses `bd` (beads) — see bead `harmony-7rq`.

**Goal:** Update harmony-athenaeum's public API to accept CID hash portions (28 bytes) for routing, and document the CID↔page-hash distinction for future maintainers.

**Architecture:** The internal page partitioning already uses SHA-256 page content hashes (32 bytes, uniformly distributed) — this is unaffected by the CID format change. The public `Encyclopedia::route()` API takes a `content_hash: &[u8; 32]` which callers treat as a CID — this needs updating to accept the 28-byte hash portion. `route_page()` stays generic (bit extraction from any byte array). `Book.cid` stays `[u8; 32]` opaque.

**Tech Stack:** Rust, `harmony-athenaeum` (no_std)

**Spec:** `docs/superpowers/specs/2026-03-21-athenaeum-cid-update-design.md`

**Scope:** Bead harmony-7rq only.

**Key insight:** The internal partitioning (`build_volume`, line 146) calls `route_page()` with SHA-256 page content hashes — NOT CIDs. `PARTITION_START_BIT=28` exists because bits 0-27 are used for PageAddr (the 28-bit `hash_bits` field). This is independent of CID format. The only CID-format-sensitive API is `Encyclopedia::route()` (line 282) and the `entries` parameter to `Encyclopedia::build()` which accepts `&[([u8; 32], &[u8])]` — the `[u8; 32]` being CIDs.

---

## File Structure

```
crates/harmony-athenaeum/src/
├── volume.rs         — Update MAX_PARTITION_DEPTH comment, keep route_page as-is
├── encyclopedia.rs   — Update Encyclopedia::route() to accept &[u8; 28], add CID-aware build variant
└── lib.rs            — Update integration tests that construct CIDs
```

---

### Task 1: Update Encyclopedia::route() to accept hash portion

**Files:**
- Modify: `crates/harmony-athenaeum/src/encyclopedia.rs`

The public `Encyclopedia::route()` function accepts a `content_hash: &[u8; 32]` and routes using `PARTITION_START_BIT` offset. With the new CID format, callers should pass the 28-byte hash portion (bytes 4-31 of the CID) instead of the full CID, since bits 0-31 are metadata.

- [ ] **Step 1: Write test for 28-byte hash routing**

Add to the existing test module in `encyclopedia.rs`:

```rust
#[test]
fn route_with_hash_portion() {
    // The hash portion is 28 bytes (bytes 4-31 of a CID).
    // route_hash should extract bits from this, starting at
    // PARTITION_START_BIT (28) which is the PageAddr boundary.
    let mut hash = [0u8; 28];
    // Set bit 28 (byte 3, bit offset 7-4=3... actually bit 28 of the
    // 28-byte array is byte 3, offset 7-(28%8)=7-4=3)
    // Bit 28 in 28-byte array = byte 3, bit 4 from MSB
    hash[3] = 0b0000_1000; // bit 28 of the hash = 1
    let path = Encyclopedia::route_hash(&hash, 5);
    assert_eq!(path & 1, 1); // first routing bit should be 1
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p harmony-athenaeum route_with_hash_portion`
Expected: FAIL — `route_hash` not defined.

- [ ] **Step 3: Implement route_hash()**

Add to `Encyclopedia` impl in `encyclopedia.rs`:

```rust
    /// Determine which partition a CID's hash maps to.
    ///
    /// `hash` is the 28-byte hash portion of a ContentId (bytes 4-31).
    /// Routing starts at `PARTITION_START_BIT` (28) because bits 0-27
    /// are used for PageAddr addressing.
    ///
    /// For callers with a full 32-byte CID: `Encyclopedia::route_hash(&cid[4..].try_into().unwrap(), depth)`
    pub fn route_hash(hash: &[u8; 28], depth: u8) -> u32 {
        // Extend to 32 bytes for route_page compatibility.
        // Prepend 4 zero bytes (the header position) so bit indices align
        // with the page content hash layout used internally.
        let mut padded = [0u8; 32];
        padded[4..].copy_from_slice(hash);
        Self::route(&padded, depth)
    }
```

Alternatively, keep the existing `route()` and add `route_hash()` as a convenience. The old `route()` still works for internal use with page content hashes.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p harmony-athenaeum route_with_hash_portion`

- [ ] **Step 5: Update MAX_PARTITION_DEPTH comment**

In `volume.rs`, update the comment on `MAX_PARTITION_DEPTH`:

```rust
/// Maximum partition depth.
///
/// Partition routing uses bits PARTITION_START_BIT..PARTITION_START_BIT+MAX_PARTITION_DEPTH
/// from the content hash. For page content hashes (SHA-256, 32 bytes), this gives
/// 228 usable bits (28..255). For CID hash portions (28 bytes from ContentId),
/// the same bit indices apply after zero-padding the 4-byte header position.
pub const MAX_PARTITION_DEPTH: u8 = 228;
```

The value stays 228 — the routing operates on padded 32-byte arrays where the hash starts at byte 4, same as the internal page hashes where bits 0-27 are used for PageAddr.

- [ ] **Step 6: Run all athenaeum tests**

Run: `cargo test -p harmony-athenaeum`
Expected: All existing tests pass + new test passes.

- [ ] **Step 7: Commit**

```bash
git add crates/harmony-athenaeum/
git commit -m "feat(athenaeum): add Encyclopedia::route_hash() for CID hash portions

Accepts 28-byte hash (bytes 4-31 of ContentId) for partition routing.
Existing route() retained for internal page content hash routing."
```

---

### Task 2: Document CID vs page-hash distinction

**Files:**
- Modify: `crates/harmony-athenaeum/src/encyclopedia.rs`
- Modify: `crates/harmony-athenaeum/src/athenaeum.rs`

Add documentation clarifying the distinction between CIDs (32-byte ContentId with 4-byte header + 28-byte hash) and page content hashes (32-byte SHA-256 of page data).

- [ ] **Step 1: Add doc comment to Encyclopedia::build()**

Update the `build()` method's doc comment to clarify the `entries` parameter:

```rust
    /// Build an Encyclopedia from a collection of books.
    ///
    /// Each entry is `(cid, data)` where:
    /// - `cid` is the full 32-byte ContentId (stored as-is in `Book.cid`)
    /// - `data` is the raw book content
    ///
    /// **CID vs page hash:** Internally, pages are deduplicated and partitioned
    /// by SHA-256 of page data (full 32 bytes, uniformly distributed), NOT by
    /// the CID. The CID is stored opaquely in `Book.cid`. For external routing
    /// lookups, use `route_hash()` with the 28-byte hash portion of the CID.
```

- [ ] **Step 2: Add doc comment to Book.cid field**

In `athenaeum.rs`, update the `cid` field doc:

```rust
    /// Full 32-byte ContentId, stored as opaque bytes.
    ///
    /// The CID format is `[4-byte header][28-byte hash]` (see harmony-content).
    /// The athenaeum does not parse the header — it stores the CID as-is and
    /// uses SHA-256 of page data (not the CID hash) for internal routing.
    pub cid: [u8; 32],
```

- [ ] **Step 3: Commit**

```bash
git add crates/harmony-athenaeum/
git commit -m "docs(athenaeum): clarify CID vs page-hash distinction in API docs"
```

---

### Task 3: Verify no regressions and cleanup

**Files:**
- Various (clippy fixes if needed)

- [ ] **Step 1: Run clippy**

Run: `cargo clippy -p harmony-athenaeum`

- [ ] **Step 2: Run workspace tests**

Run: `cargo test --workspace`

- [ ] **Step 3: Commit cleanup if needed**

```bash
git add -A
git commit -m "chore: clippy fixes for athenaeum CID update"
```

---

## Summary

| Task | Description | Key Output |
|------|-------------|------------|
| 1 | `Encyclopedia::route_hash()` for 28-byte CID hashes | New public method, backward-compatible |
| 2 | Document CID vs page-hash distinction | API docs on build(), Book.cid |
| 3 | Verify no regressions | Clippy clean, workspace tests pass |

**What doesn't change:**
- `route_page()` — stays `&[u8; 32]`, used internally with page content hashes
- `Encyclopedia::route()` — stays `&[u8; 32]`, works with page content hashes
- `PARTITION_START_BIT` — stays 28 (PageAddr boundary, not CID header)
- `MAX_PARTITION_DEPTH` — stays 228
- `Book.cid` — stays `[u8; 32]`
- `PageAddr` — unchanged (hashes page data)
- Internal page dedup/partition logic — unchanged

**Why the scope is smaller than initially estimated:** The internal partition routing uses SHA-256 page content hashes (full 32 bytes, uniformly distributed) — NOT CIDs. `PARTITION_START_BIT=28` exists because bits 0-27 are used for PageAddr addressing, not because of CID header bytes. The only CID-format-sensitive surface is the public `route()` API and documentation.
