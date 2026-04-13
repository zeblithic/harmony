# harmony-db Phase 3 Design — Incremental O(log n) Prolly Tree Mutations

## Goal

Replace the O(n) full tree rebuild on every mutation with path-local incremental
updates. Walk to the affected leaf, mutate, rechunk forward until boundaries
re-converge with the original tree, cascade new CIDs up to root. Expected cost:
O(log n) per mutation with ~4-5 node rewrites for a typical 4-level tree.

This transforms N sequential inserts from O(N*M) (where M is total entry count)
to O(N log M), making harmony-db viable for large datasets with frequent writes.

## Background

Phase 2 (ZEB-98, PR #218) shipped the Prolly Tree engine with correct
history-independent topology, O(d) diffing, and CAS persistence. However, every
mutation calls `build_tree` — a full bottom-up rebuild from the sorted entry
cache. This processes ALL entries through the CDF chunker, writes ALL leaf nodes,
and builds ALL branch levels per mutation.

The public API, read operations, diff, and commit/rebuild are all unaffected by
this change. Only the internal mutation path changes.

## Scope

**In scope:**
- Incremental insert with walk-forward rechunking and boundary convergence
- Incremental remove with walk-forward rechunking and boundary convergence
- Incremental update_meta with CID-only cascade (no rechunking — keys unchanged)
- Tree path representation for descent/ascent
- Sibling leaf loading for boundary ripple cases
- Tree height growth/shrinkage during cascade
- Equivalence tests (incremental produces same root CID as full rebuild)
- Performance validation (CAS write count per mutation)

**Out of scope:**
- Batch mutations (multiple inserts in one operation)
- Read path changes (reads still served from in-memory cache)
- Changes to diff, commit, rebuild, or public API
- Chunker config tuning
- Async/concurrent mutations

## Architecture

```text
ProllyTree::insert/remove/update_meta
    │
    ├── 1. Update cache (unchanged — binary search insert/remove/replace)
    ├── 2. Snapshot cache/root for rollback (unchanged)
    └── 3. Instead of rebuild_tree(), call:
            │
            mutate.rs
            ├── walk_to_leaf(root, key, data_dir) → TreePath
            ├── apply mutation to leaf entries
            ├── rechunk_forward(entries, path, config, data_dir) → new children
            └── cascade_up(path, new_children, config, data_dir) → new root CID
```

### File structure

```text
prolly/
├── mod.rs       # ProllyTree struct, read ops, delegates mutations to mutate.rs
├── mutate.rs    # NEW: TreePath, walk_to_leaf, rechunk_forward, cascade_up
├── chunker.rs   # Unchanged
├── node.rs      # Unchanged
└── diff.rs      # Unchanged
```

## Tree Path Representation

```rust
/// One level of the descent path from root to leaf.
struct PathLevel {
    /// The branch entries at this level (full node contents).
    entries: Vec<BranchEntry>,
    /// Which child index we followed during descent.
    child_idx: usize,
}

/// Complete path from root to a leaf, plus the leaf's entries.
struct TreePath {
    /// Branch levels from root (index 0) to parent-of-leaf (last).
    levels: Vec<PathLevel>,
    /// The leaf entries at the bottom of the path.
    leaf_entries: Vec<LeafEntry>,
}
```

**Walking algorithm** (`walk_to_leaf`):

1. Start at root CID. If None, return empty path + empty leaf.
2. Read the root node from CAS.
3. If Leaf: return empty levels + leaf entries.
4. If Branch: binary search on `boundary_key` to find child index for the target
   key. Push `PathLevel { entries, child_idx }`. Read child node. Repeat.
5. When we reach a Leaf: return accumulated levels + leaf entries.

Binary search convention: `boundary_key` is the highest key in the subtree. For
key `k`, find the first branch entry where `boundary_key >= k`. If `k` is
greater than all boundary keys, use the last child.

## Rechunk Forward (Insert/Remove)

After applying a mutation to the leaf's entry sequence, we rechunk forward until
new chunk boundaries re-converge with the original tree's boundaries.

### Algorithm

1. Start with the mutated leaf's entries (sorted, mutation applied).
2. Run the CDF chunker entry by entry using `is_boundary()`.
3. Finalize chunks as boundaries fire.
4. **Check convergence** after each finalized chunk: does the chunk's last key
   match the boundary key of one of the original leaves we've consumed?
   - **Yes (converged):** Done. Remaining original leaves are unchanged.
   - **No — under-shot (new boundary before original):** The leaf split. Continue
     chunking the remaining entries — they start a fresh chunk.
   - **No — over-shot (no boundary at original edge):** Load the next sibling
     leaf's entries from the parent's branch entries. Append to the working set.
     Continue chunking. This is the "ripple" case.
5. Repeat until convergence or all entries consumed.

### Convergence property

Content-defined chunking has a natural convergence property: after a boundary
shift, the probability of the next boundary also shifting drops exponentially.
The expected number of extra leaves loaded is < 0.1 per mutation. Pathological
cases (many consecutive shifts) require the inserted entry to produce a key hash
that disrupts multiple consecutive boundary decisions — vanishingly unlikely with
BLAKE3.

### Accessing sibling leaves

Each `PathLevel` records the full branch entries. To load the next sibling:
1. Increment `child_idx` in the deepest (parent) level.
2. Read the child node at the new index from CAS.
3. If the child is a Branch (tree depth > 2), walk down to its leftmost leaf.
4. Append the leaf entries to the working set.

If we exhaust all siblings at the parent level (e.g., the mutation is in the
last leaf of the last branch), the rechunking has consumed all remaining entries
in that subtree. The cascade then proceeds upward normally — no special handling
needed because there are no more entries to absorb.

## CID Cascade (All Mutations)

After rechunking produces new leaf nodes, cascade changes up through branch
levels to produce a new root CID.

### Algorithm

1. Start with new leaf-level children (list of `(boundary_key, child_cid)` from
   rechunking).
2. Pop the bottom `PathLevel` from the path stack.
3. Replace affected child entries with the new children:
   - **Same count (1-to-1):** Replace CID and boundary_key at `child_idx`.
   - **Split (1-to-N):** Remove entry at `child_idx`, insert N new entries.
   - **Merge (M-to-N):** Remove M entries (we consumed M original siblings
     during rechunking), insert N new entries.
4. **If child count changed:** rechunk the branch entries using the same
   walk-forward logic. This may cause the branch itself to split/merge.
5. **If child count unchanged:** write the new branch node to CAS, cascade the
   new CID upward.
6. Pop next `PathLevel`, repeat.
7. At the top: if single node remains, its CID is the new root. If rechunking
   produced multiple nodes, wrap in a new branch (tree grew taller).

### Tree height changes

- **Grows:** Root branch splits → create new root wrapping the split nodes.
- **Shrinks:** Root branch reduced to one child → child becomes new root.

### update_meta fast path

Since the key doesn't change, boundaries are guaranteed unchanged:

1. Walk to leaf.
2. Update entry metadata in the leaf.
3. Write new leaf to CAS → new CID.
4. At each level: replace child CID at `child_idx` (boundary_key unchanged,
   child count unchanged). Write new branch to CAS → new CID.
5. Repeat to root.

No rechunking at any level. Strict O(log n) with exactly one node write per
tree level.

**Implementation note:** The actual implementation uses the same rechunk path
for update_meta rather than the CID-only cascade described above. This is more
correct because changing a snippet can change `LeafEntry::approx_size()`, which
is an input to `is_boundary()`. A large snippet change could legitimately shift
chunk boundaries. Using the full rechunk path handles this correctly with
negligible overhead (the common case where boundaries don't shift converges
immediately).

## Changes to Existing Code

### `prolly/mod.rs`

- `insert()`, `remove()`, `update_meta()` call functions from `mutate.rs`
  instead of `rebuild_tree()`
- `rebuild_tree()` becomes `#[cfg(test)]` only — kept as reference for the
  `history_independence` test
- `build_tree()` stays `pub(crate)` — still used by tests as correctness oracle
- All read operations unchanged
- Cache update logic unchanged
- Rollback logic unchanged (snapshot cache/root, restore on error)

### `prolly/chunker.rs`

No changes needed. The rechunking loop in `mutate.rs` calls `is_boundary()`
directly rather than going through `chunk_items()` — the streaming
entry-by-entry boundary check is straightforward.

### No changes to

- `node.rs` — `Node::read_from_cas` and `Node::write_to_cas` reused as-is
- `diff.rs` — completely independent of mutation strategy
- `db.rs` — delegates to ProllyTree, which preserves the same interface
- `persist.rs`, `types.rs`, `error.rs`, `lib.rs` — untouched
- `examples/mail_workflow.rs` — uses only public API

## Invariants Preserved

1. **History independence:** Same entries → same root CID regardless of mutation
   order. Guaranteed by the walk-forward rechunking producing identical chunk
   boundaries to a full rebuild.
2. **Cache consistency:** Cache is updated before tree mutation (same as today).
   If tree mutation fails, cache is rolled back.
3. **CAS integrity:** All nodes written via `Node::write_to_cas` with BLAKE3 CID
   verification. All nodes read via `Node::read_from_cas` with CID verification.
4. **Rollback on failure:** Snapshot cache/root before mutation, restore on any
   error during the incremental operation.

## Testing Strategy

### Equivalence tests (incremental vs. full rebuild)

| Test | Validates |
|------|-----------|
| `incremental_insert_matches_rebuild` | Insert 1 entry into 200-entry tree, compare root with full rebuild of 201 entries |
| `incremental_remove_matches_rebuild` | Remove 1 entry from 200-entry tree, compare root with full rebuild of 199 entries |
| `incremental_update_meta_matches_rebuild` | Update metadata on 1 entry, compare root with full rebuild |
| `incremental_sequence_matches_rebuild` | Insert 100 entries one at a time incrementally, compare final root with full rebuild |
| `incremental_mixed_ops_matches_rebuild` | Mix of inserts, removes, and updates, compare final root with full rebuild |

### History independence

| Test | Validates |
|------|-----------|
| `history_independence` | Existing test — exercises incremental path now |
| `incremental_order_independence` | Insert entries in random order incrementally, compare root with sorted full rebuild |

### Edge cases

| Test | Validates |
|------|-----------|
| `insert_into_empty_tree` | First entry creates a single leaf |
| `remove_last_entry` | Removing the only entry produces None root |
| `insert_causes_leaf_split` | Insert into near-max leaf triggers split |
| `remove_causes_leaf_merge` | Remove from near-min leaf, entries absorbed into neighbor |
| `insert_causes_boundary_ripple` | Insert shifts boundaries into next sibling — convergence works |
| `cascade_causes_branch_split` | Enough leaf splits to overflow a branch node |
| `tree_height_grows` | Insert enough entries that tree grows a level |
| `tree_height_shrinks` | Remove enough entries that tree loses a level |

### Performance validation

| Test | Validates |
|------|-----------|
| `incremental_insert_node_count` | Insert 1 entry into 1000-entry tree, CAS writes ~4-5 (not ~50+) |

### Regression

All existing 56 tests pass unchanged. The `mail_workflow` example passes
unchanged.

## Performance Expectations

| Operation | Current (Phase 2) | After (Phase 3) | Notes |
|-----------|-------------------|------------------|-------|
| Insert | O(n) | O(log n) expected | ~4-5 node writes for 4-level tree |
| Remove | O(n) | O(log n) expected | Same as insert |
| Update meta | O(n) | O(log n) strict | CID cascade only, no rechunking |
| Get/Range | O(1) / O(log n + k) | Unchanged | Still served from cache |
| N sequential inserts | O(N * M) | O(N log M) | M = total entries |
| 1000 inserts into 10K entries | ~10M entry processings | ~13K node writes | ~770x improvement |

## Dependencies

No new crate dependencies. All functionality builds on existing:
- `chunker::ChunkerConfig::is_boundary` — boundary decisions
- `node::Node::read_from_cas` / `write_to_cas` — CAS I/O
- `node::LeafEntry`, `node::BranchEntry` — node entry types
