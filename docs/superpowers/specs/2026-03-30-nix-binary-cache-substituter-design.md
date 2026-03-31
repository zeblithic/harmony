# Nix Binary Cache Substituter

**Beads:** harmony-llaf (substituter), harmony-885 (NAR storage)
**Date:** 2026-03-30
**Status:** Draft

## Problem

The Harmony mesh includes multiple NixOS machines (RPi5 fleet: luna, terra,
sol, archive) and several build workstations (Mac, WSL AVALON, WSL KRILE).
When any node rebuilds NixOS or deploys a package, it fetches from
`cache.nixos.org` over the internet — even if another machine on the mesh
already built the same derivation. There is no local Nix binary cache, so
every machine redundantly downloads or rebuilds the same closures.

harmony-node already has a content-addressed storage tier that chunks, stores,
and reassembles arbitrary blobs. Making it speak the Nix binary cache HTTP
protocol turns every harmony-node into a local substituter — build once on
any workstation, fetch from the mesh everywhere.

## Solution

Add a Nix binary cache substituter to harmony-node with two capabilities:

1. **Ingest** (`harmony nar push`): Takes a Nix store path (or full closure),
   dumps the NAR, ingests it into the CAS via `dag::ingest`, builds narinfo
   metadata, signs it with a standard Nix signing key, and stores the narinfo
   as a Book with a memo mapping store hash to narinfo CID.

2. **Serve** (axum HTTP server): Three endpoints implementing the Nix binary
   cache protocol — `/nix-cache-info`, `/<hash>.narinfo`, `/nar/<cid>.nar`.
   Any Nix client configured with the node's address as a substituter can
   fetch packages from the CAS.

Any harmony-node can be both a producer (push NARs) and a server (serve the
HTTP cache). The Mac workstation, AVALON, and KRILE push builds; the RPi5
fleet and any other machine consume from the local cache.

## Architecture

### Data Model

**NAR storage:** A NAR file is a blob. `nix-store --dump /nix/store/<hash>-<name>`
produces it, `dag::ingest()` chunks it into Books and bundles them into a DAG,
returning a root CID. No NAR-specific storage format — the CAS handles it like
any other content.

**narinfo as Book:** The narinfo text (StorePath, NarHash, NarSize, References,
Sig, URL) is stored as a Book in the CAS. The `URL` field points to
`nar/<nar_root_cid_hex>.nar` — using the CAS CID as the NAR filename.

**Memo mapping:** A memo links `input CID → output CID` where:
- Input CID = `ContentId::for_book(store_hash_str.as_bytes())` — the UTF-8 bytes
  of the 32-char Nix store hash string (the base32 part before the `-name` in
  `/nix/store/<hash>-<name>`, e.g., `"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"`)
- Output CID = CID of the narinfo Book

This lets the HTTP handler look up any store hash in O(1) via the MemoStore.

### Ingestion Pipeline

`harmony nar push [--closure] /nix/store/<hash>-<name>`

**Single store path mode** (default):
1. Run `nix-store --dump <store-path>` to produce raw NAR bytes
2. Ingest NAR blob via `dag::ingest()` → `nar_root_cid`
3. Compute NarHash: SHA-256 of raw NAR bytes (Nix format: `sha256:<base64>`)
4. Gather references: `nix-store -q --references <store-path>`
5. Build narinfo text with all required fields
6. Sign narinfo fingerprint with Nix signing key (ed25519, `<keyname>:<base64sig>`)
7. Store narinfo text as Book → `narinfo_cid`
8. Create memo: input = CID from store hash, output = `narinfo_cid`
9. Persist Books via `disk_io`, memo via `memo_io`

**Closure mode** (`--closure`):
1. Run `nix-store -qR <store-path>` to enumerate full closure
2. Process each path through the single-path pipeline above
3. Skip paths where a memo already exists (idempotent)
4. Report progress: `[12/47] /nix/store/abc...-glibc-2.39`

**Pipe mode:** `nix-store --dump /nix/store/... | harmony nar push --stdin --store-path <hash>-<name>`
for environments where the store path isn't locally accessible.

### HTTP Server (axum)

Spawned when `--nix-cache-port <port>` is set (or `[nix-cache] port` in config).
Runs on its own tokio tasks alongside the existing event loop.

**`GET /nix-cache-info`** — Static response:
```
StoreDir: /nix/store
WantMassQuery: 1
Priority: 30
```
Priority 30 ranks above `cache.nixos.org` (40), so Nix tries the local cache
first. (Lower number = higher priority in Nix's substituter ordering.)

**`GET /<store-hash>.narinfo`** — Lookup:
1. Parse 32-char nix-base32 store hash from URL path
2. Compute input CID from store hash string bytes
3. Look up memo by input CID via `MemoStore::peek_by_input` (no LFU inflation)
4. Read narinfo Book from BookStore (memory hit) or disk (fallback via `disk_io::read_book`)
5. Return narinfo text with `Content-Type: text/x-nix-narinfo`
6. Return 404 if no memo found

**`GET /nar/<nar-root-cid-hex>.nar`** — Streaming reassembly:
1. Parse root CID from URL
2. Walk DAG via `dag::walk()`, collect leaf Book CIDs
3. Stream leaf Book data as HTTP response chunks via `Body::from_stream`
4. `Content-Type: application/x-nix-nar`
5. Return 404 if root CID not found

**Shared state:** axum handlers receive `Arc<BookStore>` and `Arc<MemoStore>` for
read access. The event loop owns mutable state; the HTTP layer is read-only.
No locking needed for the common serving path.

### Signing

Nix-standard ed25519 signing keys, same format used by `nix-serve` and Cachix.
Each node has its own keypair:

- Private key: path configured via `--nix-cache-signing-key` or config file
- Public key: distributed to clients (committed to repo or served via `/nix-cache-info` extension)
- Key name format: `<hostname>-1` (e.g., `harmony-luna-1`)

The narinfo fingerprint format follows the Nix specification:
`1;<StorePath>;<NarHash>;<NarSize>;<sorted References>`.

Signing uses the standard ed25519 algorithm. The `harmony-identity` crate already
has Ed25519 support, so no new crypto dependency is needed — just the Nix-specific
fingerprint formatting and base64 encoding of the signature.

### CLI Interface

```
harmony nar push [--closure] <store-path>
    --signing-key <path>     # Nix binary cache signing key (required)
    --data-dir <path>        # CAS data directory (from config if omitted)

harmony nar push --stdin --store-path <hash>-<name>
    --signing-key <path>
    --data-dir <path>
```

### Configuration

New fields in harmony-node's TOML config:

```toml
[nix-cache]
port = 5000    # 0 or absent = HTTP server disabled
```

The signing key is only needed for `harmony nar push` (ingestion), not for
serving. The HTTP server serves pre-signed narinfo Books — no key required.

CLI flag `--nix-cache-port` overrides the config value.

### Feature Flag

The `nix-cache` feature in `harmony-node/Cargo.toml` gates all Nix cache code:

```toml
[features]
nix-cache = ["dep:axum", "dep:tower-http"]
```

Nodes that don't need Nix cache serving (embedded targets, unikernel) don't
compile axum. The feature is enabled in the NixOS RPi5 build profile and
development builds.

## File Changes

### New Files

| File | Purpose |
|------|---------|
| `crates/harmony-node/src/narinfo.rs` | narinfo format: build text, compute fingerprint, sign, parse. Pure data types, no I/O |
| `crates/harmony-node/src/nar.rs` | NAR push pipeline: closure enumeration, NAR dump, CAS ingest, narinfo creation, memo persistence |
| `crates/harmony-node/src/nix_cache.rs` | axum HTTP server: router, `/nix-cache-info`, `/<hash>.narinfo`, `/nar/<cid>.nar` handlers |

### Modified Files

| File | Change |
|------|--------|
| `crates/harmony-node/src/main.rs` | Add `Nar` subcommand group (`push`), spawn HTTP server in `run` when port configured |
| `crates/harmony-node/Cargo.toml` | Add `nix-cache` feature with `axum`, `tower-http` deps |
| `nixos/harmony-node-service.nix` | Add `nixCachePort` option |
| `nixos/rpi5-base.nix` | Configure nix-cache port in service config |

### File Responsibilities

- **`narinfo.rs`** — Pure format logic. Testable without I/O, without the network,
  without the CAS. Knows how to build narinfo text, compute Nix fingerprints,
  and produce ed25519 signatures in Nix's `<keyname>:<base64>` format.

- **`nar.rs`** — CLI ingestion orchestrator. Calls `nix-store` subprocesses,
  pipes NAR bytes through `dag::ingest`, creates memos, writes to disk.
  Knows about the Nix store layout and closure enumeration.

- **`nix_cache.rs`** — HTTP serving layer. Knows axum routing and HTTP semantics.
  Reads from BookStore and MemoStore. Doesn't know about ingestion.

## Testing

### Unit Tests (narinfo.rs)

- Build narinfo from known inputs, verify text output matches expected Nix format
- Round-trip: build narinfo text, parse it back, verify all fields match
- Sign narinfo fingerprint with test key, verify signature format (`<keyname>:<base64>`)
- Verify fingerprint computation matches Nix spec: `1;StorePath;NarHash;NarSize;References`

### Unit Tests (nix_cache.rs)

- Construct axum app with pre-populated MemoryBookStore + MemoStore
- `GET /nix-cache-info` returns expected fields and Content-Type
- `GET /<hash>.narinfo` returns 200 with `text/x-nix-narinfo` when memo exists
- `GET /<hash>.narinfo` returns 404 when no memo
- `GET /nar/<cid>.nar` streams correct bytes when Books exist
- `GET /nar/<cid>.nar` returns 404 when CID missing

### Integration Tests (nar.rs)

- Create temp directory with known file content
- Use a pre-built NAR fixture (avoid dependency on `nix-store` in CI)
- Ingest fixture through push pipeline, verify memo created and narinfo stored
- Reassemble NAR from CAS, verify byte-identical to fixture
- Test closure skip logic: push same path twice, second is a no-op

### End-to-End Test (manual, requires Nix)

- Build a trivial derivation, push with `harmony nar push --closure`
- Start harmony-node with `--nix-cache-port`
- Run `nix path-info --store http://localhost:5000 <store-path>` to verify cache hit

## What is NOT in Scope

- **Compression** (follow-up bead): v1 serves `Compression: none`. zstd/xz is a
  natural follow-up once the pipeline works.
- **Pull-through proxy**: No upstream cache proxying. Nodes serve only what they have.
- **Mesh NAR distribution**: No Zenoh/Reticulum content forwarding for NARs. Nodes
  serve from local CAS only. Mesh distribution uses the existing content announcement
  infrastructure and is a future enhancement.
- **Memo-based trust chain**: Nix signing keys handle trust. Harmony memo attestation
  for NARs is a future layer on top.
- **Garbage collection**: No `harmony nar gc` command. Existing CAS eviction (LFU +
  disk quota) handles space management.
- **Multi-output derivations**: `harmony nar push` handles one store path at a time
  (or a closure). Multi-output derivation awareness is not needed.

## Error Handling

| Scenario | Behavior |
|----------|----------|
| `nix-store --dump` fails | Error with stderr from nix-store, exit non-zero |
| `nix-store -qR` fails | Error: "Cannot enumerate closure. Is the store path valid?" |
| Store path not in local store | Error: "Store path not found. Run `nix build` first." |
| Signing key file missing | Error at startup of `nar push` (not needed for serving) |
| Signing key invalid format | Error: "Invalid Nix signing key format" with expected format hint |
| Memo already exists for store hash | Skip silently (idempotent push) |
| BookStore lookup miss on serve | Try disk fallback via `disk_io::read_book`, then 404 |
| MemoStore lookup miss on serve | 404 with empty body |
| Malformed store hash in URL | 400 Bad Request |
| CID parse failure in NAR URL | 400 Bad Request |
