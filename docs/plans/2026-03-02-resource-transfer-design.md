# Resource Transfer Design

**Goal:** Implement chunked data transfer over established links, with adaptive windowing, retransmission, cancellation, and proof-of-delivery. Wire-compatible with Python Reticulum.

**Parent bead:** harmony-aaf (Implement Reticulum resource transfer)

## Architecture

Standalone `ResourceSender` and `ResourceReceiver` sans-I/O state machines in a new `resource.rs` module within `harmony-reticulum`. Both follow the event/action pattern used by `Node` and `LivelinessRouter`. A `LinkCrypto` trait abstracts encrypt/decrypt/mtu, allowing the real `Link` in production and a `MockLinkCrypto` in tests.

Two-layer encryption model (matching Python):
1. **Payload encryption:** Sender encrypts entire data blob via `LinkCrypto` before chunking. Receiver reassembles all parts, then decrypts as one unit.
2. **Per-packet encryption:** Caller encrypts/decrypts each wire packet via `LinkCrypto` (separate from resource state machines).

No internal timers or I/O. The caller drives all timing via explicit `Timeout` events and handles all encrypt/decrypt calls.

## Types and Constants

- `ResourceHash = [u8; 16]` — `SHA-256(encrypted_data + random_hash)[:16]`
- `MapHash = [u8; 4]` — `SHA-256(part_data + random_hash)[:4]`
- `RandomHash = [u8; 4]` — 4-byte random nonce per resource

Constants (from Python):

| Name | Value | Purpose |
|---|---|---|
| `MAPHASH_LEN` | 4 | Map hash truncation length |
| `MAX_EFFICIENT_SIZE` | 0xFFFFFF (~1MB) | Max resource data size |
| `MAX_RETRIES` | 16 | Receiver retry limit |
| `MAX_ADV_RETRIES` | 4 | Advertisement retry limit |
| `WINDOW_INITIAL` | 4 | Starting request window |
| `WINDOW_MIN` | 2 | Minimum window |
| `WINDOW_MAX_SLOW` | 10 | Default ceiling |
| `WINDOW_MAX_FAST` | 75 | Ceiling for fast links |
| `WINDOW_MAX_VERY_SLOW` | 4 | Ceiling for very slow links |
| `RATE_FAST` | 6250 bytes/sec | 50 kbps threshold |
| `RATE_VERY_SLOW` | 250 bytes/sec | 2 kbps threshold |
| `SENDER_GRACE_TIME` | 10s | Sender timeout padding |
| `PROCESSING_GRACE` | 1s | Advertisement timeout padding |
| `PROOF_TIMEOUT_FACTOR` | 3 | Proof timeout multiplier |
| `PART_TIMEOUT_FACTOR` | 4 | Initial part timeout multiplier |
| `PART_TIMEOUT_FACTOR_AFTER_RTT` | 2 | Part timeout after first measurement |

## LinkCrypto Trait

```rust
pub trait LinkCrypto {
    fn encrypt(&self, rng: &mut dyn CryptoRngCore, plaintext: &[u8]) -> Result<Vec<u8>, ReticulumError>;
    fn decrypt(&self, ciphertext: &[u8]) -> Result<Vec<u8>, ReticulumError>;
    fn link_id(&self) -> &[u8; 16];
    fn mdu(&self) -> usize;
}
```

`Link` implements `LinkCrypto`. Tests use `MockLinkCrypto` (identity pass-through or fixed-key).

## State Machines

### ResourceSender

```
Queued → Advertised → Transferring → AwaitingProof → Complete
                         ↓                ↓
                       Failed           Failed
              ↓ (rejected)
           Rejected
```

### ResourceReceiver

```
(accept) → Transferring → Assembling → Complete
               ↓               ↓
             Failed          Corrupt
```

## Events (inbound)

| Event | Target | Trigger |
|---|---|---|
| `AdvertisementReceived { plaintext }` | Receiver | Decrypted ResourceAdv packet |
| `RequestReceived { plaintext }` | Sender | Decrypted ResourceReq packet |
| `PartReceived { plaintext }` | Receiver | Decrypted Resource data packet |
| `HashmapUpdateReceived { plaintext }` | Receiver | Decrypted ResourceHmu packet |
| `ProofReceived { plaintext }` | Sender | Decrypted ResourcePrf packet |
| `CancelReceived` | Either | Peer sent ResourceIcl/ResourceRcl |
| `Timeout { now_ms }` | Either | Caller fires scheduled deadline |

## Actions (outbound)

| Action | Purpose |
|---|---|
| `SendPacket { context, plaintext }` | Caller encrypts and sends on wire |
| `StateChanged { new_state }` | Notify caller of transition |
| `Progress { fraction: f32 }` | Transfer progress (0.0-1.0) |
| `Completed { data }` | Reassembled + decrypted data |
| `ScheduleTimeout { deadline_ms }` | Caller should fire Timeout at this time |

## Public API

### ResourceSender

- `new(rng, crypto, data, now_ms) -> Result<Self, ReticulumError>` — Encrypt data, chunk, compute hashes. Returns Queued state.
- `advertise(now_ms) -> Vec<ResourceAction>` — Emit advertisement + schedule timeout.
- `handle_event(event) -> Vec<ResourceAction>` — Drive state machine.
- `cancel() -> Vec<ResourceAction>` — Emit ResourceIcl, move to Failed.
- `state() -> ResourceState`
- `hash() -> &ResourceHash`

### ResourceReceiver

- `accept(adv_plaintext, now_ms) -> Result<(Self, Vec<ResourceAction>), ReticulumError>` — Parse advertisement, allocate slots, emit initial ResourceReq.
- `handle_event(event) -> Vec<ResourceAction>` — Drive state machine.
- `cancel() -> Vec<ResourceAction>` — Emit ResourceRcl, move to Failed.
- `state() -> ResourceState`
- `hash() -> &ResourceHash`

## Adaptive Windowing

Window state (inside ResourceReceiver):

```
window: usize               // Current request size (starts 4)
window_min: usize            // Floor (starts 2)
window_max: usize            // Ceiling (starts 10, adapts)
outstanding_parts: usize     // Parts requested but not received
consecutive_completed: usize // Highest contiguous received index
retries_left: u8             // Starts MAX_RETRIES (16)
fast_rate_rounds: usize      // Consecutive fast-rate measurements
very_slow_rate_rounds: usize // Consecutive slow-rate measurements
```

Three adaptation rules:
1. **Growth:** Window filled (outstanding == 0) → `window += 1` up to `window_max`. Grow `window_min` to maintain flexibility gap.
2. **Rate detection:** Measure request-response RTT rate. >50kbps for 5 rounds → `window_max = 75`. <2kbps for 2 rounds → `window_max = 4`.
3. **Backoff:** Timeout → `window -= 1`, `window_max -= 1`, `retries_left -= 1`. Re-request. `retries_left == 0` → Failed.

## Retransmission

Three timeout regimes (all sans-I/O via ScheduleTimeout/Timeout events):

1. **Advertisement:** `adv_sent + rtt_estimate + 1s`. Up to 4 retries.
2. **Transfer (sender):** `last_activity + rtt * factor * max_retries + 10s`. Cancel if exceeded.
3. **Proof:** `last_part_sent + rtt * 3 + 10s`. Retry by re-requesting.

## Cancellation

- Sender cancel: emit `SendPacket { ResourceIcl, resource_hash }` → Failed.
- Receiver cancel: emit `SendPacket { ResourceRcl, resource_hash }` → Failed.
- Receiving cancel: move to Failed/Rejected.

## Wire Format

### ResourceAdvertisement (msgpack)

```
{
    "t": transfer_size (u32),
    "d": data_size (u32),
    "n": part_count (u32),
    "h": resource_hash ([u8; 16]),
    "r": random_hash ([u8; 4]),
    "o": original_hash ([u8; 16]),
    "m": hashmap_bytes (concatenated 4-byte map_hashes),
    "f": flags (u8, bit 0 = encrypted),
    "i": 1 (segment_index, always 1),
    "l": 1 (total_segments, always 1),
}
```

### ResourceReq

```
[hashmap_status_flag: u8][resource_hash: 16 bytes][requested_map_hashes: N*4 bytes]
(if flag == 0xFF, insert last_map_hash after flag: [flag][last_map_hash: 4][resource_hash][...])
```

### ResourceHmu

```
[resource_hash: 16 bytes][msgpack([segment_index, hashmap_bytes])]
```

### Part hashing

```
map_hash = SHA-256(part_data + random_hash)[:4]
resource_hash = SHA-256(encrypted_data + random_hash)[:16]
proof = SHA-256(original_data + resource_hash)[:16]
```

## Error Handling

New `ReticulumError` variants:

- `ResourceTooLarge { size, max }` — Data exceeds MAX_EFFICIENT_SIZE
- `ResourceAdvInvalid` — Malformed advertisement msgpack
- `ResourceHashMismatch` — Reassembled data hash != expected
- `ResourceProofInvalid` — Proof doesn't match expected
- `ResourceUnknownPart { map_hash }` — Part doesn't match expected hash
- `ResourceFailed` — General transfer failure
- `ResourceAlreadyComplete` — Event on completed resource

## Testing

### Rust unit tests (~25-30)
- Full transfer round-trip (small, medium, near-MTU, multi-part)
- Windowing adaptation (growth, fast detection, slow detection)
- Timeout/retry (advertisement, transfer, proof)
- Cancellation (sender, receiver, both sides)
- Advertisement rejection
- Corrupt data detection (hash mismatch)
- Proof validation (valid + invalid)
- Hashmap exhaustion + HMU flow
- Edge cases (empty data, exact-SDU-boundary, single-part)

### Python interop tests (~5-6)
- Advertisement msgpack encoding roundtrip
- Map_hash computation: SHA-256(part + random_hash)[:4]
- Resource hash computation: SHA-256(encrypted_data + random_hash)[:16]
- Proof computation: SHA-256(data + resource_hash)[:16]
- RESOURCE_REQ packet format (flag byte + map_hashes)

## Not Building (YAGNI)

- Compression (bz2)
- Multi-segment splitting (>1MB)
- Request/response RPC pattern (request_id)
- Metadata embedding
- Cache request for proofs (network-level optimization)
- Progress callbacks with ETA estimation

## Integration

New file `resource.rs`, new error variants, `LinkCrypto` trait (impl on `Link`), new lib.rs exports. Existing `Link`, `Node`, `PacketContext` code unchanged — resource contexts already defined.
