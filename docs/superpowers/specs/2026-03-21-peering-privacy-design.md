# Peering Privacy: Phase 1 Metadata Mitigations

**Date:** 2026-03-21
**Status:** Draft
**Scope:** `harmony-tunnel` (keepalive padding), `harmony-peers` (backoff jitter), `harmony-node` (stochastic dialing)
**Bead:** harmony-d63

## Overview

Harmony's tunnel peer infrastructure leaks metadata that allows network observers to deduce the social graph — who is connected to whom — without decrypting any payload. This design addresses the three lowest-cost, highest-impact mitigations identified in the Gemini research report (Phase 1), all of which are zero-bandwidth, code-only changes.

### Threat Summary

| Observer | What they learn | How |
|----------|----------------|-----|
| LAN/ISP passive | Which peers you connect to | Timing correlation: Zenoh query → QUIC dial within ms |
| Relay operator | Complete social graph | NodeId-to-NodeId routing table |
| Zenoh subscriber | All active NodeIds + IPs | Wildcard subscribe to `harmony/id/discovery/**` |
| ISP passive | Persistent relationships | Matching exponential backoff curves after disruption |
| Any observer | Tunnel existence | Fixed-size keepalive packets at exact 30s intervals |

### What this design addresses

1. **Stochastic dialing** — breaks Zenoh-query → QUIC-dial timing correlation
2. **Keepalive padding** — makes keepalive frames indistinguishable from small data packets
3. **Decoupled backoff** — makes reconnection curves unique and uncorrelatable

### What this design does NOT address (filed as future beads)

- Relay-blinding via MASQUE/OHTTP (harmony-akt, P3)
- Selective discovery opt-in with authenticated namespaces (harmony-0kq, P3)
- ZipPIR for private discovery lookups (harmony-xu2, P4)
- Dynamic rendezvous tunnels via EigenTrust (harmony-lh4, P4)

## Section 1: Stochastic Dialing Delay

### Problem

When an AnnounceRecord arrives with a tunnel hint, the runtime immediately processes it through the PeerManager, which emits `InitiateTunnel`, and the event loop immediately dials the relay. A local observer sees the Zenoh discovery query and the QUIC connection attempt within milliseconds — trivially correlating them to reveal which NodeId the target is contacting.

### Mitigation

Add a Poisson-distributed random delay (mean 2 seconds, range ~0.5-4s) between the PeerManager emitting `InitiateTunnel` and the event loop actually initiating the QUIC connection.

**Implementation:** The event loop holds `InitiateTunnel` actions in a pending queue with a scheduled fire time. Each tick checks whether any pending dials have reached their scheduled time.

**Why Poisson:** Natural network traffic follows Poisson inter-arrival patterns. A uniform random delay creates a detectable "flat plateau" in the timing distribution. Poisson arrivals are indistinguishable from organic network jitter.

**Cost:** 0.5-4 seconds additional latency on initial tunnel establishment only. Reconnections are already delayed by exponential backoff. No bandwidth cost.

## Section 2: Keepalive Padding

### Problem

Keepalive frames are always tag `0x00` with zero-length payload — exactly 5 bytes plaintext, which encrypts to a fixed ciphertext size. These fixed-size packets appear at exact 30-second intervals, creating a trivially identifiable fingerprint for Statistical Disclosure Attacks.

### Mitigation

Two changes:

1. **Random padding:** `Frame::keepalive()` generates a random-length padding payload (0 to 128 bytes, uniform). The tag stays `0x00` — the receiver already ignores keepalive payloads. The padding makes keepalive ciphertexts indistinguishable from small Reticulum packets (19-500 bytes).

2. **Interval jitter:** Instead of exactly 30 seconds, the keepalive interval is randomized per tick to 25-35 seconds (uniform). This breaks the strict periodicity.

**Where:** `crates/harmony-tunnel/src/frame.rs` (padding) and `crates/harmony-tunnel/src/session.rs` (interval jitter).

**Cost:** ~64 bytes average extra per keepalive (every ~30s) ≈ 2 bytes/second. Negligible.

## Section 3: Decoupled Exponential Backoff

### Problem

The PeerManager's exponential backoff is a clean `base * 2^retry_count`, capped at 600s. Every node reconnecting to the same peer after a network disruption produces an identical curve. An ISP-level observer seeing two nodes execute matching backoff patterns toward the same relay trivially correlates them as a persistent peer pair.

### Mitigation

Add a deterministic per-peer jitter factor derived from identity hashes:

```
jitter_seed = BLAKE3(local_identity_hash || peer_identity_hash)[:8]
jitter_factor = 0.75 + (jitter_seed_as_u64 % 512) / 1024.0  // range: 0.75 to ~1.25
interval = (base * 2^retry_count * jitter_factor).min(max_interval)
```

This produces a unique but stable backoff curve for each (local, peer) pair:
- Two nodes reconnecting to the same peer → different curves (different `local_identity_hash`)
- Same node reconnecting to two peers → different curves (different `peer_identity_hash`)
- Same node reconnecting to the same peer → same curve (deterministic, testable)

**Why deterministic (not random per retry):** Stable across retries for test reproducibility. Still uncorrelatable because observers don't know the local identity hash.

**Where:** `crates/harmony-peers/src/manager.rs` — the `probe_interval()` function. Requires passing the local identity hash into PeerManager (currently it has no concept of "self").

**Cost:** Zero bandwidth, zero latency impact. Interval varies by ±25%.
