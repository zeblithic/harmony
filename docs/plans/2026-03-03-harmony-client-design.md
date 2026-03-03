# Harmony Client Design

**Goal:** A desktop application that lets users participate in the Harmony decentralized network with a Discord-familiar interface — identity management, peer-to-peer encrypted messaging, presence indicators, and full-duplex voice calls over both internet tunnels and raw Reticulum links.

**Scope:** MVP covers 1:1 messaging, 1:1 voice, presence, and network bootstrapping via coordination points. No group features, file sharing, or video.

---

## Repository & Dependencies

**Repository:** `zeblithic/harmony-client` — separate from `zeblithic/harmony`

The core harmony crates (`harmony-crypto`, `harmony-identity`, `harmony-reticulum`, `harmony-zenoh`, `harmony-content`) are consumed as git dependencies. This keeps the core library free of async runtimes, UI frameworks, and audio codecs. The core must run on embedded devices and routers; the client is where heavyweight dependencies live.

**Tech stack:**
- **Backend:** Rust, tokio, cpal (audio), opus/codec2
- **Frontend:** Tauri v2, Svelte 5 with runes, Vite
- **Key storage:** keyring crate (OS keychain) + ChaCha20-Poly1305 encrypted file fallback

---

## Architecture: Layered Daemon + UI

```
┌─────────────────────────────────┐
│  Svelte 5 + Runes UI (Tauri)   │  ← WebView, renders state
├─────────────────────────────────┤
│  Tauri Commands (Rust ↔ JS)     │  ← IPC bridge
├─────────────────────────────────┤
│  harmony-daemon (tokio runtime) │  ← Drives all harmony state machines
│  ├─ Identity Manager            │     Keypair persistence + OS keychain
│  ├─ Transport Manager           │     TCP/UDP/WS Interface impls
│  ├─ Protocol Engine             │     Reticulum node + Zenoh session
│  ├─ Voice Engine                │     Opus + codec2, jitter buffer
│  ├─ Local API Server            │     HTTP REST + WS + Zenoh endpoint
│  └─ Coordination Client         │     Connects to bridge nodes
└─────────────────────────────────┘
```

The daemon is a **library crate** (`harmony-daemon`) with no UI dependencies. The Tauri app (`harmony-app`) is one consumer. This means:
- The daemon is independently testable
- A headless `harmonyd` binary can reuse it for servers/routers
- The `harmony-coordination` binary reuses it for bridge nodes

The daemon runs in-process with Tauri for the MVP (not a separate process), but the code boundary enables splitting later.

---

## Crate Structure

```
harmony-client/
├── Cargo.toml                          # Workspace root
├── crates/
│   ├── harmony-daemon/                 # Library: protocol engine, transport, voice
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── engine.rs               # Tokio event loop driving harmony state machines
│   │   │   ├── transport/
│   │   │   │   ├── mod.rs
│   │   │   │   ├── tcp.rs              # TCP Interface impl
│   │   │   │   ├── udp.rs              # UDP Interface impl (LAN discovery)
│   │   │   │   └── websocket.rs        # WebSocket Interface impl
│   │   │   ├── identity.rs             # Keypair persistence (keychain + encrypted file)
│   │   │   ├── voice/
│   │   │   │   ├── mod.rs
│   │   │   │   ├── opus.rs             # Opus codec wrapper
│   │   │   │   ├── codec2.rs           # codec2 wrapper (low-bandwidth)
│   │   │   │   ├── jitter.rs           # Adaptive jitter buffer
│   │   │   │   └── negotiate.rs        # Codec/transport negotiation
│   │   │   ├── coordination.rs         # Coordination point client
│   │   │   └── api/
│   │   │       ├── mod.rs
│   │   │       ├── http.rs             # REST endpoints
│   │   │       ├── websocket.rs        # Event stream
│   │   │       └── zenoh.rs            # Zenoh local endpoint
│   │   └── Cargo.toml
│   ├── harmony-app/                    # Binary: Tauri + Svelte UI
│   │   ├── src/
│   │   │   ├── main.rs                 # Tauri bootstrap
│   │   │   └── commands.rs             # Tauri IPC commands
│   │   ├── ui/                         # Svelte 5 frontend
│   │   │   ├── src/
│   │   │   │   ├── lib/
│   │   │   │   │   ├── stores/         # Runes-based reactive state
│   │   │   │   │   ├── components/     # Reusable UI components
│   │   │   │   │   └── views/          # Page-level views
│   │   │   │   ├── app.svelte
│   │   │   │   └── routes/
│   │   │   ├── package.json
│   │   │   └── vite.config.ts
│   │   └── Cargo.toml
│   └── harmony-coordination/           # Binary: bridge node server
│       ├── src/
│       │   └── main.rs
│       └── Cargo.toml
├── tauri.conf.json
└── docs/
```

---

## Protocol Engine

The engine drives harmony's sans-I/O state machines in a tokio event loop.

```rust
pub struct Engine {
    identity: PrivateIdentity,
    node: harmony_reticulum::Node,
    sessions: HashMap<PeerId, harmony_zenoh::Session>,
    pubsub: harmony_zenoh::PubSubRouter,
    liveliness: harmony_zenoh::LivelinessRouter,
    transports: TransportManager,
    voice_engine: VoiceEngine,
}
```

**Event loop pattern:**

```rust
loop {
    tokio::select! {
        bytes = transport.recv()    => engine.handle_packet(bytes),
        timer = tick.tick()         => engine.handle_timer(now),
        cmd = tauri_rx.recv()       => engine.handle_ui_command(cmd),
        cmd = api_rx.recv()         => engine.handle_api_command(cmd),
        audio = mic.recv()          => engine.handle_audio_input(audio),
    }
}
```

Every call into harmony's state machines returns `Vec<Action>` which the engine dispatches as real I/O (send packet, update UI, play audio).

**Transport Manager** implements `harmony_reticulum::Interface` for each transport:
- **TCPInterface** — connects to coordination points and direct peers
- **UDPInterface** — local network discovery (broadcast/multicast on all NICs)
- **WebSocketInterface** — browser-friendly tunnel through coordination points
- **AutoInterface** — scans local network interfaces, spawns UDP listeners

---

## Voice Engine

Two codec paths, negotiated per-call based on shared transport.

| Transport | Codec | Bitrate | Quality | Latency |
|-----------|-------|---------|---------|---------|
| Internet tunnel (TCP/UDP) | Opus | 16-64 kbps | Near-Discord | ~20ms |
| Reticulum direct (500B MTU) | codec2 | 700-3200 bps | Ham radio | ~40-80ms |

**Call setup:**
1. Caller sends voice invite via Zenoh pub/sub (control message)
2. Peers exchange transport capabilities + supported codecs
3. Best shared path selected (prefer internet tunnel if both have one)
4. Dedicated Reticulum link established for audio stream
5. Audio pipeline starts

**Audio pipeline (per direction):**

```
Mic → Resample 48kHz → Noise gate → Encode (Opus|codec2)
    → Encrypt (ChaCha20-Poly1305) → Fragment to MTU → Send

Recv → Reassemble → Decrypt → Jitter buffer (adaptive, 60-150ms)
    → Decode → Resample → Speaker
```

**Key behaviors:**
- Jitter buffer adapts: wider for Reticulum (higher jitter), tighter for UDP tunnels
- Mid-call codec switching if a better transport appears (e.g., peer joins same LAN)
- Opus at 16kbps for internet; codec2 at 1200bps fits ~2-3 frames per Reticulum packet
- Audio via `cpal` crate (macOS CoreAudio, Windows WASAPI, Linux ALSA/PulseAudio)
- Each audio frame encrypted with incrementing nonce (16 bytes auth tag overhead)

**Not in MVP voice:** group calls, video, echo cancellation (rely on OS/hardware AEC), recording.

---

## Coordination Points (Bridge Nodes)

Full harmony nodes with internet-facing transports, bridging traditional internet to the Harmony network. Anyone can run one.

```
┌─────────────────────────────────────────┐
│  harmony-coordination                    │
│  ├─ Full Reticulum node                  │  ← Participates in routing
│  ├─ Full Zenoh router                    │  ← Relays pub/sub
│  ├─ Content cache (W-TinyLFU)            │  ← Caches hot content
│  ├─ TCP listener (:7891)                 │  ← Client connections
│  ├─ WebSocket listener (:7892)           │  ← Browser/tunneled clients
│  └─ HTTPS bootstrap endpoint (:443)      │  ← GET /peers → known peers
└─────────────────────────────────────────┘
```

**Client bootstrap:**
1. Fresh client knows `harmony.zeblith.net` (default, user-configurable)
2. HTTPS GET → list of active peer addresses + transports
3. TCP connect to coordination point (now a Reticulum interface)
4. Receives announces from the network through the bridge
5. Simultaneously scans local network via UDP broadcast for LAN peers
6. Learns direct paths over time, can bypass coordination points

**Relay:** When peers can't connect directly (NAT, different networks), traffic routes through coordination points via standard Reticulum multi-hop routing. No special relay protocol — the coordination point's `Interface` wraps its TCP/WS connections like any other transport.

**Deployment:** Minimal config (port + optional domain). Regular userspace process, no special privileges.

---

## Local API Server

Two interfaces for external apps to interact with the harmony node.

### HTTP REST + WebSocket (localhost:7891)

```
REST:
  GET  /v1/identity              → user's public identity
  GET  /v1/peers                 → known peers + online status
  POST /v1/messages/send         → send message to peer
  GET  /v1/messages?peer=<addr>  → message history
  POST /v1/voice/call            → initiate call
  POST /v1/voice/hangup          → end call

WebSocket (ws://localhost:7891/v1/events):
  Events: peer_online, peer_offline, message_received, message_sent,
          call_incoming, call_connected, call_ended,
          announce_received, link_established
```

The Svelte UI connects to this same WebSocket for reactive updates. Tauri IPC commands call directly into the daemon (bypassing HTTP) for lower latency.

### Zenoh Endpoint (localhost:7892)

```
Key space:
  harmony/messages/<peer_addr>/**     → message topics
  harmony/presence/<peer_addr>        → liveliness tokens
  harmony/voice/<call_id>/**          → voice signaling
  harmony/announce/**                 → network announces
```

Zenoh-aware apps subscribe directly for lower latency. REST/WS is the universal path ("send a message from Python"); Zenoh is the harmony-native path.

---

## Svelte 5 UI

Thin reactive view layer over daemon state. Four views for MVP.

### Views

1. **Identity / Onboarding** — First-run: generate or import keypair, set display name, configure coordination points. Returning: unlock with passphrase.

2. **Peers & Presence** — Contact list with online/offline (Zenoh liveliness). Add by address or QR code. Transport quality indicator: green = direct LAN, yellow = tunneled, orange = Reticulum-only.

3. **Messaging** — Discord-familiar conversation layout. Delivery status (sent → delivered → read). E2E encrypted.

4. **Voice Call** — Call screen with active codec + transport indicator. Mute/unmute. Quality meter.

### State Management

```svelte
// Runes-based reactive state
let peers = $state<Peer[]>([]);
let activePeer = $state<Peer | null>(null);

function handlePeerOnline(addr: string) {
    const peer = peers.find(p => p.address === addr);
    if (peer) peer.online = true;
}
```

WebSocket events push into rune state. No polling, no state management library.

### Design Language

- Dark theme default (Discord-familiar)
- Compact sidebar for peers, main area for conversation
- Content-first, minimal chrome
- Transport quality via color coding
- Tauri WebView keeps memory ~50MB (vs Electron ~200MB+)

**Not in MVP UI:** group chats, file sharing UI, settings beyond identity/coordination, message search, themes.

---

## Identity & Security

### Key Storage

```
macOS   → Keychain Services (via security-framework crate)
Linux   → Secret Service D-Bus API (via keyring crate)
Windows → Credential Manager (via keyring crate)
Fallback → Argon2id(passphrase) → ChaCha20-Poly1305 encrypted file
           (~/.harmony/identity.enc)
```

On first run, generate keypair (or import from `harmony identity new`). 128-byte private key stored in OS keychain, tagged with app identifier.

### Message Encryption

```
ECDH(my_x25519, their_x25519) → shared_secret (32 bytes)
  → HKDF(shared_secret, salt) → message_key
  → ChaCha20-Poly1305(message_key, plaintext) → ciphertext
```

Uses harmony-identity ECDH + harmony-crypto ChaCha20-Poly1305. No new crypto.

### Voice Encryption

Same ECDH key exchange at call setup, separate derived key for audio stream. Each frame gets incrementing nonce. 16 bytes auth tag overhead per frame.

### Trust Model

- Trust-on-first-use (TOFU) — first encounter with a peer's public key is trusted
- Key fingerprint displayed in peer details for out-of-band verification
- No certificate authorities, no web of trust (YAGNI for now)

---

## Spartacus Key

A hardcoded, intentionally-compromised keypair embedded in the client for anonymous/testing use.

**Purpose:**
- Zero-friction onboarding: "just click Connect" without generating a keypair
- Testing and development without managing real identities
- Observing network behavior under a massively degenerate case (millions of users sharing one identity)

**Implementation:**
- Well-known keypair compiled into the binary (both public and private keys)
- UI offers "Connect as Spartacus" alongside "Create Identity" on first run
- Messages sent under the spartacus key are clearly labeled in the UI (distinct color/badge)
- The key is publicly documented so everyone knows it's compromised

**Network considerations:**
- Production coordination points SHOULD blacklist the spartacus key for stability (configurable)
- Development/testing coordination points allow it
- The spartacus key demonstrates that the network handles degenerate identity gracefully
- Multiple simultaneous spartacus users create interesting routing/announce collisions — useful for stress testing and protocol hardening

**What spartacus users CAN do:** send/receive messages (unencrypted between spartacus users since everyone shares the key), participate in presence, make voice calls.

**What they CANNOT expect:** privacy (shared key = no secrets), message delivery guarantees (routing collisions), persistence across sessions.

---

## Data Flow: Core User Stories

### "Sign in" (launch, join network)

1. Launch harmony-app → Tauri boots harmony-daemon
2. Load identity from OS keychain (or passphrase prompt, or spartacus)
3. TransportManager starts: UDP broadcast on all NICs + TCP to coordination points
4. Reticulum Node announces our identity on all interfaces
5. Zenoh liveliness token declared (we're online)
6. Peer announces arrive → path table populates
7. UI shows peer list with presence
8. Local API starts on :7891 (HTTP/WS) and :7892 (Zenoh)

### "Send message"

1. User types message → Tauri IPC → daemon.send_message(peer, text)
2. Look up peer's public key from received announces
3. ECDH → HKDF → ChaCha20-Poly1305 encrypt
4. Zenoh publish to `harmony/messages/<peer_addr>/inbox`
5. Routed via best path: direct LAN > coordination tunnel > multi-hop
6. Recipient decrypts → stores locally → WebSocket event → UI updates

### "Voice call"

1. Click "call" → daemon sends invite via Zenoh
2. Recipient sees incoming call → accepts
3. Exchange transport capabilities → select best codec
4. Dedicated Reticulum link established
5. Audio: mic → encode → encrypt → send / recv → decrypt → jitter → decode → speaker
6. Hangup → link torn down → UI updates

---

## Non-Goals (Future Work)

- Group messaging / group calls
- File sharing UI (protocol supports it via resource transfer)
- Video calls
- Message search / history sync across devices
- Themes / UI customization
- SCION data plane integration
- Web browser client (WebAssembly + WebRTC)
- Mobile apps (iOS/Android)
- Plugin/extension system
