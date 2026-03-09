# Wylene — Harmony's Fluid Interface Layer

## Summary

Wylene is Harmony's Blue-layer service: a fluid, AI-orchestrated interface that replaces the traditional app/file/folder paradigm with intent-driven interaction. The user expresses what they want and why; Wylene translates that intent into precise, auditable, capability-gated actions across the Harmony stack.

Named after a changeling — someone who adapts fluidly to any situation — Wylene is not a chatbot, not an app launcher, not a search engine. She is the entire user interface: an always-adapting translation layer between human intent and system capability.

## Design Principles

1. **Translation, not control.** Wylene translates human intent into system actions. She doesn't decide what the user wants — she helps the user express it clearly enough for the system to act. The user is always the author; Wylene is the interpreter.

2. **Orchestration, not computation.** Wylene doesn't contain AI — she routes to it. Every inference call, every search query, every content fetch is a Kitri workflow running on the best available trusted node. Wylene is the conductor, not the orchestra.

3. **Consent is granular and revocable.** Every sensor, every awareness level, every compute delegation is a UCAN capability that the user grants and can revoke at any time. No "accept all or use nothing."

4. **Accessible by construction.** The UI is composed from pre-built accessible components, not generated from pixels. Screen reader support, keyboard navigation, and alternative input methods are properties of the component library, not afterthoughts bolted onto generated output.

5. **Graceful degradation.** No internet? Wylene works with local models. Low battery? Step down awareness. No trusted nodes? Run slower locally. The experience degrades in quality, never in availability.

6. **Transparent and auditable.** Every pipeline stage is a Kitri workflow with a content-addressed audit trail. "Why did Wylene do that?" is always answerable by replaying the event log.

7. **The user model is the user's data.** It lives in their content store, encrypted by their keys, synced across their devices. If they leave Harmony, they take their model with them. No lock-in.

## Why Wylene Exists

Every existing AI assistant follows the same pattern: your data goes to a corporation's cloud, their model processes it, results come back. You trade privacy for capability.

Wylene inverts this:

| Traditional Assistant | Wylene |
|---|---|
| Cloud-first, local is degraded fallback | Local-first, mesh is capability amplifier |
| Vendor owns your interaction history | You own your user model (content-addressed, encrypted) |
| One model, one personality, everyone gets the same | Adapts to you specifically, learns your patterns |
| Bolted onto existing OS paradigm (app grid + voice) | IS the interface — no apps, just intent and outcomes |
| Always listening = always surveilling | Awareness levels with explicit capability grants |
| Trust the corporation or get nothing | Trust gradient — earn trust through observed behavior |
| Inference happens where the vendor decides | Inference routes based on YOUR trust scores |

Content-addressed storage makes the file browser metaphor dead. You don't navigate to WHERE something is — you express WHAT you want and WHY. Wylene makes that explicit.

## Color System Placement

Wylene is the **Blue service** — the transformation/translation layer. She touches every color because her job is to mediate between human intent and system capability:

| Color | Wylene's relationship |
|---|---|
| **Blue (primary)** | Wylene IS Blue's user-facing surface. Intent-to-action translation, multilingual, format negotiation, sensor-to-meaning |
| **Magenta** | Wylene generates UIs from the component library. Magenta owns the components; Wylene composes them |
| **Yellow (Oluo)** | Wylene queries Oluo to find relevant content/knowledge. "Show me photos from last weekend" = Oluo search + Wylene presentation |
| **Red (Kitri)** | Every Wylene action that needs compute is a Kitri workflow. Inference, search, transcoding — all durable, all auditable |
| **Cyan** | Wylene's awareness levels are capability-gated. Cyan enforces the boundaries. Rate-limiting on inference requests |
| **Green (Jain)** | Wylene surfaces Jain's recommendations ("you have 40GB of duplicate photos, want me to help?"). Jain is Wylene's housekeeper |
| **Lyll** | Every Wylene-generated UI, every inference result, every action taken — content-addressed and verifiable |
| **Nakaiah** | User model encryption, sensor consent management, compute delegation privacy. Nakaiah is Wylene's conscience |
| **Roxy** | When the user interacts with licensed content, Wylene handles the UX (purchase flow, expiry notices, key delivery) through Roxy |

## Core Architecture

### The Awareness State Machine

Wylene's awareness is a state machine with user-controlled transitions. Each level is a set of UCAN capabilities granted to Wylene's sensor workflows:

```
sleeping ──(wake word / tap)──→ listening ──(user grants)──→ watching ──(user grants)──→ guardian
    ↑                              ↑                            ↑                          │
    └──────────────────────────────┴────────────────────────────┴──(user revokes / timer)───┘
                                                                   (battery low / policy)
```

| Level | Sensors | Compute | Power | Use case |
|---|---|---|---|---|
| **Sleeping** | Wake word detector only (on-device, tiny model) | Near zero | Minimal | Phone in pocket, laptop closed |
| **Listening** | Microphone (on-device NLP) | Low — local models only | Low | Cooking, driving, hands-busy |
| **Watching** | Mic + cameras (periodic, configurable Hz) | Medium — local + trusted nodes | Medium | At desk, exploring, learning |
| **Guardian** | All sensors: mic, cameras, motion, light, network, GPS | Full — local + mesh | Higher | Accessibility aid, safety, travel, childcare |

**Auto-transitions** (all consent-gated):

- Battery below threshold: step down one level
- No user interaction for N minutes: step down to sleeping
- Motion/sound spike in sleeping: offer to escalate (not automatic)
- User-defined schedules: "guardian from 8am-6pm on weekdays, sleeping overnight"

### The Intent Pipeline

When the user engages Wylene, every interaction flows through five stages. Each stage is a Kitri workflow — durable, auditable, replayable. If Wylene crashes mid-pipeline, she resumes from the last checkpoint.

```
Human Input                    Wylene Pipeline                              System Output
─────────────                  ───────────────                              ─────────────
voice ──┐                     ┌─────────────┐
text ───┤                     │  Perceive   │ ← sensor fusion, STT, context
gesture─┤──────────────────→  │  (Blue)     │
tap ────┘                     └──────┬──────┘
                                     │
                              ┌──────▼──────┐
                              │ Understand  │ ← intent extraction, user model,
                              │ (Blue+Oluo) │   disambiguate, search memory
                              └──────┬──────┘
                                     │
                              ┌──────▼──────┐
                              │   Plan      │ ← select Kitri workflows, compose DAG,
                              │ (Blue+Kitri)│   estimate compute needs, pick nodes
                              └──────┬──────┘
                                     │
                              ┌──────▼──────┐     ┌──→ generated UI (Magenta components)
                              │  Present    │─────┤──→ audio response (TTS)
                              │ (Blue+Mag.) │     └──→ haptic/notification
                              └──────┬──────┘
                                     │
                              ┌──────▼──────┐
                              │   Learn     │ ← update user model, trust scores,
                              │(Blue+Nakaiah)│   preference refinement
                              └─────────────┘
```

**Perceive → Understand → Plan → Present → Learn.** This is a formalization of the OODA loop (Observe, Orient, Decide, Act) with a learning feedback step. Making each stage a Kitri workflow means the entire loop is durable and auditable — you could literally replay "what did Wylene think I meant when I said X?" from the event log.

### The User Model

Content-addressed, Nakaiah-encrypted, stored in the user's own content store. Each field is independently addressable (separate CIDs) so syncing is granular — updating `routines` on your phone doesn't re-upload your entire model.

```
UserModel (content-addressed blob, synced across devices)
├── communication_style     # voice patterns, vocabulary, formality preference
├── interaction_preferences # prefers voice vs. text vs. gesture, per-context
├── routines                # temporal patterns ("checks email at 9am")
├── relationships           # known contacts, communication patterns
├── goals                   # user-declared objectives ("learning Spanish")
├── trust_posture           # initial setting + per-node trust scores
└── device_contexts[]       # per-device instance models
    ├── device_identity     # unique 256-bit ID (fresh per boot for unikernels)
    ├── sensor_capabilities # what this device CAN perceive
    ├── local_model_inventory # what models are available on-device
    └── session_history     # recent interactions (rolling window)
```

Every Wylene instance is its own entity with its own identity and local context. The phone's Wylene and the laptop's Wylene are *siblings* — same parent user model, different lived experiences. The user model has natural layers:

- **Per-instance context** — this device, this session, this boot
- **Per-device context** — this phone knows about cameras/GPS, laptop knows about keyboard patterns
- **Cross-device user model** — preferences, communication style, goals, relationships (synced via mesh)

### UI Generation

Wylene assembles interfaces from a library of pre-built, accessible, tested components. The AI's job is *composition and layout*, not pixel-level generation.

**Phase 1 (native components):** Buttons, grids, media players, forms, lists, cards, navigation. Accessible by construction — screen reader support, keyboard navigation, color contrast, and motion reduction are properties of the component library. Fast, predictable, guaranteed accessible.

**Phase 2 (hybrid + canvas):** Add a freeform canvas for novel presentations — data visualizations, spatial layouts, artistic displays. The AI decides whether structured components or canvas best serves the intent. Canvas elements get automatic ARIA annotations generated by a separate accessibility workflow.

### Trust and Compute Routing

Trust is a living gradient, not a binary setting. It starts somewhere, evolves with evidence, and routing adapts automatically.

Three mechanisms work together:

**1. Initial trust posture** — set once by the user (conservative, balanced, adventurous). Determines willingness to route work to unproven nodes.

**2. Continuous trust scoring** — nodes earn and lose trust through observed behavior:

```
TrustScore per node:
├── initial_posture        # user's default
├── observed_latency       # rolling average response time
├── observed_correctness   # verified outputs vs. total requests
├── attestation_validity   # UCAN chain freshness, signer reputation
├── uptime_history         # availability over time
└── composite_score        # weighted combination → routing priority
```

Trust is asymmetric: hard to earn, easy to lose. A successful inference result nudges the score up (+δ). A failed or incorrect result drops it significantly (-Δ). This mirrors how human trust works.

**3. Automatic routing** — Wylene sends work to the best available node given current trust scores and data sensitivity. The user never manages a trust list; they just experience things working well (or not, in which case the system self-corrects).

Even "untrusted" nodes are useful — Harmony's cryptographic primitives mean they can perform blind computation on encrypted inputs (`kitri::seal`) and produce verifiable outputs (Lyll audit trails). They earn trust by doing good work, even on data they can't see.

Trust scores are content-addressed and stored in the user model. They sync across devices — your phone's bad experience with a node affects your laptop's routing too.

## Integration Map

### Kitri (Red — Programming Model)

Wylene's execution backbone. Every pipeline stage is a Kitri workflow:

```rust
#[kitri::workflow]
#[kitri::subscribes("wylene/input/raw")]
#[kitri::publishes("wylene/intent/resolved")]
async fn understand_intent(input: PerceptionFrame) -> KitriResult<ResolvedIntent> {
    let user_model = kitri::fetch(input.user_model_cid).await?;
    let context = kitri::query("oluo/context", &input.embedding).await?;
    let intent = kitri::infer("resolve user intent", &(input, user_model, context)).await?;
    Ok(intent)
}
```

| Wylene stage | Kitri primitive | Why |
|---|---|---|
| Perceive | `kitri::infer` (STT, vision) | Sensor fusion needs AI models |
| Understand | `kitri::query("oluo/...")` + `kitri::infer` | Search + intent extraction |
| Plan | `kitri::submit_dag` | Compose multi-step action plans |
| Present | `kitri::publish("wylene/ui/...")` | Push generated UI to renderer |
| Learn | `kitri::store` + `kitri::seal` | Persist encrypted user model updates |

Wylene never calls AI models directly. Every inference goes through `kitri::infer`, which means it's event-sourced, cached on replay, capability-gated, and routable to any trusted node based on trust scores.

### Oluo (Yellow — Discovery / Search) *stub*

Wylene's memory and knowledge. When the user asks "show me that article about mesh networking I read last month," Wylene needs to find it. Oluo is the librarian.

**Oluo's responsibilities** (to be designed in full):

- **Embedding space** — content → vector, semantic search across all formats and languages
- **Content index** — all content the user has accessed or created
- **Temporal index** — when things happened (for "last month" queries)
- **Relationship graph** — how content relates to other content, people, and goals
- **Ranking** — relevance scoring given user model + current context

**Zenoh key expressions:**

```
oluo/query/{user_hash}              # semantic search queries
oluo/index/{content_type}/*         # index updates (content created/accessed)
oluo/suggest/{user_hash}            # proactive suggestions
```

**Wylene ↔ Oluo contract:** Wylene sends a `SearchIntent` (natural language + embedding + temporal hints + user model reference). Oluo returns ranked `ContentReference`s (CIDs + relevance scores + snippets). Wylene never searches directly — Oluo is the librarian. Think of Oluo as a next-generation Dewey Decimal System in 1024 dimensions instead of one.

### Jain (Green — Maintenance / Cleanup) *stub*

Wylene's housekeeper. Jain works quietly in the background; Wylene is how Jain talks to the user. We all have so much digital data we don't want to take the effort to go through — some of it we want to keep, but so much we actually don't. Jain is the janitor/nurse that keeps systems clean, healthy, and functional.

**Jain's responsibilities** (to be designed in full):

- **Duplicate detection** — content-hash dedup + near-duplicate detection (perceptual hashing)
- **Staleness scoring** — access recency, reference count, relevance decay
- **Storage optimization** — hot/warm/cold tiering recommendations
- **Health monitoring** — per-node, per-dataset health scores
- **Composting** — graceful expiry of ephemeral content (the philosophical heart of Green)
- **Repair** — self-healing from redundant copies, error-correcting codes

**Zenoh key expressions:**

```
jain/health/{node_hash}                  # node health reports
jain/recommend/{user_hash}               # cleanup recommendations for Wylene
jain/action/{user_hash}/{action_id}      # user-approved cleanup actions
jain/stats/{node_hash}                   # storage/compute/bandwidth metrics
```

**Wylene ↔ Jain contract:** Jain publishes `CleanupRecommendation`s (what to clean, why, how much space recovered, confidence score). Wylene presents them in a friendly, non-anxious way. User approves or rejects. Wylene publishes the decision back to Jain. Jain never deletes without user consent surfaced through Wylene.

### Roxy (Red — Content Licensing)

Wylene is Roxy's storefront and concierge:

```
User: "Play that new album by $artist"
  → Wylene/Perceive: STT → text
  → Wylene/Understand: intent = play_content, query Oluo for artist
  → Wylene/Plan: check Roxy license status
    → Licensed? → fetch wrapped key, decrypt, stream
    → Not licensed? → present purchase UI (Magenta components)
      → User approves → Kitri workflow: Roxy purchase flow
      → Key wrapped for consumer, cache entry created
  → Wylene/Present: media player component + playback
  → Wylene/Learn: "user likes this artist" → user model
```

Zenoh topics Wylene subscribes to from Roxy:

```
roxy/catalog/**                     # artist content discovery
roxy/license/{consumer_hash}/*      # license status changes
roxy/cache/expiring/*               # CacheAction::NotifyExpiring → Wylene surfaces it
```

### harmony-identity (Trust Layer)

Wylene's trust engine. Trust score updates flow through the Learn pipeline stage:

```
Kitri workflow completes on remote node
  → result verified (Lyll: content hash matches, audit trail valid)
  → trust score updated: +δ for success, -Δ for failure
  → routing table updated: next inference request considers new scores
```

### harmony-content (Storage)

Everything Wylene produces or consumes is content-addressed:

| What | Stored as | CID type |
|---|---|---|
| User model fields | Encrypted blobs | ContentId per field |
| Generated UI snapshots | Component tree serialization | ContentId |
| Conversation history | Event log (Kitri audit trail) | ContentId per session |
| Trust scores | Encrypted user model field | ContentId |
| Wylene's own model weights (local) | Immutable content blob | ContentId |

### harmony-zenoh (Communication)

Wylene's full key expression namespace:

```
wylene/input/{device_hash}/raw          # raw sensor frames (local only, never leaves device)
wylene/input/{device_hash}/perception   # processed perception (may route to trusted nodes)
wylene/intent/{user_hash}/resolved      # resolved intents (internal pipeline)
wylene/ui/{device_hash}/frame           # generated UI frames for renderer
wylene/ui/{device_hash}/audio           # TTS output
wylene/awareness/{device_hash}/level    # current awareness level
wylene/model/{user_hash}/sync           # user model sync across devices
wylene/trust/{user_hash}/scores         # trust score updates
```

### harmony-crypto / harmony-reticulum (Network)

- All Wylene ↔ remote-node communication goes through Reticulum (delay-tolerant, mesh-routed)
- Inference requests to trusted nodes are encrypted end-to-end (ChaCha20-Poly1305 for Harmony-native, Fernet for Reticulum-compat nodes)
- Wylene's raw sensor data (camera frames, audio) is **never** transmitted unencrypted — even to trusted nodes, it's sealed with `kitri::seal` before leaving the device

### harmony-os (Ring Integration)

The awareness level maps to which ring is active:

| Level | Active Rings | What runs where |
|---|---|---|
| **Sleeping** | Ring 1 only | Unikernel handles wake word detection |
| **Listening** | Ring 1 + Ring 2 | Microkernel runs local STT workflow |
| **Watching** | All rings | Full pipeline with camera, UI generation |
| **Guardian** | All rings | Full sensor suite, proactive monitoring |

Ring responsibilities:

- **Ring 1 (Unikernel):** Wake word detector, low-power sensor polling. Every boot gets a unique 256-bit instance ID for auditability and as a seed for deterministic pseudo-random operations.
- **Ring 2 (Microkernel):** Wylene's Kitri workflow supervisor, capability enforcement, encryption boundary (`kitri::seal`/`kitri::open`). All encryption operations are system calls — Wylene workflows never touch raw key material.
- **Ring 3 (Full OS):** UI renderer, component library, audio I/O, full Wylene pipeline.

## What Wylene Is NOT

- **Not a chatbot.** Wylene can converse, but conversation is one modality among many. She generates UIs, plays audio, triggers automations, surfaces recommendations — whatever best serves the intent.
- **Not an app launcher.** There are no "apps" to launch. Wylene materializes the right interface for the right moment from composable components.
- **Not a search engine.** Oluo searches. Wylene asks Oluo and presents results.
- **Not a housekeeper.** Jain cleans. Wylene surfaces Jain's recommendations and relays the user's decisions.
- **Not omniscient.** Wylene is honest about what she doesn't know, what she can't do locally, and what would require more compute/trust/sensors than currently available.

## Named Services Summary

The color system is coming alive as named services, each embodying a concern:

| Service | Color | Role | Status |
|---|---|---|---|
| **Roxy** | Red | Content licensing — self-sovereign distribution | Crate implemented |
| **Kitri** | Red | Programming model — durable distributed computation | Design complete, Layer 0 planned |
| **Wylene** | Blue | Fluid interface — intent-to-outcome translation | This document |
| **Oluo** | Yellow | Discovery — semantic search, knowledge graph, content index | Stub (to be designed) |
| **Jain** | Green | Maintenance — cleanup, health monitoring, composting | Stub (to be designed) |
| **Lyll** | White | Public space guardian — verification, audit, governance | Conceptual |
| **Nakaiah** | Black | Private space guardian — encryption, consent, erasure | Conceptual |

Cyan (protection/resilience) and Magenta (UX/social/education) do not yet have named services — they may emerge as the architecture matures, or they may remain cross-cutting concerns embodied by capabilities within other services.

## Open Questions

1. **Local model selection** — which models run on-device for each awareness level? Qwen 3.5 is a strong candidate for multilingual NLP. What about vision models for the watching/guardian levels?
2. **Component library scope** — how large is the initial component set? What's the minimum viable set for Wylene to be useful?
3. **User model migration** — when the user model schema evolves, how do old models migrate forward? Content-addressed storage means old versions are immutable — need a migration workflow.
4. **Multi-user awareness** — in guardian mode with cameras, Wylene may perceive people other than the user. What are the privacy obligations? Does Wylene blur/ignore non-user faces by default?
5. **Offline-first user model** — if the user's devices can't sync for days (mesh partition), how do divergent user models reconcile? CRDTs? Last-writer-wins per field? Conflict resolution through Wylene?
6. **Accessibility of audio-only mode** — in listening mode, Wylene has no screen. How do deaf/hard-of-hearing users interact? Haptic patterns? Always require a screen for accessibility?
7. **Trust score cold start** — new nodes have no history. What's the bootstrapping mechanism? Attestation from known nodes? Proof-of-work trial period?
8. **Wylene-to-Wylene communication** — can two users' Wylene instances collaborate? ("Ask Alex's Wylene if he's free for dinner") What's the consent protocol?
