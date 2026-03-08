# Harmony Color System Architecture

**Purpose:** A taxonomy for organizing all Harmony platform capabilities by *concern* rather than by traditional layer (network, storage, compute, presentation). Six primary colors across three axes, plus a meta-axis of two guardian colors that ensure the system's integrity.

**Status:** Conceptual framework. Not yet tied to crate boundaries or implementation.

---

## Overview

| Axis | Color | Role | Complement |
|------|-------|------|------------|
| **Magic** (trust) | White (Lyll) | Public space guardian | Black (Nakaiah) |
| | Black (Nakaiah) | Private space guardian | White (Lyll) |
| **Will** (action) | Red | Creation / builder's tools | Cyan |
| | Cyan | Protection / resilience | Red |
| **Information** (knowledge) | Yellow | Discovery / observability | Blue |
| | Blue | Transformation / encoding | Yellow |
| **Experience** (humanity) | Magenta | UX / social / education | Green |
| | Green | Maintenance / entropy reduction | Magenta |

Each axis pairs a constructive force with a conservative force. Neither is complete without the other.

---

## The Meta-Axis: Magic (White / Black)

Lyll and Nakaiah are the "constitution" of the system. They don't build things or transform data — they ensure the rules themselves are trustworthy. Their entire purpose is to protect the "magic" itself: the bits, logic, integrity, and energy that make technology work.

We have more compute power and data than we know what to do with, but we've done a terrible job of protecting our data. It's been stolen and abused and we can't undo the past, but we can create a platform that lets people reclaim their information and their digital lives from today onward, forever. Lyll and Nakaiah are the "law enforcement" that help keep the system completely trustworthy and accountable for everyone it serves.

### Lyll (White) — Public Space Guardian

The things that *must* be visible to maintain trust:

- **Content addressing & verification** — every piece of data has a provable hash, every claim is verifiable
- **Public audit trails** — who changed what governance rule, when, why, with what community consent
- **Consensus protocols** — mechanisms by which the community agrees on shared truth (block validation, quorum decisions)
- **Identity attestation** — "this public key belongs to this entity" without revealing anything else
- **Open API contracts** — published schemas and protocols anyone can build against, guaranteed stable
- **Transparency reporting** — automated proof the system is behaving as promised (e.g., "no data was accessed without consent in this epoch, here's the cryptographic proof")
- **Governance frameworks** — how rules get proposed, debated, voted on, enacted. The "legislative process" of Harmony
- **Reputation ledgers** — not social credit scores, but verifiable track records: "this node has served 10M requests with 99.99% honesty"

### Nakaiah (Black) — Private Space Guardian

The things that *must* be invisible to maintain trust:

- **Private key custody** — secure generation, storage, rotation, recovery of identity keys
- **Zero-knowledge proofs** — "I'm over 18" without revealing your birthday; "I own this asset" without revealing which wallet
- **Consent management** — granular, revocable, auditable consent: who can see what, when, under what conditions
- **Sealed computation** — run code on data you can't see, return a result without learning the inputs (homomorphic encryption, secure enclaves)
- **Right to erasure** — not just deleting your copy, but cryptographically ensuring all copies become unreadable (key rotation/destruction)
- **Plausible deniability layers** — steganography, traffic analysis resistance, metadata minimization
- **Data sovereignty enforcement** — your data never leaves your control boundary without explicit, informed consent
- **Anonymous credentials** — participate in systems without revealing identity (anonymous voting, whistleblowing)

**The tension between them is the point.** Lyll says "prove it." Nakaiah says "without revealing this." Together they create a system where trust doesn't require surveillance.

---

## The Will Axis: Creation / Protection (Red / Cyan)

### Red — Builder's Tools

Everything that turns intent into reality:

- **Application frameworks** — SDKs, scaffolding, and patterns for building on Harmony. The "create-react-app" for decentralized apps
- **Workflow authoring** — visual and programmatic tools for defining computational workflows (harmony-workflow/harmony-compute territory). "If this, then that" for the decentralized web
- **Content creation primitives** — publishing, versioning, forking. Write a document, record a video, compose music — instantly addressable and distributable
- **Smart contracts / logic contracts** — user-defined rules that execute deterministically. Not just financial but general-purpose: "when someone accesses my dataset, log it and charge 0.001 credits"
- **Domain-specific languages** — tailored languages for common tasks: data queries, content policies, access rules, routing logic
- **Templating & theming** — reusable patterns that let non-technical users create sophisticated experiences by composing building blocks
- **Deployment & publishing pipelines** — "push to publish" for apps, content, services. No DevOps PhD required
- **Simulation & preview** — test your creation in a sandbox before it goes live, see how it behaves under load, verify its policies
- **Collaboration tools** — real-time co-creation, branching/merging of any content type (not just code), conflict resolution

### Cyan — Protection & Resilience

Everything that keeps creations safe and the system stable:

- **Rate limiting & circuit breakers** — prevent abuse without preventing use. Adaptive throttling that distinguishes genuine load from attack
- **Redundancy & replication** — automatic data replication across nodes with configurable durability guarantees ("this must survive 3 node failures")
- **Sandboxing & isolation** — every app/workflow runs in a contained environment. A buggy app can't crash its neighbors
- **Intrusion detection** — pattern recognition for anomalous behavior. Not just network attacks but semantic attacks (data poisoning, reputation manipulation)
- **Failover & self-healing** — when a node goes down, its responsibilities automatically redistribute. No single point of failure
- **Permission systems** — capability-based security. Fine-grained: "this app can read your calendar but not your contacts, and only during business hours"
- **Quarantine** — isolate suspicious content/behavior for inspection without destroying it. Innocent until proven guilty
- **Backup & point-in-time recovery** — "restore my data to how it looked last Tuesday at 3pm"
- **DDoS absorption** — the mesh topology itself is the defense. Attacks on one node get absorbed and redistributed across the mesh
- **Lyll/Nakaiah orchestration** — Cyan is the parent: it defines *what needs protecting* and delegates to White/Black for the *how*

---

## The Information Axis: Discovery / Transformation (Yellow / Blue)

### Yellow — Observability, Search, Knowledge

Everything about finding, understanding, and surfacing information:

- **Semantic search** — not just keyword matching but meaning matching. "Find documents about renewable energy policy in Southeast Asia" across every format, language, and data structure
- **Content discovery & recommendation** — "people who found this useful also found..." without building creepy surveillance profiles (recommendations run locally, on your data, with your model)
- **Monitoring & alerting** — system health, application metrics, SLA tracking. "Tell me when latency exceeds 200ms" or "alert when disk usage crosses 80%"
- **Knowledge graphs** — relationships between things. Not just "this document exists" but "this document references that dataset which was produced by this workflow which is owned by that organization"
- **Caching strategies** — intelligent prefetching. If you accessed something yesterday at this time, it's probably warm in your local cache today
- **DNS / naming / discovery** — human-readable names mapped to content addresses. "harmony://alice/blog/latest" resolves to the current content hash
- **Dashboards & visualization** — real-time views into any system you have access to observe. Composable, shareable, embeddable
- **Indexing infrastructure** — crawlers, indexers, and catalog builders. Distributed by nature — every node indexes what it sees
- **Anomaly detection** — "this pattern is unusual" surfaced proactively, whether it's a security concern (hand to Cyan) or an interesting trend (surface to the user)
- **Provenance tracking** — where did this data come from? What's its chain of custody? Is it authentic?

### Blue — Data Transformation & Transmutation

Everything about changing data's form while preserving its essence:

- **Codec libraries** — PNG to JPG to WebP to AVIF. WAV to MP3 to FLAC to Opus. Every format conversion imaginable
- **Compression** — not just ZIP but content-aware compression. Deduplicate at the block level before compressing. Store one copy of a popular video, not a million
- **Encryption / decryption** — the actual crypto primitives. AES, ChaCha20, Fernet. These live in Blue. Nakaiah *decides* what to encrypt; Blue *does* the encryption
- **Serialization** — JSON to CBOR to Protobuf to MessagePack. The lingua franca translators of the data world
- **Schema migration** — "this data was version 3, the current schema is version 7, here's the lossless transformation path"
- **Transcoding pipelines** — "take this 4K video, produce 1080p, 720p, and 480p variants, each with appropriate codec settings"
- **Anonymization & pseudonymization** — transform data to remove identifying information while preserving analytical utility. k-anonymity, differential privacy
- **Normalization** — "these three datasets use different date formats, units, and naming conventions. Harmonize them into one consistent representation"
- **Watermarking & steganography** — embed invisible provenance marks in content. "This image was generated by Model X on Date Y" — tamper-evident
- **Format negotiation** — "I have HEIC, you need JPG, let me automatically serve the right format for your capabilities"

**The axis tension:** Yellow wants to *see everything* to be maximally helpful. Blue ensures that what Yellow sees is appropriately encoded — encrypted where it should be, anonymized where needed, optimized for the context. Blue is Yellow's responsible handler.

---

## The Experience Axis: Polish / Maintenance (Magenta / Green)

### Magenta — User Experience, Social, Education

Everything that makes the system *humane*:

- **UI component libraries** — accessible, beautiful, consistent building blocks. The design system for all of Harmony
- **Onboarding flows** — "you just joined. Here's what Harmony is, here's your identity, here's your first experience." Progressive disclosure, not information firehose
- **Accessibility infrastructure** — screen reader support, keyboard navigation, color contrast, motion reduction — baked in at the platform level, not bolted on per-app
- **Localization / i18n** — every string, format, cultural convention. Right-to-left, ideographic, everything. The platform handles it
- **Education & tutorials** — interactive guides, contextual help, learning paths. "You're building your first app — here's a step-by-step walkthrough"
- **Social primitives** — identity profiles, messaging, groups, communities, reputation display. The social fabric that turns a network into a society
- **Notification systems** — intelligent, respectful notifications. Not dopamine-hacking attention hijacking, but genuinely useful "here's something you need to know"
- **Gamification with integrity** — achievement systems, progress indicators, challenges — designed to educate and empower, not addict
- **Feedback loops** — "was this helpful? what went wrong? how can we improve?" Built into every interaction
- **Cultural adaptation** — not just language translation but cultural context. Humor, formality, visual aesthetics that resonate locally
- **Emotion-aware design** — error messages that don't blame the user. Loading states that reduce anxiety. Confirmations that build confidence

### Green — Maintenance, Entropy Reduction, Housekeeping

Everything that keeps the system from rotting:

- **Garbage collection** — detect and reclaim data that's truly unreferenced. Not aggressive deletion but "this has zero references and no owner — safe to reclaim"
- **Deduplication** — content-addressed storage means identical data is stored once. Green actively scans for near-duplicates and suggests consolidation
- **Storage optimization** — "this data hasn't been accessed in 2 years. Migrate from hot SSD to cold archival, but keep it retrievable within 30 seconds"
- **Cache eviction** — intelligent cache management. Not just LRU but cost-aware: "this entry is cheap to regenerate, evict it before that expensive one"
- **Log rotation & compaction** — audit trails are important (Yellow) but they grow forever. Green compacts: keep the summary, archive the details, delete the noise
- **Dead reference cleanup** — links that point to content that no longer exists. Green detects and either repairs (find the new location) or marks as broken
- **Self-healing data** — error-correcting codes, automatic repair from redundant copies. "3 of 5 replicas agree, the other 2 are corrupted — fix them"
- **Resource reclamation** — nodes that went offline and came back have stale state. Green helps them catch up and release resources they no longer need
- **Defragmentation** — data across the mesh accumulates fragmentation. Green periodically reorganizes for locality and access efficiency
- **Digital composting** — the philosophical heart of Green. Data has a lifecycle. Some data should decompose naturally — ephemeral messages, temporary files, superseded versions. Green makes this graceful rather than catastrophic
- **Power/compute efficiency** — actively optimizing resource consumption. "This workflow runs every hour but could run daily with no loss of utility — suggest the change"
- **Health scoring** — every node, dataset, service gets a health score. Green focuses attention on the unhealthiest parts first

---

## Cross-Axis Interactions

The real power is in how the colors *compose*:

| Interaction | What happens |
|---|---|
| **Red + Blue** | Builder creates an app; Blue provides the format adapters and encoding layers the app needs |
| **Yellow + Cyan** | Yellow detects anomaly; Cyan activates protection |
| **Green + Yellow** | Green identifies stale data; Yellow confirms it's truly unreferenced via the knowledge graph |
| **Magenta + Green** | Magenta surfaces "your storage is 80% full" in a friendly, actionable way; Green executes the cleanup |
| **Blue + Nakaiah** | Blue encrypts; Nakaiah decides what key to use and who can decrypt |
| **Red + Magenta** | Red provides the building blocks; Magenta provides the UX patterns that make them usable |
| **Lyll + Yellow** | Lyll publishes audit proofs; Yellow indexes and makes them searchable |
| **Cyan + Green** | Cyan prevents damage; Green repairs what got through |

Each axis pair has a natural constructive/conservative dynamic:

- **Red builds, Cyan protects** what was built
- **Yellow discovers, Blue controls** how discoveries are represented
- **Magenta polishes, Green maintains** what was polished
- **Lyll reveals, Nakaiah conceals** — together they define trust

---

## Design Philosophy

Most architectures organize by *layer* (network, storage, compute, presentation). Organizing by *concern* — creation, protection, discovery, transformation, experience, maintenance — with trust as the meta-layer is more aligned with how humans actually think about their relationship with technology.

The key insight: traditional systems over-invest in Red (building) and Magenta (UX) while under-investing in Green (maintenance), Cyan (protection), and Yellow (observability). Blue (transformation) is often treated as library code rather than a core system concern. This taxonomy gives each concern equal architectural weight.

The order of magnitude reduction in humanity's digital footprint that Green promises is only achievable because Yellow knows what's actually being used, Blue can deduplicate and compress intelligently, and Cyan ensures that cleanup operations can't accidentally destroy something important. No color works alone.

---

## Case Study: Roxy (Content Licensing)

See `2026-03-08-roxy-content-licensing-design.md` for the full design. Roxy demonstrates how a single feature touches every color:

| Color | Roxy role |
|---|---|
| **Red** | License manifest authoring, catalog publishing |
| **Blue** | Key wrapping (ECDH + ChaCha20), CBOR serialization |
| **Yellow** | Zenoh catalog discovery, semantic search indexing |
| **Cyan** | UCAN access control, revocation enforcement |
| **Green** | Cache lifecycle — immediate eviction on license expiry |
| **Magenta** | Artist profiles, consumer UX, expiry notifications |
| **Lyll** | Content addressing (provable hashes), creator attestation |
| **Nakaiah** | Encrypted content storage, per-consumer key wrapping, consent |
