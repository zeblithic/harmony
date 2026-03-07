# Harmony Browser Design

## Goal

Build a next-generation content browser for the Harmony decentralized network that replaces location-addressed web browsing with content-addressed, trust-aware, privacy-first browsing. The browser renders CID-addressed content bundles and live pub/sub streams, using cryptographic trust scores instead of cookies and consent dark patterns.

## Architecture Overview

Two deliverables:

- **`harmony-browser` crate** — Pure Rust library in the `zeblithic/harmony` workspace (Apache-2.0 OR MIT). Sans-I/O state machine providing content resolution, trust policy, and subscription management. No async runtime, no GUI, no I/O.
- **`zeblithic/harmony-browser` repo** — Tauri v2 + Svelte 5 desktop app. Thin shell that drives the core state machine and renders the UI via webview. Same stack as `harmony-client`.

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Content model | Hybrid: static CIDs + live pub/sub + future WASM apps | Each serves a different purpose; all three coexist |
| Rendering layer | Pluggable: webview now, native renderer later | Rust GUI ecosystem isn't mature; trait boundary preserves the option |
| Trust model | Trust scores set defaults, UCANs enforce hard boundaries | Natural composition of two systems already designed |
| App structure | New Tauri app, separate from harmony-client | Browsing and messaging are different UX paradigms |
| MVP scope | Content viewing + live subscriptions | WASM apps deferred until UCANs are built |

## Crate Dependencies

```
harmony-browser
  ├── harmony-content   (CIDs, BlobStore, BundleBuilder)
  ├── harmony-identity  (Identity, TrustScore, sign/verify)
  ├── harmony-zenoh     (PubSubRouter, QueryableRouter, SubscriptionTable)
  └── harmony-crypto    (transitive via identity)
```

Does NOT depend on `harmony-compute` (WASM apps are Phase 2) or `harmony-reticulum` (transport is the caller's concern, not the browser core's).

## Core Types

### Navigation

```rust
/// What the user types into the address bar.
pub enum BrowseTarget {
    /// Direct CID reference (e.g., "hmy:abc123...")
    Cid(ContentId),
    /// Human-friendly name resolved via Zenoh queryable
    /// (e.g., "harmony/content/wiki/rust")
    Named(String),
    /// Live subscription to a key expression
    /// (e.g., "harmony/presence/**")
    Subscribe(String),
}
```

### Resolved Content

```rust
/// What the core resolves a target into.
pub enum ResolvedContent {
    /// Static content bundle with metadata.
    Static {
        cid: ContentId,
        mime: MimeHint,
        data: Vec<u8>,
        author: Option<Identity>,
        trust_level: TrustDecision,
    },
    /// Live subscription producing a stream of updates.
    Live {
        key_expr: String,
        subscription_id: SubscriptionId,
    },
}

/// Content type detected from InlineMetadata MIME hint.
pub enum MimeHint {
    Markdown,
    PlainText,
    Image(ImageFormat),
    HarmonyApp,            // Phase 2: WASM module
    Unknown([u8; 8]),
}
```

### Trust Decisions

```rust
/// Trust-based rendering decision for content.
pub enum TrustDecision {
    /// Render fully (text, images, everything).
    FullTrust,
    /// Render text, gate media behind one-click approval.
    Preview,
    /// Show metadata only. User must explicitly load.
    Untrusted,
    /// Author unknown to trust network. Show clear prompt.
    Unknown,
}
```

## Sans-I/O State Machine

Follows the same pattern as every Harmony crate: the caller provides events, the core returns actions.

```rust
pub enum BrowserEvent {
    /// User entered something in the address bar.
    Navigate(BrowseTarget),
    /// Content bytes arrived from network.
    ContentFetched { cid: ContentId, data: Vec<u8> },
    /// Pub/sub sample arrived.
    SubscriptionUpdate { sub_id: SubscriptionId, key_expr: String, payload: Vec<u8> },
    /// User changed trust for an author.
    TrustUpdated { address: String, score: TrustScore },
    /// User explicitly approved gated content.
    ApproveContent { cid: ContentId },
}

pub enum BrowserAction {
    /// Request content from network (caller does the I/O).
    FetchContent { cid: ContentId },
    /// Query a named resource via Zenoh.
    QueryNamed { key_expr: String },
    /// Subscribe to a key expression.
    Subscribe { key_expr: String },
    /// Deliver resolved content to the renderer.
    Render(ResolvedContent),
    /// Update trust display.
    ShowTrustInfo { author: Identity, score: TrustScore, decision: TrustDecision },
}

pub struct BrowserCore {
    pub fn new(local_identity: Identity, trust_service: TrustService) -> Self;
    pub fn handle_event(&mut self, event: BrowserEvent) -> Vec<BrowserAction>;
}
```

## Content Resolution Pipeline

```
User input
    │
    ├── CID ("hmy:7f3a...") ──────► FetchContent by CID
    │                                     │
    ├── Named ("wiki/rust") ──────► QueryNamed("harmony/content/wiki/rust")
    │                                     │
    │                               Queryable responds with root CID
    │                                     │
    │                               FetchContent by CID
    │                                     │
    └── Subscribe ("~presence/**") ► Subscribe("harmony/presence/**")
                                          │
                                    Updates arrive as SubscriptionUpdate events
                                          │
    ◄─────────────────────────────────────┘
    │
    ▼
ContentFetched { cid, data }
    │
    ▼
Validate CID hash (data matches content ID)
    │
    ▼
Parse bundle: extract InlineMetadata (MIME, size, author)
    │
    ▼
Resolve trust: TrustService.resolve(author_address) → TrustDecision
    │
    ▼
Emit BrowserAction::Render(ResolvedContent::Static { ... })
```

### Content Type Rendering

- **Markdown** — Always rendered regardless of trust level (text is safe). Links to other CIDs become navigable.
- **Images** — Gated by trust decision. `FullTrust` → inline. `Preview` → blurred placeholder + one-click load. `Untrusted`/`Unknown` → metadata only.
- **Bundles with children** — Recursively resolved. Each child inherits the parent's author and trust decision.
- **Unknown types** — Hex dump with download option.

### Live Subscription Rendering

```rust
pub enum SubscriptionDisplay {
    /// Try to parse as UTF-8 text, fall back to hex.
    Auto,
    /// Always show as JSON.
    Json,
    /// Always show as hex.
    Hex,
}
```

Each subscription update renders as a card: key expression, timestamp, payload in chosen format, author trust badge if signed.

## Trust Policy Engine — Replacing Cookies

### Resolution Chain

```
Content arrives with author Identity
    │
    ▼
Per-author trust score? ── Yes → use that score
    │ No
    ▼
Per-community trust override? ── Yes → use that
    │ No
    ▼
Trust graph transitive score? (Phase 2) ── Yes → use that
    │ No
    ▼
Global default policy
    │
    ▼
TrustDecision (FullTrust / Preview / Untrusted / Unknown)
```

This is the existing `TrustService.resolve()` chain — reused, not reinvented.

### What Replaces Web Dark Patterns

| Web Dark Pattern | Harmony Replacement |
|---|---|
| Cookie consent banners (7 clicks to refuse) | No cookies exist. Identity is your keypair. |
| "Accept all" as the big button | Trust defaults to `Unknown`. Granting trust is explicit. |
| Third-party tracking pixels | Content is CID-addressed. No referrer headers, no cross-site tracking. Publishers don't know who subscribes. |
| Browser fingerprinting | No DOM, no JavaScript, no browser APIs to fingerprint. |
| "Sign in with Google/Facebook" | Self-sovereign Ed25519 keypair. No federated login. |
| Hidden data sharing | Privacy panel shows ALL data flows in real-time. |
| "Legitimate interest" buried in settings | No concept of "legitimate interest." Zero data leaves without explicit action. |

### Trust UX Flow

1. User navigates to `wiki/quantum-computing`
2. Content resolves to a bundle authored by peer `7f3a...`
3. Browser checks: no trust score for `7f3a...` → `TrustDecision::Unknown`
4. Content pane shows: author fingerprint, content size/type, prompt: **"Unknown author. Show text only / Trust and load fully / Dismiss"**
5. "Trust and load fully" → TrustEditor opens inline, user sets score, content loads
6. Decision persists. Next time this author's content appears, it auto-loads per their score.
7. "Show text only" → text renders, media gated. No trust score saved.

**Key difference from the web:** One decision per author, not per page, not per session, not per cookie expiry. Trust is about people, not sites.

## Tauri Shell Layout

```
┌──────────────────────────────────────────────────┐
│  [Trust Badge]  [ Address Bar_________________ ] │
├───────────┬──────────────────────────────────────┤
│           │                                      │
│ Subscr.   │          Content Pane                │
│ Sidebar   │                                      │
│           │   (markdown / images / live cards     │
│ ───────── │    / trust-gated blocks)             │
│           │                                      │
│ Privacy   │                                      │
│ Panel     │                                      │
│           │                                      │
├───────────┴──────────────────────────────────────┤
│  Privacy Footer: 0 trackers · 2 active subs      │
└──────────────────────────────────────────────────┘
```

### Address Bar Input Formats

- Raw CID: `hmy:7f3a2b...` (hex) or `hmy:b32:ABCD...` (base32)
- Named path: `wiki/rust` (resolves via Zenoh queryable at `harmony/content/wiki/rust`)
- Live subscription: `~presence/**` (prefix `~` signals subscribe)

### UI Regions

1. **Address Bar** — Input + navigation history
2. **Content Pane** — Renders resolved content with trust gating
3. **Subscription Sidebar** — Active pub/sub subscriptions with update counts, unsubscribe buttons
4. **Privacy Panel** — All active connections, data flows, peer trust scores, one-click trust adjustment

### Privacy-First Design Principles

- No cookies, no local storage tokens, no tracking IDs. Identity is your keypair.
- Trust is always bidirectional and visible. You see your score for them AND theirs for you.
- One click to trust, one click to revoke. Never more.
- Content provenance is mandatory. Every piece of content shows its author and your trust relationship.

## Testing Strategy

### `harmony-browser` crate (pure Rust)

Sans-I/O makes testing trivial — no mocks, no async:

```rust
#[test]
fn navigating_to_cid_emits_fetch() {
    let mut core = BrowserCore::new(test_identity(), test_trust_service());
    let actions = core.handle_event(BrowserEvent::Navigate(
        BrowseTarget::Cid(test_cid()),
    ));
    assert!(matches!(actions[0], BrowserAction::FetchContent { .. }));
}

#[test]
fn unknown_author_gets_unknown_trust_decision() {
    let mut core = BrowserCore::new(test_identity(), test_trust_service());
    let actions = core.handle_event(BrowserEvent::ContentFetched {
        cid: test_cid(),
        data: test_markdown_bundle(unknown_author()),
    });
    match &actions[0] {
        BrowserAction::Render(ResolvedContent::Static { trust_level, .. }) => {
            assert_eq!(*trust_level, TrustDecision::Unknown);
        }
        _ => panic!("Expected Render action"),
    }
}
```

### Tauri shell (vitest + @testing-library/svelte)

- Address bar parsing (CID vs named vs subscription)
- Trust-gated content rendering (all four TrustDecision states)
- Privacy panel data flow display
- Subscription sidebar management

## Implementation Phases

| Phase | Scope | Depends On |
|-------|-------|-----------|
| **1** | `harmony-browser` crate: types, BrowserCore state machine, trust policy, content resolution | `harmony-content`, `harmony-identity`, `harmony-zenoh` (all exist) |
| **2** | Tauri shell: scaffold, address bar, content pane (markdown + trust-gated images), trust indicators | Phase 1 |
| **3** | Live subscriptions: subscription manager, sidebar, card feed | Phase 1 |
| **4** | Privacy panel: connection display, data flow visualization, inline trust editing | Phase 2 |
| **5 (future)** | WASM app hosting: CID-addressed modules, UCAN capability grants, sandboxed execution | UCANs (designed, not built), `harmony-compute` |
| **6 (future)** | Native renderer: swap webview for Xilem/Makepad behind renderer trait | Rust GUI ecosystem maturity |

Phases 1-4 are the MVP. Each phase is independently deliverable and testable.

## Gemini Report Assessment

The Gemini research report provides useful framing but diverges from reality in several ways:

**What aligns:** Content-addressed CIDs, Zenoh pub/sub reactivity, WASM sandboxing, cryptographic identity, trust scoring — all already built or designed.

**What's overstated:** Zenoh-Flow adds complexity without clear benefit for MVP. A2UI is interesting but unproven. Full custom GPU rendering (Xilem/Makepad) is premature given ecosystem maturity.

**What's missed:** UCANs (critical for Phase 5 authorization), HAMT encrypted namespaces (designed), the sans-I/O paradigm (Harmony's core architectural pattern), `harmony-compute`'s cooperative yielding + durable checkpoints + I/O request/response (more capable than described).

**Key tension resolved:** Gemini assumes WASM is always needed as a security sandbox. In Harmony, WASM is one option — but the trust network + UCANs + capability-gated OS (Ring 2) provide security without the WASM overhead for trusted code. WASM is for *untrusted third-party apps*, not for everything.
