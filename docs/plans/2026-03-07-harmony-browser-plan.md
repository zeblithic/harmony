# Harmony Browser Crate Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the `harmony-browser` crate — a sans-I/O state machine that resolves content-addressed resources, applies trust-based rendering decisions, and manages live pub/sub subscriptions.

**Architecture:** Event/action state machine (same pattern as `harmony-reticulum::Node` and `harmony-zenoh::PubSubRouter`). The caller provides `BrowserEvent`s (user navigation, network responses, subscription updates) and receives `BrowserAction`s (fetch content, subscribe, render). No async, no I/O, no GUI — pure logic.

**Tech Stack:** Rust 1.81+, edition 2021, `thiserror` for errors, depends on `harmony-content`, `harmony-identity`, `harmony-zenoh`

**Design doc:** `docs/plans/2026-03-07-harmony-browser-design.md`

---

### Task 1: Scaffold the `harmony-browser` crate

**Files:**
- Create: `crates/harmony-browser/Cargo.toml`
- Create: `crates/harmony-browser/src/lib.rs`
- Modify: `Cargo.toml` (workspace members list)

**Step 1: Create `crates/harmony-browser/Cargo.toml`**

```toml
[package]
name = "harmony-browser"
version.workspace = true
edition.workspace = true
license.workspace = true
repository.workspace = true
rust-version.workspace = true
description = "Sans-I/O browser core for Harmony content-addressed networks"

[dependencies]
harmony-content = { path = "../harmony-content" }
harmony-identity = { path = "../harmony-identity" }
harmony-zenoh = { path = "../harmony-zenoh" }
thiserror = { workspace = true }

[dev-dependencies]
rand = "0.8"
```

**Step 2: Create `crates/harmony-browser/src/lib.rs`**

```rust
pub mod error;
pub mod types;

pub use error::BrowserError;
pub use types::*;
```

**Step 3: Create `crates/harmony-browser/src/error.rs`**

```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum BrowserError {
    #[error("content error: {0}")]
    Content(#[from] harmony_content::ContentError),

    #[error("invalid CID hex: {0}")]
    InvalidCidHex(String),

    #[error("subscription not found: {0}")]
    SubscriptionNotFound(u64),
}
```

**Step 4: Create `crates/harmony-browser/src/types.rs`** (empty placeholder)

```rust
// Types will be added in subsequent tasks.
```

**Step 5: Add to workspace**

In the root `Cargo.toml`, add `"crates/harmony-browser"` to the `members` list, after `"crates/harmony-content"` (alphabetical).

**Step 6: Verify it compiles**

Run: `cargo check -p harmony-browser`
Expected: Compiles with no errors

**Step 7: Commit**

```bash
git add crates/harmony-browser/ Cargo.toml
git commit -m "feat(browser): scaffold harmony-browser crate"
```

---

### Task 2: Core types — `BrowseTarget`, `MimeHint`, `TrustDecision`

**Files:**
- Create: `crates/harmony-browser/src/types.rs`
- Test: `crates/harmony-browser/tests/types.rs`

**Step 1: Write the failing tests**

Create `crates/harmony-browser/tests/types.rs`:

```rust
use harmony_browser::{BrowseTarget, MimeHint, TrustDecision};
use harmony_content::ContentId;

#[test]
fn browse_target_cid_holds_content_id() {
    let cid = ContentId::for_blob(b"hello world").unwrap();
    let target = BrowseTarget::Cid(cid);
    match target {
        BrowseTarget::Cid(c) => assert_eq!(c, cid),
        _ => panic!("expected Cid variant"),
    }
}

#[test]
fn browse_target_named_holds_string() {
    let target = BrowseTarget::Named("wiki/rust".into());
    match target {
        BrowseTarget::Named(ref s) => assert_eq!(s, "wiki/rust"),
        _ => panic!("expected Named variant"),
    }
}

#[test]
fn browse_target_subscribe_holds_key_expr() {
    let target = BrowseTarget::Subscribe("harmony/presence/**".into());
    match target {
        BrowseTarget::Subscribe(ref s) => assert_eq!(s, "harmony/presence/**"),
        _ => panic!("expected Subscribe variant"),
    }
}

#[test]
fn mime_hint_from_bytes_recognizes_markdown() {
    assert_eq!(MimeHint::from_mime_bytes(*b"text/md\0"), MimeHint::Markdown);
}

#[test]
fn mime_hint_from_bytes_recognizes_plain_text() {
    assert_eq!(MimeHint::from_mime_bytes(*b"text/pln"), MimeHint::PlainText);
}

#[test]
fn mime_hint_from_bytes_recognizes_png() {
    assert_eq!(
        MimeHint::from_mime_bytes(*b"img/png\0"),
        MimeHint::Image(harmony_browser::ImageFormat::Png),
    );
}

#[test]
fn mime_hint_unknown_for_unrecognized() {
    let bytes = *b"foo/bar\0";
    assert_eq!(MimeHint::from_mime_bytes(bytes), MimeHint::Unknown(bytes));
}

#[test]
fn trust_decision_default_is_unknown() {
    assert_eq!(TrustDecision::default(), TrustDecision::Unknown);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-browser --test types`
Expected: FAIL — `BrowseTarget`, `MimeHint`, etc. not found

**Step 3: Implement the types**

Replace `crates/harmony-browser/src/types.rs` with:

```rust
use harmony_content::ContentId;

/// What the user types into the address bar.
#[derive(Debug, Clone)]
pub enum BrowseTarget {
    /// Direct CID reference (e.g., "hmy:abc123...").
    Cid(ContentId),
    /// Human-friendly name resolved via Zenoh queryable.
    Named(String),
    /// Live subscription to a key expression.
    Subscribe(String),
}

/// Image format for rendering decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageFormat {
    Png,
    Jpg,
    Webp,
}

/// Content type detected from InlineMetadata MIME hint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MimeHint {
    Markdown,
    PlainText,
    Image(ImageFormat),
    HarmonyApp,
    Unknown([u8; 8]),
}

impl MimeHint {
    pub fn from_mime_bytes(bytes: [u8; 8]) -> Self {
        match &bytes {
            b"text/md\0" => Self::Markdown,
            b"text/pln" => Self::PlainText,
            b"img/png\0" => Self::Image(ImageFormat::Png),
            b"img/jpg\0" => Self::Image(ImageFormat::Jpg),
            b"img/webp" => Self::Image(ImageFormat::Webp),
            b"app/hmy\0" => Self::HarmonyApp,
            _ => Self::Unknown(bytes),
        }
    }
}

/// Trust-based rendering decision for content.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

impl Default for TrustDecision {
    fn default() -> Self {
        Self::Unknown
    }
}
```

**Step 4: Update `lib.rs` exports**

```rust
pub mod error;
pub mod types;

pub use error::BrowserError;
pub use types::*;
```

**Step 5: Run tests to verify they pass**

Run: `cargo test -p harmony-browser --test types`
Expected: 8 tests PASS

**Step 6: Commit**

```bash
git add crates/harmony-browser/
git commit -m "feat(browser): add BrowseTarget, MimeHint, TrustDecision types"
```

---

### Task 3: Address bar parsing — `BrowseTarget::parse`

**Files:**
- Modify: `crates/harmony-browser/src/types.rs`
- Test: `crates/harmony-browser/tests/parse.rs`

**Step 1: Write the failing tests**

Create `crates/harmony-browser/tests/parse.rs`:

```rust
use harmony_browser::{BrowseTarget, BrowserError};
use harmony_content::ContentId;

#[test]
fn parse_cid_hex_with_hmy_prefix() {
    let cid = ContentId::for_blob(b"test data").unwrap();
    let hex = hex::encode(cid.to_bytes());
    let input = format!("hmy:{hex}");
    let target = BrowseTarget::parse(&input).unwrap();
    match target {
        BrowseTarget::Cid(c) => assert_eq!(c, cid),
        _ => panic!("expected Cid"),
    }
}

#[test]
fn parse_subscribe_with_tilde_prefix() {
    let target = BrowseTarget::parse("~presence/**").unwrap();
    match target {
        BrowseTarget::Subscribe(ref s) => assert_eq!(s, "harmony/presence/**"),
        _ => panic!("expected Subscribe"),
    }
}

#[test]
fn parse_named_path() {
    let target = BrowseTarget::parse("wiki/rust").unwrap();
    match target {
        BrowseTarget::Named(ref s) => assert_eq!(s, "harmony/content/wiki/rust"),
        _ => panic!("expected Named"),
    }
}

#[test]
fn parse_invalid_cid_hex_returns_error() {
    let result = BrowseTarget::parse("hmy:not_valid_hex_at_all");
    assert!(result.is_err());
}

#[test]
fn parse_empty_string_as_named() {
    // Empty navigates to root
    let target = BrowseTarget::parse("").unwrap();
    match target {
        BrowseTarget::Named(ref s) => assert_eq!(s, "harmony/content/"),
        _ => panic!("expected Named"),
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-browser --test parse`
Expected: FAIL — `BrowseTarget::parse` not found

**Step 3: Add `hex` dependency**

In `crates/harmony-browser/Cargo.toml`, add under `[dependencies]`:
```toml
hex = "0.4"
```

**Step 4: Implement `BrowseTarget::parse`**

Add to `crates/harmony-browser/src/types.rs`:

```rust
use crate::BrowserError;

impl BrowseTarget {
    /// Parse user address bar input into a BrowseTarget.
    ///
    /// Formats:
    /// - `hmy:<64-char hex>` → CID lookup
    /// - `~<key_expr>` → live subscription (prefixed with `harmony/`)
    /// - anything else → named content (prefixed with `harmony/content/`)
    pub fn parse(input: &str) -> Result<Self, BrowserError> {
        let trimmed = input.trim();

        if let Some(hex_str) = trimmed.strip_prefix("hmy:") {
            let bytes = hex::decode(hex_str)
                .map_err(|_| BrowserError::InvalidCidHex(hex_str.to_string()))?;
            if bytes.len() != 32 {
                return Err(BrowserError::InvalidCidHex(format!(
                    "expected 32 bytes, got {}",
                    bytes.len()
                )));
            }
            let mut arr = [0u8; 32];
            arr.copy_from_slice(&bytes);
            let cid = ContentId::from_bytes(arr);
            Ok(Self::Cid(cid))
        } else if let Some(key_expr) = trimmed.strip_prefix('~') {
            Ok(Self::Subscribe(format!("harmony/{key_expr}")))
        } else {
            Ok(Self::Named(format!("harmony/content/{trimmed}")))
        }
    }
}
```

**Step 5: Run tests to verify they pass**

Run: `cargo test -p harmony-browser --test parse`
Expected: 5 tests PASS

**Step 6: Commit**

```bash
git add crates/harmony-browser/
git commit -m "feat(browser): add BrowseTarget::parse for address bar input"
```

---

### Task 4: Trust policy — `TrustPolicy` and `TrustDecision` resolution

**Files:**
- Create: `crates/harmony-browser/src/trust.rs`
- Test: `crates/harmony-browser/tests/trust.rs`

**Step 1: Write the failing tests**

Create `crates/harmony-browser/tests/trust.rs`:

```rust
use harmony_browser::{TrustDecision, TrustPolicy};

#[test]
fn unknown_author_gets_unknown_decision() {
    let policy = TrustPolicy::new();
    assert_eq!(policy.decide(None), TrustDecision::Unknown);
}

#[test]
fn identity_0_is_untrusted() {
    let policy = TrustPolicy::new();
    // Score with identity dimension = 0 (bits 0-1 = 0)
    assert_eq!(policy.decide(Some(0b00000000)), TrustDecision::Untrusted);
}

#[test]
fn identity_1_is_untrusted() {
    let policy = TrustPolicy::new();
    // Score with identity dimension = 1
    assert_eq!(policy.decide(Some(0b00000001)), TrustDecision::Untrusted);
}

#[test]
fn identity_2_is_preview() {
    let policy = TrustPolicy::new();
    // Score with identity dimension = 2
    assert_eq!(policy.decide(Some(0b00000010)), TrustDecision::Preview);
}

#[test]
fn identity_3_is_full_trust() {
    let policy = TrustPolicy::new();
    // Score with identity dimension = 3
    assert_eq!(policy.decide(Some(0b00000011)), TrustDecision::FullTrust);
}

#[test]
fn full_score_0xff_is_full_trust() {
    let policy = TrustPolicy::new();
    assert_eq!(policy.decide(Some(0xFF)), TrustDecision::FullTrust);
}

#[test]
fn custom_threshold_overrides_default() {
    let mut policy = TrustPolicy::new();
    // Require identity=3 for even Preview
    policy.set_preview_threshold(3);
    assert_eq!(policy.decide(Some(0b00000010)), TrustDecision::Untrusted);
    assert_eq!(policy.decide(Some(0b00000011)), TrustDecision::Preview);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-browser --test trust`
Expected: FAIL — `TrustPolicy` not found

**Step 3: Implement `TrustPolicy`**

Create `crates/harmony-browser/src/trust.rs`:

```rust
use crate::TrustDecision;

/// Maps trust scores to rendering decisions.
///
/// Uses the identity dimension (bits 0-1) of the 8-bit trust score
/// to determine how content should be rendered. This matches the
/// `resolveMediaTrust` logic in harmony-client's TrustGraphService.
pub struct TrustPolicy {
    /// Minimum identity value for Preview (default: 2).
    preview_threshold: u8,
    /// Minimum identity value for FullTrust (default: 3).
    full_trust_threshold: u8,
}

impl TrustPolicy {
    pub fn new() -> Self {
        Self {
            preview_threshold: 2,
            full_trust_threshold: 3,
        }
    }

    /// Decide how to render content based on the author's trust score.
    ///
    /// - `None` → `Unknown` (author not in trust network)
    /// - Identity 0-1 → `Untrusted`
    /// - Identity 2 → `Preview`
    /// - Identity 3 → `FullTrust`
    pub fn decide(&self, score: Option<u8>) -> TrustDecision {
        let score = match score {
            Some(s) => s,
            None => return TrustDecision::Unknown,
        };
        let identity = score & 0x3;
        if identity >= self.full_trust_threshold {
            TrustDecision::FullTrust
        } else if identity >= self.preview_threshold {
            TrustDecision::Preview
        } else {
            TrustDecision::Untrusted
        }
    }

    pub fn set_preview_threshold(&mut self, threshold: u8) {
        self.preview_threshold = threshold;
    }

    pub fn set_full_trust_threshold(&mut self, threshold: u8) {
        self.full_trust_threshold = threshold;
    }
}

impl Default for TrustPolicy {
    fn default() -> Self {
        Self::new()
    }
}
```

**Step 4: Update `lib.rs`**

```rust
pub mod error;
pub mod trust;
pub mod types;

pub use error::BrowserError;
pub use trust::TrustPolicy;
pub use types::*;
```

**Step 5: Run tests to verify they pass**

Run: `cargo test -p harmony-browser --test trust`
Expected: 7 tests PASS

**Step 6: Commit**

```bash
git add crates/harmony-browser/
git commit -m "feat(browser): add TrustPolicy for trust-to-rendering decisions"
```

---

### Task 5: Browser events and actions

**Files:**
- Create: `crates/harmony-browser/src/event.rs`
- Test: `crates/harmony-browser/tests/events.rs`

**Step 1: Write the failing tests**

Create `crates/harmony-browser/tests/events.rs`:

```rust
use harmony_browser::{BrowserAction, BrowserEvent, BrowseTarget, ResolvedContent, TrustDecision, MimeHint};
use harmony_content::ContentId;
use harmony_zenoh::SubscriptionId;

#[test]
fn navigate_event_holds_browse_target() {
    let cid = ContentId::for_blob(b"hello").unwrap();
    let event = BrowserEvent::Navigate(BrowseTarget::Cid(cid));
    match event {
        BrowserEvent::Navigate(BrowseTarget::Cid(c)) => assert_eq!(c, cid),
        _ => panic!("wrong variant"),
    }
}

#[test]
fn fetch_action_holds_cid() {
    let cid = ContentId::for_blob(b"test").unwrap();
    let action = BrowserAction::FetchContent { cid };
    match action {
        BrowserAction::FetchContent { cid: c } => assert_eq!(c, cid),
        _ => panic!("wrong variant"),
    }
}

#[test]
fn render_action_holds_resolved_static() {
    let cid = ContentId::for_blob(b"# Hello").unwrap();
    let resolved = ResolvedContent::Static {
        cid,
        mime: MimeHint::Markdown,
        data: b"# Hello".to_vec(),
        author: None,
        trust_level: TrustDecision::Unknown,
    };
    match resolved {
        ResolvedContent::Static { mime, trust_level, .. } => {
            assert_eq!(mime, MimeHint::Markdown);
            assert_eq!(trust_level, TrustDecision::Unknown);
        }
        _ => panic!("wrong variant"),
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-browser --test events`
Expected: FAIL — `BrowserEvent`, `BrowserAction`, `ResolvedContent` not found

**Step 3: Implement events and actions**

Create `crates/harmony-browser/src/event.rs`:

```rust
use harmony_content::ContentId;
use harmony_identity::Identity;
use harmony_zenoh::SubscriptionId;

use crate::types::{BrowseTarget, MimeHint, TrustDecision};

/// What the core resolves a navigation target into.
#[derive(Debug, Clone)]
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

/// Events fed into the BrowserCore state machine.
#[derive(Debug)]
pub enum BrowserEvent {
    /// User entered something in the address bar.
    Navigate(BrowseTarget),
    /// Content bytes arrived from network.
    ContentFetched { cid: ContentId, data: Vec<u8> },
    /// Pub/sub sample arrived.
    SubscriptionUpdate {
        sub_id: SubscriptionId,
        key_expr: String,
        payload: Vec<u8>,
    },
    /// User changed trust for an author.
    TrustUpdated { address: [u8; 16], score: u8 },
    /// User explicitly approved gated content.
    ApproveContent { cid: ContentId },
}

/// Actions emitted by the BrowserCore state machine.
#[derive(Debug)]
pub enum BrowserAction {
    /// Request content from network (caller does the I/O).
    FetchContent { cid: ContentId },
    /// Query a named resource via Zenoh.
    QueryNamed { key_expr: String },
    /// Subscribe to a key expression.
    Subscribe { key_expr: String },
    /// Unsubscribe from a key expression.
    Unsubscribe { sub_id: SubscriptionId },
    /// Deliver resolved content to the renderer.
    Render(ResolvedContent),
    /// Deliver a subscription update to the renderer.
    DeliverUpdate {
        sub_id: SubscriptionId,
        key_expr: String,
        payload: Vec<u8>,
    },
}
```

**Step 4: Update `lib.rs`**

```rust
pub mod error;
pub mod event;
pub mod trust;
pub mod types;

pub use error::BrowserError;
pub use event::*;
pub use trust::TrustPolicy;
pub use types::*;
```

**Step 5: Run tests to verify they pass**

Run: `cargo test -p harmony-browser --test events`
Expected: 3 tests PASS

**Step 6: Commit**

```bash
git add crates/harmony-browser/
git commit -m "feat(browser): add BrowserEvent, BrowserAction, ResolvedContent"
```

---

### Task 6: `BrowserCore` state machine — navigation and content fetch

**Files:**
- Create: `crates/harmony-browser/src/core.rs`
- Test: `crates/harmony-browser/tests/core_navigation.rs`

**Step 1: Write the failing tests**

Create `crates/harmony-browser/tests/core_navigation.rs`:

```rust
use harmony_browser::{BrowserAction, BrowserCore, BrowserEvent, BrowseTarget};
use harmony_content::ContentId;

fn make_core() -> BrowserCore {
    BrowserCore::new()
}

#[test]
fn navigate_to_cid_emits_fetch() {
    let mut core = make_core();
    let cid = ContentId::for_blob(b"hello").unwrap();
    let actions = core.handle_event(BrowserEvent::Navigate(BrowseTarget::Cid(cid)));
    assert_eq!(actions.len(), 1);
    match &actions[0] {
        BrowserAction::FetchContent { cid: c } => assert_eq!(*c, cid),
        other => panic!("expected FetchContent, got {:?}", other),
    }
}

#[test]
fn navigate_to_named_emits_query() {
    let mut core = make_core();
    let actions = core.handle_event(BrowserEvent::Navigate(
        BrowseTarget::Named("harmony/content/wiki/rust".into()),
    ));
    assert_eq!(actions.len(), 1);
    match &actions[0] {
        BrowserAction::QueryNamed { key_expr } => {
            assert_eq!(key_expr, "harmony/content/wiki/rust");
        }
        other => panic!("expected QueryNamed, got {:?}", other),
    }
}

#[test]
fn navigate_to_subscribe_emits_subscribe() {
    let mut core = make_core();
    let actions = core.handle_event(BrowserEvent::Navigate(
        BrowseTarget::Subscribe("harmony/presence/**".into()),
    ));
    assert_eq!(actions.len(), 1);
    match &actions[0] {
        BrowserAction::Subscribe { key_expr } => {
            assert_eq!(key_expr, "harmony/presence/**");
        }
        other => panic!("expected Subscribe, got {:?}", other),
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-browser --test core_navigation`
Expected: FAIL — `BrowserCore` not found

**Step 3: Implement `BrowserCore` (navigation only)**

Create `crates/harmony-browser/src/core.rs`:

```rust
use harmony_content::ContentId;

use crate::event::{BrowserAction, BrowserEvent, ResolvedContent};
use crate::trust::TrustPolicy;
use crate::types::BrowseTarget;

/// Sans-I/O browser state machine.
///
/// The caller feeds events (user actions, network responses) and
/// receives actions (fetch content, subscribe, render). The caller
/// is responsible for all I/O — this struct is pure logic.
pub struct BrowserCore {
    trust_policy: TrustPolicy,
    /// Scores indexed by 16-byte address hash.
    trust_scores: hashbrown::HashMap<[u8; 16], u8>,
    /// CIDs the user has explicitly approved for loading.
    approved: hashbrown::HashSet<ContentId>,
}

impl BrowserCore {
    pub fn new() -> Self {
        Self {
            trust_policy: TrustPolicy::new(),
            trust_scores: hashbrown::HashMap::new(),
            approved: hashbrown::HashSet::new(),
        }
    }

    pub fn with_trust_policy(mut self, policy: TrustPolicy) -> Self {
        self.trust_policy = policy;
        self
    }

    pub fn handle_event(&mut self, event: BrowserEvent) -> Vec<BrowserAction> {
        match event {
            BrowserEvent::Navigate(target) => self.handle_navigate(target),
            BrowserEvent::ContentFetched { cid, data } => self.handle_content_fetched(cid, data),
            BrowserEvent::SubscriptionUpdate { sub_id, key_expr, payload } => {
                vec![BrowserAction::DeliverUpdate { sub_id, key_expr, payload }]
            }
            BrowserEvent::TrustUpdated { address, score } => {
                self.trust_scores.insert(address, score);
                vec![]
            }
            BrowserEvent::ApproveContent { cid } => {
                self.approved.insert(cid);
                vec![BrowserAction::FetchContent { cid }]
            }
        }
    }

    fn handle_navigate(&mut self, target: BrowseTarget) -> Vec<BrowserAction> {
        match target {
            BrowseTarget::Cid(cid) => vec![BrowserAction::FetchContent { cid }],
            BrowseTarget::Named(key_expr) => vec![BrowserAction::QueryNamed { key_expr }],
            BrowseTarget::Subscribe(key_expr) => vec![BrowserAction::Subscribe { key_expr }],
        }
    }

    fn handle_content_fetched(&mut self, cid: ContentId, data: Vec<u8>) -> Vec<BrowserAction> {
        // Content resolution will be implemented in the next task.
        // For now, render as unknown-trust plain data.
        use crate::types::{MimeHint, TrustDecision};
        vec![BrowserAction::Render(ResolvedContent::Static {
            cid,
            mime: MimeHint::PlainText,
            data,
            author: None,
            trust_level: TrustDecision::Unknown,
        })]
    }
}

impl Default for BrowserCore {
    fn default() -> Self {
        Self::new()
    }
}
```

**Step 4: Update `lib.rs`**

```rust
pub mod core;
pub mod error;
pub mod event;
pub mod trust;
pub mod types;

pub use crate::core::BrowserCore;
pub use error::BrowserError;
pub use event::*;
pub use trust::TrustPolicy;
pub use types::*;
```

**Step 5: Add `hashbrown` dependency**

In `crates/harmony-browser/Cargo.toml`, add:
```toml
hashbrown = { workspace = true }
```

**Step 6: Run tests to verify they pass**

Run: `cargo test -p harmony-browser --test core_navigation`
Expected: 3 tests PASS

**Step 7: Commit**

```bash
git add crates/harmony-browser/
git commit -m "feat(browser): add BrowserCore state machine with navigation"
```

---

### Task 7: Content resolution — MIME detection and trust-gated rendering

**Files:**
- Modify: `crates/harmony-browser/src/core.rs`
- Test: `crates/harmony-browser/tests/core_content.rs`

This task wires up the content pipeline: when `ContentFetched` arrives, the core detects the MIME type from the bundle's inline metadata, looks up the author's trust score, and emits a `Render` action with the appropriate `TrustDecision`.

**Step 1: Write the failing tests**

Create `crates/harmony-browser/tests/core_content.rs`:

```rust
use harmony_browser::{
    BrowserAction, BrowserCore, BrowserEvent, MimeHint, ResolvedContent, TrustDecision,
    TrustPolicy,
};
use harmony_content::{BlobStore, BundleBuilder, ContentId, MemoryBlobStore};

/// Helper: build a content bundle with inline metadata.
fn build_test_bundle(data: &[u8], mime: [u8; 8]) -> (ContentId, Vec<u8>) {
    let mut store = MemoryBlobStore::new();
    let blob_cid = store.insert(data).unwrap();

    let mut builder = BundleBuilder::new();
    builder.add(blob_cid);
    builder.with_metadata(data.len() as u64, 1, 1000, mime);
    let (bundle_bytes, bundle_cid) = builder.build().unwrap();
    (bundle_cid, bundle_bytes)
}

#[test]
fn content_fetched_detects_markdown_mime() {
    let mut core = BrowserCore::new();
    let (cid, data) = build_test_bundle(b"# Hello", *b"text/md\0");

    let actions = core.handle_event(BrowserEvent::ContentFetched { cid, data });
    assert_eq!(actions.len(), 1);
    match &actions[0] {
        BrowserAction::Render(ResolvedContent::Static { mime, .. }) => {
            assert_eq!(*mime, MimeHint::Markdown);
        }
        other => panic!("expected Render, got {:?}", other),
    }
}

#[test]
fn content_fetched_with_no_trust_score_returns_unknown() {
    let mut core = BrowserCore::new();
    let (cid, data) = build_test_bundle(b"# Hello", *b"text/md\0");

    let actions = core.handle_event(BrowserEvent::ContentFetched { cid, data });
    match &actions[0] {
        BrowserAction::Render(ResolvedContent::Static { trust_level, .. }) => {
            assert_eq!(*trust_level, TrustDecision::Unknown);
        }
        other => panic!("expected Render, got {:?}", other),
    }
}

#[test]
fn content_fetched_plain_blob_renders_as_plain_text() {
    let mut core = BrowserCore::new();
    let data = b"just plain bytes, not a bundle";
    let cid = ContentId::for_blob(data).unwrap();

    let actions = core.handle_event(BrowserEvent::ContentFetched {
        cid,
        data: data.to_vec(),
    });
    match &actions[0] {
        BrowserAction::Render(ResolvedContent::Static { mime, .. }) => {
            assert_eq!(*mime, MimeHint::PlainText);
        }
        other => panic!("expected Render, got {:?}", other),
    }
}

#[test]
fn trust_update_affects_subsequent_content() {
    let mut core = BrowserCore::new();
    let author_address = [0xAA; 16];

    // Set trust: identity=3 (full trust)
    core.handle_event(BrowserEvent::TrustUpdated {
        address: author_address,
        score: 0b00000011, // identity=3
    });

    // Simulate content from that author
    // (For MVP, author attribution comes from a separate mechanism;
    //  this test verifies the trust_scores map is populated)
    assert_eq!(
        core.trust_policy().decide(core.trust_score(&author_address)),
        TrustDecision::FullTrust,
    );
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-browser --test core_content`
Expected: FAIL — methods not found

**Step 3: Update `BrowserCore` content resolution**

In `crates/harmony-browser/src/core.rs`, replace `handle_content_fetched` and add helper methods:

```rust
    pub fn trust_policy(&self) -> &TrustPolicy {
        &self.trust_policy
    }

    pub fn trust_score(&self, address: &[u8; 16]) -> Option<u8> {
        self.trust_scores.get(address).copied()
    }

    fn handle_content_fetched(&mut self, cid: ContentId, data: Vec<u8>) -> Vec<BrowserAction> {
        use harmony_content::{parse_bundle, CidType};
        use crate::types::MimeHint;

        // Try to interpret as a bundle with inline metadata
        let (mime, content_data) = match cid.cid_type() {
            CidType::Bundle(_) => {
                match parse_bundle(&data) {
                    Ok(children) => {
                        // Look for InlineMetadata CID to get MIME hint
                        let mime = children.iter().find_map(|child| {
                            if matches!(child.cid_type(), CidType::InlineMetadata) {
                                child.parse_inline_metadata().ok().map(|(_, _, _, m)| {
                                    MimeHint::from_mime_bytes(m)
                                })
                            } else {
                                None
                            }
                        }).unwrap_or(MimeHint::PlainText);
                        (mime, data)
                    }
                    Err(_) => (MimeHint::PlainText, data),
                }
            }
            _ => (MimeHint::PlainText, data),
        };

        // Trust decision (no author attribution in MVP — defaults to Unknown)
        let trust_level = TrustDecision::Unknown;

        vec![BrowserAction::Render(ResolvedContent::Static {
            cid,
            mime,
            data: content_data,
            author: None,
            trust_level,
        })]
    }
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-browser --test core_content`
Expected: 4 tests PASS

**Step 5: Commit**

```bash
git add crates/harmony-browser/
git commit -m "feat(browser): content resolution with MIME detection and trust policy"
```

---

### Task 8: Subscription management

**Files:**
- Modify: `crates/harmony-browser/src/core.rs`
- Test: `crates/harmony-browser/tests/core_subscriptions.rs`

**Step 1: Write the failing tests**

Create `crates/harmony-browser/tests/core_subscriptions.rs`:

```rust
use harmony_browser::{BrowserAction, BrowserCore, BrowserEvent, BrowseTarget};
use harmony_zenoh::SubscriptionId;

#[test]
fn subscribe_tracks_active_subscription() {
    let mut core = BrowserCore::new();
    core.handle_event(BrowserEvent::Navigate(
        BrowseTarget::Subscribe("harmony/presence/**".into()),
    ));
    assert_eq!(core.active_subscriptions().len(), 1);
    assert!(core.active_subscriptions().contains("harmony/presence/**"));
}

#[test]
fn duplicate_subscribe_does_not_add_twice() {
    let mut core = BrowserCore::new();
    core.handle_event(BrowserEvent::Navigate(
        BrowseTarget::Subscribe("harmony/presence/**".into()),
    ));
    core.handle_event(BrowserEvent::Navigate(
        BrowseTarget::Subscribe("harmony/presence/**".into()),
    ));
    assert_eq!(core.active_subscriptions().len(), 1);
}

#[test]
fn subscription_update_emits_deliver() {
    let mut core = BrowserCore::new();
    core.handle_event(BrowserEvent::Navigate(
        BrowseTarget::Subscribe("harmony/presence/**".into()),
    ));

    let sub_id = *core.active_subscription_ids().iter().next().unwrap();
    let actions = core.handle_event(BrowserEvent::SubscriptionUpdate {
        sub_id,
        key_expr: "harmony/presence/alice".into(),
        payload: b"online".to_vec(),
    });

    assert_eq!(actions.len(), 1);
    match &actions[0] {
        BrowserAction::DeliverUpdate { key_expr, payload, .. } => {
            assert_eq!(key_expr, "harmony/presence/alice");
            assert_eq!(payload, b"online");
        }
        other => panic!("expected DeliverUpdate, got {:?}", other),
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-browser --test core_subscriptions`
Expected: FAIL — `active_subscriptions` and `active_subscription_ids` not found

**Step 3: Add subscription tracking to `BrowserCore`**

Add fields to `BrowserCore` and implement the methods. In `crates/harmony-browser/src/core.rs`:

Add to struct fields:
```rust
    /// Active subscriptions: key_expr → synthetic subscription ID.
    subscriptions: hashbrown::HashMap<String, SubscriptionId>,
    next_sub_id: u64,
```

Update `new()`:
```rust
    pub fn new() -> Self {
        Self {
            trust_policy: TrustPolicy::new(),
            trust_scores: hashbrown::HashMap::new(),
            approved: hashbrown::HashSet::new(),
            subscriptions: hashbrown::HashMap::new(),
            next_sub_id: 1,
        }
    }
```

Add methods:
```rust
    pub fn active_subscriptions(&self) -> hashbrown::HashSet<&str> {
        self.subscriptions.keys().map(|s| s.as_str()).collect()
    }

    pub fn active_subscription_ids(&self) -> hashbrown::HashSet<SubscriptionId> {
        self.subscriptions.values().copied().collect()
    }
```

Update `handle_navigate` for `Subscribe`:
```rust
    BrowseTarget::Subscribe(key_expr) => {
        if self.subscriptions.contains_key(&key_expr) {
            return vec![];
        }
        let sub_id = SubscriptionId::from_raw(self.next_sub_id);
        self.next_sub_id += 1;
        self.subscriptions.insert(key_expr.clone(), sub_id);
        vec![BrowserAction::Subscribe { key_expr }]
    }
```

Note: `SubscriptionId::from_raw` is `pub(crate)` in harmony-zenoh and gated behind `#[cfg(test)]`. For the browser crate, we need a way to create synthetic IDs. Add a newtype wrapper or use a raw u64. The simplest approach: define our own `BrowserSubId` type in `types.rs`:

```rust
/// Browser-local subscription identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BrowserSubId(pub u64);
```

Then use `BrowserSubId` instead of `SubscriptionId` in `BrowserCore` and events. Update `BrowserEvent::SubscriptionUpdate` and `BrowserAction::DeliverUpdate` to use `BrowserSubId`.

**Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-browser --test core_subscriptions`
Expected: 3 tests PASS

**Step 5: Commit**

```bash
git add crates/harmony-browser/
git commit -m "feat(browser): subscription tracking in BrowserCore"
```

---

### Task 9: Approve gated content flow

**Files:**
- Test: `crates/harmony-browser/tests/core_approve.rs`
- Modify: `crates/harmony-browser/src/core.rs` (if needed)

**Step 1: Write the failing tests**

Create `crates/harmony-browser/tests/core_approve.rs`:

```rust
use harmony_browser::{BrowserAction, BrowserCore, BrowserEvent};
use harmony_content::ContentId;

#[test]
fn approve_content_marks_cid_and_emits_fetch() {
    let mut core = BrowserCore::new();
    let cid = ContentId::for_blob(b"gated image data").unwrap();

    let actions = core.handle_event(BrowserEvent::ApproveContent { cid });
    assert_eq!(actions.len(), 1);
    match &actions[0] {
        BrowserAction::FetchContent { cid: c } => assert_eq!(*c, cid),
        other => panic!("expected FetchContent, got {:?}", other),
    }
}

#[test]
fn approved_cid_is_remembered() {
    let mut core = BrowserCore::new();
    let cid = ContentId::for_blob(b"gated image").unwrap();

    core.handle_event(BrowserEvent::ApproveContent { cid });
    assert!(core.is_approved(&cid));
}

#[test]
fn unapproved_cid_is_not_remembered() {
    let core = BrowserCore::new();
    let cid = ContentId::for_blob(b"nope").unwrap();
    assert!(!core.is_approved(&cid));
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-browser --test core_approve`
Expected: FAIL — `is_approved` not found

**Step 3: Add `is_approved` method**

In `crates/harmony-browser/src/core.rs`, add:

```rust
    pub fn is_approved(&self, cid: &ContentId) -> bool {
        self.approved.contains(cid)
    }
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-browser --test core_approve`
Expected: 3 tests PASS

**Step 5: Commit**

```bash
git add crates/harmony-browser/
git commit -m "feat(browser): content approval tracking"
```

---

### Task 10: Full integration test and workspace validation

**Files:**
- Test: `crates/harmony-browser/tests/integration.rs`

**Step 1: Write the integration test**

Create `crates/harmony-browser/tests/integration.rs`:

```rust
//! End-to-end test: navigate → fetch → render with trust.

use harmony_browser::{
    BrowserAction, BrowserCore, BrowserEvent, BrowseTarget, MimeHint,
    ResolvedContent, TrustDecision, TrustPolicy,
};
use harmony_content::{BlobStore, BundleBuilder, ContentId, MemoryBlobStore};

#[test]
fn full_navigation_flow() {
    // 1. User types a named path
    let mut core = BrowserCore::new();
    let actions = core.handle_event(BrowserEvent::Navigate(
        BrowseTarget::Named("harmony/content/wiki/rust".into()),
    ));
    assert!(matches!(&actions[0], BrowserAction::QueryNamed { .. }));

    // 2. Simulate: query resolved to a CID, caller fetched the bundle
    let mut store = MemoryBlobStore::new();
    let blob_cid = store.insert(b"# Rust Programming").unwrap();
    let mut builder = BundleBuilder::new();
    builder.add(blob_cid);
    builder.with_metadata(19, 1, 1000, *b"text/md\0");
    let (bundle_bytes, bundle_cid) = builder.build().unwrap();

    // 3. Content arrives
    let actions = core.handle_event(BrowserEvent::ContentFetched {
        cid: bundle_cid,
        data: bundle_bytes,
    });

    // 4. Should render as markdown with Unknown trust (no author)
    assert_eq!(actions.len(), 1);
    match &actions[0] {
        BrowserAction::Render(ResolvedContent::Static {
            mime, trust_level, ..
        }) => {
            assert_eq!(*mime, MimeHint::Markdown);
            assert_eq!(*trust_level, TrustDecision::Unknown);
        }
        other => panic!("expected Render, got {:?}", other),
    }
}

#[test]
fn subscription_flow() {
    let mut core = BrowserCore::new();

    // 1. User subscribes
    let actions = core.handle_event(BrowserEvent::Navigate(
        BrowseTarget::Subscribe("harmony/presence/**".into()),
    ));
    assert!(matches!(&actions[0], BrowserAction::Subscribe { .. }));
    assert_eq!(core.active_subscriptions().len(), 1);

    // 2. Update arrives
    let sub_id = *core.active_subscription_ids().iter().next().unwrap();
    let actions = core.handle_event(BrowserEvent::SubscriptionUpdate {
        sub_id,
        key_expr: "harmony/presence/bob".into(),
        payload: b"online".to_vec(),
    });
    assert!(matches!(&actions[0], BrowserAction::DeliverUpdate { .. }));
}

#[test]
fn parse_and_navigate() {
    let mut core = BrowserCore::new();
    let target = BrowseTarget::parse("~presence/**").unwrap();
    let actions = core.handle_event(BrowserEvent::Navigate(target));
    assert!(matches!(&actions[0], BrowserAction::Subscribe { .. }));
}
```

**Step 2: Run the full test suite**

Run: `cargo test -p harmony-browser`
Expected: ALL tests pass (types + parse + trust + events + core_navigation + core_content + core_subscriptions + core_approve + integration)

**Step 3: Run workspace-wide checks**

Run: `cargo test --workspace`
Expected: All 365+ existing tests plus new harmony-browser tests pass

Run: `cargo clippy -p harmony-browser`
Expected: No warnings

**Step 4: Commit**

```bash
git add crates/harmony-browser/
git commit -m "feat(browser): integration tests for full navigation and subscription flows"
```

---

## Summary

| Task | What It Builds | Test Count |
|------|---------------|------------|
| 1 | Crate scaffold | 0 (compile check) |
| 2 | BrowseTarget, MimeHint, TrustDecision | 8 |
| 3 | BrowseTarget::parse | 5 |
| 4 | TrustPolicy | 7 |
| 5 | BrowserEvent, BrowserAction, ResolvedContent | 3 |
| 6 | BrowserCore navigation | 3 |
| 7 | Content resolution + MIME detection | 4 |
| 8 | Subscription management | 3 |
| 9 | Content approval | 3 |
| 10 | Integration tests | 3 |
| **Total** | | **~39 tests** |

After completing these 10 tasks, the `harmony-browser` crate will be a fully functional sans-I/O state machine that:
- Parses address bar input into navigation targets
- Resolves content bundles with MIME detection
- Applies trust-based rendering decisions
- Manages live pub/sub subscriptions
- Tracks user content approvals

The Tauri shell (Phase 2) and privacy panel (Phase 4) would be planned separately as follow-on work.
