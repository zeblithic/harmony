use harmony_content::cid::ContentId;
use harmony_identity::Identity;

use crate::types::{BrowseTarget, MimeHint, TrustDecision};

/// Browser-local subscription identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BrowserSubId(pub u64);

/// What the core resolves a navigation target into.
#[derive(Debug, Clone)]
pub enum ResolvedContent {
    /// Static content bundle with metadata.
    Static {
        cid: ContentId,
        mime: MimeHint,
        data: Vec<u8>,
        author: Option<Box<Identity>>,
        trust_level: TrustDecision,
    },
    /// Live subscription producing a stream of updates.
    Live {
        key_expr: String,
        subscription_id: BrowserSubId,
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
        sub_id: BrowserSubId,
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
    Unsubscribe { sub_id: BrowserSubId },
    /// Deliver resolved content to the renderer.
    Render(ResolvedContent),
    /// Deliver a subscription update to the renderer.
    DeliverUpdate {
        sub_id: BrowserSubId,
        key_expr: String,
        payload: Vec<u8>,
    },
}
