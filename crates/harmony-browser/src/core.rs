use harmony_content::cid::ContentId;

use crate::event::{BrowserAction, BrowserEvent, BrowserSubId, ResolvedContent};
use crate::trust::TrustPolicy;
use crate::types::{BrowseTarget, MimeHint, TrustDecision};

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
    /// Active subscriptions: key_expr -> synthetic subscription ID.
    subscriptions: hashbrown::HashMap<String, BrowserSubId>,
    next_sub_id: u64,
}

impl BrowserCore {
    pub fn new() -> Self {
        Self {
            trust_policy: TrustPolicy::new(),
            trust_scores: hashbrown::HashMap::new(),
            approved: hashbrown::HashSet::new(),
            subscriptions: hashbrown::HashMap::new(),
            next_sub_id: 1,
        }
    }

    pub fn with_trust_policy(mut self, policy: TrustPolicy) -> Self {
        self.trust_policy = policy;
        self
    }

    pub fn trust_policy(&self) -> &TrustPolicy {
        &self.trust_policy
    }

    pub fn trust_score(&self, address: &[u8; 16]) -> Option<u8> {
        self.trust_scores.get(address).copied()
    }

    pub fn is_approved(&self, cid: &ContentId) -> bool {
        self.approved.contains(cid)
    }

    pub fn active_subscriptions(&self) -> hashbrown::HashSet<&str> {
        self.subscriptions.keys().map(|s| s.as_str()).collect()
    }

    pub fn active_subscription_ids(&self) -> hashbrown::HashSet<BrowserSubId> {
        self.subscriptions.values().copied().collect()
    }

    pub fn handle_event(&mut self, event: BrowserEvent) -> Vec<BrowserAction> {
        match event {
            BrowserEvent::Navigate(target) => self.handle_navigate(target),
            BrowserEvent::ContentFetched { cid, data } => self.handle_content_fetched(cid, data),
            BrowserEvent::SubscriptionUpdate {
                sub_id,
                key_expr,
                payload,
            } => {
                vec![BrowserAction::DeliverUpdate {
                    sub_id,
                    key_expr,
                    payload,
                }]
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
            BrowseTarget::Subscribe(key_expr) => {
                if self.subscriptions.contains_key(&key_expr) {
                    return vec![];
                }
                let sub_id = BrowserSubId(self.next_sub_id);
                self.next_sub_id += 1;
                self.subscriptions.insert(key_expr.clone(), sub_id);
                vec![BrowserAction::Subscribe { key_expr }]
            }
        }
    }

    fn handle_content_fetched(&mut self, cid: ContentId, data: Vec<u8>) -> Vec<BrowserAction> {
        use harmony_content::bundle::parse_bundle;
        use harmony_content::cid::CidType;

        // Try to interpret as a bundle with inline metadata
        let mime = match cid.cid_type() {
            CidType::Bundle(_) => match parse_bundle(&data) {
                Ok(children) => children
                    .iter()
                    .find_map(|child| {
                        if child.cid_type() == CidType::InlineMetadata {
                            child
                                .parse_inline_metadata()
                                .ok()
                                .map(|(_, _, _, m)| MimeHint::from_mime_bytes(m))
                        } else {
                            None
                        }
                    })
                    .unwrap_or(MimeHint::PlainText),
                Err(_) => MimeHint::PlainText,
            },
            _ => MimeHint::PlainText,
        };

        // Trust decision (no author attribution in MVP — defaults to Unknown)
        let trust_level = TrustDecision::Unknown;

        vec![BrowserAction::Render(ResolvedContent::Static {
            cid,
            mime,
            data,
            author: None,
            trust_level,
        })]
    }
}

impl Default for BrowserCore {
    fn default() -> Self {
        Self::new()
    }
}
