//! Core types for content lifecycle management.

use serde::{Deserialize, Serialize};

/// How content arrived on this node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum ContentOrigin {
    /// Created locally by the node operator.
    SelfCreated = 0,
    /// Replicated from a peer for redundancy.
    PeerReplicated = 1,
    /// Explicitly downloaded by the operator.
    Downloaded = 2,
    /// Cached while forwarding to another node.
    CachedInTransit = 3,
}

/// Content sensitivity level, ordered from least to most sensitive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[repr(u8)]
pub enum Sensitivity {
    /// Publicly shareable content.
    Public = 0,
    /// Content intended for limited sharing.
    Private = 1,
    /// Personal or intimate content.
    Intimate = 2,
    /// Highly confidential content.
    Confidential = 3,
}

/// Social context for content sharing, ordered from most private to most public.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[repr(u8)]
pub enum SocialContext {
    /// Only the node operator.
    Private = 0,
    /// Close companions.
    Companion = 1,
    /// Broader social circle.
    Social = 2,
    /// Professional or public context.
    Professional = 3,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_origin_serialization_round_trip() {
        for origin in [
            ContentOrigin::SelfCreated,
            ContentOrigin::PeerReplicated,
            ContentOrigin::Downloaded,
            ContentOrigin::CachedInTransit,
        ] {
            let bytes = postcard::to_allocvec(&origin).unwrap();
            let decoded: ContentOrigin = postcard::from_bytes(&bytes).unwrap();
            assert_eq!(origin, decoded);
        }
    }

    #[test]
    fn sensitivity_ordering() {
        assert!(Sensitivity::Public < Sensitivity::Private);
        assert!(Sensitivity::Private < Sensitivity::Intimate);
        assert!(Sensitivity::Intimate < Sensitivity::Confidential);
    }

    #[test]
    fn sensitivity_serialization_round_trip() {
        for sensitivity in [
            Sensitivity::Public,
            Sensitivity::Private,
            Sensitivity::Intimate,
            Sensitivity::Confidential,
        ] {
            let bytes = postcard::to_allocvec(&sensitivity).unwrap();
            let decoded: Sensitivity = postcard::from_bytes(&bytes).unwrap();
            assert_eq!(sensitivity, decoded);
        }
    }

    #[test]
    fn social_context_ordering() {
        assert!(SocialContext::Private < SocialContext::Companion);
        assert!(SocialContext::Companion < SocialContext::Social);
        assert!(SocialContext::Social < SocialContext::Professional);
    }

    #[test]
    fn social_context_serialization_round_trip() {
        for context in [
            SocialContext::Private,
            SocialContext::Companion,
            SocialContext::Social,
            SocialContext::Professional,
        ] {
            let bytes = postcard::to_allocvec(&context).unwrap();
            let decoded: SocialContext = postcard::from_bytes(&bytes).unwrap();
            assert_eq!(context, decoded);
        }
    }
}
