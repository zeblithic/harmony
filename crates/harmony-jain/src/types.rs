//! Core types for content lifecycle management.

use harmony_content::ContentId;
use harmony_roxy::catalog::ContentCategory;
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

/// Metadata record for a piece of content stored on this node.
#[derive(Debug, Clone)]
pub struct ContentRecord {
    /// Content identifier.
    pub cid: ContentId,
    /// Size of the content in bytes.
    pub size_bytes: u64,
    /// Category of the content (music, video, text, etc.).
    pub content_type: ContentCategory,
    /// How the content arrived on this node.
    pub origin: ContentOrigin,
    /// Sensitivity classification.
    pub sensitivity: Sensitivity,
    /// Timestamp (seconds since epoch) when content was first stored.
    pub stored_at: f64,
    /// Timestamp (seconds since epoch) of last access.
    pub last_accessed: f64,
    /// Total number of times the content has been accessed.
    pub access_count: u64,
    /// Number of known replicas on the network.
    pub replica_count: u8,
    /// Whether the content is pinned (exempt from eviction).
    pub pinned: bool,
    /// Whether the content has an active license.
    pub licensed: bool,
    /// Whether the local backing data is missing and a fetch is in progress.
    ///
    /// Set by [`reconcile`](super::JainEngine::reconcile) when backing data
    /// is absent. While true, `staleness_score` returns FRESH to prevent
    /// burn/archive recommendations. Cleared when new content is stored for
    /// this CID.
    pub pending_local_repair: bool,
}

/// A staleness score clamped to the unit interval [0.0, 1.0].
///
/// 0.0 means completely fresh, 1.0 means completely stale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StalenessScore(f64);

impl StalenessScore {
    /// Completely fresh content.
    pub const FRESH: Self = StalenessScore(0.0);
    /// Completely stale content.
    pub const STALE: Self = StalenessScore(1.0);

    /// Create a new staleness score, clamping to [0.0, 1.0].
    ///
    /// NaN is treated as maximally stale (1.0) to prevent corrupted
    /// timestamps from hiding content from cleanup.
    pub fn new(value: f64) -> Self {
        if value.is_nan() {
            return Self::STALE;
        }
        StalenessScore(value.clamp(0.0, 1.0))
    }

    /// Return the inner f64 value.
    pub fn value(self) -> f64 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use harmony_content::ContentFlags;

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

    #[test]
    fn staleness_score_clamps_to_unit_range() {
        assert_eq!(StalenessScore::new(-0.5).value(), 0.0);
        assert_eq!(StalenessScore::new(1.5).value(), 1.0);
        assert!((StalenessScore::new(0.42).value() - 0.42).abs() < f64::EPSILON);
    }

    #[test]
    fn staleness_score_nan_becomes_stale() {
        let score = StalenessScore::new(f64::NAN);
        assert!(
            (score.value() - 1.0).abs() < f64::EPSILON,
            "NaN should become STALE (1.0), got {}",
            score.value()
        );
    }

    #[test]
    fn staleness_score_fresh_and_stale() {
        assert!((StalenessScore::FRESH.value() - 0.0).abs() < f64::EPSILON);
        assert!((StalenessScore::STALE.value() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn content_record_tracks_basic_fields() {
        let cid = ContentId::for_blob(b"hello world", ContentFlags::default()).unwrap();
        let record = ContentRecord {
            cid,
            size_bytes: 11,
            content_type: ContentCategory::Text,
            origin: ContentOrigin::SelfCreated,
            sensitivity: Sensitivity::Public,
            stored_at: 1000.0,
            last_accessed: 2000.0,
            access_count: 5,
            replica_count: 3,
            pinned: false,
            licensed: true,
            pending_local_repair: false,
        };
        assert_eq!(record.cid, cid);
        assert_eq!(record.size_bytes, 11);
        assert_eq!(record.origin, ContentOrigin::SelfCreated);
        assert_eq!(record.sensitivity, Sensitivity::Public);
        assert_eq!(record.access_count, 5);
        assert_eq!(record.replica_count, 3);
        assert!(!record.pinned);
        assert!(record.licensed);
    }
}
