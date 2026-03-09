//! Events, actions, and decision types for the content lifecycle engine.

use harmony_content::ContentId;
use harmony_roxy::catalog::ContentCategory;
use serde::{Deserialize, Serialize};

use crate::types::{ContentOrigin, Sensitivity, StalenessScore};

/// Events that drive the content lifecycle state machine.
#[derive(Debug, Clone)]
pub enum ContentEvent {
    /// New content has been stored on this node.
    Stored {
        /// Content identifier.
        cid: ContentId,
        /// Size in bytes.
        size_bytes: u64,
        /// Category of the content.
        content_type: ContentCategory,
        /// How the content arrived.
        origin: ContentOrigin,
        /// Sensitivity classification.
        sensitivity: Sensitivity,
        /// Wall-clock timestamp when stored.
        timestamp: f64,
    },
    /// Content was accessed (read/streamed).
    Accessed {
        /// Content identifier.
        cid: ContentId,
        /// Wall-clock timestamp of the access.
        timestamp: f64,
    },
    /// Content was deleted from this node.
    Deleted {
        /// Content identifier.
        cid: ContentId,
    },
    /// Content was pinned (exempt from eviction).
    Pinned {
        /// Content identifier.
        cid: ContentId,
    },
    /// Content was unpinned.
    Unpinned {
        /// Content identifier.
        cid: ContentId,
    },
    /// A license was granted for this content.
    LicenseGranted {
        /// Content identifier.
        cid: ContentId,
    },
    /// A license expired for this content.
    LicenseExpired {
        /// Content identifier.
        cid: ContentId,
    },
    /// The replica count for this content changed.
    ReplicaChanged {
        /// Content identifier.
        cid: ContentId,
        /// Updated replica count.
        new_count: u8,
    },
}

/// Candidate content being evaluated for ingestion.
#[derive(Debug, Clone)]
pub struct IngestCandidate {
    /// Content identifier.
    pub cid: ContentId,
    /// Size in bytes.
    pub size_bytes: u64,
    /// Category of the content.
    pub content_type: ContentCategory,
    /// How the content arrived.
    pub origin: ContentOrigin,
    /// Sensitivity classification.
    pub sensitivity: Sensitivity,
}

/// Decision on whether to ingest a content candidate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IngestDecision {
    /// Index the content in the catalog and store the blob.
    IndexAndStore,
    /// Store the blob without indexing.
    StoreOnly,
    /// Reject the content.
    Reject {
        /// Why the content was rejected.
        reason: RejectReason,
    },
}

/// Reason a content candidate was rejected for ingestion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RejectReason {
    /// The storage budget would be exceeded.
    StorageBudgetExceeded,
    /// The content is a duplicate of existing content.
    DuplicateContent,
}

/// Decision on whether content should be shared in a given social context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterDecision {
    /// Content may be shared.
    Allow,
    /// Content must not be shared.
    Block,
    /// User confirmation is required before sharing.
    Confirm,
}

/// Actions emitted by the engine for the caller to execute.
#[derive(Debug, Clone)]
pub enum JainAction {
    /// Recommend burning (deleting) stale content.
    RecommendBurn {
        /// Details of the cleanup recommendation.
        recommendation: CleanupRecommendation,
    },
    /// Recommend archiving content to cold storage.
    RecommendArchive {
        /// Details of the cleanup recommendation.
        recommendation: CleanupRecommendation,
    },
    /// Recommend deduplicating content.
    RecommendDedup {
        /// CID to keep.
        keep: ContentId,
        /// CID to burn.
        burn: ContentId,
    },
    /// Query Oluo (network layer) for additional information.
    QueryOluo {
        /// Content identifier to query about.
        cid: ContentId,
        /// Hint about what to look for.
        query_hint: QueryHint,
    },
    /// Content needs more replicas.
    RepairNeeded {
        /// Content identifier.
        cid: ContentId,
        /// Current number of replicas.
        current_replicas: u8,
        /// Desired number of replicas.
        desired: u8,
    },
    /// A health alert for the node operator.
    HealthAlert {
        /// Kind of alert.
        alert: HealthAlertKind,
    },
}

/// Details of a cleanup recommendation (burn or archive).
#[derive(Debug, Clone)]
pub struct CleanupRecommendation {
    /// Content identifier.
    pub cid: ContentId,
    /// Why the cleanup is recommended.
    pub reason: CleanupReason,
    /// Staleness score at the time of recommendation.
    pub staleness: StalenessScore,
    /// Bytes that would be recovered.
    pub space_recovered_bytes: u64,
    /// Confidence in the recommendation (0.0..1.0).
    pub confidence: f64,
}

/// Reason content was recommended for cleanup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CleanupReason {
    /// Content is stale (high staleness score).
    Stale,
    /// An encrypted version of publicly available content.
    EncryptedVersionOfPublic,
    /// Node is over its storage budget.
    OverStorageBudget,
}

/// Hint for what to look for when querying the network.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QueryHint {
    /// Look for a public equivalent of encrypted content.
    FindPublicEquivalent,
    /// Look for local duplicates.
    FindLocalDuplicates,
}

/// Kinds of health alerts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthAlertKind {
    /// Storage usage is approaching capacity.
    StorageNearFull {
        /// Usage percentage times 100 (e.g. 8500 = 85.00%).
        used_percent_x100: u32,
    },
    /// Records exist in the catalog but have no backing data on disk.
    StaleReconciliation {
        /// Number of records without backing data.
        records_without_backing: u32,
    },
    /// Some content has fewer replicas than desired.
    ReplicaDeficit {
        /// Number of under-replicated records.
        affected_records: u32,
    },
}

/// A snapshot entry used during reconciliation.
#[derive(Debug, Clone)]
pub struct SnapshotEntry {
    /// Content identifier.
    pub cid: ContentId,
    /// Size of the content in bytes.
    pub size_bytes: u64,
    /// Whether the content blob exists on disk.
    pub exists_on_disk: bool,
}

/// Summary health report for the node.
#[derive(Debug, Clone, Copy)]
pub struct HealthReport {
    /// Total number of content records tracked.
    pub total_records: u32,
    /// Total bytes of content stored.
    pub total_bytes: u64,
    /// Storage usage percentage times 100 (e.g. 8500 = 85.00%).
    pub storage_used_percent_x100: u32,
    /// Number of under-replicated records.
    pub under_replicated_count: u32,
    /// Number of stale records (above archive threshold).
    pub stale_count: u32,
    /// Number of pinned records.
    pub pinned_count: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ingest_decision_equality() {
        assert_eq!(IngestDecision::IndexAndStore, IngestDecision::IndexAndStore);
        assert_eq!(IngestDecision::StoreOnly, IngestDecision::StoreOnly);
        assert_eq!(
            IngestDecision::Reject {
                reason: RejectReason::StorageBudgetExceeded,
            },
            IngestDecision::Reject {
                reason: RejectReason::StorageBudgetExceeded,
            }
        );
        assert_eq!(
            IngestDecision::Reject {
                reason: RejectReason::DuplicateContent,
            },
            IngestDecision::Reject {
                reason: RejectReason::DuplicateContent,
            }
        );
        assert_ne!(IngestDecision::IndexAndStore, IngestDecision::StoreOnly);
        assert_ne!(
            IngestDecision::Reject {
                reason: RejectReason::StorageBudgetExceeded,
            },
            IngestDecision::Reject {
                reason: RejectReason::DuplicateContent,
            }
        );
    }

    #[test]
    fn cleanup_reason_serialization_round_trip() {
        for reason in [
            CleanupReason::Stale,
            CleanupReason::EncryptedVersionOfPublic,
            CleanupReason::OverStorageBudget,
        ] {
            let bytes = postcard::to_allocvec(&reason).unwrap();
            let decoded: CleanupReason = postcard::from_bytes(&bytes).unwrap();
            assert_eq!(reason, decoded);
        }
    }

    #[test]
    fn query_hint_serialization_round_trip() {
        for hint in [QueryHint::FindPublicEquivalent, QueryHint::FindLocalDuplicates] {
            let bytes = postcard::to_allocvec(&hint).unwrap();
            let decoded: QueryHint = postcard::from_bytes(&bytes).unwrap();
            assert_eq!(hint, decoded);
        }
    }

    #[test]
    fn reject_reason_serialization_round_trip() {
        for reason in [
            RejectReason::StorageBudgetExceeded,
            RejectReason::DuplicateContent,
        ] {
            let bytes = postcard::to_allocvec(&reason).unwrap();
            let decoded: RejectReason = postcard::from_bytes(&bytes).unwrap();
            assert_eq!(reason, decoded);
        }
    }
}
