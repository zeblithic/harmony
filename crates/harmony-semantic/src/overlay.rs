// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Overlay merge logic for multi-sidecar content.
//!
//! Multiple sidecars can target the same content CID (e.g. the original
//! uploader's metadata, AI-generated tags, and user-contributed tags).
//! At query time, Oluo merges them into a single coherent metadata view.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use crate::fingerprint::ModelFingerprint;
use crate::metadata::{PrivacyTier, SidecarMetadata};
use crate::sidecar::SidecarHeader;

/// Merge two sidecar metadata records into one.
///
/// `base` is the earlier/primary record, `overlay` is applied on top.
///
/// Merge rules:
/// - `privacy_tier`: most restrictive (max value via `Ord`)
/// - `created_at`: earliest timestamp (min value)
/// - `content_type`: first non-null (base wins if both set)
/// - `language`: concatenate unique values separated by `, `
/// - `geo`: latest wins (overlay wins if both set)
/// - `description`: concatenate with ` | ` separator
/// - `tags`: union (deduplicate)
/// - `refs`: union (deduplicate, comparing `[u8; 32]` directly)
/// - `source_device`: first non-null (base wins)
/// - `ext`: union merge, key conflicts -> overlay wins
pub fn merge_metadata(base: &SidecarMetadata, overlay: &SidecarMetadata) -> SidecarMetadata {
    SidecarMetadata {
        privacy_tier: merge_privacy(base.privacy_tier, overlay.privacy_tier),
        created_at: merge_created_at(base.created_at, overlay.created_at),
        content_type: first_non_null(&base.content_type, &overlay.content_type),
        language: merge_language(&base.language, &overlay.language),
        geo: merge_geo(base.geo, overlay.geo),
        description: merge_description(&base.description, &overlay.description),
        tags: merge_tags(&base.tags, &overlay.tags),
        refs: merge_refs(&base.refs, &overlay.refs),
        source_device: first_non_null(&base.source_device, &overlay.source_device),
        ext: merge_ext(&base.ext, &overlay.ext),
    }
}

/// Select the sidecar header matching the blessed model fingerprint.
///
/// If multiple headers match, prefers the latest (last in the slice,
/// since we don't have `created_at` on headers themselves).
/// Returns `None` if no header matches.
pub fn select_embedding_header<'a>(
    headers: &[&'a SidecarHeader],
    blessed_fingerprint: &ModelFingerprint,
) -> Option<&'a SidecarHeader> {
    headers
        .iter()
        .rev()
        .find(|h| h.fingerprint == *blessed_fingerprint)
        .copied()
}

// --- internal helpers ---

/// Most restrictive privacy tier (max by `Ord`).
fn merge_privacy(a: Option<PrivacyTier>, b: Option<PrivacyTier>) -> Option<PrivacyTier> {
    match (a, b) {
        (Some(pa), Some(pb)) => Some(pa.max(pb)),
        (Some(p), None) | (None, Some(p)) => Some(p),
        (None, None) => None,
    }
}

/// Earliest creation timestamp (min).
fn merge_created_at(a: Option<u64>, b: Option<u64>) -> Option<u64> {
    match (a, b) {
        (Some(ta), Some(tb)) => Some(ta.min(tb)),
        (Some(t), None) | (None, Some(t)) => Some(t),
        (None, None) => None,
    }
}

/// First non-null string (base wins).
fn first_non_null(base: &Option<String>, overlay: &Option<String>) -> Option<String> {
    match (base, overlay) {
        (Some(b), _) => Some(b.clone()),
        (None, Some(o)) => Some(o.clone()),
        (None, None) => None,
    }
}

/// Concatenate unique language tags separated by `, `.
fn merge_language(a: &Option<String>, b: &Option<String>) -> Option<String> {
    match (a, b) {
        (Some(la), Some(lb)) => {
            // Split both on `, ` to handle already-multilingual values,
            // then deduplicate while preserving order.
            let mut seen = Vec::new();
            for tag in la.split(", ").chain(lb.split(", ")) {
                let tag = tag.trim();
                if !tag.is_empty() && !seen.contains(&tag) {
                    seen.push(tag);
                }
            }
            Some(seen.join(", "))
        }
        (Some(l), None) | (None, Some(l)) => Some(l.clone()),
        (None, None) => None,
    }
}

/// Latest geo wins (overlay wins if both set).
fn merge_geo(base: Option<(f64, f64)>, overlay: Option<(f64, f64)>) -> Option<(f64, f64)> {
    match (base, overlay) {
        (_, Some(g)) => Some(g),
        (Some(g), None) => Some(g),
        (None, None) => None,
    }
}

/// Concatenate descriptions with ` | ` separator.
fn merge_description(a: &Option<String>, b: &Option<String>) -> Option<String> {
    match (a, b) {
        (Some(da), Some(db)) => {
            let mut result = String::with_capacity(da.len() + 3 + db.len());
            result.push_str(da);
            result.push_str(" | ");
            result.push_str(db);
            Some(result)
        }
        (Some(d), None) | (None, Some(d)) => Some(d.clone()),
        (None, None) => None,
    }
}

/// Union of tags, deduplicated.
fn merge_tags(a: &Option<Vec<String>>, b: &Option<Vec<String>>) -> Option<Vec<String>> {
    match (a, b) {
        (Some(ta), Some(tb)) => {
            let mut merged = ta.clone();
            for tag in tb {
                if !merged.iter().any(|t| t == tag) {
                    merged.push(tag.clone());
                }
            }
            Some(merged)
        }
        (Some(t), None) | (None, Some(t)) => Some(t.clone()),
        (None, None) => None,
    }
}

/// Union of refs, deduplicated by comparing `[u8; 32]` directly.
fn merge_refs(a: &Option<Vec<[u8; 32]>>, b: &Option<Vec<[u8; 32]>>) -> Option<Vec<[u8; 32]>> {
    match (a, b) {
        (Some(ra), Some(rb)) => {
            let mut merged = ra.clone();
            for r in rb {
                if !merged.contains(r) {
                    merged.push(*r);
                }
            }
            Some(merged)
        }
        (Some(r), None) | (None, Some(r)) => Some(r.clone()),
        (None, None) => None,
    }
}

/// Union merge of extension maps; key conflicts -> overlay wins.
fn merge_ext(
    a: &Option<BTreeMap<String, Vec<u8>>>,
    b: &Option<BTreeMap<String, Vec<u8>>>,
) -> Option<BTreeMap<String, Vec<u8>>> {
    match (a, b) {
        (Some(ea), Some(eb)) => {
            let mut merged = ea.clone();
            for (k, v) in eb {
                merged.insert(k.clone(), v.clone());
            }
            Some(merged)
        }
        (Some(e), None) | (None, Some(e)) => Some(e.clone()),
        (None, None) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;
    use alloc::vec;
    use alloc::vec::Vec;

    #[test]
    fn merge_privacy_most_restrictive() {
        let base = SidecarMetadata {
            privacy_tier: Some(PrivacyTier::PublicDurable),
            ..SidecarMetadata::default()
        };
        let overlay = SidecarMetadata {
            privacy_tier: Some(PrivacyTier::EncryptedDurable),
            ..SidecarMetadata::default()
        };
        let merged = merge_metadata(&base, &overlay);
        assert_eq!(merged.privacy_tier, Some(PrivacyTier::EncryptedDurable));
    }

    #[test]
    fn merge_created_at_earliest() {
        let base = SidecarMetadata {
            created_at: Some(1000),
            ..SidecarMetadata::default()
        };
        let overlay = SidecarMetadata {
            created_at: Some(500),
            ..SidecarMetadata::default()
        };
        let merged = merge_metadata(&base, &overlay);
        assert_eq!(merged.created_at, Some(500));
    }

    #[test]
    fn merge_tags_union() {
        let base = SidecarMetadata {
            tags: Some(vec!["a".to_string(), "b".to_string()]),
            ..SidecarMetadata::default()
        };
        let overlay = SidecarMetadata {
            tags: Some(vec!["b".to_string(), "c".to_string()]),
            ..SidecarMetadata::default()
        };
        let merged = merge_metadata(&base, &overlay);
        let tags = merged.tags.expect("tags should be Some");
        assert_eq!(tags.len(), 3);
        assert!(tags.contains(&"a".to_string()));
        assert!(tags.contains(&"b".to_string()));
        assert!(tags.contains(&"c".to_string()));
    }

    #[test]
    fn merge_description_concatenates() {
        let base = SidecarMetadata {
            description: Some("foo".to_string()),
            ..SidecarMetadata::default()
        };
        let overlay = SidecarMetadata {
            description: Some("bar".to_string()),
            ..SidecarMetadata::default()
        };
        let merged = merge_metadata(&base, &overlay);
        assert_eq!(merged.description, Some("foo | bar".to_string()));
    }

    #[test]
    fn merge_content_type_first_non_null() {
        let base = SidecarMetadata {
            content_type: None,
            ..SidecarMetadata::default()
        };
        let overlay = SidecarMetadata {
            content_type: Some("image/jpeg".to_string()),
            ..SidecarMetadata::default()
        };
        let merged = merge_metadata(&base, &overlay);
        assert_eq!(merged.content_type, Some("image/jpeg".to_string()));
    }

    #[test]
    fn merge_geo_latest_wins() {
        let base = SidecarMetadata {
            geo: Some((1.0, 2.0)),
            ..SidecarMetadata::default()
        };
        let overlay = SidecarMetadata {
            geo: Some((3.0, 4.0)),
            ..SidecarMetadata::default()
        };
        let merged = merge_metadata(&base, &overlay);
        assert_eq!(merged.geo, Some((3.0, 4.0)));
    }

    #[test]
    fn merge_refs_union_dedup() {
        let ref1 = [0x11; 32];
        let ref2 = [0x22; 32];
        let ref3 = [0x33; 32];

        let base = SidecarMetadata {
            refs: Some(vec![ref1, ref2]),
            ..SidecarMetadata::default()
        };
        let overlay = SidecarMetadata {
            refs: Some(vec![ref2, ref3]),
            ..SidecarMetadata::default()
        };
        let merged = merge_metadata(&base, &overlay);
        let refs = merged.refs.expect("refs should be Some");
        assert_eq!(refs.len(), 3);
        assert!(refs.contains(&ref1));
        assert!(refs.contains(&ref2));
        assert!(refs.contains(&ref3));
    }

    #[test]
    fn select_embedding_matching_fingerprint() {
        use crate::fingerprint::model_fingerprint;

        let blessed = model_fingerprint("blessed-model");
        let other = model_fingerprint("other-model");

        let header_other = SidecarHeader {
            fingerprint: other,
            target_cid: [0u8; 32],
            tier1: [0u8; 8],
            tier2: [0u8; 16],
            tier3: [0u8; 32],
            tier4: [0u8; 64],
            tier5: [0u8; 128],
        };

        let header_blessed_early = SidecarHeader {
            fingerprint: blessed,
            target_cid: [0u8; 32],
            tier1: [1u8; 8],
            tier2: [0u8; 16],
            tier3: [0u8; 32],
            tier4: [0u8; 64],
            tier5: [0u8; 128],
        };

        let header_blessed_late = SidecarHeader {
            fingerprint: blessed,
            target_cid: [0u8; 32],
            tier1: [2u8; 8],
            tier2: [0u8; 16],
            tier3: [0u8; 32],
            tier4: [0u8; 64],
            tier5: [0u8; 128],
        };

        let headers: Vec<&SidecarHeader> =
            vec![&header_other, &header_blessed_early, &header_blessed_late];

        let selected = select_embedding_header(&headers, &blessed);
        assert!(selected.is_some());
        let selected = selected.unwrap();
        // Should pick the latest matching (last in slice) — tier1 = [2u8; 8]
        assert_eq!(selected.tier1, [2u8; 8]);
        assert_eq!(selected.fingerprint, blessed);

        // No match returns None
        let unrelated = model_fingerprint("unrelated-model");
        let none_result = select_embedding_header(&headers, &unrelated);
        assert!(none_result.is_none());
    }
}
