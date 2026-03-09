//! Zenoh key expression patterns for catalog discovery.

use alloc::string::String;
use alloc::vec::Vec;
use harmony_content::ContentId;
use serde::{Deserialize, Serialize};

/// Content category for organizing published content.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum ContentCategory {
    /// Audio content (songs, albums, podcasts).
    Music = 0,
    /// Video content (films, clips, livestreams).
    Video = 1,
    /// Written content (articles, books, posts).
    Text = 2,
    /// Image content (photos, illustrations, art).
    Image = 3,
    /// Software content (applications, libraries, tools).
    Software = 4,
    /// Dataset content (structured data, research data).
    Dataset = 5,
    /// A bundle of multiple content items.
    Bundle = 6,
}

impl ContentCategory {
    /// Returns the lowercase string representation of the category.
    pub fn as_str(&self) -> &'static str {
        match self {
            ContentCategory::Music => "music",
            ContentCategory::Video => "video",
            ContentCategory::Text => "text",
            ContentCategory::Image => "image",
            ContentCategory::Software => "software",
            ContentCategory::Dataset => "dataset",
            ContentCategory::Bundle => "bundle",
        }
    }
}

/// An artist's public profile for catalog discovery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtistProfile {
    /// Artist's 128-bit address hash.
    pub address_hash: [u8; 16],
    /// Human-readable display name.
    pub display_name: String,
    /// Short biography or description.
    pub bio: String,
    /// Optional CID of the artist's avatar image.
    pub avatar_cid: Option<ContentId>,
    /// CIDs of license manifests the artist has published.
    pub manifest_cids: Vec<ContentId>,
}

/// Build a Zenoh key expression for a specific catalog entry.
///
/// Returns `"roxy/catalog/{hex_hash}/{category}/{manifest_id}"`.
pub fn catalog_key(artist_hash: &[u8; 16], category: ContentCategory, manifest_id: &str) -> String {
    debug_assert!(
        !manifest_id.contains('/') && !manifest_id.contains('*'),
        "manifest_id must not contain Zenoh metacharacters"
    );
    alloc::format!(
        "roxy/catalog/{}/{}/{}",
        hex::encode(artist_hash),
        category.as_str(),
        manifest_id
    )
}

/// Build a Zenoh key expression for an artist's metadata entry.
///
/// Returns `"roxy/catalog/{hex_hash}/meta"`.
pub fn meta_key(artist_hash: &[u8; 16]) -> String {
    alloc::format!("roxy/catalog/{}/meta", hex::encode(artist_hash))
}

/// Build a Zenoh key expression for a consumer's license grant.
///
/// Returns `"roxy/license/{hex_hash}/{manifest_id}"`.
pub fn license_key(consumer_hash: &[u8; 16], manifest_id: &str) -> String {
    debug_assert!(
        !manifest_id.contains('/') && !manifest_id.contains('*'),
        "manifest_id must not contain Zenoh metacharacters"
    );
    alloc::format!(
        "roxy/license/{}/{}",
        hex::encode(consumer_hash),
        manifest_id
    )
}

/// Build a Zenoh key expression for a UCAN revocation notice.
///
/// Returns `"roxy/revocation/{hex_hash}/{ucan_hash}"`.
pub fn revocation_key(artist_hash: &[u8; 16], ucan_hash: &str) -> String {
    debug_assert!(
        !ucan_hash.contains('/') && !ucan_hash.contains('*'),
        "ucan_hash must not contain Zenoh metacharacters"
    );
    alloc::format!("roxy/revocation/{}/{}", hex::encode(artist_hash), ucan_hash)
}

/// Build a Zenoh subscription pattern for an artist's content.
///
/// If `category` is `Some`, returns `"roxy/catalog/{hex_hash}/{category}/**"`.
/// If `None`, returns `"roxy/catalog/{hex_hash}/**"`.
pub fn artist_content_pattern(artist_hash: &[u8; 16], category: Option<ContentCategory>) -> String {
    let hex_hash = hex::encode(artist_hash);
    match category {
        Some(cat) => alloc::format!("roxy/catalog/{}/{}/**", hex_hash, cat.as_str()),
        None => alloc::format!("roxy/catalog/{}/**", hex_hash),
    }
}

/// Build a Zenoh subscription pattern for all content of a given category.
///
/// Returns `"roxy/catalog/*/{category}/**"`.
pub fn global_content_pattern(category: ContentCategory) -> String {
    alloc::format!("roxy/catalog/*/{}/**", category.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_key_for_content() {
        let artist_hash = [0xABu8; 16];
        let key = catalog_key(&artist_hash, ContentCategory::Music, "deadbeef");
        assert_eq!(
            key,
            "roxy/catalog/abababababababababababababababab/music/deadbeef"
        );
    }

    #[test]
    fn catalog_key_for_meta() {
        let artist_hash = [0x01u8; 16];
        let key = meta_key(&artist_hash);
        assert_eq!(key, "roxy/catalog/01010101010101010101010101010101/meta");
    }

    #[test]
    fn license_key_for_consumer() {
        let consumer_hash = [0xCDu8; 16];
        let key = license_key(&consumer_hash, "aabbccdd");
        assert_eq!(
            key,
            "roxy/license/cdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcd/aabbccdd"
        );
    }

    #[test]
    fn revocation_key_test() {
        let artist_hash = [0xEFu8; 16];
        let key = revocation_key(&artist_hash, "11223344");
        assert_eq!(
            key,
            "roxy/revocation/efefefefefefefefefefefefefefefef/11223344"
        );
    }

    #[test]
    fn subscription_pattern_all_music_from_artist() {
        let artist_hash = [0xABu8; 16];
        let pattern = artist_content_pattern(&artist_hash, Some(ContentCategory::Music));
        assert_eq!(
            pattern,
            "roxy/catalog/abababababababababababababababab/music/**"
        );
    }

    #[test]
    fn subscription_pattern_all_from_artist() {
        let artist_hash = [0xABu8; 16];
        let pattern = artist_content_pattern(&artist_hash, None);
        assert_eq!(pattern, "roxy/catalog/abababababababababababababababab/**");
    }

    #[test]
    fn subscription_pattern_all_music() {
        let pattern = global_content_pattern(ContentCategory::Music);
        assert_eq!(pattern, "roxy/catalog/*/music/**");
    }

    #[test]
    fn content_category_serialization_round_trip() {
        for cat in [
            ContentCategory::Music,
            ContentCategory::Video,
            ContentCategory::Text,
            ContentCategory::Image,
            ContentCategory::Software,
            ContentCategory::Dataset,
            ContentCategory::Bundle,
        ] {
            let bytes = postcard::to_allocvec(&cat).unwrap();
            let decoded: ContentCategory = postcard::from_bytes(&bytes).unwrap();
            assert_eq!(cat, decoded);
        }
    }

    #[test]
    fn artist_profile_serialization_round_trip() {
        let profile = ArtistProfile {
            address_hash: [0xABu8; 16],
            display_name: alloc::string::String::from("Test Artist"),
            bio: alloc::string::String::from("A test artist"),
            avatar_cid: None,
            manifest_cids: alloc::vec![],
        };
        let bytes = postcard::to_allocvec(&profile).unwrap();
        let decoded: ArtistProfile = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(profile.display_name, decoded.display_name);
        assert_eq!(profile.address_hash, decoded.address_hash);
        assert_eq!(profile.bio, decoded.bio);
        assert_eq!(profile.avatar_cid, decoded.avatar_cid);
        assert_eq!(profile.manifest_cids, decoded.manifest_cids);
    }
}
