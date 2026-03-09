//! Zenoh key expression patterns for Jain health and recommendation topics.

use alloc::string::String;

use crate::error::JainError;

/// Validate that a key expression segment contains no Zenoh metacharacters.
fn validate_segment(s: &str) -> Result<(), JainError> {
    if s.is_empty() || s.contains('/') || s.contains('*') {
        Err(JainError::InvalidKeySegment)
    } else {
        Ok(())
    }
}

/// Build a Zenoh key expression for a node's health topic.
///
/// Returns `"jain/health/{hex}"`.
pub fn health_key(node_hash: &[u8; 16]) -> String {
    alloc::format!("jain/health/{}", hex::encode(node_hash))
}

/// Build a Zenoh key expression for a user's recommendation topic.
///
/// Returns `"jain/recommend/{hex}"`.
pub fn recommend_key(user_hash: &[u8; 16]) -> String {
    alloc::format!("jain/recommend/{}", hex::encode(user_hash))
}

/// Build a Zenoh key expression for a specific action topic.
///
/// Returns `"jain/action/{hex}/{action_id}"`.
///
/// Returns `Err(JainError::InvalidKeySegment)` if `action_id` is empty or
/// contains Zenoh metacharacters (`/` or `*`).
pub fn action_key(user_hash: &[u8; 16], action_id: &str) -> Result<String, JainError> {
    validate_segment(action_id)?;
    Ok(alloc::format!(
        "jain/action/{}/{}",
        hex::encode(user_hash),
        action_id
    ))
}

/// Build a Zenoh key expression for a node's statistics topic.
///
/// Returns `"jain/stats/{hex}"`.
pub fn stats_key(node_hash: &[u8; 16]) -> String {
    alloc::format!("jain/stats/{}", hex::encode(node_hash))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn health_key_format() {
        let hash = [0xABu8; 16];
        assert_eq!(
            health_key(&hash),
            "jain/health/abababababababababababababababab"
        );
    }

    #[test]
    fn recommend_key_format() {
        let hash = [0xCDu8; 16];
        assert_eq!(
            recommend_key(&hash),
            "jain/recommend/cdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcd"
        );
    }

    #[test]
    fn action_key_format() {
        let hash = [0xEFu8; 16];
        let key = action_key(&hash, "abc123").unwrap();
        assert_eq!(key, "jain/action/efefefefefefefefefefefefefefefef/abc123");
    }

    #[test]
    fn stats_key_format() {
        let hash = [0x01u8; 16];
        assert_eq!(
            stats_key(&hash),
            "jain/stats/01010101010101010101010101010101"
        );
    }

    #[test]
    fn action_key_rejects_metacharacters() {
        let hash = [0xABu8; 16];
        assert!(action_key(&hash, "a/b").is_err());
        assert!(action_key(&hash, "a*b").is_err());
    }

    #[test]
    fn action_key_rejects_empty_segment() {
        let hash = [0xABu8; 16];
        assert!(action_key(&hash, "").is_err());
    }
}
