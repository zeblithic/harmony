use harmony_crypto::hash;

use crate::error::ReticulumError;

/// A Reticulum destination name and its derived hashes.
///
/// The destination name is built from an app name and optional aspects,
/// joined with dots: `"app_name.aspect1.aspect2"`. Individual components
/// must not contain dots (matching Python's `expand_name` validation).
///
/// Two hashes are derived:
/// - `name_hash = SHA256("app_name.aspect1.aspect2")[:10]`
/// - `destination_hash = SHA256(name_hash || identity_address_hash)[:16]`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DestinationName {
    /// The full dot-separated name, if known (not available from wire).
    full_name: Option<String>,
    /// SHA256(full_name)[:10] — always present.
    name_hash: [u8; hash::NAME_HASH_LENGTH],
}

impl DestinationName {
    /// Create from an app name and aspects, validating that no component contains dots.
    ///
    /// # Example
    /// ```
    /// use harmony_reticulum::destination::DestinationName;
    /// let dest = DestinationName::from_name("lxmf", &["delivery"]).unwrap();
    /// ```
    pub fn from_name(app_name: &str, aspects: &[&str]) -> Result<Self, ReticulumError> {
        if app_name.contains('.') || app_name.is_empty() {
            return Err(ReticulumError::InvalidDestinationName);
        }
        for aspect in aspects {
            if aspect.contains('.') || aspect.is_empty() {
                return Err(ReticulumError::InvalidDestinationName);
            }
        }

        let mut full_name = app_name.to_string();
        for aspect in aspects {
            full_name.push('.');
            full_name.push_str(aspect);
        }

        let name_hash = hash::name_hash(full_name.as_bytes());

        Ok(Self {
            full_name: Some(full_name),
            name_hash,
        })
    }

    /// Create from a raw 10-byte name hash (parsed from wire).
    ///
    /// The original name string is not recoverable from the hash.
    pub fn from_name_hash(name_hash: [u8; hash::NAME_HASH_LENGTH]) -> Self {
        Self {
            full_name: None,
            name_hash,
        }
    }

    /// The 10-byte name hash.
    pub fn name_hash(&self) -> &[u8; hash::NAME_HASH_LENGTH] {
        &self.name_hash
    }

    /// The full name string, if constructed from `from_name`.
    pub fn full_name(&self) -> Option<&str> {
        self.full_name.as_deref()
    }

    /// Compute the 16-byte destination hash for a specific identity.
    ///
    /// `destination_hash = SHA256(name_hash || identity_address_hash)[:16]`
    pub fn destination_hash(
        &self,
        identity_address_hash: &[u8; hash::TRUNCATED_HASH_LENGTH],
    ) -> [u8; hash::TRUNCATED_HASH_LENGTH] {
        let mut material = Vec::with_capacity(hash::NAME_HASH_LENGTH + hash::TRUNCATED_HASH_LENGTH);
        material.extend_from_slice(&self.name_hash);
        material.extend_from_slice(identity_address_hash);
        hash::truncated_hash(&material)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_name() {
        let dest = DestinationName::from_name("lxmf", &["delivery"]).unwrap();
        assert_eq!(dest.full_name(), Some("lxmf.delivery"));

        let expected_hash = hash::name_hash(b"lxmf.delivery");
        assert_eq!(dest.name_hash(), &expected_hash);
    }

    #[test]
    fn name_no_aspects() {
        let dest = DestinationName::from_name("myapp", &[]).unwrap();
        assert_eq!(dest.full_name(), Some("myapp"));

        let expected_hash = hash::name_hash(b"myapp");
        assert_eq!(dest.name_hash(), &expected_hash);
    }

    #[test]
    fn name_multiple_aspects() {
        let dest = DestinationName::from_name("app", &["one", "two", "three"]).unwrap();
        assert_eq!(dest.full_name(), Some("app.one.two.three"));

        let expected_hash = hash::name_hash(b"app.one.two.three");
        assert_eq!(dest.name_hash(), &expected_hash);
    }

    #[test]
    fn dot_in_app_name_rejected() {
        let result = DestinationName::from_name("bad.name", &[]);
        assert!(matches!(result, Err(ReticulumError::InvalidDestinationName)));
    }

    #[test]
    fn dot_in_aspect_rejected() {
        let result = DestinationName::from_name("app", &["bad.aspect"]);
        assert!(matches!(result, Err(ReticulumError::InvalidDestinationName)));
    }

    #[test]
    fn empty_app_name_rejected() {
        let result = DestinationName::from_name("", &[]);
        assert!(matches!(result, Err(ReticulumError::InvalidDestinationName)));
    }

    #[test]
    fn empty_aspect_rejected() {
        let result = DestinationName::from_name("app", &[""]);
        assert!(matches!(result, Err(ReticulumError::InvalidDestinationName)));
    }

    #[test]
    fn from_name_hash_roundtrip() {
        let original = DestinationName::from_name("lxmf", &["delivery"]).unwrap();
        let from_wire = DestinationName::from_name_hash(*original.name_hash());

        assert_eq!(from_wire.name_hash(), original.name_hash());
        assert_eq!(from_wire.full_name(), None); // name not recoverable
    }

    #[test]
    fn destination_hash_computation() {
        let dest = DestinationName::from_name("test", &["app"]).unwrap();
        let identity_hash = [0xAA; 16];

        let dest_hash = dest.destination_hash(&identity_hash);
        assert_eq!(dest_hash.len(), 16);

        // Manually compute expected value
        let mut material = Vec::new();
        material.extend_from_slice(dest.name_hash());
        material.extend_from_slice(&identity_hash);
        let expected = hash::truncated_hash(&material);

        assert_eq!(dest_hash, expected);
    }

    #[test]
    fn destination_hash_deterministic() {
        let dest = DestinationName::from_name("lxmf", &["delivery"]).unwrap();
        let id_hash = [0x42; 16];

        let h1 = dest.destination_hash(&id_hash);
        let h2 = dest.destination_hash(&id_hash);
        assert_eq!(h1, h2);
    }

    #[test]
    fn different_identities_different_dest_hashes() {
        let dest = DestinationName::from_name("lxmf", &["delivery"]).unwrap();

        let h1 = dest.destination_hash(&[0xAA; 16]);
        let h2 = dest.destination_hash(&[0xBB; 16]);
        assert_ne!(h1, h2);
    }

    #[test]
    fn different_names_different_dest_hashes() {
        let d1 = DestinationName::from_name("app1", &[]).unwrap();
        let d2 = DestinationName::from_name("app2", &[]).unwrap();
        let id_hash = [0xCC; 16];

        assert_ne!(d1.destination_hash(&id_hash), d2.destination_hash(&id_hash));
    }
}
