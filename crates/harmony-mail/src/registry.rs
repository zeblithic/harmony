//! Name registration linking human-readable names to Harmony identities.
//!
//! A name registration is dual-signed: the user proves they own the identity,
//! and the gateway co-signs to prove it authorized the registration under its
//! domain.

use ed25519_dalek::{Signature, Verifier};
use harmony_identity::identity::{PUBLIC_KEY_LENGTH, SIGNATURE_LENGTH};
use harmony_identity::{Identity, PrivateIdentity};

use crate::error::MailError;

/// A name registration linking a human-readable name to a Harmony identity.
///
/// Dual-signed: the user proves they own the identity, the gateway proves
/// it authorized the registration under its domain.
pub struct NameRegistration {
    /// The human-readable name (e.g. "alice").
    pub name: String,
    /// The namespace within the domain (e.g. "mail").
    pub namespace: String,
    /// The gateway domain (e.g. "example.harmony").
    pub domain: String,
    /// The user's public identity.
    pub identity: Identity,
    /// Unix timestamp of registration.
    pub registered_at: u64,
    /// User's Ed25519 signature over the canonical bytes.
    pub user_signature: [u8; SIGNATURE_LENGTH],
    /// Gateway's Ed25519 signature over the canonical bytes.
    pub domain_signature: [u8; SIGNATURE_LENGTH],
}

impl NameRegistration {
    /// Build the canonical byte representation used for signing.
    ///
    /// Format: `name_len(u16) + name + namespace_len(u16) + namespace +
    /// domain_len(u16) + domain + identity_public_bytes(64) + registered_at(u64 BE)`
    fn canonical_bytes(
        name: &str,
        namespace: &str,
        domain: &str,
        identity: &Identity,
        registered_at: u64,
    ) -> Vec<u8> {
        let name_bytes = name.as_bytes();
        let namespace_bytes = namespace.as_bytes();
        let domain_bytes = domain.as_bytes();
        let identity_bytes = identity.to_public_bytes();

        let capacity = 2 + name_bytes.len()
            + 2 + namespace_bytes.len()
            + 2 + domain_bytes.len()
            + PUBLIC_KEY_LENGTH
            + 8;

        let mut buf = Vec::with_capacity(capacity);

        buf.extend_from_slice(&(name_bytes.len() as u16).to_be_bytes());
        buf.extend_from_slice(name_bytes);

        buf.extend_from_slice(&(namespace_bytes.len() as u16).to_be_bytes());
        buf.extend_from_slice(namespace_bytes);

        buf.extend_from_slice(&(domain_bytes.len() as u16).to_be_bytes());
        buf.extend_from_slice(domain_bytes);

        buf.extend_from_slice(&identity_bytes);

        buf.extend_from_slice(&registered_at.to_be_bytes());

        buf
    }

    /// Create a new name registration, signed by both user and gateway.
    pub fn new(
        name: String,
        namespace: String,
        domain: String,
        identity: Identity,
        registered_at: u64,
        user_private: &PrivateIdentity,
        gateway_private: &PrivateIdentity,
    ) -> Self {
        let canonical =
            Self::canonical_bytes(&name, &namespace, &domain, &identity, registered_at);

        let user_signature = user_private.sign(&canonical);
        let domain_signature = gateway_private.sign(&canonical);

        Self {
            name,
            namespace,
            domain,
            identity,
            registered_at,
            user_signature,
            domain_signature,
        }
    }

    /// Verify the user's signature over the canonical registration data.
    pub fn verify_user_signature(&self) -> bool {
        let canonical = Self::canonical_bytes(
            &self.name,
            &self.namespace,
            &self.domain,
            &self.identity,
            self.registered_at,
        );
        let sig = Signature::from_bytes(&self.user_signature);
        self.identity.verifying_key.verify(&canonical, &sig).is_ok()
    }

    /// Verify the gateway's signature over the canonical registration data.
    pub fn verify_domain_signature(&self, gateway_identity: &Identity) -> bool {
        let canonical = Self::canonical_bytes(
            &self.name,
            &self.namespace,
            &self.domain,
            &self.identity,
            self.registered_at,
        );
        let sig = Signature::from_bytes(&self.domain_signature);
        gateway_identity
            .verifying_key
            .verify(&canonical, &sig)
            .is_ok()
    }

    /// Serialize to bytes for announce payload or storage.
    ///
    /// Format: `canonical_bytes + user_signature(64) + domain_signature(64)`
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Self::canonical_bytes(
            &self.name,
            &self.namespace,
            &self.domain,
            &self.identity,
            self.registered_at,
        );
        buf.reserve(SIGNATURE_LENGTH * 2);
        buf.extend_from_slice(&self.user_signature);
        buf.extend_from_slice(&self.domain_signature);
        buf
    }

    /// Deserialize from bytes produced by [`to_bytes`](Self::to_bytes).
    pub fn from_bytes(data: &[u8]) -> Result<Self, MailError> {
        let mut pos = 0;

        let name = Self::read_length_prefixed_string(data, &mut pos, "name")?;
        let namespace = Self::read_length_prefixed_string(data, &mut pos, "namespace")?;
        let domain = Self::read_length_prefixed_string(data, &mut pos, "domain")?;

        // Identity: 64 bytes
        if data.len() < pos + PUBLIC_KEY_LENGTH {
            return Err(MailError::Truncated {
                expected: (pos + PUBLIC_KEY_LENGTH) - data.len(),
            });
        }
        let identity = Identity::from_public_bytes(&data[pos..pos + PUBLIC_KEY_LENGTH])
            .map_err(|_| MailError::InvalidIdentity)?;
        pos += PUBLIC_KEY_LENGTH;

        // registered_at: u64 big-endian
        if data.len() < pos + 8 {
            return Err(MailError::Truncated {
                expected: (pos + 8) - data.len(),
            });
        }
        let registered_at = u64::from_be_bytes(data[pos..pos + 8].try_into().unwrap());
        pos += 8;

        // user_signature: 64 bytes
        if data.len() < pos + SIGNATURE_LENGTH {
            return Err(MailError::Truncated {
                expected: (pos + SIGNATURE_LENGTH) - data.len(),
            });
        }
        let mut user_signature = [0u8; SIGNATURE_LENGTH];
        user_signature.copy_from_slice(&data[pos..pos + SIGNATURE_LENGTH]);
        pos += SIGNATURE_LENGTH;

        // domain_signature: 64 bytes
        if data.len() < pos + SIGNATURE_LENGTH {
            return Err(MailError::Truncated {
                expected: (pos + SIGNATURE_LENGTH) - data.len(),
            });
        }
        let mut domain_signature = [0u8; SIGNATURE_LENGTH];
        domain_signature.copy_from_slice(&data[pos..pos + SIGNATURE_LENGTH]);
        pos += SIGNATURE_LENGTH;

        if pos != data.len() {
            return Err(MailError::TrailingBytes {
                count: data.len() - pos,
            });
        }

        Ok(Self {
            name,
            namespace,
            domain,
            identity,
            registered_at,
            user_signature,
            domain_signature,
        })
    }

    /// Read a u16-length-prefixed UTF-8 string from `data` at `pos`, advancing `pos`.
    fn read_length_prefixed_string(
        data: &[u8],
        pos: &mut usize,
        field: &'static str,
    ) -> Result<String, MailError> {
        if data.len() < *pos + 2 {
            return Err(MailError::Truncated {
                expected: (*pos + 2) - data.len(),
            });
        }
        let len = u16::from_be_bytes(data[*pos..*pos + 2].try_into().unwrap()) as usize;
        *pos += 2;

        if data.len() < *pos + len {
            return Err(MailError::Truncated {
                expected: (*pos + len) - data.len(),
            });
        }
        let s = std::str::from_utf8(&data[*pos..*pos + len])
            .map_err(|_| MailError::InvalidUtf8 { field })?;
        *pos += len;

        Ok(s.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::OsRng;

    #[test]
    fn name_registration_sign_verify_roundtrip() {
        let user = PrivateIdentity::generate(&mut OsRng);
        let gateway = PrivateIdentity::generate(&mut OsRng);

        let reg = NameRegistration::new(
            "alice".to_string(),
            "mail".to_string(),
            "example.harmony".to_string(),
            user.public_identity().clone(),
            1_700_000_000,
            &user,
            &gateway,
        );

        assert!(reg.verify_user_signature());
        assert!(reg.verify_domain_signature(gateway.public_identity()));

        // Wrong gateway identity should fail domain verification
        let other = PrivateIdentity::generate(&mut OsRng);
        assert!(!reg.verify_domain_signature(other.public_identity()));
    }

    #[test]
    fn tampered_name_fails_verification() {
        let user = PrivateIdentity::generate(&mut OsRng);
        let gateway = PrivateIdentity::generate(&mut OsRng);

        let mut reg = NameRegistration::new(
            "alice".to_string(),
            "mail".to_string(),
            "example.harmony".to_string(),
            user.public_identity().clone(),
            1_700_000_000,
            &user,
            &gateway,
        );

        // Tamper with the name after signing
        reg.name = "mallory".to_string();

        assert!(!reg.verify_user_signature());
        assert!(!reg.verify_domain_signature(gateway.public_identity()));
    }

    #[test]
    fn serialization_roundtrip() {
        let user = PrivateIdentity::generate(&mut OsRng);
        let gateway = PrivateIdentity::generate(&mut OsRng);

        let reg = NameRegistration::new(
            "bob".to_string(),
            "chat".to_string(),
            "harmony.local".to_string(),
            user.public_identity().clone(),
            1_700_000_042,
            &user,
            &gateway,
        );

        let bytes = reg.to_bytes();
        let restored = NameRegistration::from_bytes(&bytes).expect("deserialization failed");

        assert_eq!(restored.name, "bob");
        assert_eq!(restored.namespace, "chat");
        assert_eq!(restored.domain, "harmony.local");
        assert_eq!(restored.registered_at, 1_700_000_042);
        assert_eq!(
            restored.identity.to_public_bytes(),
            reg.identity.to_public_bytes()
        );
        assert_eq!(restored.user_signature, reg.user_signature);
        assert_eq!(restored.domain_signature, reg.domain_signature);

        // Signatures should still verify after roundtrip
        assert!(restored.verify_user_signature());
        assert!(restored.verify_domain_signature(gateway.public_identity()));
    }

    #[test]
    fn trailing_bytes_rejected() {
        let user = PrivateIdentity::generate(&mut OsRng);
        let gateway = PrivateIdentity::generate(&mut OsRng);

        let reg = NameRegistration::new(
            "bob".to_string(),
            "chat".to_string(),
            "harmony.local".to_string(),
            user.public_identity().clone(),
            1_700_000_042,
            &user,
            &gateway,
        );

        let mut bytes = reg.to_bytes();
        bytes.push(0xFF); // append garbage

        match NameRegistration::from_bytes(&bytes) {
            Err(MailError::TrailingBytes { count: 1 }) => {}
            Err(other) => panic!("expected TrailingBytes {{ count: 1 }}, got {other:?}"),
            Ok(_) => panic!("expected error, got Ok"),
        }
    }
}
