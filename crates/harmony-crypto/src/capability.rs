//! Capability proof: prove possession of a shared secret by authenticating a
//! message under a purpose-bound key derived from that secret.
//!
//! ```text
//! capability_tag(secret, salt, info, message) =
//!     HMAC-SHA256( HKDF-SHA256(ikm = secret, salt, info) -> 32-byte MAC key,
//!                  message )
//! ```
//!
//! `salt` and `info` domain-separate the derived MAC key, so one secret yields
//! independent authenticators for different purposes. Verification is
//! constant-time. This is the generic kernel behind capability-based schemes
//! such as tokenless invites: the shared secret *is* the capability, and the
//! tag proves a party holds it without revealing it. Callers supply the
//! authenticated `message` bytes (any domain-specific preimage layout stays on
//! the caller's side).

use hmac::{Hmac, Mac};
use sha2::Sha256;
use subtle::ConstantTimeEq;

use crate::hkdf::DerivedKey;

type HmacSha256 = Hmac<Sha256>;

/// Length in bytes of a capability tag (an HMAC-SHA256 output).
pub const CAPABILITY_TAG_LEN: usize = 32;

/// Compute a capability proof tag over `message`.
///
/// Derives a purpose-bound MAC key from `secret` via HKDF-SHA256
/// (domain-separated by `salt` and `info`), then returns the HMAC-SHA256 tag
/// over `message`. The derived MAC key lives in a zeroize-on-drop buffer.
///
/// - `secret`: the shared secret / capability (HKDF input key material)
/// - `salt`: optional HKDF salt (HKDF defaults to 32 zero bytes when `None`)
/// - `info`: HKDF context label — the domain separator for the derived key
/// - `message`: the payload to authenticate
pub fn capability_tag(
    secret: &[u8],
    salt: Option<&[u8]>,
    info: &[u8],
    message: &[u8],
) -> [u8; CAPABILITY_TAG_LEN] {
    // 32 <= 255 * 32, so the derivation is infallible.
    let mac_key = DerivedKey::new(secret, salt, info, CAPABILITY_TAG_LEN)
        .expect("32 <= HKDF-SHA256 max output length");
    let mut mac = <HmacSha256 as Mac>::new_from_slice(mac_key.as_bytes())
        .expect("HMAC-SHA256 accepts any key length");
    mac.update(message);
    mac.finalize().into_bytes().into()
}

/// Constant-time verification of a capability tag.
///
/// Recomputes the expected tag from the same inputs and compares it against
/// `presented` in constant time (no early-exit on the first differing byte).
pub fn verify_capability_tag(
    secret: &[u8],
    salt: Option<&[u8]>,
    info: &[u8],
    message: &[u8],
    presented: &[u8; CAPABILITY_TAG_LEN],
) -> bool {
    let expected = capability_tag(secret, salt, info, message);
    expected.ct_eq(presented).into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    const INFO: &[u8] = b"open-join-auth";

    // Mirrors the harmony-client open-join capability preimage
    // (community_id ‖ joiner_identity_pub ‖ nonce ‖ timestamp_be) so this
    // golden vector doubles as the client's byte-preservation anchor.
    fn sample_message() -> Vec<u8> {
        let mut m = Vec::new();
        m.extend_from_slice(&[1u8; 16]); // community_id
        m.extend_from_slice(&[5u8; 64]); // joiner_identity_pub
        m.extend_from_slice(&[9u8; 16]); // nonce
        m.extend_from_slice(&1000u64.to_be_bytes()); // timestamp_ms
        m
    }

    #[test]
    fn valid_tag_round_trips() {
        let secret = [3u8; 32];
        let salt = [1u8; 16];
        let msg = sample_message();
        let tag = capability_tag(&secret, Some(&salt), INFO, &msg);
        assert!(verify_capability_tag(
            &secret,
            Some(&salt),
            INFO,
            &msg,
            &tag
        ));
    }

    #[test]
    fn wrong_secret_is_rejected() {
        let salt = [1u8; 16];
        let msg = sample_message();
        let tag = capability_tag(&[3u8; 32], Some(&salt), INFO, &msg);
        assert!(!verify_capability_tag(
            &[4u8; 32],
            Some(&salt),
            INFO,
            &msg,
            &tag
        ));
    }

    #[test]
    fn tampered_message_is_rejected() {
        let secret = [3u8; 32];
        let salt = [1u8; 16];
        let tag = capability_tag(&secret, Some(&salt), INFO, &sample_message());
        let mut tampered = sample_message();
        tampered[0] ^= 0x01;
        assert!(!verify_capability_tag(
            &secret,
            Some(&salt),
            INFO,
            &tampered,
            &tag
        ));
    }

    #[test]
    fn salt_and_info_domain_separate_the_key() {
        let secret = [3u8; 32];
        let msg = sample_message();
        let base = capability_tag(&secret, Some(&[1u8; 16]), INFO, &msg);
        // Different salt -> different tag.
        assert_ne!(base, capability_tag(&secret, Some(&[2u8; 16]), INFO, &msg));
        // Different info -> different tag.
        assert_ne!(
            base,
            capability_tag(&secret, Some(&[1u8; 16]), b"other-purpose", &msg)
        );
        // No salt -> different tag (salt is load-bearing).
        assert_ne!(base, capability_tag(&secret, None, INFO, &msg));
    }

    #[test]
    fn golden_vector_pins_the_construction() {
        // Fixed inputs -> fixed tag. Any change to the HKDF/HMAC construction,
        // key length, or byte handling breaks this. This exact value is also
        // asserted by harmony-client's open_join_auth byte-preservation test.
        let tag = capability_tag(&[3u8; 32], Some(&[1u8; 16]), INFO, &sample_message());
        assert_eq!(
            hex::encode(tag),
            "d17d12de45617c282087e41e3678505d6bbb11f0fa9defd55f1cfbc20a2c07ed"
        );
    }
}
