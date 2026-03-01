use harmony_crypto::{hash, hkdf};
use harmony_identity::identity::PrivateIdentity;
use zeroize::Zeroize;

use crate::error::ReticulumError;

/// Reticulum IFAC salt (from Reticulum.py line 152).
const IFAC_SALT: [u8; 32] = [
    0xad, 0xf5, 0x4d, 0x88, 0x2c, 0x9a, 0x9b, 0x80,
    0x77, 0x1e, 0xb4, 0x99, 0x5d, 0x70, 0x2d, 0x4a,
    0x3e, 0x73, 0x33, 0x91, 0xb2, 0xa0, 0xf5, 0x3f,
    0x41, 0x6d, 0x9f, 0x90, 0x7e, 0x55, 0xcf, 0xf8,
];

/// IFAC flag bit in the first byte of a packet header.
const IFAC_FLAG: u8 = 0x80;

/// Minimum packet size for IFAC operations (flags + hops).
const MIN_PACKET_SIZE: usize = 2;

/// Ed25519 signature length — the upper bound for ifac_size.
const SIGNATURE_LENGTH: usize = 64;

/// IFAC authenticator for interface access control.
///
/// Sans-I/O: transforms raw packet bytes, no actual I/O. The authenticator
/// derives an Ed25519 identity from network name/key and uses truncated
/// signatures as access codes. Packets are XOR-masked so that nodes without
/// the correct credentials see only noise.
pub struct IfacAuthenticator {
    ifac_identity: PrivateIdentity,
    ifac_key: [u8; 64],
    ifac_size: usize,
}

impl IfacAuthenticator {
    /// Create from network name and/or network key.
    ///
    /// At least one of `ifac_netname` or `ifac_netkey` must be `Some`.
    /// The `ifac_size` is in bytes (typically 8 or 16); Reticulum defaults to 8.
    pub fn new(
        ifac_netname: Option<&str>,
        ifac_netkey: Option<&str>,
        ifac_size: usize,
    ) -> Result<Self, ReticulumError> {
        if ifac_netname.is_none() && ifac_netkey.is_none() {
            return Err(ReticulumError::IfacMissingCredentials);
        }

        if ifac_size == 0 || ifac_size > SIGNATURE_LENGTH {
            return Err(ReticulumError::IfacInvalidSize(ifac_size));
        }

        // Build ifac_origin: SHA256(netname) || SHA256(netkey)
        let mut ifac_origin = Vec::new();
        if let Some(name) = ifac_netname {
            ifac_origin.extend_from_slice(&hash::full_hash(name.as_bytes()));
        }
        if let Some(key) = ifac_netkey {
            ifac_origin.extend_from_slice(&hash::full_hash(key.as_bytes()));
        }

        let ifac_origin_hash = hash::full_hash(&ifac_origin);

        // HKDF: derive 64-byte key, zeroizing the intermediate Vec
        let mut ifac_key_vec = hkdf::derive_key(
            &ifac_origin_hash,
            Some(&IFAC_SALT),
            &[],
            64,
        )?;

        let mut ifac_key = [0u8; 64];
        ifac_key.copy_from_slice(&ifac_key_vec);
        ifac_key_vec.zeroize();

        // Create Ed25519 identity from the derived key
        let ifac_identity = PrivateIdentity::from_private_bytes(&ifac_key)?;

        Ok(Self {
            ifac_identity,
            ifac_key,
            ifac_size,
        })
    }

    /// Mask an outbound packet (insert IFAC + XOR mask).
    ///
    /// Input: raw packet bytes (flags, hops, header, payload).
    /// Output: masked bytes with IFAC inserted after the 2-byte header prefix.
    pub fn mask(&self, raw: &[u8]) -> Result<Vec<u8>, ReticulumError> {
        if raw.len() < MIN_PACKET_SIZE {
            return Err(ReticulumError::PacketTooShort {
                minimum: MIN_PACKET_SIZE,
                actual: raw.len(),
            });
        }

        // 1. Compute truncated signature as access code
        let sig = self.ifac_identity.sign(raw);
        let ifac = &sig[sig.len() - self.ifac_size..];

        // 2. Generate mask via HKDF
        let mask = hkdf::derive_key(
            ifac,
            Some(&self.ifac_key),
            &[],
            raw.len() + self.ifac_size,
        )?;

        // 3. Assemble: [flags|0x80][hops][IFAC][payload...]
        let mut new_raw = Vec::with_capacity(raw.len() + self.ifac_size);
        new_raw.push(raw[0] | IFAC_FLAG);
        new_raw.push(raw[1]);
        new_raw.extend_from_slice(ifac);
        new_raw.extend_from_slice(&raw[2..]);

        // 4. Apply selective XOR mask
        let mut masked = Vec::with_capacity(new_raw.len());
        for (i, &byte) in new_raw.iter().enumerate() {
            if i == 0 {
                // Mask first byte, but force IFAC flag on
                masked.push((byte ^ mask[i]) | IFAC_FLAG);
            } else if i == 1 || i > self.ifac_size + 1 {
                // Mask second header byte and payload
                masked.push(byte ^ mask[i]);
            } else {
                // Don't mask the IFAC itself (bytes 2..2+ifac_size)
                masked.push(byte);
            }
        }

        Ok(masked)
    }

    /// Unmask an inbound packet (verify IFAC + XOR unmask + strip).
    ///
    /// Returns the original raw packet bytes (without IFAC) on success,
    /// or `IfacVerificationFailed` if the access code doesn't match.
    pub fn unmask(&self, raw: &[u8]) -> Result<Vec<u8>, ReticulumError> {
        if raw.len() < MIN_PACKET_SIZE + self.ifac_size {
            return Err(ReticulumError::PacketTooShort {
                minimum: MIN_PACKET_SIZE + self.ifac_size,
                actual: raw.len(),
            });
        }

        // Check IFAC flag (safe: length validated above)
        if raw[0] & IFAC_FLAG == 0 {
            return Err(ReticulumError::IfacVerificationFailed);
        }

        // 1. Extract IFAC (cleartext, bytes 2..2+ifac_size)
        let ifac = &raw[2..2 + self.ifac_size];

        // 2. Generate mask
        let mask = hkdf::derive_key(
            ifac,
            Some(&self.ifac_key),
            &[],
            raw.len(),
        )?;

        // 3. Unmask selectively
        let mut unmasked = Vec::with_capacity(raw.len());
        for (i, &byte) in raw.iter().enumerate() {
            if i <= 1 || i > self.ifac_size + 1 {
                // Unmask header bytes and payload
                unmasked.push(byte ^ mask[i]);
            } else {
                // Don't unmask IFAC itself
                unmasked.push(byte);
            }
        }

        // 4. Clear IFAC flag and reassemble without IFAC bytes
        let mut new_raw = Vec::with_capacity(unmasked.len() - self.ifac_size);
        new_raw.push(unmasked[0] & !IFAC_FLAG);
        new_raw.push(unmasked[1]);
        new_raw.extend_from_slice(&unmasked[2 + self.ifac_size..]);

        // 5. Verify: expected IFAC from signature of reconstructed packet
        let expected_sig = self.ifac_identity.sign(&new_raw);
        let expected_ifac = &expected_sig[expected_sig.len() - self.ifac_size..];

        if ifac != expected_ifac {
            return Err(ReticulumError::IfacVerificationFailed);
        }

        Ok(new_raw)
    }

    /// Get the IFAC size in bytes.
    pub fn ifac_size(&self) -> usize {
        self.ifac_size
    }
}

impl Drop for IfacAuthenticator {
    fn drop(&mut self) {
        self.ifac_key.zeroize();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_authenticator() -> IfacAuthenticator {
        IfacAuthenticator::new(Some("testnet"), None, 8).unwrap()
    }

    fn make_test_packet() -> Vec<u8> {
        // Minimal Type1 packet: flags(1) + hops(1) + dest_hash(16) + context(1) + data(4)
        let mut pkt = vec![0x01, 0x00]; // announce, 0 hops
        pkt.extend_from_slice(&[0xAA; 16]); // dest hash
        pkt.push(0x00); // context
        pkt.extend_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]); // data
        pkt
    }

    #[test]
    fn mask_unmask_roundtrip() {
        let auth = make_authenticator();
        let original = make_test_packet();

        let masked = auth.mask(&original).unwrap();
        let unmasked = auth.unmask(&masked).unwrap();

        assert_eq!(unmasked, original);
    }

    #[test]
    fn mask_sets_ifac_flag() {
        let auth = make_authenticator();
        let original = make_test_packet();
        assert_eq!(original[0] & IFAC_FLAG, 0, "original should not have IFAC flag");

        let masked = auth.mask(&original).unwrap();
        assert_ne!(masked[0] & IFAC_FLAG, 0, "masked should have IFAC flag set");
    }

    #[test]
    fn unmask_clears_ifac_flag() {
        let auth = make_authenticator();
        let original = make_test_packet();

        let masked = auth.mask(&original).unwrap();
        let unmasked = auth.unmask(&masked).unwrap();
        assert_eq!(unmasked[0] & IFAC_FLAG, 0, "unmasked should not have IFAC flag");
    }

    #[test]
    fn masked_packet_is_longer_by_ifac_size() {
        let auth = make_authenticator();
        let original = make_test_packet();

        let masked = auth.mask(&original).unwrap();
        assert_eq!(masked.len(), original.len() + auth.ifac_size());
    }

    #[test]
    fn tampered_packet_rejected() {
        let auth = make_authenticator();
        let original = make_test_packet();
        let mut masked = auth.mask(&original).unwrap();

        // Tamper with payload (after IFAC bytes)
        let tamper_idx = 2 + auth.ifac_size() + 5;
        if tamper_idx < masked.len() {
            masked[tamper_idx] ^= 0xFF;
        }

        let result = auth.unmask(&masked);
        assert!(
            matches!(result, Err(ReticulumError::IfacVerificationFailed)),
            "tampered packet should fail verification"
        );
    }

    #[test]
    fn wrong_credentials_rejected() {
        let auth_a = IfacAuthenticator::new(Some("network_a"), None, 8).unwrap();
        let auth_b = IfacAuthenticator::new(Some("network_b"), None, 8).unwrap();

        let original = make_test_packet();
        let masked = auth_a.mask(&original).unwrap();

        let result = auth_b.unmask(&masked);
        assert!(
            matches!(result, Err(ReticulumError::IfacVerificationFailed)),
            "wrong network credentials should fail"
        );
    }

    #[test]
    fn packet_without_ifac_flag_rejected() {
        let auth = make_authenticator();
        let original = make_test_packet();
        assert_eq!(original[0] & IFAC_FLAG, 0);

        let result = auth.unmask(&original);
        assert!(matches!(result, Err(ReticulumError::IfacVerificationFailed)));
    }

    #[test]
    fn too_short_packet_rejected_for_mask() {
        let auth = make_authenticator();
        let result = auth.mask(&[0x00]);
        assert!(matches!(result, Err(ReticulumError::PacketTooShort { .. })));
    }

    #[test]
    fn too_short_packet_rejected_for_unmask() {
        let auth = make_authenticator();
        // Set IFAC flag but packet too short to contain IFAC bytes
        let result = auth.unmask(&[0x80, 0x00]);
        assert!(matches!(result, Err(ReticulumError::PacketTooShort { .. })));
    }

    #[test]
    fn ifac_size_16_roundtrip() {
        let auth = IfacAuthenticator::new(Some("bignet"), None, 16).unwrap();
        let original = make_test_packet();

        let masked = auth.mask(&original).unwrap();
        assert_eq!(masked.len(), original.len() + 16);

        let unmasked = auth.unmask(&masked).unwrap();
        assert_eq!(unmasked, original);
    }

    #[test]
    fn netname_and_netkey_both_used() {
        let auth_name_only = IfacAuthenticator::new(Some("net"), None, 8).unwrap();
        let auth_key_only = IfacAuthenticator::new(None, Some("key"), 8).unwrap();
        let auth_both = IfacAuthenticator::new(Some("net"), Some("key"), 8).unwrap();

        let original = make_test_packet();

        // Each produces different masked output
        let m1 = auth_name_only.mask(&original).unwrap();
        let m2 = auth_key_only.mask(&original).unwrap();
        let m3 = auth_both.mask(&original).unwrap();

        assert_ne!(m1, m2);
        assert_ne!(m1, m3);
        assert_ne!(m2, m3);

        // Each can only unmask its own
        assert!(auth_name_only.unmask(&m1).is_ok());
        assert!(auth_name_only.unmask(&m2).is_err());
        assert!(auth_name_only.unmask(&m3).is_err());
    }

    #[test]
    fn mask_preserves_content_after_roundtrip() {
        let auth = make_authenticator();

        // Test with various payload sizes
        for payload_size in [0, 1, 10, 100, 400] {
            let mut pkt = vec![0x01, 0x00]; // flags, hops
            pkt.extend_from_slice(&[0xBB; 16]); // dest hash
            pkt.push(0x00); // context
            pkt.extend(std::iter::repeat(0xCC).take(payload_size));

            let masked = auth.mask(&pkt).unwrap();
            let unmasked = auth.unmask(&masked).unwrap();
            assert_eq!(unmasked, pkt, "roundtrip failed for payload_size={payload_size}");
        }
    }

    #[test]
    fn mask_is_deterministic() {
        let auth = make_authenticator();
        let original = make_test_packet();

        let m1 = auth.mask(&original).unwrap();
        let m2 = auth.mask(&original).unwrap();

        // Ed25519 signing is deterministic, so masking should be too
        assert_eq!(m1, m2);
    }

    #[test]
    fn unmask_empty_input_returns_error_not_panic() {
        let auth = make_authenticator();
        let result = auth.unmask(&[]);
        assert!(matches!(result, Err(ReticulumError::PacketTooShort { .. })));
    }

    #[test]
    fn new_with_no_credentials_rejected() {
        let result = IfacAuthenticator::new(None, None, 8);
        assert!(matches!(result, Err(ReticulumError::IfacMissingCredentials)));
    }

    #[test]
    fn new_with_zero_ifac_size_rejected() {
        let result = IfacAuthenticator::new(Some("net"), None, 0);
        assert!(matches!(result, Err(ReticulumError::IfacInvalidSize(0))));
    }

    #[test]
    fn new_with_oversized_ifac_rejected() {
        let result = IfacAuthenticator::new(Some("net"), None, 65);
        assert!(matches!(result, Err(ReticulumError::IfacInvalidSize(65))));
    }

    #[test]
    fn new_with_max_ifac_size_accepted() {
        // 64 bytes = full Ed25519 signature, should be accepted
        let auth = IfacAuthenticator::new(Some("net"), None, 64);
        assert!(auth.is_ok());
        assert_eq!(auth.unwrap().ifac_size(), 64);
    }
}
