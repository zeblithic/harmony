use harmony_crypto::hash;
use harmony_identity::identity::{
    Identity, PrivateIdentity, ADDRESS_HASH_LENGTH, PUBLIC_KEY_LENGTH, SIGNATURE_LENGTH,
};
use rand_core::CryptoRngCore;

use crate::context::PacketContext;
use crate::destination::DestinationName;
use crate::error::ReticulumError;
use crate::packet::{
    DestinationType, HeaderType, Packet, PacketFlags, PacketHeader, PacketType, PropagationType,
    MTU, HEADER_1_SIZE,
};

/// Hex-encode bytes without pulling in the `hex` crate at runtime.
fn bytes_to_hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use core::fmt::Write;
        let _ = write!(s, "{b:02x}");
    }
    s
}

/// Size of the ratchet key in the announce payload.
const RATCHET_SIZE: usize = 32;

/// Random hash length in announce payload.
const RANDOM_HASH_LENGTH: usize = 10;

/// Minimum announce payload size without ratchet:
/// pubkey(64) + name_hash(10) + random_hash(10) + signature(64) = 148
const MIN_ANNOUNCE_SIZE: usize =
    PUBLIC_KEY_LENGTH + hash::NAME_HASH_LENGTH + RANDOM_HASH_LENGTH + SIGNATURE_LENGTH;

/// Minimum announce payload size with ratchet:
/// 148 + ratchet(32) = 180
const MIN_ANNOUNCE_SIZE_RATCHET: usize = MIN_ANNOUNCE_SIZE + RATCHET_SIZE;

/// A validated announce extracted from a packet.
#[derive(Debug, Clone)]
pub struct ValidatedAnnounce {
    pub identity: Identity,
    pub destination_name: DestinationName,
    pub destination_hash: [u8; ADDRESS_HASH_LENGTH],
    pub random_hash: [u8; RANDOM_HASH_LENGTH],
    pub ratchet: Option<[u8; RATCHET_SIZE]>,
    pub app_data: Vec<u8>,
}

/// Build the 10-byte random hash for an announce.
///
/// Format: `SHA256(16 random bytes)[:5] || timestamp.to_be_bytes()[3..8]`
///
/// Sans-I/O: caller provides the RNG and timestamp.
pub fn build_random_hash(rng: &mut impl CryptoRngCore, timestamp_secs: u64) -> [u8; RANDOM_HASH_LENGTH] {
    let mut random_bytes = [0u8; hash::TRUNCATED_HASH_LENGTH];
    rng.fill_bytes(&mut random_bytes);
    let hash = hash::truncated_hash(&random_bytes);

    let ts_bytes = timestamp_secs.to_be_bytes(); // 8 bytes

    let mut result = [0u8; RANDOM_HASH_LENGTH];
    result[..5].copy_from_slice(&hash[..5]);
    result[5..].copy_from_slice(&ts_bytes[3..8]); // 5 bytes from big-endian u64
    result
}

/// Build signed data for an announce.
///
/// `dest_hash || pubkey_x25519 || pubkey_ed25519 || name_hash || random_hash || [ratchet] || [app_data]`
fn build_signed_data(
    destination_hash: &[u8; ADDRESS_HASH_LENGTH],
    public_keys: &[u8; PUBLIC_KEY_LENGTH],
    name_hash: &[u8; hash::NAME_HASH_LENGTH],
    random_hash: &[u8; RANDOM_HASH_LENGTH],
    ratchet: Option<&[u8; RATCHET_SIZE]>,
    app_data: &[u8],
) -> Vec<u8> {
    let capacity = ADDRESS_HASH_LENGTH
        + PUBLIC_KEY_LENGTH
        + hash::NAME_HASH_LENGTH
        + RANDOM_HASH_LENGTH
        + if ratchet.is_some() { RATCHET_SIZE } else { 0 }
        + app_data.len();

    let mut data = Vec::with_capacity(capacity);
    data.extend_from_slice(destination_hash);
    data.extend_from_slice(public_keys);
    data.extend_from_slice(name_hash);
    data.extend_from_slice(random_hash);
    if let Some(r) = ratchet {
        data.extend_from_slice(r);
    }
    data.extend_from_slice(app_data);
    data
}

/// Construct an announce packet.
///
/// Sans-I/O: caller provides the RNG and timestamp.
pub fn build_announce(
    private_identity: &PrivateIdentity,
    dest_name: &DestinationName,
    rng: &mut impl CryptoRngCore,
    timestamp_secs: u64,
    app_data: &[u8],
    ratchet: Option<&[u8; RATCHET_SIZE]>,
) -> Result<Packet, ReticulumError> {
    let identity = private_identity.public_identity();
    let public_keys = identity.to_public_bytes();
    let name_hash = *dest_name.name_hash();
    let destination_hash = dest_name.destination_hash(&identity.address_hash);
    let random_hash = build_random_hash(rng, timestamp_secs);

    // Build signed data and sign
    let signed_data = build_signed_data(
        &destination_hash,
        &public_keys,
        &name_hash,
        &random_hash,
        ratchet,
        app_data,
    );
    let signature = private_identity.sign(&signed_data);

    // Build payload: pubkeys || name_hash || random_hash || [ratchet] || signature || app_data
    let payload_size = PUBLIC_KEY_LENGTH
        + hash::NAME_HASH_LENGTH
        + RANDOM_HASH_LENGTH
        + if ratchet.is_some() { RATCHET_SIZE } else { 0 }
        + SIGNATURE_LENGTH
        + app_data.len();

    // Check MTU before assembling
    if HEADER_1_SIZE + payload_size > MTU {
        return Err(ReticulumError::AppDataTooLarge);
    }

    let mut payload = Vec::with_capacity(payload_size);
    payload.extend_from_slice(&public_keys);
    payload.extend_from_slice(&name_hash);
    payload.extend_from_slice(&random_hash);
    if let Some(r) = ratchet {
        payload.extend_from_slice(r);
    }
    payload.extend_from_slice(&signature);
    payload.extend_from_slice(app_data);

    let flags = PacketFlags {
        ifac: false,
        header_type: HeaderType::Type1,
        context_flag: ratchet.is_some(),
        propagation: PropagationType::Broadcast,
        destination_type: DestinationType::Single,
        packet_type: PacketType::Announce,
    };

    Ok(Packet {
        header: PacketHeader {
            flags,
            hops: 0,
            transport_id: None,
            destination_hash,
            context: PacketContext::None,
        },
        data: payload,
    })
}

/// Validate an announce packet and extract its contents.
///
/// Verification steps:
/// 1. Check packet type is Announce
/// 2. Parse payload at fixed offsets (ratchet presence from context_flag)
/// 3. Reconstruct Identity from public keys
/// 4. Rebuild signed_data and verify Ed25519 signature
/// 5. Verify destination_hash == SHA256(name_hash || identity.address_hash)[:16]
pub fn validate_announce(packet: &Packet) -> Result<ValidatedAnnounce, ReticulumError> {
    // Step 1: Check packet type
    if packet.header.flags.packet_type != PacketType::Announce {
        return Err(ReticulumError::InvalidPacketType(
            packet.header.flags.to_byte(),
        ));
    }

    let has_ratchet = packet.header.flags.context_flag;
    let min_size = if has_ratchet {
        MIN_ANNOUNCE_SIZE_RATCHET
    } else {
        MIN_ANNOUNCE_SIZE
    };

    if packet.data.len() < min_size {
        return Err(ReticulumError::AnnounceTooShort {
            minimum: min_size,
            actual: packet.data.len(),
        });
    }

    // Step 2: Parse payload at fixed offsets
    let data = &packet.data;
    let mut offset = 0;

    let public_keys: [u8; PUBLIC_KEY_LENGTH] = data[offset..offset + PUBLIC_KEY_LENGTH]
        .try_into()
        .unwrap();
    offset += PUBLIC_KEY_LENGTH;

    let mut name_hash = [0u8; hash::NAME_HASH_LENGTH];
    name_hash.copy_from_slice(&data[offset..offset + hash::NAME_HASH_LENGTH]);
    offset += hash::NAME_HASH_LENGTH;

    let mut random_hash = [0u8; RANDOM_HASH_LENGTH];
    random_hash.copy_from_slice(&data[offset..offset + RANDOM_HASH_LENGTH]);
    offset += RANDOM_HASH_LENGTH;

    let ratchet = if has_ratchet {
        let r: [u8; RATCHET_SIZE] = data[offset..offset + RATCHET_SIZE].try_into().unwrap();
        offset += RATCHET_SIZE;
        Some(r)
    } else {
        None
    };

    let signature: [u8; SIGNATURE_LENGTH] = data[offset..offset + SIGNATURE_LENGTH]
        .try_into()
        .unwrap();
    offset += SIGNATURE_LENGTH;

    let app_data = data[offset..].to_vec();

    // Step 3: Reconstruct Identity
    let x25519_pub: [u8; 32] = public_keys[..32].try_into().unwrap();
    let ed25519_pub: [u8; 32] = public_keys[32..].try_into().unwrap();
    let identity = Identity::from_public_keys(&x25519_pub, &ed25519_pub)?;

    // Step 4: Verify signature
    let signed_data = build_signed_data(
        &packet.header.destination_hash,
        &public_keys,
        &name_hash,
        &random_hash,
        ratchet.as_ref(),
        &app_data,
    );

    identity
        .verify(&signed_data, &signature)
        .map_err(|_| ReticulumError::AnnounceSignatureInvalid)?;

    // Step 5: Verify destination hash
    let dest_name = DestinationName::from_name_hash(name_hash);
    let expected_dest_hash = dest_name.destination_hash(&identity.address_hash);

    if expected_dest_hash != packet.header.destination_hash {
        return Err(ReticulumError::AnnounceDestinationMismatch {
            expected: bytes_to_hex(&expected_dest_hash),
            actual: bytes_to_hex(&packet.header.destination_hash),
        });
    }

    Ok(ValidatedAnnounce {
        identity,
        destination_name: dest_name,
        destination_hash: packet.header.destination_hash,
        random_hash,
        ratchet,
        app_data,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::OsRng;

    // ── Random hash ───────────────────────────────────────────────────

    #[test]
    fn random_hash_length() {
        let rh = build_random_hash(&mut OsRng, 1_700_000_000);
        assert_eq!(rh.len(), RANDOM_HASH_LENGTH);
    }

    #[test]
    fn random_hash_timestamp_encoding() {
        let timestamp: u64 = 0x00_11_22_33_44_55_66_77;
        let rh = build_random_hash(&mut OsRng, timestamp);
        // Last 5 bytes should be timestamp bytes [3..8] = [33, 44, 55, 66, 77]
        assert_eq!(&rh[5..], &[0x33, 0x44, 0x55, 0x66, 0x77]);
    }

    #[test]
    fn random_hash_different_each_call() {
        let rh1 = build_random_hash(&mut OsRng, 1_700_000_000);
        let rh2 = build_random_hash(&mut OsRng, 1_700_000_000);
        // Random portion (first 5 bytes) should differ
        assert_ne!(&rh1[..5], &rh2[..5]);
        // Timestamp portion (last 5 bytes) should be same
        assert_eq!(&rh1[5..], &rh2[5..]);
    }

    // ── Build + validate roundtrip ────────────────────────────────────

    #[test]
    fn announce_roundtrip_no_ratchet_no_appdata() {
        let identity = PrivateIdentity::generate(&mut OsRng);
        let dest = DestinationName::from_name("lxmf", &["delivery"]).unwrap();

        let packet = build_announce(&identity, &dest, &mut OsRng, 1_700_000_000, &[], None).unwrap();

        assert_eq!(packet.header.flags.packet_type, PacketType::Announce);
        assert!(!packet.header.flags.context_flag);
        assert_eq!(packet.data.len(), MIN_ANNOUNCE_SIZE);

        let validated = validate_announce(&packet).unwrap();
        assert_eq!(validated.identity, *identity.public_identity());
        assert_eq!(
            validated.destination_hash,
            dest.destination_hash(&identity.public_identity().address_hash)
        );
        assert!(validated.ratchet.is_none());
        assert!(validated.app_data.is_empty());
    }

    #[test]
    fn announce_roundtrip_with_ratchet() {
        let identity = PrivateIdentity::generate(&mut OsRng);
        let dest = DestinationName::from_name("lxmf", &["delivery"]).unwrap();
        let ratchet = [0x42u8; 32];

        let packet = build_announce(
            &identity,
            &dest,
            &mut OsRng,
            1_700_000_000,
            &[],
            Some(&ratchet),
        )
        .unwrap();

        assert!(packet.header.flags.context_flag);
        assert_eq!(packet.data.len(), MIN_ANNOUNCE_SIZE_RATCHET);

        let validated = validate_announce(&packet).unwrap();
        assert_eq!(validated.ratchet, Some(ratchet));
    }

    #[test]
    fn announce_roundtrip_with_appdata() {
        let identity = PrivateIdentity::generate(&mut OsRng);
        let dest = DestinationName::from_name("myapp", &["service"]).unwrap();
        let app_data = b"Hello, I am a node!";

        let packet = build_announce(
            &identity,
            &dest,
            &mut OsRng,
            1_700_000_000,
            app_data,
            None,
        )
        .unwrap();

        let validated = validate_announce(&packet).unwrap();
        assert_eq!(validated.app_data, app_data);
    }

    #[test]
    fn announce_roundtrip_with_ratchet_and_appdata() {
        let identity = PrivateIdentity::generate(&mut OsRng);
        let dest = DestinationName::from_name("app", &["x", "y"]).unwrap();
        let ratchet = [0xAB; 32];
        let app_data = b"node metadata here";

        let packet = build_announce(
            &identity,
            &dest,
            &mut OsRng,
            1_700_000_000,
            app_data,
            Some(&ratchet),
        )
        .unwrap();

        assert!(packet.header.flags.context_flag);

        let validated = validate_announce(&packet).unwrap();
        assert_eq!(validated.ratchet, Some(ratchet));
        assert_eq!(validated.app_data, app_data);
        assert_eq!(validated.identity, *identity.public_identity());
    }

    // ── Serialization roundtrip ───────────────────────────────────────

    #[test]
    fn announce_survives_serialization() {
        let identity = PrivateIdentity::generate(&mut OsRng);
        let dest = DestinationName::from_name("nomad", &["page"]).unwrap();

        let packet = build_announce(
            &identity,
            &dest,
            &mut OsRng,
            1_700_000_000,
            b"test",
            None,
        )
        .unwrap();

        // Serialize and deserialize
        let bytes = packet.to_bytes().unwrap();
        let parsed = Packet::from_bytes(&bytes).unwrap();
        let validated = validate_announce(&parsed).unwrap();

        assert_eq!(validated.identity, *identity.public_identity());
        assert_eq!(validated.app_data, b"test");
    }

    // ── Signature verification ────────────────────────────────────────

    #[test]
    fn tampered_signature_rejected() {
        let identity = PrivateIdentity::generate(&mut OsRng);
        let dest = DestinationName::from_name("app", &[]).unwrap();

        let mut packet =
            build_announce(&identity, &dest, &mut OsRng, 1_700_000_000, &[], None).unwrap();

        // Tamper with the signature (at offset 84 = 64 + 10 + 10)
        let sig_offset = PUBLIC_KEY_LENGTH + hash::NAME_HASH_LENGTH + RANDOM_HASH_LENGTH;
        packet.data[sig_offset] ^= 0x01;

        let result = validate_announce(&packet);
        assert!(matches!(
            result,
            Err(ReticulumError::AnnounceSignatureInvalid)
        ));
    }

    #[test]
    fn tampered_destination_hash_rejected() {
        let identity = PrivateIdentity::generate(&mut OsRng);
        let dest = DestinationName::from_name("app", &[]).unwrap();

        let mut packet =
            build_announce(&identity, &dest, &mut OsRng, 1_700_000_000, &[], None).unwrap();

        // Tamper with destination hash in header
        packet.header.destination_hash[0] ^= 0x01;

        let result = validate_announce(&packet);
        // Signature check will fail because dest_hash is part of signed data
        assert!(result.is_err());
    }

    #[test]
    fn tampered_appdata_rejected() {
        let identity = PrivateIdentity::generate(&mut OsRng);
        let dest = DestinationName::from_name("app", &[]).unwrap();

        let mut packet = build_announce(
            &identity,
            &dest,
            &mut OsRng,
            1_700_000_000,
            b"original",
            None,
        )
        .unwrap();

        // Tamper with app_data (last bytes of payload)
        let len = packet.data.len();
        packet.data[len - 1] ^= 0x01;

        let result = validate_announce(&packet);
        assert!(matches!(
            result,
            Err(ReticulumError::AnnounceSignatureInvalid)
        ));
    }

    // ── Error cases ───────────────────────────────────────────────────

    #[test]
    fn non_announce_packet_rejected() {
        let packet = Packet {
            header: PacketHeader {
                flags: PacketFlags {
                    ifac: false,
                    header_type: HeaderType::Type1,
                    context_flag: false,
                    propagation: PropagationType::Broadcast,
                    destination_type: DestinationType::Single,
                    packet_type: PacketType::Data, // Not an announce
                },
                hops: 0,
                transport_id: None,
                destination_hash: [0; 16],
                context: PacketContext::None,
            },
            data: vec![0; 200],
        };

        let result = validate_announce(&packet);
        assert!(matches!(result, Err(ReticulumError::InvalidPacketType(_))));
    }

    #[test]
    fn announce_too_short_rejected() {
        let packet = Packet {
            header: PacketHeader {
                flags: PacketFlags {
                    ifac: false,
                    header_type: HeaderType::Type1,
                    context_flag: false,
                    propagation: PropagationType::Broadcast,
                    destination_type: DestinationType::Single,
                    packet_type: PacketType::Announce,
                },
                hops: 0,
                transport_id: None,
                destination_hash: [0; 16],
                context: PacketContext::None,
            },
            data: vec![0; 100], // Too short for announce
        };

        let result = validate_announce(&packet);
        assert!(matches!(
            result,
            Err(ReticulumError::AnnounceTooShort {
                minimum: 148,
                actual: 100
            })
        ));
    }

    #[test]
    fn announce_with_ratchet_flag_too_short_rejected() {
        let packet = Packet {
            header: PacketHeader {
                flags: PacketFlags {
                    ifac: false,
                    header_type: HeaderType::Type1,
                    context_flag: true, // Has ratchet
                    propagation: PropagationType::Broadcast,
                    destination_type: DestinationType::Single,
                    packet_type: PacketType::Announce,
                },
                hops: 0,
                transport_id: None,
                destination_hash: [0; 16],
                context: PacketContext::None,
            },
            data: vec![0; 150], // Enough without ratchet (148), not enough with (180)
        };

        let result = validate_announce(&packet);
        assert!(matches!(
            result,
            Err(ReticulumError::AnnounceTooShort {
                minimum: 180,
                actual: 150
            })
        ));
    }

    #[test]
    fn app_data_too_large_rejected() {
        let identity = PrivateIdentity::generate(&mut OsRng);
        let dest = DestinationName::from_name("app", &[]).unwrap();

        // Max data = MTU(500) - header(19) - payload_overhead(148) = 333
        let too_large = vec![0u8; 400];
        let result = build_announce(
            &identity,
            &dest,
            &mut OsRng,
            1_700_000_000,
            &too_large,
            None,
        );
        assert!(matches!(result, Err(ReticulumError::AppDataTooLarge)));
    }

    // ── Deterministic signature ───────────────────────────────────────

    #[test]
    fn ed25519_signature_deterministic() {
        // Ed25519 signatures are deterministic (RFC 8032), so given the same
        // identity + dest + random_hash + timestamp, we get the same signature.
        let identity = PrivateIdentity::generate(&mut OsRng);
        let dest = DestinationName::from_name("test", &[]).unwrap();
        let dest_hash = dest.destination_hash(&identity.public_identity().address_hash);
        let public_keys = identity.public_identity().to_public_bytes();
        let name_hash = *dest.name_hash();
        let random_hash = [0xAA; RANDOM_HASH_LENGTH];

        let signed_data = build_signed_data(
            &dest_hash,
            &public_keys,
            &name_hash,
            &random_hash,
            None,
            b"data",
        );

        let sig1 = identity.sign(&signed_data);
        let sig2 = identity.sign(&signed_data);
        assert_eq!(sig1, sig2);
    }
}
