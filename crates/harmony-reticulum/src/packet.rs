use alloc::{sync::Arc, vec::Vec};

use harmony_crypto::hash;

use crate::context::PacketContext;
use crate::error::ReticulumError;

/// Reticulum MTU in bytes.
pub const MTU: usize = 500;

/// Type1 header size: flags(1) + hops(1) + dest_hash(16) + context(1).
pub const HEADER_1_SIZE: usize = 19;

/// Type2 header size: flags(1) + hops(1) + transport_id(16) + dest_hash(16) + context(1).
pub const HEADER_2_SIZE: usize = 35;

// ── Enums for flag fields ─────────────────────────────────────────────

/// Header format type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeaderType {
    /// Type1: direct, no transport relay.
    Type1 = 0,
    /// Type2: transport header with transport_id.
    Type2 = 1,
}

/// Propagation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PropagationType {
    Broadcast = 0,
    Transport = 1,
}

/// Destination type (2 bits).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DestinationType {
    Single = 0,
    Group = 1,
    Plain = 2,
    Link = 3,
}

impl DestinationType {
    fn from_bits(bits: u8) -> Self {
        match bits & 0x03 {
            0 => Self::Single,
            1 => Self::Group,
            2 => Self::Plain,
            3 => Self::Link,
            _ => unreachable!(),
        }
    }
}

/// Packet type (2 bits).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PacketType {
    Data = 0,
    Announce = 1,
    LinkRequest = 2,
    Proof = 3,
}

impl PacketType {
    fn from_bits(bits: u8) -> Self {
        match bits & 0x03 {
            0 => Self::Data,
            1 => Self::Announce,
            2 => Self::LinkRequest,
            3 => Self::Proof,
            _ => unreachable!(),
        }
    }
}

// ── Flags byte ────────────────────────────────────────────────────────

/// Decoded flags byte from the packet header.
///
/// Wire layout (MSB to LSB):
/// ```text
/// Bit 7: IFAC flag       (0=Open, 1=Authenticated)
/// Bit 6: Header type     (0=Type1, 1=Type2)
/// Bit 5: Context flag    (0=Unset, 1=Set)
/// Bit 4: Propagation     (0=Broadcast, 1=Transport)
/// Bits 3-2: Dest type    (00=Single, 01=Group, 10=Plain, 11=Link)
/// Bits 1-0: Packet type  (00=Data, 01=Announce, 10=LinkRequest, 11=Proof)
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PacketFlags {
    pub ifac: bool,
    pub header_type: HeaderType,
    pub context_flag: bool,
    pub propagation: PropagationType,
    pub destination_type: DestinationType,
    pub packet_type: PacketType,
}

impl PacketFlags {
    /// Decode a flags byte from the wire.
    pub fn from_byte(byte: u8) -> Self {
        Self {
            ifac: (byte & 0x80) != 0,
            header_type: if (byte & 0x40) != 0 {
                HeaderType::Type2
            } else {
                HeaderType::Type1
            },
            context_flag: (byte & 0x20) != 0,
            propagation: if (byte & 0x10) != 0 {
                PropagationType::Transport
            } else {
                PropagationType::Broadcast
            },
            destination_type: DestinationType::from_bits((byte >> 2) & 0x03),
            packet_type: PacketType::from_bits(byte & 0x03),
        }
    }

    /// Encode to a single flags byte.
    pub fn to_byte(self) -> u8 {
        let mut byte = 0u8;
        if self.ifac {
            byte |= 0x80;
        }
        if matches!(self.header_type, HeaderType::Type2) {
            byte |= 0x40;
        }
        if self.context_flag {
            byte |= 0x20;
        }
        if matches!(self.propagation, PropagationType::Transport) {
            byte |= 0x10;
        }
        byte |= (self.destination_type as u8 & 0x03) << 2;
        byte |= self.packet_type as u8 & 0x03;
        byte
    }
}

// ── Packet header ─────────────────────────────────────────────────────

/// Decoded packet header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PacketHeader {
    pub flags: PacketFlags,
    pub hops: u8,
    /// Present only for Type2 (transport) headers.
    pub transport_id: Option<[u8; hash::TRUNCATED_HASH_LENGTH]>,
    pub destination_hash: [u8; hash::TRUNCATED_HASH_LENGTH],
    pub context: PacketContext,
}

impl PacketHeader {
    /// Size of this header on the wire.
    pub fn wire_size(&self) -> usize {
        match self.flags.header_type {
            HeaderType::Type1 => HEADER_1_SIZE,
            HeaderType::Type2 => HEADER_2_SIZE,
        }
    }
}

// ── Full packet ───────────────────────────────────────────────────────

/// A Reticulum packet: header + data payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Packet {
    pub header: PacketHeader,
    pub data: Arc<[u8]>,
}

impl Packet {
    /// Parse a packet from raw bytes.
    pub fn from_bytes(raw: &[u8]) -> Result<Self, ReticulumError> {
        if raw.len() < 2 {
            return Err(ReticulumError::PacketTooShort {
                minimum: HEADER_1_SIZE,
                actual: raw.len(),
            });
        }

        let flags = PacketFlags::from_byte(raw[0]);
        let hops = raw[1];

        let (header_size, transport_id) = match flags.header_type {
            HeaderType::Type1 => (HEADER_1_SIZE, None),
            HeaderType::Type2 => {
                if raw.len() < HEADER_2_SIZE {
                    return Err(ReticulumError::PacketTooShort {
                        minimum: HEADER_2_SIZE,
                        actual: raw.len(),
                    });
                }
                let mut tid = [0u8; hash::TRUNCATED_HASH_LENGTH];
                tid.copy_from_slice(&raw[2..18]);
                (HEADER_2_SIZE, Some(tid))
            }
        };

        if raw.len() < header_size {
            return Err(ReticulumError::PacketTooShort {
                minimum: header_size,
                actual: raw.len(),
            });
        }

        // Destination hash starts after flags+hops (and transport_id for Type2)
        let dest_offset = match flags.header_type {
            HeaderType::Type1 => 2,
            HeaderType::Type2 => 18,
        };
        let mut destination_hash = [0u8; hash::TRUNCATED_HASH_LENGTH];
        destination_hash
            .copy_from_slice(&raw[dest_offset..dest_offset + hash::TRUNCATED_HASH_LENGTH]);

        let context = PacketContext::from_byte(raw[header_size - 1]);

        let data: Arc<[u8]> = Arc::from(&raw[header_size..]);

        Ok(Self {
            header: PacketHeader {
                flags,
                hops,
                transport_id,
                destination_hash,
                context,
            },
            data,
        })
    }

    /// Return the hashable portion of this packet for link_id computation.
    ///
    /// Strips transport metadata: IFAC, header_type, context_flag, propagation
    /// bits are cleared (keeping only dest_type + packet_type in the low 4 bits),
    /// the hops byte is dropped, and transport_id is dropped for Type2.
    ///
    /// Result: `[flags & 0x0F][dest_hash:16][context:1][data:N]`
    pub fn hashable_part(&self) -> Result<Vec<u8>, ReticulumError> {
        let masked_flags = self.header.flags.to_byte() & 0x0F;
        let mut buf = Vec::with_capacity(1 + 16 + 1 + self.data.len());
        buf.push(masked_flags);
        buf.extend_from_slice(&self.header.destination_hash);
        buf.push(self.header.context.as_byte());
        buf.extend_from_slice(&self.data);
        Ok(buf)
    }

    /// Serialize this packet to bytes, enforcing MTU.
    pub fn to_bytes(&self) -> Result<Vec<u8>, ReticulumError> {
        // Validate header_type ↔ transport_id consistency
        match (self.header.flags.header_type, &self.header.transport_id) {
            (HeaderType::Type2, None) => return Err(ReticulumError::MissingTransportId),
            (HeaderType::Type1, Some(_)) => return Err(ReticulumError::UnexpectedTransportId),
            _ => {}
        }

        let header_size = self.header.wire_size();
        let total = header_size + self.data.len();

        if total > MTU {
            return Err(ReticulumError::PacketExceedsMtu {
                size: total,
                mtu: MTU,
            });
        }

        let mut buf = Vec::with_capacity(total);
        buf.push(self.header.flags.to_byte());
        buf.push(self.header.hops);

        if let Some(tid) = &self.header.transport_id {
            buf.extend_from_slice(tid);
        }

        buf.extend_from_slice(&self.header.destination_hash);
        buf.push(self.header.context.as_byte());
        buf.extend_from_slice(&self.data);

        Ok(buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Flags byte roundtrip ──────────────────────────────────────────

    #[test]
    fn flags_all_zero() {
        let flags = PacketFlags::from_byte(0x00);
        assert!(!flags.ifac);
        assert_eq!(flags.header_type, HeaderType::Type1);
        assert!(!flags.context_flag);
        assert_eq!(flags.propagation, PropagationType::Broadcast);
        assert_eq!(flags.destination_type, DestinationType::Single);
        assert_eq!(flags.packet_type, PacketType::Data);
        assert_eq!(flags.to_byte(), 0x00);
    }

    #[test]
    fn flags_all_set() {
        // IFAC=1, Type2, context=1, transport, Link, Proof → 0b_1111_1111 = 0xFF
        let flags = PacketFlags::from_byte(0xFF);
        assert!(flags.ifac);
        assert_eq!(flags.header_type, HeaderType::Type2);
        assert!(flags.context_flag);
        assert_eq!(flags.propagation, PropagationType::Transport);
        assert_eq!(flags.destination_type, DestinationType::Link);
        assert_eq!(flags.packet_type, PacketType::Proof);
        assert_eq!(flags.to_byte(), 0xFF);
    }

    #[test]
    fn flags_announce_broadcast_single() {
        // Type1, broadcast, single, announce, no IFAC, no context
        // = 0b_0000_0001 = 0x01
        let flags = PacketFlags {
            ifac: false,
            header_type: HeaderType::Type1,
            context_flag: false,
            propagation: PropagationType::Broadcast,
            destination_type: DestinationType::Single,
            packet_type: PacketType::Announce,
        };
        assert_eq!(flags.to_byte(), 0x01);
        assert_eq!(PacketFlags::from_byte(0x01), flags);
    }

    #[test]
    fn flags_transport_type2_data() {
        // Type2, transport, plain destination, data
        // = 0b_0101_1000 = 0x58
        let flags = PacketFlags {
            ifac: false,
            header_type: HeaderType::Type2,
            context_flag: false,
            propagation: PropagationType::Transport,
            destination_type: DestinationType::Plain,
            packet_type: PacketType::Data,
        };
        assert_eq!(flags.to_byte(), 0x58);
        assert_eq!(PacketFlags::from_byte(0x58), flags);
    }

    #[test]
    fn flags_with_context_and_ifac() {
        // IFAC=1, Type1, context=1, broadcast, group, link_request
        // = 0b_1010_0110 = 0xA6
        let flags = PacketFlags {
            ifac: true,
            header_type: HeaderType::Type1,
            context_flag: true,
            propagation: PropagationType::Broadcast,
            destination_type: DestinationType::Group,
            packet_type: PacketType::LinkRequest,
        };
        assert_eq!(flags.to_byte(), 0xA6);
        assert_eq!(PacketFlags::from_byte(0xA6), flags);
    }

    #[test]
    fn flags_byte_roundtrip_exhaustive() {
        // Every possible byte value should roundtrip
        for byte in 0..=255u8 {
            let flags = PacketFlags::from_byte(byte);
            assert_eq!(flags.to_byte(), byte, "roundtrip failed for {byte:#04x}");
        }
    }

    // ── Type1 packet roundtrip ────────────────────────────────────────

    #[test]
    fn type1_packet_roundtrip() {
        let dest_hash = [0xAA; 16];
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
                hops: 3,
                transport_id: None,
                destination_hash: dest_hash,
                context: PacketContext::None,
            },
            data: vec![1, 2, 3, 4, 5].into(),
        };

        let bytes = packet.to_bytes().unwrap();
        assert_eq!(bytes.len(), HEADER_1_SIZE + 5);
        assert_eq!(bytes[0], 0x01); // announce
        assert_eq!(bytes[1], 3); // hops
        assert_eq!(&bytes[2..18], &dest_hash);
        assert_eq!(bytes[18], 0x00); // context None

        let parsed = Packet::from_bytes(&bytes).unwrap();
        assert_eq!(parsed, packet);
    }

    // ── Type2 packet roundtrip ────────────────────────────────────────

    #[test]
    fn type2_packet_roundtrip() {
        let transport_id = [0xBB; 16];
        let dest_hash = [0xCC; 16];
        let packet = Packet {
            header: PacketHeader {
                flags: PacketFlags {
                    ifac: false,
                    header_type: HeaderType::Type2,
                    context_flag: false,
                    propagation: PropagationType::Transport,
                    destination_type: DestinationType::Single,
                    packet_type: PacketType::Data,
                },
                hops: 7,
                transport_id: Some(transport_id),
                destination_hash: dest_hash,
                context: PacketContext::Resource,
            },
            data: vec![0xDD; 100].into(),
        };

        let bytes = packet.to_bytes().unwrap();
        assert_eq!(bytes.len(), HEADER_2_SIZE + 100);
        assert_eq!(&bytes[2..18], &transport_id);
        assert_eq!(&bytes[18..34], &dest_hash);

        let parsed = Packet::from_bytes(&bytes).unwrap();
        assert_eq!(parsed, packet);
    }

    // ── Error cases ───────────────────────────────────────────────────

    #[test]
    fn packet_too_short_rejected() {
        // Less than minimum header
        let result = Packet::from_bytes(&[0x00]);
        assert!(matches!(result, Err(ReticulumError::PacketTooShort { .. })));
    }

    #[test]
    fn type1_too_short_rejected() {
        // Type1 needs 19 bytes minimum
        let result = Packet::from_bytes(&[0x00; 18]);
        assert!(matches!(
            result,
            Err(ReticulumError::PacketTooShort { minimum: 19, .. })
        ));
    }

    #[test]
    fn type2_too_short_rejected() {
        // Set header_type bit (0x40) for Type2, needs 35 bytes
        let mut raw = [0u8; 20];
        raw[0] = 0x40; // Type2
        let result = Packet::from_bytes(&raw);
        assert!(matches!(
            result,
            Err(ReticulumError::PacketTooShort { minimum: 35, .. })
        ));
    }

    #[test]
    fn mtu_enforcement() {
        let packet = Packet {
            header: PacketHeader {
                flags: PacketFlags {
                    ifac: false,
                    header_type: HeaderType::Type1,
                    context_flag: false,
                    propagation: PropagationType::Broadcast,
                    destination_type: DestinationType::Single,
                    packet_type: PacketType::Data,
                },
                hops: 0,
                transport_id: None,
                destination_hash: [0; 16],
                context: PacketContext::None,
            },
            data: vec![0u8; MTU].into(), // header(19) + MTU(500) > 500
        };

        let result = packet.to_bytes();
        assert!(matches!(
            result,
            Err(ReticulumError::PacketExceedsMtu { .. })
        ));
    }

    #[test]
    fn max_mtu_packet_accepted() {
        // 500 - 19 = 481 bytes of data is the max for Type1
        let packet = Packet {
            header: PacketHeader {
                flags: PacketFlags {
                    ifac: false,
                    header_type: HeaderType::Type1,
                    context_flag: false,
                    propagation: PropagationType::Broadcast,
                    destination_type: DestinationType::Single,
                    packet_type: PacketType::Data,
                },
                hops: 0,
                transport_id: None,
                destination_hash: [0; 16],
                context: PacketContext::None,
            },
            data: vec![0u8; MTU - HEADER_1_SIZE].into(),
        };

        let bytes = packet.to_bytes().unwrap();
        assert_eq!(bytes.len(), MTU);
    }

    #[test]
    fn type2_missing_transport_id_rejected() {
        let packet = Packet {
            header: PacketHeader {
                flags: PacketFlags {
                    ifac: false,
                    header_type: HeaderType::Type2,
                    context_flag: false,
                    propagation: PropagationType::Transport,
                    destination_type: DestinationType::Single,
                    packet_type: PacketType::Data,
                },
                hops: 0,
                transport_id: None, // Missing!
                destination_hash: [0; 16],
                context: PacketContext::None,
            },
            data: vec![].into(),
        };

        let result = packet.to_bytes();
        assert!(matches!(result, Err(ReticulumError::MissingTransportId)));
    }

    #[test]
    fn type1_with_transport_id_rejected() {
        let packet = Packet {
            header: PacketHeader {
                flags: PacketFlags {
                    ifac: false,
                    header_type: HeaderType::Type1,
                    context_flag: false,
                    propagation: PropagationType::Broadcast,
                    destination_type: DestinationType::Single,
                    packet_type: PacketType::Data,
                },
                hops: 0,
                transport_id: Some([0xBB; 16]), // Invalid for Type1!
                destination_hash: [0; 16],
                context: PacketContext::None,
            },
            data: vec![].into(),
        };

        let result = packet.to_bytes();
        assert!(matches!(result, Err(ReticulumError::UnexpectedTransportId)));
    }

    #[test]
    fn empty_data_packet_type1() {
        let packet = Packet {
            header: PacketHeader {
                flags: PacketFlags {
                    ifac: false,
                    header_type: HeaderType::Type1,
                    context_flag: false,
                    propagation: PropagationType::Broadcast,
                    destination_type: DestinationType::Plain,
                    packet_type: PacketType::Data,
                },
                hops: 0,
                transport_id: None,
                destination_hash: [0; 16],
                context: PacketContext::None,
            },
            data: vec![].into(),
        };

        let bytes = packet.to_bytes().unwrap();
        assert_eq!(bytes.len(), HEADER_1_SIZE);

        let parsed = Packet::from_bytes(&bytes).unwrap();
        assert_eq!(parsed, packet);
    }

    #[test]
    fn header_wire_size() {
        let mut header = PacketHeader {
            flags: PacketFlags::from_byte(0x00),
            hops: 0,
            transport_id: None,
            destination_hash: [0; 16],
            context: PacketContext::None,
        };
        assert_eq!(header.wire_size(), HEADER_1_SIZE);

        header.flags.header_type = HeaderType::Type2;
        assert_eq!(header.wire_size(), HEADER_2_SIZE);
    }

    // ── hashable_part tests ─────────────────────────────────────────

    #[test]
    fn hashable_part_type1_basic() {
        let dest_hash = [0xAA; 16];
        let packet = Packet {
            header: PacketHeader {
                flags: PacketFlags {
                    ifac: false,
                    header_type: HeaderType::Type1,
                    context_flag: false,
                    propagation: PropagationType::Broadcast,
                    destination_type: DestinationType::Single,
                    packet_type: PacketType::LinkRequest,
                },
                hops: 5,
                transport_id: None,
                destination_hash: dest_hash,
                context: PacketContext::None,
            },
            data: vec![1, 2, 3].into(),
        };

        let hp = packet.hashable_part().unwrap();
        // flags masked to low 4 bits: dest_type=Single(0), packet_type=LinkRequest(2) => 0x02
        assert_eq!(hp[0], 0x02);
        assert_eq!(&hp[1..17], &dest_hash);
        assert_eq!(hp[17], 0x00); // context None
        assert_eq!(&hp[18..], &[1, 2, 3]);
        // hops is NOT included
        assert_eq!(hp.len(), 1 + 16 + 1 + 3);
    }

    #[test]
    fn hashable_part_type2_strips_transport_id() {
        let transport_id = [0xBB; 16];
        let dest_hash = [0xCC; 16];
        let packet = Packet {
            header: PacketHeader {
                flags: PacketFlags {
                    ifac: false,
                    header_type: HeaderType::Type2,
                    context_flag: false,
                    propagation: PropagationType::Transport,
                    destination_type: DestinationType::Single,
                    packet_type: PacketType::LinkRequest,
                },
                hops: 3,
                transport_id: Some(transport_id),
                destination_hash: dest_hash,
                context: PacketContext::None,
            },
            data: vec![0xDD; 10].into(),
        };

        let hp = packet.hashable_part().unwrap();
        // transport_id should NOT appear in hashable_part
        assert_eq!(hp[0], 0x02); // flags masked: Single(0) + LinkRequest(2)
        assert_eq!(&hp[1..17], &dest_hash);
        // No transport_id bytes
        assert_eq!(hp.len(), 1 + 16 + 1 + 10);
    }

    #[test]
    fn hashable_part_flags_masked_to_low_4_bits() {
        // Set all high bits: IFAC=1, Type2=1, context=1, propagation=1
        // plus dest_type=Link(3), packet_type=Proof(3) => low 4 bits = 0x0F
        let packet = Packet {
            header: PacketHeader {
                flags: PacketFlags::from_byte(0xFF),
                hops: 0,
                transport_id: Some([0; 16]),
                destination_hash: [0; 16],
                context: PacketContext::LrProof,
            },
            data: vec![].into(),
        };

        let hp = packet.hashable_part().unwrap();
        assert_eq!(hp[0], 0x0F, "only low 4 bits should survive masking");
    }
}
