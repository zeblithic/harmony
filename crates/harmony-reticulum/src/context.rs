/// Packet context values indicating the purpose of the data section.
///
/// These match Reticulum's context constants defined in `Packet.py`.
/// Unknown values are mapped to [`PacketContext::None`] (0x00).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PacketContext {
    None = 0x00,
    Resource = 0x01,
    ResourceAdv = 0x02,
    ResourceReq = 0x03,
    ResourceHmu = 0x04,
    ResourcePrf = 0x05,
    ResourceIcl = 0x06,
    ResourceRcl = 0x07,
    CacheRequest = 0x08,
    Request = 0x09,
    Response = 0x0A,
    PathResponse = 0x0B,
    Command = 0x0C,
    CommandStatus = 0x0D,
    Channel = 0x0E,
    Keepalive = 0xFA,
    LinkIdentify = 0xFB,
    LinkClose = 0xFC,
    LinkProof = 0xFD,
    Lrrtt = 0xFE,
    LrProof = 0xFF,
}

impl PacketContext {
    /// Parse a context byte. Unknown values map to `None` (0x00).
    pub fn from_byte(byte: u8) -> Self {
        match byte {
            0x00 => Self::None,
            0x01 => Self::Resource,
            0x02 => Self::ResourceAdv,
            0x03 => Self::ResourceReq,
            0x04 => Self::ResourceHmu,
            0x05 => Self::ResourcePrf,
            0x06 => Self::ResourceIcl,
            0x07 => Self::ResourceRcl,
            0x08 => Self::CacheRequest,
            0x09 => Self::Request,
            0x0A => Self::Response,
            0x0B => Self::PathResponse,
            0x0C => Self::Command,
            0x0D => Self::CommandStatus,
            0x0E => Self::Channel,
            0xFA => Self::Keepalive,
            0xFB => Self::LinkIdentify,
            0xFC => Self::LinkClose,
            0xFD => Self::LinkProof,
            0xFE => Self::Lrrtt,
            0xFF => Self::LrProof,
            _ => Self::None,
        }
    }

    /// Return the wire byte for this context.
    pub fn as_byte(self) -> u8 {
        self as u8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_known_values_roundtrip() {
        let contexts = [
            (0x00, PacketContext::None),
            (0x01, PacketContext::Resource),
            (0x02, PacketContext::ResourceAdv),
            (0x03, PacketContext::ResourceReq),
            (0x04, PacketContext::ResourceHmu),
            (0x05, PacketContext::ResourcePrf),
            (0x06, PacketContext::ResourceIcl),
            (0x07, PacketContext::ResourceRcl),
            (0x08, PacketContext::CacheRequest),
            (0x09, PacketContext::Request),
            (0x0A, PacketContext::Response),
            (0x0B, PacketContext::PathResponse),
            (0x0C, PacketContext::Command),
            (0x0D, PacketContext::CommandStatus),
            (0x0E, PacketContext::Channel),
            (0xFA, PacketContext::Keepalive),
            (0xFB, PacketContext::LinkIdentify),
            (0xFC, PacketContext::LinkClose),
            (0xFD, PacketContext::LinkProof),
            (0xFE, PacketContext::Lrrtt),
            (0xFF, PacketContext::LrProof),
        ];

        for (byte, expected) in contexts {
            let parsed = PacketContext::from_byte(byte);
            assert_eq!(parsed, expected, "from_byte({byte:#04x})");
            assert_eq!(parsed.as_byte(), byte, "as_byte for {expected:?}");
        }
    }

    #[test]
    fn unknown_values_default_to_none() {
        for byte in [0x0F, 0x10, 0x50, 0x99, 0xF9] {
            assert_eq!(
                PacketContext::from_byte(byte),
                PacketContext::None,
                "unknown byte {byte:#04x} should map to None"
            );
        }
    }

    #[test]
    fn context_none_is_zero() {
        assert_eq!(PacketContext::None.as_byte(), 0x00);
    }
}
