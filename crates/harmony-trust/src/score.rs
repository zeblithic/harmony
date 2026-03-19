use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TrustScore(u8);

impl TrustScore {
    pub fn new(raw: u8) -> Self {
        Self(raw)
    }
    pub fn raw(self) -> u8 {
        self.0
    }
    pub fn identity(self) -> u8 {
        (self.0 >> 6) & 0x03
    }
    pub fn compliance(self) -> u8 {
        (self.0 >> 4) & 0x03
    }
    pub fn association(self) -> u8 {
        (self.0 >> 2) & 0x03
    }
    pub fn endorsement(self) -> u8 {
        self.0 & 0x03
    }

    pub fn from_dimensions(identity: u8, compliance: u8, association: u8, endorsement: u8) -> Self {
        Self(
            ((identity & 0x03) << 6)
                | ((compliance & 0x03) << 4)
                | ((association & 0x03) << 2)
                | (endorsement & 0x03),
        )
    }

    pub const UNKNOWN: Self = Self(0x00);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_is_zero() {
        assert_eq!(TrustScore::UNKNOWN.raw(), 0x00);
        assert_eq!(TrustScore::UNKNOWN.identity(), 0);
        assert_eq!(TrustScore::UNKNOWN.compliance(), 0);
        assert_eq!(TrustScore::UNKNOWN.association(), 0);
        assert_eq!(TrustScore::UNKNOWN.endorsement(), 0);
    }

    #[test]
    fn from_dimensions_round_trip() {
        let score = TrustScore::from_dimensions(3, 2, 1, 0);
        assert_eq!(score.identity(), 3);
        assert_eq!(score.compliance(), 2);
        assert_eq!(score.association(), 1);
        assert_eq!(score.endorsement(), 0);
    }

    #[test]
    fn bit_layout_identity_is_msb() {
        let score = TrustScore::from_dimensions(3, 0, 0, 0);
        assert_eq!(score.raw(), 0xC0);
    }

    #[test]
    fn bit_layout_endorsement_is_lsb() {
        let score = TrustScore::from_dimensions(0, 0, 0, 3);
        assert_eq!(score.raw(), 0x03);
    }

    #[test]
    fn bit_layout_all_max() {
        let score = TrustScore::from_dimensions(3, 3, 3, 3);
        assert_eq!(score.raw(), 0xFF);
    }

    #[test]
    fn bit_layout_known_byte() {
        let score = TrustScore::new(0xF0);
        assert_eq!(score.identity(), 3);
        assert_eq!(score.compliance(), 3);
        assert_eq!(score.association(), 0);
        assert_eq!(score.endorsement(), 0);
    }

    #[test]
    fn from_dimensions_clamps_overflow() {
        let score = TrustScore::from_dimensions(0xFF, 0xFF, 0xFF, 0xFF);
        assert_eq!(score.raw(), 0xFF);
        assert_eq!(score.identity(), 3);
        assert_eq!(score.endorsement(), 3);
    }

    #[test]
    fn new_raw_round_trip() {
        for byte in 0u8..=255 {
            assert_eq!(TrustScore::new(byte).raw(), byte);
        }
    }

    #[test]
    fn serde_round_trip() {
        let score = TrustScore::from_dimensions(2, 3, 1, 0);
        let bytes = postcard::to_allocvec(&score).unwrap();
        let decoded: TrustScore = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(decoded, score);
    }
}
