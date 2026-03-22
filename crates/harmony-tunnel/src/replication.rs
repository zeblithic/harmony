//! Replication message protocol for encrypted backup delegation.
//!
//! Wire format: `[1 byte op][32 bytes CID][payload]`
//!
//! The tunnel encrypts/decrypts these as opaque `FrameTag::Replication` frames.
//! Semantic parsing happens at the runtime layer.

use alloc::vec::Vec;

use crate::error::TunnelError;

/// Replication operation types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ReplicationOp {
    /// Push encrypted book data to a peer for storage.
    Push = 0x01,
    /// Request a stored book from a peer.
    Pull = 0x02,
    /// Response to a Pull request with the book data.
    PullResponse = 0x03,
    /// Request storage usage status from a peer.
    Status = 0x04,
    /// Response to a Status request with usage and quota.
    StatusResponse = 0x05,
}

impl ReplicationOp {
    fn from_byte(b: u8) -> Result<Self, TunnelError> {
        match b {
            0x01 => Ok(Self::Push),
            0x02 => Ok(Self::Pull),
            0x03 => Ok(Self::PullResponse),
            0x04 => Ok(Self::Status),
            0x05 => Ok(Self::StatusResponse),
            other => Err(TunnelError::MalformedHandshake {
                reason: if other == 0 {
                    "unknown replication op: 0x00"
                } else {
                    "unknown replication op"
                },
            }),
        }
    }
}

/// A replication protocol message.
///
/// Encoded as `[1 byte op][32 bytes CID][payload]`.
/// For Status requests, CID is `[0; 32]` (unused).
/// For StatusResponse, payload is `[used_bytes LE 8][quota_bytes LE 8]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplicationMessage {
    pub op: ReplicationOp,
    pub cid: [u8; 32],
    pub payload: Vec<u8>,
}

/// Minimum wire size: 1 (op) + 32 (CID) = 33 bytes.
const MIN_MESSAGE_LEN: usize = 33;

impl ReplicationMessage {
    /// Encode to wire format.
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(MIN_MESSAGE_LEN + self.payload.len());
        buf.push(self.op as u8);
        buf.extend_from_slice(&self.cid);
        buf.extend_from_slice(&self.payload);
        buf
    }

    /// Decode from wire format.
    pub fn decode(data: &[u8]) -> Result<Self, TunnelError> {
        if data.len() < MIN_MESSAGE_LEN {
            return Err(TunnelError::FrameTooShort {
                expected: MIN_MESSAGE_LEN,
                got: data.len(),
            });
        }

        let op = ReplicationOp::from_byte(data[0])?;
        let mut cid = [0u8; 32];
        cid.copy_from_slice(&data[1..33]);
        let payload = data[33..].to_vec();

        Ok(Self { op, cid, payload })
    }

    /// Create a Push message.
    pub fn push(cid: [u8; 32], data: Vec<u8>) -> Self {
        Self {
            op: ReplicationOp::Push,
            cid,
            payload: data,
        }
    }

    /// Create a Pull request.
    pub fn pull(cid: [u8; 32]) -> Self {
        Self {
            op: ReplicationOp::Pull,
            cid,
            payload: Vec::new(),
        }
    }

    /// Create a Pull response with data.
    pub fn pull_response(cid: [u8; 32], data: Vec<u8>) -> Self {
        Self {
            op: ReplicationOp::PullResponse,
            cid,
            payload: data,
        }
    }

    /// Create a Status request.
    pub fn status() -> Self {
        Self {
            op: ReplicationOp::Status,
            cid: [0u8; 32],
            payload: Vec::new(),
        }
    }

    /// Create a StatusResponse with usage and quota.
    pub fn status_response(used: u64, quota: u64) -> Self {
        let mut payload = Vec::with_capacity(16);
        payload.extend_from_slice(&used.to_le_bytes());
        payload.extend_from_slice(&quota.to_le_bytes());
        Self {
            op: ReplicationOp::StatusResponse,
            cid: [0u8; 32],
            payload,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_roundtrip() {
        let cid = [0xAA; 32];
        let data = vec![1, 2, 3, 4, 5];
        let msg = ReplicationMessage::push(cid, data.clone());
        let encoded = msg.encode();
        let decoded = ReplicationMessage::decode(&encoded).unwrap();
        assert_eq!(decoded.op, ReplicationOp::Push);
        assert_eq!(decoded.cid, cid);
        assert_eq!(decoded.payload, data);
    }

    #[test]
    fn pull_roundtrip() {
        let cid = [0xBB; 32];
        let msg = ReplicationMessage::pull(cid);
        let encoded = msg.encode();
        let decoded = ReplicationMessage::decode(&encoded).unwrap();
        assert_eq!(decoded.op, ReplicationOp::Pull);
        assert_eq!(decoded.cid, cid);
        assert!(decoded.payload.is_empty());
    }

    #[test]
    fn pull_response_roundtrip() {
        let cid = [0xCC; 32];
        let data = vec![10, 20, 30];
        let msg = ReplicationMessage::pull_response(cid, data.clone());
        let encoded = msg.encode();
        let decoded = ReplicationMessage::decode(&encoded).unwrap();
        assert_eq!(decoded.op, ReplicationOp::PullResponse);
        assert_eq!(decoded.cid, cid);
        assert_eq!(decoded.payload, data);
    }

    #[test]
    fn status_roundtrip() {
        let msg = ReplicationMessage::status();
        let encoded = msg.encode();
        let decoded = ReplicationMessage::decode(&encoded).unwrap();
        assert_eq!(decoded.op, ReplicationOp::Status);
        assert_eq!(decoded.cid, [0u8; 32]);
        assert!(decoded.payload.is_empty());
    }

    #[test]
    fn status_response_roundtrip() {
        let used = 1024 * 1024 * 500; // 500 MiB
        let quota = 1024 * 1024 * 1024 * 50; // 50 GiB
        let msg = ReplicationMessage::status_response(used, quota);
        let encoded = msg.encode();
        let decoded = ReplicationMessage::decode(&encoded).unwrap();
        assert_eq!(decoded.op, ReplicationOp::StatusResponse);
        assert_eq!(decoded.payload.len(), 16);
        let decoded_used = u64::from_le_bytes(decoded.payload[0..8].try_into().unwrap());
        let decoded_quota = u64::from_le_bytes(decoded.payload[8..16].try_into().unwrap());
        assert_eq!(decoded_used, used);
        assert_eq!(decoded_quota, quota);
    }

    #[test]
    fn decode_too_short_fails() {
        let result = ReplicationMessage::decode(&[0x01; 10]);
        assert!(result.is_err());
    }

    #[test]
    fn decode_unknown_op_fails() {
        let mut data = vec![0xFF];
        data.extend_from_slice(&[0u8; 32]);
        let result = ReplicationMessage::decode(&data);
        assert!(result.is_err());
    }
}
