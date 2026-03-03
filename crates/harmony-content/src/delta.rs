use crate::bundle::{self, CID_SIZE};
use crate::cid::ContentId;
use crate::error::ContentError;

/// Opcode byte for COPY: copy CIDs from the old bundle.
const OP_COPY: u8 = 0x00;

/// Opcode byte for INSERT: inline new CID data.
const OP_INSERT: u8 = 0x01;

/// Tag byte for encode_update: full bundle follows.
const UPDATE_FULL: u8 = 0x00;

/// Tag byte for encode_update: delta follows.
const UPDATE_DELTA: u8 = 0x01;

#[cfg(test)]
mod tests {
    use super::*;

    /// Build raw bundle bytes from a slice of ContentIds.
    fn bundle_bytes(cids: &[ContentId]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(cids.len() * CID_SIZE);
        for cid in cids {
            bytes.extend_from_slice(&cid.to_bytes());
        }
        bytes
    }

    /// Create N distinct blob CIDs for testing.
    fn make_cids(n: usize) -> Vec<ContentId> {
        (0..n)
            .map(|i| {
                let data = format!("test-blob-{i}");
                ContentId::for_blob(data.as_bytes()).unwrap()
            })
            .collect()
    }

    #[test]
    fn module_compiles() {
        let _ = OP_COPY;
        let _ = OP_INSERT;
        let cids = make_cids(3);
        let _ = bundle_bytes(&cids);
    }
}
