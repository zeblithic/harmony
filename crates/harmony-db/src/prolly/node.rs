// Prolly Tree node types and CAS I/O.

use crate::error::DbError;
use crate::types::{Entry, EntryMeta};
use harmony_content::{ContentFlags, ContentId};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// A leaf-level entry in a Prolly Tree node.
///
/// Uses `[u8; 32]` for `value_cid` instead of `ContentId` because ContentId's
/// serde impl may produce different encoding than postcard expects for the
/// 32-byte array. Use `.to_bytes()` and `ContentId::from_bytes()` for conversion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct LeafEntry {
    pub key: Vec<u8>,
    pub value_cid: [u8; 32],
    pub timestamp: u64,
    pub flags: u64,
    pub snippet: String,
}

impl LeafEntry {
    /// Convert from a `types::Entry`.
    pub fn from_entry(entry: &Entry) -> Self {
        LeafEntry {
            key: entry.key.clone(),
            value_cid: entry.value_cid.to_bytes(),
            timestamp: entry.timestamp,
            flags: entry.metadata.flags,
            snippet: entry.metadata.snippet.clone(),
        }
    }

    /// Convert back to a `types::Entry`.
    pub fn to_entry(&self) -> Entry {
        Entry {
            key: self.key.clone(),
            value_cid: ContentId::from_bytes(self.value_cid),
            timestamp: self.timestamp,
            metadata: EntryMeta {
                flags: self.flags,
                snippet: self.snippet.clone(),
            },
        }
    }

    /// Approximate serialized size in bytes.
    pub fn approx_size(&self) -> usize {
        self.key.len()
            + 32 // value_cid
            + 8  // timestamp
            + 8  // flags
            + self.snippet.len()
            + 8  // length prefixes overhead estimate
    }
}

/// A branch-level entry in a Prolly Tree node.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct BranchEntry {
    pub boundary_key: Vec<u8>,
    pub child_cid: [u8; 32],
}

impl BranchEntry {
    /// Approximate serialized size in bytes.
    pub fn approx_size(&self) -> usize {
        self.boundary_key.len()
            + 32 // child_cid
            + 4  // length prefix overhead estimate
    }
}

/// A Prolly Tree node: either a leaf containing entries or a branch containing
/// child pointers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum Node {
    Leaf(Vec<LeafEntry>),
    Branch(Vec<BranchEntry>),
}

impl Node {
    /// Serialize this node with postcard, compute its ContentId, and write
    /// atomically to `commits/{hex}.bin`.
    pub fn write_to_cas(&self, data_dir: &Path) -> Result<ContentId, DbError> {
        let bytes =
            postcard::to_allocvec(self).map_err(|e| DbError::Serialize(e.to_string()))?;
        let cid = ContentId::for_book(&bytes, ContentFlags::default())
            .map_err(|e| DbError::Serialize(format!("CID error: {e:?}")))?;
        let hex = hex::encode(cid.to_bytes());
        let commits_dir = data_dir.join("commits");
        let path = commits_dir.join(format!("{hex}.bin"));
        if !path.exists() {
            let tmp = commits_dir.join(format!("{hex}.bin.tmp"));
            std::fs::write(&tmp, &bytes)?;
            std::fs::rename(&tmp, &path)?;
        }
        Ok(cid)
    }

    /// Read a node from `commits/{hex}.bin` and deserialize with postcard.
    pub fn read_from_cas(data_dir: &Path, cid: ContentId) -> Result<Node, DbError> {
        let hex = hex::encode(cid.to_bytes());
        let path = data_dir.join("commits").join(format!("{hex}.bin"));
        let bytes = std::fs::read(&path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                DbError::CommitNotFound { cid: hex.clone() }
            } else {
                DbError::Io(e)
            }
        })?;
        let node: Node =
            postcard::from_bytes(&bytes).map_err(|e| DbError::CorruptIndex(e.to_string()))?;
        Ok(node)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_leaf_entries() -> Vec<LeafEntry> {
        vec![
            LeafEntry {
                key: b"alice".to_vec(),
                value_cid: [0xAA; 32],
                timestamp: 1000,
                flags: 0,
                snippet: "first entry".to_string(),
            },
            LeafEntry {
                key: b"bob".to_vec(),
                value_cid: [0xBB; 32],
                timestamp: 2000,
                flags: 1,
                snippet: "second entry".to_string(),
            },
        ]
    }

    fn sample_branch_entries() -> Vec<BranchEntry> {
        vec![
            BranchEntry {
                boundary_key: b"alice".to_vec(),
                child_cid: [0x11; 32],
            },
            BranchEntry {
                boundary_key: b"mallory".to_vec(),
                child_cid: [0x22; 32],
            },
        ]
    }

    #[test]
    fn leaf_serialize_round_trip() {
        let entries = sample_leaf_entries();
        let node = Node::Leaf(entries.clone());
        let bytes = postcard::to_allocvec(&node).unwrap();
        let decoded: Node = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(decoded, Node::Leaf(entries));
    }

    #[test]
    fn branch_serialize_round_trip() {
        let entries = sample_branch_entries();
        let node = Node::Branch(entries.clone());
        let bytes = postcard::to_allocvec(&node).unwrap();
        let decoded: Node = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(decoded, Node::Branch(entries));
    }

    #[test]
    fn node_cas_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("commits")).unwrap();

        // Test leaf node.
        let leaf = Node::Leaf(sample_leaf_entries());
        let cid = leaf.write_to_cas(dir.path()).unwrap();
        let loaded = Node::read_from_cas(dir.path(), cid).unwrap();
        assert_eq!(loaded, leaf);

        // Test branch node.
        let branch = Node::Branch(sample_branch_entries());
        let cid2 = branch.write_to_cas(dir.path()).unwrap();
        let loaded2 = Node::read_from_cas(dir.path(), cid2).unwrap();
        assert_eq!(loaded2, branch);
    }
}
