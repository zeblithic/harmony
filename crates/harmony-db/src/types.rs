use harmony_content::ContentId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Consumer-extensible metadata stored alongside each entry in the index.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EntryMeta {
    /// Opaque bitfield for consumer use (e.g., bit 0 = read).
    pub flags: u64,
    /// Short summary for list views (truncated to 256 bytes on insert).
    pub snippet: String,
}

/// A single key-value pair in a table.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Entry {
    /// Arbitrary bytes, sorted lexicographically.
    #[serde(with = "hex_bytes")]
    pub key: Vec<u8>,
    /// BLAKE3 content address of the value bytes.
    #[serde(with = "hex_content_id")]
    pub value_cid: ContentId,
    /// Insertion time (UNIX seconds).
    pub timestamp: u64,
    /// Consumer-defined metadata.
    pub metadata: EntryMeta,
}

mod hex_bytes {
    use serde::{self, Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(bytes: &Vec<u8>, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&hex::encode(bytes))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<u8>, D::Error> {
        let s = String::deserialize(d)?;
        hex::decode(&s).map_err(serde::de::Error::custom)
    }
}

mod hex_content_id {
    use harmony_content::ContentId;
    use serde::{self, Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(cid: &ContentId, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&hex::encode(cid.to_bytes()))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<ContentId, D::Error> {
        let s = String::deserialize(d)?;
        let bytes: [u8; 32] = hex::decode(&s)
            .map_err(serde::de::Error::custom)?
            .try_into()
            .map_err(|_| serde::de::Error::custom("expected 64 hex chars for ContentId"))?;
        Ok(ContentId::from_bytes(bytes))
    }
}

/// Difference between two commits.
#[derive(Debug, Clone)]
pub struct Diff {
    pub tables: HashMap<String, TableDiff>,
}

/// Differences within a single table between two commits.
#[derive(Debug, Clone)]
pub struct TableDiff {
    pub added: Vec<Entry>,
    pub removed: Vec<Entry>,
    pub changed: Vec<(Entry, Entry)>,
}

/// Serde helper: serialize Option<ContentId> as nullable hex string.
pub(crate) fn ser_opt_cid<S: serde::Serializer>(
    cid: &Option<ContentId>,
    s: S,
) -> Result<S::Ok, S::Error> {
    match cid {
        Some(c) => s.serialize_str(&hex::encode(c.to_bytes())),
        None => s.serialize_none(),
    }
}

/// Serde helper: deserialize Option<ContentId> from nullable hex string.
pub(crate) fn de_opt_cid<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> Result<Option<ContentId>, D::Error> {
    let opt: Option<String> = Option::deserialize(d)?;
    match opt {
        None => Ok(None),
        Some(s) => {
            let bytes: [u8; 32] = hex::decode(&s)
                .map_err(serde::de::Error::custom)?
                .try_into()
                .map_err(|_| serde::de::Error::custom("expected 64 hex chars"))?;
            Ok(Some(ContentId::from_bytes(bytes)))
        }
    }
}
