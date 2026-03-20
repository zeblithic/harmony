//! Shared serde helpers for format-agnostic byte deserialization.

use alloc::vec::Vec;
use serde::de::{self, SeqAccess, Visitor};

/// A serde visitor that accepts both bytes and sequences of u8.
///
/// Handles the type mismatch between `serialize_bytes` (serde bytes type)
/// and formats that may deserialize the same data as a sequence of u8.
/// Without this, serializing via `serialize_bytes` and deserializing via
/// `Vec<u8>` (which uses `deserialize_seq`) would fail on formats like
/// JSON or CBOR where bytes and sequences encode differently.
pub(crate) struct BytesOrSeqVisitor;

impl<'de> Visitor<'de> for BytesOrSeqVisitor {
    type Value = Vec<u8>;

    fn expecting(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.write_str("byte array")
    }

    fn visit_bytes<E: de::Error>(self, v: &[u8]) -> Result<Self::Value, E> {
        Ok(v.to_vec())
    }

    fn visit_byte_buf<E: de::Error>(self, v: Vec<u8>) -> Result<Self::Value, E> {
        Ok(v)
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let mut bytes = Vec::with_capacity(seq.size_hint().unwrap_or(0));
        while let Some(byte) = seq.next_element()? {
            bytes.push(byte);
        }
        Ok(bytes)
    }
}
