use crate::OwnerError;
use ciborium::value::Value;
use serde::{de::DeserializeOwned, Serialize};

/// Encode a value to RFC 8949 §4.2 deterministic-CBOR bytes.
///
/// ciborium's standard `into_writer` produces shortest-form integers and
/// definite-length containers, but does NOT sort map keys. RFC 8949 §4.2
/// requires map keys to be sorted by length-then-bytewise lex order on
/// the canonically-encoded key bytes.
///
/// We achieve §4.2 compliance by:
/// 1. Serializing through `ciborium::value::Value` (an in-memory CBOR tree)
/// 2. Recursively walking the tree, sorting each map's entries by their
///    canonically-encoded key bytes
/// 3. Re-encoding the canonicalized tree via `into_writer`
///
/// This is overhead per encode (double pass) but our cert sizes are small
/// and encoding is not on a hot path — interop correctness matters more.
pub fn to_canonical<T: Serialize>(value: &T) -> Result<Vec<u8>, OwnerError> {
    let v = Value::serialized(value).map_err(|e| OwnerError::Cbor(e.to_string()))?;
    let canonical = canonicalize(v)?;
    let mut buf = Vec::new();
    ciborium::ser::into_writer(&canonical, &mut buf)
        .map_err(|e| OwnerError::Cbor(e.to_string()))?;
    Ok(buf)
}

pub fn from_bytes<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, OwnerError> {
    ciborium::de::from_reader(bytes).map_err(|e| OwnerError::Cbor(e.to_string()))
}

/// Recursively canonicalize a `Value`: sort map entries by encoded-key bytes
/// (length-then-bytewise lex), and recurse into nested arrays/maps.
fn canonicalize(v: Value) -> Result<Value, OwnerError> {
    match v {
        Value::Map(entries) => {
            let canon: Vec<(Value, Value)> = entries
                .into_iter()
                .map(|(k, v)| Ok::<_, OwnerError>((canonicalize(k)?, canonicalize(v)?)))
                .collect::<Result<Vec<_>, _>>()?;
            // Sort by canonically-encoded key bytes (RFC 8949 §4.2.1)
            let mut keyed: Vec<(Vec<u8>, (Value, Value))> = canon
                .into_iter()
                .map(|(k, v)| {
                    let mut buf = Vec::new();
                    ciborium::ser::into_writer(&k, &mut buf)
                        .map_err(|e| OwnerError::Cbor(e.to_string()))?;
                    Ok::<_, OwnerError>((buf, (k, v)))
                })
                .collect::<Result<Vec<_>, _>>()?;
            keyed.sort_by(|a, b| {
                // Length-first, then bytewise lex
                a.0.len().cmp(&b.0.len()).then_with(|| a.0.cmp(&b.0))
            });
            Ok(Value::Map(keyed.into_iter().map(|(_, kv)| kv).collect()))
        }
        Value::Array(items) => Ok(Value::Array(
            items
                .into_iter()
                .map(canonicalize)
                .collect::<Result<Vec<_>, _>>()?,
        )),
        Value::Tag(tag, inner) => Ok(Value::Tag(tag, Box::new(canonicalize(*inner)?))),
        // Leaf values pass through; ciborium's into_writer already handles
        // shortest-form integers and definite-length encoding for these.
        other => Ok(other),
    }
}

/// Serde codec: encode `[u8; 16]` as CBOR byte string (not an array of ints).
pub mod arr16 {
    use serde::{de::{Error, Visitor}, Deserializer, Serializer};
    use std::fmt;

    pub fn serialize<S: Serializer>(v: &[u8; 16], s: S) -> Result<S::Ok, S::Error> {
        s.serialize_bytes(v)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<[u8; 16], D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = [u8; 16];
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a 16-byte byte string")
            }
            fn visit_bytes<E: Error>(self, v: &[u8]) -> Result<Self::Value, E> {
                if v.len() != 16 {
                    return Err(E::invalid_length(v.len(), &self));
                }
                let mut out = [0u8; 16];
                out.copy_from_slice(v);
                Ok(out)
            }
            fn visit_borrowed_bytes<E: Error>(self, v: &'de [u8]) -> Result<Self::Value, E> {
                self.visit_bytes(v)
            }
            fn visit_byte_buf<E: Error>(self, v: Vec<u8>) -> Result<Self::Value, E> {
                self.visit_bytes(&v)
            }
        }
        d.deserialize_bytes(V)
    }
}

/// Serde codec: encode `Vec<[u8; 16]>` as a CBOR array of byte strings
/// (not an array of arrays of ints).
pub mod arr16_vec {
    use serde::{de::{Error, SeqAccess, Visitor}, ser::SerializeSeq, Deserializer, Serializer};
    use std::fmt;

    pub fn serialize<S: Serializer>(v: &Vec<[u8; 16]>, s: S) -> Result<S::Ok, S::Error> {
        let mut seq = s.serialize_seq(Some(v.len()))?;
        for arr in v {
            seq.serialize_element(serde_bytes::Bytes::new(arr))?;
        }
        seq.end()
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<[u8; 16]>, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = Vec<[u8; 16]>;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a sequence of 16-byte byte strings")
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let mut out = Vec::new();
                while let Some(b) = seq.next_element::<serde_bytes::ByteBuf>()? {
                    if b.len() != 16 {
                        return Err(A::Error::invalid_length(b.len(), &"16-byte byte string"));
                    }
                    let mut a = [0u8; 16];
                    a.copy_from_slice(&b);
                    out.push(a);
                }
                Ok(out)
            }
        }
        d.deserialize_seq(V)
    }
}

/// Serde codec: encode `Vec<Vec<u8>>` (variable-length signature list) as a
/// CBOR array of byte strings.
pub mod bytes_vec {
    use serde::{de::{SeqAccess, Visitor}, ser::SerializeSeq, Deserializer, Serializer};
    use std::fmt;

    pub fn serialize<S: Serializer>(v: &Vec<Vec<u8>>, s: S) -> Result<S::Ok, S::Error> {
        let mut seq = s.serialize_seq(Some(v.len()))?;
        for bytes in v {
            seq.serialize_element(serde_bytes::Bytes::new(bytes))?;
        }
        seq.end()
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<Vec<u8>>, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = Vec<Vec<u8>>;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a sequence of byte strings")
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let mut out = Vec::new();
                while let Some(b) = seq.next_element::<serde_bytes::ByteBuf>()? {
                    out.push(b.into_vec());
                }
                Ok(out)
            }
        }
        d.deserialize_seq(V)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Serialize, Deserialize, PartialEq)]
    struct Sample {
        a: u32,
        b: String,
    }

    #[test]
    fn roundtrip() {
        let v = Sample { a: 42, b: "hello".into() };
        let bytes = to_canonical(&v).unwrap();
        let decoded: Sample = from_bytes(&bytes).unwrap();
        assert_eq!(v, decoded);
    }

    #[test]
    fn deterministic_encoding() {
        let v = Sample { a: 42, b: "hello".into() };
        let b1 = to_canonical(&v).unwrap();
        let b2 = to_canonical(&v).unwrap();
        assert_eq!(b1, b2, "encoding must be deterministic");
    }

    #[test]
    fn arr16_encodes_as_byte_string() {
        use serde::Serialize;
        #[derive(Serialize, serde::Deserialize, PartialEq, Debug)]
        struct W(#[serde(with = "super::arr16")] [u8; 16]);
        let v = W([0xAB; 16]);
        let bytes = to_canonical(&v).unwrap();
        // CBOR major type 2 (bstr) with length 16 = 0x50; followed by 16 bytes
        assert_eq!(bytes[0], 0x50);
        assert_eq!(bytes.len(), 17);
        let decoded: W = from_bytes(&bytes).unwrap();
        assert_eq!(v, decoded);
    }

    #[test]
    fn arr16_vec_encodes_as_array_of_byte_strings() {
        use serde::Serialize;
        #[derive(Serialize, serde::Deserialize, PartialEq, Debug)]
        struct W(#[serde(with = "super::arr16_vec")] Vec<[u8; 16]>);
        let v = W(vec![[1u8; 16], [2u8; 16]]);
        let bytes = to_canonical(&v).unwrap();
        // Outer = CBOR array of length 2 (0x82); each element = bstr-16 (0x50 + 16 bytes)
        assert_eq!(bytes[0], 0x82);
        assert_eq!(bytes[1], 0x50);
        let decoded: W = from_bytes(&bytes).unwrap();
        assert_eq!(v, decoded);
    }

    #[test]
    fn bytes_vec_encodes_as_array_of_byte_strings() {
        use serde::Serialize;
        #[derive(Serialize, serde::Deserialize, PartialEq, Debug)]
        struct W(#[serde(with = "super::bytes_vec")] Vec<Vec<u8>>);
        let v = W(vec![vec![0xAA; 64], vec![0xBB; 64]]);
        let bytes = to_canonical(&v).unwrap();
        // Outer array of 2; each element bstr-64 (0x58 0x40 ... 64 bytes)
        assert_eq!(bytes[0], 0x82);
        assert_eq!(bytes[1], 0x58);  // bstr with 1-byte length
        assert_eq!(bytes[2], 0x40);  // length = 64
        let decoded: W = from_bytes(&bytes).unwrap();
        assert_eq!(v, decoded);
    }

    #[test]
    fn map_keys_sorted_canonically() {
        // Build two structs that ciborium serializes with different field orders;
        // both should produce IDENTICAL canonical bytes.
        #[derive(Serialize)]
        struct AB {
            a: u32,
            b: u32,
        }
        #[derive(Serialize)]
        struct BA {
            b: u32,
            a: u32,
        }
        let ab = AB { a: 1, b: 2 };
        let ba = BA { b: 2, a: 1 };
        let bytes_ab = to_canonical(&ab).unwrap();
        let bytes_ba = to_canonical(&ba).unwrap();
        assert_eq!(
            bytes_ab, bytes_ba,
            "field-order-different structs must canonicalize to the same bytes"
        );
    }
}
