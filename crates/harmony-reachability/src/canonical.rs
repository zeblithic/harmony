//! Byte-stable CBOR encode/decode + the byte-string serde helpers used by the
//! reachability record. Moved verbatim from harmony-client
//! (`owner_state_crypto::canonical_cbor_encode` + `owner_state_types` bstr
//! helpers) so the on-wire encoding is byte-identical; see the record's golden
//! vector for the compat lock.

use serde::{de::DeserializeOwned, Deserializer, Serialize, Serializer};

#[derive(Debug, thiserror::Error)]
pub enum CborError {
    #[error("CBOR encode failed: {0}")]
    Encode(String),
    #[error("CBOR decode failed: {0}")]
    Decode(String),
}

/// Deterministic CBOR encode: a thin `ciborium::into_writer` wrapper. Byte-stable
/// across instances for a serde tree with same-length map keys at each level, no
/// f32/f64, no HashMap, no CBOR tags (the reachability record satisfies this).
pub fn canonical_cbor_encode<T: Serialize>(value: &T) -> Result<Vec<u8>, CborError> {
    let mut buf = Vec::new();
    ciborium::into_writer(value, &mut buf).map_err(|e| CborError::Encode(format!("{e}")))?;
    Ok(buf)
}

/// Decoder paired with [`canonical_cbor_encode`]; rejects trailing bytes so a
/// fingerprinted/signed encoding can't be extended to a distinct byte sequence
/// that decodes to the same value.
pub fn canonical_cbor_decode<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, CborError> {
    let mut cursor = std::io::Cursor::new(bytes);
    let value = ciborium::from_reader(&mut cursor).map_err(|e| CborError::Decode(format!("{e}")))?;
    if cursor.position() as usize != bytes.len() {
        return Err(CborError::Decode(format!(
            "trailing bytes after canonical value: consumed {} of {}",
            cursor.position(),
            bytes.len()
        )));
    }
    Ok(value)
}

/// Serialize a byte array as a CBOR byte string (major type 2), NOT an array of
/// u8. Load-bearing for wire compat — see the record's golden vector.
pub fn serialize_bytes_as_bstr<const N: usize, S>(b: &[u8; N], s: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    s.serialize_bytes(b)
}

/// Deserialize a CBOR byte string into a fixed-size array.
pub fn deserialize_bytes_from_bstr<'de, const N: usize, D>(d: D) -> Result<[u8; N], D::Error>
where
    D: Deserializer<'de>,
{
    use serde::de::Visitor;
    use std::fmt;

    struct BytesVisitor<const N: usize>;

    impl<'de, const N: usize> Visitor<'de> for BytesVisitor<N> {
        type Value = [u8; N];

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            write!(formatter, "a byte array of length {}", N)
        }

        fn visit_bytes<E>(self, value: &[u8]) -> Result<[u8; N], E>
        where
            E: serde::de::Error,
        {
            if value.len() != N {
                return Err(E::custom(format!("expected {} bytes, got {}", N, value.len())));
            }
            let mut arr = [0u8; N];
            arr.copy_from_slice(value);
            Ok(arr)
        }

        fn visit_byte_buf<E>(self, v: Vec<u8>) -> Result<[u8; N], E>
        where
            E: serde::de::Error,
        {
            self.visit_bytes(&v)
        }
    }

    d.deserialize_bytes(BytesVisitor::<N>)
}

/// `skip_serializing_if` predicate: a zero stamp means "no butler set" and is
/// elided so legacy blobs stay byte-identical.
pub fn is_zero_u64(v: &u64) -> bool {
    *v == 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Serialize, Deserialize, PartialEq, Debug)]
    struct Probe {
        #[serde(serialize_with = "serialize_bytes_as_bstr", deserialize_with = "deserialize_bytes_from_bstr")]
        b: [u8; 4],
    }

    #[test]
    fn bstr_is_major_type_2_and_round_trips() {
        let p = Probe { b: [1, 2, 3, 4] };
        let bytes = canonical_cbor_encode(&p).expect("encode");
        // map(1) "b" -> bstr(4) 01020304. The bstr header for a 4-byte string is
        // 0x44 (major type 2, length 4) — NOT 0x84 (major type 4, array len 4).
        assert!(bytes.windows(1).any(|w| w[0] == 0x44), "value must encode as a CBOR byte string");
        let back: Probe = canonical_cbor_decode(&bytes).expect("decode");
        assert_eq!(back, p);
    }

    #[test]
    fn decode_rejects_trailing_bytes() {
        let p = Probe { b: [9; 4] };
        let mut bytes = canonical_cbor_encode(&p).expect("encode");
        bytes.push(0x00);
        assert!(canonical_cbor_decode::<Probe>(&bytes).is_err());
    }
}
