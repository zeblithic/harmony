use crate::OwnerError;
use serde::{de::DeserializeOwned, Serialize};

/// Encode a value to deterministic (canonical) CBOR bytes.
///
/// Per RFC 8949 §4.2: shortest-form integers, definite-length containers,
/// keys sorted by major type then byte-wise lexicographic order. ciborium
/// produces canonical output by default for these constraints.
pub fn to_canonical<T: Serialize>(value: &T) -> Result<Vec<u8>, OwnerError> {
    let mut buf = Vec::new();
    ciborium::ser::into_writer(value, &mut buf).map_err(|e| OwnerError::Cbor(e.to_string()))?;
    Ok(buf)
}

pub fn from_bytes<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, OwnerError> {
    ciborium::de::from_reader(bytes).map_err(|e| OwnerError::Cbor(e.to_string()))
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
}
