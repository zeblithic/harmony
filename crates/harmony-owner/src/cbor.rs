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
