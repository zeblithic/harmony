//! RFC 8785 JSON Canonicalization Scheme.
//!
//! Provides deterministic JSON serialization for cryptographic signing.
//! The canonical form sorts object keys by UTF-16 code unit order and
//! formats numbers per ES6 `Number.prototype.toString()` (ECMA-262 7.1.12.1).

/// Serialize a JSON value to its RFC 8785 canonical form.
pub fn canonicalize(value: &serde_json::Value) -> Vec<u8> {
    let mut buf = Vec::new();
    serialize_value(value, &mut buf);
    buf
}

fn serialize_value(_value: &serde_json::Value, _buf: &mut Vec<u8>) {
    // TODO: implement in Task 2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_object() {
        let val: serde_json::Value = serde_json::from_str("{}").unwrap();
        assert_eq!(canonicalize(&val), b"{}");
    }

    #[test]
    fn empty_array() {
        let val: serde_json::Value = serde_json::from_str("[]").unwrap();
        assert_eq!(canonicalize(&val), b"[]");
    }
}
