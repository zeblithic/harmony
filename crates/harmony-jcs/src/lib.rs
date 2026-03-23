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

fn serialize_value(value: &serde_json::Value, buf: &mut Vec<u8>) {
    match value {
        serde_json::Value::Null => buf.extend_from_slice(b"null"),
        serde_json::Value::Bool(true) => buf.extend_from_slice(b"true"),
        serde_json::Value::Bool(false) => buf.extend_from_slice(b"false"),
        serde_json::Value::String(s) => serialize_string(s, buf),
        serde_json::Value::Number(n) => serialize_number(n, buf),
        serde_json::Value::Array(arr) => {
            buf.push(b'[');
            for (i, item) in arr.iter().enumerate() {
                if i > 0 {
                    buf.push(b',');
                }
                serialize_value(item, buf);
            }
            buf.push(b']');
        }
        serde_json::Value::Object(map) => {
            let mut entries: Vec<(&String, &serde_json::Value)> = map.iter().collect();
            entries.sort_by(|(a, _), (b, _)| utf16_cmp(a, b));
            buf.push(b'{');
            for (i, (key, val)) in entries.iter().enumerate() {
                if i > 0 {
                    buf.push(b',');
                }
                serialize_string(key, buf);
                buf.push(b':');
                serialize_value(val, buf);
            }
            buf.push(b'}');
        }
    }
}

fn serialize_string(s: &str, buf: &mut Vec<u8>) {
    buf.push(b'"');
    for ch in s.chars() {
        match ch {
            '\x08' => buf.extend_from_slice(b"\\b"),
            '\x09' => buf.extend_from_slice(b"\\t"),
            '\x0A' => buf.extend_from_slice(b"\\n"),
            '\x0C' => buf.extend_from_slice(b"\\f"),
            '\x0D' => buf.extend_from_slice(b"\\r"),
            '"' => buf.extend_from_slice(b"\\\""),
            '\\' => buf.extend_from_slice(b"\\\\"),
            '\x00'..='\x1F' => {
                buf.extend_from_slice(format!("\\u{:04x}", ch as u32).as_bytes());
            }
            _ => {
                let mut tmp = [0u8; 4];
                let encoded = ch.encode_utf8(&mut tmp);
                buf.extend_from_slice(encoded.as_bytes());
            }
        }
    }
    buf.push(b'"');
}

fn utf16_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    a.encode_utf16().cmp(b.encode_utf16())
}

fn serialize_number(n: &serde_json::Number, buf: &mut Vec<u8>) {
    buf.extend_from_slice(n.to_string().as_bytes());
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

    #[test]
    fn null_value() {
        let val = serde_json::Value::Null;
        assert_eq!(canonicalize(&val), b"null");
    }

    #[test]
    fn bool_true() {
        let val = serde_json::Value::Bool(true);
        assert_eq!(canonicalize(&val), b"true");
    }

    #[test]
    fn bool_false() {
        let val = serde_json::Value::Bool(false);
        assert_eq!(canonicalize(&val), b"false");
    }

    #[test]
    fn simple_string() {
        let val = serde_json::Value::String("hello".into());
        assert_eq!(canonicalize(&val), b"\"hello\"");
    }

    #[test]
    fn string_with_escapes() {
        let val = serde_json::Value::String("\x08\t\n\x0c\r\"\\".into());
        assert_eq!(canonicalize(&val), b"\"\\b\\t\\n\\f\\r\\\"\\\\\"");
    }

    #[test]
    fn string_control_char_hex_escape() {
        let val = serde_json::Value::String("\x01".into());
        assert_eq!(canonicalize(&val), b"\"\\u0001\"");
    }

    #[test]
    fn string_del_not_escaped() {
        let val = serde_json::Value::String("\x7f".into());
        assert_eq!(canonicalize(&val), b"\"\x7f\"");
    }

    #[test]
    fn string_unicode_passthrough() {
        let val = serde_json::Value::String("é".into());
        assert_eq!(canonicalize(&val), "\"é\"".as_bytes());
    }

    #[test]
    fn simple_array() {
        let val: serde_json::Value = serde_json::from_str("[1,2,3]").unwrap();
        assert_eq!(canonicalize(&val), b"[1,2,3]");
    }

    #[test]
    fn key_sorting_ascii() {
        let val: serde_json::Value = serde_json::from_str(r#"{"b":2,"a":1}"#).unwrap();
        assert_eq!(canonicalize(&val), b"{\"a\":1,\"b\":2}");
    }

    #[test]
    fn nested_structures() {
        let val: serde_json::Value =
            serde_json::from_str(r#"{"z":{"b":2,"a":1},"a":[3,1,2]}"#).unwrap();
        assert_eq!(
            canonicalize(&val),
            b"{\"a\":[3,1,2],\"z\":{\"a\":1,\"b\":2}}"
        );
    }

    #[test]
    fn key_sorting_unicode_utf16() {
        // U+10000 encodes as surrogate pair D800 DC00 in UTF-16 (sorts lower)
        // U+FEFF encodes as FEFF in UTF-16 (sorts higher)
        // But in UTF-8, U+10000 = F0 90 80 80 sorts HIGHER than U+FEFF = EF BB BF
        // JCS requires UTF-16 order, so U+10000 comes first
        let json = format!("{{\"{0}\":1,\"{1}\":2}}", '\u{FEFF}', '\u{10000}',);
        let val: serde_json::Value = serde_json::from_str(&json).unwrap();
        let result = String::from_utf8(canonicalize(&val)).unwrap();
        let expected = format!("{{\"{1}\":2,\"{0}\":1}}", '\u{FEFF}', '\u{10000}',);
        assert_eq!(result, expected);
    }
}
