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
                // Zero-allocation hex escape. Control chars are U+0000–U+001F,
                // so the upper two hex digits are always '0','0'.
                const HEX: &[u8; 16] = b"0123456789abcdef";
                let b = ch as u8;
                buf.extend_from_slice(&[
                    b'\\', b'u', b'0', b'0',
                    HEX[(b >> 4) as usize],
                    HEX[(b & 0xF) as usize],
                ]);
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

/// Maximum safe integer for IEEE 754 double (2^53 - 1).
/// Beyond this, i64/u64 values may not round-trip through f64 exactly,
/// so they must go through the f64 formatting path per RFC 8785 §3.2.2.3.
const MAX_SAFE_INTEGER: i64 = (1i64 << 53) - 1;

fn serialize_number(n: &serde_json::Number, buf: &mut Vec<u8>) {
    // Fast path for integers within the IEEE 754 safe range.
    // RFC 8785 requires all numbers to be treated as IEEE 754 doubles.
    // For integers within ±(2^53 - 1), the i64 representation is exact.
    if let Some(i) = n.as_i64() {
        if i.abs() <= MAX_SAFE_INTEGER {
            buf.extend_from_slice(i.to_string().as_bytes());
            return;
        }
    }
    if let Some(u) = n.as_u64() {
        if u <= MAX_SAFE_INTEGER as u64 {
            buf.extend_from_slice(u.to_string().as_bytes());
            return;
        }
    }

    let f = n.as_f64().expect("serde_json::Number is i64, u64, or f64");

    // Negative zero and positive zero both serialize as "0".
    if f == 0.0 {
        buf.push(b'0');
        return;
    }

    format_es6_double(f, buf);
}

/// Format an f64 as ES6 `Number.prototype.toString()` (ECMA-262 §7.1.12.1).
fn format_es6_double(f: f64, buf: &mut Vec<u8>) {
    let negative = f < 0.0;
    if negative {
        buf.push(b'-');
    }

    let mut ryu_buf = ryu::Buffer::new();
    let ryu_str = ryu_buf.format_finite(f.abs());

    let (digits, n) = parse_ryu_output(ryu_str);
    let k = digits.len() as i32;

    if k <= n && n <= 21 {
        // Integer with trailing zeros: e.g., 100, 100000000000000000000
        buf.extend_from_slice(digits.as_bytes());
        for _ in 0..(n - k) {
            buf.push(b'0');
        }
    } else if 0 < n && n <= 21 {
        // Decimal notation: e.g., 1.5, 123.456
        buf.extend_from_slice(&digits.as_bytes()[..n as usize]);
        buf.push(b'.');
        buf.extend_from_slice(&digits.as_bytes()[n as usize..]);
    } else if -6 < n && n <= 0 {
        // Leading zeros: e.g., 0.001, 0.000001
        buf.extend_from_slice(b"0.");
        for _ in 0..(-n) {
            buf.push(b'0');
        }
        buf.extend_from_slice(digits.as_bytes());
    } else if k == 1 {
        // Single-digit exponential: e.g., 1e+21, 5e-7
        buf.push(digits.as_bytes()[0]);
        buf.push(b'e');
        let e = n - 1;
        if e > 0 {
            buf.push(b'+');
        }
        buf.extend_from_slice(e.to_string().as_bytes());
    } else {
        // Multi-digit exponential: e.g., 1.5e+20
        buf.push(digits.as_bytes()[0]);
        buf.push(b'.');
        buf.extend_from_slice(&digits.as_bytes()[1..]);
        buf.push(b'e');
        let e = n - 1;
        if e > 0 {
            buf.push(b'+');
        }
        buf.extend_from_slice(e.to_string().as_bytes());
    }
}

/// Parse a ryu-formatted finite float string into `(significant_digits, n)`.
///
/// `n` is the decimal point position such that `value = digits × 10^(n−k)`,
/// where `k = digits.len()`.  Examples:
/// - `"1.5"`   → `("15", 1)`
/// - `"100.0"` → `("1", 3)` (trailing zeros removed)
/// - `"1e20"`  → `("1", 21)`
/// - `"1e-7"`  → `("1", -6)`
/// - `"1.5e20"` → `("15", 21)`
fn parse_ryu_output(s: &str) -> (String, i32) {
    // Split on 'e' / 'E' to get mantissa and optional exponent.
    let (mantissa, exp_part) = if let Some(pos) = s.find(['e', 'E']) {
        (&s[..pos], s[pos + 1..].parse::<i32>().unwrap())
    } else {
        (s, 0)
    };

    if let Some(dot_pos) = mantissa.find('.') {
        // Mantissa has a decimal point, e.g. "1.5" or "100.0".
        let digits: String = mantissa.chars().filter(|&c| c != '.').collect();
        // n = number of digits before the decimal + exponent
        let n = exp_part + dot_pos as i32;
        let trimmed = digits.trim_end_matches('0');
        if trimmed.is_empty() {
            ("0".to_string(), 1)
        } else {
            (trimmed.to_string(), n)
        }
    } else {
        // No decimal point, e.g. "1", "100", "1e20".
        let trimmed = mantissa.trim_end_matches('0');
        if trimmed.is_empty() {
            ("0".to_string(), 1)
        } else {
            // n = total digits in mantissa + exponent
            (trimmed.to_string(), exp_part + mantissa.len() as i32)
        }
    }
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
    fn number_integer() {
        let val: serde_json::Value = serde_json::from_str("1").unwrap();
        assert_eq!(canonicalize(&val), b"1");
    }

    #[test]
    fn number_negative() {
        let val: serde_json::Value = serde_json::from_str("-1").unwrap();
        assert_eq!(canonicalize(&val), b"-1");
    }

    #[test]
    fn number_decimal() {
        let val: serde_json::Value = serde_json::from_str("1.5").unwrap();
        assert_eq!(canonicalize(&val), b"1.5");
    }

    #[test]
    fn number_negative_zero() {
        let val: serde_json::Value = serde_json::from_str("-0").unwrap();
        assert_eq!(canonicalize(&val), b"0");
    }

    #[test]
    fn number_large_no_exponential() {
        // 10^20 should NOT use exponential (threshold is 10^21)
        let val: serde_json::Value = serde_json::from_str("1e20").unwrap();
        assert_eq!(canonicalize(&val), b"100000000000000000000");
    }

    #[test]
    fn number_large_exponential() {
        // 10^21 SHOULD use exponential
        let val: serde_json::Value = serde_json::from_str("1e21").unwrap();
        assert_eq!(canonicalize(&val), b"1e+21");
    }

    #[test]
    fn number_small_no_exponential() {
        let val: serde_json::Value = serde_json::from_str("0.000001").unwrap();
        assert_eq!(canonicalize(&val), b"0.000001");
    }

    #[test]
    fn number_small_exponential() {
        let val: serde_json::Value = serde_json::from_str("1e-7").unwrap();
        assert_eq!(canonicalize(&val), b"1e-7");
    }

    #[test]
    fn number_float_integer_value() {
        // A float that is an exact integer should have no decimal point
        let val: serde_json::Value = serde_json::from_str("1.0").unwrap();
        assert_eq!(canonicalize(&val), b"1");
    }

    #[test]
    fn number_max_safe_integer() {
        let val: serde_json::Value = serde_json::from_str("9007199254740991").unwrap();
        assert_eq!(canonicalize(&val), b"9007199254740991");
    }

    #[test]
    fn number_pi() {
        let val: serde_json::Value = serde_json::from_str("3.141592653589793").unwrap();
        assert_eq!(canonicalize(&val), b"3.141592653589793");
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

    /// RFC 8785 — key ordering with special characters
    #[test]
    fn rfc8785_structure_key_ordering() {
        use serde_json::json;
        let val = json!({
            "\r": "Carriage Return",
            "1": "One",
            "": "Empty"
        });
        let result = String::from_utf8(canonicalize(&val)).unwrap();
        // UTF-16 order: "" (empty) < "\r" (000D) < "1" (0031)
        assert_eq!(result, r#"{"":"Empty","\r":"Carriage Return","1":"One"}"#);
    }

    /// Verify Unicode key ordering above BMP
    #[test]
    fn rfc8785_structure_unicode_keys() {
        use serde_json::json;
        let val = json!({
            "\u{20ac}": "Euro Sign",
            "\u{00f6}": "Latin Small O Diaeresis"
        });
        let result = String::from_utf8(canonicalize(&val)).unwrap();
        // UTF-16 order: U+00F6 (00F6) < U+20AC (20AC)
        let f6_pos = result.find('\u{00f6}').unwrap();
        let euro_pos = result.find('\u{20ac}').unwrap();
        assert!(
            f6_pos < euro_pos,
            "U+00F6 should sort before U+20AC in UTF-16"
        );
    }

    /// Batch number vector tests from RFC 8785 Appendix B
    #[test]
    fn rfc8785_number_vectors() {
        let cases = vec![
            ("0", "0"),
            ("0.0", "0"),
            ("-0", "0"),
            ("-0.0", "0"),
            ("1", "1"),
            ("-1", "-1"),
            ("1.5", "1.5"),
            ("1e20", "100000000000000000000"),
            ("1e21", "1e+21"),
            ("-1e21", "-1e+21"),
            ("1e-6", "0.000001"),
            ("1e-7", "1e-7"),
            ("9007199254740991", "9007199254740991"),
            ("-9007199254740991", "-9007199254740991"),
            ("0.1", "0.1"),
            // serde_json parses 999999999999999900000 to the f64 1e21 (the nearest
            // representable double), so the canonical form is "1e+21".
            ("999999999999999900000", "1e+21"),
            ("1e+23", "1e+23"),
        ];

        for (input, expected) in cases {
            let val: serde_json::Value = serde_json::from_str(input).unwrap();
            let result = String::from_utf8(canonicalize(&val)).unwrap();
            assert_eq!(result, expected, "input: {input}");
        }
    }

    #[test]
    fn no_whitespace() {
        let input = r#"{
            "a" : 1 ,
            "b" : [ 2 , 3 ]
        }"#;
        let val: serde_json::Value = serde_json::from_str(input).unwrap();
        let result = canonicalize(&val);
        assert!(!result.contains(&b' '));
        assert!(!result.contains(&b'\n'));
        assert!(!result.contains(&b'\t'));
    }

    /// Deeply nested structure
    #[test]
    fn deeply_nested() {
        let val: serde_json::Value =
            serde_json::from_str(r#"{"a":{"b":{"c":{"d":"deep"}}}}"#).unwrap();
        assert_eq!(
            canonicalize(&val),
            b"{\"a\":{\"b\":{\"c\":{\"d\":\"deep\"}}}}"
        );
    }

    /// Mixed types in array
    #[test]
    fn mixed_array() {
        let val: serde_json::Value =
            serde_json::from_str(r#"[null, true, false, 42, "hello", {}, []]"#).unwrap();
        assert_eq!(canonicalize(&val), b"[null,true,false,42,\"hello\",{},[]]");
    }

    /// String with all control characters U+0000 through U+001F
    #[test]
    fn all_control_characters() {
        // Build a string with all 32 control characters
        let mut s = String::new();
        for i in 0u8..=0x1F {
            s.push(char::from(i));
        }
        let val = serde_json::Value::String(s);
        let result = String::from_utf8(canonicalize(&val)).unwrap();

        // Verify shorthand escapes are used where applicable
        assert!(result.contains("\\b")); // U+0008
        assert!(result.contains("\\t")); // U+0009
        assert!(result.contains("\\n")); // U+000A
        assert!(result.contains("\\f")); // U+000C
        assert!(result.contains("\\r")); // U+000D

        // Verify hex escapes for others
        assert!(result.contains("\\u0000")); // U+0000
        assert!(result.contains("\\u0001")); // U+0001
        assert!(result.contains("\\u001f")); // U+001F (lowercase hex)

        // Verify NO shorthand for U+000B (vertical tab) — should be \u000b
        assert!(result.contains("\\u000b"));
    }

    /// Empty string
    #[test]
    fn empty_string() {
        let val = serde_json::Value::String(String::new());
        assert_eq!(canonicalize(&val), b"\"\"");
    }

    /// Canonical output is idempotent
    #[test]
    fn idempotent() {
        let input = r#"{"z":1,"a":2,"m":{"b":3,"a":4}}"#;
        let val: serde_json::Value = serde_json::from_str(input).unwrap();
        let first = canonicalize(&val);
        // Parse the canonical output and re-canonicalize
        let val2: serde_json::Value = serde_json::from_slice(&first).unwrap();
        let second = canonicalize(&val2);
        assert_eq!(first, second, "canonicalization should be idempotent");
    }
}
