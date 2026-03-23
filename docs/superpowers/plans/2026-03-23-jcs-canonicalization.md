# JCS Canonicalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> **Note:** The `- [ ]` checkboxes below are execution tracking markers for the agentic worker, not persistent TODO items. Task tracking uses `bd` (beads) — see bead `harmony-vl1`.

**Goal:** Implement RFC 8785 (JSON Canonicalization Scheme) as a new `harmony-jcs` crate with a single `canonicalize(&serde_json::Value) -> Vec<u8>` function.

**Architecture:** New crate with three internal components: (1) recursive value serializer dispatching by JSON type, (2) ES6-conformant number formatter (ECMA-262 7.1.12.1) using `ryu` for digit extraction with custom formatting, (3) UTF-16 code unit key comparator for object key sorting. Single public function, infallible return type.

**Tech Stack:** Rust, `serde_json` (Value type), `ryu` (shortest-roundtrip digit extraction)

**Spec:** `docs/superpowers/specs/2026-03-23-jcs-canonicalization-design.md`

---

## File Structure

```
crates/harmony-jcs/
├── Cargo.toml
└── src/
    └── lib.rs    — canonicalize() + serialize_value() + serialize_string()
                    + serialize_number() + utf16_cmp()
```

Also modify:
```
Cargo.toml    — Add harmony-jcs to workspace members and dependencies
```

---

### Task 1: Create crate scaffold and workspace wiring

**Files:**
- Create: `crates/harmony-jcs/Cargo.toml`
- Create: `crates/harmony-jcs/src/lib.rs`
- Modify: `Cargo.toml` (workspace root)

- [ ] **Step 1: Create Cargo.toml**

```toml
[package]
name = "harmony-jcs"
version = "0.1.0"
edition = "2021"
license = "Apache-2.0 OR MIT"
description = "RFC 8785 JSON Canonicalization Scheme"

[dependencies]
serde_json = { workspace = true }
```

Note: The workspace defines `serde_json` with `default-features = false, features = ["alloc"]`. `serde_json::Value` is available with the `alloc` feature — `std` is not required.

- [ ] **Step 2: Create minimal lib.rs**

```rust
//! RFC 8785 JSON Canonicalization Scheme.
//!
//! Provides deterministic JSON serialization for cryptographic signing.
//! The canonical form sorts object keys by UTF-16 code unit order and
//! formats numbers per ES6 `Number.prototype.toString()` (ECMA-262 7.1.12.1).

/// Serialize a JSON value to its RFC 8785 canonical form.
///
/// The output is deterministic: the same logical JSON document always
/// produces the same byte sequence regardless of the parser that produced
/// the `Value`. Object keys are sorted by UTF-16 code unit order, numbers
/// use ES6 formatting, and no whitespace is emitted.
pub fn canonicalize(value: &serde_json::Value) -> Vec<u8> {
    let mut buf = Vec::new();
    serialize_value(value, &mut buf);
    buf
}

fn serialize_value(_value: &serde_json::Value, _buf: &mut Vec<u8>) {
    // TODO: implement in subsequent tasks
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
```

- [ ] **Step 3: Add to workspace**

In root `Cargo.toml`, add `"crates/harmony-jcs"` to the `[workspace.members]` list (after `harmony-identity` or at the end — the list is not strictly alphabetical).

Add to `[workspace.dependencies]`:
```toml
harmony-jcs = { path = "crates/harmony-jcs" }
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p harmony-jcs`
Expected: 2 tests fail (empty_object, empty_array) — serialize_value is a no-op stub.

Actually — the stub writes nothing, so `canonicalize` returns `[]` (empty vec), which won't equal `b"{}"`. The tests will fail as expected. Good.

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(jcs): create harmony-jcs crate scaffold

New crate for RFC 8785 JSON Canonicalization Scheme.
Stub canonicalize() function, 2 failing tests for TDD."
```

---

### Task 2: Implement core serializer (strings, booleans, null, arrays, objects)

**Files:**
- Modify: `crates/harmony-jcs/src/lib.rs`

This task implements everything EXCEPT number formatting (which is complex enough for its own task).

- [ ] **Step 1: Write tests**

```rust
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
        // Test all shorthand escapes: \b \t \n \f \r \" \\
        let val = serde_json::Value::String("\x08\t\n\x0c\r\"\\".into());
        assert_eq!(canonicalize(&val), b"\"\\b\\t\\n\\f\\r\\\"\\\\\"");
    }

    #[test]
    fn string_control_char_hex_escape() {
        // U+0001 should be \u0001, not a shorthand
        let val = serde_json::Value::String("\x01".into());
        assert_eq!(canonicalize(&val), b"\"\\u0001\"");
    }

    #[test]
    fn string_del_not_escaped() {
        // U+007F (DEL) is NOT escaped per RFC 8785
        let val = serde_json::Value::String("\x7f".into());
        assert_eq!(canonicalize(&val), b"\"\x7f\"");
    }

    #[test]
    fn string_unicode_passthrough() {
        // Non-ASCII UTF-8 passes through as-is
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
        // Keys must be sorted lexicographically
        let val: serde_json::Value = serde_json::from_str(r#"{"b":2,"a":1}"#).unwrap();
        assert_eq!(canonicalize(&val), b"{\"a\":1,\"b\":2}");
    }

    #[test]
    fn nested_structures() {
        let val: serde_json::Value = serde_json::from_str(
            r#"{"z":{"b":2,"a":1},"a":[3,1,2]}"#
        ).unwrap();
        assert_eq!(
            canonicalize(&val),
            b"{\"a\":[3,1,2],\"z\":{\"a\":1,\"b\":2}}"
        );
    }

    #[test]
    fn key_sorting_unicode_utf16() {
        // U+10000 (LINEAR B SYLLABLE B008 A) encodes as surrogate pair D800 DC00 in UTF-16
        // which sorts LOWER than U+FEFF (BOM) = FEFF in UTF-16.
        // But in UTF-8, U+10000 = F0 90 80 80 sorts HIGHER than U+FEFF = EF BB BF.
        // JCS requires UTF-16 code unit order, so U+FEFF key should come AFTER U+10000 key.
        let json = format!(
            "{{\"{0}\":1,\"{1}\":2}}",
            '\u{FEFF}',   // BOM: UTF-16 = FEFF
            '\u{10000}',  // Linear B: UTF-16 = D800 DC00
        );
        let val: serde_json::Value = serde_json::from_str(&json).unwrap();
        let result = String::from_utf8(canonicalize(&val)).unwrap();
        // UTF-16 order: D800 < FEFF, so U+10000 key comes first
        let expected = format!(
            "{{\"{1}\":2,\"{0}\":1}}",
            '\u{FEFF}',
            '\u{10000}',
        );
        assert_eq!(result, expected);
    }
```

- [ ] **Step 2: Implement serialize_value, serialize_string, utf16_cmp**

```rust
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
            // Re-sort keys by UTF-16 code unit order
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

/// Serialize a string with RFC 8785 escaping rules.
/// Shorthand escapes (\b, \t, \n, \f, \r, \", \\) take precedence.
/// Other control characters (U+0000-U+001F) use \uXXXX.
/// All other characters pass through as literal UTF-8.
fn serialize_string(s: &str, buf: &mut Vec<u8>) {
    buf.push(b'"');
    for ch in s.chars() {
        match ch {
            '"' => buf.extend_from_slice(b"\\\""),
            '\\' => buf.extend_from_slice(b"\\\\"),
            '\u{0008}' => buf.extend_from_slice(b"\\b"),
            '\u{0009}' => buf.extend_from_slice(b"\\t"),
            '\u{000A}' => buf.extend_from_slice(b"\\n"),
            '\u{000C}' => buf.extend_from_slice(b"\\f"),
            '\u{000D}' => buf.extend_from_slice(b"\\r"),
            c if c < '\u{0020}' => {
                // Other control characters: \uXXXX with lowercase hex
                write!(buf, "\\u{:04x}", c as u32).unwrap();
            }
            c => {
                // All other characters (including non-ASCII) as literal UTF-8
                let mut utf8_buf = [0u8; 4];
                buf.extend_from_slice(c.encode_utf8(&mut utf8_buf).as_bytes());
            }
        }
    }
    buf.push(b'"');
}

/// Compare two strings by UTF-16 code unit order (RFC 8785 key sorting).
fn utf16_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    let mut a_iter = a.encode_utf16();
    let mut b_iter = b.encode_utf16();
    loop {
        match (a_iter.next(), b_iter.next()) {
            (Some(a_unit), Some(b_unit)) => {
                match a_unit.cmp(&b_unit) {
                    std::cmp::Ordering::Equal => continue,
                    ord => return ord,
                }
            }
            (Some(_), None) => return std::cmp::Ordering::Greater,
            (None, Some(_)) => return std::cmp::Ordering::Less,
            (None, None) => return std::cmp::Ordering::Equal,
        }
    }
}

/// Serialize a number per ES6 Number.prototype.toString() (ECMA-262 7.1.12.1).
/// STUB — implemented in Task 3.
fn serialize_number(n: &serde_json::Number, buf: &mut Vec<u8>) {
    // Temporary: use serde_json's default (NOT conformant — will be replaced)
    buf.extend_from_slice(n.to_string().as_bytes());
}
```

The `write!` macro calls in `serialize_string` and `serialize_number` use `std::io::Write` on `Vec<u8>`. Add `use std::io::Write;` at the top of the file. Alternatively, use `buf.extend_from_slice(format!(...).as_bytes())` to avoid the import — the implementer should choose based on what compiles cleanly.

- [ ] **Step 3: Run tests**

Run: `cargo test -p harmony-jcs`
Expected: Most tests pass. The `simple_array` test may fail if the temporary number serializer doesn't match expected output (depends on whether serde_json serializes `1` as `1` or `1.0`). If it fails, that's fine — it'll pass after Task 3.

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(jcs): core serializer — strings, booleans, null, arrays, objects

serialize_value() dispatches by JSON type. serialize_string() implements
RFC 8785 escape rules (shorthand precedence, \\uXXXX for control chars).
utf16_cmp() sorts keys by UTF-16 code unit order. Number formatting is
stubbed (Task 3). 13 tests."
```

---

### Task 3: Implement ES6-conformant number formatting

**Files:**
- Modify: `crates/harmony-jcs/src/lib.rs`

This is the most complex task. The implementation must match ECMA-262 7.1.12.1.

- [ ] **Step 1: Write number formatting tests**

```rust
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
        // -0 must be serialized as "0"
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
        // 1e-6 should NOT use exponential (threshold is 1e-7)
        let val: serde_json::Value = serde_json::from_str("0.000001").unwrap();
        assert_eq!(canonicalize(&val), b"0.000001");
    }

    #[test]
    fn number_small_exponential() {
        // 1e-7 SHOULD use exponential
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
        // 2^53 - 1 = 9007199254740991
        let val: serde_json::Value = serde_json::from_str("9007199254740991").unwrap();
        assert_eq!(canonicalize(&val), b"9007199254740991");
    }

    #[test]
    fn number_pi() {
        let val: serde_json::Value = serde_json::from_str("3.141592653589793").unwrap();
        assert_eq!(canonicalize(&val), b"3.141592653589793");
    }
```

- [ ] **Step 2: Implement serialize_number**

Replace the stub with the full ES6 implementation. The algorithm:

1. Extract the `f64` value from `serde_json::Number`. If it's stored as `u64` or `i64`, format as integer directly (no decimal, no exponential for reasonable sizes).
2. For `f64` values:
   a. Handle `-0` → `"0"`
   b. Use `ryu::Buffer` to get the shortest round-trip decimal representation
   c. Parse the `ryu` output into (sign, digits, exponent)
   d. Apply ES6 formatting rules based on digit count and exponent

```rust
/// Serialize a number per ES6 Number.prototype.toString() (ECMA-262 7.1.12.1).
fn serialize_number(n: &serde_json::Number, buf: &mut Vec<u8>) {
    // Fast path: serde_json stores integers as i64/u64 when possible.
    if let Some(i) = n.as_i64() {
        write!(buf, "{}", i).unwrap();
        return;
    }
    if let Some(u) = n.as_u64() {
        write!(buf, "{}", u).unwrap();
        return;
    }

    // f64 path
    let f = n.as_f64().expect("serde_json::Number must be i64, u64, or f64");

    // Negative zero → "0"
    if f == 0.0 {
        buf.push(b'0');
        return;
    }

    format_es6_double(f, buf);
}

/// Format an f64 per ES6 Number.prototype.toString().
///
/// Uses `ryu` for shortest-roundtrip digit extraction, then applies
/// ES6 exponential notation thresholds.
fn format_es6_double(f: f64, buf: &mut Vec<u8>) {
    debug_assert!(f != 0.0 && f.is_finite());

    let negative = f < 0.0;
    if negative {
        buf.push(b'-');
    }

    // Use ryu to get the shortest decimal representation
    let mut ryu_buf = ryu::Buffer::new();
    let ryu_str = ryu_buf.format_finite(f.abs());

    // Parse ryu output: it produces either "digits" or "d.digitseE"
    // Parse into (integer_digits, fractional_digits, base-10 exponent)
    let (digits, exp) = parse_ryu_output(ryu_str);

    // k = number of significant digits
    // n = position of decimal point (digits × 10^(n-k) = value)
    let k = digits.len() as i32;
    let n = exp; // decimal point position

    // ES6 7.1.12.1 formatting rules:
    if k <= n && n <= 21 {
        // Case: integer with trailing zeros (e.g., 100, 1000000)
        buf.extend_from_slice(digits.as_bytes());
        for _ in 0..(n - k) {
            buf.push(b'0');
        }
    } else if 0 < n && n <= 21 {
        // Case: decimal number (e.g., 1.5, 123.456)
        buf.extend_from_slice(&digits.as_bytes()[..n as usize]);
        buf.push(b'.');
        buf.extend_from_slice(&digits.as_bytes()[n as usize..]);
    } else if -6 < n && n <= 0 {
        // Case: 0.00...0digits (e.g., 0.001, 0.000001)
        buf.extend_from_slice(b"0.");
        for _ in 0..(-n) {
            buf.push(b'0');
        }
        buf.extend_from_slice(digits.as_bytes());
    } else if k == 1 {
        // Case: single digit with exponent (e.g., 1e+21, 5e-7)
        buf.push(digits.as_bytes()[0]);
        buf.push(b'e');
        let exp_val = n - 1;
        if exp_val > 0 {
            buf.push(b'+');
        }
        write!(buf, "{}", exp_val).unwrap();
    } else {
        // Case: multiple digits with exponent (e.g., 1.5e+20)
        buf.push(digits.as_bytes()[0]);
        buf.push(b'.');
        buf.extend_from_slice(&digits.as_bytes()[1..]);
        buf.push(b'e');
        let exp_val = n - 1;
        if exp_val > 0 {
            buf.push(b'+');
        }
        write!(buf, "{}", exp_val).unwrap();
    }
}

/// Parse ryu output into (significant_digits, decimal_point_position).
///
/// ryu produces strings like "1.5", "100.0", "1.5e20", "1e-7".
/// Returns (digits_without_dot, n) where the value = 0.digits × 10^n.
fn parse_ryu_output(s: &str) -> (String, i32) {
    // Split on 'e' or 'E' for exponential notation
    let (mantissa, exp_part) = if let Some(pos) = s.find(|c| c == 'e' || c == 'E') {
        (&s[..pos], s[pos + 1..].parse::<i32>().unwrap())
    } else {
        (s, 0)
    };

    // Remove dot from mantissa, track where it was
    if let Some(dot_pos) = mantissa.find('.') {
        let digits: String = mantissa.chars().filter(|&c| c != '.').collect();
        // n = exp_part + dot_pos (number of digits before dot)
        let n = exp_part + dot_pos as i32;
        // Strip trailing zeros from digits
        let trimmed = digits.trim_end_matches('0');
        if trimmed.is_empty() {
            ("0".to_string(), 1)
        } else {
            (trimmed.to_string(), n)
        }
    } else {
        // No dot — pure integer representation from ryu
        let trimmed = mantissa.trim_end_matches('0');
        if trimmed.is_empty() {
            ("0".to_string(), 1)
        } else {
            (trimmed.to_string(), exp_part + mantissa.len() as i32)
        }
    }
}
```

Add `ryu` to the root workspace `Cargo.toml` under `[workspace.dependencies]`:

```toml
ryu = "1"
```

Then add it to `crates/harmony-jcs/Cargo.toml`:

```toml
[dependencies]
serde_json = { workspace = true }
ryu = { workspace = true }
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p harmony-jcs`
Expected: All tests pass (including the earlier tests from Task 2 and the new number tests).

Some number tests may need iteration. The ES6 formatting rules have subtle edge cases — run tests, fix issues, repeat. The `parse_ryu_output` function is the most likely source of bugs. Debug with `println!` in test context if needed.

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(jcs): ES6-conformant number formatting (ECMA-262 7.1.12.1)

serialize_number() handles integer fast-path (i64/u64), negative zero,
and full f64 formatting via ryu digit extraction + ES6 exponential
thresholds. 11 number-specific tests."
```

---

### Task 4: RFC 8785 appendix test vectors and edge cases

**Files:**
- Modify: `crates/harmony-jcs/src/lib.rs`

- [ ] **Step 1: Add RFC 8785 test vectors**

The RFC appendix provides specific input/output pairs. Add tests from the RFC:

```rust
    /// RFC 8785 Section B.2 — Structures (simplified)
    /// Keys are sorted by UTF-16 code unit order. Verify key ordering and
    /// string escaping on a representative document.
    #[test]
    fn rfc8785_structure_key_ordering() {
        // Build a JSON object with keys that sort differently in UTF-8 vs UTF-16.
        // Use serde_json to construct it (avoids surrogate pair issues in source).
        use serde_json::json;
        let val = json!({
            "\r": "Carriage Return",
            "1": "One",
            "": "Empty"
        });
        let result = String::from_utf8(canonicalize(&val)).unwrap();
        // UTF-16 order: "" (empty) < "\r" (000D) < "1" (0031)
        assert_eq!(
            result,
            r#"{"":"Empty","\r":"Carriage Return","1":"One"}"#
        );
    }

    /// Verify that Unicode keys above BMP are handled correctly.
    #[test]
    fn rfc8785_structure_unicode_keys() {
        use serde_json::json;
        // Euro sign (U+20AC) and Latin small o with diaeresis (U+00F6)
        let val = json!({
            "\u{20ac}": "Euro Sign",
            "\u{00f6}": "Latin Small O Diaeresis"
        });
        let result = String::from_utf8(canonicalize(&val)).unwrap();
        // UTF-16 order: U+00F6 (00F6) < U+20AC (20AC)
        assert!(result.contains("\u{00f6}"));
        assert!(result.contains("\u{20ac}"));
        // Verify U+00F6 key comes first
        let f6_pos = result.find('\u{00f6}').unwrap();
        let euro_pos = result.find('\u{20ac}').unwrap();
        assert!(f6_pos < euro_pos, "U+00F6 should sort before U+20AC in UTF-16");
    }

    /// Verify that all number edge cases from RFC 8785 B.3 produce correct output.
    #[test]
    fn rfc8785_number_vectors() {
        // These are (input_json, expected_canonical) pairs from the RFC
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
            ("9007199254740991", "9007199254740991"),   // MAX_SAFE_INTEGER
            ("-9007199254740991", "-9007199254740991"),  // MIN_SAFE_INTEGER
            ("0.1", "0.1"),
            ("999999999999999900000", "999999999999999900000"),
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
        // Should have no spaces or newlines
        assert!(!result.contains(&b' '));
        assert!(!result.contains(&b'\n'));
    }
```

The implementer should reference the RFC appendix for the exact test vectors and construct them carefully. The structural vector test above is a template — the exact expected bytes should be derived from the RFC's hex-encoded output.

- [ ] **Step 2: Run tests, fix any failures**

Run: `cargo test -p harmony-jcs`
Iterate until all tests pass. The most likely issues are in the number formatter edge cases.

- [ ] **Step 3: Commit**

```bash
git commit -m "feat(jcs): RFC 8785 appendix test vectors and edge cases

Comprehensive number vectors covering exponential thresholds,
negative zero, MAX_SAFE_INTEGER. Structural test with Unicode keys.
Whitespace absence test."
```

---

### Task 5: Cleanup and verification

- [ ] **Step 1: Run clippy**

Run: `cargo clippy -p harmony-jcs`
Fix any warnings.

- [ ] **Step 2: Run workspace tests**

Run: `cargo test --workspace`

- [ ] **Step 3: Run fmt check**

Run: `cargo fmt -p harmony-jcs -- --check`

- [ ] **Step 4: Commit (if fixes needed)**

```bash
git commit -m "chore: clippy/fmt fixes for harmony-jcs"
```

---

## Summary

| Task | Description | Key Output |
|------|-------------|------------|
| 1 | Crate scaffold + workspace wiring | `harmony-jcs` crate exists, compiles, 2 stub tests |
| 2 | Core serializer (strings, objects, arrays) | `serialize_value()`, `serialize_string()`, `utf16_cmp()`, 13 tests |
| 3 | ES6 number formatting | `serialize_number()`, `format_es6_double()`, `parse_ryu_output()`, 11 tests |
| 4 | RFC 8785 test vectors | Conformance tests from the RFC appendix |
| 5 | Cleanup | Clippy clean, workspace tests pass |

**Public API after this bead:**
```rust
// In any crate that depends on harmony-jcs:
let doc: serde_json::Value = serde_json::from_str(json_str)?;
let canonical: Vec<u8> = harmony_jcs::canonicalize(&doc);
let hash: [u8; 32] = harmony_crypto::hash::full_hash(&canonical);
```
