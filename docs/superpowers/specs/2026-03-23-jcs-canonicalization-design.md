# JCS Canonicalization (RFC 8785)

**Date:** 2026-03-23
**Status:** Draft
**Scope:** New crate `harmony-jcs`
**Bead:** harmony-vl1

## Problem

The credential system's JSON-LD export (`harmony-credential/jsonld.rs`) uses
custom cryptosuite names (`harmony-eddsa-2022`, `harmony-mldsa65-2025`)
because proofs sign the postcard binary payload, not canonicalized JSON.
This makes exported credentials incompatible with standard W3C Data
Integrity verifiers that expect JCS-based proofs (`mldsa65-jcs-2024`,
`slhdsa128-jcs-2024`).

Without JSON canonicalization, there is no deterministic JSON byte string
to sign — different serializers produce different whitespace, key ordering,
and number formatting for the same logical document.

Note: the existing `jsonld.rs` comments mention RDFC-2022 as the missing
canonicalization. JCS suites (`*-jcs-2024`) are a simpler alternative to
RDFC-2022 suites — they require only JSON key sorting, not RDF graph
processing. Both are valid W3C Data Integrity canonicalization methods.

## Solution

New crate `harmony-jcs` implementing RFC 8785 (JSON Canonicalization
Scheme). A single public function takes a parsed `serde_json::Value` and
returns deterministic canonical UTF-8 bytes. Consumers combine this with
`harmony-crypto` hash functions for Data Integrity proof generation and
verification.

## Design Decisions

### New crate (not a module in harmony-crypto or harmony-credential)

RFC 8785 is a well-defined standard with no Harmony-specific semantics.
A dedicated crate keeps the dependency graph clean — `harmony-crypto` stays
JSON-free, and `harmony-credential` doesn't couple a general-purpose
standard to credential logic. Any crate can depend on `harmony-jcs`
independently.

### std-only (no_std deferred)

The primary consumer (`harmony-credential` JSON-LD export) already requires
`std` for its `jsonld` feature. Writing a custom `no_std` JSON parser for
a standard nobody consumes in a `no_std` context is premature. The crate
structure supports adding a `no_std` feature later via feature gates.

### Single function API

```rust
pub fn canonicalize(value: &serde_json::Value) -> Vec<u8>
```

Takes a parsed JSON value, returns canonical UTF-8 bytes. The caller
handles parsing (`serde_json::from_str`) and hashing (`harmony-crypto`).
Composable, does one thing well. No hash-integrated convenience functions
— different proof suites use different hash algorithms.

The function is infallible (`Vec<u8>`, not `Result`) because
`serde_json::Value` cannot represent invalid states (no NaN, no Infinity,
no duplicate keys). If a future `no_std` path accepts raw bytes or a
different JSON type, that path would use a separate fallible function.

## RFC 8785 Rules

### Objects

Keys sorted by **UTF-16 code unit order** (not UTF-8 byte order). For
ASCII-only keys (the common case), UTF-8 and UTF-16 sort identically.
For keys containing characters above U+FFFF (non-BMP), surrogate pair
ordering differs from UTF-8 byte ordering. Each key:value pair is
serialized recursively after sorting.

Implementation: `serde_json::Map` is a `BTreeMap<String, Value>` sorted
by Rust's `Ord` for `String` (UTF-8 byte order). For correctness, we
re-sort using a UTF-16 code unit comparator:
`a.encode_utf16().cmp(b.encode_utf16())`.

In practice this is a no-op for ASCII keys, but the implementation must
handle the general case.

### Arrays

Elements serialized in order. No sorting.

### Strings

JSON-escaped with minimal escaping. The **shorthand escapes take
precedence** over the generic `\uXXXX` form for the specific code points
they cover:

- `\b` (U+0008), `\t` (U+0009), `\n` (U+000A), `\f` (U+000C), `\r` (U+000D)
- `\"` (U+0022), `\\` (U+005C)
- All other control characters U+0000–U+001F → `\uXXXX` (lowercase hex)
- All other characters passed through as-is (including non-ASCII UTF-8)

Note: U+007F (DEL) is NOT escaped — it passes through as a literal byte.
Characters above U+FFFF are emitted as literal UTF-8, never as surrogate
pair escapes. `serde_json` already handles surrogate pair decoding during
parsing, so supplementary characters arrive as proper Unicode code points.

### Numbers (CRITICAL — custom formatting required)

RFC 8785 requires ES6 `Number.prototype.toString()` semantics (ECMA-262
section 7.1.12.1). **`serde_json`'s default serializer does NOT produce
conformant output** — it uses `ryu` for shortest-round-trip representation,
which differs from ES6 in exponential notation thresholds.

Custom number formatting is required for all `f64` values. This is the
most complex part of the implementation (~40-60% of effort).

ES6 number serialization rules (ECMA-262 7.1.12.1):

1. **NaN/Infinity:** not valid JSON — `serde_json::Value` cannot represent
   these, so not a concern.
2. **Negative zero:** `-0` → `"0"` (normalize to positive zero).
3. **Integers:** values with no fractional part are serialized without a
   decimal point: `1.0` → `"1"`, `100.0` → `"100"`.
4. **Exponential thresholds:**
   - If the decimal representation would have more than 21 digits before
     the decimal point, use exponential: `1e+21` not `1000000000000000000000`.
   - If the number of leading zeros after the decimal would be 6 or more,
     use exponential: `1e-7` not `0.0000001`.
5. **Exponential format:** lowercase `e`, explicit `+` or `-` sign on
   exponent (e.g., `1.5e+20`, `1e-7`).
6. **Shortest representation:** use the fewest digits that round-trip to
   the same IEEE 754 double.

Key differences from `ryu` (serde_json's default):
- `ryu`: `1e20` → ES6: `100000000000000000000` (no exponential under 10^21)
- `ryu`: `5e-7` → ES6: `5e-7` (same in this case, but thresholds differ)
- `ryu`: may produce `1.2e10` → ES6: `12000000000`

The implementation should use `ryu` for shortest-round-trip digit
extraction, then apply ES6 formatting rules to the digits and exponent.

### Booleans and null

Literal `true`, `false`, `null`. No special handling needed.

### Whitespace

None. No spaces, no newlines between any tokens.

## Invariants from serde_json

The following RFC 8785 requirements are guaranteed by `serde_json::Value`:

- **No duplicate keys:** `serde_json::Map` is backed by `BTreeMap`, which
  deduplicates during parsing (last value wins). RFC 8785 Section 3.2.3
  requires rejecting duplicates — our function operates post-parse, so
  duplicates are already resolved.
- **No NaN/Infinity:** `serde_json` rejects these during parsing.
- **Surrogate pairs decoded:** supplementary characters arrive as proper
  Unicode code points, not surrogate pair escapes.

A future `no_std` path with a custom parser must enforce these invariants
explicitly.

## Crate Structure

```
crates/harmony-jcs/
├── Cargo.toml
└── src/
    └── lib.rs    — canonicalize() + serialize_string() + serialize_number()
```

Single file. Expect ~200-300 lines including the ES6 number formatter.

The workspace `Cargo.toml` needs `harmony-jcs` added to `[workspace.members]`
and `[workspace.dependencies]`.

**Cargo.toml dependencies:**
```toml
[dependencies]
serde_json = { workspace = true }
```

No other dependencies needed. The crate does not depend on `serde` directly
— it operates on `serde_json::Value` which is already parsed.

## Testing

RFC 8785 specifies test vectors (Section 4 / Appendix B). Key test
categories:

- `empty_object` — `{}` → `{}`
- `empty_array` — `[]` → `[]`
- `string_escaping` — shorthand escapes, `\uXXXX` for control chars, passthrough for non-ASCII
- `string_del_not_escaped` — U+007F passes through as literal
- `number_integer` — `1.0` → `1`, `100.0` → `100`
- `number_negative_zero` — `-0` → `0`
- `number_exponential_thresholds` — verify ES6 boundary at 10^21 and 10^-7
- `number_shortest_roundtrip` — verify fewest digits that round-trip
- `key_sorting_ascii` — lexicographic sort of ASCII keys
- `key_sorting_unicode` — UTF-16 code unit order for non-BMP keys
- `nested_structures` — objects within arrays within objects
- `rfc8785_appendix_vectors` — official examples from the RFC

## Integration Points

After this crate exists, the following integrations become possible
(all out of scope for this bead):

- `harmony-credential` `jsonld` feature: replace custom cryptosuite names
  with standard `mldsa65-jcs-2024` by signing `SHA-256(canonicalize(doc))`
- `harmony-sdjwt`: JCS-based SD-JWT alternative (future spec)
- W3C Data Integrity proof verification: verify proofs from external issuers

## What is NOT in Scope

- Integration with `harmony-credential` (separate bead)
- RDFC-2022 (RDF Dataset Canonicalization — different standard, much more complex)
- `no_std` implementation (deferred, feature-gated when needed)
- Hash-integrated convenience functions
- Data Integrity proof generation/verification logic
