//! Address resolution for Harmony mail.
//!
//! Parses the local part of an email address (before the `@`) into one of
//! three variants and formats outbound sender addresses.

use crate::message::ADDRESS_HASH_LEN;

// ── LocalPart ────────────────────────────────────────────────────────

/// Parsed local part of an email address (the part before @).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LocalPart {
    /// 32 hex characters decoded to a 16-byte Harmony address hash.
    Hex([u8; ADDRESS_HASH_LEN]),
    /// name_namespace format (e.g., "jake_z" -> name="jake", namespace="z").
    Named { name: String, namespace: String },
    /// Anything else — resolved via operator-defined alias table.
    Alias(String),
}

// ── Parsing ──────────────────────────────────────────────────────────

/// Parse the local part of an email address into a [`LocalPart`].
///
/// Resolution order:
/// 1. If exactly 32 hex characters (case-insensitive) -> `Hex` (decode to 16 bytes).
/// 2. If contains an underscore, split at the **last** underscore: name (left)
///    and namespace (right). Both parts must be non-empty, and namespace must
///    not contain underscores. -> `Named`.
/// 3. Otherwise -> `Alias`.
///
/// All parsing normalizes to lowercase.
pub fn parse_local_part(local: &str) -> LocalPart {
    let lower = local.to_lowercase();

    // 1. Exactly 32 hex characters -> Hex
    if lower.len() == 32 && lower.bytes().all(|b| b.is_ascii_hexdigit()) {
        let mut bytes = [0u8; ADDRESS_HASH_LEN];
        for (i, chunk) in lower.as_bytes().chunks(2).enumerate() {
            // SAFETY: we already validated all bytes are ASCII hex digits,
            // so from_str_radix will always succeed.
            let s = std::str::from_utf8(chunk).unwrap();
            bytes[i] = u8::from_str_radix(s, 16).unwrap();
        }
        return LocalPart::Hex(bytes);
    }

    // 2. Contains underscore -> split at last underscore
    if let Some(last_us) = lower.rfind('_') {
        let name = &lower[..last_us];
        let namespace = &lower[last_us + 1..];

        // Both parts must be non-empty, namespace must not contain underscores.
        if !name.is_empty() && !namespace.is_empty() && !namespace.contains('_') {
            return LocalPart::Named {
                name: name.to_string(),
                namespace: namespace.to_string(),
            };
        }
    }

    // 3. Fallback -> Alias
    LocalPart::Alias(lower)
}

// ── Formatting ───────────────────────────────────────────────────────

/// Format the outbound sender (From) address for an SMTP envelope.
///
/// - If `registered_name` is `Some((name, namespace))`, produces
///   `"name_namespace@domain"`.
/// - If `None`, produces `"user-{first 4 bytes hex}@domain"` — the `user-`
///   prefix avoids SpamAssassin `FROM_LOCAL_HEX` penalties.
pub fn format_outbound_sender(
    registered_name: Option<(&str, &str)>,
    address_hash: &[u8; ADDRESS_HASH_LEN],
    domain: &str,
) -> String {
    match registered_name {
        Some((name, namespace)) => {
            format!("{}_{}@{}", name, namespace, domain)
        }
        None => {
            let prefix: String = address_hash[..4]
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect();
            format!("user-{}@{}", prefix, domain)
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_hex_address() {
        let local = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6";
        let result = parse_local_part(local);
        let expected_bytes: [u8; ADDRESS_HASH_LEN] = [
            0xa1, 0xb2, 0xc3, 0xd4, 0xe5, 0xf6, 0xa7, 0xb8, 0xc9, 0xd0, 0xe1, 0xf2, 0xa3, 0xb4,
            0xc5, 0xd6,
        ];
        assert_eq!(result, LocalPart::Hex(expected_bytes));
    }

    #[test]
    fn parse_named_address() {
        let result = parse_local_part("jake_z");
        assert_eq!(
            result,
            LocalPart::Named {
                name: "jake".to_string(),
                namespace: "z".to_string(),
            }
        );
    }

    #[test]
    fn parse_named_address_longer_namespace() {
        let result = parse_local_part("victoria_vrk");
        assert_eq!(
            result,
            LocalPart::Named {
                name: "victoria".to_string(),
                namespace: "vrk".to_string(),
            }
        );
    }

    #[test]
    fn parse_vanity_alias() {
        let result = parse_local_part("support");
        assert_eq!(result, LocalPart::Alias("support".to_string()));
    }

    #[test]
    fn parse_vanity_alias_with_dots() {
        let result = parse_local_part("first.last");
        assert_eq!(result, LocalPart::Alias("first.last".to_string()));
    }

    #[test]
    fn hex_address_case_insensitive() {
        let upper = parse_local_part("A1B2C3D4E5F6A7B8C9D0E1F2A3B4C5D6");
        let lower = parse_local_part("a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6");
        assert_eq!(upper, lower);
        assert!(matches!(upper, LocalPart::Hex(_)));
    }

    #[test]
    fn hex_address_wrong_length_is_alias() {
        // 30 hex chars — not a valid 16-byte address
        let result = parse_local_part("a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5");
        assert!(matches!(result, LocalPart::Alias(_)));
    }

    #[test]
    fn format_outbound_from_named() {
        let hash = [0x00u8; ADDRESS_HASH_LEN];
        let result = format_outbound_sender(Some(("jake", "z")), &hash, "q8.fyi");
        assert_eq!(result, "jake_z@q8.fyi");
    }

    #[test]
    fn format_outbound_from_hex_uses_prefix() {
        let mut hash = [0x00u8; ADDRESS_HASH_LEN];
        hash[0] = 0xa1;
        hash[1] = 0xb2;
        hash[2] = 0xc3;
        hash[3] = 0xd4;
        let result = format_outbound_sender(None, &hash, "q8.fyi");
        assert_eq!(result, "user-a1b2c3d4@q8.fyi");
    }

    #[test]
    fn parse_multi_underscore_splits_at_last() {
        let result = parse_local_part("my_name_z");
        assert_eq!(
            result,
            LocalPart::Named {
                name: "my_name".to_string(),
                namespace: "z".to_string(),
            }
        );
    }
}
