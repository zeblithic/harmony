//! DID Document utilities: URL mapping and document parsing.
//!
//! This module provides:
//! - [`did_web_to_url`] — pure, no-I/O conversion from a `did:web` DID to its
//!   canonical HTTPS fetch URL (W3C did:web spec).

use alloc::format;
use alloc::string::String;

use crate::did::DidError;

/// Convert a `did:web` DID string into the HTTPS URL where its DID Document
/// is published, following the W3C did:web specification.
///
/// Rules:
/// - `did:web:example.com` → `https://example.com/.well-known/did.json`
/// - `did:web:example.com:path:to:resource` → `https://example.com/path/to/resource/did.json`
/// - Percent-encoded characters in the domain (e.g. `%3A` for `:`) are decoded
///   before the URL is assembled.
///
/// Returns [`DidError::UnsupportedMethod`] if the DID does not start with `did:web:`,
/// and [`DidError::MalformedDid`] if the method-specific identifier is empty.
pub fn did_web_to_url(did: &str) -> Result<String, DidError> {
    let method_specific_id = did
        .strip_prefix("did:web:")
        .ok_or_else(|| DidError::UnsupportedMethod(String::from("web")))?;

    if method_specific_id.is_empty() {
        return Err(DidError::MalformedDid(String::from(
            "did:web method-specific identifier is empty",
        )));
    }

    // Split on ':' to separate domain from optional path segments.
    // The first segment is the domain (possibly percent-encoded), and
    // any remaining segments become path components.
    let mut parts = method_specific_id.split(':');
    let raw_domain = parts.next().expect("split always yields at least one element");

    // Decode percent-encoded characters in the domain (e.g. %3A → :)
    let domain = percent_decode(raw_domain)?;

    // Collect remaining segments as path components
    let path_segments: alloc::vec::Vec<&str> = parts.collect();

    let url = if path_segments.is_empty() {
        format!("https://{}/.well-known/did.json", domain)
    } else {
        format!("https://{}/{}/did.json", domain, path_segments.join("/"))
    };

    Ok(url)
}

/// Decode percent-encoded ASCII sequences (e.g. `%3A` → `:`) in a string.
///
/// Only two-hex-digit sequences following `%` are decoded. A bare `%` not
/// followed by exactly two hex digits is returned as-is (lenient decoding).
fn percent_decode(input: &str) -> Result<String, DidError> {
    let mut output = String::with_capacity(input.len());
    let bytes = input.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let hi = bytes[i + 1];
            let lo = bytes[i + 2];
            if hi.is_ascii_hexdigit() && lo.is_ascii_hexdigit() {
                let decoded = hex_to_byte(hi, lo);
                // Only decode single-byte ASCII code points
                output.push(decoded as char);
                i += 3;
                continue;
            }
        }
        output.push(bytes[i] as char);
        i += 1;
    }

    Ok(output)
}

/// Convert two ASCII hex digit bytes to a single byte value.
fn hex_to_byte(hi: u8, lo: u8) -> u8 {
    fn digit(b: u8) -> u8 {
        match b {
            b'0'..=b'9' => b - b'0',
            b'a'..=b'f' => b - b'a' + 10,
            b'A'..=b'F' => b - b'A' + 10,
            _ => 0,
        }
    }
    (digit(hi) << 4) | digit(lo)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn did_web_root_domain() {
        let url = did_web_to_url("did:web:example.com").unwrap();
        assert_eq!(url, "https://example.com/.well-known/did.json");
    }

    #[test]
    fn did_web_path_based() {
        let url = did_web_to_url("did:web:example.com:issuers:1").unwrap();
        assert_eq!(url, "https://example.com/issuers/1/did.json");
    }

    #[test]
    fn did_web_percent_encoded_port() {
        let url = did_web_to_url("did:web:example.com%3A8443").unwrap();
        assert_eq!(url, "https://example.com:8443/.well-known/did.json");
    }

    #[test]
    fn did_web_percent_encoded_port_with_path() {
        let url = did_web_to_url("did:web:example.com%3A8443:users:alice").unwrap();
        assert_eq!(url, "https://example.com:8443/users/alice/did.json");
    }

    #[test]
    fn did_web_not_did_web_prefix() {
        assert!(did_web_to_url("did:key:z6Mk...").is_err());
    }

    #[test]
    fn did_web_empty_identifier() {
        assert!(did_web_to_url("did:web:").is_err());
    }
}
