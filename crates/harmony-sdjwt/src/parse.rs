use crate::error::SdJwtError;

/// Extract the raw base64url header and payload strings from a JWS compact serialization.
///
/// Returns `(signing_input, signature_b64)` where `signing_input` is the raw
/// `header.payload` string suitable for signature verification, and `signature_b64`
/// is the base64url-encoded signature.
///
/// This function does NOT require `serde_json` and works in `no_std` environments.
pub fn signing_input(compact: &str) -> Result<(&str, &str), SdJwtError> {
    if compact.is_empty() {
        return Err(SdJwtError::EmptyInput);
    }

    // The input may be an SD-JWT with disclosures appended after `~`.
    // The JWS is always the first segment before any `~`.
    let jws = compact.split('~').next().unwrap_or(compact);

    // A JWS compact serialization has exactly 3 dot-separated parts.
    let mut dots = 0usize;
    let mut first_dot = None;
    let mut second_dot = None;
    for (i, b) in jws.bytes().enumerate() {
        if b == b'.' {
            dots += 1;
            match dots {
                1 => first_dot = Some(i),
                2 => second_dot = Some(i),
                _ => return Err(SdJwtError::MalformedCompact),
            }
        }
    }

    let first_dot = first_dot.ok_or(SdJwtError::MalformedCompact)?;
    let second_dot = second_dot.ok_or(SdJwtError::MalformedCompact)?;

    if dots != 2 {
        return Err(SdJwtError::MalformedCompact);
    }

    let signing_input_str = &jws[..second_dot];
    let signature_b64 = &jws[second_dot + 1..];

    // Validate that header and payload parts are non-empty.
    if first_dot == 0 || second_dot == first_dot + 1 {
        return Err(SdJwtError::MalformedCompact);
    }

    // Signature may be empty (unsecured JWS), but we require it for SD-JWT.
    if signature_b64.is_empty() {
        return Err(SdJwtError::MalformedCompact);
    }

    Ok((signing_input_str, signature_b64))
}

/// Parse an SD-JWT compact serialization into a fully decoded [`SdJwt`].
///
/// The input format is: `header.payload.signature~disclosure1~disclosure2~...~`
///
/// This function requires the `std` feature because it uses `serde_json` for
/// JSON deserialization.
#[cfg(feature = "std")]
pub fn parse(compact: &str) -> Result<crate::types::SdJwt, SdJwtError> {
    let _ = compact;
    todo!()
}
