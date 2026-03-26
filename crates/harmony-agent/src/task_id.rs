//! Deterministic task ID generation via BLAKE3.

/// Generate a task ID from submitter address, params, and nonce.
///
/// `id = hex(BLAKE3(submitter_addr || json(params) || nonce_le))`
///
/// The nonce ensures unique IDs even for identical params.
pub fn generate_task_id(
    submitter_addr: &[u8; 16],
    params: &serde_json::Value,
    nonce: u64,
) -> String {
    // serde_json::to_vec on a Value always succeeds.
    let params_bytes = serde_json::to_vec(params).unwrap_or_default();
    let mut input = Vec::with_capacity(16 + params_bytes.len() + 8);
    input.extend_from_slice(submitter_addr);
    input.extend_from_slice(&params_bytes);
    input.extend_from_slice(&nonce.to_le_bytes());
    hex::encode(harmony_crypto::hash::blake3_hash(&input))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_same_inputs() {
        let addr = [0xAA; 16];
        let params = serde_json::json!({"prompt": "Hello"});
        assert_eq!(generate_task_id(&addr, &params, 42), generate_task_id(&addr, &params, 42));
    }

    #[test]
    fn different_nonce_different_id() {
        let addr = [0xAA; 16];
        let params = serde_json::json!({"prompt": "Hello"});
        assert_ne!(generate_task_id(&addr, &params, 1), generate_task_id(&addr, &params, 2));
    }

    #[test]
    fn different_params_different_id() {
        let addr = [0xAA; 16];
        let p1 = serde_json::json!({"prompt": "Hello"});
        let p2 = serde_json::json!({"prompt": "World"});
        assert_ne!(generate_task_id(&addr, &p1, 0), generate_task_id(&addr, &p2, 0));
    }

    #[test]
    fn different_submitter_different_id() {
        let params = serde_json::json!({"prompt": "Hello"});
        assert_ne!(generate_task_id(&[0xAA; 16], &params, 0), generate_task_id(&[0xBB; 16], &params, 0));
    }

    #[test]
    fn id_is_hex_encoded() {
        let id = generate_task_id(&[0; 16], &serde_json::json!({}), 0);
        assert_eq!(id.len(), 64);
        assert!(id.chars().all(|c| c.is_ascii_hexdigit()));
    }
}
