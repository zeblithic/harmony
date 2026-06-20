//! Wire-format pin: canonical CBOR bytes of one `PkarrRoutingRecord`
//! produced with deterministic-test inputs.
//!
//! If this test breaks, the canonical CBOR encoding of `PkarrRoutingRecord`
//! has drifted — every published pkarr record under the v1 wire format
//! becomes undecodable without a v2 migration. Regenerate the expected hex
//! only with full understanding.

use ed25519_dalek::SigningKey;
use harmony_pkarr::PkarrRoutingRecord;

#[test]
fn canonical_cbor_bytes_pinned() {
    // Deterministic inputs.
    let sk_bytes = [42u8; 32];
    let sk = SigningKey::from_bytes(&sk_bytes);
    let mut identity_pub = [0u8; 64];
    identity_pub[32..].copy_from_slice(&sk.verifying_key().to_bytes());
    let rec = PkarrRoutingRecord::sign_new(
        b"deterministic-routing-blob".to_vec(),
        identity_pub,
        1_700_000_000_000u64,                  // fixed announced_at
        1_700_000_000_000u64 + 604_800_000u64, // fixed valid_until (+7d)
        &sk,
    )
    .expect("sign");

    let cbor = rec.to_canonical_cbor().expect("encode");
    let cbor_hex = hex::encode(&cbor);

    // Pin: regenerate only with full understanding — a change breaks every
    // v1 pkarr record published under this wire format.
    let expected = "a5627264581a64657465726d696e69737469632d726f7574696e672d626c6f6262697058400000000000000000000000000000000000000000000000000000000000000000197f6b23e16c8532c6abc838facd5ea789be0c76b2920334039bfa8b3d368d616261741b0000018bcfe568006276751b0000018bf3f1ec00627367584013ad53aa88471f0195a735bef2600e034bc9c84d709f7a159a3e3119683684db137184ef30c84f548a1a9194a439d1561020e56299a4212415c14781469ec506";
    assert_eq!(
        cbor_hex, expected,
        "PkarrRoutingRecord canonical CBOR must not drift"
    );
}
