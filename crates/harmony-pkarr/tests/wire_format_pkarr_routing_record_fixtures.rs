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
        1_700_000_000_000u64, // fixed wall-clock
        &sk,
    )
    .expect("sign");

    let cbor = rec.to_canonical_cbor().expect("encode");
    let cbor_hex = hex::encode(&cbor);

    // Pin: regenerate only with full understanding — a change breaks every
    // v1 pkarr record published under this wire format.
    let expected = "a4627264581a64657465726d696e69737469632d726f7574696e672d626c6f6262697058400000000000000000000000000000000000000000000000000000000000000000197f6b23e16c8532c6abc838facd5ea789be0c76b2920334039bfa8b3d368d616261741b0000018bcfe568006273675840e25d13d3e7227edda334a7d186770b07c9a254d41881ab66c6a91d6331d48db787b81e57c977b9e9079bf796841dda52b659d69c6f3574056ab690ba645bd50d";
    assert_eq!(
        cbor_hex, expected,
        "PkarrRoutingRecord canonical CBOR must not drift"
    );
}
