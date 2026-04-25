//! Cross-implementation interop fixtures. Each test produces a deterministic
//! cert from fixed seeds and asserts the byte-exact CBOR encoding. If a
//! second implementation produces these same bytes from the same inputs,
//! the wire format is unambiguous.
//!
//! On encoding changes that bump BindingFormatVersion, regenerate the
//! fixtures and bump the version byte in the test data.

use harmony_owner::{
    cbor,
    certs::EnrollmentCert,
    pubkey_bundle::{ClassicalKeys, PubKeyBundle},
};
use ed25519_dalek::SigningKey;

// Golden vector: byte-exact CBOR encoding of the deterministic EnrollmentCert v1
// fixture. If this changes unintentionally, the wire format has drifted; bump
// the format version and intentionally regenerate the fixture before merging.
const EXPECTED_ENROLLMENT_CERT_V1_HEX: &str = "a86776657273696f6e01686f776e65725f6964901864187d18e818d5185318b518fc18a6184e183a18eb188c187618691856184a696465766963655f696490186a187418e9184d1877185f181f18be18c9161846185e18491418cf18586e6465766963655f7075626b657973a269636c6173736963616ca26e656432353531395f7665726966795820a7f6dfaf8f38b89ba8ce649b594f91e4d01fdc57f9c9493df43b5e50a99873676a7832353531395f707562582002020202020202020202020202020202020202020202020202020202020202026c706f73745f7175616e74756df6696973737565645f61741a6553f1006a657870697265735f6174f666697373756572a1664d6173746572a16d6d61737465725f7075626b6579a269636c6173736963616ca26e656432353531395f7665726966795820197f6b23e16c8532c6abc838facd5ea789be0c76b2920334039bfa8b3d368d616a7832353531395f707562582001010101010101010101010101010101010101010101010101010101010101016c706f73745f7175616e74756df6697369676e61747572655840266fcbb35a354d1dd7693c8f8415a10072a1e1c5765463f6e4f69f9f74a2aaf16b4536b0634b9b97701c25027205dc293bec0841335737d5ced37d18c7a8ae0e";

fn deterministic_master_sk() -> SigningKey {
    let seed = [42u8; 32];
    SigningKey::from_bytes(&seed)
}

fn deterministic_device_sk() -> SigningKey {
    let seed = [99u8; 32];
    SigningKey::from_bytes(&seed)
}

#[test]
fn master_enrollment_cert_v1_is_deterministic() {
    let master_sk = deterministic_master_sk();
    let master_bundle = PubKeyBundle {
        classical: ClassicalKeys {
            ed25519_verify: master_sk.verifying_key().to_bytes(),
            x25519_pub: [1u8; 32], // fixed for fixture
        },
        post_quantum: None,
    };
    let device_sk = deterministic_device_sk();
    let device_bundle = PubKeyBundle {
        classical: ClassicalKeys {
            ed25519_verify: device_sk.verifying_key().to_bytes(),
            x25519_pub: [2u8; 32],
        },
        post_quantum: None,
    };
    let device_id = device_bundle.identity_hash();

    let cert = EnrollmentCert::sign_master(
        &master_sk,
        master_bundle,
        device_id,
        device_bundle,
        1_700_000_000,
        None,
    ).unwrap();

    let bytes_a = cbor::to_canonical(&cert).unwrap();
    assert_eq!(
        hex::encode(&bytes_a),
        EXPECTED_ENROLLMENT_CERT_V1_HEX,
        "wire format changed; bump BindingFormatVersion and regenerate fixture intentionally"
    );
    let bytes_b = cbor::to_canonical(&cert).unwrap();
    assert_eq!(bytes_a, bytes_b, "encoding must be deterministic across runs");

    // Round-trip
    let decoded: EnrollmentCert = cbor::from_bytes(&bytes_a).unwrap();
    assert_eq!(cert, decoded);
    decoded.verify().unwrap();

    // Print bytes for documentation; fixtures consumers can save these.
    println!("EnrollmentCert (Master) v1 bytes ({}): {}", bytes_a.len(), hex::encode(&bytes_a));
}
