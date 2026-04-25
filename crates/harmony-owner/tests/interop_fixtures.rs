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
//
// v1.0.0 wire format: identity_hash hashes only signing material (ed25519_verify
// + optional ml_dsa_verify), so encryption-key rotation does not change identity
// (matches Matrix/Signal). LivenessCert carries owner_id for cross-owner
// domain separation. EnrollmentCert encoding itself is unchanged structurally,
// but `owner_id` and `device_id` reflect the new signing-only identity hash.
//
// The encoding is RFC 8949 §4.2 deterministic CBOR: map entries are sorted by
// length-then-bytewise-lex on the canonically-encoded key bytes. Implemented
// in `harmony_owner::cbor::to_canonical` by serializing through
// `ciborium::value::Value`, recursively sorting map entries, then re-encoding.
// This guarantees byte-for-byte interop with any RFC 8949 §4.2-compliant CBOR
// library; second implementations of this protocol are not coupled to ciborium.
const EXPECTED_ENROLLMENT_CERT_V1_HEX: &str = "a866697373756572a1664d6173746572a16d6d61737465725f7075626b6579a269636c6173736963616ca26a7832353531395f707562582001010101010101010101010101010101010101010101010101010101010101016e656432353531395f7665726966795820197f6b23e16c8532c6abc838facd5ea789be0c76b2920334039bfa8b3d368d616c706f73745f7175616e74756df66776657273696f6e01686f776e65725f69649012186b18f7186918f818cb18d418ee18f218890418ee18b3189f1898187b696465766963655f696490188f182718a418cf185d18ca1868183f18dd0d0b189d18f118d1181b183e696973737565645f61741a6553f100697369676e617475726558409843c89f940c6d89a86cfaed20d7d4d9fd596276431ac12e4d08d08d95205cd1d01b5785af5c69835aee0b1293cc069b136008af9b3dd6362cdf15a9b79df00d6a657870697265735f6174f66e6465766963655f7075626b657973a269636c6173736963616ca26a7832353531395f707562582002020202020202020202020202020202020202020202020202020202020202026e656432353531395f7665726966795820a7f6dfaf8f38b89ba8ce649b594f91e4d01fdc57f9c9493df43b5e50a99873676c706f73745f7175616e74756df6";

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
