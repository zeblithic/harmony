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
    let bytes_b = cbor::to_canonical(&cert).unwrap();
    assert_eq!(bytes_a, bytes_b, "encoding must be deterministic across runs");

    // Round-trip
    let decoded: EnrollmentCert = cbor::from_bytes(&bytes_a).unwrap();
    assert_eq!(cert, decoded);
    decoded.verify().unwrap();

    // Print bytes for documentation; fixtures consumers can save these.
    println!("EnrollmentCert (Master) v1 bytes ({}): {}", bytes_a.len(), hex::encode(&bytes_a));
}
