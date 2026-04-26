//! Cross-implementation interop fixtures. Each test produces a deterministic
//! cert from fixed seeds and asserts the byte-exact CBOR encoding. If a
//! second implementation produces these same bytes from the same inputs,
//! the wire format is unambiguous.
//!
//! On encoding changes that bump BindingFormatVersion, regenerate the
//! fixtures and bump the version byte in the test data.

use harmony_owner::{
    cbor,
    certs::{EnrollmentCert, EnrollmentIssuer},
    pubkey_bundle::{ClassicalKeys, PubKeyBundle},
    signing::{sign_with_tag, tags},
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
//
// All byte-typed fields (IdentityHash [u8;16], signature, issuer_data, and
// per-element entries of signers/signatures lists) are encoded as CBOR byte
// strings (major type 2), NOT as arrays of integers (major type 4). This
// matches the natural CBOR encoding for byte-typed data and is what second
// implementations using any standard CBOR library will produce. See
// `harmony_owner::cbor::{arr16, arr16_vec, bytes_vec}` for the serde codecs.
const EXPECTED_ENROLLMENT_CERT_V1_HEX: &str = "a866697373756572a1664d6173746572a16d6d61737465725f7075626b6579a269636c6173736963616ca26a7832353531395f707562582001010101010101010101010101010101010101010101010101010101010101016e656432353531395f7665726966795820197f6b23e16c8532c6abc838facd5ea789be0c76b2920334039bfa8b3d368d616c706f73745f7175616e74756df66776657273696f6e01686f776e65725f696450126bf769f8cbd4eef28904eeb39f987b696465766963655f6964508f27a4cf5dca683fdd0d0b9df1d11b3e696973737565645f61741a6553f100697369676e6174757265584061947fad5d8d3a8e2cd567b84eb30b77a95c9b77b7312bfad1152258dd7735348e6ba6744e1a2316852cdb78d2b3ac9d576f98816796d30fe8bf6b9bd205f00c6a657870697265735f6174f66e6465766963655f7075626b657973a269636c6173736963616ca26a7832353531395f707562582002020202020202020202020202020202020202020202020202020202020202026e656432353531395f7665726966795820a7f6dfaf8f38b89ba8ce649b594f91e4d01fdc57f9c9493df43b5e50a99873676c706f73745f7175616e74756df6";

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

fn deterministic_signer_a() -> SigningKey {
    SigningKey::from_bytes(&[7u8; 32])
}

fn deterministic_signer_b() -> SigningKey {
    SigningKey::from_bytes(&[8u8; 32])
}

// Quorum enrollment cert wire format. Locks down the encoding of:
//   - EnrollmentIssuer::Quorum (signers as arr16_vec, signatures as bytes_vec)
//   - signers list as a CBOR array of 16-byte byte strings
//   - signatures list as a CBOR array of 64-byte byte strings (ed25519)
//
// Signatures here are produced over `cbor(signers)` (matching the issuer_data
// derivation used internally for Quorum signing payloads). They are not
// expected to round-trip through standalone EnrollmentCert::verify (which only
// performs structural checks for Quorum and delegates signature verification
// to OwnerState that has access to signer enrollments). The fixture's purpose
// is byte-exact encoding stability, not end-to-end verification.
const EXPECTED_QUORUM_ENROLLMENT_HEX: &str = "a866697373756572a16651756f72756da2677369676e657273825033eb0926cb807d52aaecc4fcba3649ad50a739288ca7857e7b6495ff79e5d65bb66a7369676e61747572657382584001a2c7c2496157f285962f2f938c494bb06dc775e6cf2cc95cd0805efa9e773e1488db6a6d06c419d91f57afaa1a98b395d93e2657ee6ea57279de2298d550005840d785f14dcfd9bf6e23e0ec443b7d2edea9bfc9783b808a18d7947bf312ffb76503039f046bf74c33c5828b2c7cdfa565abdf9e35138cbfe689e2a7660238470c6776657273696f6e01686f776e65725f6964502a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a696465766963655f6964508f27a4cf5dca683fdd0d0b9df1d11b3e696973737565645f61741a6553f100697369676e6174757265406a657870697265735f6174f66e6465766963655f7075626b657973a269636c6173736963616ca26a7832353531395f707562582003030303030303030303030303030303030303030303030303030303030303036e656432353531395f7665726966795820a7f6dfaf8f38b89ba8ce649b594f91e4d01fdc57f9c9493df43b5e50a99873676c706f73745f7175616e74756df6";

#[test]
fn quorum_enrollment_cert_v1_is_deterministic() {
    let sk_a = deterministic_signer_a();
    let sk_b = deterministic_signer_b();
    let bundle_a = PubKeyBundle::classical_only(sk_a.verifying_key().to_bytes());
    let bundle_b = PubKeyBundle::classical_only(sk_b.verifying_key().to_bytes());
    let id_a = bundle_a.identity_hash();
    let id_b = bundle_b.identity_hash();

    let device_sk = SigningKey::from_bytes(&[99u8; 32]);
    let device_bundle = PubKeyBundle {
        classical: ClassicalKeys {
            ed25519_verify: device_sk.verifying_key().to_bytes(),
            x25519_pub: [3u8; 32],
        },
        post_quantum: None,
    };
    let device_id = device_bundle.identity_hash();

    let owner_id = [42u8; 16]; // canonical owner_id for this fixture
    let signers = vec![id_a, id_b];

    // Build deterministic signatures over a fixed payload (issuer_data = cbor(signers)).
    // For the encoding-stability fixture we don't need verify() to succeed —
    // we just need the cert's CBOR bytes to be reproducible.
    let issuer_data = cbor::to_canonical(&signers).unwrap();
    let cert = EnrollmentCert {
        version: 1,
        owner_id,
        device_id,
        device_pubkeys: device_bundle,
        issued_at: 1_700_000_000,
        expires_at: None,
        issuer: EnrollmentIssuer::Quorum {
            signers: signers.clone(),
            signatures: vec![
                sign_with_tag(&sk_a, tags::ENROLLMENT, &issuer_data),
                sign_with_tag(&sk_b, tags::ENROLLMENT, &issuer_data),
            ],
        },
        signature: Vec::new(),
    };

    let bytes_a = cbor::to_canonical(&cert).unwrap();
    let bytes_b = cbor::to_canonical(&cert).unwrap();
    assert_eq!(bytes_a, bytes_b, "encoding must be deterministic across runs");

    // Round-trip
    let decoded: EnrollmentCert = cbor::from_bytes(&bytes_a).unwrap();
    assert_eq!(cert, decoded);

    println!(
        "EnrollmentCert (Quorum) v1 bytes ({}): {}",
        bytes_a.len(),
        hex::encode(&bytes_a)
    );

    assert_eq!(
        hex::encode(&bytes_a),
        EXPECTED_QUORUM_ENROLLMENT_HEX,
        "quorum wire format changed; bump version and regenerate fixture intentionally"
    );
}
