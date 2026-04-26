//! Wire-format pin for the encrypted recovery file. Decoding the committed
//! fixture must yield the expected seed and metadata; any accidental
//! serialization change (CBOR canonicalization drift, KDF param tweak,
//! header layout shift) breaks this test loudly.
//!
//! Regenerate the fixture for an intentional format change:
//!
//!     HARMONY_REGENERATE_RECOVERY_WIRE_FIXTURE=1 \
//!         cargo test --features test-fixtures \
//!         --test recovery_wire_format_fixture
//!
//! Then commit the new bytes alongside the version bump.
//!
//! Requires the `test-fixtures` Cargo feature for the deterministic
//! `encrypt_with_params_for_test` helper.
//!
//! ## Encoding decisions documented for this format
//!
//! - **Option<None> as CBOR null**: when `mint_at` or `comment` is `None`,
//!   the serde default encodes the field as CBOR null (0xf6) inside a
//!   4-entry map, NOT omitted from the map. This produces a schema-stable
//!   wire shape. Changing to `#[serde(skip_serializing_if = "Option::is_none")]`
//!   would shrink files with absent metadata but would require a
//!   `format_version` bump.
//!
//!   NOTE: this fixture pins the FULL-METADATA case (both fields `Some(...)`).
//!   The Option<None> encoding is documented above but not byte-pinned by
//!   this specific fixture. A future commit could add a second fixture
//!   (e.g., `recovery_v1_no_metadata.bin`) to pin the None case explicitly.

use harmony_owner::lifecycle::mint::RecoveryArtifact;
use harmony_owner::recovery::{
    encrypt_with_params_for_test, FORMAT_STRING, NONCE_LEN, RecoveryFileBody, SALT_LEN,
};
use secrecy::SecretString;
use std::path::PathBuf;

const FIXTURE_REL_PATH: &str = "tests/fixtures/recovery_v1.bin";
const FIXTURE_PASSPHRASE: &str = "harmony-recovery-fixture-v1";
const FIXTURE_SEED: [u8; 32] = [
    0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef,
    0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54, 0x32, 0x10,
    0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
    0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff,
];
const FIXTURE_SALT: [u8; SALT_LEN] = [0x5A; SALT_LEN];
const FIXTURE_NONCE: [u8; NONCE_LEN] = [0xA5; NONCE_LEN];
const FIXTURE_MINT_AT: u64 = 1_700_000_000;
const FIXTURE_COMMENT: &str = "ZEB-175 wire format pin v1";

fn fixture_path() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push(FIXTURE_REL_PATH);
    p
}

fn deterministic_bytes() -> Vec<u8> {
    let body = RecoveryFileBody {
        format: FORMAT_STRING.into(),
        seed: FIXTURE_SEED,
        mint_at: Some(FIXTURE_MINT_AT),
        comment: Some(FIXTURE_COMMENT.into()),
    };
    encrypt_with_params_for_test(
        &SecretString::from(FIXTURE_PASSPHRASE.to_string()),
        &body,
        &FIXTURE_SALT,
        &FIXTURE_NONCE,
    )
}

#[test]
fn wire_format_v1_pinned() {
    let path = fixture_path();
    let regen = std::env::var("HARMONY_REGENERATE_RECOVERY_WIRE_FIXTURE").is_ok();

    if regen {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(&path, deterministic_bytes()).unwrap();
        // Explicit regeneration run — operator opted into rewrite.
        return;
    }
    assert!(
        path.exists(),
        "wire-format fixture {FIXTURE_REL_PATH} is missing. The committed \
         binary fixture is the load-bearing pin for the v1 wire format. \
         If this is intentional (you're regenerating the fixture for a \
         format change), re-run with HARMONY_REGENERATE_RECOVERY_WIRE_FIXTURE=1 \
         and commit the new bytes alongside the format_version bump."
    );

    let on_disk = std::fs::read(&path).unwrap();
    let expected = deterministic_bytes();
    assert_eq!(
        on_disk, expected,
        "fixture {FIXTURE_REL_PATH} no longer matches the deterministic encode. \
         If this was an intentional format change, regenerate via \
         HARMONY_REGENERATE_RECOVERY_WIRE_FIXTURE=1 and commit the new bytes."
    );

    // Round-trip decode to confirm the format-string and metadata path.
    let restored = RecoveryArtifact::from_encrypted_file(
        &on_disk,
        &SecretString::from(FIXTURE_PASSPHRASE.to_string()),
    )
    .unwrap();
    assert_eq!(restored.artifact.as_bytes(), &FIXTURE_SEED);
    assert_eq!(restored.metadata.mint_at, Some(FIXTURE_MINT_AT));
    assert_eq!(restored.metadata.comment.as_deref(), Some(FIXTURE_COMMENT));
}
