//! Identity backup / restore — BIP39-24 mnemonic and encrypted-file
//! encodings of the master `RecoveryArtifact` seed.
//!
//! See `docs/superpowers/specs/2026-04-26-identity-backup-restore-design.md`
//! for the design and `docs/superpowers/plans/2026-04-26-identity-backup-restore.md`
//! for the build sequence.

pub mod error;
pub mod wire;
pub mod mnemonic;
pub mod encrypted_file;

// pub use error::RecoveryError;  // re-enabled in Task 3 once the enum exists
