//! # harmony-owner
//!
//! Two-tier owner→device identity binding for the Harmony network.
//!
//! Spec: `docs/superpowers/specs/2026-04-25-harmony-owner-device-binding-design.md`
//!
//! ## Concepts
//!
//! - **Owner identity `M`**: master keypair defining a single human user.
//!   Lives only in the recovery artifact + transient RAM during enrollment
//!   ceremonies.
//! - **Device identity `D`**: per-device keypair. Persists locally.
//! - **Enrollment Cert**: authorizes `D` under `M`. Signed by `M` or by a
//!   K=2 quorum of already-enrolled siblings.
//! - **Vouching Cert**: per-(signer, target) attestation. LWW CRDT.
//! - **Liveness Cert**: periodic timestamped heartbeat.
//! - **Revocation Cert**: monotonic Remove-Wins. Once present, never reversed.
//! - **Reclamation Cert**: time-bounded claim of continuity from a prior
//!   identity, after total loss.
//!
//! ## Trust evaluation
//!
//! [`trust::evaluate_trust`] returns Full / Provisional / Refused for a
//! target device given current state and time. v1 threshold: N=1 active
//! sibling vouch.
//!
//! ## Out of scope here
//!
//! Network propagation (Zenoh gossip + queryable) and harmony-client
//! integration are separate plans/crates.

pub mod cbor;
pub mod certs;
pub mod crdt;
pub mod error;
pub mod lifecycle;
pub mod pubkey_bundle;
pub mod signing;
pub mod state;
pub mod trust;

pub use error::OwnerError;
