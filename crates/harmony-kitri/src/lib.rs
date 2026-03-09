// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Kitri — Harmony's native programming model for durable distributed computation.
//!
//! Layer 0: sans-I/O core types, traits, and manifest parsing.
//! This crate defines the *protocol* of Kitri — no runtime, no I/O.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod checkpoint;
pub mod compensation;
pub mod dag;
pub mod engine;
pub mod error;
pub mod event;
pub mod io;
pub mod manifest;
pub mod program;
pub mod retry;
pub mod staging;
pub mod trust;

pub use checkpoint::KitriCheckpoint;
pub use compensation::{CompensationLog, Compensator};
pub use dag::{DagStep, KitriDag};
pub use engine::{KitriAction, KitriEngine, KitriEngineEvent, KitriWorkflowStatus};
pub use error::{KitriError, KitriResult};
pub use event::KitriEventLog;
pub use io::{KitriIoOp, KitriIoResult};
pub use manifest::{DeployConfig, KitriManifest, RuntimeConfig, TrustConfig};
pub use program::{kitri_workflow_id, KitriProgram};
pub use retry::{BackoffStrategy, FailureKind, RetryPolicy};
pub use staging::{StagedWrite, StagingBuffer};
pub use trust::{CapabilityDecl, CapabilitySet, TrustTier};
