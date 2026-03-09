// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Error types for Kitri workflows.

use alloc::string::String;

/// Errors surfaced to Kitri workflow authors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KitriError {
    /// Placeholder — real variants added in Task 2.
    WorkflowFailed { reason: String },
}

/// Convenience result alias.
pub type KitriResult<T> = core::result::Result<T, KitriError>;
