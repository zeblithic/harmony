// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Sans-I/O runtime orchestration for Harmony network nodes.
//!
//! This crate provides [`NodeRuntime`], the core state machine that wires
//! Tier 1 (Reticulum routing), Tier 2 (content storage), and Tier 3
//! (compute scheduling) into a unified event/action pipeline.
//!
//! Consumers feed events via [`NodeRuntime::push_event`] and process
//! returned actions from [`NodeRuntime::tick`] through platform-specific
//! I/O (UDP sockets, Zenoh sessions, disk, etc.).

pub mod inference_types;
pub mod page_index;
pub mod runtime;

// Re-export primary types for ergonomic imports
pub use runtime::{NodeConfig, NodeRuntime, RuntimeAction, RuntimeEvent};
pub use runtime::{AdaptiveCompute, TierSchedule};
