//! Structured telemetry event types for the Harmony mesh network.
//!
//! Nodes publish `TelemetryEvent` messages to Zenoh pub/sub topics
//! at `harmony/telemetry/{node_addr}/{intent}`. Events carry an intent
//! tag, an opaque JSON payload, and optional metadata (confidence, source).

pub mod types;
pub mod wire;

pub use types::TelemetryEvent;
pub use wire::{decode_event, encode_event, TelemetryError};
