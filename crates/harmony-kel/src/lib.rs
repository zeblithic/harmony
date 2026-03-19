#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;

pub mod commitment;
pub mod error;
pub mod event;
pub mod log;

pub use commitment::{compute_commitment, verify_commitment};
pub use error::KelError;
pub use event::{
    serialize_event_payload, serialize_inception_payload, serialize_interaction_payload,
    serialize_rotation_payload, InceptionEvent, InteractionEvent, KeyEvent, RotationEvent,
};
pub use log::KeyEventLog;
