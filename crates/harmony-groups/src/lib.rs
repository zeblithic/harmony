#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;

pub mod dag;
pub mod error;
pub mod op;
pub mod resolver;
pub mod sync;
pub mod types;

pub use error::ResolveError;
pub use resolver::resolve;
pub use sync::ops_to_send;
pub use types::{
    GroupAction, GroupId, GroupMode, GroupOp, GroupOpPayload, GroupState, MemberAddr, MemberEntry,
    OpId, Role,
};
