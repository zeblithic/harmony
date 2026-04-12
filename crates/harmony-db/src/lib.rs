//! Content-addressed key-value database with named tables, atomic commits,
//! history diffing, and portable index rebuild from CAS.

mod error;
mod persist;
mod table;
mod types;

pub use error::DbError;
pub use types::{Entry, EntryMeta};
