//! Content-addressed key-value database with named tables, atomic commits,
//! history diffing, and portable index rebuild from CAS.

mod db;
mod error;
mod persist;
mod prolly;
mod types;

pub use db::HarmonyDb;
pub use error::DbError;
pub use types::{Diff, Entry, EntryMeta, TableDiff};
