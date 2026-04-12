//! Content-addressed key-value database with named tables, atomic commits,
//! history diffing, and portable index rebuild from CAS.

mod commit;
mod db;
mod error;
mod persist;
mod prolly;
mod table;
mod types;

pub use db::HarmonyDb;
pub use error::DbError;
pub use types::{Diff, Entry, EntryMeta, TableDiff};
