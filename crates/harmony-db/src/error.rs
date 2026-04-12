/// Errors returned by harmony-db operations.
#[derive(Debug, thiserror::Error)]
pub enum DbError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("serialization error: {0}")]
    Serialize(String),

    #[error("corrupt index: {0}")]
    CorruptIndex(String),

    #[error("commit not found: {cid}")]
    CommitNotFound { cid: String },

    #[error("table not found: {name}")]
    TableNotFound { name: String },

    #[error("entry not found: key in table {table}")]
    EntryNotFound { table: String },

    #[error("value blob missing: {cid}")]
    BlobMissing { cid: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn io_error_display() {
        let err: DbError = std::io::Error::new(std::io::ErrorKind::NotFound, "gone").into();
        assert!(err.to_string().contains("I/O error"));
    }

    #[test]
    fn serialize_error_display() {
        let err = DbError::Serialize("bad json".into());
        assert_eq!(err.to_string(), "serialization error: bad json");
    }

    #[test]
    fn commit_not_found_display() {
        let err = DbError::CommitNotFound { cid: "abc123".into() };
        assert_eq!(err.to_string(), "commit not found: abc123");
    }

    #[test]
    fn table_not_found_display() {
        let err = DbError::TableNotFound { name: "inbox".into() };
        assert_eq!(err.to_string(), "table not found: inbox");
    }

    #[test]
    fn entry_not_found_display() {
        let err = DbError::EntryNotFound { table: "inbox".into() };
        assert_eq!(err.to_string(), "entry not found: key in table inbox");
    }

    #[test]
    fn blob_missing_display() {
        let err = DbError::BlobMissing { cid: "deadbeef".into() };
        assert_eq!(err.to_string(), "value blob missing: deadbeef");
    }

    #[test]
    fn corrupt_index_display() {
        let err = DbError::CorruptIndex("truncated".into());
        assert_eq!(err.to_string(), "corrupt index: truncated");
    }
}
