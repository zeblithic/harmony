use crate::blob::BlobStore;
use crate::bundle::{self, BundleBuilder, MAX_BUNDLE_ENTRIES};
use crate::chunker::{chunk_all, ChunkerConfig};
use crate::cid::{CidType, ContentId};
use crate::error::ContentError;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blob::MemoryBlobStore;
    use crate::chunker::ChunkerConfig;

    /// Small chunker config for fast tests (min=64, avg=128, max=256 bytes).
    fn test_config() -> ChunkerConfig {
        ChunkerConfig {
            min_chunk: 64,
            avg_chunk: 128,
            max_chunk: 256,
        }
    }

    #[test]
    fn module_compiles() {
        let _store = MemoryBlobStore::new();
        let _config = test_config();
    }
}
