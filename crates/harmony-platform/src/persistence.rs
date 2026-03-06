use alloc::string::String;
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::collections::HashMap;
#[cfg(not(feature = "std"))]
use hashbrown::HashMap;

use crate::error::PlatformError;

/// Platform-provided key-value persistence for node state.
///
/// Callers serialize their own state to bytes. Implementations may
/// back this with flash, files, a database, or in-memory (volatile).
pub trait PersistentState {
    /// Save a byte blob under the given key, overwriting any prior value.
    fn save(&mut self, key: &str, data: &[u8]) -> Result<(), PlatformError>;

    /// Load a previously saved blob. Returns `None` if the key doesn't exist.
    fn load(&self, key: &str) -> Option<Vec<u8>>;

    /// Delete a key and its data. No-op if key doesn't exist.
    fn delete(&mut self, key: &str) -> Result<(), PlatformError>;

    /// List all stored keys.
    fn keys(&self) -> Vec<String>;
}

/// In-memory volatile implementation of [`PersistentState`].
///
/// Useful for tests and for nodes that don't need persistence across
/// restarts (e.g., ephemeral unikernel appliances).
pub struct MemoryPersistentState {
    data: HashMap<String, Vec<u8>>,
}

impl MemoryPersistentState {
    pub fn new() -> Self {
        MemoryPersistentState {
            data: HashMap::new(),
        }
    }

    /// Number of stored entries.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl Default for MemoryPersistentState {
    fn default() -> Self {
        Self::new()
    }
}

impl PersistentState for MemoryPersistentState {
    fn save(&mut self, key: &str, data: &[u8]) -> Result<(), PlatformError> {
        self.data.insert(key.into(), data.to_vec());
        Ok(())
    }

    fn load(&self, key: &str) -> Option<Vec<u8>> {
        self.data.get(key).cloned()
    }

    fn delete(&mut self, key: &str) -> Result<(), PlatformError> {
        self.data.remove(key);
        Ok(())
    }

    fn keys(&self) -> Vec<String> {
        self.data.keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_and_load() {
        let mut store = MemoryPersistentState::new();
        store.save("identity", b"my-64-byte-key-material").unwrap();
        let loaded = store.load("identity").unwrap();
        assert_eq!(loaded, b"my-64-byte-key-material");
    }

    #[test]
    fn load_missing_key_returns_none() {
        let store = MemoryPersistentState::new();
        assert!(store.load("nonexistent").is_none());
    }

    #[test]
    fn save_overwrites_existing() {
        let mut store = MemoryPersistentState::new();
        store.save("config", b"v1").unwrap();
        store.save("config", b"v2").unwrap();
        assert_eq!(store.load("config").unwrap(), b"v2");
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn delete_removes_key() {
        let mut store = MemoryPersistentState::new();
        store.save("temp", b"data").unwrap();
        assert_eq!(store.len(), 1);
        store.delete("temp").unwrap();
        assert!(store.load("temp").is_none());
        assert!(store.is_empty());
    }

    #[test]
    fn delete_missing_key_is_noop() {
        let mut store = MemoryPersistentState::new();
        store.delete("ghost").unwrap();
    }

    #[test]
    fn keys_lists_all_stored() {
        let mut store = MemoryPersistentState::new();
        store.save("identity", b"key").unwrap();
        store.save("path_table", b"paths").unwrap();
        store.save("node_config", b"cfg").unwrap();
        let mut keys = store.keys();
        keys.sort();
        assert_eq!(keys, vec!["identity", "node_config", "path_table"]);
    }

    #[test]
    fn default_is_empty() {
        let store = MemoryPersistentState::default();
        assert!(store.is_empty());
    }

    #[test]
    fn works_as_trait_object() {
        let mut store = MemoryPersistentState::new();
        let ps: &mut dyn PersistentState = &mut store;
        ps.save("via-dyn", b"dynamic dispatch").unwrap();
        assert_eq!(ps.load("via-dyn").unwrap(), b"dynamic dispatch");
    }
}
