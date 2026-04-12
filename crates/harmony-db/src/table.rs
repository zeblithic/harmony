use crate::types::{Entry, EntryMeta};

/// A named collection of sorted key-value entries.
///
/// Maintains lexicographic order by key. Uses binary search for lookups
/// and slice views for range scans. No filesystem I/O — pure data structure.
#[derive(Debug, Clone)]
pub(crate) struct Table {
    pub(crate) entries: Vec<Entry>,
}

impl Table {
    pub fn new() -> Self {
        Table { entries: Vec::new() }
    }

    /// Insert or replace an entry. Maintains sorted order by key.
    /// Returns the previous entry if the key already existed.
    pub fn upsert(&mut self, entry: Entry) -> Option<Entry> {
        match self.entries.binary_search_by(|e| e.key.cmp(&entry.key)) {
            Ok(idx) => {
                let old = std::mem::replace(&mut self.entries[idx], entry);
                Some(old)
            }
            Err(idx) => {
                self.entries.insert(idx, entry);
                None
            }
        }
    }

    /// Look up an entry by key.
    pub fn get_entry(&self, key: &[u8]) -> Option<&Entry> {
        let idx = self.entries.binary_search_by(|e| e.key.as_slice().cmp(key)).ok()?;
        Some(&self.entries[idx])
    }

    /// Remove an entry by key. Returns the removed entry if found.
    pub fn remove(&mut self, key: &[u8]) -> Option<Entry> {
        let idx = self.entries.binary_search_by(|e| e.key.as_slice().cmp(key)).ok()?;
        Some(self.entries.remove(idx))
    }

    /// Update metadata for an existing entry. Returns false if key not found.
    pub fn update_meta(&mut self, key: &[u8], meta: EntryMeta) -> bool {
        if let Ok(idx) = self.entries.binary_search_by(|e| e.key.as_slice().cmp(key)) {
            self.entries[idx].metadata = meta;
            true
        } else {
            false
        }
    }

    /// Ordered slice over a key range [start, end).
    pub fn range(&self, start: &[u8], end: &[u8]) -> &[Entry] {
        let lo = match self.entries.binary_search_by(|e| e.key.as_slice().cmp(start)) {
            Ok(i) | Err(i) => i,
        };
        let hi = match self.entries.binary_search_by(|e| e.key.as_slice().cmp(end)) {
            Ok(i) | Err(i) => i,
        };
        &self.entries[lo..hi]
    }

    /// All entries in sorted order.
    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use harmony_content::{ContentFlags, ContentId};

    fn make_entry(key: &[u8], flags: u64, snippet: &str) -> Entry {
        let cid = ContentId::for_book(key, ContentFlags::default()).unwrap();
        Entry {
            key: key.to_vec(),
            value_cid: cid,
            timestamp: 1000,
            metadata: EntryMeta { flags, snippet: snippet.to_string() },
        }
    }

    #[test]
    fn insert_and_get_entry() {
        let mut t = Table::new();
        let e = make_entry(b"hello", 0, "greet");
        assert!(t.upsert(e.clone()).is_none());
        let found = t.get_entry(b"hello").unwrap();
        assert_eq!(found.metadata.snippet, "greet");
    }

    #[test]
    fn upsert_replaces_existing() {
        let mut t = Table::new();
        t.upsert(make_entry(b"key", 0, "old"));
        let old = t.upsert(make_entry(b"key", 1, "new"));
        assert!(old.is_some());
        assert_eq!(old.unwrap().metadata.snippet, "old");
        assert_eq!(t.get_entry(b"key").unwrap().metadata.snippet, "new");
        assert_eq!(t.len(), 1);
    }

    #[test]
    fn sorted_order_maintained() {
        let mut t = Table::new();
        t.upsert(make_entry(b"cherry", 0, ""));
        t.upsert(make_entry(b"apple", 0, ""));
        t.upsert(make_entry(b"banana", 0, ""));
        let keys: Vec<&[u8]> = t.entries().iter().map(|e| e.key.as_slice()).collect();
        assert_eq!(keys, vec![b"apple".as_slice(), b"banana", b"cherry"]);
    }

    #[test]
    fn remove_entry() {
        let mut t = Table::new();
        t.upsert(make_entry(b"a", 0, ""));
        t.upsert(make_entry(b"b", 0, ""));
        let removed = t.remove(b"a");
        assert!(removed.is_some());
        assert!(t.get_entry(b"a").is_none());
        assert_eq!(t.len(), 1);
    }

    #[test]
    fn remove_nonexistent_returns_none() {
        let mut t = Table::new();
        assert!(t.remove(b"nope").is_none());
    }

    #[test]
    fn update_meta() {
        let mut t = Table::new();
        t.upsert(make_entry(b"key", 0, "old"));
        let ok = t.update_meta(b"key", EntryMeta { flags: 1, snippet: "new".into() });
        assert!(ok);
        assert_eq!(t.get_entry(b"key").unwrap().metadata.flags, 1);
    }

    #[test]
    fn update_meta_nonexistent_returns_false() {
        let mut t = Table::new();
        assert!(!t.update_meta(b"nope", EntryMeta { flags: 0, snippet: "".into() }));
    }

    #[test]
    fn range_query() {
        let mut t = Table::new();
        t.upsert(make_entry(b"a", 0, ""));
        t.upsert(make_entry(b"b", 0, ""));
        t.upsert(make_entry(b"c", 0, ""));
        t.upsert(make_entry(b"d", 0, ""));
        let range = t.range(b"b", b"d");
        let keys: Vec<&[u8]> = range.iter().map(|e| e.key.as_slice()).collect();
        assert_eq!(keys, vec![b"b".as_slice(), b"c"]);
    }

    #[test]
    fn range_empty() {
        let mut t = Table::new();
        t.upsert(make_entry(b"a", 0, ""));
        assert!(t.range(b"x", b"z").is_empty());
    }

    #[test]
    fn get_entry_nonexistent() {
        let t = Table::new();
        assert!(t.get_entry(b"nope").is_none());
    }
}
