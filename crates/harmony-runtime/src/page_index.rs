// SPDX-License-Identifier: Apache-2.0 OR MIT
//! PageIndex — maps mode-00 PageAddr to indexed book pages for fast lookup.
//!
//! The primary index key is the mode-00 (Sha256Msb) address. Each entry
//! stores the full set of 4 algorithm variants so callers can cross-reference
//! any mode without re-hashing.

use std::collections::HashMap;

use harmony_athenaeum::{Book, PageAddr, ALGO_COUNT};
use harmony_content::ContentId;

/// A single indexed page: which book it belongs to, its page number within
/// that book, and all 4 algorithm-variant addresses.
#[derive(Debug, Clone)]
pub struct PageIndexEntry {
    pub book_cid: ContentId,
    pub page_num: u8,
    pub addrs: [PageAddr; ALGO_COUNT],
}

/// In-memory index keyed by mode-00 (Sha256Msb) page addresses.
pub struct PageIndex {
    by_addr00: HashMap<PageAddr, Vec<PageIndexEntry>>,
    total: usize,
}

impl PageIndex {
    /// Create an empty index.
    pub fn new() -> Self {
        PageIndex {
            by_addr00: HashMap::new(),
            total: 0,
        }
    }

    /// Index all data pages from a book.
    ///
    /// Each data page is inserted under its mode-00 (Sha256Msb) address.
    pub fn insert_book(&mut self, cid: ContentId, book: &Book) {
        for (i, addrs) in book.data_pages().iter().enumerate() {
            assert!(
                i <= u8::MAX as usize,
                "data page index {i} exceeds u8::MAX — BOOK_MAX_SIZE guarantees at most 256 pages"
            );
            let entry = PageIndexEntry {
                book_cid: cid,
                page_num: i as u8,
                addrs: *addrs,
            };
            self.by_addr00.entry(addrs[0]).or_default().push(entry);
            self.total += 1;
        }
    }

    /// Look up all entries matching a mode-00 address.
    pub fn lookup(&self, addr_00: &PageAddr) -> &[PageIndexEntry] {
        self.by_addr00
            .get(addr_00)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Multi-field query: look up by mode-00 address, then filter by any
    /// combination of the other mode addresses, book CID, and page number.
    ///
    /// `addr_00` is required for the initial lookup; remaining fields narrow
    /// the result set.
    #[allow(clippy::too_many_arguments)]
    pub fn match_query(
        &self,
        addr_00: Option<&PageAddr>,
        addr_01: Option<&PageAddr>,
        addr_10: Option<&PageAddr>,
        addr_11: Option<&PageAddr>,
        book_cid: Option<&ContentId>,
        page_num: Option<u8>,
    ) -> Vec<&PageIndexEntry> {
        let candidates: Box<dyn Iterator<Item = &PageIndexEntry>> = match addr_00 {
            Some(a) => Box::new(self.lookup(a).iter()),
            None => Box::new(self.by_addr00.values().flatten()),
        };

        candidates
            .filter(|e| addr_01.is_none_or(|a| e.addrs[1] == *a))
            .filter(|e| addr_10.is_none_or(|a| e.addrs[2] == *a))
            .filter(|e| addr_11.is_none_or(|a| e.addrs[3] == *a))
            .filter(|e| book_cid.is_none_or(|c| e.book_cid == *c))
            .filter(|e| page_num.is_none_or(|n| e.page_num == n))
            .collect()
    }

    /// Total number of indexed page entries.
    pub fn len(&self) -> usize {
        self.total
    }

    /// Returns `true` if the index contains no entries.
    pub fn is_empty(&self) -> bool {
        self.total == 0
    }

    /// Iterate over all indexed mode-00 (Sha256Msb) page addresses.
    ///
    /// Used by the page Bloom filter builder to enumerate local page addresses
    /// without exposing the internal HashMap.
    pub fn addr00_iter(&self) -> impl Iterator<Item = &PageAddr> {
        self.by_addr00.keys()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use harmony_athenaeum::PAGE_SIZE;

    #[test]
    fn insert_and_lookup() {
        let data = vec![0xAA; PAGE_SIZE];
        let cid_bytes = [0x11u8; 32];
        let book = Book::from_book(cid_bytes, &data).unwrap();
        let cid = ContentId::from_bytes(cid_bytes);

        let mut idx = PageIndex::new();
        idx.insert_book(cid, &book);

        assert_eq!(idx.len(), 1);
        let addr_00 = book.data_pages()[0][0];
        let results = idx.lookup(&addr_00);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].book_cid, cid);
        assert_eq!(results[0].page_num, 0);
    }

    #[test]
    fn multiple_pages_indexed() {
        // 8192 bytes -> 2 data pages
        let data = vec![0xAA; PAGE_SIZE * 2];
        let cid_bytes = [0x11u8; 32];
        let book = Book::from_book(cid_bytes, &data).unwrap();
        let cid = ContentId::from_bytes(cid_bytes);

        let mut idx = PageIndex::new();
        idx.insert_book(cid, &book);

        assert_eq!(idx.len(), 2);

        for (i, addrs) in book.data_pages().iter().enumerate() {
            let results = idx.lookup(&addrs[0]);
            assert!(!results.is_empty());
            let entry = results.iter().find(|e| e.page_num == i as u8).unwrap();
            assert_eq!(entry.book_cid, cid);
        }
    }

    #[test]
    fn match_query_filters_by_page_num() {
        // Use distinct data per page so each page gets a unique address.
        let mut data = vec![0u8; PAGE_SIZE * 3];
        data[0] = 0x01; // page 0 differs
        data[PAGE_SIZE] = 0x02; // page 1 differs
        data[PAGE_SIZE * 2] = 0x03; // page 2 differs
        let cid_bytes = [0x22u8; 32];
        let book = Book::from_book(cid_bytes, &data).unwrap();
        let cid = ContentId::from_bytes(cid_bytes);

        let mut idx = PageIndex::new();
        idx.insert_book(cid, &book);

        // Query for page 1 specifically
        let addr_00 = book.data_pages()[1][0];
        let results = idx.match_query(Some(&addr_00), None, None, None, None, Some(1));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].page_num, 1);

        // Query for page 0 against page 1's address should yield nothing
        let results = idx.match_query(Some(&addr_00), None, None, None, None, Some(0));
        assert!(results.is_empty());
    }

    #[test]
    fn match_query_filters_by_book_cid() {
        let data_a = vec![0xCC; PAGE_SIZE];
        let data_b = vec![0xDD; PAGE_SIZE];
        let cid_a_bytes = [0x33u8; 32];
        let cid_b_bytes = [0x44u8; 32];
        let book_a = Book::from_book(cid_a_bytes, &data_a).unwrap();
        let book_b = Book::from_book(cid_b_bytes, &data_b).unwrap();
        let cid_a = ContentId::from_bytes(cid_a_bytes);
        let cid_b = ContentId::from_bytes(cid_b_bytes);

        let mut idx = PageIndex::new();
        idx.insert_book(cid_a, &book_a);
        idx.insert_book(cid_b, &book_b);

        assert_eq!(idx.len(), 2);

        // Filter by book_a's CID — should only return book_a's page
        let addr_00_a = book_a.data_pages()[0][0];
        let results = idx.match_query(Some(&addr_00_a), None, None, None, Some(&cid_a), None);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].book_cid, cid_a);

        // Filter by book_b's CID against book_a's address — should return nothing
        let results = idx.match_query(Some(&addr_00_a), None, None, None, Some(&cid_b), None);
        assert!(results.is_empty());
    }

    #[test]
    fn multiple_books_same_addr() {
        // Two books with identical data but different CIDs produce the same
        // page addresses.  Both must appear under the same mode-00 key.
        let data = vec![0xEE; PAGE_SIZE];
        let cid_a_bytes = [0x55u8; 32];
        let cid_b_bytes = [0x66u8; 32];
        let book_a = Book::from_book(cid_a_bytes, &data).unwrap();
        let book_b = Book::from_book(cid_b_bytes, &data).unwrap();
        let cid_a = ContentId::from_bytes(cid_a_bytes);
        let cid_b = ContentId::from_bytes(cid_b_bytes);

        // Verify the mode-00 addresses actually collide.
        assert_eq!(
            book_a.data_pages()[0][0],
            book_b.data_pages()[0][0],
            "same data should produce the same page address"
        );

        let mut idx = PageIndex::new();
        idx.insert_book(cid_a, &book_a);
        idx.insert_book(cid_b, &book_b);

        assert_eq!(idx.len(), 2);

        // Both entries live under the same mode-00 address.
        let addr_00 = book_a.data_pages()[0][0];
        let results = idx.lookup(&addr_00);
        assert_eq!(results.len(), 2);

        // Can disambiguate by book CID.
        let only_a = idx.match_query(Some(&addr_00), None, None, None, Some(&cid_a), None);
        assert_eq!(only_a.len(), 1);
        assert_eq!(only_a[0].book_cid, cid_a);

        let only_b = idx.match_query(Some(&addr_00), None, None, None, Some(&cid_b), None);
        assert_eq!(only_b.len(), 1);
        assert_eq!(only_b[0].book_cid, cid_b);
    }

    #[test]
    fn empty_index_returns_empty() {
        let idx = PageIndex::new();
        assert!(idx.is_empty());
        assert_eq!(idx.len(), 0);

        // Arbitrary address lookup should return empty
        let dummy =
            harmony_athenaeum::PageAddr::new(0x0000_0000, harmony_athenaeum::Algorithm::Sha256Msb);
        assert!(idx.lookup(&dummy).is_empty());
    }
}
