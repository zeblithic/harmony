//! Example: mail-like workflow using harmony-db.
//!
//! Demonstrates: multi-table DB, insert, mark-read via update_meta,
//! move-to-trash via remove+insert, commit, diff, rebuild.
//!
//! Run: cargo run -p harmony-db --example mail_workflow

use harmony_content::book::MemoryBookStore;
use harmony_db::{EntryMeta, HarmonyDb};

fn meta(read: bool, snippet: &str) -> EntryMeta {
    EntryMeta {
        flags: if read { 1 } else { 0 },
        snippet: snippet.to_string(),
    }
}

fn main() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = HarmonyDb::open(dir.path()).unwrap();

    println!("=== Receive 5 messages ===");
    for i in 0..5 {
        let key = format!("msg{i:04}");
        let body = format!("Message body #{i}");
        let snippet = format!("Subject {i}");
        db.insert("inbox", key.as_bytes(), body.as_bytes(), meta(false, &snippet))
            .unwrap();
    }
    println!("Inbox: {} messages", db.table_len("inbox"));

    // Initial commit.
    let mut store = MemoryBookStore::new();
    let root1 = db.commit(Some(&mut store)).unwrap();
    println!(
        "Commit 1: {}",
        hex::encode(root1.to_bytes())
    );

    println!("\n=== Mark 2 as read ===");
    db.update_meta("inbox", b"msg0000", meta(true, "Subject 0"))
        .unwrap();
    db.update_meta("inbox", b"msg0001", meta(true, "Subject 1"))
        .unwrap();

    println!("\n=== Move msg0002 to trash ===");
    let _entry = db.remove("inbox", b"msg0002").unwrap().unwrap();
    db.insert(
        "trash",
        b"msg0002",
        b"Message body #2",
        meta(false, "Subject 2"),
    )
    .unwrap();
    println!(
        "Inbox: {}, Trash: {}",
        db.table_len("inbox"),
        db.table_len("trash")
    );

    println!("\n=== Send a message ===");
    db.insert("sent", b"out0001", b"Hey there!", meta(true, "Outgoing"))
        .unwrap();

    // Second commit.
    let root2 = db.commit(Some(&mut store)).unwrap();
    println!(
        "Commit 2: {}",
        hex::encode(root2.to_bytes())
    );

    println!("\n=== Diff commit 1 vs 2 ===");
    let diff = db.diff(root1, root2).unwrap();
    for (table, td) in &diff.tables {
        println!(
            "  {table}: +{} added, -{} removed, ~{} changed",
            td.added.len(),
            td.removed.len(),
            td.changed.len()
        );
    }

    println!("\n=== Rebuild from CAS ===");
    let dir2 = tempfile::tempdir().unwrap();
    let db2 = HarmonyDb::open_from_cas(dir2.path(), root2, &store).unwrap();
    println!("Rebuilt inbox: {} messages", db2.table_len("inbox"));
    println!("Rebuilt sent:  {} messages", db2.table_len("sent"));
    println!("Rebuilt trash: {} messages", db2.table_len("trash"));

    // Verify round-trip.
    assert_eq!(db2.table_len("inbox"), 4);
    assert_eq!(db2.table_len("sent"), 1);
    assert_eq!(db2.table_len("trash"), 1);
    assert_eq!(
        db2.get_entry("inbox", b"msg0000").unwrap().metadata.flags,
        1,
    );
    assert!(db2.get("sent", b"out0001").unwrap().is_some());

    println!("\nAll assertions passed!");
}
