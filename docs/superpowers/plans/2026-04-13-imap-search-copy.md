# IMAP SEARCH + COPY/MOVE Wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Wire IMAP SEARCH and COPY/MOVE to real operations, completing the IMAP command set for harmony-mail.

**Architecture:** A pure `matches_criteria` function evaluates search criteria against message metadata + optional CAS content. COPY/MOVE uses existing `ImapStore::copy_messages` with flag+expunge for MOVE. Both reuse `resolve_sequence_set` from PR #225.

**Tech Stack:** Rust, tokio, harmony-content (CAS/DAG), harmony-db (DiskBookStore), rusqlite

---

## File Structure

| File | Responsibility |
|------|---------------|
| `crates/harmony-mail/src/server.rs` | Replace SEARCH and COPY stubs, add `parse_imap_date`, `matches_single_criterion`, `matches_criteria` helpers |

No new files — all changes in existing server.rs.

---

### Task 1: Implement parse_imap_date with tests

**Files:**
- Modify: `crates/harmony-mail/src/server.rs`

- [ ] **Step 1: Write the tests**

Add to the `sequence_set_tests` module (or a new `search_tests` module) in `crates/harmony-mail/src/server.rs`:

```rust
    #[test]
    fn parse_imap_date_standard() {
        // 13-Apr-2026 00:00:00 UTC
        let ts = super::parse_imap_date("13-Apr-2026");
        // 2026-04-13 = days since epoch: 20,556 * 86400 = 1776038400
        assert!(ts.is_some());
        let t = ts.unwrap();
        // April 13, 2026 is 20556 days after Jan 1 1970
        assert_eq!(t, 20556 * 86400);
    }

    #[test]
    fn parse_imap_date_all_months() {
        assert!(super::parse_imap_date("01-Jan-2000").is_some());
        assert!(super::parse_imap_date("01-Feb-2000").is_some());
        assert!(super::parse_imap_date("01-Mar-2000").is_some());
        assert!(super::parse_imap_date("01-Apr-2000").is_some());
        assert!(super::parse_imap_date("01-May-2000").is_some());
        assert!(super::parse_imap_date("01-Jun-2000").is_some());
        assert!(super::parse_imap_date("01-Jul-2000").is_some());
        assert!(super::parse_imap_date("01-Aug-2000").is_some());
        assert!(super::parse_imap_date("01-Sep-2000").is_some());
        assert!(super::parse_imap_date("01-Oct-2000").is_some());
        assert!(super::parse_imap_date("01-Nov-2000").is_some());
        assert!(super::parse_imap_date("01-Dec-2000").is_some());
    }

    #[test]
    fn parse_imap_date_epoch() {
        let ts = super::parse_imap_date("01-Jan-1970");
        assert_eq!(ts, Some(0));
    }

    #[test]
    fn parse_imap_date_invalid() {
        assert!(super::parse_imap_date("invalid").is_none());
        assert!(super::parse_imap_date("32-Jan-2026").is_none());
        assert!(super::parse_imap_date("01-Xyz-2026").is_none());
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-mail parse_imap_date -- --nocapture`
Expected: FAIL — `parse_imap_date` not found

- [ ] **Step 3: Implement parse_imap_date**

Add this function near the bottom of `crates/harmony-mail/src/server.rs`, above the test modules:

```rust
/// Parse an IMAP date string ("DD-Mon-YYYY") to a Unix timestamp (midnight UTC).
/// Returns None if the format is invalid.
fn parse_imap_date(s: &str) -> Option<u64> {
    let parts: Vec<&str> = s.split('-').collect();
    if parts.len() != 3 {
        return None;
    }
    let day: u32 = parts[0].parse().ok()?;
    if day == 0 || day > 31 {
        return None;
    }
    let month: u32 = match parts[1].to_ascii_lowercase().as_str() {
        "jan" => 1,
        "feb" => 2,
        "mar" => 3,
        "apr" => 4,
        "may" => 5,
        "jun" => 6,
        "jul" => 7,
        "aug" => 8,
        "sep" => 9,
        "oct" => 10,
        "nov" => 11,
        "dec" => 12,
        _ => return None,
    };
    let year: u64 = parts[2].parse().ok()?;
    if year < 1970 {
        return None;
    }

    // Days from epoch to start of year
    let mut days: u64 = 0;
    for y in 1970..year {
        days += if is_leap(y) { 366 } else { 365 };
    }
    // Days from start of year to start of month
    let month_days = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    for m in 1..month {
        days += month_days[m as usize] as u64;
        if m == 2 && is_leap(year) {
            days += 1;
        }
    }
    days += (day - 1) as u64;
    Some(days * 86400)
}

fn is_leap(year: u64) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-mail parse_imap_date -- --nocapture`
Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-mail/src/server.rs
git commit -m "feat(mail): add parse_imap_date helper for SEARCH (ZEB-111)"
```

---

### Task 2: Implement matches_criteria with tests

**Files:**
- Modify: `crates/harmony-mail/src/server.rs`

- [ ] **Step 1: Write the tests**

Add to the test module in `crates/harmony-mail/src/server.rs`:

```rust
    use crate::imap_parse::SearchKey;

    fn msg_row_with(uid: u32, internal_date: u64, rfc822_size: u32) -> MessageRow {
        MessageRow {
            id: uid as i64,
            mailbox_id: 1,
            uid,
            harmony_msg_id: [0u8; MESSAGE_ID_LEN],
            message_cid: None,
            internal_date,
            rfc822_size,
        }
    }

    #[test]
    fn search_all() {
        let row = msg_row_with(1, 1000, 100);
        assert!(super::matches_criteria(&[SearchKey::All], &row, &[], 1, None));
    }

    #[test]
    fn search_flag_seen() {
        let row = msg_row_with(1, 1000, 100);
        let seen = vec!["\\Seen".to_string()];
        assert!(super::matches_criteria(&[SearchKey::Seen], &row, &seen, 1, None));
        assert!(!super::matches_criteria(&[SearchKey::Unseen], &row, &seen, 1, None));
        assert!(!super::matches_criteria(&[SearchKey::Seen], &row, &[], 1, None));
        assert!(super::matches_criteria(&[SearchKey::Unseen], &row, &[], 1, None));
    }

    #[test]
    fn search_flag_deleted_draft_flagged_answered() {
        let row = msg_row_with(1, 1000, 100);
        let flags = vec!["\\Deleted".to_string(), "\\Draft".to_string()];
        assert!(super::matches_criteria(&[SearchKey::Deleted], &row, &flags, 1, None));
        assert!(super::matches_criteria(&[SearchKey::Draft], &row, &flags, 1, None));
        assert!(!super::matches_criteria(&[SearchKey::Undeleted], &row, &flags, 1, None));
        assert!(!super::matches_criteria(&[SearchKey::Undraft], &row, &flags, 1, None));
        assert!(!super::matches_criteria(&[SearchKey::Flagged], &row, &flags, 1, None));
        assert!(super::matches_criteria(&[SearchKey::Unflagged], &row, &flags, 1, None));
        assert!(!super::matches_criteria(&[SearchKey::Answered], &row, &flags, 1, None));
        assert!(super::matches_criteria(&[SearchKey::Unanswered], &row, &flags, 1, None));
    }

    #[test]
    fn search_recent_new_old() {
        let row = msg_row_with(1, 1000, 100);
        let recent = vec!["\\Recent".to_string()];
        let recent_seen = vec!["\\Recent".to_string(), "\\Seen".to_string()];
        assert!(super::matches_criteria(&[SearchKey::Recent], &row, &recent, 1, None));
        assert!(!super::matches_criteria(&[SearchKey::Old], &row, &recent, 1, None));
        // New = Recent AND Unseen
        assert!(super::matches_criteria(&[SearchKey::New], &row, &recent, 1, None));
        assert!(!super::matches_criteria(&[SearchKey::New], &row, &recent_seen, 1, None));
    }

    #[test]
    fn search_size() {
        let row = msg_row_with(1, 1000, 500);
        assert!(super::matches_criteria(&[SearchKey::Larger(499)], &row, &[], 1, None));
        assert!(!super::matches_criteria(&[SearchKey::Larger(500)], &row, &[], 1, None));
        assert!(super::matches_criteria(&[SearchKey::Smaller(501)], &row, &[], 1, None));
        assert!(!super::matches_criteria(&[SearchKey::Smaller(500)], &row, &[], 1, None));
    }

    #[test]
    fn search_date_since_before_on() {
        // internal_date = 86400 (Jan 2, 1970)
        let row = msg_row_with(1, 86400, 100);
        assert!(super::matches_criteria(&[SearchKey::Since("02-Jan-1970".to_string())], &row, &[], 1, None));
        assert!(super::matches_criteria(&[SearchKey::Since("01-Jan-1970".to_string())], &row, &[], 1, None));
        assert!(!super::matches_criteria(&[SearchKey::Since("03-Jan-1970".to_string())], &row, &[], 1, None));
        assert!(super::matches_criteria(&[SearchKey::Before("03-Jan-1970".to_string())], &row, &[], 1, None));
        assert!(!super::matches_criteria(&[SearchKey::Before("02-Jan-1970".to_string())], &row, &[], 1, None));
        assert!(super::matches_criteria(&[SearchKey::On("02-Jan-1970".to_string())], &row, &[], 1, None));
        assert!(!super::matches_criteria(&[SearchKey::On("01-Jan-1970".to_string())], &row, &[], 1, None));
    }

    #[test]
    fn search_not_and_or() {
        let row = msg_row_with(1, 1000, 100);
        let seen = vec!["\\Seen".to_string()];
        // NOT Seen = Unseen
        assert!(!super::matches_criteria(&[SearchKey::Not(Box::new(SearchKey::Seen))], &row, &seen, 1, None));
        assert!(super::matches_criteria(&[SearchKey::Not(Box::new(SearchKey::Seen))], &row, &[], 1, None));
        // OR(Seen, Flagged) — Seen is true
        assert!(super::matches_criteria(
            &[SearchKey::Or(Box::new(SearchKey::Seen), Box::new(SearchKey::Flagged))],
            &row, &seen, 1, None
        ));
        // OR(Flagged, Deleted) — neither is true
        assert!(!super::matches_criteria(
            &[SearchKey::Or(Box::new(SearchKey::Flagged), Box::new(SearchKey::Deleted))],
            &row, &[], 1, None
        ));
    }

    #[test]
    fn search_multi_criteria_and() {
        let row = msg_row_with(1, 1000, 500);
        let seen = vec!["\\Seen".to_string()];
        // Seen AND Larger(100) — both true
        assert!(super::matches_criteria(&[SearchKey::Seen, SearchKey::Larger(100)], &row, &seen, 1, None));
        // Seen AND Larger(600) — Larger is false
        assert!(!super::matches_criteria(&[SearchKey::Seen, SearchKey::Larger(600)], &row, &seen, 1, None));
    }

    #[test]
    fn search_subject_body() {
        let row = msg_row_with(1, 1000, 100);
        let msg = test_message(); // from existing tests — subject="Test subject", body="Hello, world!"
        assert!(super::matches_criteria(
            &[SearchKey::Subject("test".to_string())], &row, &[], 1, Some(&msg)
        ));
        assert!(super::matches_criteria(
            &[SearchKey::Subject("TEST SUBJECT".to_string())], &row, &[], 1, Some(&msg)
        ));
        assert!(!super::matches_criteria(
            &[SearchKey::Subject("missing".to_string())], &row, &[], 1, Some(&msg)
        ));
        assert!(super::matches_criteria(
            &[SearchKey::Body("hello".to_string())], &row, &[], 1, Some(&msg)
        ));
        assert!(!super::matches_criteria(
            &[SearchKey::Body("missing".to_string())], &row, &[], 1, Some(&msg)
        ));
    }

    #[test]
    fn search_content_without_message_returns_false() {
        let row = msg_row_with(1, 1000, 100);
        assert!(!super::matches_criteria(&[SearchKey::Subject("test".to_string())], &row, &[], 1, None));
        assert!(!super::matches_criteria(&[SearchKey::Body("hello".to_string())], &row, &[], 1, None));
        assert!(!super::matches_criteria(&[SearchKey::From("user".to_string())], &row, &[], 1, None));
    }

    #[test]
    fn search_keyword() {
        let row = msg_row_with(1, 1000, 100);
        let flags = vec!["$Important".to_string()];
        assert!(super::matches_criteria(&[SearchKey::Keyword("$Important".to_string())], &row, &flags, 1, None));
        assert!(!super::matches_criteria(&[SearchKey::Keyword("$Other".to_string())], &row, &flags, 1, None));
        assert!(!super::matches_criteria(&[SearchKey::Unkeyword("$Important".to_string())], &row, &flags, 1, None));
        assert!(super::matches_criteria(&[SearchKey::Unkeyword("$Other".to_string())], &row, &flags, 1, None));
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-mail search_ -- --nocapture`
Expected: FAIL — `matches_criteria` not found

- [ ] **Step 3: Implement matches_criteria and matches_single_criterion**

Add these functions in `crates/harmony-mail/src/server.rs`, near `parse_imap_date`:

```rust
/// Evaluate IMAP SEARCH criteria against a message.
/// Multiple criteria are AND'd per RFC 9051.
fn matches_criteria(
    criteria: &[imap_parse::SearchKey],
    msg_row: &imap_store::MessageRow,
    flags: &[String],
    seqnum: u32,
    msg: Option<&crate::message::HarmonyMessage>,
) -> bool {
    criteria.iter().all(|key| matches_single_criterion(key, msg_row, flags, seqnum, msg))
}

/// Evaluate a single IMAP SEARCH criterion.
fn matches_single_criterion(
    key: &imap_parse::SearchKey,
    msg_row: &imap_store::MessageRow,
    flags: &[String],
    seqnum: u32,
    msg: Option<&crate::message::HarmonyMessage>,
) -> bool {
    use imap_parse::SearchKey;

    let has_flag = |f: &str| flags.iter().any(|fl| fl == f);

    match key {
        SearchKey::All => true,
        SearchKey::Seen => has_flag("\\Seen"),
        SearchKey::Unseen => !has_flag("\\Seen"),
        SearchKey::Flagged => has_flag("\\Flagged"),
        SearchKey::Unflagged => !has_flag("\\Flagged"),
        SearchKey::Answered => has_flag("\\Answered"),
        SearchKey::Unanswered => !has_flag("\\Answered"),
        SearchKey::Deleted => has_flag("\\Deleted"),
        SearchKey::Undeleted => !has_flag("\\Deleted"),
        SearchKey::Draft => has_flag("\\Draft"),
        SearchKey::Undraft => !has_flag("\\Draft"),
        SearchKey::Recent => has_flag("\\Recent"),
        SearchKey::New => has_flag("\\Recent") && !has_flag("\\Seen"),
        SearchKey::Old => !has_flag("\\Recent"),
        SearchKey::Keyword(s) => has_flag(s),
        SearchKey::Unkeyword(s) => !has_flag(s),

        SearchKey::Larger(n) => msg_row.rfc822_size > *n,
        SearchKey::Smaller(n) => msg_row.rfc822_size < *n,

        SearchKey::Since(date) => {
            parse_imap_date(date).map_or(false, |d| msg_row.internal_date >= d)
        }
        SearchKey::Before(date) => {
            parse_imap_date(date).map_or(false, |d| msg_row.internal_date < d)
        }
        SearchKey::On(date) => {
            parse_imap_date(date).map_or(false, |d| {
                // Same calendar day: internal_date falls within [d, d+86400)
                msg_row.internal_date >= d && msg_row.internal_date < d + 86400
            })
        }

        SearchKey::Uid(set) => {
            // Check if msg_row.uid is in the sequence set (treated as UID set)
            let dummy_msgs = [imap_store::MessageRow {
                id: msg_row.id,
                mailbox_id: msg_row.mailbox_id,
                uid: msg_row.uid,
                harmony_msg_id: msg_row.harmony_msg_id,
                message_cid: msg_row.message_cid,
                internal_date: msg_row.internal_date,
                rfc822_size: msg_row.rfc822_size,
            }];
            !resolve_sequence_set(set, true, &dummy_msgs).is_empty()
        }
        SearchKey::SequenceSet(set) => {
            let dummy_msgs = [imap_store::MessageRow {
                id: msg_row.id,
                mailbox_id: msg_row.mailbox_id,
                uid: msg_row.uid,
                harmony_msg_id: msg_row.harmony_msg_id,
                message_cid: msg_row.message_cid,
                internal_date: msg_row.internal_date,
                rfc822_size: msg_row.rfc822_size,
            }];
            !resolve_sequence_set(set, false, &dummy_msgs).is_empty()
        }

        SearchKey::Subject(s) => {
            msg.map_or(false, |m| m.subject.to_lowercase().contains(&s.to_lowercase()))
        }
        SearchKey::Body(s) => {
            msg.map_or(false, |m| m.body.to_lowercase().contains(&s.to_lowercase()))
        }
        SearchKey::From(s) => {
            msg.map_or(false, |m| {
                let sender_hex = hex::encode(m.sender_address);
                sender_hex.to_lowercase().contains(&s.to_lowercase())
            })
        }
        SearchKey::To(s) => {
            msg.map_or(false, |m| {
                m.recipients.iter().any(|r| {
                    let addr_hex = hex::encode(r.address_hash);
                    addr_hex.to_lowercase().contains(&s.to_lowercase())
                })
            })
        }
        SearchKey::Header(name, value) => {
            msg.map_or(false, |m| {
                let field_content = match name.to_lowercase().as_str() {
                    "subject" => Some(m.subject.as_str()),
                    "from" => None, // From is an address hash, not a text field
                    _ => None,
                };
                match field_content {
                    Some(content) => content.to_lowercase().contains(&value.to_lowercase()),
                    None => {
                        // For From header, match against hex-encoded sender
                        if name.eq_ignore_ascii_case("from") {
                            let sender_hex = hex::encode(m.sender_address);
                            sender_hex.to_lowercase().contains(&value.to_lowercase())
                        } else {
                            false // Unknown header
                        }
                    }
                }
            })
        }

        SearchKey::Not(inner) => !matches_single_criterion(inner, msg_row, flags, seqnum, msg),
        SearchKey::Or(a, b) => {
            matches_single_criterion(a, msg_row, flags, seqnum, msg)
                || matches_single_criterion(b, msg_row, flags, seqnum, msg)
        }
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-mail search_ -- --nocapture`
Expected: all 11 tests PASS

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-mail/src/server.rs
git commit -m "feat(mail): add matches_criteria for IMAP SEARCH evaluation (ZEB-111)"
```

---

### Task 3: Wire SEARCH handler

**Files:**
- Modify: `crates/harmony-mail/src/server.rs`

- [ ] **Step 1: Write a SEARCH integration test**

Add to the test module in `crates/harmony-mail/src/server.rs`:

```rust
    #[test]
    fn search_by_flags_integration() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("imap.db");
        let imap_store = ImapStore::open(&db_path).unwrap();
        imap_store.initialize_default_mailboxes().unwrap();

        // Insert 3 messages
        let uid1 = imap_store.insert_message("INBOX", &[1u8; MESSAGE_ID_LEN], None, 1000, 100).unwrap();
        let uid2 = imap_store.insert_message("INBOX", &[2u8; MESSAGE_ID_LEN], None, 2000, 200).unwrap();
        let uid3 = imap_store.insert_message("INBOX", &[3u8; MESSAGE_ID_LEN], None, 3000, 300).unwrap();

        // Flag message 1 as Seen, message 2 as Seen+Flagged
        let mbox = imap_store.get_mailbox("INBOX").unwrap().unwrap();
        let msgs = imap_store.get_messages(mbox.id).unwrap();
        imap_store.add_flags(msgs[0].id, &["\\Seen"]).unwrap();
        imap_store.add_flags(msgs[1].id, &["\\Seen", "\\Flagged"]).unwrap();

        // SEARCH SEEN should match UIDs 1 and 2
        let mut results = Vec::new();
        for (idx, msg) in msgs.iter().enumerate() {
            let flags = imap_store.get_flags(msg.id).unwrap();
            if super::matches_criteria(&[SearchKey::Seen], msg, &flags, (idx + 1) as u32, None) {
                results.push(msg.uid);
            }
        }
        assert_eq!(results, vec![uid1, uid2]);

        // SEARCH FLAGGED should match UID 2 only
        let mut results = Vec::new();
        for (idx, msg) in msgs.iter().enumerate() {
            let flags = imap_store.get_flags(msg.id).unwrap();
            if super::matches_criteria(&[SearchKey::Flagged], msg, &flags, (idx + 1) as u32, None) {
                results.push(msg.uid);
            }
        }
        assert_eq!(results, vec![uid2]);

        // SEARCH UNSEEN should match UID 3 only
        let mut results = Vec::new();
        for (idx, msg) in msgs.iter().enumerate() {
            let flags = imap_store.get_flags(msg.id).unwrap();
            if super::matches_criteria(&[SearchKey::Unseen], msg, &flags, (idx + 1) as u32, None) {
                results.push(msg.uid);
            }
        }
        assert_eq!(results, vec![uid3]);
    }
```

- [ ] **Step 2: Run integration test to verify it passes**

Run: `cargo test -p harmony-mail search_by_flags_integration -- --nocapture`
Expected: PASS (uses already-implemented matches_criteria)

- [ ] **Step 3: Replace the SEARCH stub**

In `crates/harmony-mail/src/server.rs`, replace the `ImapAction::Search { .. }` match arm (currently returns `NO [CANNOT] SEARCH not yet implemented`) with:

```rust
            ImapAction::Search { criteria, uid_mode } => {
                let search_result: Result<Vec<u32>, String> = (|| {
                    let mailbox_name = match &session.state {
                        ImapState::Selected { mailbox, .. } => mailbox.name.clone(),
                        _ => return Err("no mailbox selected".to_string()),
                    };
                    let mbox = store
                        .get_mailbox(&mailbox_name)
                        .map_err(|e| e.to_string())?
                        .ok_or_else(|| "mailbox not found".to_string())?;
                    let all_msgs = store.get_messages(mbox.id).map_err(|e| e.to_string())?;

                    // Check if any criterion needs CAS content
                    let needs_cas = criteria_need_cas(criteria);

                    let mut results = Vec::new();
                    for (idx, msg_row) in all_msgs.iter().enumerate() {
                        let seqnum = (idx + 1) as u32;
                        let flags = store.get_flags(msg_row.id).map_err(|e| e.to_string())?;

                        let harmony_msg = if needs_cas {
                            match msg_row.message_cid {
                                Some(cid_bytes) => {
                                    let csp_inner = content_store_path.to_path_buf();
                                    // Inline CAS retrieval (blocking — acceptable for search)
                                    let book_store = harmony_db::DiskBookStore::new(&csp_inner);
                                    let content_id = harmony_content::cid::ContentId::from_bytes(cid_bytes);
                                    match harmony_content::dag::reassemble(&content_id, &book_store) {
                                        Ok(bytes) => crate::message::HarmonyMessage::from_bytes(&bytes).ok(),
                                        Err(_) => None,
                                    }
                                }
                                None => None,
                            }
                        } else {
                            None
                        };

                        if matches_criteria(criteria, msg_row, &flags, seqnum, harmony_msg.as_ref()) {
                            results.push(if *uid_mode { msg_row.uid } else { seqnum });
                        }
                    }

                    Ok(results)
                })();

                match search_result {
                    Ok(results) => {
                        let callback = session.handle(ImapEvent::SearchComplete { results });
                        execute_imap_actions(&callback, writer).await?;
                    }
                    Err(reason) => {
                        if let Some(tag) = session.pending_tag.take() {
                            writer
                                .write_all(format!("{tag} NO {reason}\r\n").as_bytes())
                                .await?;
                            writer.flush().await?;
                        }
                    }
                }
            }
```

Also add this helper function near `matches_criteria`:

```rust
/// Check if any search criterion requires CAS message content.
fn criteria_need_cas(criteria: &[imap_parse::SearchKey]) -> bool {
    use imap_parse::SearchKey;
    criteria.iter().any(|key| match key {
        SearchKey::From(_) | SearchKey::To(_) | SearchKey::Subject(_)
        | SearchKey::Body(_) | SearchKey::Header(_, _) => true,
        SearchKey::Not(inner) => criteria_need_cas(std::slice::from_ref(inner.as_ref())),
        SearchKey::Or(a, b) => {
            criteria_need_cas(std::slice::from_ref(a.as_ref()))
                || criteria_need_cas(std::slice::from_ref(b.as_ref()))
        }
        _ => false,
    })
}
```

Note: The SEARCH handler uses inline CAS retrieval (not `spawn_blocking`) because `process_imap_async_actions` already runs inside the connection's async task, and search iterates many messages — spawning a blocking task per message would flood the thread pool. The DiskBookStore file reads are fast (cached by OS) and acceptable inline for search.

- [ ] **Step 4: Verify compilation and all tests pass**

Run: `cargo check -p harmony-mail`
Run: `cargo test -p harmony-mail -- --nocapture`
Expected: compiles, all tests pass

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-mail/src/server.rs
git commit -m "feat(mail): wire IMAP SEARCH to criteria evaluation (ZEB-111)"
```

---

### Task 4: Wire COPY/MOVE handler

**Files:**
- Modify: `crates/harmony-mail/src/server.rs`

- [ ] **Step 1: Write COPY and MOVE tests**

Add to the test module in `crates/harmony-mail/src/server.rs`:

```rust
    #[test]
    fn copy_messages_preserves_flags() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("imap.db");
        let imap_store = ImapStore::open(&db_path).unwrap();
        imap_store.initialize_default_mailboxes().unwrap();
        imap_store.create_mailbox("Archive").unwrap();

        // Insert message with flags
        let uid = imap_store.insert_message("INBOX", &[1u8; MESSAGE_ID_LEN], None, 1000, 100).unwrap();
        let mbox = imap_store.get_mailbox("INBOX").unwrap().unwrap();
        let msgs = imap_store.get_messages(mbox.id).unwrap();
        imap_store.add_flags(msgs[0].id, &["\\Seen", "\\Flagged"]).unwrap();

        // Copy to Archive
        let mapping = imap_store.copy_messages(mbox.id, &[uid], "Archive").unwrap();
        assert_eq!(mapping.len(), 1);
        let (src_uid, dst_uid) = mapping[0];
        assert_eq!(src_uid, uid);

        // Verify source still exists
        let src_msgs = imap_store.get_messages(mbox.id).unwrap();
        assert_eq!(src_msgs.len(), 1);

        // Verify destination has the message with flags
        let dst_mbox = imap_store.get_mailbox("Archive").unwrap().unwrap();
        let dst_msgs = imap_store.get_messages(dst_mbox.id).unwrap();
        assert_eq!(dst_msgs.len(), 1);
        assert_eq!(dst_msgs[0].uid, dst_uid);
        let dst_flags = imap_store.get_flags(dst_msgs[0].id).unwrap();
        assert!(dst_flags.contains(&"\\Flagged".to_string()));
        assert!(dst_flags.contains(&"\\Seen".to_string()));
    }

    #[test]
    fn move_removes_source() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("imap.db");
        let imap_store = ImapStore::open(&db_path).unwrap();
        imap_store.initialize_default_mailboxes().unwrap();
        imap_store.create_mailbox("Trash").unwrap();

        // Insert 2 messages
        let uid1 = imap_store.insert_message("INBOX", &[1u8; MESSAGE_ID_LEN], None, 1000, 100).unwrap();
        let uid2 = imap_store.insert_message("INBOX", &[2u8; MESSAGE_ID_LEN], None, 2000, 200).unwrap();
        let mbox = imap_store.get_mailbox("INBOX").unwrap().unwrap();

        // Move uid1 to Trash (copy + flag \Deleted + expunge)
        let mapping = imap_store.copy_messages(mbox.id, &[uid1], "Trash").unwrap();
        assert_eq!(mapping.len(), 1);

        // Flag source as deleted
        let msgs = imap_store.get_messages(mbox.id).unwrap();
        let msg1 = msgs.iter().find(|m| m.uid == uid1).unwrap();
        imap_store.add_flags(msg1.id, &["\\Deleted"]).unwrap();

        // Expunge
        let expunged = imap_store.expunge(mbox.id).unwrap();
        assert_eq!(expunged, vec![uid1]);

        // Verify source only has uid2
        let remaining = imap_store.get_messages(mbox.id).unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].uid, uid2);

        // Verify destination has the moved message
        let dst_mbox = imap_store.get_mailbox("Trash").unwrap().unwrap();
        let dst_msgs = imap_store.get_messages(dst_mbox.id).unwrap();
        assert_eq!(dst_msgs.len(), 1);
    }
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cargo test -p harmony-mail copy_messages_preserves -- --nocapture`
Run: `cargo test -p harmony-mail move_removes_source -- --nocapture`
Expected: both PASS (tests the store layer which is already implemented)

- [ ] **Step 3: Replace the COPY/MOVE stub**

In `crates/harmony-mail/src/server.rs`, replace the `ImapAction::CopyMessages { .. }` match arm with:

```rust
            ImapAction::CopyMessages {
                sequence_set,
                destination,
                uid_mode,
                is_move,
            } => {
                // Phase 1: sync — resolve mailbox, sequence set, do the copy
                enum CopyOutcome {
                    Done(Vec<(u32, u32)>, Vec<imap_store::MessageRow>, i64), // mapping, all_msgs, mailbox_id
                    Empty,
                    TryCreate,
                }

                let copy_result: Result<CopyOutcome, String> = (|| {
                    let mailbox_name = match &session.state {
                        ImapState::Selected { mailbox, .. } => mailbox.name.clone(),
                        _ => return Err("no mailbox selected".to_string()),
                    };
                    let mbox = store
                        .get_mailbox(&mailbox_name)
                        .map_err(|e| e.to_string())?
                        .ok_or_else(|| "mailbox not found".to_string())?;
                    let all_msgs = store.get_messages(mbox.id).map_err(|e| e.to_string())?;
                    let resolved = resolve_sequence_set(sequence_set, *uid_mode, &all_msgs);
                    let uids: Vec<u32> = resolved.iter().map(|(uid, _)| *uid).collect();

                    if uids.is_empty() {
                        return Ok(CopyOutcome::Empty);
                    }

                    match store.copy_messages(mbox.id, &uids, destination) {
                        Ok(mapping) => Ok(CopyOutcome::Done(mapping, all_msgs, mbox.id)),
                        Err(imap_store::StoreError::MailboxNotFound(_)) => Ok(CopyOutcome::TryCreate),
                        Err(e) => Err(e.to_string()),
                    }
                })();

                // Phase 2: async — write responses, handle MOVE expunge
                match copy_result {
                    Ok(CopyOutcome::Done(mapping, all_msgs, mailbox_id)) => {
                        if *is_move {
                            // Flag source messages as \Deleted
                            let rows_by_uid: std::collections::HashMap<u32, &imap_store::MessageRow> =
                                all_msgs.iter().map(|m| (m.uid, m)).collect();
                            for (src_uid, _dst_uid) in &mapping {
                                if let Some(row) = rows_by_uid.get(src_uid) {
                                    let _ = store.add_flags(row.id, &["\\Deleted"]);
                                }
                            }

                            let uid_to_seqnum: std::collections::HashMap<u32, u32> = all_msgs
                                .iter()
                                .enumerate()
                                .map(|(i, m)| (m.uid, (i + 1) as u32))
                                .collect();
                            let expunged_uids = store.expunge(mailbox_id).unwrap_or_default();
                            let mut seqnums: Vec<u32> = expunged_uids
                                .iter()
                                .filter_map(|uid| uid_to_seqnum.get(uid).copied())
                                .collect();
                            seqnums.sort_unstable_by(|a, b| b.cmp(a)); // descending

                            for seqnum in &seqnums {
                                writer
                                    .write_all(format!("* {seqnum} EXPUNGE\r\n").as_bytes())
                                    .await?;
                            }

                            if let Some(tag) = session.pending_tag.take() {
                                writer
                                    .write_all(format!("{tag} OK MOVE completed\r\n").as_bytes())
                                    .await?;
                            }
                        } else {
                            if let Some(tag) = session.pending_tag.take() {
                                writer
                                    .write_all(format!("{tag} OK COPY completed\r\n").as_bytes())
                                    .await?;
                            }
                        }
                        writer.flush().await?;
                    }
                    Ok(CopyOutcome::Empty) => {
                        let cmd = if *is_move { "MOVE" } else { "COPY" };
                        if let Some(tag) = session.pending_tag.take() {
                            writer
                                .write_all(format!("{tag} OK {cmd} completed\r\n").as_bytes())
                                .await?;
                        }
                        writer.flush().await?;
                    }
                    Ok(CopyOutcome::TryCreate) => {
                        if let Some(tag) = session.pending_tag.take() {
                            writer
                                .write_all(
                                    format!("{tag} NO [TRYCREATE] mailbox not found\r\n").as_bytes(),
                                )
                                .await?;
                        }
                        writer.flush().await?;
                    }
                    Err(reason) => {
                        if let Some(tag) = session.pending_tag.take() {
                            writer
                                .write_all(format!("{tag} NO {reason}\r\n").as_bytes())
                                .await?;
                            writer.flush().await?;
                        }
                    }
                }
            }
```

Note: This handler uses a two-phase pattern — a sync closure for store operations (Phase 1) returning a `CopyOutcome` enum, then async code for writing responses (Phase 2). This avoids unstable async closures and matches the FETCH handler's pattern.

- [ ] **Step 4: Verify compilation and all tests pass**

Run: `cargo check -p harmony-mail`
Run: `cargo test -p harmony-mail -- --nocapture`
Expected: compiles, all tests pass

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-mail/src/server.rs
git commit -m "feat(mail): wire IMAP COPY/MOVE to store operations (ZEB-111)"
```

---
