# IMAP SEARCH + COPY/MOVE Wiring (ZEB-111 Part 2)

## Overview

Wire IMAP SEARCH and COPY/MOVE to real operations. The protocol parsing, state machine actions/events, and store methods are all complete. This spec covers the I/O layer wiring that replaces the `NO [CANNOT]` stubs.

## Scope

Items #2 (remainder) from ZEB-111:

- **SEARCH**: Evaluate search criteria against messages (flags from SQLite, content from CAS), return matching UIDs/seqnums
- **COPY**: Copy messages between mailboxes via existing `ImapStore::copy_messages`
- **MOVE**: Copy + flag `\Deleted` + expunge source messages

## Design Decisions

- **`matches_criteria` in I/O layer** — A pure function evaluates `&[SearchKey]` against message metadata + optional HarmonyMessage. Multiple criteria are AND'd per RFC 9051. Easy to test, handles all 32 SearchKey variants uniformly.
- **CAS retrieval only when needed** — Same optimization as FETCH: check if any criterion requires content (From/To/Subject/Body/Header) before attempting CAS. If CAS fails or CID is missing, content criteria evaluate to `false`.
- **I/O layer writes COPY/MOVE responses directly** — Rather than going through `CopyComplete`/`ExpungeComplete` events, the handler writes responses using `pending_tag`. This avoids changing the state machine and keeps MOVE's interleaved EXPUNGE notifications self-contained.
- **MOVE = COPY + flag + expunge** — Per RFC 9051 §6.4.8, MOVE is semantically COPY + STORE +FLAGS.SILENT (\Deleted) + EXPUNGE. Uses existing `copy_messages`, `add_flags`, and `expunge` store methods.

## SEARCH Wiring

### Data Flow

```text
1. Extract selected mailbox from session.state
2. store.get_messages(mailbox_id)                    -> Vec<MessageRow>
3. Determine if any criterion needs CAS content
4. For each message:
   a. store.get_flags(msg_row.id)                    -> flags
   b. If CAS needed and message_cid is Some:
      spawn_blocking: DiskBookStore + dag::reassemble -> HarmonyMessage
   c. matches_criteria(&criteria, msg_row, &flags, seqnum, harmony_msg.as_ref()) -> bool
   d. If match: collect uid (uid_mode) or seqnum (not uid_mode)
5. session.handle(ImapEvent::SearchComplete { results })
6. Execute returned actions (writes "* SEARCH ..." + tagged OK)
```

### matches_criteria Function

```rust
fn matches_criteria(
    criteria: &[SearchKey],
    msg_row: &MessageRow,
    flags: &[String],
    seqnum: u32,
    msg: Option<&HarmonyMessage>,
) -> bool
```

Multiple criteria are AND'd. Each `SearchKey` variant maps to:

| Criterion | Source | Evaluation |
|-----------|--------|------------|
| `All` | — | `true` |
| `Seen` | flags | `flags.contains("\\Seen")` |
| `Unseen` | flags | `!flags.contains("\\Seen")` |
| `Flagged` | flags | `flags.contains("\\Flagged")` |
| `Unflagged` | flags | `!flags.contains("\\Flagged")` |
| `Answered` | flags | `flags.contains("\\Answered")` |
| `Unanswered` | flags | `!flags.contains("\\Answered")` |
| `Deleted` | flags | `flags.contains("\\Deleted")` |
| `Undeleted` | flags | `!flags.contains("\\Deleted")` |
| `Draft` | flags | `flags.contains("\\Draft")` |
| `Undraft` | flags | `!flags.contains("\\Draft")` |
| `Recent` | flags | `flags.contains("\\Recent")` |
| `New` | flags | `\\Recent` present AND `\\Seen` absent |
| `Old` | flags | `!flags.contains("\\Recent")` |
| `From(s)` | HarmonyMessage | case-insensitive: sender address contains `s` |
| `To(s)` | HarmonyMessage | case-insensitive: any recipient address contains `s` |
| `Subject(s)` | HarmonyMessage | case-insensitive: subject contains `s` |
| `Body(s)` | HarmonyMessage | case-insensitive: body contains `s` |
| `Header(name, value)` | HarmonyMessage | match `name` against known fields (From/To/Subject), case-insensitive value check |
| `Since(date)` | msg_row.internal_date | `internal_date >= parsed_date` |
| `Before(date)` | msg_row.internal_date | `internal_date < parsed_date` |
| `On(date)` | msg_row.internal_date | same calendar day |
| `Larger(n)` | msg_row.rfc822_size | `rfc822_size > n` |
| `Smaller(n)` | msg_row.rfc822_size | `rfc822_size < n` |
| `Uid(set)` | msg_row.uid | UID is in the resolved set |
| `SequenceSet(set)` | seqnum | seqnum is in the resolved set |
| `Not(key)` | recursive | `!matches_single(key, ...)` |
| `Or(a, b)` | recursive | `matches_single(a, ...) \|\| matches_single(b, ...)` |
| `Keyword(s)` | flags | `flags.contains(s)` |
| `Unkeyword(s)` | flags | `!flags.contains(s)` |

For content criteria (From/To/Subject/Body/Header): if `msg` is `None` (no CID or CAS failure), the criterion evaluates to `false`.

### Date Parsing

IMAP dates are strings like `"13-Apr-2026"`. A simple parser converts to Unix timestamp for comparison against `msg_row.internal_date`. The format is `DD-Mon-YYYY` per RFC 9051 §9 formal syntax.

## COPY/MOVE Wiring

### COPY Flow

```text
1. Extract selected mailbox from session.state
2. store.get_messages(mailbox_id)                     -> Vec<MessageRow>
3. resolve_sequence_set(set, uid_mode, &messages)     -> Vec<(uid, seqnum)>
4. Collect UIDs from resolved pairs
5. store.copy_messages(mailbox_id, &uids, &destination) -> Vec<(src_uid, dst_uid)>
6. pending_tag.take() -> write "{tag} OK COPY completed\r\n"
7. writer.flush()
```

### MOVE Flow

```text
1-4. Same as COPY
5. store.copy_messages(mailbox_id, &uids, &destination) -> mapping
6. For each source message: store.add_flags(msg_id, &["\\Deleted"])
7. Build uid_to_seqnum map from all_msgs (same pattern as Expunge)
8. store.expunge(mailbox_id)                          -> expunged UIDs
9. Map expunged UIDs to seqnums (descending order)
10. Write untagged "* {seqnum} EXPUNGE" for each
11. pending_tag.take() -> write "{tag} OK MOVE completed\r\n"
12. writer.flush()
```

### Error Handling

- Destination mailbox not found: `store.copy_messages` returns `StoreError::MailboxNotFound` → tagged `NO [TRYCREATE] mailbox not found`
- No mailbox selected: tagged `NO no mailbox selected`
- Store errors: tagged `NO {error}`

## Files Changed

### Modified

- `crates/harmony-mail/src/server.rs` — Replace SEARCH and COPY stubs. Add `matches_criteria` and `parse_imap_date` helper functions.

### No New Files

All changes are wiring in the existing server.rs file.

## Testing Strategy

### matches_criteria (TDD — pure function)

- **Flag criteria**: Seen/Unseen/Flagged/Answered/Deleted/Draft with messages that have and don't have the flag
- **Date criteria**: Since/Before/On with known timestamps
- **Size criteria**: Larger/Smaller with known rfc822_size
- **UID/SequenceSet**: Membership checks
- **Content criteria**: From/To/Subject/Body with HarmonyMessage substring matching
- **Composite**: Not(Seen), Or(Seen, Flagged), multi-criteria AND
- **Content unavailable**: Content criterion with `msg: None` returns false

### COPY

- Insert messages in INBOX with flags, copy to "Sent", verify both mailboxes have the messages with correct flags and UIDs

### MOVE

- Insert messages, move to another mailbox, verify source messages are removed and destination has them

### parse_imap_date

- Parse `"13-Apr-2026"` to correct Unix timestamp
- Handle all 12 month abbreviations
