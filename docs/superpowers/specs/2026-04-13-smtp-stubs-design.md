# SMTP Integration Stubs: Local Delivery (ZEB-111 Item #5)

## Overview

Wire the three SMTP stub operations to real implementations, enabling end-to-end local delivery: SMTP in -> translate -> store in IMAP mailbox -> IMAP out. Remote (Zenoh network) delivery is tracked separately in ZEB-113.

## Scope

- **CheckSpf** — Wire to `mail-auth` SPF resolver, store result for spam scoring
- **ResolveHarmonyAddress** — Resolve addresses against local IMAP user store
- **DeliverToHarmony** — Translate RFC 5322, store in CAS, deliver to local IMAP mailbox with spam scoring

Non-local recipients are rejected with 550 (remote delivery requires ZEB-113).

## Design Decisions

- **SPF result stored in I/O layer, not state machine** — The SMTP state machine's `CheckSpf` action is fire-and-forget with no callback event. Rather than changing the sans-I/O state machine contract, the I/O layer stores the `SpfResult` in a per-connection variable and uses it when building `SpamSignals` during delivery.
- **`MessageAuthenticator` shared via `Arc`** — Created once at server startup, shared across all SMTP connections. The `mail-auth` authenticator uses system DNS config.
- **Spam scoring before delivery** — The spam check happens before inserting into the IMAP store. If the score exceeds `reject_threshold`, we return `DeliveryResult { success: false }` and the SMTP state machine sends 451.
- **CAS storage for translated messages** — The translated `HarmonyMessage` is serialized and stored in CAS (via `DiskBookStore`), producing a CID that's stored alongside the message in the IMAP store. This matches the existing FETCH/SEARCH pattern that retrieves messages from CAS by CID.

## CheckSpf

### Data Flow

```text
1. SMTP MAIL FROM triggers SmtpAction::CheckSpf { sender_domain, peer_ip }
2. process_async_actions calls authenticator.verify_spf() async
3. Maps result via auth::map_spf_result() -> SpfResult
4. Stores SpfResult in per-connection variable (passed by &mut reference)
5. Later used in DeliverToHarmony to build SpamSignals
```

### Error Handling

- DNS timeout/failure: log warning, default to `SpfResult::None` (don't block delivery for DNS issues)
- Null sender (MAIL FROM:<>): state machine already skips `CheckSpf` emission

## ResolveHarmonyAddress

### Data Flow

```text
1. SMTP RCPT TO triggers SmtpAction::ResolveHarmonyAddress { local_part, domain }
2. Check if domain matches gateway's configured domain
3. If non-local domain: identity = None (550 rejection)
4. If local domain: parse local_part via address::parse_local_part()
   a. Hex(hash) -> identity = Some(hash) directly
   b. Named { name, .. } -> store.get_user(&name) -> user.harmony_address
   c. Alias(alias) -> store.get_user(&alias) (same path for now)
5. Feed SmtpEvent::HarmonyResolved { local_part, identity } back to state machine
```

### Error Handling

- User not found: `identity = None` -> state machine sends 550
- Store error: log warning, `identity = None` -> 550

## DeliverToHarmony

### Data Flow

```text
1. SMTP DATA complete triggers SmtpAction::DeliverToHarmony { recipients, data }
2. Build SpamSignals with stored SpfResult, run spam::score()
3. If score >= reject_threshold: return DeliveryResult { success: false } -> 451
4. translate::translate_inbound(&data) -> TranslatedMessage { message, attachment_data }
5. Store attachment blobs in CAS via DiskBookStore
6. Serialize HarmonyMessage via to_bytes(), store in CAS -> get CID
7. For each recipient hash:
   a. store.get_user_by_address(&hash) -> find local user
   b. If found: store.insert_message("INBOX", &msg_id, Some(&cid), timestamp, rfc822_size)
8. Return DeliveryResult { success: delivered_count > 0 }
```

### Error Handling

- Translation failure: log error, return `DeliveryResult { success: false }` -> 451
- CAS storage failure: log error, still deliver (insert_message with `message_cid: None`)
- No local recipients found: return `DeliveryResult { success: false }` -> 451
- Partial delivery (some recipients found, some not): succeed for found ones

## Files Changed

### Modified

- **`crates/harmony-mail/src/server.rs`** — Wire all three stubs in `process_async_actions`. Add parameters: `authenticator: &MessageAuthenticator`, `imap_store: &Arc<ImapStore>`, `local_domain: &str`, `content_store_path: &Path`, `spf_result: &mut SpfResult`, `reject_threshold: i32`. Thread these through `handle_connection`, `handle_connection_generic`, and all SMTP listener spawn sites. The `run()` function already creates an `Arc<ImapStore>` for IMAP — the same Arc is cloned and passed to SMTP handlers so both protocols share the same user/message store.
- **`crates/harmony-mail/src/imap_store.rs`** — Add `get_user_by_address(hash: &[u8; ADDRESS_HASH_LEN]) -> Result<Option<UserRow>, StoreError>` method.

### No New Files

All changes in existing files.

## Testing

### get_user_by_address (unit test)

Create users with known addresses, verify lookup by address returns correct user, missing address returns None.

### ResolveHarmonyAddress (integration test)

Create a user in the IMAP store. Simulate RCPT TO with the user's address. Verify `HarmonyResolved` returns the correct identity hash. Verify non-existent user returns None (550).

### DeliverToHarmony (integration test)

Create a user. Run a full SMTP DATA transaction with RFC 5322 message bytes. Verify the message appears in the user's IMAP INBOX via `store.get_messages()`. Verify the message has a CID (CAS storage worked). Verify spam rejection by sending with `SpfResult::Fail` + other negative signals.

### CheckSpf (unit test)

Test the `map_spf_result` mapping (already tested). The actual `verify_spf` call requires live DNS and is not suitable for unit tests. The wiring path is exercised by the DeliverToHarmony integration test which checks that the stored SPF result flows into spam scoring.
