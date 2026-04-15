//! Merkle mailbox manager — maintains per-user CAS mailbox trees.
//!
//! Each registered user gets a MailRoot with 4 folders (inbox, sent, drafts,
//! trash). On SMTP delivery, the inbox tree is updated and the new root CID
//! is published to Zenoh for harmony-client consumption.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::sync::{Notify, Semaphore};
use tokio_util::sync::CancellationToken;

/// Maximum in-flight raw-mail publish tasks at any given moment.
///
/// Bounded so a stalled Zenoh session cannot let per-recipient spawn tasks
/// accumulate without limit (each holds an Arc<Vec<u8>> payload). 256 is
/// larger than MAX_RECIPIENTS (100) so a single max-fan-out message never
/// pressure-stalls on itself, while leaving headroom for concurrent SMTP
/// deliveries. Excess spawns wait on the semaphore and are released when
/// a publish completes, times out, or is cancelled.
const RAW_PUBLISH_CONCURRENCY: usize = 256;

/// Hard ceiling for a single `session.put()` operation.
///
/// A healthy Zenoh put returns in milliseconds — anything beyond this
/// indicates a stalled link. Abort rather than pile up.
const RAW_PUBLISH_TIMEOUT: Duration = Duration::from_secs(10);

use crate::mailbox::{FolderKind, MailFolder, MailPage, MailRoot, MessageEntry, FOLDER_COUNT, MAX_SNIPPET_LEN};
use crate::message::{ADDRESS_HASH_LEN, CID_LEN, MESSAGE_ID_LEN};

/// Zenoh publisher for mailbox notifications.
///
/// Two independent publish paths share one Zenoh session:
///
/// 1. **Root CID updates** (`notify` → `harmony/mail/v1/{addr}/root`): coalesced
///    per-address. `MailboxManager` runs in sync context (`spawn_blocking`) and
///    calls `notify` whenever a user's mailbox root CID changes. A latest-only
///    map bounded by active user count is drained by a background task that
///    wakes on `Notify`. Stale roots never outrun fresh ones, and there is no
///    drop/queue-full path (so no log storm on stall).
///
/// 2. **Raw message bytes** (`publish_raw_mail` → `harmony/mail/v1/{addr}`):
///    fire-and-forget per-message. Each SMTP-delivered recipient gets one
///    tokio::spawn that issues a `session.put(bytes)`. No coalescing — each
///    message is distinct content and must be delivered individually.
///    Compatible with the Phase 0 harmony-client, which subscribes to
///    `harmony/mail/v1/{own_hex}` and decodes the payload via
///    `HarmonyMessage::from_bytes`.
///
/// Both paths skip new publishes once `cancel` fires, and the drain task
/// drops its session clone on cancel for clean disconnect.
pub struct ZenohPublisher {
    /// Coalescing map: pending root CIDs to publish. Drained by the
    /// background task on each wake. Keyed by `addr_hex` because the drain
    /// task formats `harmony/mail/v1/{addr_hex}/root` topics directly from
    /// these keys; converting to raw bytes would force a hex-encode inside
    /// the publish hot path for zero benefit.
    latest: Arc<Mutex<HashMap<String, [u8; CID_LEN]>>>,
    /// Current root CID per address (never drained). Populated by every
    /// `notify()` call and seeded from persisted roots on `set_publisher`.
    /// Read by the root queryable for cold-start sync so replies are
    /// deterministic regardless of drain-task timing.
    ///
    /// Keyed by the 32-byte address (matching `MailboxManager.roots`) so
    /// seeding is a direct copy with no per-entry hex allocation.
    ///
    /// `std::sync::Mutex` (not `tokio::sync::Mutex`) is correct here: the
    /// guard scope is a single `HashMap::get` + `Copy` of `[u8; 32]`, held
    /// for sub-microsecond, and never across an `.await`. Matches the
    /// existing `latest` mutex pattern in this file.
    current: Arc<Mutex<HashMap<[u8; ADDRESS_HASH_LEN], [u8; CID_LEN]>>>,
    wake: Arc<Notify>,
    raw_sink: RawSink,
}

/// Back-end for raw-mail publishes. Real builds hold a session + cancel
/// token; test builds capture publish attempts into a shared vec for
/// assertions without opening a Zenoh session.
enum RawSink {
    Session {
        session: zenoh::Session,
        cancel: CancellationToken,
        /// Caps concurrent in-flight `session.put()` tasks. Acquired inside
        /// the spawned task so the spawn itself is cheap — backpressure
        /// happens at the permit-acquire point, with cancellation awareness
        /// in case the semaphore is saturated during shutdown.
        permits: Arc<Semaphore>,
    },
    #[cfg(test)]
    Captured {
        raw: Arc<Mutex<Vec<(String, Arc<Vec<u8>>)>>>,
        sealed_unicast: Arc<Mutex<Vec<(String, Arc<Vec<u8>>)>>>,
    },
}

/// Test handles returned by `ZenohPublisher::inert_for_test` so callers can
/// assert on both coalesced root-CID updates and raw-mail publishes without
/// spawning a drain task or opening a Zenoh session.
#[cfg(test)]
pub struct InertHandles {
    pub latest: Arc<Mutex<HashMap<String, [u8; CID_LEN]>>>,
    pub current: Arc<Mutex<HashMap<[u8; ADDRESS_HASH_LEN], [u8; CID_LEN]>>>,
    pub raw: Arc<Mutex<Vec<(String, Arc<Vec<u8>>)>>>,
    pub sealed_unicast: Arc<Mutex<Vec<(String, Arc<Vec<u8>>)>>>,
}

impl ZenohPublisher {
    /// Create a new publisher backed by a Zenoh session.
    ///
    /// Spawns a background task that waits for wake-ups, drains the latest
    /// map, and publishes each user's most recent root CID to
    /// `harmony/mail/v1/{addr_hex}/root`.
    ///
    /// Topic namespace: `harmony/mail/v1/*` is the canonical mail prefix
    /// (Phase 0 client consumes `harmony/mail/v1/{addr}` for raw messages
    /// and reserves the `/root` suffix for root CID updates).
    ///
    /// The drain task observes `cancel`: on cancellation the loop exits and
    /// the Zenoh session is dropped explicitly, allowing a clean disconnect
    /// during graceful shutdown rather than holding the socket until the
    /// Tokio runtime is dropped.
    pub fn new(session: zenoh::Session, cancel: CancellationToken) -> Self {
        let latest: Arc<Mutex<HashMap<String, [u8; CID_LEN]>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let current: Arc<Mutex<HashMap<[u8; ADDRESS_HASH_LEN], [u8; CID_LEN]>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let wake = Arc::new(Notify::new());

        // zenoh::Session is Arc-internal, so cloning just bumps the refcount.
        // The drain task owns one clone (dropped on cancel for clean
        // disconnect); we keep another on the publisher for raw publishes.
        let drain_session = session.clone();

        let drain_latest = Arc::clone(&latest);
        let drain_wake = Arc::clone(&wake);
        let drain_cancel = cancel.clone();
        tokio::spawn(async move {
            let mut cancelled = false;
            loop {
                if !cancelled {
                    // Wait for a wake-up OR cancellation. If both are pending
                    // simultaneously, `select!` may pick either branch — so
                    // we always drain at least once more AFTER observing cancel
                    // (see the post-loop final-drain pass below) to ensure an
                    // update that raced the cancel signal is still published.
                    tokio::select! {
                        _ = drain_wake.notified() => {}
                        _ = drain_cancel.cancelled() => { cancelled = true; }
                    }
                }
                // Snapshot+clear under the sync lock so notify() can keep
                // inserting while we publish. Held briefly (O(active users));
                // no .await inside this scope.
                let snapshot: Vec<(String, [u8; CID_LEN])> = {
                    let mut map = drain_latest
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    map.drain().collect()
                };
                // Publish every entry in the snapshot, timeout-bounded, with
                // no inner cancel race. Two reasons:
                //
                // 1. Once a snapshot is drained from the coalescing map, the
                //    entries are consumed — if we abort mid-snapshot, those
                //    updates are lost until the next `insert_message` bumps
                //    the same addr. Committing to the full snapshot (each
                //    entry timeout-bounded) preserves the "drain at least
                //    once" invariant the coalescing contract depends on.
                // 2. `RAW_PUBLISH_TIMEOUT` (10s) already caps stall risk per
                //    entry. Total shutdown latency is bounded by
                //    (timeout × snapshot_len); cancel is observed at the
                //    outer `select!` between iterations, which is plenty
                //    responsive for a typical snapshot size (bounded by the
                //    number of active users whose mailboxes updated in the
                //    current wake cycle).
                //
                // Cancel fires during a put → the current put's timeout
                // still applies (worst case 10s to notice), then the outer
                // loop's select observes cancel, snapshots once more to
                // catch any late arrivals, drains, and breaks.
                for (addr_hex, root_cid) in snapshot {
                    let topic = format!("harmony/mail/v1/{addr_hex}/root");
                    match tokio::time::timeout(
                        RAW_PUBLISH_TIMEOUT,
                        drain_session.put(&topic, &root_cid[..]),
                    )
                    .await
                    {
                        Ok(Ok(())) => {}
                        Ok(Err(e)) => {
                            tracing::warn!(
                                error = %e,
                                %topic,
                                "Zenoh root CID publish failed"
                            );
                        }
                        Err(_elapsed) => {
                            tracing::warn!(
                                %topic,
                                timeout_ms = RAW_PUBLISH_TIMEOUT.as_millis() as u64,
                                "Zenoh root CID publish timed out — aborted"
                            );
                        }
                    }
                }
                if cancelled {
                    break;
                }
            }
            // Explicit drop ensures Zenoh disconnects promptly on shutdown
            // rather than waiting for the runtime tear-down.
            drop(drain_session);
            tracing::debug!("ZenohPublisher drain task exited on cancel");
        });

        // ── Queryable: respond to root-CID lookups (cold-start sync support) ──
        //
        // Clients query `harmony/mail/v1/{addr_hex}/root` to retrieve the current
        // root CID for an address. Same key as the publish topic — Zenoh routes
        // queries and puts independently. Reply payload is the raw 32 bytes, or
        // an empty reply if the address has no mail yet.
        let query_session = session.clone();
        let query_current = Arc::clone(&current);
        let query_cancel = cancel.clone();
        // Reply payload is the raw 32-byte CID for the address, OR an empty
        // payload if the address has no mail yet (sentinel for "no root yet").
        // Clients distinguish empty from absent reply by checking payload length.
        tokio::spawn(async move {
            let queryable = match query_session
                .declare_queryable("harmony/mail/v1/*/root")
                .await
            {
                Ok(q) => q,
                Err(e) => {
                    tracing::error!(error = %e, "failed to declare root queryable; cold-start sync unavailable");
                    return;
                }
            };
            loop {
                tokio::select! {
                    _ = query_cancel.cancelled() => break,
                    query = queryable.recv_async() => {
                        let Ok(query) = query else { break };
                        let key = query.key_expr().as_str().to_owned();
                        // Extract addr_hex between "harmony/mail/v1/" and "/root".
                        let Some(addr_hex) = key
                            .strip_prefix("harmony/mail/v1/")
                            .and_then(|s| s.strip_suffix("/root"))
                        else {
                            tracing::debug!(
                                %key,
                                "root queryable rejected malformed key (missing prefix/suffix)"
                            );
                            let _ = query.reply_err("invalid key").await;
                            continue;
                        };
                        // Decode the hex addr from the query key into raw
                        // bytes (the map key type). Malformed hex or wrong
                        // length → treat as missing and reply empty, same as
                        // an unknown address.
                        let payload: Option<[u8; CID_LEN]> = match hex::decode(addr_hex) {
                            Ok(bytes) if bytes.len() == ADDRESS_HASH_LEN => {
                                let mut addr = [0u8; ADDRESS_HASH_LEN];
                                addr.copy_from_slice(&bytes);
                                query_current
                                    .lock()
                                    .unwrap_or_else(|p| p.into_inner())
                                    .get(&addr)
                                    .copied()
                            }
                            _ => None,
                        };
                        let reply_result = match payload {
                            Some(cid) => query.reply(&key, &cid[..]).await,
                            None => query.reply(&key, &[][..]).await,
                        };
                        if let Err(e) = reply_result {
                            tracing::warn!(error = %e, %key, "failed to reply to root query");
                        }
                    }
                }
            }
            drop(query_session);
            tracing::debug!("ZenohPublisher root queryable task exited on cancel");
        });

        Self {
            latest,
            current,
            wake,
            raw_sink: RawSink::Session {
                session,
                cancel,
                permits: Arc::new(Semaphore::new(RAW_PUBLISH_CONCURRENCY)),
            },
        }
    }

    /// Test-only: create a publisher with no drain task, exposing the
    /// coalescing map and the raw-publish capture buffer for assertions.
    /// `notify()` still updates the map and `publish_raw_mail()` still
    /// appends to the raw buffer, so callers can observe what would have
    /// been published without opening a Zenoh session.
    #[cfg(test)]
    pub fn inert_for_test() -> (Self, InertHandles) {
        let latest: Arc<Mutex<HashMap<String, [u8; CID_LEN]>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let current: Arc<Mutex<HashMap<[u8; ADDRESS_HASH_LEN], [u8; CID_LEN]>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let raw: Arc<Mutex<Vec<(String, Arc<Vec<u8>>)>>> = Arc::new(Mutex::new(Vec::new()));
        let sealed_unicast: Arc<Mutex<Vec<(String, Arc<Vec<u8>>)>>> =
            Arc::new(Mutex::new(Vec::new()));
        let wake = Arc::new(Notify::new());
        let publisher = Self {
            latest: Arc::clone(&latest),
            current: Arc::clone(&current),
            wake,
            raw_sink: RawSink::Captured {
                raw: Arc::clone(&raw),
                sealed_unicast: Arc::clone(&sealed_unicast),
            },
        };
        (
            publisher,
            InertHandles {
                latest,
                current,
                raw,
                sealed_unicast,
            },
        )
    }

    /// Announce a new root CID for a user.
    ///
    /// Overwrites any previous pending CID for the same address and wakes the
    /// drain task. Callable from sync context.
    ///
    /// Takes the raw 32-byte address: this is the byte-keyed type used by
    /// `current` (matches `MailboxManager.roots`). A hex string is computed
    /// locally once per call for the drain-task's `latest` map, which needs
    /// it to build the `harmony/mail/v1/{addr_hex}/root` topic.
    pub fn notify(&self, address: [u8; ADDRESS_HASH_LEN], root_cid: [u8; CID_LEN]) {
        let addr_hex = hex::encode(address);
        let mut latest = self
            .latest
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut current = self
            .current
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        latest.insert(addr_hex, root_cid);
        current.insert(address, root_cid);
        drop(latest);
        drop(current);
        self.wake.notify_one();
    }

    /// Seed the queryable's `current` map without triggering a publish.
    ///
    /// Called from `MailboxManager::set_publisher` to pre-populate `current`
    /// from persisted roots loaded on startup. Without this, the root
    /// queryable would return empty for existing users until the next message
    /// triggered `notify()` — defeating cold-start sync after a server restart.
    ///
    /// Deliberately does NOT touch `latest`: seeding `latest` would trigger
    /// a publish storm of unchanged roots on startup.
    pub fn seed_current(&self, address: [u8; ADDRESS_HASH_LEN], root_cid: [u8; CID_LEN]) {
        let mut current = self
            .current
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        current.insert(address, root_cid);
    }

    /// Publish raw `HarmonyMessage` bytes to `harmony/mail/v1/{addr_hex}`.
    ///
    /// Fire-and-forget: spawns a task that issues one `session.put()` and
    /// logs on failure. Each call is a distinct message; the publisher does
    /// no coalescing or retry. Callers are expected to hand in the exact
    /// bytes produced by `HarmonyMessage::to_bytes()` so subscribers (e.g.
    /// Phase 0 harmony-client) can decode directly.
    ///
    /// **Resource bounds:**
    /// - Payload is `Arc<Vec<u8>>` so a single encoded message can fan out
    ///   to many recipients without per-recipient payload copies —
    ///   `Arc::clone` is a refcount bump, and the spawned task dereferences
    ///   to `&[u8]` for `session.put`.
    /// - Permits are acquired **synchronously at the call site** via
    ///   `try_acquire_owned`. If all `RAW_PUBLISH_CONCURRENCY` permits are
    ///   held, the publish is dropped (logged) rather than queued — this
    ///   bounds both concurrent `session.put` calls AND the number of
    ///   pending tokio tasks. Without call-site acquire, nothing would cap
    ///   how many task frames can pile up waiting for a permit.
    /// - Each publish has a `RAW_PUBLISH_TIMEOUT` hard cap and observes the
    ///   shared cancellation token, so stalled puts are aborted instead of
    ///   holding a permit forever.
    ///
    /// Raw publishes are a best-effort real-time notification. IMAP delivery
    /// and the Merkle root publish remain the durable paths — a dropped
    /// raw publish just means the client catches up on its next poll or
    /// root refresh.
    ///
    /// Short-circuits when the cancel token has already fired — prevents
    /// post-shutdown spawns from keeping the runtime alive.
    pub fn publish_raw_mail(&self, addr_hex: String, bytes: Arc<Vec<u8>>) {
        match &self.raw_sink {
            RawSink::Session {
                session,
                cancel,
                permits,
            } => {
                if cancel.is_cancelled() {
                    return;
                }
                // Acquire at the CALL SITE, not inside the spawn. Dropping
                // on backpressure prevents unbounded task accumulation
                // during a Zenoh stall — if we spawned first and acquired
                // second, every waiting spawn would hold its Arc<Vec<u8>>
                // refcount and task frame open for minutes.
                let permit = match Arc::clone(permits).try_acquire_owned() {
                    Ok(p) => p,
                    Err(tokio::sync::TryAcquireError::NoPermits) => {
                        tracing::warn!(
                            addr = %addr_hex,
                            "raw-mail publish dropped: in-flight limit ({}) reached — best-effort path, client will catch up via IMAP/root",
                            RAW_PUBLISH_CONCURRENCY,
                        );
                        return;
                    }
                    Err(tokio::sync::TryAcquireError::Closed) => return,
                };
                let session = session.clone();
                let cancel = cancel.clone();
                let topic = format!("harmony/mail/v1/{addr_hex}");
                tokio::spawn(async move {
                    // Move the already-acquired permit into the task so it
                    // releases on completion/timeout/cancel. Timeout-bounded,
                    // cancel-aware put.
                    let _permit = permit;
                    tokio::select! {
                        res = tokio::time::timeout(
                            RAW_PUBLISH_TIMEOUT,
                            session.put(&topic, &bytes[..]),
                        ) => match res {
                            Ok(Ok(())) => {}
                            Ok(Err(e)) => {
                                tracing::warn!(
                                    error = %e,
                                    %topic,
                                    "Zenoh raw mail publish failed"
                                );
                            }
                            Err(_elapsed) => {
                                tracing::warn!(
                                    %topic,
                                    timeout_ms = RAW_PUBLISH_TIMEOUT.as_millis() as u64,
                                    "Zenoh raw mail publish timed out — aborted"
                                );
                            }
                        },
                        _ = cancel.cancelled() => {
                            tracing::debug!(
                                %topic,
                                "Zenoh raw mail publish cancelled during shutdown"
                            );
                        }
                    }
                });
            }
            #[cfg(test)]
            RawSink::Captured { raw, .. } => {
                raw.lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .push((addr_hex, bytes));
            }
        }
    }

    /// Publish a sealed unicast envelope to
    /// `harmony/msg/v1/unicast/{recipient_hash_hex}`.
    ///
    /// Fire-and-forget: spawns a task that issues one `session.put()` and
    /// logs on failure. Mirrors `publish_raw_mail` in structure — shares the
    /// same permit pool and timeout/cancellation machinery — but targets the
    /// `msg::unicast` keyspace used by sender-to-recipient sealed delivery
    /// (ZEB-113). Callers hand in the already-encoded sealed envelope bytes
    /// so recipients can decode without further transformation.
    ///
    /// Sharing `RAW_PUBLISH_CONCURRENCY` / `RAW_PUBLISH_TIMEOUT` with
    /// `publish_raw_mail` is intentional: a single SMTP delivery fans out
    /// through one or the other, not both, so a shared budget bounds total
    /// outbound puts without over-provisioning two independent pools.
    ///
    /// Short-circuits when the cancel token has already fired — prevents
    /// post-shutdown spawns from keeping the runtime alive.
    pub fn publish_sealed_unicast(
        &self,
        recipient_hash_hex: String,
        envelope: Arc<Vec<u8>>,
    ) {
        match &self.raw_sink {
            RawSink::Session {
                session,
                cancel,
                permits,
            } => {
                if cancel.is_cancelled() {
                    return;
                }
                let permit = match Arc::clone(permits).try_acquire_owned() {
                    Ok(p) => p,
                    Err(tokio::sync::TryAcquireError::NoPermits) => {
                        tracing::warn!(
                            recipient = %recipient_hash_hex,
                            "sealed-unicast publish dropped: in-flight limit ({}) reached — best-effort path",
                            RAW_PUBLISH_CONCURRENCY,
                        );
                        return;
                    }
                    Err(tokio::sync::TryAcquireError::Closed) => return,
                };
                let session = session.clone();
                let cancel = cancel.clone();
                let topic = harmony_zenoh::namespace::msg::unicast_key(&recipient_hash_hex);
                tokio::spawn(async move {
                    let _permit = permit;
                    tokio::select! {
                        res = tokio::time::timeout(
                            RAW_PUBLISH_TIMEOUT,
                            session.put(&topic, &envelope[..]),
                        ) => match res {
                            Ok(Ok(())) => {}
                            Ok(Err(e)) => {
                                tracing::warn!(
                                    error = %e,
                                    %topic,
                                    "Zenoh sealed-unicast publish failed"
                                );
                            }
                            Err(_elapsed) => {
                                tracing::warn!(
                                    %topic,
                                    timeout_ms = RAW_PUBLISH_TIMEOUT.as_millis() as u64,
                                    "Zenoh sealed-unicast publish timed out — aborted"
                                );
                            }
                        },
                        _ = cancel.cancelled() => {
                            tracing::debug!(
                                %topic,
                                "Zenoh sealed-unicast publish cancelled during shutdown"
                            );
                        }
                    }
                });
            }
            #[cfg(test)]
            RawSink::Captured { sealed_unicast, .. } => {
                let topic = harmony_zenoh::namespace::msg::unicast_key(&recipient_hash_hex);
                sealed_unicast
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .push((topic, envelope));
            }
        }
    }
}

/// Errors from mailbox manager operations.
#[derive(Debug, thiserror::Error)]
pub enum ManagerError {
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("CAS error: {0}")]
    Cas(String),
    #[error("mailbox format error: {0}")]
    Format(#[from] crate::error::MailError),
    #[error("no mailbox for user {}", hex::encode(.0))]
    NoMailbox([u8; ADDRESS_HASH_LEN]),
}

impl From<harmony_mailbox::MailboxError> for ManagerError {
    fn from(e: harmony_mailbox::MailboxError) -> Self {
        ManagerError::Format(crate::error::MailError::Mailbox(e))
    }
}

pub struct MailboxManager {
    /// Per-user current root CIDs.
    roots: HashMap<[u8; ADDRESS_HASH_LEN], [u8; CID_LEN]>,
    /// CAS storage path for DiskBookStore access.
    content_store_path: PathBuf,
    /// Persistence for root CID pointers.
    db: rusqlite::Connection,
    /// Optional Zenoh publisher for root CID notifications.
    publisher: Option<Arc<ZenohPublisher>>,
}

const SCHEMA_SQL: &str = "
CREATE TABLE IF NOT EXISTS mailbox_roots (
    address BLOB NOT NULL UNIQUE,
    root_cid BLOB NOT NULL
);
";

impl MailboxManager {
    /// Open or create the mailbox roots database.
    pub fn open(db_path: &Path, content_store_path: &Path) -> Result<Self, ManagerError> {
        let db = rusqlite::Connection::open(db_path)?;
        db.execute_batch(SCHEMA_SQL)?;

        // Load existing roots into memory.
        // The statement borrow must be dropped before `db` is moved into Self,
        // so we collect all rows eagerly inside a nested block.
        let mut roots = HashMap::new();
        {
            let mut stmt = db.prepare("SELECT address, root_cid FROM mailbox_roots")?;
            let pairs: Vec<(Vec<u8>, Vec<u8>)> = stmt
                .query_map([], |row| {
                    let addr_blob: Vec<u8> = row.get(0)?;
                    let cid_blob: Vec<u8> = row.get(1)?;
                    Ok((addr_blob, cid_blob))
                })?
                .collect::<Result<_, _>>()?;
            for (addr_blob, cid_blob) in pairs {
                if addr_blob.len() != ADDRESS_HASH_LEN || cid_blob.len() != CID_LEN {
                    return Err(ManagerError::Cas(format!(
                        "corrupt mailbox_roots row: address_len={}, root_cid_len={}",
                        addr_blob.len(),
                        cid_blob.len()
                    )));
                }
                let mut addr = [0u8; ADDRESS_HASH_LEN];
                addr.copy_from_slice(&addr_blob);
                let mut cid = [0u8; CID_LEN];
                cid.copy_from_slice(&cid_blob);
                roots.insert(addr, cid);
            }
        }

        Ok(Self {
            roots,
            content_store_path: content_store_path.to_path_buf(),
            db,
            publisher: None,
        })
    }

    /// Attach a Zenoh publisher for root CID notifications.
    ///
    /// Takes `Arc<ZenohPublisher>` so the same publisher can be shared with
    /// the SMTP server's raw-publish path without routing raw publishes
    /// through this manager's mutex (raw publish doesn't need the manager
    /// state, only the publisher).
    ///
    /// Call this after opening the manager if Zenoh is configured and enabled.
    pub fn set_publisher(&mut self, publisher: Arc<ZenohPublisher>) {
        // Seed publisher's `current` map from persisted roots so cold-start
        // queries return the correct CIDs immediately after attach (without
        // requiring a new insert_message to populate `current` per-address).
        // We deliberately do NOT seed `latest` — that would trigger a publish
        // storm of unchanged roots on startup.
        for (address, root_cid) in &self.roots {
            publisher.seed_current(*address, *root_cid);
        }
        self.publisher = Some(publisher);
    }

    /// Get the current root CID for a user, if one exists.
    pub fn get_root(&self, address: &[u8; ADDRESS_HASH_LEN]) -> Option<&[u8; CID_LEN]> {
        self.roots.get(address)
    }

    /// Number of users with initialized Merkle mailboxes.
    pub fn user_count(&self) -> usize {
        self.roots.len()
    }

    /// Ingest raw bytes into CAS and return the resulting CID as a 32-byte array.
    fn cas_ingest(&self, data: &[u8]) -> Result<[u8; CID_LEN], ManagerError> {
        let mut book = harmony_db::DiskBookStore::new(&self.content_store_path);
        let cid = harmony_content::dag::ingest(
            data,
            &harmony_content::chunker::ChunkerConfig::DEFAULT,
            &mut book,
        )
        .map_err(|e| ManagerError::Cas(e.to_string()))?;
        Ok(cid.to_bytes())
    }

    /// Load raw bytes from CAS by CID, reassembling chunked DAG entries.
    fn cas_load(&self, cid: &[u8; CID_LEN]) -> Result<Vec<u8>, ManagerError> {
        let book = harmony_db::DiskBookStore::new(&self.content_store_path);
        let content_id = harmony_content::cid::ContentId::from_bytes(*cid);
        harmony_content::dag::reassemble(&content_id, &book)
            .map_err(|e| ManagerError::Cas(e.to_string()))
    }

    /// Ensure a user has a Merkle mailbox tree. If one already exists, this is a no-op.
    /// If not, creates an empty MailRoot with 4 empty folders, ingests everything
    /// into CAS, and persists the root CID.
    pub fn ensure_user_mailbox(
        &mut self,
        address: &[u8; ADDRESS_HASH_LEN],
    ) -> Result<(), ManagerError> {
        if self.roots.contains_key(address) {
            return Ok(());
        }

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Create an empty page (shared by all empty folders)
        let empty_page = MailPage::new_empty();
        let page_bytes = empty_page.to_bytes()?;
        let page_cid = self.cas_ingest(&page_bytes)?;

        // Create 4 empty folders, each pointing to the empty page
        let mut folder_cids = [[0u8; CID_LEN]; FOLDER_COUNT];
        for folder_cid in &mut folder_cids {
            let folder = MailFolder {
                version: crate::mailbox::MAILBOX_VERSION,
                message_count: 0,
                unread_count: 0,
                page_cids: vec![page_cid],
            };
            let folder_bytes = folder.to_bytes()?;
            *folder_cid = self.cas_ingest(&folder_bytes)?;
        }

        // Create the root
        let root = MailRoot {
            version: crate::mailbox::MAILBOX_VERSION,
            owner_address: *address,
            updated_at: now,
            folders: folder_cids,
        };
        let root_bytes = root.to_bytes();
        let root_cid = self.cas_ingest(&root_bytes)?;

        self.persist_root(address, &root_cid)?;

        // Mirror the new root into the queryable's `current` map so cold-start
        // queries succeed immediately after creation (without waiting for the
        // first `notify()` or a restart-time `seed_current` backfill).
        // Deliberately does NOT touch `latest` — that would cause a publish of
        // an unchanged root.
        if let Some(ref publisher) = self.publisher {
            publisher.seed_current(*address, root_cid);
        }

        Ok(())
    }

    /// Insert a message into a user's inbox Merkle tree.
    ///
    /// Loads the current tree from CAS, prepends the entry to the head page
    /// (splitting if full), writes all changed blobs to CAS, and persists the
    /// new root CID.
    pub fn insert_message(
        &mut self,
        user_address: &[u8; ADDRESS_HASH_LEN],
        message_cid: &[u8; CID_LEN],
        msg_id: &[u8; MESSAGE_ID_LEN],
        sender_address: &[u8; ADDRESS_HASH_LEN],
        timestamp: u64,
        subject: &str,
    ) -> Result<(), ManagerError> {
        // Auto-initialize if this user doesn't have a Merkle tree yet
        // (e.g., user created after startup, or roots DB was reset).
        if !self.roots.contains_key(user_address) {
            self.ensure_user_mailbox(user_address)?;
        }
        let root_cid = *self
            .roots
            .get(user_address)
            .ok_or(ManagerError::NoMailbox(*user_address))?;

        // Load current root
        let root_bytes = self.cas_load(&root_cid)?;
        let root = MailRoot::from_bytes(&root_bytes)?;

        // Load inbox folder
        let inbox_cid = *root.folder_cid(FolderKind::Inbox);
        let folder_bytes = self.cas_load(&inbox_cid)?;
        let mut folder = MailFolder::from_bytes(&folder_bytes)?;

        // Build the new entry
        let snippet = crate::mailbox::truncate_utf8(subject, MAX_SNIPPET_LEN).to_string();
        let entry = MessageEntry {
            message_cid: *message_cid,
            message_id: *msg_id,
            sender_address: *sender_address,
            timestamp,
            subject_snippet: snippet,
            read: false,
        };

        // Load head page (or create one if folder has no pages)
        if folder.page_cids.is_empty() {
            // No pages yet — create one with just this entry
            let page = MailPage {
                version: crate::mailbox::MAILBOX_VERSION,
                next_page: None,
                entries: vec![entry],
            };
            let page_cid = self.cas_ingest(&page.to_bytes()?)?;
            folder.page_cids.push(page_cid);
        } else {
            let head_cid = folder.page_cids[0];
            let head_bytes = self.cas_load(&head_cid)?;
            let mut head_page = MailPage::from_bytes(&head_bytes)?;

            if head_page.is_full() {
                // Page is full — create a new head page with just this entry.
                // The old head page becomes the second page (unchanged in CAS).
                let new_page = MailPage {
                    version: crate::mailbox::MAILBOX_VERSION,
                    next_page: Some(head_cid),
                    entries: vec![entry],
                };
                let new_page_cid = self.cas_ingest(&new_page.to_bytes()?)?;
                folder.page_cids.insert(0, new_page_cid);
            } else {
                // Prepend to existing head page
                head_page.entries.insert(0, entry);
                // Update next_page pointer if there's a following page
                head_page.next_page = folder.page_cids.get(1).copied();
                let new_head_cid = self.cas_ingest(&head_page.to_bytes()?)?;
                folder.page_cids[0] = new_head_cid;
            }
        }

        // Update folder counts
        folder.message_count += 1;
        folder.unread_count += 1;

        // Write updated folder to CAS
        let new_folder_cid = self.cas_ingest(&folder.to_bytes()?)?;

        // Update root with new inbox CID
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let new_root = root.with_folder(FolderKind::Inbox, new_folder_cid, now);
        let new_root_cid = self.cas_ingest(&new_root.to_bytes())?;

        // Persist
        self.persist_root(user_address, &new_root_cid)?;

        // Notify Zenoh publisher (non-critical path — errors are logged and swallowed).
        if let Some(ref publisher) = self.publisher {
            publisher.notify(*user_address, new_root_cid);
        }

        Ok(())
    }

    /// Persist a root CID for a user (inserts or updates).
    fn persist_root(
        &mut self,
        address: &[u8; ADDRESS_HASH_LEN],
        root_cid: &[u8; CID_LEN],
    ) -> Result<(), ManagerError> {
        self.db.execute(
            "INSERT INTO mailbox_roots (address, root_cid) VALUES (?1, ?2)
             ON CONFLICT(address) DO UPDATE SET root_cid = excluded.root_cid",
            rusqlite::params![address.as_slice(), root_cid.as_slice()],
        )?;
        self.roots.insert(*address, *root_cid);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use harmony_content::book::BookStore as _;
    use crate::mailbox::{FolderKind, PAGE_CAPACITY};
    use crate::message::MESSAGE_ID_LEN;

    fn dummy_msg_cid(tag: u8) -> [u8; CID_LEN] {
        let mut cid = [0u8; CID_LEN];
        cid[0] = tag;
        cid
    }

    fn dummy_msg_id(tag: u8) -> [u8; MESSAGE_ID_LEN] {
        [tag; MESSAGE_ID_LEN]
    }

    fn dummy_sender() -> [u8; ADDRESS_HASH_LEN] {
        [0xEEu8; ADDRESS_HASH_LEN]
    }

    #[test]
    fn open_creates_schema() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("roots.db");
        let cas_path = dir.path().join("content");
        std::fs::create_dir_all(cas_path.join("commits")).unwrap();
        std::fs::create_dir_all(cas_path.join("blobs")).unwrap();

        let mgr = MailboxManager::open(&db_path, &cas_path).unwrap();
        assert!(mgr.roots.is_empty());
    }

    #[test]
    fn persist_and_reload_root() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("roots.db");
        let cas_path = dir.path().join("content");
        std::fs::create_dir_all(cas_path.join("commits")).unwrap();
        std::fs::create_dir_all(cas_path.join("blobs")).unwrap();

        let addr = [0xAAu8; ADDRESS_HASH_LEN];
        let cid = [0xBBu8; CID_LEN];

        // Persist
        {
            let mut mgr = MailboxManager::open(&db_path, &cas_path).unwrap();
            mgr.persist_root(&addr, &cid).unwrap();
            assert_eq!(mgr.get_root(&addr), Some(&cid));
        }

        // Reload from disk
        {
            let mgr = MailboxManager::open(&db_path, &cas_path).unwrap();
            assert_eq!(mgr.get_root(&addr), Some(&cid));
        }
    }

    #[test]
    fn persist_root_upserts() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("roots.db");
        let cas_path = dir.path().join("content");
        std::fs::create_dir_all(cas_path.join("commits")).unwrap();
        std::fs::create_dir_all(cas_path.join("blobs")).unwrap();

        let addr = [0xAAu8; ADDRESS_HASH_LEN];
        let cid1 = [0x11u8; CID_LEN];
        let cid2 = [0x22u8; CID_LEN];

        let mut mgr = MailboxManager::open(&db_path, &cas_path).unwrap();
        mgr.persist_root(&addr, &cid1).unwrap();
        assert_eq!(mgr.get_root(&addr), Some(&cid1));

        mgr.persist_root(&addr, &cid2).unwrap();
        assert_eq!(mgr.get_root(&addr), Some(&cid2));
    }

    #[test]
    fn ensure_user_mailbox_creates_tree() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("roots.db");
        let cas_path = dir.path().join("content");
        std::fs::create_dir_all(cas_path.join("commits")).unwrap();
        std::fs::create_dir_all(cas_path.join("blobs")).unwrap();

        let addr = [0xAAu8; ADDRESS_HASH_LEN];
        let mut mgr = MailboxManager::open(&db_path, &cas_path).unwrap();

        // No root yet
        assert!(mgr.get_root(&addr).is_none());

        // Create the empty tree
        mgr.ensure_user_mailbox(&addr).unwrap();

        // Root CID now exists
        let root_cid = mgr.get_root(&addr).expect("root should exist");
        assert_ne!(root_cid, &[0u8; CID_LEN], "root CID should not be empty");

        // Verify the root can be loaded from CAS
        let book = harmony_db::DiskBookStore::new(&cas_path);
        let root_content_id = harmony_content::cid::ContentId::from_bytes(*root_cid);
        let root_bytes = book.get(&root_content_id).expect("root should be in CAS");
        let root = MailRoot::from_bytes(root_bytes).unwrap();
        assert_eq!(root.owner_address, addr);

        // Verify inbox folder exists in CAS
        let inbox_cid = root.folder_cid(FolderKind::Inbox);
        assert_ne!(inbox_cid, &[0u8; CID_LEN]);
        let folder_content_id = harmony_content::cid::ContentId::from_bytes(*inbox_cid);
        let folder_bytes = book.get(&folder_content_id).expect("inbox folder should be in CAS");
        let folder = MailFolder::from_bytes(folder_bytes).unwrap();
        assert_eq!(folder.message_count, 0);
        assert_eq!(folder.unread_count, 0);
    }

    #[test]
    fn ensure_user_mailbox_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("roots.db");
        let cas_path = dir.path().join("content");
        std::fs::create_dir_all(cas_path.join("commits")).unwrap();
        std::fs::create_dir_all(cas_path.join("blobs")).unwrap();

        let addr = [0xAAu8; ADDRESS_HASH_LEN];
        let mut mgr = MailboxManager::open(&db_path, &cas_path).unwrap();

        mgr.ensure_user_mailbox(&addr).unwrap();
        let cid1 = *mgr.get_root(&addr).unwrap();

        // Second call should not change the root CID
        mgr.ensure_user_mailbox(&addr).unwrap();
        let cid2 = *mgr.get_root(&addr).unwrap();
        assert_eq!(cid1, cid2);
    }

    #[test]
    fn insert_into_empty_mailbox() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("roots.db");
        let cas_path = dir.path().join("content");
        std::fs::create_dir_all(cas_path.join("commits")).unwrap();
        std::fs::create_dir_all(cas_path.join("blobs")).unwrap();

        let addr = [0xAAu8; ADDRESS_HASH_LEN];
        let mut mgr = MailboxManager::open(&db_path, &cas_path).unwrap();
        mgr.ensure_user_mailbox(&addr).unwrap();
        let initial_cid = *mgr.get_root(&addr).unwrap();

        mgr.insert_message(
            &addr,
            &dummy_msg_cid(1),
            &dummy_msg_id(1),
            &dummy_sender(),
            1700000000,
            "Hello World",
        )
        .unwrap();

        // Root CID should have changed
        let new_cid = *mgr.get_root(&addr).unwrap();
        assert_ne!(initial_cid, new_cid);

        // Load and verify the tree
        let book = harmony_db::DiskBookStore::new(&cas_path);
        let root = MailRoot::from_bytes(
            book.get(&harmony_content::cid::ContentId::from_bytes(new_cid)).unwrap(),
        )
        .unwrap();

        let inbox_cid = root.folder_cid(FolderKind::Inbox);
        let folder = MailFolder::from_bytes(
            book.get(&harmony_content::cid::ContentId::from_bytes(*inbox_cid)).unwrap(),
        )
        .unwrap();
        assert_eq!(folder.message_count, 1);
        assert_eq!(folder.unread_count, 1);

        let page = MailPage::from_bytes(
            book.get(&harmony_content::cid::ContentId::from_bytes(folder.page_cids[0])).unwrap(),
        )
        .unwrap();
        assert_eq!(page.entries.len(), 1);
        assert_eq!(page.entries[0].message_cid, dummy_msg_cid(1));
        assert_eq!(page.entries[0].subject_snippet, "Hello World");
        assert!(!page.entries[0].read);
    }

    #[test]
    fn insert_splits_page_at_capacity() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("roots.db");
        let cas_path = dir.path().join("content");
        std::fs::create_dir_all(cas_path.join("commits")).unwrap();
        std::fs::create_dir_all(cas_path.join("blobs")).unwrap();

        let addr = [0xAAu8; ADDRESS_HASH_LEN];
        let mut mgr = MailboxManager::open(&db_path, &cas_path).unwrap();
        mgr.ensure_user_mailbox(&addr).unwrap();

        // Fill first page to capacity
        for i in 0..PAGE_CAPACITY {
            mgr.insert_message(
                &addr,
                &dummy_msg_cid(i as u8),
                &dummy_msg_id(i as u8),
                &dummy_sender(),
                1700000000 + i as u64,
                &format!("msg {i}"),
            )
            .unwrap();
        }

        // Verify: 1 page with PAGE_CAPACITY entries
        let book = harmony_db::DiskBookStore::new(&cas_path);
        let root_cid = *mgr.get_root(&addr).unwrap();
        let root = MailRoot::from_bytes(
            book.get(&harmony_content::cid::ContentId::from_bytes(root_cid)).unwrap(),
        )
        .unwrap();
        let folder = MailFolder::from_bytes(
            book.get(&harmony_content::cid::ContentId::from_bytes(*root.folder_cid(FolderKind::Inbox))).unwrap(),
        )
        .unwrap();
        assert_eq!(folder.message_count as usize, PAGE_CAPACITY);
        // The initial empty page + PAGE_CAPACITY replacements = still 1 page CID
        assert_eq!(folder.page_cids.len(), 1);

        // Insert one more — should trigger page split
        mgr.insert_message(
            &addr,
            &dummy_msg_cid(0xFF),
            &dummy_msg_id(0xFF),
            &dummy_sender(),
            1700000000 + PAGE_CAPACITY as u64,
            "overflow",
        )
        .unwrap();

        // Verify: 2 pages now
        let root_cid = *mgr.get_root(&addr).unwrap();
        let root = MailRoot::from_bytes(
            book.get(&harmony_content::cid::ContentId::from_bytes(root_cid)).unwrap(),
        )
        .unwrap();
        let folder = MailFolder::from_bytes(
            book.get(&harmony_content::cid::ContentId::from_bytes(*root.folder_cid(FolderKind::Inbox))).unwrap(),
        )
        .unwrap();
        assert_eq!(folder.message_count as usize, PAGE_CAPACITY + 1);
        assert_eq!(folder.page_cids.len(), 2);

        // Head page has 1 entry (the newest)
        let head_page = MailPage::from_bytes(
            book.get(&harmony_content::cid::ContentId::from_bytes(folder.page_cids[0])).unwrap(),
        )
        .unwrap();
        assert_eq!(head_page.entries.len(), 1);
        assert_eq!(head_page.entries[0].message_cid, dummy_msg_cid(0xFF));
    }

    #[test]
    fn per_user_isolation() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("roots.db");
        let cas_path = dir.path().join("content");
        std::fs::create_dir_all(cas_path.join("commits")).unwrap();
        std::fs::create_dir_all(cas_path.join("blobs")).unwrap();

        let alice = [0xAAu8; ADDRESS_HASH_LEN];
        let bob = [0xBBu8; ADDRESS_HASH_LEN];
        let mut mgr = MailboxManager::open(&db_path, &cas_path).unwrap();
        mgr.ensure_user_mailbox(&alice).unwrap();
        mgr.ensure_user_mailbox(&bob).unwrap();

        // Send 3 messages to alice, 1 to bob
        for i in 0..3u8 {
            mgr.insert_message(
                &alice,
                &dummy_msg_cid(i),
                &dummy_msg_id(i),
                &dummy_sender(),
                1700000000 + i as u64,
                &format!("alice msg {i}"),
            )
            .unwrap();
        }
        mgr.insert_message(
            &bob,
            &dummy_msg_cid(10),
            &dummy_msg_id(10),
            &dummy_sender(),
            1700000000,
            "bob msg",
        )
        .unwrap();

        // Verify alice has 3, bob has 1
        let book = harmony_db::DiskBookStore::new(&cas_path);

        let alice_root = MailRoot::from_bytes(
            book.get(&harmony_content::cid::ContentId::from_bytes(*mgr.get_root(&alice).unwrap())).unwrap(),
        )
        .unwrap();
        let alice_folder = MailFolder::from_bytes(
            book.get(&harmony_content::cid::ContentId::from_bytes(*alice_root.folder_cid(FolderKind::Inbox))).unwrap(),
        )
        .unwrap();
        assert_eq!(alice_folder.message_count, 3);

        let bob_root = MailRoot::from_bytes(
            book.get(&harmony_content::cid::ContentId::from_bytes(*mgr.get_root(&bob).unwrap())).unwrap(),
        )
        .unwrap();
        let bob_folder = MailFolder::from_bytes(
            book.get(&harmony_content::cid::ContentId::from_bytes(*bob_root.folder_cid(FolderKind::Inbox))).unwrap(),
        )
        .unwrap();
        assert_eq!(bob_folder.message_count, 1);
    }

    #[test]
    fn root_cid_survives_restart() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("roots.db");
        let cas_path = dir.path().join("content");
        std::fs::create_dir_all(cas_path.join("commits")).unwrap();
        std::fs::create_dir_all(cas_path.join("blobs")).unwrap();

        let addr = [0xAAu8; ADDRESS_HASH_LEN];

        // First session: create tree and insert messages
        {
            let mut mgr = MailboxManager::open(&db_path, &cas_path).unwrap();
            mgr.ensure_user_mailbox(&addr).unwrap();
            for i in 0..3u8 {
                mgr.insert_message(
                    &addr,
                    &dummy_msg_cid(i),
                    &dummy_msg_id(i),
                    &dummy_sender(),
                    1700000000 + i as u64,
                    &format!("msg {i}"),
                )
                .unwrap();
            }
        }

        // Second session: reopen and verify
        {
            let mgr = MailboxManager::open(&db_path, &cas_path).unwrap();
            let root_cid = mgr.get_root(&addr).expect("root should persist");

            let book = harmony_db::DiskBookStore::new(&cas_path);
            let root = MailRoot::from_bytes(
                book.get(&harmony_content::cid::ContentId::from_bytes(*root_cid)).unwrap(),
            )
            .unwrap();
            let folder = MailFolder::from_bytes(
                book.get(&harmony_content::cid::ContentId::from_bytes(*root.folder_cid(FolderKind::Inbox))).unwrap(),
            )
            .unwrap();
            assert_eq!(folder.message_count, 3);

            let page = MailPage::from_bytes(
                book.get(&harmony_content::cid::ContentId::from_bytes(folder.page_cids[0])).unwrap(),
            )
            .unwrap();
            assert_eq!(page.entries.len(), 3);
            // Newest first
            assert_eq!(page.entries[0].message_cid, dummy_msg_cid(2));
            assert_eq!(page.entries[2].message_cid, dummy_msg_cid(0));
        }
    }

    #[tokio::test]
    async fn mailbox_manager_publishes_root_cid() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("roots.db");
        let cas_path = dir.path().join("content");
        std::fs::create_dir_all(cas_path.join("commits")).unwrap();
        std::fs::create_dir_all(cas_path.join("blobs")).unwrap();

        let (publisher, handles) = ZenohPublisher::inert_for_test();
        let latest = handles.latest;

        let addr = [0xAAu8; ADDRESS_HASH_LEN];
        let mut mgr = MailboxManager::open(&db_path, &cas_path).unwrap();
        mgr.set_publisher(Arc::new(publisher));
        mgr.ensure_user_mailbox(&addr).unwrap();

        mgr.insert_message(
            &addr,
            &dummy_msg_cid(1),
            &dummy_msg_id(1),
            &dummy_sender(),
            1700000000,
            "Test Subject",
        )
        .unwrap();

        // notify() is synchronous: the coalescing map is updated before
        // insert_message returns, so no polling or timeout is needed.
        let map = latest
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let expected_addr_hex = hex::encode(addr);
        let cid = map
            .get(&expected_addr_hex)
            .expect("ZenohPublisher did not record a notification");
        assert_eq!(*cid, *mgr.get_root(&addr).unwrap());
        assert_eq!(map.len(), 1);
    }

    #[tokio::test]
    async fn zenoh_publisher_coalesces_per_addr() {
        // Two bursts of notifications for the same addr must collapse to the
        // latest CID, and the drain task must snapshot the newest value.
        let (publisher, handles) = ZenohPublisher::inert_for_test();
        let latest = handles.latest;

        let addr = [0xAAu8; ADDRESS_HASH_LEN];
        let addr_hex = hex::encode(addr);
        let cid_old = [0x01u8; CID_LEN];
        let cid_mid = [0x02u8; CID_LEN];
        let cid_new = [0x03u8; CID_LEN];

        publisher.notify(addr, cid_old);
        publisher.notify(addr, cid_mid);
        publisher.notify(addr, cid_new);

        let map = latest
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        assert_eq!(map.len(), 1, "coalescing must keep one entry per addr");
        assert_eq!(
            map.get(&addr_hex).copied(),
            Some(cid_new),
            "latest CID must win; older values must not linger"
        );
    }

    #[tokio::test]
    async fn zenoh_publisher_keeps_distinct_addrs_separate() {
        // Updates for different addrs must not clobber each other.
        let (publisher, handles) = ZenohPublisher::inert_for_test();
        let latest = handles.latest;

        let alice = [0xAAu8; ADDRESS_HASH_LEN];
        let bob = [0xBBu8; ADDRESS_HASH_LEN];
        let alice_hex = hex::encode(alice);
        let bob_hex = hex::encode(bob);
        let alice_cid = [0x11u8; CID_LEN];
        let bob_cid = [0x22u8; CID_LEN];

        publisher.notify(alice, alice_cid);
        publisher.notify(bob, bob_cid);

        let map = latest
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        assert_eq!(map.len(), 2);
        assert_eq!(map.get(&alice_hex).copied(), Some(alice_cid));
        assert_eq!(map.get(&bob_hex).copied(), Some(bob_cid));
    }

    #[tokio::test]
    async fn zenoh_publisher_captures_raw_mail() {
        // publish_raw_mail must route each call into the raw capture buffer
        // (in production this is a Zenoh put; the inert scaffold records it
        // so tests can assert on what would have been published). Payloads
        // flow through as `Arc<Vec<u8>>` so fan-out callers can clone a
        // refcount handle instead of duplicating the buffer.
        let (publisher, handles) = ZenohPublisher::inert_for_test();

        let alice_hex = "aa".repeat(ADDRESS_HASH_LEN);
        let bob_hex = "bb".repeat(ADDRESS_HASH_LEN);
        let payload_a = Arc::new(vec![0x01, 0x02, 0x03]);
        let payload_b = Arc::new(vec![0xFFu8]);
        let payload_c = Arc::new(vec![0x04, 0x05]);
        publisher.publish_raw_mail(alice_hex.clone(), Arc::clone(&payload_a));
        publisher.publish_raw_mail(bob_hex.clone(), Arc::clone(&payload_b));
        publisher.publish_raw_mail(alice_hex.clone(), Arc::clone(&payload_c));

        let raw = handles
            .raw
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        assert_eq!(raw.len(), 3, "each call must record one publish — no coalescing");
        assert_eq!(raw[0].0, alice_hex);
        assert!(Arc::ptr_eq(&raw[0].1, &payload_a), "shared-buffer handle must be preserved");
        assert_eq!(raw[1].0, bob_hex);
        assert!(Arc::ptr_eq(&raw[1].1, &payload_b));
        assert_eq!(raw[2].0, alice_hex);
        assert!(Arc::ptr_eq(&raw[2].1, &payload_c));
    }

    #[tokio::test]
    async fn zenoh_publisher_fanout_shares_buffer() {
        // Fanning out one payload to many recipients must Arc::clone the same
        // allocation rather than duplicate it — this is the fix for the
        // Qodo-reported OOM hazard (16MiB × 100 recipients ≈ 1.6GiB copies).
        let (publisher, handles) = ZenohPublisher::inert_for_test();
        let payload = Arc::new(vec![0xAAu8; 64]);

        for i in 0..10u8 {
            let addr_hex = format!("{:02x}", i).repeat(ADDRESS_HASH_LEN);
            publisher.publish_raw_mail(addr_hex, Arc::clone(&payload));
        }

        let raw = handles
            .raw
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        assert_eq!(raw.len(), 10);
        for entry in raw.iter() {
            assert!(
                Arc::ptr_eq(&entry.1, &payload),
                "every fan-out recipient must share the same underlying buffer"
            );
        }
    }

    #[tokio::test]
    async fn publish_sealed_unicast_captures_notification_on_test_publisher() {
        let (publisher, handles) = ZenohPublisher::inert_for_test();
        let recipient_hash_hex = "aabbccddeeff00112233445566778899".to_string();
        let envelope: Arc<Vec<u8>> = Arc::new(vec![0xDE, 0xAD, 0xBE, 0xEF]);

        publisher.publish_sealed_unicast(recipient_hash_hex.clone(), Arc::clone(&envelope));

        let sealed = handles
            .sealed_unicast
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        assert_eq!(sealed.len(), 1, "expected exactly one sealed-unicast capture");
        let (key, payload) = &sealed[0];
        assert_eq!(key, "harmony/msg/v1/unicast/aabbccddeeff00112233445566778899");
        assert_eq!(payload.as_slice(), envelope.as_slice());
    }

    #[tokio::test]
    async fn mailbox_manager_no_publish_without_publisher() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("roots.db");
        let cas_path = dir.path().join("content");
        std::fs::create_dir_all(cas_path.join("commits")).unwrap();
        std::fs::create_dir_all(cas_path.join("blobs")).unwrap();

        let addr = [0xBBu8; ADDRESS_HASH_LEN];
        let mut mgr = MailboxManager::open(&db_path, &cas_path).unwrap();
        // No publisher set — insert_message should succeed silently.
        mgr.insert_message(
            &addr,
            &dummy_msg_cid(2),
            &dummy_msg_id(2),
            &dummy_sender(),
            1700000001,
            "No Publisher",
        )
        .unwrap();
        // Root CID should be present in memory.
        assert!(mgr.get_root(&addr).is_some());
    }

    /// Exercises the REAL `ZenohPublisher::new()` drain task (not the mpsc
    /// test double) end-to-end: opens two in-process Zenoh sessions in peer
    /// mode over loopback, creates the publisher with one session, declares
    /// a subscriber on the other, inserts a message, and verifies the raw
    /// root CID bytes reach the subscriber through `session.put()`.
    ///
    /// This catches regressions the `from_sender` test cannot — e.g., if the
    /// drain task stops spawning, if the topic format breaks, or if
    /// `session.put()` starts returning errors silently.
    ///
    /// `#[ignore]`'d by default: starts two Zenoh sessions which bind sockets
    /// and rely on peer discovery; can be flaky under loaded CI runners.
    /// Run locally with `cargo test -p harmony-mail -- --ignored`.
    #[tokio::test]
    #[ignore]
    async fn mailbox_manager_real_zenoh_end_to_end() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("roots.db");
        let cas_path = dir.path().join("content");
        std::fs::create_dir_all(cas_path.join("commits")).unwrap();
        std::fs::create_dir_all(cas_path.join("blobs")).unwrap();

        // Two peer-mode sessions that discover each other on localhost.
        let pub_session = zenoh::open(zenoh::Config::default()).await.unwrap();
        let sub_session = zenoh::open(zenoh::Config::default()).await.unwrap();

        let addr = [0xCCu8; ADDRESS_HASH_LEN];
        let addr_hex = hex::encode(addr);
        let topic = format!("harmony/mail/v1/{addr_hex}/root");

        let subscriber = sub_session.declare_subscriber(&topic).await.unwrap();

        // Brief settle time for peer discovery on loopback.
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        let cancel = CancellationToken::new();
        let publisher = ZenohPublisher::new(pub_session, cancel.clone());

        let mut mgr = MailboxManager::open(&db_path, &cas_path).unwrap();
        mgr.set_publisher(Arc::new(publisher));
        mgr.ensure_user_mailbox(&addr).unwrap();

        mgr.insert_message(
            &addr,
            &dummy_msg_cid(7),
            &dummy_msg_id(7),
            &dummy_sender(),
            1700000042,
            "Real Zenoh",
        )
        .unwrap();

        // The drain task must pick up the coalesced update and run
        // session.put().
        let sample = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            subscriber.recv_async(),
        )
        .await
        .expect("subscriber never received the real zenoh publish within 5s")
        .expect("subscriber channel closed");

        let payload = sample.payload().to_bytes();
        assert_eq!(payload.len(), CID_LEN);
        let received_cid: [u8; CID_LEN] = (&payload[..]).try_into().unwrap();
        assert_eq!(received_cid, *mgr.get_root(&addr).unwrap());

        // Signal shutdown and give the drain task a chance to exit cleanly.
        cancel.cancel();
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    }

    /// Regression test for the drain-task shutdown path.
    ///
    /// The publisher's drain task must observe the shared `CancellationToken`
    /// so graceful shutdown can tear it down deterministically instead of
    /// leaking it until runtime drop. This test opens a real Zenoh session
    /// (so we exercise the real `new` path, not the `inert_for_test` scaffold),
    /// cancels the token, and confirms a subsequent `notify()` produces no
    /// publish — i.e., the drain task has stopped consuming wake-ups.
    ///
    /// `#[ignore]`'d for the same reason as the end-to-end test — relies on
    /// peer discovery over loopback.
    #[tokio::test]
    #[ignore]
    async fn zenoh_publisher_drain_task_stops_on_cancel() {
        let pub_session = zenoh::open(zenoh::Config::default()).await.unwrap();
        let sub_session = zenoh::open(zenoh::Config::default()).await.unwrap();

        let addr = [0xDDu8; ADDRESS_HASH_LEN];
        let addr_hex = hex::encode(addr);
        let topic = format!("harmony/mail/v1/{addr_hex}/root");
        let subscriber = sub_session.declare_subscriber(&topic).await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        let cancel = CancellationToken::new();
        let publisher = ZenohPublisher::new(pub_session, cancel.clone());

        // Sanity: the drain task is alive and publishes before cancel.
        publisher.notify(addr, [0x01u8; CID_LEN]);
        let before = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            subscriber.recv_async(),
        )
        .await
        .expect("pre-cancel publish should succeed")
        .expect("subscriber channel closed");
        assert_eq!(before.payload().to_bytes().len(), CID_LEN);

        // Cancel and give the drain task a moment to exit.
        cancel.cancel();
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        // A post-cancel notify should wake nothing — no further publish
        // arrives within a short window.
        publisher.notify(addr, [0x02u8; CID_LEN]);
        let after = tokio::time::timeout(
            std::time::Duration::from_millis(500),
            subscriber.recv_async(),
        )
        .await;
        assert!(
            after.is_err(),
            "drain task should have stopped after cancel; got unexpected publish"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn root_queryable_returns_current_root() {
        let cancel = CancellationToken::new();
        let session = zenoh::open(zenoh::Config::default()).await.unwrap();
        let publisher = ZenohPublisher::new(session.clone(), cancel.clone());

        // Allow the queryable declaration to settle within the session before
        // the first query.
        tokio::time::sleep(Duration::from_millis(100)).await;

        let addr = {
            let mut a = [0u8; ADDRESS_HASH_LEN];
            // Arbitrary deterministic bytes (hex: "00112233..ee...")
            for (i, slot) in a.iter_mut().enumerate() {
                *slot = (i as u8).wrapping_mul(0x11);
            }
            a
        };
        let addr_hex = hex::encode(addr);
        let root_cid = [0xAB; CID_LEN];
        publisher.notify(addr, root_cid);

        // Wait long enough that the drain task definitely consumed `latest`.
        // The queryable reads from `current`, which is never drained, so the
        // reply is deterministic regardless of drain-task timing.
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Drain all replies from the `session.get()` channel. Because every
        // test in this module opens its own Zenoh session on loopback, peer
        // scouting discovers sibling publishers running in parallel tests,
        // and each of their queryables will also receive our query and reply
        // with an empty payload (addr not in their `current`). We accept those
        // empty cross-session replies and assert our own queryable returned
        // the expected CID at least once. `ConsolidationMode::None` disables
        // Zenoh's default reply consolidation so we see every replier's
        // payload (consolidation would otherwise collapse them into one).
        let topic = format!("harmony/mail/v1/{addr_hex}/root");
        let replies = session
            .get(&topic)
            .consolidation(zenoh::query::ConsolidationMode::None)
            .await
            .unwrap();
        let mut payloads: Vec<Vec<u8>> = Vec::new();
        // Wrap the drain in a hard timeout so a slow/stalled Zenoh session
        // fails the test fast instead of hanging the suite.
        let drain_result = tokio::time::timeout(Duration::from_secs(3), async {
            while let Ok(reply) = replies.recv_async().await {
                if let Ok(sample) = reply.result() {
                    assert_eq!(
                        sample.key_expr().as_str(),
                        &topic,
                        "reply key should match query topic"
                    );
                    payloads.push(sample.payload().to_bytes().to_vec());
                }
            }
        })
        .await;
        assert!(
            drain_result.is_ok(),
            "drain timed out; payloads collected: {payloads:?}"
        );
        assert!(
            payloads.iter().any(|p| p.as_slice() == &root_cid[..]),
            "expected at least one reply with the root CID; got {payloads:?}"
        );
        cancel.cancel();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn root_queryable_empty_for_unknown_addr() {
        let cancel = CancellationToken::new();
        let session = zenoh::open(zenoh::Config::default()).await.unwrap();
        let _publisher = ZenohPublisher::new(session.clone(), cancel.clone());

        // Allow the queryable declaration to settle before querying.
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Drain every reply. With gossip scouting across parallel tests,
        // multiple queryables may reply, but none of them should have this
        // addr in `current`, so every reply must carry an empty payload.
        // `ConsolidationMode::None` disables reply consolidation so we see
        // every replier — including our own session-local queryable.
        let unknown = hex::encode([0xFFu8; ADDRESS_HASH_LEN]);
        let topic = format!("harmony/mail/v1/{unknown}/root");
        let replies = session
            .get(&topic)
            .consolidation(zenoh::query::ConsolidationMode::None)
            .await
            .unwrap();
        let mut got_any = false;
        let mut payloads: Vec<Vec<u8>> = Vec::new();
        // Wrap the drain in a hard timeout so a slow/stalled Zenoh session
        // fails the test fast instead of hanging the suite.
        let drain_result = tokio::time::timeout(Duration::from_secs(3), async {
            while let Ok(reply) = replies.recv_async().await {
                if let Ok(sample) = reply.result() {
                    got_any = true;
                    let bytes = sample.payload().to_bytes().to_vec();
                    assert!(
                        bytes.is_empty(),
                        "unknown address must yield an empty payload; got {bytes:?}"
                    );
                    payloads.push(bytes);
                }
            }
        })
        .await;
        assert!(
            drain_result.is_ok(),
            "drain timed out; payloads collected: {payloads:?}"
        );
        assert!(got_any, "expected at least one reply from our queryable");
        cancel.cancel();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn root_queryable_returns_latest_after_multiple_updates() {
        let cancel = CancellationToken::new();
        let session = zenoh::open(zenoh::Config::default()).await.unwrap();
        let publisher = ZenohPublisher::new(session.clone(), cancel.clone());

        // Allow the queryable declaration to settle before querying.
        tokio::time::sleep(Duration::from_millis(100)).await;

        let addr = {
            let mut a = [0u8; ADDRESS_HASH_LEN];
            // Arbitrary deterministic, distinct from other tests.
            for (i, slot) in a.iter_mut().enumerate() {
                *slot = 0x10u8.wrapping_add(i as u8);
            }
            a
        };
        let addr_hex = hex::encode(addr);
        publisher.notify(addr, [0x01; CID_LEN]);
        publisher.notify(addr, [0x02; CID_LEN]);
        publisher.notify(addr, [0x03; CID_LEN]);

        // Wait long enough that the drain task has definitely consumed
        // `latest`. Queryable reads from `current` which is never drained, so
        // this is deterministic.
        tokio::time::sleep(Duration::from_millis(200)).await;

        // As in the other queryable tests: drain all replies, tolerate empty
        // replies from sibling-test queryables (peer-discovered on loopback),
        // and assert our own queryable returned the latest CID at least once.
        // Specifically, no earlier CID (0x01 or 0x02) should ever appear —
        // `current` is strictly last-write-wins, not drained.
        // `ConsolidationMode::None` disables reply consolidation so every
        // replier's payload is delivered (our own local one included).
        let topic = format!("harmony/mail/v1/{addr_hex}/root");
        let replies = session
            .get(&topic)
            .consolidation(zenoh::query::ConsolidationMode::None)
            .await
            .unwrap();
        let mut payloads: Vec<Vec<u8>> = Vec::new();
        // Wrap the drain in a hard timeout so a slow/stalled Zenoh session
        // fails the test fast instead of hanging the suite.
        let drain_result = tokio::time::timeout(Duration::from_secs(3), async {
            while let Ok(reply) = replies.recv_async().await {
                if let Ok(sample) = reply.result() {
                    payloads.push(sample.payload().to_bytes().to_vec());
                }
            }
        })
        .await;
        assert!(
            drain_result.is_ok(),
            "drain timed out; payloads collected: {payloads:?}"
        );
        let expected = [0x03; CID_LEN];
        assert!(
            payloads.iter().any(|p| p.as_slice() == &expected[..]),
            "queryable should return the latest CID after multiple updates; got {payloads:?}"
        );
        // Neither of the earlier CIDs should leak through — last-write-wins.
        for bad in [[0x01u8; CID_LEN], [0x02u8; CID_LEN]] {
            assert!(
                !payloads.iter().any(|p| p.as_slice() == &bad[..]),
                "queryable must not return a stale CID {bad:?}; got {payloads:?}"
            );
        }
        cancel.cancel();
    }

    /// Regression test for the cold-start-sync path: after a server restart,
    /// `MailboxManager::open` rehydrates `self.roots` from SQLite, then
    /// `set_publisher` must seed the publisher's `current` map so the root
    /// queryable can answer for existing users IMMEDIATELY — without waiting
    /// for a new inbound message to repopulate `current` via `notify()`.
    ///
    /// Builds a manager, inserts one message (so a root is persisted),
    /// THEN attaches a fresh publisher, queries, and expects the persisted
    /// root in the replies.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn root_queryable_returns_persisted_root_after_set_publisher() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("roots.db");
        let cas_path = dir.path().join("content");
        std::fs::create_dir_all(cas_path.join("commits")).unwrap();
        std::fs::create_dir_all(cas_path.join("blobs")).unwrap();

        // Populate a persisted root BEFORE attaching the publisher, so the
        // seed path (not `notify`) is what puts the entry into `current`.
        let addr = [0xCDu8; ADDRESS_HASH_LEN];
        let mut mgr = MailboxManager::open(&db_path, &cas_path).unwrap();
        mgr.ensure_user_mailbox(&addr).unwrap();
        mgr.insert_message(
            &addr,
            &dummy_msg_cid(7),
            &dummy_msg_id(7),
            &dummy_sender(),
            1_700_000_042,
            "persisted",
        )
        .unwrap();
        let persisted_root = *mgr.get_root(&addr).unwrap();

        // Open a Zenoh session and attach a fresh publisher — this must seed
        // `current` from the persisted roots.
        let cancel = CancellationToken::new();
        let session = zenoh::open(zenoh::Config::default()).await.unwrap();
        let publisher = Arc::new(ZenohPublisher::new(session.clone(), cancel.clone()));
        mgr.set_publisher(Arc::clone(&publisher));

        // Allow queryable declaration + peer discovery to settle.
        tokio::time::sleep(Duration::from_millis(100)).await;

        let addr_hex = hex::encode(addr);
        let topic = format!("harmony/mail/v1/{addr_hex}/root");
        let replies = session
            .get(&topic)
            .consolidation(zenoh::query::ConsolidationMode::None)
            .await
            .unwrap();

        let mut payloads: Vec<Vec<u8>> = Vec::new();
        let drain_result = tokio::time::timeout(Duration::from_secs(3), async {
            while let Ok(reply) = replies.recv_async().await {
                if let Ok(sample) = reply.result() {
                    payloads.push(sample.payload().to_bytes().to_vec());
                }
            }
        })
        .await;
        assert!(
            drain_result.is_ok(),
            "drain timed out; collected: {payloads:?}"
        );

        let found_persisted = payloads.iter().any(|p| p.as_slice() == &persisted_root[..]);
        assert!(
            found_persisted,
            "expected persisted root in replies; got payloads: {payloads:?}"
        );
        cancel.cancel();
    }

    /// Regression test for the create-path sync: when a user's mailbox is
    /// created AFTER the publisher is attached (i.e., a brand-new user hits
    /// the gateway for the first time), the persisted root must appear in the
    /// queryable's `current` map immediately — without waiting for a first
    /// incoming message to trigger `notify()`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn root_queryable_returns_root_for_freshly_created_mailbox() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("roots.db");
        let cas_path = dir.path().join("content");
        std::fs::create_dir_all(cas_path.join("commits")).unwrap();
        std::fs::create_dir_all(cas_path.join("blobs")).unwrap();

        let mut mgr = MailboxManager::open(&db_path, &cas_path).unwrap();

        // Attach the publisher FIRST, then create the mailbox — this exercises
        // the create-path mirror (not the startup seed path).
        let cancel = CancellationToken::new();
        let session = zenoh::open(zenoh::Config::default()).await.unwrap();
        let publisher = Arc::new(ZenohPublisher::new(session.clone(), cancel.clone()));
        mgr.set_publisher(Arc::clone(&publisher));

        let addr = [0xEFu8; ADDRESS_HASH_LEN];
        mgr.ensure_user_mailbox(&addr).unwrap();
        let expected_root = *mgr.get_root(&addr).unwrap();

        // Allow queryable declaration to settle.
        tokio::time::sleep(Duration::from_millis(100)).await;

        let addr_hex = hex::encode(addr);
        let topic = format!("harmony/mail/v1/{addr_hex}/root");
        let replies = session
            .get(&topic)
            .consolidation(zenoh::query::ConsolidationMode::None)
            .await
            .unwrap();

        let mut payloads: Vec<Vec<u8>> = Vec::new();
        let drain_result = tokio::time::timeout(Duration::from_secs(3), async {
            while let Ok(reply) = replies.recv_async().await {
                if let Ok(sample) = reply.result() {
                    payloads.push(sample.payload().to_bytes().to_vec());
                }
            }
        })
        .await;
        assert!(
            drain_result.is_ok(),
            "drain timed out; collected: {payloads:?}"
        );

        let found = payloads.iter().any(|p| p.as_slice() == &expected_root[..]);
        assert!(
            found,
            "expected freshly-created mailbox root in replies; got payloads: {payloads:?}"
        );
        cancel.cancel();
    }
}
