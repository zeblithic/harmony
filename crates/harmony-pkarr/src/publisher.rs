//! `PkarrPublisher` — background task that publishes registered keys on a
//! schedule and republishes ahead of DHT-record TTL expiry.
//!
//! Spec Section 6 (publication lifecycles per case).
//!
//! Caller responsibility: `register(handle, key, record_builder)` adds an
//! active publication; `unregister(handle)` removes it. The publisher itself
//! is case-agnostic — case-specific lifecycle logic lives in harmony-client.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use ed25519_dalek::SigningKey;
use tokio::sync::{Mutex, Notify};

use crate::epoch::{current_epoch_id, epoch_start_ms, EPOCH_DURATION_MS};
use crate::error::PkarrError;
use crate::record::PkarrRoutingRecord;
use crate::relay::RelayClient;

/// Callback that produces a fresh `PkarrRoutingRecord` for a publication.
///
/// Closure-typed so callers can capture per-publication state (which iroh
/// routing to encode, what `announced_at_ms` to stamp).
///
/// The `u64` argument is the current time in milliseconds since Unix epoch.
pub type RecordBuilder = Arc<dyn Fn(u64) -> PkarrRoutingRecord + Send + Sync>;

/// Callback that produces the current epoch's ephemeral `SigningKey` for a
/// publication. Invoked on every publish so that the key rotates as the
/// epoch advances — registering once at the start of epoch N would otherwise
/// keep publishing under the epoch-N key after the boundary, where resolvers
/// query under epoch N+1 ± tolerance and never find the record.
///
/// The `u64` argument is the current time in milliseconds since Unix epoch.
pub type EphemeralKeyBuilder = Arc<dyn Fn(u64) -> SigningKey + Send + Sync>;

#[derive(Clone)]
struct ActivePublication {
    handle: String,
    ephemeral_key_builder: EphemeralKeyBuilder,
    builder: RecordBuilder,
    next_publish_at: Instant,
    /// Set to `true` when the publication is unregistered or replaced.
    ///
    /// `publish_one` checks this flag before executing the PUT so that
    /// in-flight publishes cloned before the lock was released are
    /// short-circuited rather than racing with the updated state.
    ///
    /// Fix 2 (unregister allows in-flight publish) and Fix 3
    /// (mid-flight register skips new publish) both rely on this flag.
    cancelled: Arc<AtomicBool>,
}

/// Background task that holds a set of active publications and republishes
/// them on a schedule derived from the BEP44 epoch lifecycle.
///
/// Wrap in `Arc` and call [`spawn`][PkarrPublisher::spawn] to start the
/// background driver. Use [`register`][PkarrPublisher::register] /
/// [`unregister`][PkarrPublisher::unregister] to manage publications.
pub struct PkarrPublisher {
    relay: Arc<RelayClient>,
    /// `tokio::sync::Mutex`. `drive_pending` acquires it only for brief
    /// synchronous critical sections — snapshotting the due set, then (after
    /// each network PUT completes) updating that entry's schedule — and never
    /// holds it across an `.await`. `try_active_handles`'s `try_lock` therefore
    /// contends only during those sub-millisecond windows.
    state: Arc<Mutex<HashMap<String, ActivePublication>>>,
    /// Poked from `register` so the background loop wakes immediately for
    /// new publications rather than waiting out the current sleep.
    wakeup: Arc<Notify>,
}

impl PkarrPublisher {
    /// Create a new publisher backed by the given relay client.
    pub fn new(relay: Arc<RelayClient>) -> Self {
        Self {
            relay,
            state: Arc::new(Mutex::new(HashMap::new())),
            wakeup: Arc::new(Notify::new()),
        }
    }

    /// Add or replace an active publication. Schedules an immediate publish
    /// on the next loop tick by setting `next_publish_at` to `Instant::now()`
    /// and notifying the background loop.
    ///
    /// `ephemeral_key_builder` is invoked on every publish so the key rotates
    /// with the epoch — see [`EphemeralKeyBuilder`] for why this is a closure
    /// rather than a fixed `SigningKey`.
    ///
    /// If the handle was already registered, the prior entry's `cancelled` flag
    /// is set so any in-flight publish cloned from the old entry short-circuits
    /// before executing its PUT (Fix 3).
    pub async fn register(
        &self,
        handle: String,
        ephemeral_key_builder: EphemeralKeyBuilder,
        builder: RecordBuilder,
    ) {
        let cancelled = Arc::new(AtomicBool::new(false));
        let pub_state = ActivePublication {
            handle: handle.clone(),
            ephemeral_key_builder,
            builder,
            next_publish_at: Instant::now(),
            cancelled: Arc::clone(&cancelled),
        };
        let mut state = self.state.lock().await;
        // Cancel the old entry before replacing it so that any clone already
        // in-flight from drive_pending doesn't overwrite the new entry's
        // next_publish_at after we insert the replacement.
        if let Some(old) = state.get(&handle) {
            old.cancelled.store(true, Ordering::Release);
        }
        state.insert(handle, pub_state);
        drop(state);
        self.wakeup.notify_one();
    }

    /// Remove an active publication. Future publish ticks will skip this key.
    ///
    /// The removed entry's `cancelled` flag is set so any in-flight publish
    /// clone short-circuits before executing its PUT (Fix 2).
    pub async fn unregister(&self, handle: &str) {
        if let Some(old) = self.state.lock().await.remove(handle) {
            old.cancelled.store(true, Ordering::Release);
        }
    }

    /// Returns the handles of all currently-registered publications.
    pub async fn active_handles(&self) -> Vec<String> {
        self.state.lock().await.keys().cloned().collect()
    }

    /// Non-blocking variant of [`active_handles`][Self::active_handles] for
    /// synchronous callers (e.g. `network_health`'s `PkarrSnapshot`, which is a
    /// sync trait and cannot await). Returns `Some(handles)` when the state
    /// mutex is uncontended, or `None` if it is momentarily locked by the
    /// background driver. `drive_pending` never holds this lock across an
    /// `await` — it clones the due set under the lock, then drops it before
    /// each network PUT — so contention windows are sub-millisecond and `None`
    /// is rare; callers treat it as "unknown, fall back".
    pub fn try_active_handles(&self) -> Option<Vec<String>> {
        // Collect under the lock, then release the guard before sorting so the
        // critical section stays minimal. Sorted output keeps synchronous
        // snapshots / equality assertions deterministic (HashMap order isn't).
        let mut handles: Vec<String> = self.state.try_lock().ok()?.keys().cloned().collect();
        handles.sort();
        Some(handles)
    }

    /// Spawn the background driver. Caller keeps the returned `JoinHandle` so
    /// it can `abort()` on shutdown.
    pub fn spawn(self: Arc<Self>) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            loop {
                // Wait until the next scheduled publish, or until a new
                // registration wakes us early.
                let sleep_until = self.next_wakeup().await;
                tokio::select! {
                    _ = tokio::time::sleep_until(sleep_until) => {}
                    _ = self.wakeup.notified() => {}
                }
                self.drive_pending().await;
            }
        })
    }

    /// Publish all publications whose `next_publish_at` has passed.
    ///
    /// Schedule is updated AFTER each publish attempt so that a failure does
    /// not silently delay the next attempt by a full epoch window:
    /// - success → next scheduled epoch slot (`compute_next_publish_at`)
    /// - failure → 60 s retry backoff
    async fn drive_pending(&self) {
        let now = Instant::now();
        // Collect due publications without advancing their schedule yet.
        let due: Vec<ActivePublication> = {
            let state = self.state.lock().await;
            state
                .values()
                .filter(|p| p.next_publish_at <= now)
                .cloned()
                .collect()
        };
        for pub_state in due {
            match self.publish_one(&pub_state).await {
                Ok(()) => {
                    let mut state = self.state.lock().await;
                    // Only update next_publish_at if this entry has not been
                    // cancelled (replaced or unregistered). If it was cancelled,
                    // the new entry's next_publish_at (set to Instant::now() in
                    // register()) must not be overwritten (Fix 3).
                    if let Some(entry) = state.get_mut(&pub_state.handle) {
                        if !pub_state.cancelled.load(Ordering::Acquire) {
                            entry.next_publish_at = compute_next_publish_at(now_ms());
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        handle = %pub_state.handle,
                        error = ?e,
                        "pkarr publish failed — retrying in 60s"
                    );
                    let mut state = self.state.lock().await;
                    if let Some(entry) = state.get_mut(&pub_state.handle) {
                        if !pub_state.cancelled.load(Ordering::Acquire) {
                            entry.next_publish_at = Instant::now() + Duration::from_secs(60);
                        }
                    }
                }
            }
        }
    }

    async fn publish_one(&self, pub_state: &ActivePublication) -> Result<(), PkarrError> {
        // Short-circuit if this entry was cancelled (unregistered or replaced
        // by a new register() call) between when drive_pending cloned it and
        // now. This prevents in-flight publishes from hitting the relay after
        // the caller has already called unregister() (Fix 2 + Fix 3).
        if pub_state.cancelled.load(Ordering::Acquire) {
            return Ok(());
        }
        let now = now_ms();
        // Re-derive the ephemeral key for the current epoch on every publish.
        // Bug history: this used to be a fixed SigningKey captured at register
        // time, which silently broke discovery after the first epoch boundary
        // because the publisher kept writing under the previous epoch's DHT
        // key while resolvers queried under the current epoch ± tolerance.
        let ephemeral_key = (pub_state.ephemeral_key_builder)(now);
        let record = (pub_state.builder)(now);
        let (key_z32, payload) = crate::wire::build_relay_payload(&ephemeral_key, &record)?;
        self.relay.put(&key_z32, &payload).await
    }

    async fn next_wakeup(&self) -> tokio::time::Instant {
        let state = self.state.lock().await;
        let std_instant = state
            .values()
            .map(|p| p.next_publish_at)
            .min()
            .unwrap_or_else(|| Instant::now() + Duration::from_secs(3600));
        // Convert std::time::Instant → tokio::time::Instant.
        let from_now = std_instant.saturating_duration_since(Instant::now());
        tokio::time::Instant::now() + from_now
    }
}

/// Compute the next scheduled publish time relative to `now_ms`.
///
/// Schedule per spec Section 5.4: `epoch_start + 30 min` and
/// `epoch_start + 3.5 days`. Returns the nearest future candidate; if both
/// candidates within the current epoch are already past, returns
/// next-epoch `+30 min`.
pub(crate) fn compute_next_publish_at(now_ms: u64) -> Instant {
    let epoch_id = current_epoch_id(now_ms);
    let epoch_start = epoch_start_ms(epoch_id);
    let candidate_a = epoch_start + 30 * 60 * 1_000; // epoch_start + 30 min
    let candidate_b = epoch_start + EPOCH_DURATION_MS / 2; // epoch_start + 3.5 d
    let candidate_c = epoch_start_ms(epoch_id + 1) + 30 * 60 * 1_000; // next epoch + 30 min

    let next_ms = [candidate_a, candidate_b, candidate_c]
        .iter()
        .copied()
        .filter(|&t| t > now_ms)
        .min()
        .unwrap_or(now_ms + EPOCH_DURATION_MS);

    Instant::now() + Duration::from_millis(next_ms - now_ms)
}

fn now_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before Unix epoch is unsupported")
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::MockPkarrRelay;
    use ed25519_dalek::SigningKey;
    use rand::rngs::OsRng;

    #[tokio::test]
    async fn registered_publication_publishes_immediately() {
        let relay = MockPkarrRelay::start().await;
        let pool = crate::relay::RelayPool::new(vec![relay.base_url.clone()]);
        let client = Arc::new(crate::relay::RelayClient::new(pool));
        let publisher = Arc::new(PkarrPublisher::new(client));
        let _handle = Arc::clone(&publisher).spawn();

        let ephemeral = SigningKey::generate(&mut OsRng);
        let pk_z32 = crate::wire::z32_for_verifying_key(&ephemeral.verifying_key()).expect("z32");
        let ephemeral_for_builder = ephemeral.clone();
        let key_builder: EphemeralKeyBuilder = Arc::new(move |_| ephemeral_for_builder.clone());

        let identity_sk = SigningKey::generate(&mut OsRng);
        let mut identity_pub = [0u8; 64];
        identity_pub[32..].copy_from_slice(&identity_sk.verifying_key().to_bytes());
        let identity_sk_for_builder = identity_sk.clone();
        let builder: RecordBuilder = Arc::new(move |at_ms| {
            PkarrRoutingRecord::sign_new(
                b"test-routing".to_vec(),
                identity_pub,
                at_ms,
                at_ms + 604_800_000,
                &identity_sk_for_builder,
            )
            .expect("sign")
        });

        publisher
            .register("test-pub".to_string(), key_builder, builder)
            .await;

        // Poll the mock relay for up to 2.5s for the publication to land.
        let mut retries = 0;
        while retries < 50 {
            tokio::time::sleep(Duration::from_millis(50)).await;
            let resp = reqwest::get(format!("{}/{}", relay.base_url, pk_z32))
                .await
                .expect("http get");
            if resp.status() == 200 {
                let body = resp.bytes().await.expect("body");
                // Real pkarr relay payload is non-empty (sig + seq + DNS packet).
                assert!(!body.is_empty(), "payload should be non-empty");
                return;
            }
            retries += 1;
        }
        panic!("publisher did not push to mock relay within 2.5s");
    }

    #[tokio::test]
    async fn unregister_stops_future_publishes() {
        let relay = MockPkarrRelay::start().await;
        let pool = crate::relay::RelayPool::new(vec![relay.base_url.clone()]);
        let client = Arc::new(crate::relay::RelayClient::new(pool));
        let publisher = Arc::new(PkarrPublisher::new(client));
        let _handle = Arc::clone(&publisher).spawn();

        let ephemeral = SigningKey::generate(&mut OsRng);
        let key_builder: EphemeralKeyBuilder = Arc::new(move |_| ephemeral.clone());
        let identity_sk = SigningKey::generate(&mut OsRng);
        let mut identity_pub = [0u8; 64];
        identity_pub[32..].copy_from_slice(&identity_sk.verifying_key().to_bytes());
        let identity_sk_for_builder = identity_sk.clone();
        let builder: RecordBuilder = Arc::new(move |at_ms| {
            PkarrRoutingRecord::sign_new(
                b"blob".to_vec(),
                identity_pub,
                at_ms,
                at_ms + 604_800_000,
                &identity_sk_for_builder,
            )
            .expect("sign")
        });

        publisher
            .register("to-remove".to_string(), key_builder, builder)
            .await;
        publisher.unregister("to-remove").await;

        assert!(publisher.active_handles().await.is_empty());
    }

    /// `try_active_handles` returns the registered handle set without awaiting,
    /// so a synchronous caller (e.g. the Network Health snapshot) can read
    /// publish state. Returns `Some` whenever the state mutex is uncontended,
    /// and the handles are sorted for deterministic snapshot output.
    #[tokio::test]
    async fn try_active_handles_returns_sorted_registered_handles() {
        let relay = MockPkarrRelay::start().await;
        let pool = crate::relay::RelayPool::new(vec![relay.base_url.clone()]);
        let client = Arc::new(crate::relay::RelayClient::new(pool));
        let publisher = Arc::new(PkarrPublisher::new(Arc::clone(&client)));

        // Empty (but readable) before any registration.
        assert_eq!(publisher.try_active_handles(), Some(Vec::new()));

        let key_builder: EphemeralKeyBuilder =
            Arc::new(move |_at_ms| SigningKey::generate(&mut OsRng));
        let identity_sk = SigningKey::generate(&mut OsRng);
        let mut identity_pub = [0u8; 64];
        identity_pub[32..].copy_from_slice(&identity_sk.verifying_key().to_bytes());
        let id_sk = identity_sk.clone();
        let builder: RecordBuilder = Arc::new(move |at_ms| {
            PkarrRoutingRecord::sign_new(
                b"blob".to_vec(),
                identity_pub,
                at_ms,
                at_ms + 604_800_000,
                &id_sk,
            )
            .expect("sign")
        });

        // Register two handles out of sorted order; the accessor returns them
        // deterministically sorted regardless of HashMap iteration order.
        publisher
            .register(
                "z-last".to_string(),
                Arc::clone(&key_builder),
                Arc::clone(&builder),
            )
            .await;
        publisher
            .register("identity".to_string(), key_builder, builder)
            .await;

        let handles = publisher.try_active_handles().expect("uncontended");
        assert_eq!(handles, vec!["identity".to_string(), "z-last".to_string()]);
    }

    /// Regression: every publish must invoke `ephemeral_key_builder` so the
    /// key tracks the current epoch. Before this fix, `register` snapshotted
    /// a single SigningKey at registration time and reused it for every
    /// subsequent publish — meaning publication silently kept writing to the
    /// old epoch's DHT key after the boundary, while resolvers (which derive
    /// keys against the current epoch ± tolerance) stopped finding the
    /// record. This test verifies the builder is called multiple times by
    /// forcing two publish cycles and observing distinct relay URLs.
    #[tokio::test]
    async fn ephemeral_key_builder_invoked_each_publish() {
        let relay = MockPkarrRelay::start().await;
        let pool = crate::relay::RelayPool::new(vec![relay.base_url.clone()]);
        let client = Arc::new(crate::relay::RelayClient::new(pool));
        let publisher = Arc::new(PkarrPublisher::new(Arc::clone(&client)));
        let _handle = Arc::clone(&publisher).spawn();

        // Two distinct keys; the builder returns key_a then key_b on
        // subsequent calls so we can verify both surfaces appear on the relay.
        let key_a = SigningKey::generate(&mut OsRng);
        let key_b = SigningKey::generate(&mut OsRng);
        let pk_a_z32 = crate::wire::z32_for_verifying_key(&key_a.verifying_key()).expect("z32 a");
        let pk_b_z32 = crate::wire::z32_for_verifying_key(&key_b.verifying_key()).expect("z32 b");
        let call_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let call_count_clone = Arc::clone(&call_count);
        let key_a_clone = key_a.clone();
        let key_b_clone = key_b.clone();
        let key_builder: EphemeralKeyBuilder = Arc::new(move |_now_ms| {
            let n = call_count_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if n == 0 {
                key_a_clone.clone()
            } else {
                key_b_clone.clone()
            }
        });

        let identity_sk = SigningKey::generate(&mut OsRng);
        let mut identity_pub = [0u8; 64];
        identity_pub[32..].copy_from_slice(&identity_sk.verifying_key().to_bytes());
        let identity_sk_for_builder = identity_sk.clone();
        let builder: RecordBuilder = Arc::new(move |at_ms| {
            PkarrRoutingRecord::sign_new(
                b"rotation-test".to_vec(),
                identity_pub,
                at_ms,
                at_ms + 604_800_000,
                &identity_sk_for_builder,
            )
            .expect("sign")
        });

        publisher
            .register("rotation".to_string(), key_builder, builder)
            .await;

        // Wait for the first publish (under key_a) to land.
        let mut tries = 0;
        loop {
            tokio::time::sleep(Duration::from_millis(50)).await;
            tries += 1;
            assert!(tries < 60, "first publish (key_a) did not land");
            let resp = reqwest::get(format!("{}/{}", relay.base_url, pk_a_z32))
                .await
                .expect("http get");
            if resp.status() == 200 {
                break;
            }
        }

        // Force a second publish by unregister + re-register. This drives a
        // fresh publish cycle that should invoke the builder again, which
        // returns key_b — simulating the epoch-boundary case where the same
        // logical publication wakes up after the epoch rolled and re-derives
        // its key.
        publisher.unregister("rotation").await;
        // Re-build with a new pair of closures so the registry sees a new
        // entry. The second builder call returns key_b (call_count == 1).
        let call_count_clone2 = Arc::clone(&call_count);
        let key_a2 = key_a.clone();
        let key_b2 = key_b.clone();
        let key_builder2: EphemeralKeyBuilder = Arc::new(move |_now_ms| {
            let n = call_count_clone2.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if n == 0 {
                key_a2.clone()
            } else {
                key_b2.clone()
            }
        });
        let identity_sk_for_builder2 = identity_sk.clone();
        let builder2: RecordBuilder = Arc::new(move |at_ms| {
            PkarrRoutingRecord::sign_new(
                b"rotation-test".to_vec(),
                identity_pub,
                at_ms,
                at_ms + 604_800_000,
                &identity_sk_for_builder2,
            )
            .expect("sign")
        });
        publisher
            .register("rotation".to_string(), key_builder2, builder2)
            .await;

        // Wait for key_b's record to appear on the relay.
        let mut tries = 0;
        loop {
            tokio::time::sleep(Duration::from_millis(50)).await;
            tries += 1;
            assert!(tries < 60, "second publish (key_b) did not land");
            let resp = reqwest::get(format!("{}/{}", relay.base_url, pk_b_z32))
                .await
                .expect("http get");
            if resp.status() == 200 {
                break;
            }
        }

        // Builder must have been invoked at least twice. Concrete count is
        // ≥ 2 because the loop may have retried during the first poll, but
        // observing distinct URLs proves the key did rotate.
        assert!(
            call_count.load(std::sync::atomic::Ordering::SeqCst) >= 2,
            "builder must be invoked on every publish, not just at register"
        );
    }

    #[tokio::test]
    async fn publishes_real_pkarr_payload_to_strict_mock() {
        let relay = MockPkarrRelay::start_strict().await;
        let client = Arc::new(crate::relay::RelayClient::new(
            crate::relay::RelayPool::new(vec![relay.base_url.clone()]),
        ));
        let publisher = Arc::new(PkarrPublisher::new(client));

        let id_sk = ed25519_dalek::SigningKey::from_bytes(&[3u8; 32]);
        let mut id_pub = [0u8; 64];
        id_pub[32..].copy_from_slice(&id_sk.verifying_key().to_bytes());
        let id_sk_for_builder = id_sk.clone();

        publisher
            .register(
                "h1".to_string(),
                Arc::new(|_now| ed25519_dalek::SigningKey::from_bytes(&[9u8; 32])),
                Arc::new(move |now| {
                    crate::record::PkarrRoutingRecord::sign_new(
                        b"routing".to_vec(),
                        id_pub,
                        now,
                        now + 604_800_000,
                        &id_sk_for_builder,
                    )
                    .expect("sign record")
                }),
            )
            .await;

        let handle = Arc::clone(&publisher).spawn();
        // Give the background loop time to publish once.
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        handle.abort();

        // Strict mock accepted the PUT → the payload is real pkarr format.
        let ephemeral = ed25519_dalek::SigningKey::from_bytes(&[9u8; 32]);
        let z32 = crate::wire::z32_for_verifying_key(&ephemeral.verifying_key()).unwrap();
        let client2 = reqwest::Client::new();
        let got = client2
            .get(format!("{}/{}", relay.base_url, z32))
            .send()
            .await
            .expect("get");
        assert_eq!(got.status(), 200);
    }
}
