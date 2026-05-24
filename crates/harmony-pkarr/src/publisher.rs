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

#[derive(Clone)]
struct ActivePublication {
    handle: String,
    ephemeral_key: SigningKey,
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
    /// `tokio::sync::Mutex` — held across await points in `drive_pending`.
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
    /// If the handle was already registered, the prior entry's `cancelled` flag
    /// is set so any in-flight publish cloned from the old entry short-circuits
    /// before executing its PUT (Fix 3).
    pub async fn register(
        &self,
        handle: String,
        ephemeral_key: SigningKey,
        builder: RecordBuilder,
    ) {
        let cancelled = Arc::new(AtomicBool::new(false));
        let pub_state = ActivePublication {
            handle: handle.clone(),
            ephemeral_key,
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
        let record = (pub_state.builder)(now);
        let cbor = record.to_canonical_cbor()?;
        let envelope = wrap_bep44_envelope(&pub_state.ephemeral_key, &cbor, now)?;
        let key_z32 = z32_encode_public(&pub_state.ephemeral_key.verifying_key().to_bytes());
        self.relay.put(&key_z32, &envelope).await
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

/// Build a BEP44 envelope using the simple in-house wire format for
/// mock-relay testing:
///
/// ```text
/// envelope = sig(64 bytes) ‖ seq(8 bytes, LE u64) ‖ payload(N bytes)
/// ```
///
/// The signature covers the BEP44 v=0 salt:
/// `"3:seqi{seq}e1:v{len}:{payload}"`.
///
/// `seq` is milliseconds since Unix epoch as `u64` — avoids the ~49.7-day
/// wrap that a `u32` cast would introduce.
///
/// This is the format the companion `PkarrResolver` decodes.
/// Real pkarr-relay bencoded compatibility is Phase 3 hardening.
fn wrap_bep44_envelope(
    ephemeral_key: &SigningKey,
    payload: &[u8],
    seq: u64,
) -> Result<Vec<u8>, PkarrError> {
    use ed25519_dalek::Signer;
    // BEP44 v=0 sign material (same as real pkarr relays expect).
    let mut to_sign = Vec::new();
    to_sign.extend_from_slice(format!("3:seqi{}e1:v{}:", seq, payload.len()).as_bytes());
    to_sign.extend_from_slice(payload);
    let sig = ephemeral_key.sign(&to_sign);

    // Envelope: sig(64) ‖ seq(8 LE u64) ‖ payload.
    let mut envelope = Vec::with_capacity(64 + 8 + payload.len());
    envelope.extend_from_slice(&sig.to_bytes());
    envelope.extend_from_slice(&seq.to_le_bytes());
    envelope.extend_from_slice(payload);
    Ok(envelope)
}

/// Encode a 32-byte Ed25519 public key as a string for use in relay URL paths.
///
/// Uses lowercase hex for now — the mock relay (Task 5) accepts arbitrary
/// strings. Phase 3 hardening will replace this with real z-base-32 (e.g.,
/// the `zbase32` crate) for production pkarr-relay interop.
fn z32_encode_public(pk: &[u8; 32]) -> String {
    hex::encode(pk)
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
        let pk_hex = hex::encode(ephemeral.verifying_key().to_bytes());

        let identity_sk = SigningKey::generate(&mut OsRng);
        let mut identity_pub = [0u8; 64];
        identity_pub[32..].copy_from_slice(&identity_sk.verifying_key().to_bytes());
        let identity_sk_for_builder = identity_sk.clone();
        let builder: RecordBuilder = Arc::new(move |at_ms| {
            PkarrRoutingRecord::sign_new(
                b"test-routing".to_vec(),
                identity_pub,
                at_ms,
                &identity_sk_for_builder,
            )
            .expect("sign")
        });

        publisher
            .register("test-pub".to_string(), ephemeral, builder)
            .await;

        // Poll the mock relay for up to 2.5s for the publication to land.
        let mut retries = 0;
        while retries < 50 {
            tokio::time::sleep(Duration::from_millis(50)).await;
            let resp = reqwest::get(format!("{}/{}", relay.base_url, pk_hex))
                .await
                .expect("http get");
            if resp.status() == 200 {
                let body = resp.bytes().await.expect("body");
                // Envelope = 64-byte sig + 8-byte seq (u64 LE) + CBOR payload (>= ~70 bytes).
                assert!(
                    body.len() > 64 + 8,
                    "envelope too short: {} bytes",
                    body.len()
                );
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
        let identity_sk = SigningKey::generate(&mut OsRng);
        let mut identity_pub = [0u8; 64];
        identity_pub[32..].copy_from_slice(&identity_sk.verifying_key().to_bytes());
        let identity_sk_for_builder = identity_sk.clone();
        let builder: RecordBuilder = Arc::new(move |at_ms| {
            PkarrRoutingRecord::sign_new(
                b"blob".to_vec(),
                identity_pub,
                at_ms,
                &identity_sk_for_builder,
            )
            .expect("sign")
        });

        publisher
            .register("to-remove".to_string(), ephemeral, builder)
            .await;
        publisher.unregister("to-remove").await;

        assert!(publisher.active_handles().await.is_empty());
    }
}
