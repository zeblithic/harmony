//! E-mail → identity resolution facade (spec §5.5 + §6).
//!
//! `EmailResolver` is the async trait that the SMTP server calls.
//! `DefaultEmailResolver` is the production implementation backed by a real
//! DNS client, a real HTTP client, and the four-level cache from `cache.rs`.
//! Cold-path fetch logic is implemented in Task 15.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use dashmap::DashMap;
use ed25519_dalek::{Signature as EdSignature, Verifier, VerifyingKey};
use harmony_identity::IdentityHash;
use tokio::sync::{Mutex, Semaphore};
use tokio::task::JoinHandle;
use tracing::{error, warn};

use crate::cache::{CacheLimits, ResolverCaches, TimeSource};
use crate::claim::{
    canonical_cbor, hashed_local_part, DomainRecord, MasterPubkey, RevocationList, RevocationView,
    Signature, VerifyError,
};
use crate::dns::{DnsClient, DnsFetchError};
use crate::http::{ClaimFetchResult, HttpClient, HttpFetchError, RevocationFetchResult};

// ─── Trait ───────────────────────────────────────────────────────────────────

/// Resolve an e-mail address to an `IdentityHash` (or report why it can't).
///
/// Implementations must be `Send + Sync + 'static` so they can be held in
/// shared server state and called from any async task.
#[async_trait]
pub trait EmailResolver: Send + Sync + 'static {
    async fn resolve(&self, local_part: &str, domain: &str) -> ResolveOutcome;
}

// ─── Outcome ─────────────────────────────────────────────────────────────────

/// Every possible result of an `EmailResolver::resolve` call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolveOutcome {
    /// The address is registered and maps to this identity.
    Resolved(IdentityHash),
    /// The domain has no Harmony DNS record — not a participant.
    DomainDoesNotParticipate,
    /// The domain participates but has no claim for this local part.
    UserUnknown,
    /// A recoverable infrastructure error; the caller should treat the
    /// address as temporarily unresolvable.
    Transient { reason: &'static str },
    /// The claim exists but has been revoked.
    Revoked,
}

// ─── Config ──────────────────────────────────────────────────────────────────

/// Tuning knobs for `DefaultEmailResolver`.  All `_secs` fields are seconds.
#[derive(Debug, Clone)]
pub struct ResolverConfig {
    /// How long (seconds) after last-successful-DNS we still soft-accept a
    /// domain whose DNS is now unreachable (spec §5.5 72 h window).
    pub soft_fail_window_secs: u64,
    /// How long (seconds) to cache a negative (UserUnknown) result.
    pub negative_cache_secs: u64,
    /// How often (seconds) to proactively refresh the revocation list for an
    /// active domain.
    pub revocation_refresh_secs: u64,
    /// Hard safety valve (seconds): if the revocation list hasn't been
    /// refreshed in this long, stop accepting claims from that domain.
    pub revocation_safety_valve_secs: u64,
    /// Allowed clock drift between the local clock and the `not_before` /
    /// `not_after` fields in a claim (seconds).
    pub clock_skew_tolerance_secs: u64,
    /// Default TTL (seconds) to use for DNS records that don't carry their own
    /// TTL.
    pub dns_ttl_default_secs: u64,
    /// Capacity limits for the in-process caches.
    pub cache_limits: CacheLimits,
    /// How often the background sweep task wakes up to evict expired entries.
    pub background_sweep_interval: Duration,
}

impl Default for ResolverConfig {
    fn default() -> Self {
        Self {
            soft_fail_window_secs: 72 * 3600,        // 72 hours (spec §5.5)
            negative_cache_secs: 60,                 // 1 minute (spec §6)
            revocation_refresh_secs: 6 * 3600,       // 6 hours  (spec §5.4)
            revocation_safety_valve_secs: 24 * 3600, // 24 hours (spec §5.4)
            clock_skew_tolerance_secs: 60,           // 1 minute (spec §4.3)
            dns_ttl_default_secs: 3600,              // 1 hour   (spec §5.1)
            cache_limits: CacheLimits::default(),
            background_sweep_interval: Duration::from_secs(15 * 60), // 15 minutes
        }
    }
}

// ─── DefaultEmailResolver ────────────────────────────────────────────────────

/// Production resolver.  Holds Arc-wrapped clients so it can be cheaply
/// cloned into background tasks.
pub struct DefaultEmailResolver {
    dns: Arc<dyn DnsClient>,
    http: Arc<dyn HttpClient>,
    caches: Arc<ResolverCaches>,
    time: Arc<dyn TimeSource>,
    config: ResolverConfig,
    /// One `Semaphore(1)` per domain, used to serialise concurrent revocation
    /// refreshes for the same domain without blocking unrelated domains.
    ///
    /// NOTE: this map grows once per distinct domain ever resolved and is
    /// never pruned. For the expected footprint of a single gateway this is
    /// a bounded cost, but Task 19's background sweep should evict entries
    /// for domains that have not been seen recently (cf. `domain_last_seen`).
    revocation_refresh_locks: Arc<dashmap::DashMap<String, Arc<Semaphore>>>,
    /// Highest observed claim serial per `(domain, hashed_local_part)`.
    /// Used to detect serial rollback attacks: a claim whose serial is
    /// strictly less than the previously-seen highest is rejected as
    /// `Transient("claim_serial_rollback")`.
    ///
    /// NOTE: this tracker lives only in memory, so a resolver restart
    /// resets the high-water mark. Persistence is a Phase-2 concern
    /// (spec §5.4.3 treats this protection as best-effort).
    highest_serial: Arc<DashMap<(String, [u8; 32]), u64>>,
    /// Handle to the background sweep task; held so the task is cancelled when
    /// the resolver is dropped. Wired up by the background-sweep task (Task 17+).
    background_task: Mutex<Option<JoinHandle<()>>>,
}

impl DefaultEmailResolver {
    /// Create a new resolver with the given config and clients.
    ///
    /// A `TimeSource` is injected so tests can control time without mocking
    /// the whole resolver.
    pub fn new(
        dns: Arc<dyn DnsClient>,
        http: Arc<dyn HttpClient>,
        time: Arc<dyn TimeSource>,
        config: ResolverConfig,
    ) -> Self {
        let caches = Arc::new(ResolverCaches::new(config.cache_limits.clone()));
        Self {
            dns,
            http,
            caches,
            time,
            config,
            revocation_refresh_locks: Arc::new(dashmap::DashMap::new()),
            highest_serial: Arc::new(DashMap::new()),
            background_task: Mutex::new(None),
        }
    }

    /// Expose the cache collection (used by background sweep and tests).
    pub fn caches(&self) -> &Arc<ResolverCaches> {
        &self.caches
    }

    /// Expose the current config.
    pub fn config(&self) -> &ResolverConfig {
        &self.config
    }

    /// Expose the injected time source.
    pub fn time(&self) -> &Arc<dyn TimeSource> {
        &self.time
    }
}

// ─── Cold-path resolve impl (Task 15) ────────────────────────────────────────

#[async_trait]
impl EmailResolver for DefaultEmailResolver {
    async fn resolve(&self, local_part: &str, domain: &str) -> ResolveOutcome {
        let domain = domain.to_ascii_lowercase();
        let now = self.time.now();

        // 1. Get or fetch the domain record.
        let domain_record = match self.caches.get_master_key(&domain, now) {
            Some(rec) => rec,
            None => match crate::dns::fetch_domain_record(self.dns.as_ref(), &domain).await {
                Ok(rec) => {
                    self.caches.put_master_key(
                        &domain,
                        rec.clone(),
                        now.saturating_add(self.config.dns_ttl_default_secs),
                    );
                    rec
                }
                Err(DnsFetchError::NoRecord) => {
                    if self.caches.was_domain_seen_within(
                        &domain,
                        now,
                        self.config.soft_fail_window_secs,
                    ) {
                        return ResolveOutcome::Transient {
                            reason: "dns_no_record_soft_fail",
                        };
                    }
                    return ResolveOutcome::DomainDoesNotParticipate;
                }
                Err(DnsFetchError::UnsupportedVersion(_)) => {
                    return ResolveOutcome::DomainDoesNotParticipate;
                }
                Err(DnsFetchError::MultipleRecords) => {
                    return ResolveOutcome::Transient {
                        reason: "dns_multiple_records",
                    };
                }
                Err(DnsFetchError::Malformed(_)) => {
                    if self.caches.was_domain_seen_within(
                        &domain,
                        now,
                        self.config.soft_fail_window_secs,
                    ) {
                        return ResolveOutcome::Transient {
                            reason: "dns_malformed",
                        };
                    }
                    return ResolveOutcome::DomainDoesNotParticipate;
                }
                Err(DnsFetchError::Transient(_)) => {
                    return ResolveOutcome::Transient {
                        reason: "dns_error",
                    };
                }
            },
        };

        // 2. Compute cache key.
        let h = hashed_local_part(local_part, &domain_record.domain_salt);
        let cache_key = (domain.clone(), h);

        // 3. Negative-cache short-circuit.
        if self.caches.is_negative(&cache_key, now) {
            return ResolveOutcome::UserUnknown;
        }

        // 4. Ensure we have a usable revocation view (bootstrap closed).
        let revocations = match self.ensure_revocation_view(&domain, now).await {
            Ok(view) => view,
            Err(reason) => return ResolveOutcome::Transient { reason },
        };

        // 5. Claim cache hit?
        let claim = match self.caches.get_claim(&cache_key, now) {
            Some(cached) => cached,
            None => match crate::http::fetch_claim(self.http.as_ref(), &domain, &h).await {
                Ok(ClaimFetchResult::Found(c)) => c,
                Ok(ClaimFetchResult::NotFound) => {
                    self.caches.mark_negative(
                        cache_key.clone(),
                        now.saturating_add(self.config.negative_cache_secs),
                    );
                    return ResolveOutcome::UserUnknown;
                }
                Err(HttpFetchError::MalformedCbor(_)) => {
                    return ResolveOutcome::Transient {
                        reason: "claim_parse",
                    };
                }
                Err(HttpFetchError::Server(_)) | Err(HttpFetchError::Transport(_)) => {
                    return ResolveOutcome::Transient {
                        reason: "http_error",
                    };
                }
            },
        };

        // 6. Verify.
        match claim.verify_against(
            &domain,
            &domain_record,
            &revocations,
            now,
            self.config.clock_skew_tolerance_secs,
        ) {
            Ok(binding) => {
                let identity_hash: IdentityHash = binding.identity_hash;
                let signing_key_id = claim.cert.signing_key_id;
                let cert_valid_until = claim.cert.valid_until;
                let claim_expires_at = claim.payload.expires_at;
                let serial = binding.serial;
                // Serial rollback check: reject if this claim's serial is
                // strictly less than the highest serial we have ever seen
                // for this (domain, hashed_local_part) pair.
                //
                // Read-and-update is performed inside a single `entry` so the
                // shard lock keeps check-and-write atomic: two concurrent
                // resolves observing different serials cannot let the lower
                // one overwrite the higher after both passed the check.
                let mut rollback_prev: Option<u64> = None;
                self.highest_serial
                    .entry(cache_key.clone())
                    .and_modify(|stored| {
                        if serial < *stored {
                            rollback_prev = Some(*stored);
                        } else if serial > *stored {
                            *stored = serial;
                        }
                    })
                    .or_insert(serial);
                if let Some(prev) = rollback_prev {
                    warn!(%domain, local_part = %local_part, prev, new = serial,
                          "claim serial rollback detected");
                    return ResolveOutcome::Transient {
                        reason: "claim_serial_rollback",
                    };
                }
                let cert_clone = claim.cert.clone();
                self.caches
                    .put_claim(cache_key, claim, claim_expires_at, serial);
                self.caches.put_signing_key(
                    (domain.clone(), signing_key_id),
                    cert_clone,
                    cert_valid_until,
                );
                self.caches.mark_domain_seen(&domain, now);
                ResolveOutcome::Resolved(identity_hash)
            }
            Err(VerifyError::CertRevoked { .. }) => ResolveOutcome::Revoked,
            Err(e) => {
                warn!(error = %e, %domain, %local_part, "claim verification failed");
                self.caches.mark_negative(
                    (domain.clone(), h),
                    now.saturating_add(self.config.negative_cache_secs),
                );
                ResolveOutcome::Transient {
                    reason: verify_err_reason(&e),
                }
            }
        }
    }
}

fn verify_err_reason(e: &VerifyError) -> &'static str {
    match e {
        VerifyError::DomainMismatch => "claim_domain_mismatch",
        VerifyError::HashedLocalPartMismatch => "claim_lp_mismatch",
        VerifyError::CertSignatureInvalid => "cert_sig",
        VerifyError::ClaimSignatureInvalid => "claim_sig",
        VerifyError::CertNotYetValid { .. } => "cert_future",
        VerifyError::CertExpired { .. } => "cert_expired",
        VerifyError::CertRevoked { .. } => "cert_revoked",
        VerifyError::ClaimExpired { .. } => "claim_expired",
        VerifyError::UnsupportedVersion(_) => "unsupported_version",
        VerifyError::UnsupportedAlgorithm => "unsupported_alg",
        VerifyError::EncodingFailed => "encoding_failed",
    }
}

impl DefaultEmailResolver {
    /// Fetch or reuse the revocation view for `domain`. Spec §7.4:
    /// "No cached list ever + fetch fails" is `Transient` — fail
    /// closed on first-ever bootstrap. Subsequent refreshes leave the
    /// previous cache in place on failure.
    async fn ensure_revocation_view(
        &self,
        domain: &str,
        now: u64,
    ) -> Result<RevocationView, &'static str> {
        if let Some((view, expires_at, last_refreshed)) = self.caches.get_revocation(domain) {
            // 24h safety valve: if we haven't refreshed successfully in >24h, fail.
            if now.saturating_sub(last_refreshed) > self.config.revocation_safety_valve_secs {
                return Err("revocation_stale");
            }
            if now < expires_at {
                return Ok(view);
            }
            // Expired: try to refresh; on failure, keep serving previous.
            match self.refresh_revocation_view(domain, now).await {
                Ok(fresh) => Ok(fresh),
                Err(_) => Ok(view),
            }
        } else {
            // No prior cache — bootstrap-closed.
            self.refresh_revocation_view(domain, now).await
        }
    }

    async fn refresh_revocation_view(
        &self,
        domain: &str,
        now: u64,
    ) -> Result<RevocationView, &'static str> {
        // Per-domain concurrency cap. We clone the Arc out of the
        // DashMap entry RefMut and immediately drop the RefMut (end of
        // statement) to avoid holding the shard lock across the await.
        let sem: Arc<Semaphore> = self
            .revocation_refresh_locks
            .entry(domain.to_string())
            .or_insert_with(|| Arc::new(Semaphore::new(1)))
            .value()
            .clone();
        let _permit = sem
            .acquire_owned()
            .await
            .map_err(|_| "revocation_bootstrap_failed")?;

        // Get master pubkey for verifying the list.
        let Some(domain_record) = self.caches.get_master_key(domain, now) else {
            return Err("revocation_bootstrap_failed");
        };

        match crate::http::fetch_revocation_list(self.http.as_ref(), domain).await {
            Ok(RevocationFetchResult::Found(list)) => {
                if let Err(reason) = verify_revocation_list(&list, &domain_record) {
                    error!(%domain, %reason, "revocation list signature failed — potential attack");
                    return Err("revocation_bootstrap_failed");
                }
                let view = build_revocation_view(&list);
                self.caches.put_revocation(
                    domain,
                    view.clone(),
                    now.saturating_add(self.config.revocation_refresh_secs),
                    now,
                );
                Ok(view)
            }
            Ok(RevocationFetchResult::Empty) => {
                let view = RevocationView::empty();
                self.caches.put_revocation(
                    domain,
                    view.clone(),
                    now.saturating_add(self.config.revocation_refresh_secs),
                    now,
                );
                Ok(view)
            }
            Err(_) => Err("revocation_bootstrap_failed"),
        }
    }
}

fn verify_revocation_list(
    list: &RevocationList,
    domain_record: &DomainRecord,
) -> Result<(), &'static str> {
    let MasterPubkey::Ed25519(k) = domain_record.master_pubkey;
    let vk = VerifyingKey::from_bytes(&k).map_err(|_| "master_key_parse")?;
    let bytes = canonical_cbor(&list.signable()).map_err(|_| "encoding_failed")?;
    let Signature::Ed25519(sig_bytes) = list.master_signature;
    let sig = EdSignature::from_bytes(&sig_bytes);
    vk.verify(&bytes, &sig).map_err(|_| "master_sig")?;
    Ok(())
}

fn build_revocation_view(list: &RevocationList) -> RevocationView {
    let mut view = RevocationView::empty();
    for cert in &list.revoked_certs {
        view.insert(cert.signing_key_id, cert.valid_until);
    }
    view
}

impl DefaultEmailResolver {
    /// Spawn the background maintenance task. Safe to call multiple
    /// times — subsequent calls are no-ops. The task exits when the
    /// resolver is dropped (it holds a Weak reference to caches/time).
    pub async fn spawn_background_refresh(self: &Arc<Self>) {
        let mut guard = self.background_task.lock().await;
        if guard.is_some() {
            return;
        }

        let weak = Arc::downgrade(self);
        let handle = tokio::spawn(async move {
            let mut ticker = tokio::time::interval({
                // Upgrade once to read interval, then drop.
                let Some(this) = weak.upgrade() else {
                    return;
                };
                this.config.background_sweep_interval
            });
            ticker.tick().await; // immediate first tick — discard
            loop {
                ticker.tick().await;
                let Some(this) = weak.upgrade() else {
                    break;
                };
                let now = this.time.now();
                this.caches.sweep_expired(now);
                // Per-domain revocation refresh scheduling: collect
                // domains whose last_refreshed + refresh_secs < now.
                let to_refresh: Vec<String> = this
                    .caches
                    .revocation_refresh_candidates(now, this.config.revocation_refresh_secs);
                for domain in to_refresh {
                    let this2 = this.clone();
                    tokio::spawn(async move {
                        let _ = this2
                            .refresh_revocation_view(&domain, this2.time.now())
                            .await;
                    });
                }
            }
        });
        *guard = Some(handle);
    }
}
