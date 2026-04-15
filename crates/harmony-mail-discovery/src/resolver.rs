//! E-mail → identity resolution facade (spec §5.5 + §6).
//!
//! `EmailResolver` is the async trait that the SMTP server calls.
//! `DefaultEmailResolver` is the production implementation backed by a real
//! DNS client, a real HTTP client, and the four-level cache from `cache.rs`.
//! Cold-path fetch logic is implemented in Task 15.

// Task 14: skeleton only; Task 15 consumes these imports.
#![allow(unused_imports)]

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use harmony_identity::IdentityHash;
use tokio::sync::{Mutex, Semaphore};
use tokio::task::JoinHandle;
use tracing::{debug, error, info, warn};

use crate::cache::{CacheLimits, ResolverCaches, TimeSource};
use crate::claim::{hashed_local_part, DomainRecord, RevocationView, SignedClaim, VerifyError};
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
// Task 14: fields wired up in Task 15; suppress dead_code until then.
#[allow(dead_code)]
pub struct DefaultEmailResolver {
    dns: Arc<dyn DnsClient>,
    http: Arc<dyn HttpClient>,
    caches: Arc<ResolverCaches>,
    time: Arc<dyn TimeSource>,
    config: ResolverConfig,
    /// One `Semaphore(1)` per domain, used to serialise concurrent revocation
    /// refreshes for the same domain without blocking unrelated domains.
    revocation_refresh_locks: Arc<dashmap::DashMap<String, Arc<Semaphore>>>,
    /// Handle to the background sweep task; held so the task is cancelled when
    /// the resolver is dropped.
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
