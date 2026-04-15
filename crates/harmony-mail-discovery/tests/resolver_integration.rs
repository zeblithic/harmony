//! Component tests for `DefaultEmailResolver` with injected fakes.
//! See spec §8.2.

// Integration tests under tests/ are compiled as separate crates, so the
// crate-level `cfg_attr(test, allow(unwrap_used))` in lib.rs does not
// reach them. Allow it here per file.
#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use harmony_mail_discovery::claim::{canonical_cbor, MasterPubkey};
use harmony_mail_discovery::dns::DnsError;
use harmony_mail_discovery::http::HttpResponse;
use harmony_mail_discovery::resolver::{
    DefaultEmailResolver, EmailResolver, ResolveOutcome, ResolverConfig,
};
use harmony_mail_discovery::test_support::{
    ClaimBuilder, FakeDnsClient, FakeHttpClient, FakeTimeSource, TestDomain,
};
use rand_core::OsRng;

const NOW: u64 = 2_000_000_000;

fn build_dns_txt(d: &TestDomain) -> String {
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use base64::Engine;
    let MasterPubkey::Ed25519(k) = d.record().master_pubkey;
    let k = URL_SAFE_NO_PAD.encode(k);
    let salt = URL_SAFE_NO_PAD.encode(d.salt);
    format!("v=harmony1; k={k}; salt={salt}; alg=ed25519")
}

fn setup() -> (
    TestDomain,
    Arc<FakeDnsClient>,
    Arc<FakeHttpClient>,
    Arc<FakeTimeSource>,
    DefaultEmailResolver,
) {
    let mut rng = OsRng;
    let d = TestDomain::new(&mut rng, "q8.fyi");
    let dns = Arc::new(FakeDnsClient::new());
    let http = Arc::new(FakeHttpClient::new());
    let time = Arc::new(FakeTimeSource::new(NOW));
    let resolver = DefaultEmailResolver::new(
        dns.clone(),
        http.clone(),
        time.clone(),
        ResolverConfig::default(),
    );
    (d, dns, http, time, resolver)
}

#[tokio::test]
async fn cold_path_resolves_and_caches() {
    let (d, dns, http, _time, resolver) = setup();
    let mut rng = OsRng;
    let sk = d.mint_signing_key(&mut rng, NOW - 100, NOW + 90 * 86_400);
    let claim = ClaimBuilder::new(&d, &sk, NOW)
        .identity_hash([0x42; 16])
        .build();

    dns.set("_harmony.q8.fyi", Ok(vec![build_dns_txt(&d)]));
    let claim_url =
        harmony_mail_discovery::http::claim_url(&d.domain, &claim.payload.hashed_local_part);
    http.set(
        &claim_url,
        Ok(HttpResponse {
            status: 200,
            body: canonical_cbor(&claim).unwrap(),
        }),
    );
    http.set_not_found(&harmony_mail_discovery::http::revocation_url(&d.domain));

    let out = resolver.resolve("alice", "q8.fyi").await;
    // `IdentityHash` is a type alias for `[u8; 16]` (see harmony-identity),
    // so destructuring as a tuple struct does not apply.
    let bytes = match &out {
        ResolveOutcome::Resolved(h) => *h,
        other => panic!("expected Resolved, got {other:?}"),
    };
    assert_eq!(bytes, [0x42; 16]);

    // Subsequent call: zero network.
    // On the first call the resolver populated (a) the master-key cache,
    // (b) the revocation cache (from the 404 → authoritative empty), and
    // (c) the claim cache. All three remain fresh, so the second resolve
    // must not touch either client.
    let pre_dns = dns.call_count();
    let pre_http = http.call_count();
    let _ = resolver.resolve("alice", "q8.fyi").await;
    assert_eq!(dns.call_count(), pre_dns, "no new DNS calls on cache hit");
    assert_eq!(
        http.call_count(),
        pre_http,
        "no new HTTP calls on cache hit"
    );
}

// Coverage gap tracked for later: `DnsFetchError::UnsupportedVersion` and
// `DnsFetchError::MultipleRecords` map to `DomainDoesNotParticipate` and
// `Transient("dns_multiple_records")` respectively in resolver.rs but are
// only exercised via dns.rs parser unit tests today. Add resolver-layer
// assertions when Task 18 expands soft-fail / safety-valve coverage.

#[tokio::test]
async fn dns_nxdomain_returns_domain_does_not_participate() {
    let (_d, dns, _http, _time, resolver) = setup();
    dns.set("_harmony.q8.fyi", Err(DnsError::NoRecord));
    assert_eq!(
        resolver.resolve("alice", "q8.fyi").await,
        ResolveOutcome::DomainDoesNotParticipate,
    );
}

#[tokio::test]
async fn dns_nxdomain_within_soft_fail_window_returns_transient() {
    let (d, dns, http, time, resolver) = setup();
    let mut rng = OsRng;
    let sk = d.mint_signing_key(&mut rng, NOW - 100, NOW + 90 * 86_400);
    let claim = ClaimBuilder::new(&d, &sk, NOW).build();
    // First, succeed once so the domain is "seen".
    dns.set("_harmony.q8.fyi", Ok(vec![build_dns_txt(&d)]));
    http.set(
        &harmony_mail_discovery::http::claim_url(&d.domain, &claim.payload.hashed_local_part),
        Ok(HttpResponse {
            status: 200,
            body: canonical_cbor(&claim).unwrap(),
        }),
    );
    http.set_not_found(&harmony_mail_discovery::http::revocation_url(&d.domain));
    assert!(matches!(
        resolver.resolve("alice", "q8.fyi").await,
        ResolveOutcome::Resolved(_)
    ));

    // Now expire the master-key cache and flip DNS to NXDOMAIN.
    time.advance(2 * 3600); // past 1h DNS TTL
    dns.set("_harmony.q8.fyi", Err(DnsError::NoRecord));
    match resolver.resolve("alice", "q8.fyi").await {
        ResolveOutcome::Transient { reason } => {
            assert_eq!(reason, "dns_no_record_soft_fail");
        }
        other => panic!("expected Transient soft-fail, got {other:?}"),
    }
}

#[tokio::test]
async fn dns_timeout_returns_transient() {
    let (_d, dns, _http, _time, resolver) = setup();
    dns.set(
        "_harmony.q8.fyi",
        Err(DnsError::Transient("timeout".into())),
    );
    match resolver.resolve("alice", "q8.fyi").await {
        ResolveOutcome::Transient { reason } => assert_eq!(reason, "dns_error"),
        other => panic!("expected Transient, got {other:?}"),
    }
}

#[tokio::test]
async fn http_404_returns_user_unknown_and_is_negative_cached() {
    let (d, dns, http, _time, resolver) = setup();
    dns.set("_harmony.q8.fyi", Ok(vec![build_dns_txt(&d)]));
    // Claim 404 — but we still need revocation list to bootstrap.
    http.set_not_found(&harmony_mail_discovery::http::revocation_url(&d.domain));
    // The claim URL is for "alice" so we don't set it — default is 404.
    let out = resolver.resolve("alice", "q8.fyi").await;
    assert_eq!(out, ResolveOutcome::UserUnknown);

    let pre_http = http.call_count();
    let _ = resolver.resolve("alice", "q8.fyi").await;
    // Re-query hits negative cache, no new HTTP call for the claim URL.
    assert_eq!(http.call_count(), pre_http, "negative cache short-circuits");
}

#[tokio::test]
async fn http_500_returns_transient() {
    let (d, dns, http, _time, resolver) = setup();
    dns.set("_harmony.q8.fyi", Ok(vec![build_dns_txt(&d)]));
    http.set_not_found(&harmony_mail_discovery::http::revocation_url(&d.domain));
    // FakeHttpClient matches on exact URL; compute the exact hash for alice.
    let h = harmony_mail_discovery::claim::hashed_local_part("alice", &d.salt);
    http.set(
        &harmony_mail_discovery::http::claim_url(&d.domain, &h),
        Ok(HttpResponse {
            status: 500,
            body: vec![],
        }),
    );
    match resolver.resolve("alice", "q8.fyi").await {
        ResolveOutcome::Transient { reason } => assert_eq!(reason, "http_error"),
        other => panic!("expected Transient, got {other:?}"),
    }
}

#[tokio::test]
async fn http_malformed_cbor_returns_transient() {
    let (d, dns, http, _time, resolver) = setup();
    dns.set("_harmony.q8.fyi", Ok(vec![build_dns_txt(&d)]));
    http.set_not_found(&harmony_mail_discovery::http::revocation_url(&d.domain));
    let h = harmony_mail_discovery::claim::hashed_local_part("alice", &d.salt);
    http.set(
        &harmony_mail_discovery::http::claim_url(&d.domain, &h),
        Ok(HttpResponse {
            status: 200,
            body: vec![0xff, 0xff, 0xff],
        }),
    );
    match resolver.resolve("alice", "q8.fyi").await {
        ResolveOutcome::Transient { reason } => assert_eq!(reason, "claim_parse"),
        other => panic!("expected Transient, got {other:?}"),
    }
}

#[tokio::test]
async fn revocation_bootstrap_failure_fails_closed() {
    let (d, dns, http, _time, resolver) = setup();
    let mut rng = OsRng;
    let sk = d.mint_signing_key(&mut rng, NOW - 100, NOW + 90 * 86_400);
    let claim = ClaimBuilder::new(&d, &sk, NOW).build();
    dns.set("_harmony.q8.fyi", Ok(vec![build_dns_txt(&d)]));
    http.set(
        &harmony_mail_discovery::http::claim_url(&d.domain, &claim.payload.hashed_local_part),
        Ok(HttpResponse {
            status: 200,
            body: canonical_cbor(&claim).unwrap(),
        }),
    );
    // The claim response above is scripted for completeness, but should
    // never be consumed: `ensure_revocation_view` runs before claim fetch,
    // and it fails closed on a 5xx with no prior cache.
    http.set(
        &harmony_mail_discovery::http::revocation_url(&d.domain),
        Ok(HttpResponse {
            status: 500,
            body: vec![],
        }),
    );
    match resolver.resolve("alice", "q8.fyi").await {
        ResolveOutcome::Transient { reason } => assert_eq!(reason, "revocation_bootstrap_failed"),
        other => panic!("expected bootstrap-closed Transient, got {other:?}"),
    }
}
