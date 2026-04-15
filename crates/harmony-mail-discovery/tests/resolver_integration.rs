//! Component tests for `DefaultEmailResolver` with injected fakes.
//! See spec §8.2.

// Integration tests under tests/ are compiled as separate crates, so the
// crate-level `cfg_attr(test, allow(unwrap_used))` in lib.rs does not
// reach them. Allow it here per file.
#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use harmony_mail_discovery::claim::{canonical_cbor, MasterPubkey};
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
