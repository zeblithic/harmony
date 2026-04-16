//! Test fixtures exposed to integration tests and external crates
//! (via the `test-support` Cargo feature).
//!
//! This module exists so that `harmony-mail`'s integration tests can
//! construct valid `SignedClaim`s without duplicating the signing
//! ceremony. It is NOT compiled into release binaries unless the
//! consumer explicitly enables `test-support`.

// `canonical_cbor` panics only when the encoder fails on well-formed
// structs — that is the correct loud-failure behaviour in test fixtures.
// Mutex::lock().unwrap() is intentional in test-only fakes.
#![allow(clippy::expect_used)]
#![allow(clippy::unwrap_used)]

use ed25519_dalek::{Signer, SigningKey as EdSigningKey, VerifyingKey as EdVerifyingKey};
use rand_core::{CryptoRng, RngCore};

use crate::claim::{
    canonical_cbor, hashed_local_part, ClaimPayload, DomainRecord, MasterPubkey, RevocationList,
    Signature, SignatureAlg, SignedClaim, SigningKeyCert, SigningPubkey,
};

/// A master keypair bound to a test domain. Mirrors spec §2.1.
pub struct TestDomain {
    pub domain: String,
    pub salt: [u8; 16],
    pub master_sk: EdSigningKey,
    pub master_pk: EdVerifyingKey,
}

impl TestDomain {
    pub fn new(rng: &mut (impl RngCore + CryptoRng), domain: impl Into<String>) -> Self {
        let master_sk = EdSigningKey::generate(rng);
        let master_pk = master_sk.verifying_key();
        let mut salt = [0u8; 16];
        rng.fill_bytes(&mut salt);
        Self {
            domain: domain.into(),
            salt,
            master_sk,
            master_pk,
        }
    }

    pub fn record(&self) -> DomainRecord {
        DomainRecord {
            version: 1,
            master_pubkey: MasterPubkey::Ed25519(self.master_pk.to_bytes()),
            domain_salt: self.salt,
            alg: SignatureAlg::Ed25519,
        }
    }

    pub fn mint_signing_key(
        &self,
        rng: &mut (impl RngCore + CryptoRng),
        valid_from: u64,
        valid_until: u64,
    ) -> TestSigningKey {
        let signing_sk = EdSigningKey::generate(rng);
        let signing_pk = signing_sk.verifying_key();
        let mut key_id = [0u8; 8];
        rng.fill_bytes(&mut key_id);
        let cert = SigningKeyCert {
            version: 1,
            signing_key_id: key_id,
            signing_pubkey: SigningPubkey::Ed25519(signing_pk.to_bytes()),
            valid_from,
            valid_until,
            domain: self.domain.clone(),
            master_signature: Signature::Ed25519([0u8; 64]), // placeholder
        };
        let bytes = canonical_cbor(&cert.signable()).expect("encode cert signable");
        let sig = self.master_sk.sign(&bytes);
        let cert = SigningKeyCert {
            master_signature: Signature::Ed25519(sig.to_bytes()),
            ..cert
        };
        TestSigningKey { cert, signing_sk }
    }

    pub fn revocation_list(
        &self,
        issued_at: u64,
        revoked_certs: Vec<SigningKeyCert>,
    ) -> RevocationList {
        let list = RevocationList {
            version: 1,
            domain: self.domain.clone(),
            issued_at,
            revoked_certs,
            master_signature: Signature::Ed25519([0u8; 64]),
        };
        let bytes = canonical_cbor(&list.signable()).expect("encode rev-list");
        let sig = self.master_sk.sign(&bytes);
        RevocationList {
            master_signature: Signature::Ed25519(sig.to_bytes()),
            ..list
        }
    }
}

pub struct TestSigningKey {
    pub cert: SigningKeyCert,
    pub signing_sk: EdSigningKey,
}

impl TestSigningKey {
    /// Turn this cert into a revocation cert by setting `valid_until`
    /// to `revoked_at` and re-signing under the master key.
    pub fn revoke(mut self, master_sk: &EdSigningKey, revoked_at: u64) -> SigningKeyCert {
        self.cert.valid_until = revoked_at;
        self.cert.master_signature = Signature::Ed25519([0u8; 64]);
        let bytes = canonical_cbor(&self.cert.signable()).expect("encode cert signable");
        let sig = master_sk.sign(&bytes);
        self.cert.master_signature = Signature::Ed25519(sig.to_bytes());
        self.cert
    }
}

/// Fluent builder for `SignedClaim`. Defaults to a valid claim bound
/// to `TestDomain` + `TestSigningKey`; tests mutate fields then call
/// `build` (which re-signs under the signing key).
pub struct ClaimBuilder<'a> {
    domain: &'a TestDomain,
    sk: &'a TestSigningKey,
    email: String,
    identity_hash: [u8; 16],
    issued_at: u64,
    expires_at: u64,
    serial: u64,
    payload_version: u8,
    override_hashed_local_part: Option<[u8; 32]>,
    override_payload_domain: Option<String>,
    tamper_sig: bool,
}

impl<'a> ClaimBuilder<'a> {
    pub fn new(domain: &'a TestDomain, sk: &'a TestSigningKey, now: u64) -> Self {
        Self {
            domain,
            sk,
            email: format!("alice@{}", domain.domain),
            identity_hash: [0x11; 16],
            issued_at: now,
            expires_at: now + 7 * 86_400,
            serial: 1,
            payload_version: 1,
            override_hashed_local_part: None,
            override_payload_domain: None,
            tamper_sig: false,
        }
    }

    pub fn email(mut self, e: impl Into<String>) -> Self {
        self.email = e.into();
        self
    }
    pub fn identity_hash(mut self, h: [u8; 16]) -> Self {
        self.identity_hash = h;
        self
    }
    pub fn issued_at(mut self, t: u64) -> Self {
        self.issued_at = t;
        self
    }
    pub fn expires_at(mut self, t: u64) -> Self {
        self.expires_at = t;
        self
    }
    pub fn serial(mut self, s: u64) -> Self {
        self.serial = s;
        self
    }
    pub fn payload_version(mut self, v: u8) -> Self {
        self.payload_version = v;
        self
    }
    pub fn hashed_local_part_override(mut self, h: [u8; 32]) -> Self {
        self.override_hashed_local_part = Some(h);
        self
    }
    pub fn payload_domain_override(mut self, d: impl Into<String>) -> Self {
        self.override_payload_domain = Some(d.into());
        self
    }
    pub fn tamper_claim_signature(mut self) -> Self {
        self.tamper_sig = true;
        self
    }

    pub fn build(self) -> SignedClaim {
        let local_part = self.email.split('@').next().unwrap_or("").to_string();
        let h = self
            .override_hashed_local_part
            .unwrap_or_else(|| hashed_local_part(&local_part, &self.domain.salt));
        let payload = ClaimPayload {
            version: self.payload_version,
            domain: self
                .override_payload_domain
                .unwrap_or_else(|| self.domain.domain.clone()),
            hashed_local_part: h,
            email: self.email,
            identity_hash: self.identity_hash,
            issued_at: self.issued_at,
            expires_at: self.expires_at,
            serial: self.serial,
            signing_key_id: self.sk.cert.signing_key_id,
        };
        let bytes = canonical_cbor(&payload).expect("encode payload");
        let sig = self.sk.signing_sk.sign(&bytes);
        let mut sig_bytes = sig.to_bytes();
        if self.tamper_sig {
            sig_bytes[0] ^= 0xff;
        }
        SignedClaim {
            payload,
            cert: self.sk.cert.clone(),
            claim_signature: Signature::Ed25519(sig_bytes),
        }
    }
}

use std::sync::atomic::{AtomicU64, Ordering};

use crate::cache::TimeSource;

#[derive(Debug)]
pub struct FakeTimeSource(AtomicU64);

impl FakeTimeSource {
    pub fn new(start: u64) -> Self {
        Self(AtomicU64::new(start))
    }
    pub fn advance(&self, secs: u64) {
        self.0.fetch_add(secs, Ordering::Relaxed);
    }
    pub fn set(&self, secs: u64) {
        self.0.store(secs, Ordering::Relaxed);
    }
}

impl TimeSource for FakeTimeSource {
    fn now(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }
}

use std::collections::HashMap;
use std::sync::Mutex;

use async_trait::async_trait;

use crate::dns::{DnsClient, DnsError};
use crate::http::{HttpClient, HttpError, HttpResponse};

/// Scripted DNS responder. Keys are fully-qualified names
/// (e.g. `_harmony.q8.fyi`). Default response is `NoRecord`.
#[derive(Default)]
pub struct FakeDnsClient {
    answers: Mutex<HashMap<String, Result<Vec<String>, DnsError>>>,
    call_count: Mutex<usize>,
}

impl FakeDnsClient {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn set(&self, name: &str, answer: Result<Vec<String>, DnsError>) {
        self.answers
            .lock()
            .unwrap()
            .insert(name.to_string(), answer);
    }
    pub fn call_count(&self) -> usize {
        *self.call_count.lock().unwrap()
    }
}

#[async_trait]
impl DnsClient for FakeDnsClient {
    async fn fetch_txt(&self, name: &str) -> Result<Vec<String>, DnsError> {
        *self.call_count.lock().unwrap() += 1;
        self.answers
            .lock()
            .unwrap()
            .get(name)
            .cloned()
            .unwrap_or(Err(DnsError::NoRecord))
    }
}

/// Scripted HTTP responder keyed on exact URL. Default response is 404.
#[derive(Default)]
pub struct FakeHttpClient {
    answers: Mutex<HashMap<String, Result<HttpResponse, HttpError>>>,
    call_count: Mutex<usize>,
    calls_by_url: Mutex<HashMap<String, usize>>,
}

impl FakeHttpClient {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn set(&self, url: &str, answer: Result<HttpResponse, HttpError>) {
        self.answers.lock().unwrap().insert(url.to_string(), answer);
    }
    pub fn set_not_found(&self, url: &str) {
        self.set(
            url,
            Ok(HttpResponse {
                status: 404,
                body: vec![],
            }),
        );
    }
    pub fn call_count(&self) -> usize {
        *self.call_count.lock().unwrap()
    }
    pub fn call_count_for(&self, url: &str) -> usize {
        self.calls_by_url
            .lock()
            .unwrap()
            .get(url)
            .copied()
            .unwrap_or(0)
    }
}

#[async_trait]
impl HttpClient for FakeHttpClient {
    async fn get(&self, url: &str) -> Result<HttpResponse, HttpError> {
        *self.call_count.lock().unwrap() += 1;
        *self
            .calls_by_url
            .lock()
            .unwrap()
            .entry(url.to_string())
            .or_default() += 1;
        self.answers
            .lock()
            .unwrap()
            .get(url)
            .cloned()
            .unwrap_or(Ok(HttpResponse {
                status: 404,
                body: vec![],
            }))
    }
}
