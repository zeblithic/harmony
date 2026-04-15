//! End-to-end integration test for ZEB-113 online-recipient remote delivery.
//!
//! Harness:
//!   - Gateway A: full SMTP server path (ZenohPublisher writing to session_a,
//!     plus a RecipientResolver stub that returns Bob's Identity).
//!   - Gateway B: independent Zenoh session; subscribes to
//!     harmony/msg/v1/unicast/{bob_hash}. Holds Bob's PrivateIdentity.
//!
//! Deliver an SMTP message addressed to bob@remote.example via Gateway A's
//! process_async_actions. Gateway B's subscriber receives the sealed
//! envelope; HarmonyEnvelope::open recovers the plaintext; we parse the
//! plaintext back into a HarmonyMessage and assert the subject matches.

use std::sync::Arc;
use std::time::Duration;

use harmony_identity::{Identity, IdentityHash, PrivateIdentity};
use harmony_mail::mailbox_manager::ZenohPublisher;
use harmony_mail::message::HarmonyMessage;
use harmony_mail::remote_delivery::RecipientResolver;
use harmony_mail::server::process_async_actions;
use harmony_mail::smtp::{SmtpAction, SmtpConfig, SmtpSession};
use harmony_mail::spam::SpfResult;
use harmony_zenoh::envelope::{HarmonyEnvelope, MessageType};
use harmony_zenoh::namespace::msg;
use rand_core::OsRng;
use tokio_util::sync::CancellationToken;

struct StubResolver {
    bob_hash: IdentityHash,
    bob_identity: Identity,
}

impl RecipientResolver for StubResolver {
    fn resolve(&self, address_hash: &IdentityHash) -> Option<Identity> {
        if address_hash == &self.bob_hash {
            Some(self.bob_identity.clone())
        } else {
            None
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn smtp_remote_delivery_round_trips_through_zenoh_to_recipient() {
    let mut rng = OsRng;
    let gateway_a_priv = Arc::new(PrivateIdentity::generate(&mut rng));
    let bob_priv = PrivateIdentity::generate(&mut rng);
    let bob_pub = bob_priv.public_identity().clone();
    let bob_hash = bob_pub.address_hash;

    // ── Two independent Zenoh sessions ────────────────────────────
    let session_a = zenoh::open(zenoh::Config::default())
        .await
        .expect("open session_a");
    let session_b = zenoh::open(zenoh::Config::default())
        .await
        .expect("open session_b");

    // Subscribe Bob's gateway to the unicast key BEFORE we publish.
    let sub_key = msg::unicast_key(&hex::encode(bob_hash));
    let sub = session_b
        .declare_subscriber(&sub_key)
        .await
        .expect("declare subscriber");

    // Bounded readiness probe — replaces a blind 250ms sleep so the test
    // fails fast on real propagation issues and doesn't flake on slow
    // runners. Publish a throwaway byte on a probe keyspace from session_a;
    // session_b's throwaway subscriber acknowledges first receipt.
    //
    // 5s total budget. Empirically the first receipt lands in <50ms on a
    // warm runner and <400ms cold.
    //
    // The probe_key is a distinct literal key from the real `sub_key`
    // above (non-wildcard exact-key matching in Zenoh 1.x), so the real
    // subscriber will not observe probe traffic.
    let probe_key = "harmony/msg/v1/unicast/_readiness_probe_zeb113";
    let probe_sub = session_b
        .declare_subscriber(probe_key)
        .await
        .expect("probe subscriber declare");

    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    loop {
        if std::time::Instant::now() >= deadline {
            panic!("peer discovery did not converge within 5s");
        }
        session_a
            .put(probe_key, b"probe".to_vec())
            .await
            .expect("probe put");
        match tokio::time::timeout(Duration::from_millis(100), probe_sub.recv_async()).await {
            Ok(Ok(_)) => break,
            _ => continue,
        }
    }
    drop(probe_sub);

    // ── ZenohPublisher against session_a (real, not inert) ────────
    let cancel = CancellationToken::new();
    let publisher = Arc::new(ZenohPublisher::new(session_a.clone(), cancel.clone()));

    // ── IMAP store with NO local user for bob ──────────────────────
    let tmp = tempfile::tempdir().expect("tempdir");
    let imap = Arc::new(
        harmony_mail::imap_store::ImapStore::open(&tmp.path().join("a.db"))
            .expect("open imap store"),
    );

    // ── Resolver maps bob_hash → bob's public Identity ────────────
    let resolver: Arc<dyn RecipientResolver> = Arc::new(StubResolver {
        bob_hash,
        bob_identity: bob_pub.clone(),
    });

    // ── RFC 5322 message. Match the field style of the existing
    //    delivers_remote_warn_and_skip_when_resolver_returns_none test. ──
    let rfc822 = b"From: alice@local.example\r\n\
                   To: bob@remote.example\r\n\
                   Subject: hello from A\r\n\
                   Date: Tue, 15 Apr 2026 12:00:00 +0000\r\n\
                   Message-ID: <test-zeb113-integration@local.example>\r\n\
                   \r\n\
                   Hello Bob, this is Alice.\r\n";

    let actions = vec![SmtpAction::DeliverToHarmony {
        recipients: vec![bob_hash],
        data: rfc822.to_vec(),
    }];

    let smtp_config = SmtpConfig {
        domain: "local.example".to_string(),
        mx_host: "mail.local.example".to_string(),
        max_message_size: 10 * 1024 * 1024,
        max_recipients: 100,
        tls_available: false,
    };

    let mut session = SmtpSession::new(smtp_config);
    let mut writer = Vec::<u8>::new();
    let mut spf_result = SpfResult::None;

    process_async_actions(
        &actions,
        &mut session,
        &mut writer,
        &imap,
        &None,
        "local.example",
        tmp.path(),
        &mut spf_result,
        1000, // reject_threshold high — no spam rejection
        &None,
        &Some(Arc::clone(&publisher)),
        &Some(Arc::clone(&gateway_a_priv)),
        &Some(resolver),
    )
    .await
    .expect("process_async_actions failed");

    // ── Gateway B should receive a sealed envelope on the unicast key ──
    let sample = tokio::time::timeout(Duration::from_secs(3), sub.recv_async())
        .await
        .expect("subscriber timeout — no publish observed within 3s")
        .expect("subscriber channel closed");
    let sealed_bytes = sample.payload().to_bytes().to_vec();

    // ── Open the envelope with Bob's private identity ──────────────
    let opened = HarmonyEnvelope::open(
        &bob_priv,
        gateway_a_priv.public_identity(),
        &sealed_bytes,
    )
    .expect("HarmonyEnvelope::open failed");

    // ── The recovered plaintext is the serialized HarmonyMessage ───
    let recovered = HarmonyMessage::from_bytes(&opened.plaintext)
        .expect("recovered bytes should parse as a HarmonyMessage");
    assert_eq!(recovered.subject, "hello from A");
    assert_eq!(
        opened.sender_address,
        gateway_a_priv.public_identity().address_hash,
    );

    // Lock in the MessageType decision from the plan (Design Decision #2).
    assert_eq!(
        opened.msg_type,
        MessageType::Put,
        "envelope msg_type should be Put for mail",
    );

    // Body survived translation (not just headers).
    assert!(
        recovered.body.contains("Hello Bob"),
        "recovered HarmonyMessage body missing expected text: {:?}",
        recovered.body,
    );

    // Recipient list preserved through serialize/seal/open/deserialize.
    // The recipients Vec inside the HarmonyMessage body is populated by
    // translate.rs from the RFC 5322 `To:` header, where each address
    // is hashed via blake3("bob@remote.example") truncated to 16 bytes
    // — NOT the random `bob_hash` identity used for envelope sealing.
    let expected_to_hash = {
        let h = blake3::hash(b"bob@remote.example");
        let mut out = [0u8; 16];
        out.copy_from_slice(&h.as_bytes()[..16]);
        out
    };
    assert!(
        recovered
            .recipients
            .iter()
            .any(|r| r.address_hash == expected_to_hash),
        "recovered HarmonyMessage recipients missing bob@remote.example hash: {:?}",
        recovered.recipients,
    );

    // ── Clean up: cancel publisher + drop sessions ────────────────
    cancel.cancel();
    drop(sub);
    drop(session_b);
    drop(session_a);
}
