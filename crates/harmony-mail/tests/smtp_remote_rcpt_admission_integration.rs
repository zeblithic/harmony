//! End-to-end integration test for ZEB-120 RCPT admission via EmailResolver.
//!
//! Drives the full SMTP state machine flow: Connected -> EHLO -> MAIL FROM ->
//! RCPT TO (non-local domain) -> DATA -> DataComplete.
//!
//! The RCPT TO triggers ResolveHarmonyAddress, which calls a FakeEmailResolver
//! that resolves "alice@remote.example" to Bob's address hash. That hash feeds
//! into a StubResolver (RecipientResolver) that returns Bob's public Identity
//! for sealing. The sealed envelope is published via Zenoh and received by a
//! subscriber on session_b. The envelope is opened with Bob's private key and
//! the plaintext is verified.

use std::sync::Arc;
use std::time::Duration;

use harmony_identity::{Identity, IdentityHash, PrivateIdentity};
use harmony_mail::imap_store::ImapStore;
use harmony_mail::mailbox_manager::ZenohPublisher;
use harmony_mail::message::HarmonyMessage;
use harmony_mail::remote_delivery::RecipientResolver;
use harmony_mail::server::process_async_actions;
use harmony_mail::smtp::{SmtpCommand, SmtpConfig, SmtpEvent, SmtpSession};
use harmony_mail::spam::SpfResult;
use harmony_zenoh::envelope::{HarmonyEnvelope, MessageType};
use harmony_zenoh::namespace::msg;
use rand_core::OsRng;
use tokio_util::sync::CancellationToken;

// ── FakeEmailResolver ───────────────────────────────────────────────────

/// Resolves "alice@remote.example" to a known address hash (bob's).
struct FakeEmailResolver {
    target_hash: IdentityHash,
}

#[async_trait::async_trait]
impl harmony_mail_discovery::resolver::EmailResolver for FakeEmailResolver {
    async fn resolve(
        &self,
        local_part: &str,
        domain: &str,
    ) -> harmony_mail_discovery::resolver::ResolveOutcome {
        if local_part == "alice" && domain == "remote.example" {
            harmony_mail_discovery::resolver::ResolveOutcome::Resolved(self.target_hash)
        } else {
            harmony_mail_discovery::resolver::ResolveOutcome::UserUnknown
        }
    }
}

// ── StubResolver (RecipientResolver) ────────────────────────────────────

/// Maps a known address hash to Bob's public Identity.
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
async fn smtp_rcpt_to_remote_domain_resolves_seals_publishes() {
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

    // Bounded readiness probe — ensures Zenoh peer convergence before the
    // real publish. Same pattern as smtp_remote_delivery_integration.rs.
    let probe_key = format!(
        "harmony/msg/v1/unicast/_readiness_probe_zeb120_{}",
        hex::encode(bob_hash),
    );
    let probe_sub = session_b
        .declare_subscriber(&probe_key)
        .await
        .expect("probe subscriber declare");

    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    loop {
        if std::time::Instant::now() >= deadline {
            panic!("peer discovery did not converge within 5s");
        }
        session_a
            .put(&probe_key, b"probe".to_vec())
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
        ImapStore::open(&tmp.path().join("a.db")).expect("open imap store"),
    );

    // ── Resolvers ─────────────────────────────────────────────────
    let email_resolver: Arc<dyn harmony_mail_discovery::resolver::EmailResolver> =
        Arc::new(FakeEmailResolver {
            target_hash: bob_hash,
        });
    let recipient_resolver: Arc<dyn RecipientResolver> = Arc::new(StubResolver {
        bob_hash,
        bob_identity: bob_pub.clone(),
    });

    // ── SMTP config: local domain is "local.example" — distinct from
    //    "remote.example" so RCPT TO goes through EmailResolver ───
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

    // Shared parameters for process_async_actions calls
    let local_domain = "local.example";

    // ── 1. Connected ──────────────────────────────────────────────
    let actions = session.handle(SmtpEvent::Connected {
        peer_ip: "127.0.0.1".parse().unwrap(),
        tls: false,
    });
    // SendResponse(220, ...) — write to buffer for completeness
    process_async_actions(
        &actions,
        &mut session,
        &mut writer,
        &imap,
        &None,
        local_domain,
        tmp.path(),
        &mut spf_result,
        1000,
        &None,
        &Some(Arc::clone(&publisher)),
        &Some(Arc::clone(&gateway_a_priv)),
        &Some(Arc::clone(&recipient_resolver)),
        &Some(Arc::clone(&email_resolver)),
    )
    .await
    .expect("process Connected actions");

    // ── 2. EHLO ───────────────────────────────────────────────────
    let actions = session.handle(SmtpEvent::Command(SmtpCommand::Ehlo {
        domain: "test.example".to_string(),
    }));
    process_async_actions(
        &actions,
        &mut session,
        &mut writer,
        &imap,
        &None,
        local_domain,
        tmp.path(),
        &mut spf_result,
        1000,
        &None,
        &Some(Arc::clone(&publisher)),
        &Some(Arc::clone(&gateway_a_priv)),
        &Some(Arc::clone(&recipient_resolver)),
        &Some(Arc::clone(&email_resolver)),
    )
    .await
    .expect("process EHLO actions");

    // ── 3. MAIL FROM ──────────────────────────────────────────────
    let actions = session.handle(SmtpEvent::Command(SmtpCommand::MailFrom {
        address: "sender@test.example".to_string(),
    }));
    // Contains SendResponse(250) + CheckSpf (no-op without authenticator)
    process_async_actions(
        &actions,
        &mut session,
        &mut writer,
        &imap,
        &None,
        local_domain,
        tmp.path(),
        &mut spf_result,
        1000,
        &None,
        &Some(Arc::clone(&publisher)),
        &Some(Arc::clone(&gateway_a_priv)),
        &Some(Arc::clone(&recipient_resolver)),
        &Some(Arc::clone(&email_resolver)),
    )
    .await
    .expect("process MAIL FROM actions");

    // ── 4. RCPT TO (non-local domain) ─────────────────────────────
    // This emits ResolveHarmonyAddress { local_part: "alice", domain: "remote.example" }.
    // process_async_actions calls FakeEmailResolver -> resolves to bob_hash ->
    // feeds HarmonyResolved back to session -> session transitions to RcptToReceived.
    let actions = session.handle(SmtpEvent::Command(SmtpCommand::RcptTo {
        address: "alice@remote.example".to_string(),
    }));
    process_async_actions(
        &actions,
        &mut session,
        &mut writer,
        &imap,
        &None,
        local_domain,
        tmp.path(),
        &mut spf_result,
        1000,
        &None,
        &Some(Arc::clone(&publisher)),
        &Some(Arc::clone(&gateway_a_priv)),
        &Some(Arc::clone(&recipient_resolver)),
        &Some(Arc::clone(&email_resolver)),
    )
    .await
    .expect("process RCPT TO actions");

    // Verify session accepted the recipient
    assert_eq!(
        session.state,
        harmony_mail::smtp::SmtpState::RcptToReceived,
        "RCPT TO should have been admitted after EmailResolver returned bob_hash"
    );

    // Check that the writer received the 250 OK from the HarmonyResolved callback
    let writer_str = String::from_utf8_lossy(&writer);
    assert!(
        writer_str.contains("250 OK"),
        "writer should contain 250 OK from RCPT admission: {writer_str}"
    );

    // ── 5. DATA ───────────────────────────────────────────────────
    let actions = session.handle(SmtpEvent::Command(SmtpCommand::Data));
    process_async_actions(
        &actions,
        &mut session,
        &mut writer,
        &imap,
        &None,
        local_domain,
        tmp.path(),
        &mut spf_result,
        1000,
        &None,
        &Some(Arc::clone(&publisher)),
        &Some(Arc::clone(&gateway_a_priv)),
        &Some(Arc::clone(&recipient_resolver)),
        &Some(Arc::clone(&email_resolver)),
    )
    .await
    .expect("process DATA actions");

    // ── 6. DataComplete — triggers DeliverToHarmony ───────────────
    let rfc822 = b"From: sender@test.example\r\n\
                   To: alice@remote.example\r\n\
                   Subject: RCPT admission test\r\n\
                   Date: Wed, 15 Apr 2026 12:00:00 +0000\r\n\
                   Message-ID: <test-zeb120-rcpt-admission@test.example>\r\n\
                   \r\n\
                   This message was admitted via EmailResolver.\r\n";

    let actions = session.handle(SmtpEvent::DataComplete(rfc822.to_vec()));
    process_async_actions(
        &actions,
        &mut session,
        &mut writer,
        &imap,
        &None,
        local_domain,
        tmp.path(),
        &mut spf_result,
        1000,
        &None,
        &Some(Arc::clone(&publisher)),
        &Some(Arc::clone(&gateway_a_priv)),
        &Some(Arc::clone(&recipient_resolver)),
        &Some(Arc::clone(&email_resolver)),
    )
    .await
    .expect("process DataComplete actions");

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
    assert_eq!(recovered.subject, "RCPT admission test");
    assert_eq!(
        opened.sender_address,
        gateway_a_priv.public_identity().address_hash,
    );

    // Envelope message type should be Put.
    assert_eq!(
        opened.msg_type,
        MessageType::Put,
        "envelope msg_type should be Put for mail",
    );

    // Body survived translation.
    assert!(
        recovered.body.contains("EmailResolver"),
        "recovered HarmonyMessage body missing expected text: {:?}",
        recovered.body,
    );

    // Recipient list preserved through the full pipeline.
    let expected_to_hash = {
        let h = blake3::hash(b"alice@remote.example");
        let mut out = [0u8; harmony_mail::message::ADDRESS_HASH_LEN];
        out.copy_from_slice(&h.as_bytes()[..harmony_mail::message::ADDRESS_HASH_LEN]);
        out
    };
    assert!(
        recovered
            .recipients
            .iter()
            .any(|r| r.address_hash == expected_to_hash),
        "recovered HarmonyMessage recipients missing alice@remote.example hash: {:?}",
        recovered.recipients,
    );

    // ── Clean up: cancel publisher + drop sessions ────────────────
    cancel.cancel();
    drop(sub);
    drop(session_b);
    drop(session_a);
}
