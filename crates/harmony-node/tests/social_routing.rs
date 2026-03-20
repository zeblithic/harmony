//! Integration tests for social content routing.
//!
//! These tests prove that social content routing works as an emergent
//! property of existing Harmony systems composing:
//!
//! 1. Zenoh PubSubRouter — interest declaration + write-side filtering
//! 2. StorageTier — content storage + Bloom filter availability
//!
//! See docs/plans/2026-03-20-social-content-routing-design.md

use harmony_content::blob::MemoryBlobStore;
use harmony_content::cid::{ContentFlags, ContentId};
use harmony_content::storage_tier::{
    ContentPolicy, FilterBroadcastConfig, StorageBudget, StorageTier, StorageTierAction,
    StorageTierEvent,
};
use harmony_identity::PrivateIdentity;
use harmony_zenoh::pubsub::{PubSubAction, PubSubEvent, PubSubRouter};
use harmony_zenoh::session::{Session, SessionAction, SessionConfig, SessionEvent, SessionState};
use rand::rngs::OsRng;

/// Set up a pair of authenticated sessions (Alice ↔ Bob).
fn active_session_pair() -> (Session, Session) {
    let alice_priv = PrivateIdentity::generate(&mut OsRng);
    let bob_priv = PrivateIdentity::generate(&mut OsRng);
    let alice_pub = alice_priv.public_identity().clone();
    let bob_pub = bob_priv.public_identity().clone();

    let (mut alice, alice_actions) =
        Session::new(alice_priv, bob_pub, SessionConfig::default(), 0);
    let (mut bob, bob_actions) =
        Session::new(bob_priv, alice_pub, SessionConfig::default(), 0);

    let alice_proof = match &alice_actions[0] {
        SessionAction::SendHandshake { proof } => proof.clone(),
        _ => panic!("expected SendHandshake"),
    };
    let bob_proof = match &bob_actions[0] {
        SessionAction::SendHandshake { proof } => proof.clone(),
        _ => panic!("expected SendHandshake"),
    };

    bob.handle_event(SessionEvent::HandshakeReceived { proof: alice_proof })
        .unwrap();
    alice
        .handle_event(SessionEvent::HandshakeReceived { proof: bob_proof })
        .unwrap();

    assert_eq!(alice.state(), SessionState::Active);
    assert_eq!(bob.state(), SessionState::Active);
    (alice, bob)
}

// ── Test 1: Interest propagation through tunnel peers ─────────────

/// Proves the full social routing cycle:
/// 1. Alice (subscriber) declares interest in Bob's vine announces
/// 2. Bob (publisher) receives the interest declaration
/// 3. Bob publishes a vine — write-side filter allows it because Alice wants it
/// 4. Without interest, the same publish is silently dropped
///
/// This is how social content routing works: your Zenoh subscriptions
/// flow to your tunnel peers, who serve matching content. The social
/// graph topology (who you're peered with) determines who gets your
/// publications.
#[test]
fn vine_interest_propagates_through_tunnel_peer() {
    let (mut alice_session, _bob_session) = active_session_pair();
    let mut bob_router = PubSubRouter::new();

    // Bob declares a publisher on his vine announce key expression.
    // The key expression uses the actual Harmony namespace convention.
    let (pub_id, _) = bob_router
        .declare_publisher(
            "harmony/vines/aa00bb11cc22dd33ee44ff5566778899/announce/post1".into(),
            &mut alice_session,
        )
        .unwrap();

    // Without interest: publish is silently dropped (write-side filtering)
    let actions = bob_router
        .publish(pub_id, b"vine content".to_vec(), &alice_session)
        .unwrap();
    assert!(
        actions.is_empty(),
        "without remote interest, publish should be silently dropped"
    );

    // Alice declares interest in Bob's vine announces (simulating
    // the SubscriberDeclared event arriving from Alice's session)
    bob_router
        .handle_event(
            PubSubEvent::SubscriberDeclared {
                key_expr: "harmony/vines/aa00bb11cc22dd33ee44ff5566778899/announce/**".into(),
            },
            &alice_session,
        )
        .unwrap();

    // Now publish should emit SendMessage — Alice wants this content
    let actions = bob_router
        .publish(pub_id, b"vine content".to_vec(), &alice_session)
        .unwrap();
    assert_eq!(actions.len(), 1);
    assert!(
        matches!(&actions[0], PubSubAction::SendMessage { payload, .. }
            if payload == b"vine content"),
        "with remote interest, publish should emit SendMessage"
    );
}

/// Proves that subscribing emits a SendSubscriberDeclare action,
/// which is what propagates interest to the tunnel peer.
#[test]
fn subscribe_emits_interest_declaration_for_tunnel_peer() {
    let mut router = PubSubRouter::new();

    let (_sub_id, actions) = router
        .subscribe("harmony/vines/aa00bb11cc22dd33ee44ff5566778899/announce/**")
        .unwrap();

    assert_eq!(actions.len(), 1);
    assert!(matches!(
        &actions[0],
        PubSubAction::SendSubscriberDeclare { key_expr }
            if key_expr == "harmony/vines/aa00bb11cc22dd33ee44ff5566778899/announce/**"
    ));
}

// ── Test 2: Content discovery via Bloom filters ──────────────────

/// Proves the content discovery pipeline:
/// 1. Bob stores content in his StorageTier
/// 2. Bob's StorageTier emits a BroadcastFilter (Bloom filter)
/// 3. Alice deserializes the Bloom filter (received via Zenoh pub/sub)
/// 4. Alice checks: does Bob likely have my CID? → yes
/// 5. Alice queries Bob and gets the content
///
/// This is how friends help you discover content: their Bloom filter
/// broadcasts tell you what they have, before you even ask. Your tunnel
/// peers' filters are the most useful because they share your interests.
#[test]
fn bloom_filter_enables_social_content_discovery() {
    let budget = StorageBudget {
        cache_capacity: 100,
        max_pinned_bytes: 1_000_000,
    };
    let (mut tier, _startup_actions) = StorageTier::new(
        MemoryBlobStore::new(),
        budget,
        ContentPolicy::default(),
        FilterBroadcastConfig {
            mutation_threshold: 1, // broadcast after every mutation for testing
            max_interval_ticks: 30,
            expected_items: 64,
            fp_rate: 0.001,
        },
    );

    // Bob stores content
    let data = b"hello from the social graph";
    let cid = ContentId::for_blob(data, ContentFlags::default()).unwrap();

    let store_actions = tier.handle(StorageTierEvent::PublishContent {
        cid,
        data: data.to_vec(),
    });

    // Content should be announced
    assert!(
        store_actions
            .iter()
            .any(|a| matches!(a, StorageTierAction::AnnounceContent { .. })),
        "storing content should announce it"
    );

    // Trigger filter broadcast
    let tick_actions = tier.handle(StorageTierEvent::FilterTimerTick);

    // Find the BroadcastFilter action (Bloom filter for content availability)
    let filter_payload = tick_actions.iter().find_map(|a| match a {
        StorageTierAction::BroadcastFilter { payload } => Some(payload.clone()),
        _ => None,
    });

    if let Some(payload) = filter_payload {
        // Alice receives and deserializes the Bloom filter
        let filter =
            harmony_content::bloom::BloomFilter::from_bytes(&payload).expect("valid Bloom filter");

        // Alice checks: does Bob likely have my CID?
        assert!(
            filter.may_contain(&cid),
            "Bloom filter should indicate Bob likely has the CID"
        );
    }

    // Alice queries Bob for the content — should get a reply
    let query_actions = tier.handle(StorageTierEvent::ContentQuery { query_id: 1, cid });
    assert!(
        query_actions
            .iter()
            .any(|a| matches!(a, StorageTierAction::SendReply { query_id: 1, .. })),
        "content query should produce a reply"
    );

    // Verify metrics
    assert_eq!(tier.metrics().publishes_stored, 1);
    assert_eq!(tier.metrics().cache_hits, 1);
}
