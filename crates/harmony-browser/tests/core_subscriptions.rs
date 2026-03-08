use harmony_browser::{BrowseTarget, BrowserAction, BrowserCore, BrowserEvent, BrowserSubId};

#[test]
fn subscribe_tracks_active_subscription() {
    let mut core = BrowserCore::new();
    let _ = core.handle_event(BrowserEvent::Navigate(BrowseTarget::Subscribe(
        "harmony/presence/**".into(),
    )));
    assert_eq!(core.active_subscriptions().len(), 1);
    assert!(core.active_subscriptions().contains("harmony/presence/**"));
}

#[test]
fn duplicate_subscribe_does_not_add_twice() {
    let mut core = BrowserCore::new();
    let _ = core.handle_event(BrowserEvent::Navigate(BrowseTarget::Subscribe(
        "harmony/presence/**".into(),
    )));
    // Second subscribe to same key_expr should be a no-op
    let actions = core.handle_event(BrowserEvent::Navigate(BrowseTarget::Subscribe(
        "harmony/presence/**".into(),
    )));
    assert_eq!(core.active_subscriptions().len(), 1);
    assert!(actions.is_empty()); // No duplicate Subscribe action emitted
}

#[test]
fn subscription_update_emits_deliver() {
    let mut core = BrowserCore::new();
    let _ = core.handle_event(BrowserEvent::Navigate(BrowseTarget::Subscribe(
        "harmony/presence/**".into(),
    )));

    let sub_id = *core.active_subscription_ids().iter().next().unwrap();
    let actions = core.handle_event(BrowserEvent::SubscriptionUpdate {
        sub_id,
        key_expr: "harmony/presence/alice".into(),
        payload: b"online".to_vec(),
    });

    assert_eq!(actions.len(), 1);
    match &actions[0] {
        BrowserAction::DeliverUpdate {
            key_expr, payload, ..
        } => {
            assert_eq!(key_expr, "harmony/presence/alice");
            assert_eq!(payload, b"online");
        }
        other => panic!("expected DeliverUpdate, got {:?}", other),
    }
}

#[test]
fn unknown_sub_id_is_ignored() {
    let mut core = BrowserCore::new();
    let actions = core.handle_event(BrowserEvent::SubscriptionUpdate {
        sub_id: BrowserSubId(999),
        key_expr: "harmony/presence/alice".into(),
        payload: b"online".to_vec(),
    });
    assert!(actions.is_empty());
}
