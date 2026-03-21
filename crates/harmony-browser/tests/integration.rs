//! End-to-end test: navigate -> fetch -> render with trust.

use harmony_browser::{
    BrowseTarget, BrowserAction, BrowserCore, BrowserEvent, MimeHint, ResolvedContent,
    TrustDecision,
};
use harmony_content::book::{BookStore, MemoryBookStore};
use harmony_content::bundle::BundleBuilder;

#[test]
fn full_navigation_flow() {
    // 1. User types a named path
    let mut core = BrowserCore::new();
    let actions = core.handle_event(BrowserEvent::Navigate(BrowseTarget::Named(
        "harmony/content/wiki/rust".into(),
    )));
    assert!(matches!(&actions[0], BrowserAction::QueryNamed { .. }));

    // 2. Simulate: query resolved to a CID, caller fetched the bundle
    let mut store = MemoryBookStore::new();
    let book_cid = store.insert(b"# Rust Programming").unwrap();
    let mut builder = BundleBuilder::new();
    builder.add(book_cid);
    builder.with_metadata(19, 1, 1000, *b"text/md\0");
    let (bundle_bytes, bundle_cid) = builder.build().unwrap();

    // 3. Content arrives
    let actions = core.handle_event(BrowserEvent::ContentFetched {
        cid: bundle_cid,
        data: bundle_bytes,
    });

    // 4. Should render as markdown with Unknown trust (no author)
    assert_eq!(actions.len(), 1);
    match &actions[0] {
        BrowserAction::Render(ResolvedContent::Static {
            mime, trust_level, ..
        }) => {
            assert_eq!(*mime, MimeHint::Markdown);
            assert_eq!(*trust_level, TrustDecision::Unknown);
        }
        other => panic!("expected Render, got {:?}", other),
    }
}

#[test]
fn subscription_flow() {
    let mut core = BrowserCore::new();

    // 1. User subscribes
    let actions = core.handle_event(BrowserEvent::Navigate(BrowseTarget::Subscribe(
        "harmony/presence/**".into(),
    )));
    assert!(matches!(&actions[0], BrowserAction::Subscribe { .. }));
    assert_eq!(core.active_subscriptions().len(), 1);

    // 2. Update arrives
    let sub_id = *core.active_subscription_ids().iter().next().unwrap();
    let actions = core.handle_event(BrowserEvent::SubscriptionUpdate {
        sub_id,
        key_expr: "harmony/presence/bob".into(),
        payload: b"online".to_vec(),
    });
    assert!(matches!(&actions[0], BrowserAction::DeliverUpdate { .. }));
}

#[test]
fn parse_and_navigate() {
    let mut core = BrowserCore::new();
    let target = BrowseTarget::parse("~presence/**").unwrap();
    let actions = core.handle_event(BrowserEvent::Navigate(target));
    assert!(matches!(&actions[0], BrowserAction::Subscribe { .. }));
}
