use harmony_browser::{BrowseTarget, BrowserAction, BrowserCore, BrowserEvent};
use harmony_content::cid::{ContentFlags, ContentId};

fn make_core() -> BrowserCore {
    BrowserCore::new()
}

#[test]
fn navigate_to_cid_emits_fetch() {
    let mut core = make_core();
    let cid = ContentId::for_book(b"hello", ContentFlags::default()).unwrap();
    let actions = core.handle_event(BrowserEvent::Navigate(BrowseTarget::Cid(cid)));
    assert_eq!(actions.len(), 1);
    match &actions[0] {
        BrowserAction::FetchContent { cid: c } => assert_eq!(*c, cid),
        other => panic!("expected FetchContent, got {:?}", other),
    }
}

#[test]
fn navigate_to_named_emits_query() {
    let mut core = make_core();
    let actions = core.handle_event(BrowserEvent::Navigate(BrowseTarget::Named(
        "harmony/content/wiki/rust".into(),
    )));
    assert_eq!(actions.len(), 1);
    match &actions[0] {
        BrowserAction::QueryNamed { key_expr } => {
            assert_eq!(key_expr, "harmony/content/wiki/rust");
        }
        other => panic!("expected QueryNamed, got {:?}", other),
    }
}

#[test]
fn navigate_to_subscribe_emits_subscribe() {
    let mut core = make_core();
    let actions = core.handle_event(BrowserEvent::Navigate(BrowseTarget::Subscribe(
        "harmony/presence/**".into(),
    )));
    assert_eq!(actions.len(), 1);
    match &actions[0] {
        BrowserAction::Subscribe { key_expr, sub_id } => {
            assert_eq!(key_expr, "harmony/presence/**");
            assert_eq!(*sub_id, harmony_browser::BrowserSubId(1));
        }
        other => panic!("expected Subscribe, got {:?}", other),
    }
}
