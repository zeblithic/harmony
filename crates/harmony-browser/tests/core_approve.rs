use harmony_browser::{BrowserAction, BrowserCore, BrowserEvent};
use harmony_content::cid::ContentId;

#[test]
fn approve_content_marks_cid_and_emits_fetch() {
    let mut core = BrowserCore::new();
    let cid = ContentId::for_blob(b"gated image data").unwrap();

    let actions = core.handle_event(BrowserEvent::ApproveContent { cid });
    assert_eq!(actions.len(), 1);
    match &actions[0] {
        BrowserAction::FetchContent { cid: c } => assert_eq!(*c, cid),
        other => panic!("expected FetchContent, got {:?}", other),
    }
}

#[test]
fn approved_cid_is_remembered() {
    let mut core = BrowserCore::new();
    let cid = ContentId::for_blob(b"gated image").unwrap();

    let _ = core.handle_event(BrowserEvent::ApproveContent { cid });
    assert!(core.is_approved(&cid));
}

#[test]
fn unapproved_cid_is_not_remembered() {
    let core = BrowserCore::new();
    let cid = ContentId::for_blob(b"nope").unwrap();
    assert!(!core.is_approved(&cid));
}
