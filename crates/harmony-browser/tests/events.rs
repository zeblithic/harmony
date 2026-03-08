use harmony_browser::{
    BrowseTarget, BrowserAction, BrowserEvent, MimeHint, ResolvedContent, TrustDecision,
};
use harmony_content::cid::{ContentFlags, ContentId};

#[test]
fn navigate_event_holds_browse_target() {
    let cid = ContentId::for_blob(b"hello", ContentFlags::default()).unwrap();
    let event = BrowserEvent::Navigate(BrowseTarget::Cid(cid));
    match event {
        BrowserEvent::Navigate(BrowseTarget::Cid(c)) => assert_eq!(c, cid),
        _ => panic!("wrong variant"),
    }
}

#[test]
fn fetch_action_holds_cid() {
    let cid = ContentId::for_blob(b"test", ContentFlags::default()).unwrap();
    let action = BrowserAction::FetchContent { cid };
    match action {
        BrowserAction::FetchContent { cid: c } => assert_eq!(c, cid),
        _ => panic!("wrong variant"),
    }
}

#[test]
fn render_action_holds_resolved_static() {
    let cid = ContentId::for_blob(b"# Hello", ContentFlags::default()).unwrap();
    let resolved = ResolvedContent::Static {
        cid,
        mime: MimeHint::Markdown,
        data: b"# Hello".to_vec(),
        author: None,
        trust_level: TrustDecision::Unknown,
    };
    match resolved {
        ResolvedContent::Static {
            mime, trust_level, ..
        } => {
            assert_eq!(mime, MimeHint::Markdown);
            assert_eq!(trust_level, TrustDecision::Unknown);
        }
        _ => panic!("wrong variant"),
    }
}
