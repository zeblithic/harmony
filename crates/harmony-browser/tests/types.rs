use harmony_browser::{BrowseTarget, MimeHint, TrustDecision};
use harmony_content::cid::ContentId;

#[test]
fn browse_target_cid_holds_content_id() {
    let cid = ContentId::for_blob(b"hello world").unwrap();
    let target = BrowseTarget::Cid(cid);
    match target {
        BrowseTarget::Cid(c) => assert_eq!(c, cid),
        _ => panic!("expected Cid variant"),
    }
}

#[test]
fn browse_target_named_holds_string() {
    let target = BrowseTarget::Named("wiki/rust".into());
    match target {
        BrowseTarget::Named(ref s) => assert_eq!(s, "wiki/rust"),
        _ => panic!("expected Named variant"),
    }
}

#[test]
fn browse_target_subscribe_holds_key_expr() {
    let target = BrowseTarget::Subscribe("harmony/presence/**".into());
    match target {
        BrowseTarget::Subscribe(ref s) => assert_eq!(s, "harmony/presence/**"),
        _ => panic!("expected Subscribe variant"),
    }
}

#[test]
fn mime_hint_from_bytes_recognizes_markdown() {
    assert_eq!(MimeHint::from_mime_bytes(*b"text/md\0"), MimeHint::Markdown);
}

#[test]
fn mime_hint_from_bytes_recognizes_plain_text() {
    assert_eq!(MimeHint::from_mime_bytes(*b"text/pln"), MimeHint::PlainText);
}

#[test]
fn mime_hint_from_bytes_recognizes_png() {
    assert_eq!(
        MimeHint::from_mime_bytes(*b"img/png\0"),
        MimeHint::Image(harmony_browser::ImageFormat::Png),
    );
}

#[test]
fn mime_hint_unknown_for_unrecognized() {
    let bytes = *b"foo/bar\0";
    assert_eq!(MimeHint::from_mime_bytes(bytes), MimeHint::Unknown(bytes));
}

#[test]
fn mime_hint_video() {
    assert_eq!(MimeHint::from_mime_bytes(*b"vine/vid"), MimeHint::Video);
}

#[test]
fn trust_decision_default_is_unknown() {
    assert_eq!(TrustDecision::default(), TrustDecision::Unknown);
}
