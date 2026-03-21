use harmony_browser::BrowseTarget;
use harmony_content::cid::{ContentFlags, ContentId};

#[test]
fn parse_cid_hex_with_hmy_prefix() {
    let cid = ContentId::for_book(b"test data", ContentFlags::default()).unwrap();
    let hex = hex::encode(cid.to_bytes());
    let input = format!("hmy:{hex}");
    let target = BrowseTarget::parse(&input).unwrap();
    match target {
        BrowseTarget::Cid(c) => assert_eq!(c, cid),
        _ => panic!("expected Cid"),
    }
}

#[test]
fn parse_subscribe_with_tilde_prefix() {
    let target = BrowseTarget::parse("~presence/**").unwrap();
    match target {
        BrowseTarget::Subscribe(ref s) => assert_eq!(s, "harmony/presence/**"),
        _ => panic!("expected Subscribe"),
    }
}

#[test]
fn parse_named_path() {
    let target = BrowseTarget::parse("wiki/rust").unwrap();
    match target {
        BrowseTarget::Named(ref s) => assert_eq!(s, "harmony/content/wiki/rust"),
        _ => panic!("expected Named"),
    }
}

#[test]
fn parse_invalid_cid_hex_returns_error() {
    let result = BrowseTarget::parse("hmy:not_valid_hex_at_all");
    assert!(result.is_err());
}

#[test]
fn parse_empty_string_as_named() {
    let target = BrowseTarget::parse("").unwrap();
    match target {
        BrowseTarget::Named(ref s) => assert_eq!(s, "harmony/content/"),
        _ => panic!("expected Named"),
    }
}
