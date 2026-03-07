use harmony_browser::{
    BrowserAction, BrowserCore, BrowserEvent, MimeHint, ResolvedContent, TrustDecision,
};
use harmony_content::blob::{BlobStore, MemoryBlobStore};
use harmony_content::bundle::BundleBuilder;
use harmony_content::cid::ContentId;

/// Helper: build a content bundle with inline metadata.
fn build_test_bundle(data: &[u8], mime: [u8; 8]) -> (ContentId, Vec<u8>) {
    let mut store = MemoryBlobStore::new();
    let blob_cid = store.insert(data).unwrap();

    let mut builder = BundleBuilder::new();
    builder.add(blob_cid);
    builder.with_metadata(data.len() as u64, 1, 1000, mime);
    let (bundle_bytes, bundle_cid) = builder.build().unwrap();
    (bundle_cid, bundle_bytes)
}

#[test]
fn content_fetched_detects_markdown_mime() {
    let mut core = BrowserCore::new();
    let (cid, data) = build_test_bundle(b"# Hello", *b"text/md\0");

    let actions = core.handle_event(BrowserEvent::ContentFetched { cid, data });
    assert_eq!(actions.len(), 1);
    match &actions[0] {
        BrowserAction::Render(ResolvedContent::Static { mime, .. }) => {
            assert_eq!(*mime, MimeHint::Markdown);
        }
        other => panic!("expected Render, got {:?}", other),
    }
}

#[test]
fn content_fetched_with_no_trust_score_returns_unknown() {
    let mut core = BrowserCore::new();
    let (cid, data) = build_test_bundle(b"# Hello", *b"text/md\0");

    let actions = core.handle_event(BrowserEvent::ContentFetched { cid, data });
    match &actions[0] {
        BrowserAction::Render(ResolvedContent::Static { trust_level, .. }) => {
            assert_eq!(*trust_level, TrustDecision::Unknown);
        }
        other => panic!("expected Render, got {:?}", other),
    }
}

#[test]
fn content_fetched_plain_blob_renders_as_plain_text() {
    let mut core = BrowserCore::new();
    let data = b"just plain bytes, not a bundle";
    let cid = ContentId::for_blob(data).unwrap();

    let actions = core.handle_event(BrowserEvent::ContentFetched {
        cid,
        data: data.to_vec(),
    });
    match &actions[0] {
        BrowserAction::Render(ResolvedContent::Static { mime, .. }) => {
            assert_eq!(*mime, MimeHint::PlainText);
        }
        other => panic!("expected Render, got {:?}", other),
    }
}

#[test]
fn trust_update_affects_subsequent_content() {
    let mut core = BrowserCore::new();
    let author_address = [0xAA; 16];

    // Set trust: identity=3 (full trust)
    let _ = core.handle_event(BrowserEvent::TrustUpdated {
        address: author_address,
        score: 0b00000011, // identity=3
    });

    // Verify the trust_scores map is populated and policy resolves correctly
    assert_eq!(
        core.trust_policy().decide(core.trust_score(&author_address)),
        TrustDecision::FullTrust,
    );
}

#[test]
fn tampered_content_is_rejected() {
    let mut core = BrowserCore::new();
    let (cid, mut data) = build_test_bundle(b"# Hello", *b"text/md\0");

    // Tamper with the data so the hash no longer matches
    if let Some(byte) = data.last_mut() {
        *byte ^= 0xFF;
    }

    let actions = core.handle_event(BrowserEvent::ContentFetched { cid, data });
    assert!(actions.is_empty(), "tampered content should produce no actions");
}

#[test]
fn approved_content_renders_with_full_trust() {
    let mut core = BrowserCore::new();
    let data = b"gated image data";
    let cid = ContentId::for_blob(data).unwrap();

    // Approve the CID first
    let _ = core.handle_event(BrowserEvent::ApproveContent { cid });

    // Now when the content arrives, it should render with FullTrust
    let actions = core.handle_event(BrowserEvent::ContentFetched {
        cid,
        data: data.to_vec(),
    });
    assert_eq!(actions.len(), 1);
    match &actions[0] {
        BrowserAction::Render(ResolvedContent::Static { trust_level, .. }) => {
            assert_eq!(*trust_level, TrustDecision::FullTrust);
        }
        other => panic!("expected Render with FullTrust, got {:?}", other),
    }
}
