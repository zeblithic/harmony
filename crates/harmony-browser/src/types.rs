use harmony_content::cid::ContentId;

/// What the user types into the address bar.
#[derive(Debug, Clone)]
pub enum BrowseTarget {
    /// Direct CID reference (e.g., "hmy:abc123...").
    Cid(ContentId),
    /// Human-friendly name resolved via Zenoh queryable.
    Named(String),
    /// Live subscription to a key expression.
    Subscribe(String),
}

/// Image format for rendering decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageFormat {
    Png,
    Jpg,
    Webp,
}

/// Content type detected from InlineMetadata MIME hint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MimeHint {
    Markdown,
    PlainText,
    Image(ImageFormat),
    HarmonyApp,
    Unknown([u8; 8]),
}

impl MimeHint {
    pub fn from_mime_bytes(bytes: [u8; 8]) -> Self {
        match &bytes {
            b"text/md\0" => Self::Markdown,
            b"text/pln" => Self::PlainText,
            b"img/png\0" => Self::Image(ImageFormat::Png),
            b"img/jpg\0" => Self::Image(ImageFormat::Jpg),
            b"img/webp" => Self::Image(ImageFormat::Webp),
            b"app/hmy\0" => Self::HarmonyApp,
            _ => Self::Unknown(bytes),
        }
    }
}

/// Trust-based rendering decision for content.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TrustDecision {
    /// Render fully (text, images, everything).
    FullTrust,
    /// Render text, gate media behind one-click approval.
    Preview,
    /// Show metadata only. User must explicitly load.
    Untrusted,
    /// Author unknown to trust network. Show clear prompt.
    #[default]
    Unknown,
}
