use crate::BrowserError;
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

impl BrowseTarget {
    /// Parse user address bar input into a BrowseTarget.
    ///
    /// Formats:
    /// - `hmy:<64-char hex>` -> CID lookup
    /// - `~<key_expr>` -> live subscription (prefixed with `harmony/`)
    /// - anything else -> named content (prefixed with `harmony/content/`)
    pub fn parse(input: &str) -> Result<Self, BrowserError> {
        let trimmed = input.trim();

        if let Some(hex_str) = trimmed.strip_prefix("hmy:") {
            let bytes = hex::decode(hex_str)
                .map_err(|_| BrowserError::InvalidCidHex(hex_str.to_string()))?;
            if bytes.len() != 32 {
                return Err(BrowserError::InvalidCidHex(format!(
                    "expected 32 bytes, got {}",
                    bytes.len()
                )));
            }
            let mut arr = [0u8; 32];
            arr.copy_from_slice(&bytes);
            let cid = ContentId::from_bytes(arr);
            Ok(Self::Cid(cid))
        } else if let Some(key_expr) = trimmed.strip_prefix('~') {
            Ok(Self::Subscribe(format!("harmony/{key_expr}")))
        } else {
            Ok(Self::Named(format!("harmony/content/{trimmed}")))
        }
    }
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
    Video,
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
            b"vine/vid" => Self::Video,
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
