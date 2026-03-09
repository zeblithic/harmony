// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Trust model — UCAN-based tiering for Kitri program execution.

use alloc::string::String;
use alloc::vec::Vec;

/// Trust tier assigned to a Kitri program at deploy time.
///
/// Determined by inspecting the program's UCAN capability chain.
/// Higher tiers get lower-overhead execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TrustTier {
    /// Anonymous, expired chain, or unknown signer.
    /// Forced into WASM sandbox with full synchronous audit.
    Untrusted = 0,
    /// Valid UCAN chain from a trusted issuer.
    /// Native execution with capability check + rate limiting.
    Delegated = 1,
    /// Signed by the node's own identity.
    /// Native execution with capability check only (fast path).
    Owner = 2,
}

impl TrustTier {
    /// Whether I/O content must be audited before commit.
    pub fn requires_content_audit(self) -> bool {
        self == Self::Untrusted
    }

    /// Whether I/O operations are rate-limited.
    pub fn requires_rate_limit(self) -> bool {
        self != Self::Owner
    }

    /// Whether this tier can run as native Rust (vs. forced WASM).
    pub fn allows_native_execution(self) -> bool {
        self != Self::Untrusted
    }
}

/// A declared capability requirement from `Kitri.toml`.
///
/// At deployment, the runtime issues a UCAN token scoped to exactly
/// these permissions. Any I/O outside scope returns Unauthorized.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CapabilityDecl {
    /// Can subscribe to messages on this topic pattern.
    Subscribe { topic: String },
    /// Can publish messages to this topic pattern.
    Publish { topic: String },
    /// Can fetch content under this CID namespace.
    Fetch { namespace: String },
    /// Can store content under this CID namespace.
    Store { namespace: String },
    /// Can invoke AI models.
    Infer,
    /// Can spawn a specific child workflow by name.
    Spawn { workflow: String },
    /// Can encrypt/decrypt via kernel.
    Seal,
}

/// The set of capabilities declared by a Kitri program.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CapabilitySet {
    pub declarations: Vec<CapabilityDecl>,
}

impl CapabilitySet {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, decl: CapabilityDecl) {
        self.declarations.push(decl);
    }

    pub fn is_empty(&self) -> bool {
        self.declarations.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trust_tier_ordering() {
        // Owner > Delegated > Untrusted
        assert!(TrustTier::Owner > TrustTier::Delegated);
        assert!(TrustTier::Delegated > TrustTier::Untrusted);
        assert!(TrustTier::Owner > TrustTier::Untrusted);
    }

    #[test]
    fn trust_tier_requires_audit() {
        assert!(!TrustTier::Owner.requires_content_audit());
        assert!(!TrustTier::Delegated.requires_content_audit());
        assert!(TrustTier::Untrusted.requires_content_audit());
    }

    #[test]
    fn trust_tier_requires_rate_limit() {
        assert!(!TrustTier::Owner.requires_rate_limit());
        assert!(TrustTier::Delegated.requires_rate_limit());
        assert!(TrustTier::Untrusted.requires_rate_limit());
    }

    #[test]
    fn trust_tier_allows_native() {
        assert!(TrustTier::Owner.allows_native_execution());
        assert!(TrustTier::Delegated.allows_native_execution());
        assert!(!TrustTier::Untrusted.allows_native_execution());
    }

    #[test]
    fn capability_declaration_variants() {
        let cap = CapabilityDecl::Subscribe {
            topic: "foo/**".into(),
        };
        assert!(matches!(cap, CapabilityDecl::Subscribe { .. }));

        let cap = CapabilityDecl::Infer;
        assert!(matches!(cap, CapabilityDecl::Infer));
    }
}
