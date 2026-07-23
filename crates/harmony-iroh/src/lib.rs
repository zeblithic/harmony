//! Reusable iroh transport substrate: a persistent `Endpoint` wrapper with
//! relay/rebind lifecycle, plus an ALPN-keyed inbound-connection dispatch seam.
//!
//! App-agnostic: no identity/keychain, no hardcoded ALPN wire strings, no
//! concrete protocol acceptors. Extracted from harmony-client (ZEB-739).

pub mod dispatch;
pub mod endpoint;
mod error;

pub use error::IrohEndpointError;
