//! Error type for the iroh endpoint wrapper.

/// Failures constructing or driving an [`crate::endpoint::IrohEndpoint`].
///
/// Extracted from harmony-client's `IrohEndpointError`, reduced to the
/// transport-substrate surface: the client's `Keychain`/`Vault` variants
/// covered app-side key loading (which is no longer this crate's concern —
/// the caller injects the [`iroh::SecretKey`]), so only the bind failure
/// remains here.
#[derive(Debug, thiserror::Error)]
pub enum IrohEndpointError {
    /// The underlying [`iroh::Endpoint::bind`] failed. Boxed so this crate's
    /// error surface stays independent of iroh's internal error types.
    #[error("iroh endpoint bind failed")]
    Bind(#[source] Box<dyn std::error::Error + Send + Sync + 'static>),
}
