//! Sans-I/O model registry state machine.

/// Tracks local and remote model availability.
pub struct ModelRegistry;

/// Inbound events for the registry.
pub enum ModelRegistryEvent {}

/// Outbound actions from the registry.
pub enum ModelRegistryAction {}

/// Whether a model is available locally, remotely, or both.
pub enum Source {}
