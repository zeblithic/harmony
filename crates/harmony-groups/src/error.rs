use crate::types::OpId;

/// Errors returned when resolving a group DAG into a `GroupState`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolveError {
    /// The op list was empty.
    EmptyDag,
    /// No genesis (Create with no parents) op was found.
    NoGenesis,
    /// More than one op claims to be the genesis.
    MultipleGenesis,
    /// An op references a parent that is not present in the op list.
    MissingParent { op: OpId, parent: OpId },
    /// A cycle was detected in the DAG.
    CycleDetected,
    /// The genesis op does not have a `Create` action or has parents.
    InvalidGenesis,
    /// An op's stored ID does not match the canonical hash of its payload.
    InvalidOpId { op: OpId },
}

impl core::fmt::Display for ResolveError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyDag => write!(f, "DAG is empty"),
            Self::NoGenesis => write!(f, "no genesis op found"),
            Self::MultipleGenesis => write!(f, "multiple genesis ops found"),
            Self::MissingParent { op, parent } => write!(
                f,
                "op {:02x?} references missing parent {:02x?}",
                &op[..4],
                &parent[..4]
            ),
            Self::CycleDetected => write!(f, "cycle detected in DAG"),
            Self::InvalidGenesis => write!(f, "genesis op is invalid (must be Create with no parents)"),
            Self::InvalidOpId { op } => write!(f, "op {:02x?} has invalid content-addressed ID", &op[..4]),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ResolveError {}

