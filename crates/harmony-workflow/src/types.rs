use crate::offload::ComputeHint;

/// Content-addressed workflow identifier.
///
/// Computed as `BLAKE3(module_hash || input)`. Deterministic: same module +
/// same input always produces the same workflow ID, enabling deduplication
/// across nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WorkflowId([u8; 32]);

impl WorkflowId {
    /// Compute a workflow ID from the module hash and input bytes.
    pub fn new(module_hash: &[u8; 32], input: &[u8]) -> Self {
        let mut buf = Vec::with_capacity(32 + input.len());
        buf.extend_from_slice(module_hash);
        buf.extend_from_slice(input);
        Self(harmony_crypto::hash::blake3_hash(&buf))
    }

    /// Create a WorkflowId from raw bytes (for deserialization).
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// The raw 32 bytes of this ID.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl std::fmt::Display for WorkflowId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for byte in &self.0[..4] {
            write!(f, "{byte:02x}")?;
        }
        write!(f, "...")
    }
}

/// A single recorded IO event in the workflow's execution history.
#[derive(Debug, Clone)]
pub enum HistoryEvent {
    /// WASM module requested content by CID (fetch_content call).
    IoRequested { cid: [u8; 32] },
    /// Content was resolved — data is `Some` if found, `None` if not found.
    IoResolved {
        cid: [u8; 32],
        data: Option<Vec<u8>>,
    },
}

/// The complete event log for a workflow — this IS the durable checkpoint.
///
/// On crash recovery, the workflow engine replays the WASM module from
/// scratch, feeding cached IO responses from this history. WASM determinism
/// guarantees identical execution paths: same module + same input + same IO
/// responses = same `fetch_content` calls in the same order.
#[derive(Debug, Clone)]
pub struct WorkflowHistory {
    pub workflow_id: WorkflowId,
    pub module_hash: [u8; 32],
    /// The original input bytes passed to the WASM module.
    pub input: Vec<u8>,
    /// Ordered sequence of IO events recorded during execution.
    pub events: Vec<HistoryEvent>,
    /// Total fuel consumed across all execution rounds.
    pub total_fuel_consumed: u64,
}

/// Current lifecycle state of a workflow.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkflowStatus {
    /// Queued, waiting for execution.
    Pending,
    /// Currently executing in the compute tier.
    Executing,
    /// Suspended waiting for external IO resolution.
    WaitingForIo { cid: [u8; 32] },
    /// Completed successfully.
    Complete,
    /// Failed with an error.
    Failed,
}

/// Inbound events for the workflow engine.
#[derive(Debug, Clone)]
pub enum WorkflowEvent {
    /// Submit a new workflow with inline module bytes.
    Submit {
        module: Vec<u8>,
        input: Vec<u8>,
        hint: ComputeHint,
    },
    /// Submit a workflow referencing a module by CID (fetched from storage).
    SubmitByCid {
        module_cid: [u8; 32],
        input: Vec<u8>,
        hint: ComputeHint,
    },
    /// Module bytes fetched from storage (for SubmitByCid).
    ModuleFetched { cid: [u8; 32], module: Vec<u8> },
    /// Module fetch failed.
    ModuleFetchFailed { cid: [u8; 32] },
    /// External IO resolved — content data is available.
    ContentFetched { cid: [u8; 32], data: Vec<u8> },
    /// External IO failed — content not found.
    ContentFetchFailed { cid: [u8; 32] },
}

/// Outbound actions returned by the workflow engine.
#[derive(Debug, Clone)]
pub enum WorkflowAction {
    /// Fetch a WASM module by CID from storage.
    FetchModule { cid: [u8; 32] },
    /// Fetch content by CID (IO request from WASM during execution).
    FetchContent {
        workflow_id: WorkflowId,
        cid: [u8; 32],
    },
    /// Workflow completed successfully — here is the output.
    WorkflowComplete {
        workflow_id: WorkflowId,
        output: Vec<u8>,
    },
    /// Workflow failed.
    WorkflowFailed {
        workflow_id: WorkflowId,
        error: String,
    },
    /// The workflow's history changed — caller should persist it.
    PersistHistory { workflow_id: WorkflowId },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workflow_id_deterministic() {
        let hash = [0xAB; 32];
        let input = b"hello";
        let id1 = WorkflowId::new(&hash, input);
        let id2 = WorkflowId::new(&hash, input);
        assert_eq!(id1, id2);
    }

    #[test]
    fn workflow_id_different_inputs_differ() {
        let hash = [0xAB; 32];
        let id1 = WorkflowId::new(&hash, b"hello");
        let id2 = WorkflowId::new(&hash, b"world");
        assert_ne!(id1, id2);
    }

    #[test]
    fn workflow_id_display() {
        let id = WorkflowId::from_bytes([0xDE; 32]);
        let display = format!("{id}");
        assert_eq!(display, "dededede...");
    }

    #[test]
    fn workflow_id_round_trip() {
        let hash = [0x42; 32];
        let id = WorkflowId::new(&hash, b"test");
        let bytes = *id.as_bytes();
        let id2 = WorkflowId::from_bytes(bytes);
        assert_eq!(id, id2);
    }

    #[test]
    fn history_event_construction() {
        let cid = [0xFF; 32];
        let req = HistoryEvent::IoRequested { cid };
        assert!(matches!(req, HistoryEvent::IoRequested { cid: c } if c == [0xFF; 32]));

        let resolved = HistoryEvent::IoResolved {
            cid,
            data: Some(vec![1, 2, 3]),
        };
        assert!(
            matches!(resolved, HistoryEvent::IoResolved { data: Some(d), .. } if d == vec![1, 2, 3])
        );
    }

    #[test]
    fn workflow_status_variants() {
        assert_eq!(WorkflowStatus::Pending, WorkflowStatus::Pending);
        assert_eq!(WorkflowStatus::Complete, WorkflowStatus::Complete);
        assert_ne!(WorkflowStatus::Pending, WorkflowStatus::Executing);

        let cid = [0x11; 32];
        assert_eq!(
            WorkflowStatus::WaitingForIo { cid },
            WorkflowStatus::WaitingForIo { cid }
        );
    }
}
