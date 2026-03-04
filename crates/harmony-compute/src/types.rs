use crate::error::ComputeError;

/// How many fuel units (approximately instructions) to allow before yielding.
#[derive(Debug, Clone, Copy)]
pub struct InstructionBudget {
    pub fuel: u64,
}

/// Serializable snapshot of WASM execution state for durable recovery.
#[derive(Debug, Clone)]
pub struct Checkpoint {
    /// BLAKE3 hash of the original WASM module bytes.
    pub module_hash: [u8; 32],
    /// Snapshot of WASM linear memory at time of checkpoint.
    pub memory: Vec<u8>,
    /// Total fuel consumed before this checkpoint was taken.
    pub fuel_consumed: u64,
}

/// An I/O request from WASM code (for future NeedsIO support).
#[derive(Debug, Clone)]
pub enum IORequest {
    /// Request to fetch content by CID.
    FetchContent { cid: [u8; 32] },
}

/// Response to an I/O request, provided by the caller after resolving externally.
#[derive(Debug, Clone)]
pub enum IOResponse {
    /// Content was found and is ready.
    ContentReady { data: Vec<u8> },
    /// Content was not found.
    ContentNotFound,
}

/// Result of executing a WASM computation slice.
#[derive(Debug)]
pub enum ComputeResult {
    /// Execution completed successfully.
    Complete { output: Vec<u8> },
    /// Execution yielded due to fuel exhaustion (can resume).
    Yielded { fuel_consumed: u64 },
    /// Execution needs external I/O before continuing.
    NeedsIO { request: IORequest },
    /// Execution failed with an error.
    Failed { error: ComputeError },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instruction_budget_construction() {
        let budget = InstructionBudget { fuel: 10_000 };
        assert_eq!(budget.fuel, 10_000);
    }

    #[test]
    fn checkpoint_construction() {
        let cp = Checkpoint {
            module_hash: [0xAB; 32],
            memory: vec![1, 2, 3, 4],
            fuel_consumed: 500,
        };
        assert_eq!(cp.module_hash, [0xAB; 32]);
        assert_eq!(cp.memory, vec![1, 2, 3, 4]);
        assert_eq!(cp.fuel_consumed, 500);
    }

    #[test]
    fn io_request_construction() {
        let req = IORequest::FetchContent { cid: [0xFF; 32] };
        assert!(matches!(
            req,
            IORequest::FetchContent { cid } if cid == [0xFF; 32]
        ));
    }

    #[test]
    fn compute_result_complete() {
        let result = ComputeResult::Complete {
            output: vec![42, 43],
        };
        assert!(matches!(result, ComputeResult::Complete { output } if output == vec![42, 43]));
    }

    #[test]
    fn compute_result_yielded() {
        let result = ComputeResult::Yielded {
            fuel_consumed: 5000,
        };
        assert!(matches!(
            result,
            ComputeResult::Yielded {
                fuel_consumed: 5000
            }
        ));
    }

    #[test]
    fn compute_result_needs_io() {
        let result = ComputeResult::NeedsIO {
            request: IORequest::FetchContent { cid: [0; 32] },
        };
        assert!(matches!(result, ComputeResult::NeedsIO { .. }));
    }

    #[test]
    fn compute_result_failed() {
        let result = ComputeResult::Failed {
            error: ComputeError::NoPendingExecution,
        };
        assert!(matches!(result, ComputeResult::Failed { .. }));
    }

    #[test]
    fn io_response_content_ready() {
        let resp = IOResponse::ContentReady {
            data: vec![1, 2, 3],
        };
        assert!(matches!(resp, IOResponse::ContentReady { data } if data == vec![1, 2, 3]));
    }

    #[test]
    fn io_response_content_not_found() {
        let resp = IOResponse::ContentNotFound;
        assert!(matches!(resp, IOResponse::ContentNotFound));
    }
}
