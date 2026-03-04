use crate::error::ComputeError;
use crate::types::{Checkpoint, ComputeResult, InstructionBudget};

/// Abstract WASM execution engine with cooperative yielding.
///
/// Implementations execute a bounded slice of WASM instructions and return
/// without blocking. The caller controls scheduling by choosing fuel budgets
/// and deciding when to resume yielded executions.
///
/// **WASM ABI:** Modules must export `memory` (linear memory) and
/// `compute(input_ptr: i32, input_len: i32) -> i32`. Input bytes are written
/// to memory at offset 0. The function returns the number of output bytes
/// written starting at offset `input_len`.
pub trait ComputeRuntime {
    /// Execute a WASM module's `compute` export with the given input and fuel budget.
    fn execute(&mut self, module: &[u8], input: &[u8], budget: InstructionBudget) -> ComputeResult;

    /// Resume a previously yielded execution with additional fuel.
    fn resume(&mut self, budget: InstructionBudget) -> ComputeResult;

    /// Whether there is a suspended execution that can be resumed.
    fn has_pending(&self) -> bool;

    /// Take a serializable snapshot of the current execution state.
    fn snapshot(&self) -> Result<Checkpoint, ComputeError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockRuntime;

    impl ComputeRuntime for MockRuntime {
        fn execute(
            &mut self,
            _module: &[u8],
            _input: &[u8],
            _budget: InstructionBudget,
        ) -> ComputeResult {
            ComputeResult::Complete {
                output: vec![1, 2, 3],
            }
        }

        fn resume(&mut self, _budget: InstructionBudget) -> ComputeResult {
            ComputeResult::Failed {
                error: ComputeError::NoPendingExecution,
            }
        }

        fn has_pending(&self) -> bool {
            false
        }

        fn snapshot(&self) -> Result<Checkpoint, ComputeError> {
            Err(ComputeError::NoPendingExecution)
        }
    }

    #[test]
    fn mock_runtime_execute() {
        let mut rt = MockRuntime;
        let result = rt.execute(b"", b"input", InstructionBudget { fuel: 1000 });
        assert!(matches!(result, ComputeResult::Complete { output } if output == vec![1, 2, 3]));
    }

    #[test]
    fn mock_runtime_has_no_pending() {
        let rt = MockRuntime;
        assert!(!rt.has_pending());
    }
}
