// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Pluggable compute backend traits and descriptors.

/// Descriptor for an available compute backend.
#[derive(Debug, Clone)]
pub struct ComputeBackendDescriptor {
    /// Unique identifier for this backend instance.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// What kind of backend this is.
    pub kind: ComputeBackendKind,
    /// Capabilities this backend advertises.
    pub capabilities: Vec<ComputeCapability>,
}

/// Classification of compute backend type.
#[derive(Debug, Clone)]
pub enum ComputeBackendKind {
    /// HTTP inference server (LM Studio, ollama, vLLM, etc.).
    HttpInference { endpoint: String },
    /// Direct GPU access (future: CUDA, Metal, Vulkan compute).
    DirectGpu { device_id: String },
    /// CPU-only compute.
    Cpu,
}

/// A specific compute capability offered by a backend.
#[derive(Debug, Clone)]
pub enum ComputeCapability {
    /// LLM inference with a specific model.
    Inference {
        model_id: String,
        context_length: u32,
    },
    /// WASM module execution.
    WasmExecution { fuel_budget: u64 },
    /// Raw tensor compute (future).
    TensorCompute { flops_estimate: u64 },
}

/// Execute compute work on a specific backend.
///
/// Implementations are platform-specific. The runtime never calls
/// this directly — it emits `RuntimeAction::RunInference` and the
/// platform's event loop dispatches to the appropriate backend.
pub trait ComputeBackend {
    type Error: core::fmt::Debug;

    fn run_inference(
        &mut self,
        model_id: &str,
        input: &[u8],
        params: &[u8],
    ) -> Result<Vec<u8>, Self::Error>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_backend_kind_variants_exist() {
        let _ = ComputeBackendKind::HttpInference {
            endpoint: "http://localhost:1234".to_string(),
        };
        let _ = ComputeBackendKind::DirectGpu {
            device_id: "gpu0".to_string(),
        };
        let _ = ComputeBackendKind::Cpu;
    }

    #[test]
    fn compute_capability_variants_exist() {
        let _ = ComputeCapability::Inference {
            model_id: "llama-3.1".to_string(),
            context_length: 8192,
        };
        let _ = ComputeCapability::WasmExecution {
            fuel_budget: 100_000,
        };
        let _ = ComputeCapability::TensorCompute {
            flops_estimate: 1_000_000,
        };
    }

    #[test]
    fn compute_backend_descriptor_construction() {
        let desc = ComputeBackendDescriptor {
            id: "lmstudio-0".to_string(),
            name: "LM Studio".to_string(),
            kind: ComputeBackendKind::HttpInference {
                endpoint: "http://localhost:1234".to_string(),
            },
            capabilities: vec![ComputeCapability::Inference {
                model_id: "qwen-2.5".to_string(),
                context_length: 32768,
            }],
        };
        assert_eq!(desc.id, "lmstudio-0");
        assert_eq!(desc.capabilities.len(), 1);
    }

    struct MockBackend;

    impl ComputeBackend for MockBackend {
        type Error = String;

        fn run_inference(
            &mut self,
            _model_id: &str,
            _input: &[u8],
            _params: &[u8],
        ) -> Result<Vec<u8>, Self::Error> {
            Ok(vec![0x42])
        }
    }

    #[test]
    fn mock_backend_implements_trait() {
        let mut backend = MockBackend;
        let result = backend.run_inference("test", b"hello", b"").unwrap();
        assert_eq!(result, vec![0x42]);
    }
}
