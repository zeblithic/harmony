//! Model metadata types for Harmony CAS-distributed ML models.

use harmony_content::ContentId;
use serde::{Deserialize, Serialize};

/// Describes a model stored in Harmony CAS.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelManifest {
    /// Human-readable name (e.g., "Qwen3-0.6B-Q4_K_M").
    pub name: String,
    /// Model family identifier (e.g., "qwen3", "llama", "phi").
    pub family: String,
    /// Parameter count (e.g., 600_000_000 for 0.6B).
    pub parameter_count: u64,
    /// Serialization format.
    pub format: ModelFormat,
    /// Quantization method (e.g., "Q4_K_M", "F16"). None for full precision.
    pub quantization: Option<String>,
    /// Context window size in tokens.
    pub context_length: u32,
    /// Vocabulary size.
    pub vocab_size: u32,
    /// Estimated memory required in bytes.
    pub memory_estimate: u64,
    /// Tasks this model supports.
    pub tasks: Vec<ModelTask>,
    /// CID of the model data (DAG root for large models, book for small).
    pub data_cid: ContentId,
    /// CID of the tokenizer (JSON book). None if embedded in model file.
    pub tokenizer_cid: Option<ContentId>,
}

/// Model serialization format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelFormat {
    Gguf,
    Safetensors,
}

/// Task type a model supports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelTask {
    TextGeneration,
    Embedding,
    Vision,
    AudioTranscription,
}

/// Compact advertisement for Zenoh discovery.
///
/// Contains enough metadata for quick filtering without fetching the full
/// manifest from CAS. Published to `harmony/model/{manifest_cid}/{node_addr}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelAdvertisement {
    /// Manifest CID (redundant with key, self-contained for consumers).
    pub manifest_cid: ContentId,
    /// Human-readable model name.
    pub name: String,
    /// Model family identifier.
    pub family: String,
    /// Parameter count.
    pub parameter_count: u64,
    /// Quantization method.
    pub quantization: Option<String>,
    /// Supported tasks.
    pub tasks: Vec<ModelTask>,
    /// Estimated memory required in bytes.
    pub memory_estimate: u64,
}
