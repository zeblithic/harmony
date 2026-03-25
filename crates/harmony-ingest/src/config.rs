//! TOML config file parsing and validation for Engram ingestion.

use serde::Deserialize;
use std::path::Path;

/// Parsed Engram ingestion config.
#[derive(Debug, Clone, Deserialize)]
pub struct EngramConfig {
    /// Table version identifier (e.g. "v1").
    pub version: String,
    /// Embeddings per shard (typically 200).
    pub shard_size: u32,
    /// Per-head xxhash64 seeds.
    pub hash_seeds: Vec<u64>,
    /// Name of the tensor to extract from the safetensors file.
    pub tensor: String,
}

impl EngramConfig {
    /// Load and validate from a TOML file.
    pub fn load(path: &Path) -> Result<Self, String> {
        let text =
            std::fs::read_to_string(path).map_err(|e| format!("failed to read config: {e}"))?;
        let config: Self =
            toml::from_str(&text).map_err(|e| format!("invalid config TOML: {e}"))?;
        config.validate()?;
        Ok(config)
    }

    /// Parse and validate from a TOML string (for testing).
    #[allow(dead_code)]
    pub fn parse(text: &str) -> Result<Self, String> {
        let config: Self = toml::from_str(text).map_err(|e| format!("invalid config TOML: {e}"))?;
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), String> {
        if self.shard_size == 0 {
            return Err("shard_size must be > 0".into());
        }
        if self.hash_seeds.is_empty() {
            return Err("hash_seeds must not be empty".into());
        }
        if self.tensor.is_empty() {
            return Err("tensor name must not be empty".into());
        }
        Ok(())
    }

    /// Number of hash heads (implied by hash_seeds length).
    pub fn num_heads(&self) -> u32 {
        self.hash_seeds.len() as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_config() {
        let toml = r#"
            version = "v1"
            shard_size = 200
            hash_seeds = [111, 222, 333, 444]
            tensor = "model.engram_table.weight"
        "#;
        let config = EngramConfig::parse(toml).unwrap();
        assert_eq!(config.version, "v1");
        assert_eq!(config.shard_size, 200);
        assert_eq!(config.hash_seeds.len(), 4);
        assert_eq!(config.num_heads(), 4);
        assert_eq!(config.tensor, "model.engram_table.weight");
    }

    #[test]
    fn zero_shard_size() {
        let toml = r#"
            version = "v1"
            shard_size = 0
            hash_seeds = [111]
            tensor = "t"
        "#;
        let err = EngramConfig::parse(toml).unwrap_err();
        assert!(err.contains("shard_size"));
    }

    #[test]
    fn empty_hash_seeds() {
        let toml = r#"
            version = "v1"
            shard_size = 200
            hash_seeds = []
            tensor = "t"
        "#;
        let err = EngramConfig::parse(toml).unwrap_err();
        assert!(err.contains("hash_seeds"));
    }

    #[test]
    fn missing_field() {
        let toml = r#"
            version = "v1"
            shard_size = 200
        "#;
        let err = EngramConfig::parse(toml).unwrap_err();
        assert!(err.contains("invalid config"));
    }
}
