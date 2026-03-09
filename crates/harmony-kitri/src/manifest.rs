// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Manifest types — parsed representation of `Kitri.toml`.

use alloc::string::String;
use alloc::vec::Vec;

use crate::retry::RetryPolicy;
use crate::trust::CapabilitySet;

/// Parsed `Kitri.toml` manifest.
#[derive(Debug, Clone)]
pub struct KitriManifest {
    pub name: String,
    pub version: String,
    pub runtime: RuntimeConfig,
    pub capabilities: CapabilitySet,
    pub trust: TrustConfig,
    pub deploy: DeployConfig,
}

/// Runtime configuration from `[runtime]` section.
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    pub max_retries: u32,
    pub retry_policy: RetryPolicy,
    pub fuel_budget: u64,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            retry_policy: RetryPolicy::default(),
            fuel_budget: 1_000_000,
        }
    }
}

/// Trust configuration from `[trust]` section.
#[derive(Debug, Clone, Default)]
pub struct TrustConfig {
    /// DID keys of trusted UCAN issuers.
    pub signers: Vec<String>,
}

/// Deployment preferences from `[deploy]` section.
#[derive(Debug, Clone)]
pub struct DeployConfig {
    pub prefer_native: bool,
    pub replicas: u32,
}

impl Default for DeployConfig {
    fn default() -> Self {
        Self {
            prefer_native: true,
            replicas: 1,
        }
    }
}

// ── TOML parsing (std-only) ─────────────────────────────────────────

#[cfg(feature = "std")]
mod parsing {
    use super::*;
    use crate::error::KitriError;
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct RawManifest {
        package: RawPackage,
        runtime: Option<RawRuntime>,
        deploy: Option<RawDeploy>,
    }

    #[derive(Deserialize)]
    struct RawPackage {
        name: String,
        version: String,
    }

    #[derive(Deserialize)]
    struct RawRuntime {
        max_retries: Option<u32>,
        fuel_budget: Option<u64>,
    }

    #[derive(Deserialize)]
    struct RawDeploy {
        prefer_native: Option<bool>,
        replicas: Option<u32>,
    }

    impl KitriManifest {
        /// Parse a `Kitri.toml` string into a manifest.
        pub fn from_toml(toml_str: &str) -> Result<Self, KitriError> {
            let raw: RawManifest =
                toml::from_str(toml_str).map_err(|e| KitriError::ManifestInvalid {
                    reason: e.to_string(),
                })?;

            let mut runtime = RuntimeConfig::default();
            if let Some(rt) = raw.runtime {
                if let Some(mr) = rt.max_retries {
                    runtime.max_retries = mr;
                }
                if let Some(fb) = rt.fuel_budget {
                    runtime.fuel_budget = fb;
                }
            }

            let mut deploy = DeployConfig::default();
            if let Some(dep) = raw.deploy {
                if let Some(pn) = dep.prefer_native {
                    deploy.prefer_native = pn;
                }
                if let Some(r) = dep.replicas {
                    deploy.replicas = r;
                }
            }

            Ok(KitriManifest {
                name: raw.package.name,
                version: raw.package.version,
                runtime,
                capabilities: CapabilitySet::new(),
                trust: TrustConfig::default(),
                deploy,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::retry::RetryPolicy;
    use crate::trust::CapabilitySet;

    #[test]
    fn manifest_creation() {
        let manifest = KitriManifest {
            name: "shipment-verifier".into(),
            version: "0.1.0".into(),
            runtime: RuntimeConfig {
                max_retries: 3,
                retry_policy: RetryPolicy::default(),
                fuel_budget: 1_000_000,
            },
            capabilities: CapabilitySet::new(),
            trust: TrustConfig { signers: vec![] },
            deploy: DeployConfig {
                prefer_native: true,
                replicas: 3,
            },
        };
        assert_eq!(manifest.name, "shipment-verifier");
        assert!(manifest.deploy.prefer_native);
        assert_eq!(manifest.deploy.replicas, 3);
    }

    #[test]
    fn runtime_config_defaults() {
        let config = RuntimeConfig::default();
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.fuel_budget, 1_000_000);
    }

    #[test]
    fn deploy_config_defaults() {
        let config = DeployConfig::default();
        assert!(config.prefer_native);
        assert_eq!(config.replicas, 1);
    }

    #[cfg(feature = "std")]
    #[test]
    fn parse_kitri_toml() {
        let toml_str = r#"
[package]
name = "test-workflow"
version = "0.1.0"

[runtime]
max_retries = 5
fuel_budget = 2000000

[deploy]
prefer_native = false
replicas = 2
"#;
        let manifest = KitriManifest::from_toml(toml_str).unwrap();
        assert_eq!(manifest.name, "test-workflow");
        assert_eq!(manifest.version, "0.1.0");
        assert_eq!(manifest.runtime.max_retries, 5);
        assert_eq!(manifest.runtime.fuel_budget, 2_000_000);
        assert!(!manifest.deploy.prefer_native);
        assert_eq!(manifest.deploy.replicas, 2);
    }
}
