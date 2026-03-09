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
    pub retry_policy: RetryPolicy,
    pub fuel_budget: u64,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
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
    use crate::trust::CapabilityDecl;
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct RawManifest {
        package: RawPackage,
        runtime: Option<RawRuntime>,
        capabilities: Option<RawCapabilities>,
        trust: Option<RawTrust>,
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

    #[derive(Deserialize, Default)]
    struct RawCapabilities {
        subscribe: Option<Vec<String>>,
        publish: Option<Vec<String>>,
        fetch: Option<Vec<String>>,
        store: Option<Vec<String>>,
        infer: Option<bool>,
        spawn: Option<Vec<String>>,
        seal: Option<bool>,
    }

    #[derive(Deserialize, Default)]
    struct RawTrust {
        signers: Option<Vec<String>>,
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
                    runtime.retry_policy.max_retries = mr;
                }
                if let Some(fb) = rt.fuel_budget {
                    runtime.fuel_budget = fb;
                }
            }

            let mut capabilities = CapabilitySet::new();
            if let Some(caps) = raw.capabilities {
                for topic in caps.subscribe.unwrap_or_default() {
                    capabilities.add(CapabilityDecl::Subscribe { topic });
                }
                for topic in caps.publish.unwrap_or_default() {
                    capabilities.add(CapabilityDecl::Publish { topic });
                }
                for namespace in caps.fetch.unwrap_or_default() {
                    capabilities.add(CapabilityDecl::Fetch { namespace });
                }
                for namespace in caps.store.unwrap_or_default() {
                    capabilities.add(CapabilityDecl::Store { namespace });
                }
                if caps.infer.unwrap_or(false) {
                    capabilities.add(CapabilityDecl::Infer);
                }
                for workflow in caps.spawn.unwrap_or_default() {
                    capabilities.add(CapabilityDecl::Spawn { workflow });
                }
                if caps.seal.unwrap_or(false) {
                    capabilities.add(CapabilityDecl::Seal);
                }
            }

            let trust = if let Some(t) = raw.trust {
                TrustConfig {
                    signers: t.signers.unwrap_or_default(),
                }
            } else {
                TrustConfig::default()
            };

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
                capabilities,
                trust,
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
        assert_eq!(config.retry_policy.max_retries, 3);
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
        assert_eq!(manifest.runtime.retry_policy.max_retries, 5);
        assert_eq!(manifest.runtime.fuel_budget, 2_000_000);
        assert!(!manifest.deploy.prefer_native);
        assert_eq!(manifest.deploy.replicas, 2);
    }

    #[cfg(feature = "std")]
    #[test]
    fn parse_kitri_toml_with_capabilities_and_trust() {
        use crate::trust::CapabilityDecl;

        let toml_str = r#"
[package]
name = "supply-chain"
version = "0.2.0"

[capabilities]
subscribe = ["orders/**", "events/user/*"]
publish = ["results/output"]
fetch = ["content/images"]
store = ["content/processed"]
infer = true
spawn = ["child-verifier"]
seal = true

[trust]
signers = ["did:key:z6MkFirst", "did:key:z6MkSecond"]
"#;
        let manifest = KitriManifest::from_toml(toml_str).unwrap();
        assert_eq!(manifest.name, "supply-chain");

        // Capabilities parsed correctly.
        let caps = &manifest.capabilities.declarations;
        assert_eq!(caps.len(), 8); // 2 subscribe + 1 publish + 1 fetch + 1 store + 1 infer + 1 spawn + 1 seal
        assert!(caps.contains(&CapabilityDecl::Subscribe {
            topic: "orders/**".into()
        }));
        assert!(caps.contains(&CapabilityDecl::Subscribe {
            topic: "events/user/*".into()
        }));
        assert!(caps.contains(&CapabilityDecl::Publish {
            topic: "results/output".into()
        }));
        assert!(caps.contains(&CapabilityDecl::Fetch {
            namespace: "content/images".into()
        }));
        assert!(caps.contains(&CapabilityDecl::Store {
            namespace: "content/processed".into()
        }));
        assert!(caps.contains(&CapabilityDecl::Infer));
        assert!(caps.contains(&CapabilityDecl::Spawn {
            workflow: "child-verifier".into()
        }));
        assert!(caps.contains(&CapabilityDecl::Seal));

        // Trust signers parsed correctly.
        assert_eq!(manifest.trust.signers.len(), 2);
        assert_eq!(manifest.trust.signers[0], "did:key:z6MkFirst");
        assert_eq!(manifest.trust.signers[1], "did:key:z6MkSecond");
    }

    #[cfg(feature = "std")]
    #[test]
    fn parse_kitri_toml_omitted_capabilities_and_trust_default() {
        let toml_str = r#"
[package]
name = "minimal"
version = "0.1.0"
"#;
        let manifest = KitriManifest::from_toml(toml_str).unwrap();
        assert!(manifest.capabilities.is_empty());
        assert!(manifest.trust.signers.is_empty());
    }
}
