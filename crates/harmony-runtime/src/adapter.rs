// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Platform adapter traits for multi-platform Harmony participation.

/// Metadata returned by platform initialization.
#[derive(Debug, Clone)]
pub struct PlatformInfo {
    /// Platform hostname or node name.
    pub hostname: String,
    /// Interfaces discovered at init time.
    pub interfaces: Vec<InterfaceDescriptor>,
}

/// Descriptor for a network transport available on the platform.
#[derive(Debug, Clone)]
pub struct InterfaceDescriptor {
    /// Interface name (matches Node's interface_name convention).
    pub name: String,
    /// Transport type.
    pub kind: InterfaceKind,
    /// Maximum transmission unit in bytes.
    pub mtu: usize,
    /// Estimated bandwidth in bytes/sec. 0 = unknown.
    pub bandwidth_estimate: u64,
}

/// Network transport type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InterfaceKind {
    Udp,
    Tcp,
    Serial,
    LoRa,
    ZenohTunnel,
    IrohQuic,
    Other(String),
}

/// What a platform must provide to host a Harmony runtime.
///
/// Extends the foundation traits (EntropySource, PersistentState)
/// with lifecycle, networking, attestation, and compute discovery.
pub trait PlatformAdapter {
    type Error: core::fmt::Debug;

    // --- Lifecycle ---
    fn init(&mut self) -> Result<PlatformInfo, Self::Error>;
    fn shutdown(&mut self) -> Result<(), Self::Error>;

    // --- Networking ---
    fn available_interfaces(&self) -> Vec<InterfaceDescriptor>;
    fn send(&mut self, interface: &str, data: &[u8]) -> Result<(), Self::Error>;
    fn receive(&mut self) -> Option<(String, Vec<u8>)>;

    // --- Attestation ---
    fn attestation(&self) -> crate::attestation::AttestationReport;

    // --- Compute ---
    fn compute_backends(&self) -> Vec<crate::compute_backend::ComputeBackendDescriptor>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interface_descriptor_construction() {
        let desc = InterfaceDescriptor {
            name: "udp0".to_string(),
            kind: InterfaceKind::Udp,
            mtu: 500,
            bandwidth_estimate: 1_000_000,
        };
        assert_eq!(desc.name, "udp0");
        assert_eq!(desc.mtu, 500);
    }

    #[test]
    fn interface_kind_variants_exist() {
        let _ = InterfaceKind::Udp;
        let _ = InterfaceKind::Tcp;
        let _ = InterfaceKind::Serial;
        let _ = InterfaceKind::LoRa;
        let _ = InterfaceKind::ZenohTunnel;
        let _ = InterfaceKind::IrohQuic;
        let _ = InterfaceKind::Other("custom".to_string());
    }

    #[test]
    fn platform_info_construction() {
        let info = PlatformInfo {
            hostname: "test-node".to_string(),
            interfaces: vec![],
        };
        assert_eq!(info.hostname, "test-node");
        assert!(info.interfaces.is_empty());
    }

    struct MockAdapter;

    impl PlatformAdapter for MockAdapter {
        type Error = String;

        fn init(&mut self) -> Result<PlatformInfo, Self::Error> {
            Ok(PlatformInfo {
                hostname: "mock".to_string(),
                interfaces: vec![],
            })
        }

        fn shutdown(&mut self) -> Result<(), Self::Error> {
            Ok(())
        }

        fn available_interfaces(&self) -> Vec<InterfaceDescriptor> {
            vec![]
        }

        fn send(&mut self, _interface: &str, _data: &[u8]) -> Result<(), Self::Error> {
            Ok(())
        }

        fn receive(&mut self) -> Option<(String, Vec<u8>)> {
            None
        }

        fn attestation(&self) -> crate::attestation::AttestationReport {
            crate::attestation::AttestationReport {
                tier: crate::attestation::AttestationTier::Unattested,
                evidence: None,
            }
        }

        fn compute_backends(&self) -> Vec<crate::compute_backend::ComputeBackendDescriptor> {
            vec![]
        }
    }

    #[test]
    fn mock_adapter_implements_trait() {
        let mut adapter = MockAdapter;
        let info = adapter.init().unwrap();
        assert_eq!(info.hostname, "mock");
        assert!(adapter.receive().is_none());
        adapter.shutdown().unwrap();
    }
}
