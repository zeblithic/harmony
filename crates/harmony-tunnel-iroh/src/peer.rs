//! Seam A (ZEB-739): the crate-owned peer descriptor the tunnel driver dials +
//! authenticates against, replacing the app-coupled
//! `harmony-client::owner_state_types::DeviceTunnelContact`.
//!
//! The app maps its own device-contact type into a [`TunnelPeer`] at the call
//! boundary (`TunnelManager::send_dm`), so this crate carries no
//! owner-state/CRDT coupling.

/// Everything the tunnel driver needs to dial and authenticate one peer device.
///
/// Fields are exactly what the driver's `dial_addr` + `peer_pq_identity`
/// consume from the client's `DeviceTunnelContact`, and no more:
///
/// * `node_id` — the peer's **iroh** `EndpointId` bytes (the ed25519 endpoint
///   key), used to build the dial [`iroh::EndpointAddr`]. This is NOT the tunnel
///   node id (`blake3(ML-DSA pubkey)`); that one is derived separately by
///   [`crate::manager::node_id_from_dsa_pubkey`] and keyed into the session map.
/// * `pq_identity` — the peer's reconstructed post-quantum public identity
///   (ML-KEM + ML-DSA). The client stored raw KEM/DSA pubkey bytes and rebuilt a
///   `PqIdentity` inside the dial; holding the already-reconstructed identity
///   here moves that fallible parse to the mapping boundary.
/// * `home_relay` — the peer's home relay, pre-parsed to an [`iroh::RelayUrl`]
///   (the client carried a `String` and parsed it in `dial_addr`).
///
/// Deviation from the task brief's field sketch: the sketch also listed
/// `direct_addrs: Vec<SocketAddr>`, but neither `dial_addr` nor
/// `peer_pq_identity` consumes direct addresses (the client's `dial_addr` builds
/// a node-id + relay `EndpointAddr` only), so per the brief's "include only
/// those [fields] consumed" directive the field is omitted. See the task-4
/// report.
#[derive(Clone)]
pub struct TunnelPeer {
    /// The peer's iroh `EndpointId` bytes (ed25519 endpoint key).
    pub node_id: [u8; 32],
    /// The peer's reconstructed post-quantum public identity (ML-KEM + ML-DSA).
    pub pq_identity: harmony_identity::PqIdentity,
    /// The peer's home relay, pre-parsed. `None` when the peer advertised none.
    pub home_relay: Option<iroh::RelayUrl>,
}
