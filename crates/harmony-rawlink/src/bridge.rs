//! Async bridge connecting a [`RawSocket`] to a zenoh session.
//!
//! The bridge runs a continuous loop that:
//! 1. Broadcasts Scout frames on a jittered timer.
//! 2. Receives inbound L2 frames and publishes Data payloads into zenoh.
//! 3. Drains the zenoh subscriber and broadcasts outbound Data frames.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::time::{Duration, Instant};

use rand::Rng;
use tracing::{debug, trace, warn};

use crate::{
    error::RawLinkError,
    frame::{self, BROADCAST_MAC},
    frame_type,
    peer_table::PeerTable,
    socket::RawSocket,
};

/// Configuration for the [`Bridge`].
pub struct BridgeConfig {
    /// 128-bit identity hash broadcast in Scout frames.
    pub identity_hash: [u8; 16],
    /// Zenoh key expression pattern to subscribe to (e.g. `"harmony/**"`).
    pub subscribe_pattern: String,
    /// Base interval between Scout broadcasts (jittered to 1x–2x).
    pub scout_interval: Duration,
    /// Time-to-live for peer table entries.
    pub peer_ttl: Duration,
}

impl Default for BridgeConfig {
    fn default() -> Self {
        Self {
            identity_hash: [0u8; 16],
            subscribe_pattern: "harmony/**".to_string(),
            scout_interval: std::time::Duration::from_secs(5),
            peer_ttl: std::time::Duration::from_secs(30),
        }
    }
}

/// An async bridge that shuttles frames between a raw L2 socket and a zenoh session.
pub struct Bridge<S: RawSocket> {
    socket: S,
    session: zenoh::Session,
    config: BridgeConfig,
    peer_table: PeerTable,
}

impl<S: RawSocket> Bridge<S> {
    /// Create a new bridge.
    ///
    /// The zenoh session must already be open. The bridge does **not** own the
    /// session lifetime — callers keep a handle for other uses.
    pub fn new(socket: S, session: zenoh::Session, config: BridgeConfig) -> Self {
        let peer_table = PeerTable::new(config.peer_ttl);
        Self {
            socket,
            session,
            config,
            peer_table,
        }
    }

    /// Run the bridge loop forever (or until the socket errors out).
    ///
    /// This is the main entry point — call it from a `tokio::spawn`.
    pub async fn run(&mut self) -> Result<(), RawLinkError> {
        let subscriber = self
            .session
            .declare_subscriber(&self.config.subscribe_pattern)
            .await
            .map_err(|e| RawLinkError::SocketError(format!("zenoh subscribe failed: {e}")))?;

        let mut next_scout = Instant::now();
        let mut next_purge = Instant::now() + self.config.peer_ttl;
        let local_mac = self.socket.local_mac();

        let mut consecutive_errors: u32 = 0;

        loop {
            let now = Instant::now();

            // Run the iteration body, catching transient errors.
            let result: Result<(), RawLinkError> = async {
                // 1. Scout — broadcast presence if timer expired.
                if now >= next_scout {
                    self.send_scout()?;
                    let jitter = self.jittered_scout_interval();
                    next_scout = now + jitter;
                }

                // 2. Purge expired peer table entries periodically.
                if now >= next_purge {
                    self.peer_table.purge_expired();
                    next_purge = now + self.config.peer_ttl;
                }

                // 3. Process inbound L2 frames → publish to zenoh.
                let inbound_hashes = self.process_inbound_frames(&local_mac).await?;

                // 4. Drain outbound zenoh samples → broadcast as Data frames.
                //    Skip samples whose content hash matches something we just
                //    published from inbound L2 (echo prevention).
                while let Ok(Some(sample)) = subscriber.try_recv() {
                    let h = content_hash(sample.key_expr().as_str(), &sample.payload().to_bytes());
                    if inbound_hashes.contains(&h) {
                        trace!("skipping echo of L2-originated publish");
                        continue;
                    }
                    self.process_outbound_sample(&sample)?;
                }

                Ok(())
            }
            .await;

            match result {
                Ok(()) => {
                    consecutive_errors = 0;
                }
                Err(ref e) if consecutive_errors < 10 => {
                    consecutive_errors += 1;
                    warn!(
                        err = %e,
                        consecutive_errors,
                        "rawlink bridge iteration error (retrying)"
                    );
                    // Back off on repeated errors to avoid log spam.
                    let backoff_ms = 100 * (1u64 << consecutive_errors.min(6));
                    tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
                    continue;
                }
                Err(e) => {
                    tracing::error!(
                        err = %e,
                        "rawlink bridge giving up after {consecutive_errors} consecutive errors"
                    );
                    return Err(e);
                }
            }

            // 5. Yield to avoid busy-looping.
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    /// Broadcast a Scout frame carrying our identity hash.
    fn send_scout(&mut self) -> Result<(), RawLinkError> {
        // Build payload inline: [SCOUT tag][16-byte identity_hash]
        let mut payload = [0u8; 17];
        payload[0] = frame_type::SCOUT;
        payload[1..17].copy_from_slice(&self.config.identity_hash);
        self.socket.send_frame(BROADCAST_MAC, &payload)?;
        trace!(
            identity = hex::encode(self.config.identity_hash),
            "scout broadcast sent"
        );
        Ok(())
    }

    /// Process all pending inbound L2 frames.
    ///
    /// Buffers frames during the sync `recv_frames` callback, then publishes
    /// to zenoh asynchronously after the callback returns. Returns the key
    /// expressions that were published (for echo prevention).
    async fn process_inbound_frames(&mut self, local_mac: &[u8; 6]) -> Result<HashSet<u64>, RawLinkError> {
        let mut inbound_data: Vec<(String, Vec<u8>)> = Vec::new();
        let allowed_prefix = &self.config.subscribe_pattern;

        let Self {
            socket, peer_table, ..
        } = self;

        socket.recv_frames(&mut |src_mac, payload| {
            // Skip frames from ourselves (loopback on the interface).
            if src_mac == local_mac {
                return;
            }
            if payload.is_empty() {
                return;
            }

            match payload[0] {
                frame_type::SCOUT => {
                    if payload.len() < 1 + 16 {
                        debug!(len = payload.len(), "scout frame too short, ignoring");
                        return;
                    }
                    let mut identity_hash = [0u8; 16];
                    identity_hash.copy_from_slice(&payload[1..17]);
                    peer_table.update(identity_hash, *src_mac);
                    debug!(
                        identity = hex::encode(identity_hash),
                        src_mac = hex::encode(src_mac),
                        "peer scouted"
                    );
                }
                frame_type::DATA => {
                    // Data payload: [DATA tag][6-byte origin_mac][u16 BE key_len][key][payload]
                    if payload.len() < 1 + 6 + 2 {
                        debug!(len = payload.len(), "data frame too short, ignoring");
                        return;
                    }
                    let mut origin_mac = [0u8; 6];
                    origin_mac.copy_from_slice(&payload[1..7]);
                    // Discard frames we sent ourselves (L2 loopback).
                    if &origin_mac == local_mac {
                        trace!("discarding self-originated data frame");
                        return;
                    }
                    let key_len = u16::from_be_bytes([payload[7], payload[8]]) as usize;
                    let key_start = 9;
                    let key_end = key_start + key_len;
                    if payload.len() < key_end {
                        debug!(
                            key_len,
                            frame_len = payload.len(),
                            "data frame truncated key_expr, ignoring"
                        );
                        return;
                    }
                    let key_expr = match std::str::from_utf8(&payload[key_start..key_end]) {
                        Ok(s) => s,
                        Err(_) => {
                            debug!("data frame has invalid UTF-8 key_expr, ignoring");
                            return;
                        }
                    };
                    // Validate key expression against allowed namespace to prevent
                    // unauthenticated injection into arbitrary zenoh key spaces.
                    // Ensure prefix ends with '/' to prevent "harmony_evil" matching "harmony/".
                    let mut prefix = allowed_prefix.trim_end_matches("**").trim_end_matches('/').to_string();
                    prefix.push('/');
                    if !key_expr.starts_with(&prefix) {
                        debug!(
                            key_expr,
                            "data frame key_expr outside allowed namespace, ignoring"
                        );
                        return;
                    }
                    let data_payload = payload[key_end..].to_vec();
                    inbound_data.push((key_expr.to_string(), data_payload));
                }
                other => {
                    trace!(frame_type = other, "ignoring unknown frame type");
                }
            }
        })?;

        // Publish buffered inbound data to zenoh (async, non-blocking).
        // Track content hashes for echo prevention on the outbound path.
        let mut published_hashes = HashSet::new();
        for (key_expr, data) in inbound_data {
            let h = content_hash(&key_expr, &data);
            // TODO: Use zenoh SHM (shared memory) for zero-copy inbound delivery.
            if let Err(e) = self.session.put(&key_expr, data).await {
                warn!(key_expr, %e, "failed to publish inbound data to zenoh");
            } else {
                published_hashes.insert(h);
            }
        }

        Ok(published_hashes)
    }

    /// Encode and broadcast a zenoh sample as a Data frame.
    fn process_outbound_sample(
        &mut self,
        sample: &zenoh::sample::Sample,
    ) -> Result<(), RawLinkError> {
        let key_expr = sample.key_expr().as_str();
        let payload = sample.payload().to_bytes();

        // Guard against key expression length overflow (u16 max = 65535).
        if key_expr.len() > u16::MAX as usize {
            warn!(key_expr, "outbound key_expr exceeds u16 max length, dropping frame");
            return Ok(());
        }

        // Guard against oversized frames (standard Ethernet MTU = 1500 bytes).
        // ETH_MTU is the max Ethernet PAYLOAD (excludes the 14-byte header).
        // send_frame receives only the post-header payload, so we don't count
        // ETH_HEADER_LEN here. Overhead within the payload:
        //   1 (type tag) + 6 (origin_mac) + 2 (key_len) + key_expr.len()
        const ETH_MTU: usize = 1500;
        let frame_overhead = 1 + 6 + 2 + key_expr.len();
        if payload.len() > ETH_MTU.saturating_sub(frame_overhead) {
            trace!(
                key_expr,
                payload_len = payload.len(),
                "outbound payload exceeds Ethernet MTU, dropping frame"
            );
            return Ok(());
        }

        // Build frame payload: [DATA tag][6-byte origin_mac][u16 BE key_len][key_bytes][payload]
        let local_mac = self.socket.local_mac();
        let mut frame_payload = Vec::with_capacity(1 + 6 + 2 + key_expr.len() + payload.len());
        frame_payload.push(frame_type::DATA);
        frame_payload.extend_from_slice(&local_mac);
        frame_payload.extend_from_slice(&(key_expr.len() as u16).to_be_bytes());
        frame_payload.extend_from_slice(key_expr.as_bytes());
        frame_payload.extend_from_slice(&payload);

        self.socket.send_frame(BROADCAST_MAC, &frame_payload)?;
        trace!(
            key_expr,
            payload_len = payload.len(),
            "outbound data frame sent"
        );
        Ok(())
    }

    /// Returns a jittered scout interval between 1x and 2x the configured base.
    fn jittered_scout_interval(&self) -> Duration {
        let base = self.config.scout_interval.as_millis() as u64;
        let jitter = rand::thread_rng().gen_range(0..=base);
        Duration::from_millis(base + jitter)
    }

    /// Returns a reference to the peer table.
    pub fn peer_table(&self) -> &PeerTable {
        &self.peer_table
    }
}

/// Hash a (key_expr, payload) pair for content-based echo deduplication.
fn content_hash(key_expr: &str, payload: &[u8]) -> u64 {
    let mut hasher = DefaultHasher::new();
    key_expr.hash(&mut hasher);
    payload.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::socket::MockSocket;

    const MAC_A: [u8; 6] = [0x00, 0x11, 0x22, 0x33, 0x44, 0x55];
    const MAC_B: [u8; 6] = [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF];
    const IDENTITY: [u8; 16] = [
        0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
        0x10,
    ];

    /// Build a scout payload as the bridge would send it (post-Ethernet-header).
    fn make_scout_payload(identity_hash: &[u8; 16]) -> Vec<u8> {
        let mut buf = Vec::with_capacity(1 + 16);
        buf.push(frame_type::SCOUT);
        buf.extend_from_slice(identity_hash);
        buf
    }

    /// Build a data payload as the bridge would send it (post-Ethernet-header).
    fn make_data_payload(origin_mac: [u8; 6], key_expr: &str, data: &[u8]) -> Vec<u8> {
        let key_bytes = key_expr.as_bytes();
        let mut buf = Vec::with_capacity(1 + 6 + 2 + key_bytes.len() + data.len());
        buf.push(frame_type::DATA);
        buf.extend_from_slice(&origin_mac);
        buf.extend_from_slice(&(key_bytes.len() as u16).to_be_bytes());
        buf.extend_from_slice(key_bytes);
        buf.extend_from_slice(data);
        buf
    }

    #[test]
    fn scout_payload_round_trip() {
        let payload = make_scout_payload(&IDENTITY);
        assert_eq!(payload[0], frame_type::SCOUT);
        assert_eq!(payload.len(), 17);

        let mut decoded = [0u8; 16];
        decoded.copy_from_slice(&payload[1..17]);
        assert_eq!(decoded, IDENTITY);
    }

    #[test]
    fn data_payload_round_trip() {
        let key = "harmony/test/topic";
        let data = b"hello world";
        let origin = [0xCC; 6];
        let payload = make_data_payload(origin, key, data);

        assert_eq!(payload[0], frame_type::DATA);
        assert_eq!(&payload[1..7], &origin);
        let key_len = u16::from_be_bytes([payload[7], payload[8]]) as usize;
        assert_eq!(key_len, key.len());

        let decoded_key = std::str::from_utf8(&payload[9..9 + key_len]).expect("valid UTF-8");
        assert_eq!(decoded_key, key);

        let decoded_data = &payload[9 + key_len..];
        assert_eq!(decoded_data, data);
    }

    #[test]
    fn inbound_scout_updates_peer_table() {
        // Inject a scout frame via MockSocket side B, then run
        // process_inbound_frames on side A and verify peer_table is updated.
        let (mut socket_a, mut socket_b) = MockSocket::pair(MAC_A, MAC_B);

        // B sends a scout that A will receive.
        let scout_payload = make_scout_payload(&IDENTITY);
        socket_b
            .send_frame(MAC_A, &scout_payload)
            .expect("mock send");

        // Build a minimal Bridge around socket_a (no real zenoh session needed
        // for this test — we only call process_inbound_frames which uses
        // session.put for Data frames, not for Scout).
        let config = BridgeConfig {
            identity_hash: [0xFFu8; 16],
            subscribe_pattern: "harmony/**".into(),
            scout_interval: Duration::from_secs(5),
            peer_ttl: Duration::from_secs(30),
        };

        // We can't create a real zenoh::Session in a unit test without a
        // runtime, so we test the peer_table update logic directly.
        let mut peer_table = PeerTable::new(config.peer_ttl);

        // Simulate what process_inbound_frames does for Scout frames.
        socket_a
            .recv_frames(&mut |src_mac, payload| {
                if payload.is_empty() {
                    return;
                }
                if payload[0] == frame_type::SCOUT && payload.len() >= 17 {
                    let mut identity_hash = [0u8; 16];
                    identity_hash.copy_from_slice(&payload[1..17]);
                    peer_table.update(identity_hash, *src_mac);
                }
            })
            .expect("recv should succeed");

        // Verify the peer was added.
        assert_eq!(peer_table.peer_count(), 1);
        assert_eq!(peer_table.lookup(&IDENTITY), Some(MAC_B));
    }

    #[test]
    fn data_payload_contains_origin_mac() {
        // Verify that the bridge's Data payload includes the origin MAC
        // for cross-node echo prevention.
        let key = "harmony/mesh/status";
        let data = b"online";
        let origin = MAC_A;

        let payload = make_data_payload(origin, key, data);
        assert_eq!(payload[0], frame_type::DATA);
        assert_eq!(&payload[1..7], &origin);
        // Key expression follows after origin MAC
        let key_len = u16::from_be_bytes([payload[7], payload[8]]) as usize;
        assert_eq!(key_len, key.len());
    }

    #[test]
    fn scout_too_short_ignored() {
        // A scout payload with less than 17 bytes should be silently ignored.
        let (mut socket_a, mut socket_b) = MockSocket::pair(MAC_A, MAC_B);

        // Send a truncated scout (only tag + 8 bytes instead of 16).
        let mut short_scout = vec![frame_type::SCOUT];
        short_scout.extend_from_slice(&[0u8; 8]);
        socket_b.send_frame(MAC_A, &short_scout).expect("mock send");

        let mut peer_table = PeerTable::new(Duration::from_secs(30));

        socket_a
            .recv_frames(&mut |src_mac, payload| {
                if payload.is_empty() {
                    return;
                }
                if payload[0] == frame_type::SCOUT && payload.len() >= 17 {
                    let mut identity_hash = [0u8; 16];
                    identity_hash.copy_from_slice(&payload[1..17]);
                    peer_table.update(identity_hash, *src_mac);
                }
            })
            .expect("recv should succeed");

        assert_eq!(
            peer_table.peer_count(),
            0,
            "truncated scout must be ignored"
        );
    }

    #[test]
    fn unknown_frame_type_ignored() {
        let (mut socket_a, mut socket_b) = MockSocket::pair(MAC_A, MAC_B);

        // Send a frame with an unknown type tag (0xFF).
        socket_b
            .send_frame(MAC_A, &[0xFF, 0x01, 0x02])
            .expect("mock send");

        let mut count = 0usize;
        socket_a
            .recv_frames(&mut |_, payload| {
                if !payload.is_empty()
                    && payload[0] != frame_type::SCOUT
                    && payload[0] != frame_type::DATA
                {
                    count += 1;
                }
            })
            .expect("recv should succeed");

        assert_eq!(count, 1, "unknown frame type should be seen but not crash");
    }
}
