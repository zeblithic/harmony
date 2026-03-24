//! Async bridge connecting a [`RawSocket`] to a zenoh session.
//!
//! The bridge runs a continuous loop that:
//! 1. Broadcasts Scout frames on a jittered timer.
//! 2. Receives inbound L2 frames and publishes Data payloads into zenoh.
//! 3. Drains the zenoh subscriber and broadcasts outbound Data frames.

use std::time::{Duration, Instant};

use rand::Rng;
use tracing::{debug, trace, warn};
use zenoh::Wait;

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

        loop {
            let now = Instant::now();

            // 1. Scout — broadcast presence if timer expired.
            if now >= next_scout {
                self.send_scout()?;
                let jitter = self.jittered_scout_interval();
                next_scout = now + jitter;
            }

            // 2. Process inbound L2 frames.
            self.process_inbound_frames()?;

            // 3. Drain outbound zenoh samples → broadcast as Data frames.
            while let Ok(Some(sample)) = subscriber.try_recv() {
                self.process_outbound_sample(&sample)?;
            }

            // 4. Yield to avoid busy-looping.
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    /// Broadcast a Scout frame carrying our identity hash.
    fn send_scout(&mut self) -> Result<(), RawLinkError> {
        let local_mac = self.socket.local_mac();
        let full_frame = frame::encode_scout_frame(local_mac, &self.config.identity_hash);
        // send_frame expects payload *after* the Ethernet header, but our
        // encode_* helpers produce a complete frame including the header.
        // The RawSocket trait says payload is everything after the Ethernet
        // header, so strip the first 14 bytes.
        let payload = &full_frame[crate::ETH_HEADER_LEN..];
        self.socket.send_frame(BROADCAST_MAC, payload)?;
        trace!(
            identity = hex::encode(self.config.identity_hash),
            "scout broadcast sent"
        );
        Ok(())
    }

    /// Process all pending inbound L2 frames.
    ///
    /// Destructures `self` to satisfy the borrow checker — the closure needs
    /// `&mut peer_table` and `&session` while `recv_frames` borrows `&mut socket`.
    fn process_inbound_frames(&mut self) -> Result<(), RawLinkError> {
        // Buffer inbound data frames so we can publish to zenoh after the
        // callback returns (avoids borrow-checker issues with &self.session
        // inside the FnMut closure).
        let mut inbound_data: Vec<(String, Vec<u8>)> = Vec::new();

        let Self {
            socket, peer_table, ..
        } = self;

        socket.recv_frames(&mut |src_mac, payload| {
            if payload.is_empty() {
                return;
            }

            match payload[0] {
                frame_type::SCOUT => {
                    // Scout payload: [SCOUT tag][16-byte identity_hash]
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
                    // Data payload: [DATA tag][u16 BE key_len][key_bytes][payload_bytes]
                    if payload.len() < 1 + 2 {
                        debug!(len = payload.len(), "data frame too short, ignoring");
                        return;
                    }
                    let key_len = u16::from_be_bytes([payload[1], payload[2]]) as usize;
                    let key_start = 3;
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
                        Ok(s) => s.to_string(),
                        Err(_) => {
                            debug!("data frame has invalid UTF-8 key_expr, ignoring");
                            return;
                        }
                    };
                    let data_payload = payload[key_end..].to_vec();
                    inbound_data.push((key_expr, data_payload));
                }
                other => {
                    trace!(frame_type = other, "ignoring unknown frame type");
                }
            }
        })?;

        // Publish buffered inbound data to zenoh.
        for (key_expr, data) in inbound_data {
            // TODO: Use zenoh SHM (shared memory) for zero-copy inbound delivery.
            // With the `shared-memory` feature enabled, we could allocate an SHM
            // buffer via ShmProvider, memcpy the payload, and publish as SHM-backed
            // ZBytes. For now, we copy the payload into a Vec.
            if let Err(e) = self.session.put(&key_expr, data).wait() {
                warn!(key_expr, %e, "failed to publish inbound data to zenoh");
            }
        }

        Ok(())
    }

    /// Encode and broadcast a zenoh sample as a Data frame.
    fn process_outbound_sample(
        &mut self,
        sample: &zenoh::sample::Sample,
    ) -> Result<(), RawLinkError> {
        let key_expr = sample.key_expr().as_str();
        let payload = sample.payload().to_bytes();

        // Build frame payload: [DATA tag][u16 BE key_len][key_bytes][payload_bytes]
        let mut frame_payload = Vec::with_capacity(1 + 2 + key_expr.len() + payload.len());
        frame_payload.push(frame_type::DATA);
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
    fn make_data_payload(key_expr: &str, data: &[u8]) -> Vec<u8> {
        let key_bytes = key_expr.as_bytes();
        let mut buf = Vec::with_capacity(1 + 2 + key_bytes.len() + data.len());
        buf.push(frame_type::DATA);
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
        let payload = make_data_payload(key, data);

        assert_eq!(payload[0], frame_type::DATA);
        let key_len = u16::from_be_bytes([payload[1], payload[2]]) as usize;
        assert_eq!(key_len, key.len());

        let decoded_key = std::str::from_utf8(&payload[3..3 + key_len]).expect("valid UTF-8");
        assert_eq!(decoded_key, key);

        let decoded_data = &payload[3 + key_len..];
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
    fn data_payload_encoding_matches_frame_module() {
        // Verify that the bridge's Data payload encoding is consistent with
        // the frame module's encode_data_frame (minus the Ethernet header).
        let key = "harmony/mesh/status";
        let data = b"online";

        let bridge_payload = make_data_payload(key, data);
        let full_frame = frame::encode_data_frame(MAC_A, BROADCAST_MAC, key, data);
        let frame_payload = &full_frame[crate::ETH_HEADER_LEN..];

        assert_eq!(bridge_payload, frame_payload);
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
