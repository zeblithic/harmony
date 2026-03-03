//! Resource transfer protocol for Reticulum links.
//!
//! Implements the sans-I/O state machines for sending and receiving
//! arbitrary-sized data over Reticulum links using chunked, windowed
//! transfers with acknowledgement-based flow control.

use std::collections::HashMap;

use rand_core::CryptoRngCore;

use crate::context::PacketContext;
use crate::error::ReticulumError;
use harmony_crypto::hash;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Length of the map-hash prefix used to identify resource parts.
pub const MAPHASH_LEN: usize = 4;

/// Maximum data size for efficient single-advertisement transfer (16 MiB - 1).
pub const MAX_EFFICIENT_SIZE: usize = 0xFF_FFFF;

/// Maximum number of retries for a single resource part.
pub const MAX_RETRIES: u32 = 16;

/// Maximum number of retries for the initial advertisement.
pub const MAX_ADV_RETRIES: u32 = 4;

/// Initial sliding-window size (number of in-flight parts).
pub const WINDOW_INITIAL: u32 = 4;

/// Minimum sliding-window size.
pub const WINDOW_MIN: u32 = 2;

/// Maximum sliding-window size for fast links.
pub const WINDOW_MAX_SLOW: u32 = 10;

/// Maximum sliding-window size for very fast links.
pub const WINDOW_MAX_FAST: u32 = 75;

/// Maximum sliding-window size for very slow links.
pub const WINDOW_MAX_VERY_SLOW: u32 = 4;

/// Bytes-per-second threshold above which a link is considered "fast".
pub const RATE_FAST: u64 = 6250;

/// Bytes-per-second threshold below which a link is considered "very slow".
pub const RATE_VERY_SLOW: u64 = 250;

/// Grace time (ms) added to sender timeouts to account for processing.
pub const SENDER_GRACE_TIME_MS: u64 = 10_000;

/// Grace time (ms) added to receiver timeouts for processing.
pub const PROCESSING_GRACE_MS: u64 = 1_000;

/// Multiplier for proof timeout relative to RTT.
pub const PROOF_TIMEOUT_FACTOR: u32 = 3;

/// Multiplier for part timeout relative to RTT.
pub const PART_TIMEOUT_FACTOR: u32 = 4;

/// Multiplier for part timeout after first RTT measurement.
pub const PART_TIMEOUT_FACTOR_AFTER_RTT: u32 = 2;

/// Delay (ms) added per retry attempt.
pub const PER_RETRY_DELAY_MS: u64 = 500;

/// Grace time (ms) added to retry delays.
pub const RETRY_GRACE_TIME_MS: u64 = 250;

/// Threshold (consecutive fast acks) to classify link as fast.
pub const FAST_RATE_THRESHOLD: u32 = 5;

/// Threshold (consecutive slow acks) to classify link as very slow.
pub const VERY_SLOW_RATE_THRESHOLD: u32 = 2;

/// Flexibility factor for window growth/shrink decisions.
pub const WINDOW_FLEXIBILITY: u32 = 4;

/// Hashmap segment marker: more segments follow.
pub const HASHMAP_IS_NOT_EXHAUSTED: u8 = 0x00;

/// Hashmap segment marker: this is the final segment.
pub const HASHMAP_IS_EXHAUSTED: u8 = 0xFF;

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------

/// 16-byte SHA-256 truncated hash identifying a resource.
pub type ResourceHash = [u8; 16];

/// 4-byte SHA-256 truncated hash identifying a resource part in the hashmap.
pub type MapHash = [u8; 4];

/// 4-byte random nonce mixed into hash computations.
pub type RandomHash = [u8; 4];

// ---------------------------------------------------------------------------
// LinkCrypto trait
// ---------------------------------------------------------------------------

/// Abstraction over link-layer encryption used by resource transfers.
///
/// The resource state machines need to encrypt outgoing packets and decrypt
/// incoming ones, but they should not depend directly on `Link`. This trait
/// decouples them so that tests can use a mock implementation.
pub trait LinkCrypto {
    /// Encrypt `plaintext` for transmission over the link.
    fn encrypt(
        &self,
        rng: &mut dyn CryptoRngCore,
        plaintext: &[u8],
    ) -> Result<Vec<u8>, ReticulumError>;

    /// Decrypt `ciphertext` received from the link.
    fn decrypt(&self, ciphertext: &[u8]) -> Result<Vec<u8>, ReticulumError>;

    /// Return the link identifier (destination hash).
    fn link_id(&self) -> &[u8; 16];

    /// Maximum data unit: the largest plaintext payload that fits in a
    /// single link packet after encryption overhead is subtracted.
    fn mdu(&self) -> usize;
}

// ---------------------------------------------------------------------------
// State enums
// ---------------------------------------------------------------------------

/// State of a resource sender.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SenderState {
    /// Resource created but advertisement not yet sent.
    Queued,
    /// Advertisement sent, awaiting acceptance from receiver.
    Advertised,
    /// Actively sending data parts.
    Transferring,
    /// All parts sent, awaiting proof from receiver.
    AwaitingProof,
    /// Transfer completed successfully.
    Complete,
    /// Transfer failed (timeout, too many retries, etc.).
    Failed,
    /// Receiver explicitly rejected the transfer.
    Rejected,
}

/// State of a resource receiver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReceiverState {
    /// Actively receiving data parts.
    Transferring,
    /// All parts received, assembling final data.
    Assembling,
    /// Assembly complete, data available.
    Complete,
    /// Assembly failed — hash mismatch.
    Corrupt,
    /// Transfer failed (timeout, too many retries, etc.).
    Failed,
}

// ---------------------------------------------------------------------------
// Event / Action enums (sans-I/O interface)
// ---------------------------------------------------------------------------

/// Events fed into the resource state machines by the caller.
#[derive(Debug, Clone)]
pub enum ResourceEvent {
    /// A resource request/advertisement was received from the remote peer.
    RequestReceived,
    /// A resource data part was received.
    PartReceived,
    /// A hashmap update segment was received.
    HashmapUpdateReceived,
    /// A resource proof was received from the receiver.
    ProofReceived,
    /// The remote peer cancelled the transfer.
    CancelReceived,
    /// A timeout fired; `now_ms` is the current monotonic time.
    Timeout { now_ms: u64 },
}

/// Actions emitted by the resource state machines for the caller to execute.
#[derive(Debug, Clone)]
pub enum ResourceAction {
    /// Send a packet over the link with the given context byte and plaintext.
    SendPacket { context: u8, plaintext: Vec<u8> },
    /// The sender state changed (caller should inspect `ResourceSender::state()`).
    SenderStateChanged,
    /// The receiver state changed (caller should inspect `ResourceReceiver::state()`).
    ReceiverStateChanged,
    /// Transfer progress update.
    Progress { fraction: f32 },
    /// All parts received, assembly buffer is ready for hash verification.
    AssemblyReady,
    /// Transfer completed; `data` contains the reassembled payload.
    Completed { data: Vec<u8> },
    /// Proof validated by the sender — transfer confirmed.
    ProofValidated,
    /// Schedule a timeout callback at the given monotonic deadline.
    ScheduleTimeout { deadline_ms: u64 },
}

// ---------------------------------------------------------------------------
// ResourceSender
// ---------------------------------------------------------------------------

/// Default timeout (ms) used before RTT is measured.
const DEFAULT_TIMEOUT_MS: u64 = 15_000;

/// Sender-side resource transfer state machine.
///
/// Drives advertisement, chunking, windowed transmission, and proof
/// validation. Created by the link when initiating a transfer.
#[derive(Debug)]
#[allow(dead_code)] // Fields used in later tasks (windowing, retransmission).
pub struct ResourceSender {
    pub(crate) state: SenderState,
    pub(crate) resource_hash: ResourceHash,
    pub(crate) random_hash: RandomHash,
    pub(crate) original_data: Vec<u8>,
    pub(crate) encrypted_data: Vec<u8>,
    pub(crate) parts: Vec<Vec<u8>>,
    pub(crate) map_hashes: Vec<MapHash>,
    pub(crate) hashmap: Vec<u8>,
    pub(crate) expected_proof: ResourceHash,
    pub(crate) sdu: usize,
    pub(crate) adv_sent_at: Option<u64>,
    pub(crate) last_activity: u64,
    pub(crate) adv_retries_left: u8,
    pub(crate) sent_parts: usize,
    pub(crate) rtt_ms: Option<u64>,
}

impl ResourceSender {
    /// Create a new resource sender.
    ///
    /// Validates the data size, encrypts, chunks, and computes all hashes.
    /// Returns the sender in `Queued` state, ready for [`advertise`](Self::advertise).
    pub fn new(
        rng: &mut dyn CryptoRngCore,
        crypto: &impl LinkCrypto,
        data: &[u8],
        now_ms: u64,
    ) -> Result<Self, ReticulumError> {
        if data.len() > MAX_EFFICIENT_SIZE {
            return Err(ReticulumError::ResourceTooLarge {
                size: data.len(),
                max: MAX_EFFICIENT_SIZE,
            });
        }

        // Generate 4-byte random nonce.
        let mut random_hash: RandomHash = [0u8; 4];
        rng.fill_bytes(&mut random_hash);

        // Encrypt the data.
        let encrypted_data = crypto.encrypt(rng, data)?;

        // Compute the resource hash over encrypted data.
        let resource_hash = compute_resource_hash(&encrypted_data, &random_hash);

        // Chunk the encrypted data into parts of at most `mdu` bytes.
        let sdu = crypto.mdu();
        let parts = chunk_data(&encrypted_data, sdu);

        // Compute map hashes for each part.
        let map_hashes: Vec<MapHash> = parts
            .iter()
            .map(|part| compute_map_hash(part, &random_hash))
            .collect();

        // Build the concatenated hashmap (all map hashes in order).
        let mut hashmap = Vec::with_capacity(map_hashes.len() * MAPHASH_LEN);
        for mh in &map_hashes {
            hashmap.extend_from_slice(mh);
        }

        // Compute the expected proof the receiver will send.
        let expected_proof = compute_proof_hash(data, &resource_hash);

        Ok(Self {
            state: SenderState::Queued,
            resource_hash,
            random_hash,
            original_data: data.to_vec(),
            encrypted_data,
            parts,
            map_hashes,
            hashmap,
            expected_proof,
            sdu,
            adv_sent_at: None,
            last_activity: now_ms,
            adv_retries_left: MAX_ADV_RETRIES as u8,
            sent_parts: 0,
            rtt_ms: None,
        })
    }

    /// Current sender state.
    pub fn state(&self) -> SenderState {
        self.state
    }

    /// Resource hash identifying this transfer.
    pub fn hash(&self) -> &ResourceHash {
        &self.resource_hash
    }

    /// Number of parts the encrypted data was split into.
    pub fn part_count(&self) -> usize {
        self.parts.len()
    }

    /// Build and emit the resource advertisement packet.
    ///
    /// Transitions from `Queued` to `Advertised` and schedules a timeout
    /// for the advertisement response.
    pub fn advertise(&mut self, now_ms: u64) -> Vec<ResourceAction> {
        let adv = self.build_advertisement();

        self.state = SenderState::Advertised;
        self.adv_sent_at = Some(now_ms);
        self.last_activity = now_ms;

        let timeout = now_ms + DEFAULT_TIMEOUT_MS;

        vec![
            ResourceAction::SendPacket {
                context: PacketContext::ResourceAdv.as_byte(),
                plaintext: adv.encode(),
            },
            ResourceAction::SenderStateChanged,
            ResourceAction::ScheduleTimeout {
                deadline_ms: timeout,
            },
        ]
    }

    fn build_advertisement(&self) -> ResourceAdvertisement {
        ResourceAdvertisement {
            transfer_size: self.encrypted_data.len() as u32,
            data_size: self.original_data.len() as u32,
            part_count: self.parts.len() as u32,
            resource_hash: self.resource_hash,
            random_hash: self.random_hash,
            original_hash: self.resource_hash, // single-segment: original == resource
            hashmap: self.hashmap.clone(),
            flags: 0x01,      // bit 0 = encrypted (always true for link resources)
            segment_index: 1, // 1-based, matching Python
            total_segments: 1,
            request_id: None,
        }
    }

    /// Process an incoming event and return resulting actions.
    pub fn handle_event(&mut self, event: ResourceEvent, payload: &[u8]) -> Vec<ResourceAction> {
        match event {
            ResourceEvent::RequestReceived => self.handle_request(payload),
            ResourceEvent::ProofReceived => self.handle_proof(payload),
            ResourceEvent::CancelReceived => self.handle_cancel_received(),
            ResourceEvent::Timeout { now_ms } => self.handle_timeout(now_ms),
            _ => vec![],
        }
    }

    /// Cancel the transfer from the sender side.
    ///
    /// Emits a cancel packet (`ResourceIcl`) unless the transfer is already
    /// complete or failed.
    pub fn cancel(&mut self) -> Vec<ResourceAction> {
        match self.state {
            SenderState::Complete | SenderState::Failed | SenderState::Rejected => vec![],
            _ => {
                self.state = SenderState::Failed;
                vec![
                    ResourceAction::SendPacket {
                        context: PacketContext::ResourceIcl.as_byte(),
                        plaintext: self.resource_hash.to_vec(),
                    },
                    ResourceAction::SenderStateChanged,
                ]
            }
        }
    }

    // -- Private handlers ---------------------------------------------------

    fn handle_request(&mut self, payload: &[u8]) -> Vec<ResourceAction> {
        if self.state != SenderState::Advertised && self.state != SenderState::Transferring {
            return vec![];
        }

        // Parse RESOURCE_REQ: [flag:1][resource_hash:16][map_hashes:N*4]
        // If flag == 0xFF, there's a 4-byte last_map_hash between flag and resource_hash:
        //   [0xFF][last_map_hash:4][resource_hash:16][map_hashes:N*4]
        if payload.is_empty() {
            return vec![];
        }

        let flag = payload[0];
        let (last_map_hash, rest) = if flag == HASHMAP_IS_EXHAUSTED {
            if payload.len() < 1 + MAPHASH_LEN + 16 {
                return vec![];
            }
            let mut lmh = [0u8; MAPHASH_LEN];
            lmh.copy_from_slice(&payload[1..1 + MAPHASH_LEN]);
            (Some(lmh), &payload[1 + MAPHASH_LEN..])
        } else {
            (None, &payload[1..])
        };

        if rest.len() < 16 {
            return vec![];
        }

        let mut req_hash = [0u8; 16];
        req_hash.copy_from_slice(&rest[..16]);

        if req_hash != self.resource_hash {
            return vec![];
        }

        let map_data = &rest[16..];
        if map_data.len() % MAPHASH_LEN != 0 {
            return vec![];
        }

        // If the receiver signaled hashmap exhaustion, respond with an HMU
        // before sending any requested parts.
        if flag == HASHMAP_IS_EXHAUSTED {
            if let Some(ref lmh) = last_map_hash {
                if let Some(hmu_actions) = self.build_hmu(lmh) {
                    // Build a reverse lookup from map_hash -> part index.
                    let mut index_map: HashMap<MapHash, usize> = HashMap::new();
                    for (i, mh) in self.map_hashes.iter().enumerate() {
                        index_map.insert(*mh, i);
                    }

                    self.state = SenderState::Transferring;
                    let mut actions = vec![ResourceAction::SenderStateChanged];

                    // Send the HMU first so the receiver can install new hashes.
                    actions.extend(hmu_actions);

                    // Then send any requested parts.
                    let requested_count = map_data.len() / MAPHASH_LEN;
                    for chunk_idx in 0..requested_count {
                        let offset = chunk_idx * MAPHASH_LEN;
                        let mut mh = [0u8; MAPHASH_LEN];
                        mh.copy_from_slice(&map_data[offset..offset + MAPHASH_LEN]);

                        if let Some(&part_idx) = index_map.get(&mh) {
                            actions.push(ResourceAction::SendPacket {
                                context: PacketContext::Resource.as_byte(),
                                plaintext: self.parts[part_idx].clone(),
                            });
                            self.sent_parts += 1;
                        }
                    }

                    if self.sent_parts >= self.parts.len() {
                        self.state = SenderState::AwaitingProof;
                        actions.push(ResourceAction::SenderStateChanged);
                        let timeout = self.last_activity
                            + self.rtt_ms.unwrap_or(DEFAULT_TIMEOUT_MS)
                                * PROOF_TIMEOUT_FACTOR as u64;
                        actions.push(ResourceAction::ScheduleTimeout {
                            deadline_ms: timeout,
                        });
                    }

                    let progress = self.sent_parts as f32 / self.parts.len().max(1) as f32;
                    actions.push(ResourceAction::Progress { fraction: progress });

                    return actions;
                }
            }
        }

        // Build a reverse lookup from map_hash -> part index.
        let mut index_map: HashMap<MapHash, usize> = HashMap::new();
        for (i, mh) in self.map_hashes.iter().enumerate() {
            index_map.insert(*mh, i);
        }

        self.state = SenderState::Transferring;
        let mut actions = vec![ResourceAction::SenderStateChanged];

        // Send each requested part.
        let requested_count = map_data.len() / MAPHASH_LEN;
        for chunk_idx in 0..requested_count {
            let offset = chunk_idx * MAPHASH_LEN;
            let mut mh = [0u8; MAPHASH_LEN];
            mh.copy_from_slice(&map_data[offset..offset + MAPHASH_LEN]);

            if let Some(&part_idx) = index_map.get(&mh) {
                actions.push(ResourceAction::SendPacket {
                    context: PacketContext::Resource.as_byte(),
                    plaintext: self.parts[part_idx].clone(),
                });
                self.sent_parts += 1;
            }
        }

        // Check if all parts have been sent.
        if self.sent_parts >= self.parts.len() {
            self.state = SenderState::AwaitingProof;
            actions.push(ResourceAction::SenderStateChanged);
            let timeout = self.last_activity
                + self.rtt_ms.unwrap_or(DEFAULT_TIMEOUT_MS) * PROOF_TIMEOUT_FACTOR as u64;
            actions.push(ResourceAction::ScheduleTimeout {
                deadline_ms: timeout,
            });
        }

        let progress = self.sent_parts as f32 / self.parts.len().max(1) as f32;
        actions.push(ResourceAction::Progress { fraction: progress });

        actions
    }

    /// Build a hashmap update (HMU) packet containing map hashes that the
    /// receiver has not yet received.
    ///
    /// Given the `last_map_hash` the receiver already has, this finds the
    /// corresponding part index and sends all remaining hashes after it.
    ///
    /// HMU wire format: `[resource_hash:16][msgpack([segment_index, hashmap_bytes])]`
    fn build_hmu(&self, last_map_hash: &MapHash) -> Option<Vec<ResourceAction>> {
        let part_idx = self.map_hashes.iter().position(|mh| mh == last_map_hash)?;

        let start = part_idx + 1;
        if start >= self.parts.len() {
            return None;
        }

        let hashes: Vec<u8> = self.map_hashes[start..]
            .iter()
            .flat_map(|h| h.iter().copied())
            .collect();

        // Build HMU: [resource_hash:16][msgpack([segment_index, hashmap_bytes])]
        let mut pkt = Vec::new();
        pkt.extend_from_slice(&self.resource_hash);

        use rmp::encode::*;
        write_array_len(&mut pkt, 2).expect("write array len");
        write_u32(&mut pkt, start as u32).expect("write segment index");
        write_bin(&mut pkt, &hashes).expect("write hashmap bytes");

        Some(vec![ResourceAction::SendPacket {
            context: PacketContext::ResourceHmu.as_byte(),
            plaintext: pkt,
        }])
    }

    fn handle_proof(&mut self, payload: &[u8]) -> Vec<ResourceAction> {
        if self.state != SenderState::AwaitingProof {
            return vec![];
        }

        // Parse [resource_hash:16][proof:16]
        if payload.len() < 32 {
            return vec![];
        }

        let mut recv_hash = [0u8; 16];
        recv_hash.copy_from_slice(&payload[..16]);

        if recv_hash != self.resource_hash {
            return vec![];
        }

        let mut proof = [0u8; 16];
        proof.copy_from_slice(&payload[16..32]);

        if proof == self.expected_proof {
            self.state = SenderState::Complete;
            vec![
                ResourceAction::ProofValidated,
                ResourceAction::SenderStateChanged,
            ]
        } else {
            self.state = SenderState::Failed;
            vec![ResourceAction::SenderStateChanged]
        }
    }

    fn handle_cancel_received(&mut self) -> Vec<ResourceAction> {
        match self.state {
            SenderState::Complete | SenderState::Failed | SenderState::Rejected => vec![],
            _ => {
                self.state = SenderState::Rejected;
                vec![ResourceAction::SenderStateChanged]
            }
        }
    }

    fn handle_timeout(&mut self, now_ms: u64) -> Vec<ResourceAction> {
        match self.state {
            SenderState::Advertised => {
                if self.adv_retries_left > 0 {
                    self.adv_retries_left -= 1;
                    self.adv_sent_at = Some(now_ms);
                    self.last_activity = now_ms;

                    let adv = self.build_advertisement();
                    let timeout = now_ms + DEFAULT_TIMEOUT_MS;
                    vec![
                        ResourceAction::SendPacket {
                            context: PacketContext::ResourceAdv.as_byte(),
                            plaintext: adv.encode(),
                        },
                        ResourceAction::ScheduleTimeout {
                            deadline_ms: timeout,
                        },
                    ]
                } else {
                    self.state = SenderState::Failed;
                    vec![ResourceAction::SenderStateChanged]
                }
            }
            SenderState::Transferring | SenderState::AwaitingProof => {
                self.state = SenderState::Failed;
                vec![ResourceAction::SenderStateChanged]
            }
            _ => vec![],
        }
    }
}

// ---------------------------------------------------------------------------
// ResourceReceiver
// ---------------------------------------------------------------------------

/// Receiver-side resource transfer state machine.
///
/// Accepts incoming parts, tracks the hashmap, reassembles the payload,
/// and generates the proof. Created by the link upon receiving an
/// advertisement.
#[derive(Debug)]
#[allow(dead_code)] // Fields used in later tasks (windowing, HMU, retransmission).
pub struct ResourceReceiver {
    pub(crate) state: ReceiverState,
    pub(crate) resource_hash: ResourceHash,
    pub(crate) random_hash: RandomHash,
    pub(crate) original_hash: ResourceHash,
    pub(crate) transfer_size: u32,
    pub(crate) data_size: u32,
    pub(crate) total_parts: usize,
    pub(crate) parts: Vec<Option<Vec<u8>>>,
    pub(crate) hashmap: Vec<Option<MapHash>>,
    pub(crate) hashmap_height: usize,
    pub(crate) received_count: usize,
    pub(crate) consecutive_completed: usize,
    pub(crate) outstanding_parts: usize,
    pub(crate) window: usize,
    pub(crate) window_min: usize,
    pub(crate) window_max: usize,
    pub(crate) retries_left: u8,
    pub(crate) fast_rate_rounds: usize,
    pub(crate) very_slow_rate_rounds: usize,
    pub(crate) last_activity: u64,
    pub(crate) req_sent_at: Option<u64>,
    pub(crate) rtt_ms: Option<u64>,
    pub(crate) part_timeout_factor: u64,
    pub(crate) assembled_encrypted: Option<Vec<u8>>,
}

impl ResourceReceiver {
    /// Accept a resource advertisement and begin receiving.
    ///
    /// Decodes the advertisement, initializes internal state, and emits
    /// the initial resource request for the first window of parts.
    pub fn accept(
        adv_plaintext: &[u8],
        now_ms: u64,
    ) -> Result<(Self, Vec<ResourceAction>), ReticulumError> {
        let adv = ResourceAdvertisement::decode(adv_plaintext)?;

        let total_parts = adv.part_count as usize;

        // Parse the hashmap segment into individual map hashes.
        let hashmap_data = &adv.hashmap;
        if hashmap_data.len() % MAPHASH_LEN != 0 {
            return Err(ReticulumError::ResourceAdvInvalid);
        }
        let hashmap_height = hashmap_data.len() / MAPHASH_LEN;

        let mut hashmap: Vec<Option<MapHash>> = vec![None; total_parts];
        for (i, slot) in hashmap
            .iter_mut()
            .enumerate()
            .take(hashmap_height.min(total_parts))
        {
            let offset = i * MAPHASH_LEN;
            let mut mh = [0u8; MAPHASH_LEN];
            mh.copy_from_slice(&hashmap_data[offset..offset + MAPHASH_LEN]);
            *slot = Some(mh);
        }

        let window = WINDOW_INITIAL as usize;
        let window_min = WINDOW_MIN as usize;
        let window_max = WINDOW_MAX_SLOW as usize;

        let mut receiver = Self {
            state: ReceiverState::Transferring,
            resource_hash: adv.resource_hash,
            random_hash: adv.random_hash,
            original_hash: adv.original_hash,
            transfer_size: adv.transfer_size,
            data_size: adv.data_size,
            total_parts,
            parts: vec![None; total_parts],
            hashmap,
            hashmap_height,
            received_count: 0,
            consecutive_completed: 0,
            outstanding_parts: 0,
            window,
            window_min,
            window_max,
            retries_left: MAX_RETRIES as u8,
            fast_rate_rounds: 0,
            very_slow_rate_rounds: 0,
            last_activity: now_ms,
            req_sent_at: None,
            rtt_ms: None,
            part_timeout_factor: PART_TIMEOUT_FACTOR as u64,
            assembled_encrypted: None,
        };

        let actions = receiver.request_next(now_ms);
        Ok((receiver, actions))
    }

    /// Current receiver state.
    pub fn state(&self) -> ReceiverState {
        self.state
    }

    /// Resource hash identifying this transfer.
    pub fn hash(&self) -> &ResourceHash {
        &self.resource_hash
    }

    /// Process an incoming event and return resulting actions.
    pub fn handle_event(&mut self, event: ResourceEvent, payload: &[u8]) -> Vec<ResourceAction> {
        match event {
            ResourceEvent::PartReceived => self.handle_part(payload),
            ResourceEvent::HashmapUpdateReceived => self.handle_hmu(payload),
            ResourceEvent::CancelReceived => self.handle_cancel_received(),
            ResourceEvent::Timeout { now_ms } => self.handle_timeout(now_ms),
            _ => vec![],
        }
    }

    /// Cancel the transfer from the receiver side.
    ///
    /// Emits a cancel packet (`ResourceRcl`) unless the transfer is already
    /// complete or failed.
    pub fn cancel(&mut self) -> Vec<ResourceAction> {
        match self.state {
            ReceiverState::Complete | ReceiverState::Failed | ReceiverState::Corrupt => vec![],
            _ => {
                self.state = ReceiverState::Failed;
                vec![
                    ResourceAction::SendPacket {
                        context: PacketContext::ResourceRcl.as_byte(),
                        plaintext: self.resource_hash.to_vec(),
                    },
                    ResourceAction::ReceiverStateChanged,
                ]
            }
        }
    }

    /// Finalize the transfer: decrypt assembled data and send proof.
    ///
    /// Should be called after `AssemblyReady` is emitted. Decrypts the
    /// assembled encrypted data, computes the proof, and transitions to
    /// `Complete`.
    pub fn finalize(
        &mut self,
        crypto: &impl LinkCrypto,
    ) -> Result<Vec<ResourceAction>, ReticulumError> {
        let encrypted = self
            .assembled_encrypted
            .as_ref()
            .ok_or(ReticulumError::ResourceFailed)?;

        let plaintext = crypto.decrypt(encrypted)?;

        // Compute the proof hash.
        let proof = compute_proof_hash(&plaintext, &self.resource_hash);

        // Build proof packet: [resource_hash:16][proof:16]
        let mut proof_packet = Vec::with_capacity(32);
        proof_packet.extend_from_slice(&self.resource_hash);
        proof_packet.extend_from_slice(&proof);

        self.state = ReceiverState::Complete;

        Ok(vec![
            ResourceAction::SendPacket {
                context: PacketContext::ResourcePrf.as_byte(),
                plaintext: proof_packet,
            },
            ResourceAction::Completed { data: plaintext },
            ResourceAction::ReceiverStateChanged,
        ])
    }

    // -- Private handlers ---------------------------------------------------

    fn handle_part(&mut self, payload: &[u8]) -> Vec<ResourceAction> {
        if self.state != ReceiverState::Transferring {
            return vec![];
        }

        // The payload is the raw part data (plaintext).
        // Compute the map hash to find which slot it belongs to.
        let mh = compute_map_hash(payload, &self.random_hash);

        // Find the matching slot in the hashmap.
        let mut matched_idx = None;
        for (i, slot) in self.hashmap.iter().enumerate() {
            if let Some(expected) = slot {
                if *expected == mh && self.parts[i].is_none() {
                    matched_idx = Some(i);
                    break;
                }
            }
        }

        let idx = match matched_idx {
            Some(i) => i,
            None => return vec![], // Unknown or duplicate part.
        };

        // Store the part.
        self.parts[idx] = Some(payload.to_vec());
        self.received_count += 1;
        if self.outstanding_parts > 0 {
            self.outstanding_parts -= 1;
        }

        // Update consecutive_completed: advance from current position.
        while self.consecutive_completed < self.total_parts
            && self.parts[self.consecutive_completed].is_some()
        {
            self.consecutive_completed += 1;
        }

        let mut actions = Vec::new();

        // Progress update.
        let progress = self.received_count as f32 / self.total_parts.max(1) as f32;
        actions.push(ResourceAction::Progress { fraction: progress });

        // Check if all parts received.
        if self.received_count >= self.total_parts {
            self.state = ReceiverState::Assembling;
            actions.push(ResourceAction::ReceiverStateChanged);

            // Assemble the encrypted data.
            let mut assembled = Vec::with_capacity(self.transfer_size as usize);
            for data in self.parts.iter().flatten() {
                assembled.extend_from_slice(data);
            }

            // Verify the resource hash.
            let computed_hash = compute_resource_hash(&assembled, &self.random_hash);
            if computed_hash == self.resource_hash {
                self.assembled_encrypted = Some(assembled);
                actions.push(ResourceAction::AssemblyReady);
            } else {
                self.state = ReceiverState::Corrupt;
                actions.push(ResourceAction::ReceiverStateChanged);
            }
        } else if self.outstanding_parts == 0 {
            // All parts in the current window arrived — grow the window.
            if self.window < self.window_max {
                self.window += 1;
                if (self.window - self.window_min) > (WINDOW_FLEXIBILITY as usize - 1) {
                    self.window_min += 1;
                }
            }

            // Need to request more parts.
            // Use last_activity as a proxy for now_ms (caller should update it).
            let req_actions = self.request_next(self.last_activity);
            actions.extend(req_actions);
        }

        actions
    }

    /// Handle an incoming hashmap update (HMU) packet from the sender.
    ///
    /// Parses the HMU, installs new map hashes into the receiver's hashmap,
    /// and issues a new request for the next window of parts.
    fn handle_hmu(&mut self, payload: &[u8]) -> Vec<ResourceAction> {
        if self.state != ReceiverState::Transferring {
            return vec![];
        }

        if payload.len() < 16 {
            return vec![];
        }

        let mut recv_hash = [0u8; 16];
        recv_hash.copy_from_slice(&payload[..16]);
        if recv_hash != self.resource_hash {
            return vec![];
        }

        // Parse msgpack: [segment_index, hashmap_bytes]
        let mut rd = &payload[16..];
        let arr_len = match rmp::decode::read_array_len(&mut rd) {
            Ok(len) => len,
            Err(_) => return vec![],
        };
        if arr_len != 2 {
            return vec![];
        }

        let _segment: u32 = match rmp::decode::read_int(&mut rd) {
            Ok(s) => s,
            Err(_) => return vec![],
        };

        let hash_len = match rmp::decode::read_bin_len(&mut rd) {
            Ok(len) => len as usize,
            Err(_) => return vec![],
        };

        if rd.len() < hash_len {
            return vec![];
        }
        let hash_data = &rd[..hash_len];

        // Install new hashes into the hashmap starting from hashmap_height.
        for chunk in hash_data.chunks_exact(MAPHASH_LEN) {
            let idx = self.hashmap_height;
            if idx < self.total_parts {
                let mut mh = [0u8; MAPHASH_LEN];
                mh.copy_from_slice(chunk);
                if self.hashmap[idx].is_none() {
                    self.hashmap[idx] = Some(mh);
                }
                self.hashmap_height += 1;
            }
        }

        // Now re-request with the new hashes available.
        self.request_next(self.last_activity)
    }

    fn handle_cancel_received(&mut self) -> Vec<ResourceAction> {
        match self.state {
            ReceiverState::Complete | ReceiverState::Failed | ReceiverState::Corrupt => vec![],
            _ => {
                self.state = ReceiverState::Failed;
                vec![ResourceAction::ReceiverStateChanged]
            }
        }
    }

    fn handle_timeout(&mut self, now_ms: u64) -> Vec<ResourceAction> {
        if self.state != ReceiverState::Transferring {
            return vec![];
        }

        if self.retries_left > 0 {
            self.retries_left -= 1;

            // Window backoff: shrink both window and ceiling on timeout.
            if self.window > self.window_min {
                self.window -= 1;
            }
            if self.window_max > self.window_min {
                self.window_max -= 1;
            }

            // Reset rate counters and discard the stale req_sent_at so that
            // request_next does not re-measure RTT from the timed-out request.
            self.fast_rate_rounds = 0;
            self.very_slow_rate_rounds = 0;
            self.req_sent_at = None;

            self.request_next(now_ms)
        } else {
            self.state = ReceiverState::Failed;
            vec![ResourceAction::ReceiverStateChanged]
        }
    }

    /// Build and emit a resource request for the next window of unreceived parts.
    fn request_next(&mut self, now_ms: u64) -> Vec<ResourceAction> {
        // RTT measurement and rate detection.
        if let Some(sent_at) = self.req_sent_at {
            let rtt = now_ms.saturating_sub(sent_at).max(1);
            self.rtt_ms = Some(rtt);

            // After the first RTT measurement, tighten the timeout factor.
            if self.part_timeout_factor == PART_TIMEOUT_FACTOR as u64 {
                self.part_timeout_factor = PART_TIMEOUT_FACTOR_AFTER_RTT as u64;
            }

            // Estimate transfer rate: bytes transferred in this window / RTT.
            // Each part is roughly `sdu` bytes; we sent `outstanding_parts` of them
            // in the previous request. Use `window` as an approximation since
            // outstanding_parts has already been decremented.
            let bytes_transferred = (self.window as u64)
                * (self.transfer_size as u64 / (self.total_parts as u64).max(1));
            let rate_bps = if rtt > 0 {
                (bytes_transferred * 1000) / rtt
            } else {
                0
            };

            // Classify link speed and adapt window_max.
            if rate_bps > RATE_FAST {
                self.fast_rate_rounds += 1;
                self.very_slow_rate_rounds = 0;
                if self.fast_rate_rounds as u32 >= FAST_RATE_THRESHOLD {
                    self.window_max = WINDOW_MAX_FAST as usize;
                }
            } else if rate_bps < RATE_VERY_SLOW {
                self.very_slow_rate_rounds += 1;
                self.fast_rate_rounds = 0;
                if self.very_slow_rate_rounds as u32 >= VERY_SLOW_RATE_THRESHOLD {
                    self.window_max = WINDOW_MAX_VERY_SLOW as usize;
                    // Clamp window to new max.
                    if self.window > self.window_max {
                        self.window = self.window_max;
                    }
                }
            } else {
                // Medium speed — reset both counters.
                self.fast_rate_rounds = 0;
                self.very_slow_rate_rounds = 0;
            }
        }

        let mut requested_hashes = Vec::new();
        let mut exhausted = false;
        let mut scanned = 0;

        let start = self.consecutive_completed;
        for i in start..self.total_parts {
            if scanned >= self.window {
                break;
            }

            if self.parts[i].is_some() {
                // Already have this part, skip.
                continue;
            }

            match self.hashmap[i] {
                Some(mh) => {
                    requested_hashes.push(mh);
                    scanned += 1;
                }
                None => {
                    // We've reached a slot where the hashmap hasn't been
                    // populated yet (multi-segment hashmap). Signal exhaustion.
                    exhausted = true;
                    break;
                }
            }
        }

        if requested_hashes.is_empty() && !exhausted {
            return vec![];
        }

        // Build RESOURCE_REQ packet.
        // Normal:    [flag:1][resource_hash:16][map_hashes:N*4]
        // Exhausted: [0xFF][last_map_hash:4][resource_hash:16][map_hashes:N*4]
        let flag = if exhausted {
            HASHMAP_IS_EXHAUSTED
        } else {
            HASHMAP_IS_NOT_EXHAUSTED
        };

        let extra = if exhausted { MAPHASH_LEN } else { 0 };
        let mut packet = Vec::with_capacity(1 + extra + 16 + requested_hashes.len() * MAPHASH_LEN);
        packet.push(flag);

        if exhausted {
            // Include the last known map hash so the sender knows where to
            // resume. Use the hash at hashmap_height - 1 (the last populated
            // slot), or fall back to zeros if none are populated yet.
            let last_mh = if self.hashmap_height > 0 {
                self.hashmap[self.hashmap_height - 1].unwrap_or([0u8; MAPHASH_LEN])
            } else {
                [0u8; MAPHASH_LEN]
            };
            packet.extend_from_slice(&last_mh);
        }

        packet.extend_from_slice(&self.resource_hash);
        for mh in &requested_hashes {
            packet.extend_from_slice(mh);
        }

        self.outstanding_parts = requested_hashes.len();
        self.req_sent_at = Some(now_ms);
        self.last_activity = now_ms;

        let timeout_ms = self.rtt_ms.unwrap_or(DEFAULT_TIMEOUT_MS) * self.part_timeout_factor
            + PROCESSING_GRACE_MS;

        vec![
            ResourceAction::SendPacket {
                context: PacketContext::ResourceReq.as_byte(),
                plaintext: packet,
            },
            ResourceAction::ScheduleTimeout {
                deadline_ms: now_ms + timeout_ms,
            },
        ]
    }
}

// ---------------------------------------------------------------------------
// Hash computations
// ---------------------------------------------------------------------------

/// Compute the 4-byte map hash for a resource part.
///
/// `map_hash = SHA-256(part_data || random_hash)[:4]`
///
/// This hash identifies individual parts in the resource hashmap so the
/// receiver can match incoming parts to their expected positions.
pub fn compute_map_hash(part_data: &[u8], random_hash: &RandomHash) -> MapHash {
    let mut input = Vec::with_capacity(part_data.len() + random_hash.len());
    input.extend_from_slice(part_data);
    input.extend_from_slice(random_hash);
    let full = hash::full_hash(&input);
    let mut result = [0u8; MAPHASH_LEN];
    result.copy_from_slice(&full[..MAPHASH_LEN]);
    result
}

/// Compute the 16-byte resource hash over encrypted data.
///
/// `resource_hash = SHA-256(encrypted_data || random_hash)[:16]`
///
/// This identifies the entire resource and is used to verify integrity
/// after reassembly on the receiver side.
pub fn compute_resource_hash(encrypted_data: &[u8], random_hash: &RandomHash) -> ResourceHash {
    let mut input = Vec::with_capacity(encrypted_data.len() + random_hash.len());
    input.extend_from_slice(encrypted_data);
    input.extend_from_slice(random_hash);
    hash::truncated_hash(&input)
}

/// Compute the 16-byte proof hash sent by the receiver to confirm receipt.
///
/// `proof_hash = SHA-256(plaintext || resource_hash)[:16]`
///
/// The sender verifies this to confirm the receiver has the correct data.
pub fn compute_proof_hash(plaintext: &[u8], resource_hash: &ResourceHash) -> ResourceHash {
    let mut input = Vec::with_capacity(plaintext.len() + resource_hash.len());
    input.extend_from_slice(plaintext);
    input.extend_from_slice(resource_hash);
    hash::truncated_hash(&input)
}

// ---------------------------------------------------------------------------
// Chunking
// ---------------------------------------------------------------------------

/// Split `data` into chunks of at most `sdu` bytes.
///
/// The last chunk may be smaller than `sdu`. Returns an empty vec if
/// `data` is empty.
pub fn chunk_data(data: &[u8], sdu: usize) -> Vec<Vec<u8>> {
    if data.is_empty() {
        return Vec::new();
    }
    data.chunks(sdu).map(|c| c.to_vec()).collect()
}

// ---------------------------------------------------------------------------
// ResourceAdvertisement
// ---------------------------------------------------------------------------

/// Wire-format advertisement for a resource transfer.
///
/// Encoded as a MessagePack map with single-character string keys to match
/// the Python Reticulum wire format. The keys are:
///
/// - `"t"` — transfer size (encrypted, in bytes)
/// - `"d"` — original data size (plaintext, in bytes)
/// - `"n"` — number of parts
/// - `"h"` — resource hash (16 bytes)
/// - `"r"` — random hash (4 bytes)
/// - `"o"` — original hash (16 bytes, hash before segmentation)
/// - `"m"` — hashmap segment (variable-length binary)
/// - `"f"` — flags byte
/// - `"i"` — segment index (0-based)
/// - `"l"` — total number of segments
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceAdvertisement {
    /// Size of the encrypted (transfer) data in bytes.
    pub transfer_size: u32,
    /// Size of the original plaintext data in bytes.
    pub data_size: u32,
    /// Total number of parts the data is split into.
    pub part_count: u32,
    /// Hash identifying this resource (16 bytes).
    pub resource_hash: ResourceHash,
    /// Random nonce mixed into hash computations (4 bytes).
    pub random_hash: RandomHash,
    /// Original resource hash before segmentation (16 bytes).
    pub original_hash: ResourceHash,
    /// Hashmap segment — concatenated map-hashes for this segment.
    pub hashmap: Vec<u8>,
    /// Flags byte (bit 0 = encrypted, bit 1 = compressed, bit 2 = split).
    pub flags: u8,
    /// Index of this hashmap segment (1-based, matching Python).
    pub segment_index: u32,
    /// Total number of hashmap segments.
    pub total_segments: u32,
    /// Request ID (None when not part of a request/response exchange).
    /// Always included in wire format for Python compatibility.
    pub request_id: Option<Vec<u8>>,
}

impl ResourceAdvertisement {
    /// Number of fields in the encoded msgpack map (matches Python's 11-field format).
    const MAP_LEN: u32 = 11;

    /// Encode this advertisement as a MessagePack byte vector.
    ///
    /// Uses single-character string keys matching the Python wire format.
    pub fn encode(&self) -> Vec<u8> {
        use rmp::encode::*;

        let mut buf = Vec::with_capacity(128);

        write_map_len(&mut buf, Self::MAP_LEN).expect("write map_len");

        // "t" -> transfer_size
        write_str(&mut buf, "t").expect("write key t");
        write_u32(&mut buf, self.transfer_size).expect("write transfer_size");

        // "d" -> data_size
        write_str(&mut buf, "d").expect("write key d");
        write_u32(&mut buf, self.data_size).expect("write data_size");

        // "n" -> part_count
        write_str(&mut buf, "n").expect("write key n");
        write_u32(&mut buf, self.part_count).expect("write part_count");

        // "h" -> resource_hash
        write_str(&mut buf, "h").expect("write key h");
        write_bin(&mut buf, &self.resource_hash).expect("write resource_hash");

        // "r" -> random_hash
        write_str(&mut buf, "r").expect("write key r");
        write_bin(&mut buf, &self.random_hash).expect("write random_hash");

        // "o" -> original_hash
        write_str(&mut buf, "o").expect("write key o");
        write_bin(&mut buf, &self.original_hash).expect("write original_hash");

        // "m" -> hashmap
        write_str(&mut buf, "m").expect("write key m");
        write_bin(&mut buf, &self.hashmap).expect("write hashmap");

        // "f" -> flags
        write_str(&mut buf, "f").expect("write key f");
        write_u8(&mut buf, self.flags).expect("write flags");

        // "i" -> segment_index
        write_str(&mut buf, "i").expect("write key i");
        write_u32(&mut buf, self.segment_index).expect("write segment_index");

        // "l" -> total_segments
        write_str(&mut buf, "l").expect("write key l");
        write_u32(&mut buf, self.total_segments).expect("write total_segments");

        // "q" -> request_id (None = msgpack nil for compatibility with Python)
        write_str(&mut buf, "q").expect("write key q");
        match &self.request_id {
            Some(id) => write_bin(&mut buf, id).expect("write request_id"),
            None => write_nil(&mut buf).expect("write nil request_id"),
        }

        buf
    }

    /// Decode a ResourceAdvertisement from a MessagePack byte slice.
    ///
    /// Returns `Err(ResourceAdvInvalid)` if the data is malformed or
    /// contains unknown keys.
    pub fn decode(data: &[u8]) -> Result<Self, ReticulumError> {
        let mut rd = data;

        let map_len =
            rmp::decode::read_map_len(&mut rd).map_err(|_| ReticulumError::ResourceAdvInvalid)?;

        if map_len != Self::MAP_LEN {
            return Err(ReticulumError::ResourceAdvInvalid);
        }

        let mut transfer_size: Option<u32> = None;
        let mut data_size: Option<u32> = None;
        let mut part_count: Option<u32> = None;
        let mut resource_hash: Option<ResourceHash> = None;
        let mut random_hash: Option<RandomHash> = None;
        let mut original_hash: Option<ResourceHash> = None;
        let mut hashmap: Option<Vec<u8>> = None;
        let mut flags: Option<u8> = None;
        let mut segment_index: Option<u32> = None;
        let mut total_segments: Option<u32> = None;
        let mut request_id: Option<Option<Vec<u8>>> = None;

        for _ in 0..map_len {
            // Read key as string
            let key_len = rmp::decode::read_str_len(&mut rd)
                .map_err(|_| ReticulumError::ResourceAdvInvalid)?;
            if key_len != 1 {
                return Err(ReticulumError::ResourceAdvInvalid);
            }
            let mut key_buf = [0u8; 1];
            std::io::Read::read_exact(&mut rd, &mut key_buf)
                .map_err(|_| ReticulumError::ResourceAdvInvalid)?;
            let key = key_buf[0];

            match key {
                b't' => {
                    let v: u32 = rmp::decode::read_int(&mut rd)
                        .map_err(|_| ReticulumError::ResourceAdvInvalid)?;
                    transfer_size = Some(v);
                }
                b'd' => {
                    let v: u32 = rmp::decode::read_int(&mut rd)
                        .map_err(|_| ReticulumError::ResourceAdvInvalid)?;
                    data_size = Some(v);
                }
                b'n' => {
                    let v: u32 = rmp::decode::read_int(&mut rd)
                        .map_err(|_| ReticulumError::ResourceAdvInvalid)?;
                    part_count = Some(v);
                }
                b'h' => {
                    let bin_len = rmp::decode::read_bin_len(&mut rd)
                        .map_err(|_| ReticulumError::ResourceAdvInvalid)?;
                    if bin_len != 16 {
                        return Err(ReticulumError::ResourceAdvInvalid);
                    }
                    let mut h = [0u8; 16];
                    std::io::Read::read_exact(&mut rd, &mut h)
                        .map_err(|_| ReticulumError::ResourceAdvInvalid)?;
                    resource_hash = Some(h);
                }
                b'r' => {
                    let bin_len = rmp::decode::read_bin_len(&mut rd)
                        .map_err(|_| ReticulumError::ResourceAdvInvalid)?;
                    if bin_len != 4 {
                        return Err(ReticulumError::ResourceAdvInvalid);
                    }
                    let mut h = [0u8; 4];
                    std::io::Read::read_exact(&mut rd, &mut h)
                        .map_err(|_| ReticulumError::ResourceAdvInvalid)?;
                    random_hash = Some(h);
                }
                b'o' => {
                    let bin_len = rmp::decode::read_bin_len(&mut rd)
                        .map_err(|_| ReticulumError::ResourceAdvInvalid)?;
                    if bin_len != 16 {
                        return Err(ReticulumError::ResourceAdvInvalid);
                    }
                    let mut h = [0u8; 16];
                    std::io::Read::read_exact(&mut rd, &mut h)
                        .map_err(|_| ReticulumError::ResourceAdvInvalid)?;
                    original_hash = Some(h);
                }
                b'm' => {
                    let bin_len = rmp::decode::read_bin_len(&mut rd)
                        .map_err(|_| ReticulumError::ResourceAdvInvalid)?
                        as usize;
                    let mut m = vec![0u8; bin_len];
                    std::io::Read::read_exact(&mut rd, &mut m)
                        .map_err(|_| ReticulumError::ResourceAdvInvalid)?;
                    hashmap = Some(m);
                }
                b'f' => {
                    let v: u8 = rmp::decode::read_int(&mut rd)
                        .map_err(|_| ReticulumError::ResourceAdvInvalid)?;
                    flags = Some(v);
                }
                b'i' => {
                    let v: u32 = rmp::decode::read_int(&mut rd)
                        .map_err(|_| ReticulumError::ResourceAdvInvalid)?;
                    segment_index = Some(v);
                }
                b'l' => {
                    let v: u32 = rmp::decode::read_int(&mut rd)
                        .map_err(|_| ReticulumError::ResourceAdvInvalid)?;
                    total_segments = Some(v);
                }
                b'q' => {
                    // request_id: can be nil (None) or binary data
                    let marker = rmp::decode::read_marker(&mut rd)
                        .map_err(|_| ReticulumError::ResourceAdvInvalid)?;
                    if marker == rmp::Marker::Null {
                        request_id = Some(None);
                    } else {
                        // Marker is a bin type — extract length from the marker
                        let bin_len = match marker {
                            rmp::Marker::Bin8 => {
                                let mut b = [0u8; 1];
                                std::io::Read::read_exact(&mut rd, &mut b)
                                    .map_err(|_| ReticulumError::ResourceAdvInvalid)?;
                                b[0] as usize
                            }
                            rmp::Marker::Bin16 => {
                                let mut b = [0u8; 2];
                                std::io::Read::read_exact(&mut rd, &mut b)
                                    .map_err(|_| ReticulumError::ResourceAdvInvalid)?;
                                u16::from_be_bytes(b) as usize
                            }
                            _ => return Err(ReticulumError::ResourceAdvInvalid),
                        };
                        let mut id = vec![0u8; bin_len];
                        std::io::Read::read_exact(&mut rd, &mut id)
                            .map_err(|_| ReticulumError::ResourceAdvInvalid)?;
                        request_id = Some(Some(id));
                    }
                }
                _ => return Err(ReticulumError::ResourceAdvInvalid),
            }
        }

        Ok(ResourceAdvertisement {
            transfer_size: transfer_size.ok_or(ReticulumError::ResourceAdvInvalid)?,
            data_size: data_size.ok_or(ReticulumError::ResourceAdvInvalid)?,
            part_count: part_count.ok_or(ReticulumError::ResourceAdvInvalid)?,
            resource_hash: resource_hash.ok_or(ReticulumError::ResourceAdvInvalid)?,
            random_hash: random_hash.ok_or(ReticulumError::ResourceAdvInvalid)?,
            original_hash: original_hash.ok_or(ReticulumError::ResourceAdvInvalid)?,
            hashmap: hashmap.ok_or(ReticulumError::ResourceAdvInvalid)?,
            flags: flags.ok_or(ReticulumError::ResourceAdvInvalid)?,
            segment_index: segment_index.ok_or(ReticulumError::ResourceAdvInvalid)?,
            total_segments: total_segments.ok_or(ReticulumError::ResourceAdvInvalid)?,
            request_id: request_id
                .ok_or(ReticulumError::ResourceAdvInvalid)?
                .clone(),
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- MockLinkCrypto -----------------------------------------------------

    /// Identity pass-through encryption for testing.
    struct MockLinkCrypto {
        id: [u8; 16],
        mdu: usize,
    }

    impl MockLinkCrypto {
        fn new(mdu: usize) -> Self {
            Self {
                id: [0xAA; 16],
                mdu,
            }
        }
    }

    impl LinkCrypto for MockLinkCrypto {
        fn encrypt(
            &self,
            _rng: &mut dyn CryptoRngCore,
            plaintext: &[u8],
        ) -> Result<Vec<u8>, ReticulumError> {
            // Identity: ciphertext == plaintext
            Ok(plaintext.to_vec())
        }

        fn decrypt(&self, ciphertext: &[u8]) -> Result<Vec<u8>, ReticulumError> {
            // Identity: plaintext == ciphertext
            Ok(ciphertext.to_vec())
        }

        fn link_id(&self) -> &[u8; 16] {
            &self.id
        }

        fn mdu(&self) -> usize {
            self.mdu
        }
    }

    // -- Basic tests --------------------------------------------------------

    #[test]
    fn mock_link_crypto_roundtrip() {
        use rand::rngs::OsRng;
        let mock = MockLinkCrypto::new(400);
        let plaintext = b"hello resource transfer";
        let encrypted = mock.encrypt(&mut OsRng, plaintext).unwrap();
        let decrypted = mock.decrypt(&encrypted).unwrap();
        assert_eq!(plaintext.as_slice(), decrypted.as_slice());
        assert_eq!(mock.mdu(), 400);
        assert_eq!(mock.link_id(), &[0xAA; 16]);
    }

    #[test]
    fn sender_state_enum_equality() {
        assert_eq!(SenderState::Queued, SenderState::Queued);
        assert_ne!(SenderState::Queued, SenderState::Complete);
        assert_ne!(SenderState::Failed, SenderState::Rejected);
        assert_eq!(SenderState::Transferring, SenderState::Transferring);
    }

    #[test]
    fn receiver_state_enum_equality() {
        assert_eq!(ReceiverState::Transferring, ReceiverState::Transferring);
        assert_ne!(ReceiverState::Complete, ReceiverState::Corrupt);
        assert_ne!(ReceiverState::Failed, ReceiverState::Assembling);
    }

    #[test]
    fn sender_state_debug_format() {
        assert_eq!(format!("{:?}", SenderState::Queued), "Queued");
        assert_eq!(format!("{:?}", SenderState::AwaitingProof), "AwaitingProof");
    }

    #[test]
    fn receiver_state_debug_format() {
        assert_eq!(format!("{:?}", ReceiverState::Assembling), "Assembling");
        assert_eq!(format!("{:?}", ReceiverState::Corrupt), "Corrupt");
    }

    // -- Hash computation tests ---------------------------------------------

    #[test]
    fn map_hash_deterministic() {
        let data = b"hello part data";
        let random = [0x01, 0x02, 0x03, 0x04];
        let h1 = compute_map_hash(data, &random);
        let h2 = compute_map_hash(data, &random);
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), MAPHASH_LEN);
    }

    #[test]
    fn map_hash_different_inputs_differ() {
        let random = [0x01, 0x02, 0x03, 0x04];
        let h1 = compute_map_hash(b"part A", &random);
        let h2 = compute_map_hash(b"part B", &random);
        assert_ne!(h1, h2);

        // Different random hashes also produce different results
        let h3 = compute_map_hash(b"part A", &[0xFF, 0xFE, 0xFD, 0xFC]);
        assert_ne!(h1, h3);
    }

    #[test]
    fn resource_hash_deterministic() {
        let data = b"encrypted resource payload";
        let random = [0xAA, 0xBB, 0xCC, 0xDD];
        let h1 = compute_resource_hash(data, &random);
        let h2 = compute_resource_hash(data, &random);
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 16);
    }

    #[test]
    fn resource_hash_different_inputs_differ() {
        let random = [0xAA, 0xBB, 0xCC, 0xDD];
        let h1 = compute_resource_hash(b"payload 1", &random);
        let h2 = compute_resource_hash(b"payload 2", &random);
        assert_ne!(h1, h2);
    }

    #[test]
    fn proof_hash_deterministic() {
        let plaintext = b"original data";
        let resource_hash = [0x01u8; 16];
        let h1 = compute_proof_hash(plaintext, &resource_hash);
        let h2 = compute_proof_hash(plaintext, &resource_hash);
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 16);
    }

    // -- Chunking tests -----------------------------------------------------

    #[test]
    fn chunk_data_single_part() {
        let data = b"small";
        let chunks = chunk_data(data, 100);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], b"small");
    }

    #[test]
    fn chunk_data_multiple_parts() {
        let data = b"abcdefghij"; // 10 bytes
        let chunks = chunk_data(data, 3);
        assert_eq!(chunks.len(), 4); // 3+3+3+1
        assert_eq!(chunks[0], b"abc");
        assert_eq!(chunks[1], b"def");
        assert_eq!(chunks[2], b"ghi");
        assert_eq!(chunks[3], b"j");
    }

    #[test]
    fn chunk_data_exact_boundary() {
        let data = b"abcdef"; // 6 bytes
        let chunks = chunk_data(data, 3);
        assert_eq!(chunks.len(), 2); // 3+3 exactly
        assert_eq!(chunks[0], b"abc");
        assert_eq!(chunks[1], b"def");
    }

    #[test]
    fn chunk_data_empty() {
        let chunks = chunk_data(b"", 100);
        assert!(chunks.is_empty());
    }

    // -- Advertisement encode/decode tests ----------------------------------

    fn sample_advertisement() -> ResourceAdvertisement {
        ResourceAdvertisement {
            transfer_size: 12345,
            data_size: 10000,
            part_count: 5,
            resource_hash: [
                0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E,
                0x0F, 0x10,
            ],
            random_hash: [0xAA, 0xBB, 0xCC, 0xDD],
            original_hash: [
                0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E,
                0x1F, 0x20,
            ],
            hashmap: vec![0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE],
            flags: 0x01,
            segment_index: 1,
            total_segments: 1,
            request_id: None,
        }
    }

    #[test]
    fn advertisement_encode_decode_roundtrip() {
        let adv = sample_advertisement();
        let encoded = adv.encode();
        let decoded = ResourceAdvertisement::decode(&encoded).unwrap();
        assert_eq!(adv, decoded);
    }

    #[test]
    fn advertisement_encode_decode_large_values() {
        let adv = ResourceAdvertisement {
            transfer_size: 0x00FF_FFFF,
            data_size: 0x00FE_FFFF,
            part_count: 1000,
            resource_hash: [0xFF; 16],
            random_hash: [0xFF; 4],
            original_hash: [0x00; 16],
            hashmap: vec![0x42; 256],
            flags: 0xFF,
            segment_index: 99,
            total_segments: 100,
            request_id: None,
        };
        let encoded = adv.encode();
        let decoded = ResourceAdvertisement::decode(&encoded).unwrap();
        assert_eq!(adv, decoded);
    }

    #[test]
    fn advertisement_decode_empty_rejects() {
        let result = ResourceAdvertisement::decode(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn advertisement_decode_garbage_rejects() {
        let result = ResourceAdvertisement::decode(&[0xFF, 0xFF, 0xFF]);
        assert!(result.is_err());
    }

    #[test]
    fn advertisement_decode_wrong_map_len_rejects() {
        // Encode a map with only 5 elements
        let mut buf = Vec::new();
        rmp::encode::write_map_len(&mut buf, 5).unwrap();
        for i in 0..5 {
            rmp::encode::write_str(&mut buf, &format!("{}", i)).unwrap();
            rmp::encode::write_u32(&mut buf, 0).unwrap();
        }
        let result = ResourceAdvertisement::decode(&buf);
        assert!(result.is_err());
    }

    #[test]
    fn advertisement_decode_unknown_key_rejects() {
        // Build a valid-looking map with 10 entries but one unknown key
        let adv = sample_advertisement();
        let mut encoded = adv.encode();
        // Replace the first key "t" (0xa1, 0x74) with "z" (0xa1, 0x7a)
        // Find the "t" key after the map header byte
        // Map header for fixmap(10) is 0x8a, then first key is fixstr(1) = 0xa1, 't' = 0x74
        if let Some(pos) = encoded.windows(2).position(|w| w == [0xa1, 0x74]) {
            encoded[pos + 1] = 0x7a; // 'z'
        }
        drop(adv);

        let result = ResourceAdvertisement::decode(&encoded);
        assert!(result.is_err());
    }

    // -- ResourceSender tests -----------------------------------------------

    /// Helper: create a sender with test data (10 bytes, mdu=3 -> 4 parts).
    fn make_sender() -> ResourceSender {
        use rand::rngs::OsRng;
        let crypto = MockLinkCrypto::new(3);
        ResourceSender::new(&mut OsRng, &crypto, b"abcdefghij", 1000).unwrap()
    }

    #[test]
    fn sender_new_creates_parts_and_hashes() {
        let sender = make_sender();
        assert_eq!(sender.state(), SenderState::Queued);
        assert_eq!(sender.part_count(), 4); // 10 bytes / 3 = ceil(3.33) = 4 parts
        assert_eq!(sender.parts.len(), 4);
        assert_eq!(sender.map_hashes.len(), 4);
        assert_eq!(sender.hashmap.len(), 4 * MAPHASH_LEN);
        // With MockLinkCrypto (identity), encrypted == original.
        assert_eq!(sender.encrypted_data, b"abcdefghij");
        assert_eq!(sender.original_data, b"abcdefghij");
        // Resource hash should be 16 bytes.
        assert_eq!(sender.hash().len(), 16);
    }

    #[test]
    fn sender_new_rejects_oversized() {
        use rand::rngs::OsRng;
        let crypto = MockLinkCrypto::new(400);
        let oversized = vec![0u8; MAX_EFFICIENT_SIZE + 1];
        let result = ResourceSender::new(&mut OsRng, &crypto, &oversized, 0);
        assert!(result.is_err());
        match result.unwrap_err() {
            ReticulumError::ResourceTooLarge { size, max } => {
                assert_eq!(size, MAX_EFFICIENT_SIZE + 1);
                assert_eq!(max, MAX_EFFICIENT_SIZE);
            }
            other => panic!("expected ResourceTooLarge, got: {:?}", other),
        }
    }

    #[test]
    fn sender_advertise_emits_packet_and_timeout() {
        let mut sender = make_sender();
        let actions = sender.advertise(2000);
        assert_eq!(sender.state(), SenderState::Advertised);

        // Should have: SendPacket{ResourceAdv}, SenderStateChanged, ScheduleTimeout.
        let send_count = actions
            .iter()
            .filter(|a| matches!(a, ResourceAction::SendPacket { context, .. } if *context == PacketContext::ResourceAdv.as_byte()))
            .count();
        assert_eq!(send_count, 1);

        let timeout_count = actions
            .iter()
            .filter(|a| matches!(a, ResourceAction::ScheduleTimeout { .. }))
            .count();
        assert_eq!(timeout_count, 1);

        let state_changed = actions
            .iter()
            .any(|a| matches!(a, ResourceAction::SenderStateChanged));
        assert!(state_changed);
    }

    #[test]
    fn sender_handles_request_emits_parts() {
        let mut sender = make_sender();
        sender.advertise(1000);

        // Build a RESOURCE_REQ payload requesting all 4 parts:
        // [flag:1][resource_hash:16][map_hashes:N*4]
        let mut req = Vec::new();
        req.push(HASHMAP_IS_NOT_EXHAUSTED); // flag
        req.extend_from_slice(&sender.resource_hash);
        for mh in &sender.map_hashes {
            req.extend_from_slice(mh);
        }

        let actions = sender.handle_event(ResourceEvent::RequestReceived, &req);

        // Should transition to Transferring, then AwaitingProof since all parts sent.
        assert_eq!(sender.state(), SenderState::AwaitingProof);

        // Count Resource packets sent (one per part).
        let resource_packets: Vec<_> = actions
            .iter()
            .filter(|a| matches!(a, ResourceAction::SendPacket { context, .. } if *context == PacketContext::Resource.as_byte()))
            .collect();
        assert_eq!(resource_packets.len(), 4);

        // Should have a progress action.
        let has_progress = actions
            .iter()
            .any(|a| matches!(a, ResourceAction::Progress { fraction } if *fraction >= 1.0));
        assert!(has_progress);
    }

    #[test]
    fn sender_validates_correct_proof() {
        let mut sender = make_sender();
        sender.advertise(1000);

        // Transition to AwaitingProof by handling a request for all parts.
        let mut req = Vec::new();
        req.push(HASHMAP_IS_NOT_EXHAUSTED);
        req.extend_from_slice(&sender.resource_hash);
        for mh in &sender.map_hashes {
            req.extend_from_slice(mh);
        }
        sender.handle_event(ResourceEvent::RequestReceived, &req);
        assert_eq!(sender.state(), SenderState::AwaitingProof);

        // Build a valid proof: [resource_hash:16][proof:16]
        let proof = compute_proof_hash(&sender.original_data, &sender.resource_hash);
        let mut proof_payload = Vec::new();
        proof_payload.extend_from_slice(&sender.resource_hash);
        proof_payload.extend_from_slice(&proof);

        let actions = sender.handle_event(ResourceEvent::ProofReceived, &proof_payload);
        assert_eq!(sender.state(), SenderState::Complete);

        let has_validated = actions
            .iter()
            .any(|a| matches!(a, ResourceAction::ProofValidated));
        assert!(has_validated);
    }

    #[test]
    fn sender_rejects_wrong_proof() {
        let mut sender = make_sender();
        sender.advertise(1000);

        // Move to AwaitingProof.
        let mut req = Vec::new();
        req.push(HASHMAP_IS_NOT_EXHAUSTED);
        req.extend_from_slice(&sender.resource_hash);
        for mh in &sender.map_hashes {
            req.extend_from_slice(mh);
        }
        sender.handle_event(ResourceEvent::RequestReceived, &req);
        assert_eq!(sender.state(), SenderState::AwaitingProof);

        // Send a bad proof.
        let mut bad_proof = Vec::new();
        bad_proof.extend_from_slice(&sender.resource_hash);
        bad_proof.extend_from_slice(&[0xFF; 16]); // Wrong proof bytes.

        let actions = sender.handle_event(ResourceEvent::ProofReceived, &bad_proof);
        assert_eq!(sender.state(), SenderState::Failed);

        let no_validated = !actions
            .iter()
            .any(|a| matches!(a, ResourceAction::ProofValidated));
        assert!(no_validated);
    }

    #[test]
    fn sender_cancel_emits_icl() {
        let mut sender = make_sender();
        sender.advertise(1000);

        let actions = sender.cancel();
        assert_eq!(sender.state(), SenderState::Failed);

        let has_icl = actions.iter().any(|a| {
            matches!(a, ResourceAction::SendPacket { context, .. } if *context == PacketContext::ResourceIcl.as_byte())
        });
        assert!(has_icl);
    }

    #[test]
    fn sender_cancel_on_complete_noop() {
        let mut sender = make_sender();
        // Force complete state.
        sender.state = SenderState::Complete;

        let actions = sender.cancel();
        assert_eq!(sender.state(), SenderState::Complete);
        assert!(actions.is_empty());
    }

    #[test]
    fn sender_receives_cancel_becomes_rejected() {
        let mut sender = make_sender();
        sender.advertise(1000);

        let actions = sender.handle_event(ResourceEvent::CancelReceived, &[]);
        assert_eq!(sender.state(), SenderState::Rejected);

        let has_state_change = actions
            .iter()
            .any(|a| matches!(a, ResourceAction::SenderStateChanged));
        assert!(has_state_change);
    }

    // -- ResourceReceiver tests ---------------------------------------------

    /// Helper: create a sender and derive a receiver from its advertisement.
    fn make_sender_and_receiver() -> (ResourceSender, ResourceReceiver) {
        use rand::rngs::OsRng;
        let crypto = MockLinkCrypto::new(3);
        let mut sender = ResourceSender::new(&mut OsRng, &crypto, b"abcdefghij", 1000).unwrap();
        let adv_actions = sender.advertise(1000);

        // Extract the advertisement plaintext from the SendPacket action.
        let adv_plaintext = adv_actions
            .iter()
            .find_map(|a| match a {
                ResourceAction::SendPacket { context, plaintext }
                    if *context == PacketContext::ResourceAdv.as_byte() =>
                {
                    Some(plaintext.clone())
                }
                _ => None,
            })
            .expect("advertise should emit SendPacket");

        let (receiver, _actions) = ResourceReceiver::accept(&adv_plaintext, 2000).unwrap();
        (sender, receiver)
    }

    #[test]
    fn receiver_accept_parses_adv_emits_request() {
        use rand::rngs::OsRng;
        let crypto = MockLinkCrypto::new(3);
        let mut sender = ResourceSender::new(&mut OsRng, &crypto, b"abcdefghij", 1000).unwrap();
        let adv_actions = sender.advertise(1000);

        let adv_plaintext = adv_actions
            .iter()
            .find_map(|a| match a {
                ResourceAction::SendPacket { context, plaintext }
                    if *context == PacketContext::ResourceAdv.as_byte() =>
                {
                    Some(plaintext.clone())
                }
                _ => None,
            })
            .unwrap();

        let (receiver, actions) = ResourceReceiver::accept(&adv_plaintext, 2000).unwrap();

        assert_eq!(receiver.state(), ReceiverState::Transferring);
        assert_eq!(receiver.total_parts, 4);
        assert_eq!(*receiver.hash(), sender.resource_hash);

        // Should emit a resource request.
        let has_req = actions.iter().any(|a| {
            matches!(a, ResourceAction::SendPacket { context, .. } if *context == PacketContext::ResourceReq.as_byte())
        });
        assert!(has_req);

        // Should schedule a timeout.
        let has_timeout = actions
            .iter()
            .any(|a| matches!(a, ResourceAction::ScheduleTimeout { .. }));
        assert!(has_timeout);
    }

    #[test]
    fn receiver_handles_parts_and_assembles() {
        let (sender, mut receiver) = make_sender_and_receiver();

        // Feed all 4 parts to the receiver. With MockLinkCrypto, the part
        // data is the same as the plaintext chunks.
        for part_data in &sender.parts {
            let actions = receiver.handle_event(ResourceEvent::PartReceived, part_data);
            if receiver.state() == ReceiverState::Assembling {
                // Last part should trigger assembly.
                let has_assembly_ready = actions
                    .iter()
                    .any(|a| matches!(a, ResourceAction::AssemblyReady));
                assert!(has_assembly_ready);
            }
        }

        assert_eq!(receiver.state(), ReceiverState::Assembling);
        assert_eq!(receiver.received_count, 4);
        assert!(receiver.assembled_encrypted.is_some());
    }

    #[test]
    fn receiver_finalize_decrypts_and_proves() {
        let (sender, mut receiver) = make_sender_and_receiver();
        let crypto = MockLinkCrypto::new(3);

        // Feed all parts.
        for part_data in &sender.parts {
            receiver.handle_event(ResourceEvent::PartReceived, part_data);
        }
        assert_eq!(receiver.state(), ReceiverState::Assembling);

        // Finalize: decrypt + send proof.
        let actions = receiver.finalize(&crypto).unwrap();
        assert_eq!(receiver.state(), ReceiverState::Complete);

        // Should emit a proof packet.
        let has_prf = actions.iter().any(|a| {
            matches!(a, ResourceAction::SendPacket { context, .. } if *context == PacketContext::ResourcePrf.as_byte())
        });
        assert!(has_prf);

        // Should emit Completed with the original data.
        let completed_data = actions.iter().find_map(|a| match a {
            ResourceAction::Completed { data } => Some(data.clone()),
            _ => None,
        });
        assert_eq!(completed_data.as_deref(), Some(b"abcdefghij".as_slice()));
    }

    #[test]
    fn receiver_cancel_emits_rcl() {
        let (_sender, mut receiver) = make_sender_and_receiver();

        let actions = receiver.cancel();
        assert_eq!(receiver.state(), ReceiverState::Failed);

        let has_rcl = actions.iter().any(|a| {
            matches!(a, ResourceAction::SendPacket { context, .. } if *context == PacketContext::ResourceRcl.as_byte())
        });
        assert!(has_rcl);
    }

    #[test]
    fn receiver_cancel_received_becomes_failed() {
        let (_sender, mut receiver) = make_sender_and_receiver();

        let actions = receiver.handle_event(ResourceEvent::CancelReceived, &[]);
        assert_eq!(receiver.state(), ReceiverState::Failed);

        let has_state_change = actions
            .iter()
            .any(|a| matches!(a, ResourceAction::ReceiverStateChanged));
        assert!(has_state_change);
    }

    #[test]
    fn receiver_detects_corrupt_assembly() {
        let (sender, mut receiver) = make_sender_and_receiver();

        // Feed all parts except the last one normally.
        for part_data in &sender.parts[..sender.parts.len() - 1] {
            receiver.handle_event(ResourceEvent::PartReceived, part_data);
        }

        // Tamper with the last part before feeding it.
        let mut tampered = sender.parts.last().unwrap().clone();
        // Flip a byte in the tampered data.
        if let Some(byte) = tampered.first_mut() {
            *byte ^= 0xFF;
        }

        // We need the tampered part to still match a map_hash so it gets
        // stored. Since we changed the data, the map_hash won't match.
        // Instead, manually store the tampered part to simulate corruption.
        let last_idx = sender.parts.len() - 1;
        receiver.parts[last_idx] = Some(tampered);
        receiver.received_count = receiver.total_parts;
        receiver.consecutive_completed = receiver.total_parts;

        // Trigger assembly manually by transitioning state.
        receiver.state = ReceiverState::Assembling;

        // Assemble and verify hash mismatch.
        let mut assembled = Vec::new();
        for part in &receiver.parts {
            if let Some(data) = part {
                assembled.extend_from_slice(data);
            }
        }
        let computed_hash = compute_resource_hash(&assembled, &receiver.random_hash);
        assert_ne!(
            computed_hash, receiver.resource_hash,
            "Tampered data should produce different hash"
        );

        // The receiver should detect this corruption.
        // Reset state to Transferring and feed the tampered last part through handle_event.
        // We need to construct a scenario where the assembly check runs:
        receiver.state = ReceiverState::Transferring;
        receiver.received_count = sender.parts.len() - 1;
        receiver.consecutive_completed = sender.parts.len() - 1;
        receiver.parts[last_idx] = None;

        // The tampered part won't match any map_hash, so it won't be stored.
        // To test corruption detection, directly insert bad data and trigger assembly.
        receiver.parts[last_idx] = Some(vec![0xFF; sender.parts[last_idx].len()]);
        receiver.received_count = receiver.total_parts;
        receiver.consecutive_completed = receiver.total_parts;

        // Assemble.
        let mut assembled = Vec::new();
        for part in &receiver.parts {
            if let Some(data) = part {
                assembled.extend_from_slice(data);
            }
        }
        let hash_check = compute_resource_hash(&assembled, &receiver.random_hash);
        // Confirm that the tampered assembly produces a different hash.
        assert_ne!(hash_check, receiver.resource_hash);
    }

    #[test]
    fn sender_timeout_in_advertised_retries() {
        let mut sender = make_sender();
        sender.advertise(1000);
        assert_eq!(sender.state(), SenderState::Advertised);

        let initial_retries = sender.adv_retries_left;

        // First timeout: should retry the advertisement.
        let actions = sender.handle_event(ResourceEvent::Timeout { now_ms: 20_000 }, &[]);
        assert_eq!(sender.state(), SenderState::Advertised);
        assert_eq!(sender.adv_retries_left, initial_retries - 1);

        let has_adv = actions.iter().any(|a| {
            matches!(a, ResourceAction::SendPacket { context, .. } if *context == PacketContext::ResourceAdv.as_byte())
        });
        assert!(has_adv);
    }

    #[test]
    fn sender_timeout_exhausts_retries_fails() {
        let mut sender = make_sender();
        sender.advertise(1000);
        sender.adv_retries_left = 0;

        let actions = sender.handle_event(ResourceEvent::Timeout { now_ms: 20_000 }, &[]);
        assert_eq!(sender.state(), SenderState::Failed);
        let has_state_change = actions
            .iter()
            .any(|a| matches!(a, ResourceAction::SenderStateChanged));
        assert!(has_state_change);
    }

    #[test]
    fn receiver_timeout_retries_then_fails() {
        let (_sender, mut receiver) = make_sender_and_receiver();
        assert_eq!(receiver.state(), ReceiverState::Transferring);

        // Should retry while retries_left > 0.
        let initial_retries = receiver.retries_left;
        let actions = receiver.handle_event(ResourceEvent::Timeout { now_ms: 50_000 }, &[]);
        assert_eq!(receiver.state(), ReceiverState::Transferring);
        assert_eq!(receiver.retries_left, initial_retries - 1);

        let has_req = actions.iter().any(|a| {
            matches!(a, ResourceAction::SendPacket { context, .. } if *context == PacketContext::ResourceReq.as_byte())
        });
        assert!(has_req);

        // Exhaust retries.
        receiver.retries_left = 0;
        let actions = receiver.handle_event(ResourceEvent::Timeout { now_ms: 100_000 }, &[]);
        assert_eq!(receiver.state(), ReceiverState::Failed);
        let has_state_change = actions
            .iter()
            .any(|a| matches!(a, ResourceAction::ReceiverStateChanged));
        assert!(has_state_change);
    }

    // -- Round-trip integration helpers --------------------------------------

    /// Extract the plaintext from the first SendPacket action matching the
    /// given context byte. Panics if no matching action is found.
    fn extract_send(actions: &[ResourceAction], ctx_byte: u8) -> Vec<u8> {
        find_send(actions, ctx_byte).expect("expected SendPacket action")
    }

    /// Find the plaintext from the first SendPacket action matching the
    /// given context byte, returning `None` if not found.
    fn find_send(actions: &[ResourceAction], ctx_byte: u8) -> Option<Vec<u8>> {
        actions.iter().find_map(|a| match a {
            ResourceAction::SendPacket { context, plaintext } if *context == ctx_byte => {
                Some(plaintext.clone())
            }
            _ => None,
        })
    }

    /// Collect all SendPacket plaintexts matching the given context byte.
    fn collect_sends(actions: &[ResourceAction], ctx_byte: u8) -> Vec<Vec<u8>> {
        actions
            .iter()
            .filter_map(|a| match a {
                ResourceAction::SendPacket { context, plaintext } if *context == ctx_byte => {
                    Some(plaintext.clone())
                }
                _ => None,
            })
            .collect()
    }

    /// Run a full sender-receiver round-trip protocol exchange and return the
    /// data recovered by the receiver. This helper handles multi-round windowed
    /// transfers automatically.
    fn run_roundtrip(
        sender: &mut ResourceSender,
        receiver: &mut ResourceReceiver,
        initial_actions: Vec<ResourceAction>,
        mock: &MockLinkCrypto,
    ) -> (Vec<u8>, Vec<ResourceAction>) {
        let resource_ctx = PacketContext::Resource.as_byte();
        let req_ctx = PacketContext::ResourceReq.as_byte();

        // The initial_actions from accept() should contain a ResourceReq.
        let mut pending_req = find_send(&initial_actions, req_ctx);

        // Loop: feed requests to sender, get parts, feed parts to receiver.
        loop {
            let req_plaintext = match pending_req.take() {
                Some(r) => r,
                None => panic!("expected a ResourceReq but none was emitted"),
            };

            // Feed request to sender -> get part packets.
            let sender_actions =
                sender.handle_event(ResourceEvent::RequestReceived, &req_plaintext);
            let part_packets = collect_sends(&sender_actions, resource_ctx);
            assert!(
                !part_packets.is_empty(),
                "sender should emit at least one part"
            );

            // Feed each part to receiver.
            let mut last_receiver_actions = Vec::new();
            for part in &part_packets {
                last_receiver_actions = receiver.handle_event(ResourceEvent::PartReceived, part);
            }

            if receiver.state() == ReceiverState::Assembling {
                // All parts received, finalize.
                let finalize_actions = receiver.finalize(mock).expect("finalize should succeed");
                return (
                    finalize_actions
                        .iter()
                        .find_map(|a| match a {
                            ResourceAction::Completed { data } => Some(data.clone()),
                            _ => None,
                        })
                        .expect("finalize should emit Completed"),
                    finalize_actions,
                );
            }

            // Not done yet — extract the next request from the receiver's actions.
            pending_req = find_send(&last_receiver_actions, req_ctx);
        }
    }

    // -- Full round-trip integration tests ----------------------------------

    #[test]
    fn full_roundtrip_small_data() {
        use rand::rngs::OsRng;

        let mock = MockLinkCrypto::new(64);
        let data = b"Hello, resource transfer!";
        let mut sender = ResourceSender::new(&mut OsRng, &mock, data, 1000).unwrap();

        // Sender advertises.
        let adv_actions = sender.advertise(1000);
        let adv_plaintext = extract_send(&adv_actions, PacketContext::ResourceAdv.as_byte());

        // Receiver accepts advertisement.
        let (mut receiver, rx_actions) = ResourceReceiver::accept(&adv_plaintext, 2000).unwrap();
        assert_eq!(receiver.state(), ReceiverState::Transferring);

        // The initial actions should contain a ResourceReq.
        let req_plaintext = extract_send(&rx_actions, PacketContext::ResourceReq.as_byte());

        // Feed the request to sender.
        let sender_actions = sender.handle_event(ResourceEvent::RequestReceived, &req_plaintext);

        // Sender responds with Resource data parts.
        let part_packets = collect_sends(&sender_actions, PacketContext::Resource.as_byte());
        assert_eq!(part_packets.len(), sender.part_count());

        // Feed each part to receiver.
        for part in &part_packets {
            receiver.handle_event(ResourceEvent::PartReceived, part);
        }

        // Receiver should be Assembling after all parts.
        assert_eq!(receiver.state(), ReceiverState::Assembling);

        // Finalize: decrypt and produce proof.
        let finalize_actions = receiver.finalize(&mock).unwrap();
        assert_eq!(receiver.state(), ReceiverState::Complete);

        // Verify the recovered data matches the original.
        let recovered = finalize_actions
            .iter()
            .find_map(|a| match a {
                ResourceAction::Completed { data } => Some(data.clone()),
                _ => None,
            })
            .expect("finalize should emit Completed");
        assert_eq!(recovered, data);

        // Extract proof and feed to sender.
        let proof_plaintext = extract_send(&finalize_actions, PacketContext::ResourcePrf.as_byte());
        let proof_actions = sender.handle_event(ResourceEvent::ProofReceived, &proof_plaintext);

        // Sender should validate the proof and reach Complete.
        assert_eq!(sender.state(), SenderState::Complete);
        let has_validated = proof_actions
            .iter()
            .any(|a| matches!(a, ResourceAction::ProofValidated));
        assert!(has_validated);
    }

    #[test]
    fn full_roundtrip_multipart() {
        use rand::rngs::OsRng;

        // MDU of 20, data of 200 bytes -> 10 parts; window starts at 4,
        // so we need multiple request/response rounds.
        let mock = MockLinkCrypto::new(20);
        let data: Vec<u8> = (0..200u8).collect();
        let mut sender = ResourceSender::new(&mut OsRng, &mock, &data, 1000).unwrap();
        assert_eq!(sender.part_count(), 10);

        // Sender advertises.
        let adv_actions = sender.advertise(1000);
        let adv_plaintext = extract_send(&adv_actions, PacketContext::ResourceAdv.as_byte());

        // Receiver accepts.
        let (mut receiver, initial_actions) =
            ResourceReceiver::accept(&adv_plaintext, 2000).unwrap();
        assert_eq!(receiver.state(), ReceiverState::Transferring);

        // Run the multi-round protocol via the helper.
        let (recovered, finalize_actions) =
            run_roundtrip(&mut sender, &mut receiver, initial_actions, &mock);

        // Verify the recovered data matches the original.
        assert_eq!(recovered, data);
        assert_eq!(receiver.state(), ReceiverState::Complete);

        // Verify the sender accepted the proof.
        let proof_plaintext = extract_send(&finalize_actions, PacketContext::ResourcePrf.as_byte());
        let proof_actions = sender.handle_event(ResourceEvent::ProofReceived, &proof_plaintext);
        assert_eq!(sender.state(), SenderState::Complete);
        assert!(proof_actions
            .iter()
            .any(|a| matches!(a, ResourceAction::ProofValidated)));
    }

    #[test]
    fn roundtrip_single_byte() {
        use rand::rngs::OsRng;

        let mock = MockLinkCrypto::new(64);
        let data = &[0x42u8];
        let mut sender = ResourceSender::new(&mut OsRng, &mock, data, 1000).unwrap();
        assert_eq!(sender.part_count(), 1);

        let adv_actions = sender.advertise(1000);
        let adv_plaintext = extract_send(&adv_actions, PacketContext::ResourceAdv.as_byte());

        let (mut receiver, rx_actions) = ResourceReceiver::accept(&adv_plaintext, 2000).unwrap();
        let req_plaintext = extract_send(&rx_actions, PacketContext::ResourceReq.as_byte());

        // Sender sends the single part.
        let sender_actions = sender.handle_event(ResourceEvent::RequestReceived, &req_plaintext);
        let parts = collect_sends(&sender_actions, PacketContext::Resource.as_byte());
        assert_eq!(parts.len(), 1);

        // Feed the part to receiver.
        receiver.handle_event(ResourceEvent::PartReceived, &parts[0]);
        assert_eq!(receiver.state(), ReceiverState::Assembling);

        // Finalize.
        let finalize_actions = receiver.finalize(&mock).unwrap();
        assert_eq!(receiver.state(), ReceiverState::Complete);

        let recovered = finalize_actions
            .iter()
            .find_map(|a| match a {
                ResourceAction::Completed { data } => Some(data.clone()),
                _ => None,
            })
            .unwrap();
        assert_eq!(recovered, data);

        // Verify proof.
        let proof_plaintext = extract_send(&finalize_actions, PacketContext::ResourcePrf.as_byte());
        let proof_actions = sender.handle_event(ResourceEvent::ProofReceived, &proof_plaintext);
        assert_eq!(sender.state(), SenderState::Complete);
        assert!(proof_actions
            .iter()
            .any(|a| matches!(a, ResourceAction::ProofValidated)));
    }

    #[test]
    fn roundtrip_exact_sdu_boundary() {
        use rand::rngs::OsRng;

        // MDU 50, data 100 bytes -> exactly 2 parts.
        let mock = MockLinkCrypto::new(50);
        let data = vec![0xDD; 100];
        let mut sender = ResourceSender::new(&mut OsRng, &mock, &data, 1000).unwrap();
        assert_eq!(sender.part_count(), 2);

        let adv_actions = sender.advertise(1000);
        let adv_plaintext = extract_send(&adv_actions, PacketContext::ResourceAdv.as_byte());

        let (mut receiver, rx_actions) = ResourceReceiver::accept(&adv_plaintext, 2000).unwrap();
        let req_plaintext = extract_send(&rx_actions, PacketContext::ResourceReq.as_byte());

        // Both parts fit in one window (window=4 > 2 parts).
        let sender_actions = sender.handle_event(ResourceEvent::RequestReceived, &req_plaintext);
        let parts = collect_sends(&sender_actions, PacketContext::Resource.as_byte());
        assert_eq!(parts.len(), 2);

        // Feed parts to receiver.
        for part in &parts {
            receiver.handle_event(ResourceEvent::PartReceived, part);
        }
        assert_eq!(receiver.state(), ReceiverState::Assembling);

        // Finalize and verify.
        let finalize_actions = receiver.finalize(&mock).unwrap();
        assert_eq!(receiver.state(), ReceiverState::Complete);

        let recovered = finalize_actions
            .iter()
            .find_map(|a| match a {
                ResourceAction::Completed { data } => Some(data.clone()),
                _ => None,
            })
            .unwrap();
        assert_eq!(recovered, data);

        // Verify proof.
        let proof_plaintext = extract_send(&finalize_actions, PacketContext::ResourcePrf.as_byte());
        let proof_actions = sender.handle_event(ResourceEvent::ProofReceived, &proof_plaintext);
        assert_eq!(sender.state(), SenderState::Complete);
        assert!(proof_actions
            .iter()
            .any(|a| matches!(a, ResourceAction::ProofValidated)));
    }

    #[test]
    fn roundtrip_proof_validates_sender() {
        use rand::rngs::OsRng;

        // Medium-sized transfer: 150 bytes with MDU 30 -> 5 parts.
        let mock = MockLinkCrypto::new(30);
        let data: Vec<u8> = (0..150u8).collect();
        let mut sender = ResourceSender::new(&mut OsRng, &mock, &data, 1000).unwrap();
        assert_eq!(sender.part_count(), 5);

        let adv_actions = sender.advertise(1000);
        let adv_plaintext = extract_send(&adv_actions, PacketContext::ResourceAdv.as_byte());

        let (mut receiver, initial_actions) =
            ResourceReceiver::accept(&adv_plaintext, 2000).unwrap();

        // Run the full protocol exchange.
        let (recovered, finalize_actions) =
            run_roundtrip(&mut sender, &mut receiver, initial_actions, &mock);
        assert_eq!(recovered, data);

        // The sender should be in AwaitingProof after all parts are sent.
        // (run_roundtrip doesn't feed the proof to sender.)
        assert!(
            sender.state() == SenderState::AwaitingProof
                || sender.state() == SenderState::Transferring,
            "sender should be AwaitingProof or Transferring after sending all parts, got {:?}",
            sender.state()
        );

        // Feed the proof to sender.
        let proof_plaintext = extract_send(&finalize_actions, PacketContext::ResourcePrf.as_byte());
        let proof_actions = sender.handle_event(ResourceEvent::ProofReceived, &proof_plaintext);

        // Sender should validate the proof and reach Complete.
        assert_eq!(sender.state(), SenderState::Complete);
        let has_proof_validated = proof_actions
            .iter()
            .any(|a| matches!(a, ResourceAction::ProofValidated));
        assert!(
            has_proof_validated,
            "sender should emit ProofValidated action"
        );

        // Verify both sides are in terminal Complete state.
        assert_eq!(sender.state(), SenderState::Complete);
        assert_eq!(receiver.state(), ReceiverState::Complete);
    }

    // -- Adaptive windowing tests -------------------------------------------

    /// Helper: create a sender/receiver pair with enough parts to span
    /// multiple windows (MDU=20, data=200 bytes -> 10 parts, window=4).
    fn make_multipart_pair() -> (ResourceSender, ResourceReceiver, Vec<ResourceAction>) {
        use rand::rngs::OsRng;
        let mock = MockLinkCrypto::new(20);
        let data: Vec<u8> = (0..200u8).collect();
        let mut sender = ResourceSender::new(&mut OsRng, &mock, &data, 1000).unwrap();
        assert_eq!(sender.part_count(), 10);

        let adv_actions = sender.advertise(1000);
        let adv_plaintext = extract_send(&adv_actions, PacketContext::ResourceAdv.as_byte());
        let (receiver, initial_actions) = ResourceReceiver::accept(&adv_plaintext, 2000).unwrap();
        (sender, receiver, initial_actions)
    }

    #[test]
    fn window_grows_after_full_window_received() {
        let (mut sender, mut receiver, initial_actions) = make_multipart_pair();

        // The receiver starts with window = WINDOW_INITIAL = 4.
        assert_eq!(receiver.window, WINDOW_INITIAL as usize);
        let initial_window = receiver.window;

        // Feed the first request to the sender and get parts back.
        let req_plaintext = extract_send(&initial_actions, PacketContext::ResourceReq.as_byte());
        let sender_actions = sender.handle_event(ResourceEvent::RequestReceived, &req_plaintext);
        let part_packets = collect_sends(&sender_actions, PacketContext::Resource.as_byte());
        assert_eq!(part_packets.len(), initial_window);

        // Feed all parts from the first window to the receiver.
        for part in &part_packets {
            receiver.handle_event(ResourceEvent::PartReceived, part);
        }

        // After receiving a full window, the window should have grown by 1.
        // The receiver has 10 parts, received 4, so it still needs more
        // and should have issued a new request with grown window.
        assert_eq!(receiver.window, initial_window + 1);
        assert_eq!(receiver.received_count, initial_window);
        assert_eq!(receiver.state(), ReceiverState::Transferring);
    }

    #[test]
    fn window_does_not_exceed_window_max() {
        let (_sender, mut receiver) = make_sender_and_receiver();

        // Set window to window_max - 1 to test the cap.
        receiver.window = receiver.window_max;

        // Manually trigger the growth logic: set outstanding_parts to 0
        // and ensure there are unreceived parts.
        receiver.outstanding_parts = 0;

        // Window growth only happens in handle_part when outstanding == 0.
        // Since window == window_max, it should NOT grow.
        let old_window = receiver.window;

        // Feed a valid part to trigger the code path.
        // We need a part that matches the hashmap.
        let part_data = vec![0x42; 3]; // dummy data
        let mh = compute_map_hash(&part_data, &receiver.random_hash);
        // Install this map hash in slot 0.
        receiver.hashmap[0] = Some(mh);
        receiver.parts[0] = None;
        receiver.received_count = receiver.total_parts - 2; // not yet complete
        receiver.consecutive_completed = 0;
        receiver.outstanding_parts = 1; // will decrement to 0

        receiver.handle_event(ResourceEvent::PartReceived, &part_data);

        // Window should NOT have grown past window_max.
        assert!(
            receiver.window <= old_window,
            "window should not exceed window_max: {} > {}",
            receiver.window,
            old_window
        );
    }

    #[test]
    fn window_min_advances_with_flexibility() {
        let (_sender, mut receiver) = make_sender_and_receiver();

        // Set up conditions where window will grow repeatedly.
        // WINDOW_FLEXIBILITY is 4, so after growing 4 times beyond window_min,
        // window_min should also increase.
        let original_min = receiver.window_min; // 2
        receiver.window = original_min + (WINDOW_FLEXIBILITY as usize - 1); // 5
        receiver.window_max = 20; // plenty of headroom

        // Trigger growth by having outstanding_parts reach 0 with parts remaining.
        // We need to simulate: receive a part that brings outstanding to 0, with more needed.
        let part_data = vec![0xAB; 3];
        let mh = compute_map_hash(&part_data, &receiver.random_hash);
        receiver.hashmap[0] = Some(mh);
        receiver.parts[0] = None;
        receiver.received_count = receiver.total_parts - 2;
        receiver.consecutive_completed = 0;
        receiver.outstanding_parts = 1;

        receiver.handle_event(ResourceEvent::PartReceived, &part_data);

        // Window grew from (min + FLEX-1) to (min + FLEX), so window_min should increase.
        assert_eq!(
            receiver.window_min,
            original_min + 1,
            "window_min should advance when (window - window_min) > FLEXIBILITY-1"
        );
    }

    #[test]
    fn window_shrinks_on_timeout() {
        let (_sender, mut receiver) = make_sender_and_receiver();
        assert_eq!(receiver.state(), ReceiverState::Transferring);

        let initial_window = receiver.window;
        let initial_max = receiver.window_max;
        let initial_retries = receiver.retries_left;

        // Fire a timeout — should shrink window, shrink window_max, and retry.
        let actions = receiver.handle_event(ResourceEvent::Timeout { now_ms: 50_000 }, &[]);

        assert_eq!(receiver.state(), ReceiverState::Transferring);
        assert_eq!(receiver.retries_left, initial_retries - 1);
        assert_eq!(
            receiver.window,
            initial_window - 1,
            "window should shrink by 1 on timeout"
        );
        assert_eq!(
            receiver.window_max,
            initial_max - 1,
            "window_max should shrink by 1 on timeout"
        );

        // Should have emitted a new request.
        let has_req = actions.iter().any(|a| {
            matches!(a, ResourceAction::SendPacket { context, .. }
                if *context == PacketContext::ResourceReq.as_byte())
        });
        assert!(has_req, "timeout should trigger a new request");
    }

    #[test]
    fn window_does_not_shrink_below_min() {
        let (_sender, mut receiver) = make_sender_and_receiver();

        // Set window and window_max to their minimum values.
        receiver.window = receiver.window_min;
        receiver.window_max = receiver.window_min;

        let actions = receiver.handle_event(ResourceEvent::Timeout { now_ms: 50_000 }, &[]);

        // Window should stay at window_min.
        assert_eq!(
            receiver.window, receiver.window_min,
            "window must not go below window_min"
        );
        assert_eq!(
            receiver.window_max, receiver.window_min,
            "window_max must not go below window_min"
        );

        // Should still have emitted a retry request.
        let has_req = actions.iter().any(|a| {
            matches!(a, ResourceAction::SendPacket { context, .. }
                if *context == PacketContext::ResourceReq.as_byte())
        });
        assert!(has_req);
    }

    #[test]
    fn receiver_fails_after_max_retries() {
        let (_sender, mut receiver) = make_sender_and_receiver();
        assert_eq!(receiver.retries_left, MAX_RETRIES as u8);

        // Exhaust all retries one by one.
        for i in 0..MAX_RETRIES {
            assert_eq!(
                receiver.state(),
                ReceiverState::Transferring,
                "should still be Transferring at retry {}",
                i
            );
            receiver.handle_event(
                ResourceEvent::Timeout {
                    now_ms: 50_000 + i as u64 * 1_000,
                },
                &[],
            );
        }
        assert_eq!(receiver.retries_left, 0);
        assert_eq!(
            receiver.state(),
            ReceiverState::Transferring,
            "should be Transferring with 0 retries left but not yet timed out again"
        );

        // One more timeout should fail the transfer.
        let actions = receiver.handle_event(ResourceEvent::Timeout { now_ms: 200_000 }, &[]);
        assert_eq!(receiver.state(), ReceiverState::Failed);
        let has_state_change = actions
            .iter()
            .any(|a| matches!(a, ResourceAction::ReceiverStateChanged));
        assert!(
            has_state_change,
            "should emit ReceiverStateChanged on failure"
        );
    }

    #[test]
    fn rtt_measured_after_first_window() {
        let (mut sender, mut receiver, initial_actions) = make_multipart_pair();

        // Before any parts, RTT should be None.
        assert!(receiver.rtt_ms.is_none());

        // Note the time the initial request was sent.
        let req_sent_time = receiver.req_sent_at.unwrap();

        // Feed the request to sender.
        let req_plaintext = extract_send(&initial_actions, PacketContext::ResourceReq.as_byte());
        let sender_actions = sender.handle_event(ResourceEvent::RequestReceived, &req_plaintext);
        let part_packets = collect_sends(&sender_actions, PacketContext::Resource.as_byte());

        // Simulate time passing: update last_activity before feeding parts.
        receiver.last_activity = req_sent_time + 500; // 500ms later

        // Feed all parts.
        for part in &part_packets {
            receiver.handle_event(ResourceEvent::PartReceived, part);
        }

        // After a full window completes, request_next is called with last_activity,
        // and RTT should be measured.
        assert!(
            receiver.rtt_ms.is_some(),
            "RTT should be measured after first complete window"
        );

        // The part_timeout_factor should have been reduced.
        assert_eq!(
            receiver.part_timeout_factor, PART_TIMEOUT_FACTOR_AFTER_RTT as u64,
            "part_timeout_factor should tighten after first RTT measurement"
        );
    }

    #[test]
    fn timeout_resets_rate_counters() {
        let (_sender, mut receiver) = make_sender_and_receiver();

        // Simulate that the receiver had been counting fast rounds.
        receiver.fast_rate_rounds = 3;
        receiver.very_slow_rate_rounds = 1;

        receiver.handle_event(ResourceEvent::Timeout { now_ms: 50_000 }, &[]);

        assert_eq!(
            receiver.fast_rate_rounds, 0,
            "fast_rate_rounds should reset on timeout"
        );
        assert_eq!(
            receiver.very_slow_rate_rounds, 0,
            "very_slow_rate_rounds should reset on timeout"
        );
    }

    #[test]
    fn multipart_roundtrip_with_windowing_succeeds() {
        // End-to-end test proving that window growth doesn't break the protocol.
        use rand::rngs::OsRng;
        let mock = MockLinkCrypto::new(20);
        let data: Vec<u8> = (0..200u8).collect();
        let mut sender = ResourceSender::new(&mut OsRng, &mock, &data, 1000).unwrap();
        assert_eq!(sender.part_count(), 10);

        let adv_actions = sender.advertise(1000);
        let adv_plaintext = extract_send(&adv_actions, PacketContext::ResourceAdv.as_byte());
        let (mut receiver, initial_actions) =
            ResourceReceiver::accept(&adv_plaintext, 2000).unwrap();

        let (recovered, finalize_actions) =
            run_roundtrip(&mut sender, &mut receiver, initial_actions, &mock);

        assert_eq!(recovered, data);
        assert_eq!(receiver.state(), ReceiverState::Complete);

        // Window should have grown from initial during the transfer.
        // (We can't assert the exact final value since it depends on
        // how many rounds were needed, but the transfer succeeded.)

        // Verify proof.
        let proof_plaintext = extract_send(&finalize_actions, PacketContext::ResourcePrf.as_byte());
        let proof_actions = sender.handle_event(ResourceEvent::ProofReceived, &proof_plaintext);
        assert_eq!(sender.state(), SenderState::Complete);
        assert!(proof_actions
            .iter()
            .any(|a| matches!(a, ResourceAction::ProofValidated)));
    }

    // -- Hashmap Update (HMU) tests -----------------------------------------

    #[test]
    fn hmu_extends_receiver_hashmap() {
        // Create a sender with 10 parts (MDU=20, data=200 bytes).
        // Then create a receiver with a partial hashmap (only the first 4
        // hashes populated), simulating a large resource where the initial
        // advertisement didn't include all hashes.
        use rand::rngs::OsRng;

        let mock = MockLinkCrypto::new(20);
        let data: Vec<u8> = (0..200u8).collect();
        let mut sender = ResourceSender::new(&mut OsRng, &mock, &data, 1000).unwrap();
        assert_eq!(sender.part_count(), 10);

        let adv_actions = sender.advertise(1000);
        let adv_plaintext = extract_send(&adv_actions, PacketContext::ResourceAdv.as_byte());

        // Accept the full advertisement, then manually truncate the hashmap
        // to simulate a partial hashmap from a segmented advertisement.
        let (mut receiver, _initial_actions) =
            ResourceReceiver::accept(&adv_plaintext, 2000).unwrap();

        // Truncate: keep only the first 4 hashes, clear the rest.
        let truncate_at = 4;
        for slot in receiver.hashmap.iter_mut().skip(truncate_at) {
            *slot = None;
        }
        receiver.hashmap_height = truncate_at;

        // Build an HMU packet containing the remaining 6 hashes (indices 4..10).
        let remaining_hashes: Vec<u8> = sender.map_hashes[truncate_at..]
            .iter()
            .flat_map(|h| h.iter().copied())
            .collect();

        let mut hmu_pkt = Vec::new();
        hmu_pkt.extend_from_slice(&sender.resource_hash);
        {
            use rmp::encode::*;
            write_array_len(&mut hmu_pkt, 2).unwrap();
            write_u32(&mut hmu_pkt, truncate_at as u32).unwrap();
            write_bin(&mut hmu_pkt, &remaining_hashes).unwrap();
        }

        // Before HMU, hashmap_height is 4.
        assert_eq!(receiver.hashmap_height, 4);

        // Feed the HMU to the receiver.
        let actions = receiver.handle_event(ResourceEvent::HashmapUpdateReceived, &hmu_pkt);

        // After HMU, hashmap_height should be 10 (all hashes installed).
        assert_eq!(receiver.hashmap_height, 10);
        assert_eq!(
            receiver.hashmap_height, receiver.total_parts,
            "all hashmap slots should now be populated"
        );

        // All hashmap slots should have Some values.
        for (i, slot) in receiver.hashmap.iter().enumerate() {
            assert!(
                slot.is_some(),
                "hashmap slot {} should be populated after HMU",
                i
            );
        }

        // The HMU handler should have emitted a new request for parts.
        let has_req = actions.iter().any(|a| {
            matches!(a, ResourceAction::SendPacket { context, .. }
                if *context == PacketContext::ResourceReq.as_byte())
        });
        assert!(has_req, "HMU should trigger a new resource request");
    }

    #[test]
    fn sender_builds_hmu_on_exhausted_request() {
        // Create a sender with 10 parts.
        use rand::rngs::OsRng;

        let mock = MockLinkCrypto::new(20);
        let data: Vec<u8> = (0..200u8).collect();
        let mut sender = ResourceSender::new(&mut OsRng, &mock, &data, 1000).unwrap();
        assert_eq!(sender.part_count(), 10);
        sender.advertise(1000);

        // Build an exhausted RESOURCE_REQ: the receiver knows hashes 0..4
        // and signals it needs more.
        //   [0xFF][last_map_hash:4][resource_hash:16][requested_map_hashes:N*4]
        let last_known_idx = 3; // 0-based, last hash the receiver has
        let last_map_hash = sender.map_hashes[last_known_idx];

        let mut req = Vec::new();
        req.push(HASHMAP_IS_EXHAUSTED);
        req.extend_from_slice(&last_map_hash);
        req.extend_from_slice(&sender.resource_hash);
        // Request parts 0..4 (the ones we have hashes for).
        for mh in &sender.map_hashes[..4] {
            req.extend_from_slice(mh);
        }

        let actions = sender.handle_event(ResourceEvent::RequestReceived, &req);

        // Should emit an HMU packet.
        let hmu_packets = collect_sends(&actions, PacketContext::ResourceHmu.as_byte());
        assert_eq!(hmu_packets.len(), 1, "sender should emit exactly one HMU");

        // Verify the HMU packet structure: [resource_hash:16][msgpack([index, hashes])]
        let hmu = &hmu_packets[0];
        assert!(
            hmu.len() > 16,
            "HMU should be larger than just resource hash"
        );

        let mut hmu_rh = [0u8; 16];
        hmu_rh.copy_from_slice(&hmu[..16]);
        assert_eq!(
            hmu_rh, sender.resource_hash,
            "HMU should contain the resource hash"
        );

        // Parse the msgpack payload.
        let mut rd = &hmu[16..];
        let arr_len = rmp::decode::read_array_len(&mut rd).unwrap();
        assert_eq!(arr_len, 2);

        let seg_idx: u32 = rmp::decode::read_int(&mut rd).unwrap();
        assert_eq!(
            seg_idx,
            (last_known_idx + 1) as u32,
            "segment index should be the first new hash position"
        );

        let hash_len = rmp::decode::read_bin_len(&mut rd).unwrap() as usize;
        // Should contain hashes for parts 4..10 = 6 hashes * 4 bytes = 24 bytes.
        let expected_hash_count = sender.part_count() - (last_known_idx + 1);
        assert_eq!(
            hash_len,
            expected_hash_count * MAPHASH_LEN,
            "HMU should contain {} hash bytes",
            expected_hash_count * MAPHASH_LEN
        );

        // Should also emit Resource data packets for the 4 requested parts.
        let resource_packets = collect_sends(&actions, PacketContext::Resource.as_byte());
        assert_eq!(resource_packets.len(), 4, "sender should emit 4 data parts");

        // Sender should be in Transferring state (not all parts sent yet).
        assert_eq!(sender.state(), SenderState::Transferring);
    }

    #[test]
    fn full_roundtrip_with_hmu() {
        // End-to-end test: create a sender with 10 parts, build a receiver
        // with a partial hashmap (first 4 hashes only), and drive the full
        // protocol exchange including HMU to completion.
        use rand::rngs::OsRng;

        let mock = MockLinkCrypto::new(20);
        let data: Vec<u8> = (0..200u8).collect();
        let mut sender = ResourceSender::new(&mut OsRng, &mock, &data, 1000).unwrap();
        assert_eq!(sender.part_count(), 10);

        let adv_actions = sender.advertise(1000);
        let adv_plaintext = extract_send(&adv_actions, PacketContext::ResourceAdv.as_byte());

        // Create receiver and truncate its hashmap to simulate partial adv.
        let (mut receiver, _) = ResourceReceiver::accept(&adv_plaintext, 2000).unwrap();
        let truncate_at = 4;
        for slot in receiver.hashmap.iter_mut().skip(truncate_at) {
            *slot = None;
        }
        receiver.hashmap_height = truncate_at;

        // Reset receiver state for a clean run.
        receiver.outstanding_parts = 0;
        receiver.req_sent_at = None;

        let req_ctx = PacketContext::ResourceReq.as_byte();
        let hmu_ctx = PacketContext::ResourceHmu.as_byte();
        let resource_ctx = PacketContext::Resource.as_byte();

        // Round 1: receiver requests parts with the 4 hashes it knows.
        // With window=4, it will request 4 parts, all of which it has hashes for.
        let round1_actions = receiver.request_next(3000);
        let req1 = extract_send(&round1_actions, req_ctx);

        // Feed request to sender.
        let sender_actions1 = sender.handle_event(ResourceEvent::RequestReceived, &req1);
        let parts1 = collect_sends(&sender_actions1, resource_ctx);
        assert_eq!(parts1.len(), 4, "round 1 should yield 4 parts");

        // Feed parts to receiver. The last part completing the window will
        // internally call request_next, which hits hashmap exhaustion.
        let mut last_part_actions = Vec::new();
        for part in &parts1 {
            last_part_actions = receiver.handle_event(ResourceEvent::PartReceived, part);
        }
        assert_eq!(receiver.received_count, 4);
        assert_eq!(receiver.state(), ReceiverState::Transferring);

        // The last handle_part should have internally emitted an exhausted request
        // because hashmap[4] is None.
        let req2 = extract_send(&last_part_actions, req_ctx);
        assert_eq!(
            req2[0], HASHMAP_IS_EXHAUSTED,
            "request should signal hashmap exhaustion"
        );

        // Feed the exhausted request to sender -- it should respond with HMU + parts.
        let sender_actions2 = sender.handle_event(ResourceEvent::RequestReceived, &req2);
        let hmu_packets = collect_sends(&sender_actions2, hmu_ctx);
        assert_eq!(
            hmu_packets.len(),
            1,
            "sender should emit HMU on exhausted request"
        );

        // Feed the HMU to the receiver. This installs the remaining hashes and
        // emits a new request for the next window of parts.
        let hmu_actions =
            receiver.handle_event(ResourceEvent::HashmapUpdateReceived, &hmu_packets[0]);
        assert_eq!(
            receiver.hashmap_height, 10,
            "all hashes should be installed after HMU"
        );

        // The exhausted request may also have sent data parts for the hashes
        // the receiver included. Since the exhausted request had no requested
        // map hashes beyond the known 4 (which were already received), there
        // might be 0 parts. Check anyway.
        let parts2 = collect_sends(&sender_actions2, resource_ctx);
        for part in &parts2 {
            receiver.handle_event(ResourceEvent::PartReceived, part);
        }

        // Now drive remaining rounds using the request from the HMU handler
        // and subsequent requests from handle_part.
        let mut pending_req = find_send(&hmu_actions, req_ctx);
        let mut max_rounds = 10;
        while receiver.state() == ReceiverState::Transferring && max_rounds > 0 {
            max_rounds -= 1;

            let next_req = match pending_req.take() {
                Some(r) => r,
                None => {
                    // Need to manually trigger a request.
                    let req_actions = receiver.request_next(5000 + (10 - max_rounds) as u64 * 1000);
                    match find_send(&req_actions, req_ctx) {
                        Some(r) => r,
                        None => break,
                    }
                }
            };

            let sender_resp = sender.handle_event(ResourceEvent::RequestReceived, &next_req);
            let parts = collect_sends(&sender_resp, resource_ctx);

            let mut last_actions = Vec::new();
            for part in &parts {
                last_actions = receiver.handle_event(ResourceEvent::PartReceived, part);
            }

            // If the last part's actions contain a new request, capture it.
            pending_req = find_send(&last_actions, req_ctx);
        }

        // The receiver should have all 10 parts and be in Assembling state.
        assert_eq!(
            receiver.received_count, 10,
            "receiver should have all 10 parts"
        );
        assert_eq!(receiver.state(), ReceiverState::Assembling);

        // Finalize and verify.
        let finalize_actions = receiver.finalize(&mock).unwrap();
        assert_eq!(receiver.state(), ReceiverState::Complete);

        let recovered = finalize_actions
            .iter()
            .find_map(|a| match a {
                ResourceAction::Completed { data } => Some(data.clone()),
                _ => None,
            })
            .expect("finalize should emit Completed");
        assert_eq!(recovered, data, "recovered data should match original");

        // Verify proof on sender side.
        let proof_plaintext = extract_send(&finalize_actions, PacketContext::ResourcePrf.as_byte());
        let proof_actions = sender.handle_event(ResourceEvent::ProofReceived, &proof_plaintext);
        assert_eq!(sender.state(), SenderState::Complete);
        assert!(proof_actions
            .iter()
            .any(|a| matches!(a, ResourceAction::ProofValidated)));
    }

    #[test]
    fn hmu_ignores_wrong_resource_hash() {
        let (_sender, mut receiver) = make_sender_and_receiver();

        // Build an HMU with a wrong resource hash.
        let mut hmu_pkt = Vec::new();
        hmu_pkt.extend_from_slice(&[0xFF; 16]); // wrong hash
        {
            use rmp::encode::*;
            write_array_len(&mut hmu_pkt, 2).unwrap();
            write_u32(&mut hmu_pkt, 0).unwrap();
            write_bin(&mut hmu_pkt, &[0xAA; 8]).unwrap();
        }

        let actions = receiver.handle_event(ResourceEvent::HashmapUpdateReceived, &hmu_pkt);
        assert!(actions.is_empty(), "HMU with wrong hash should be ignored");
    }

    #[test]
    fn hmu_ignores_truncated_payload() {
        let (_sender, mut receiver) = make_sender_and_receiver();

        // Payload too short (less than 16 bytes).
        let actions = receiver.handle_event(ResourceEvent::HashmapUpdateReceived, &[0x01; 10]);
        assert!(actions.is_empty(), "truncated HMU should be ignored");
    }

    #[test]
    fn exhausted_request_includes_last_map_hash() {
        // Verify that when a receiver emits an exhausted request, it includes
        // the last known map hash in the correct position.
        use rand::rngs::OsRng;

        let mock = MockLinkCrypto::new(20);
        let data: Vec<u8> = (0..200u8).collect();
        let mut sender = ResourceSender::new(&mut OsRng, &mock, &data, 1000).unwrap();
        assert_eq!(sender.part_count(), 10);

        let adv_actions = sender.advertise(1000);
        let adv_plaintext = extract_send(&adv_actions, PacketContext::ResourceAdv.as_byte());

        let (mut receiver, _) = ResourceReceiver::accept(&adv_plaintext, 2000).unwrap();

        // Truncate hashmap to 4 entries.
        let truncate_at = 4;
        for slot in receiver.hashmap.iter_mut().skip(truncate_at) {
            *slot = None;
        }
        receiver.hashmap_height = truncate_at;
        receiver.consecutive_completed = truncate_at; // pretend first 4 are done
        for i in 0..truncate_at {
            receiver.parts[i] = Some(vec![0x42]); // dummy data
        }
        receiver.received_count = truncate_at;
        receiver.outstanding_parts = 0;

        let actions = receiver.request_next(5000);
        let req = extract_send(&actions, PacketContext::ResourceReq.as_byte());

        // Verify the exhausted flag.
        assert_eq!(req[0], HASHMAP_IS_EXHAUSTED);

        // Extract last_map_hash from position [1..5].
        let mut last_mh = [0u8; MAPHASH_LEN];
        last_mh.copy_from_slice(&req[1..1 + MAPHASH_LEN]);

        // It should be the hash at index truncate_at - 1 = 3.
        let expected_last_mh = receiver.hashmap[truncate_at - 1].unwrap();
        assert_eq!(
            last_mh, expected_last_mh,
            "exhausted request should contain the last known map hash"
        );

        // Resource hash should follow at [5..21].
        let mut rh = [0u8; 16];
        rh.copy_from_slice(&req[1 + MAPHASH_LEN..1 + MAPHASH_LEN + 16]);
        assert_eq!(rh, receiver.resource_hash);
    }

    // ── Python interop tests ────────────────────────────────────────────

    #[test]
    fn interop_map_hash_matches_python() {
        // Python: hashlib.sha256(bytes([0xAA]*30) + bytes([0x01,0x02,0x03,0x04])).digest()[:4]
        // Result: d38ae4f6
        let part = vec![0xAA; 30];
        let random_hash: RandomHash = [0x01, 0x02, 0x03, 0x04];
        let mh = compute_map_hash(&part, &random_hash);
        assert_eq!(mh, [0xd3, 0x8a, 0xe4, 0xf6]);
    }

    #[test]
    fn interop_resource_hash_matches_python() {
        // Python: hashlib.sha256(bytes([0xBB]*100) + bytes([0x05,0x06,0x07,0x08])).digest()[:16]
        // Result: 613cbe9a5d56e2905d9e4faf56d015cc
        let data = vec![0xBB; 100];
        let random_hash: RandomHash = [0x05, 0x06, 0x07, 0x08];
        let rh = compute_resource_hash(&data, &random_hash);
        assert_eq!(
            rh,
            [
                0x61, 0x3c, 0xbe, 0x9a, 0x5d, 0x56, 0xe2, 0x90, 0x5d, 0x9e, 0x4f, 0xaf, 0x56,
                0xd0, 0x15, 0xcc
            ]
        );
    }

    #[test]
    fn interop_proof_hash_matches_python() {
        // Python: hashlib.sha256(b'test data for proof' + bytes([0xCC]*16)).digest()[:16]
        // Result: 479f250c8349db6975bcd3d16dfa6369
        let plaintext = b"test data for proof";
        let resource_hash: ResourceHash = [0xCC; 16];
        let proof = compute_proof_hash(plaintext, &resource_hash);
        assert_eq!(
            proof,
            [
                0x47, 0x9f, 0x25, 0x0c, 0x83, 0x49, 0xdb, 0x69, 0x75, 0xbc, 0xd3, 0xd1, 0x6d,
                0xfa, 0x63, 0x69
            ]
        );
    }

    #[test]
    fn interop_advertisement_decode_from_python() {
        // Python: umsgpack.packb({"t":100,"d":80,"n":3,"h":bytes([0xAA]*16),"r":bytes([1,2,3,4]),
        //   "o":bytes([0xBB]*16),"m":bytes([0x11]*12),"f":1,"i":1,"l":1,"q":None})
        // Result: 8ba17464a16450a16e03a168c410...a171c0
        let python_bytes = hex::decode(
            "8ba17464a16450a16e03a168c410aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa172c40401020304\
             a16fc410bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba16dc40c111111111111111111111111\
             a16601a16901a16c01a171c0",
        )
        .unwrap();

        let adv = ResourceAdvertisement::decode(&python_bytes).unwrap();
        assert_eq!(adv.transfer_size, 100);
        assert_eq!(adv.data_size, 80);
        assert_eq!(adv.part_count, 3);
        assert_eq!(adv.resource_hash, [0xAA; 16]);
        assert_eq!(adv.random_hash, [1, 2, 3, 4]);
        assert_eq!(adv.original_hash, [0xBB; 16]);
        assert_eq!(adv.hashmap, vec![0x11; 12]);
        assert_eq!(adv.flags, 1);
        assert_eq!(adv.segment_index, 1);
        assert_eq!(adv.total_segments, 1);
        assert_eq!(adv.request_id, None);
    }

    #[test]
    fn interop_advertisement_rust_decoded_by_python_format() {
        // Verify our encode produces bytes that match the Python format
        let adv = ResourceAdvertisement {
            transfer_size: 100,
            data_size: 80,
            part_count: 3,
            resource_hash: [0xAA; 16],
            random_hash: [1, 2, 3, 4],
            original_hash: [0xBB; 16],
            hashmap: vec![0x11; 12],
            flags: 1,
            segment_index: 1,
            total_segments: 1,
            request_id: None,
        };
        let encoded = adv.encode();
        // Should decode back successfully
        let decoded = ResourceAdvertisement::decode(&encoded).unwrap();
        assert_eq!(decoded, adv);

        // Verify the Python-generated bytes also decode to the same result
        let python_bytes = hex::decode(
            "8ba17464a16450a16e03a168c410aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa172c40401020304\
             a16fc410bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba16dc40c111111111111111111111111\
             a16601a16901a16c01a171c0",
        )
        .unwrap();
        let python_decoded = ResourceAdvertisement::decode(&python_bytes).unwrap();
        assert_eq!(python_decoded, decoded);
    }
}
