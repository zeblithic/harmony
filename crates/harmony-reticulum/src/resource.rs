//! Resource transfer protocol for Reticulum links.
//!
//! Implements the sans-I/O state machines for sending and receiving
//! arbitrary-sized data over Reticulum links using chunked, windowed
//! transfers with acknowledgement-based flow control.

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
    fn encrypt(&self, plaintext: &[u8]) -> Result<Vec<u8>, ReticulumError>;

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
// Struct skeletons
// ---------------------------------------------------------------------------

/// Sender-side resource transfer state machine.
///
/// Drives advertisement, chunking, windowed transmission, and proof
/// validation. Created by the link when initiating a transfer.
pub struct ResourceSender {
    state: SenderState,
}

impl ResourceSender {
    /// Current sender state.
    pub fn state(&self) -> SenderState {
        self.state
    }
}

/// Receiver-side resource transfer state machine.
///
/// Accepts incoming parts, tracks the hashmap, reassembles the payload,
/// and generates the proof. Created by the link upon receiving an
/// advertisement.
pub struct ResourceReceiver {
    state: ReceiverState,
}

impl ResourceReceiver {
    /// Current receiver state.
    pub fn state(&self) -> ReceiverState {
        self.state
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
    /// Flags byte (reserved for future use).
    pub flags: u8,
    /// Index of this hashmap segment (0-based).
    pub segment_index: u32,
    /// Total number of hashmap segments.
    pub total_segments: u32,
}

impl ResourceAdvertisement {
    /// Number of fields in the encoded msgpack map.
    const MAP_LEN: u32 = 10;

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

        buf
    }

    /// Decode a ResourceAdvertisement from a MessagePack byte slice.
    ///
    /// Returns `Err(ResourceAdvInvalid)` if the data is malformed or
    /// contains unknown keys.
    pub fn decode(data: &[u8]) -> Result<Self, ReticulumError> {
        let mut rd = data;

        let map_len = rmp::decode::read_map_len(&mut rd)
            .map_err(|_| ReticulumError::ResourceAdvInvalid)?;

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
                        .map_err(|_| ReticulumError::ResourceAdvInvalid)? as usize;
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
        fn encrypt(&self, plaintext: &[u8]) -> Result<Vec<u8>, ReticulumError> {
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
        let mock = MockLinkCrypto::new(400);
        let plaintext = b"hello resource transfer";
        let encrypted = mock.encrypt(plaintext).unwrap();
        let decrypted = mock.decrypt(&encrypted).unwrap();
        assert_eq!(plaintext.as_slice(), decrypted.as_slice());
        assert_eq!(mock.mdu(), 400);
        assert_eq!(mock.link_id(), &[0xAA; 16]);
    }

    #[test]
    fn sender_state_initial() {
        let sender = ResourceSender {
            state: SenderState::Queued,
        };
        assert_eq!(sender.state(), SenderState::Queued);
    }

    #[test]
    fn receiver_state_initial() {
        let receiver = ResourceReceiver {
            state: ReceiverState::Transferring,
        };
        assert_eq!(receiver.state(), ReceiverState::Transferring);
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
            resource_hash: [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
                            0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10],
            random_hash: [0xAA, 0xBB, 0xCC, 0xDD],
            original_hash: [0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18,
                            0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20],
            hashmap: vec![0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE],
            flags: 0x00,
            segment_index: 0,
            total_segments: 1,
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
}
