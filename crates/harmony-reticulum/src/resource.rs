//! Resource transfer protocol for Reticulum links.
//!
//! Implements the sans-I/O state machines for sending and receiving
//! arbitrary-sized data over Reticulum links using chunked, windowed
//! transfers with acknowledgement-based flow control.

use crate::error::ReticulumError;

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
}
