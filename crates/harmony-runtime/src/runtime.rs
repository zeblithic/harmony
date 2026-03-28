//! Node runtime: priority event loop wiring Tier 1 (Router) + Tier 2 (Storage) + Tier 3 (Compute).

// Remaining dead code (RuntimeEvent, push_event, tick, metrics, etc.) is consumed
// only by tests until the async event loop is wired.
#![allow(dead_code)]

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use harmony_compute::InstructionBudget;
use harmony_contacts::ContactStore;
use harmony_content::bloom::BloomFilter;
use harmony_content::book::BookStore;
use harmony_content::cid::ContentId;
use harmony_content::cuckoo::CuckooFilter;
use harmony_content::replica::{MemoryReplicaStore, ReplicaStore};
use harmony_content::storage_tier::{
    ContentPolicy, FilterBroadcastConfig, StorageBudget, StorageMetrics, StorageTier,
    StorageTierAction, StorageTierEvent,
};
use harmony_discovery::DiscoveryManager;
use harmony_memo::store::MemoStore;
use harmony_peers::{PeerAction, PeerEvent, PeerManager};
use harmony_reticulum::node::{Node, NodeAction, NodeEvent};
use harmony_workflow::{ComputeHint, WorkflowAction, WorkflowEngine, WorkflowEvent, WorkflowId};
use harmony_zenoh::namespace::content as content_ns;
use harmony_zenoh::namespace::{announce as announce_ns, page as page_ns};
use harmony_zenoh::queryable::{QueryableAction, QueryableEvent, QueryableId, QueryableRouter};

/// Ticks before an in-flight memo fetch expires (20 × 250ms = 5s).
const MEMO_FETCH_TIMEOUT_TICKS: u64 = 20;

/// Ticks before a DSD session with no VerifyResponse is evicted (40 × 250ms = ~10s).
#[cfg(feature = "inference")]
const DSD_SESSION_TIMEOUT_TICKS: u64 = 40;

/// Maximum number of memos in a single query response.
const MAX_MEMO_RESPONSE_COUNT: usize = 256;

/// Maximum total response payload size (1 MiB).
const MAX_MEMO_RESPONSE_BYTES: usize = 1_048_576;

/// Configuration for a Harmony node runtime.
#[derive(Debug, Clone)]
pub struct NodeConfig {
    /// Storage tier capacity limits.
    pub storage_budget: StorageBudget,
    /// Compute tier per-slice instruction budget.
    pub compute_budget: InstructionBudget,
    /// Per-tick scheduling strategy for the three-tier event loop.
    pub schedule: TierSchedule,
    /// Content acceptance / announcement policy.
    pub content_policy: ContentPolicy,
    /// Configuration for periodic Bloom filter broadcasts.
    pub filter_broadcast_config: FilterBroadcastConfig,
    /// This node's address (hex-encoded), used for filter broadcast keys.
    /// Must be unique per node — set from identity address_hash at startup.
    /// Defaults to `"local"` as a placeholder until identity is wired.
    pub node_addr: String,
    /// This node's identity hash (16-byte address), passed to PeerManager so
    /// probe-interval jitter is unique per local-peer pair rather than zero.
    /// Defaults to all-zeros; must be set from the loaded identity at startup.
    pub local_identity_hash: harmony_identity::IdentityHash,
    /// This node's PQ identity hash (16-byte address derived from ML-KEM + ML-DSA keys).
    /// Used as the issuer identity for Discovery UCAN tokens. Distinct from
    /// `local_identity_hash` (Ed25519-derived, used for Reticulum/PeerManager).
    /// Defaults to all-zeros; must be set from the PQ identity at startup.
    pub local_pq_identity_hash: harmony_identity::IdentityHash,
    /// This node's ML-DSA-65 public verifying key bytes.
    /// Used to verify Discovery UCAN tokens (which are issued by this node).
    /// Defaults to empty; must be set from the loaded identity at startup.
    pub local_dsa_pubkey: Vec<u8>,
    /// Hex-decoded 32-byte CID of the GGUF model file in CAS (for inference).
    pub inference_gguf_cid: Option<[u8; 32]>,
    /// Hex-decoded 32-byte CID of the tokenizer.json file in CAS (for inference).
    pub inference_tokenizer_cid: Option<[u8; 32]>,
    /// Whether disk persistence is enabled (true when data_dir is configured).
    /// Controls whether StorageTier emits PersistToDisk and DiskLookup actions.
    pub disk_enabled: bool,
    /// CID+size entries discovered by scanning the data directory at startup.
    pub disk_entries: Vec<(ContentId, u64)>,
    /// Maximum bytes allowed on disk. `None` means no quota enforced.
    pub disk_quota: Option<u64>,
    /// Whether S3 fallback is enabled for durable content.
    /// Controls whether StorageTier emits S3Lookup actions on cache miss.
    pub s3_enabled: bool,
}

/// Per-tick scheduling strategy for the three-tier event loop.
#[derive(Debug, Clone)]
pub struct TierSchedule {
    /// Max router events to process per tick. `None` = drain all.
    pub router_max_per_tick: Option<usize>,
    /// Max storage events to process per tick. `None` = drain all.
    pub storage_max_per_tick: Option<usize>,
    /// Adaptive compute fuel scaling under data-plane load.
    pub adaptive_compute: AdaptiveCompute,
    /// Ticks without processing before a tier is promoted in tick order.
    /// A value of `0` means all tiers are always considered starved and no
    /// reordering occurs (equivalent to disabling starvation-based promotion).
    pub starvation_threshold: u32,
}

/// Controls how compute fuel scales with data-plane queue depth.
#[derive(Debug, Clone)]
pub struct AdaptiveCompute {
    /// Combined router+storage queue depth at which fuel starts shrinking.
    pub high_water: usize,
    /// Minimum fuel as fraction of base budget (0.0..=1.0).
    pub floor_fraction: f64,
}

impl Default for TierSchedule {
    fn default() -> Self {
        Self {
            router_max_per_tick: None,
            storage_max_per_tick: None,
            adaptive_compute: AdaptiveCompute::default(),
            starvation_threshold: 10,
        }
    }
}

impl Default for AdaptiveCompute {
    fn default() -> Self {
        Self {
            high_water: 50,
            floor_fraction: 0.1,
        }
    }
}

impl Default for NodeConfig {
    fn default() -> Self {
        Self {
            storage_budget: StorageBudget {
                cache_capacity: 1024,
                max_pinned_bytes: 100_000_000,
            },
            compute_budget: InstructionBudget { fuel: 100_000 },
            schedule: TierSchedule::default(),
            content_policy: ContentPolicy::default(),
            filter_broadcast_config: FilterBroadcastConfig::default(),
            node_addr: "0000".to_string(),
            local_identity_hash: [0u8; 16],
            local_pq_identity_hash: [0u8; 16],
            local_dsa_pubkey: Vec::new(),
            inference_gguf_cid: None,
            inference_tokenizer_cid: None,
            disk_enabled: false,
            disk_entries: Vec::new(),
            disk_quota: None,
            s3_enabled: false,
        }
    }
}

/// Inbound events fed into the node runtime.
#[derive(Debug)]
pub enum RuntimeEvent {
    /// Tier 1: Reticulum packet received on a network interface.
    InboundPacket {
        interface_name: String,
        raw: Vec<u8>,
        now: u64,
    },
    /// Tier 1: Periodic timer tick for path expiry, announce scheduling.
    /// `now` is monotonic millis-since-start. `unix_now` is Unix epoch seconds.
    TimerTick { now: u64, unix_now: u64 },
    /// Tier 2: Zenoh query received (content fetch or stats request).
    QueryReceived {
        query_id: u64,
        key_expr: String,
        payload: Vec<u8>,
    },
    /// Tier 2: Zenoh subscription message (transit or publish).
    SubscriptionMessage { key_expr: String, payload: Vec<u8> },
    /// Tier 3: Compute query received (activity request).
    ComputeQuery {
        query_id: u64,
        key_expr: String,
        payload: Vec<u8>,
    },
    /// Tier 3: Module fetch response for a CID-referenced compute request.
    ModuleFetchResponse {
        cid: [u8; 32],
        result: Result<Vec<u8>, String>,
    },
    /// Tier 3: Content fetch response for a suspended compute task.
    ContentFetchResponse {
        cid: [u8; 32],
        result: Result<Vec<u8>, String>,
    },
    /// A tunnel handshake completed — register as a point-to-point interface.
    TunnelHandshakeComplete {
        interface_name: String,
        peer_node_id: [u8; 32],
    },
    /// A Reticulum packet arrived via a tunnel interface.
    TunnelReticulumReceived {
        interface_name: String,
        packet: Vec<u8>,
        now: u64,
    },
    /// A tunnel was closed.
    TunnelClosed { interface_name: String },
    /// A discovery announce record was received from the network.
    /// `unix_now` is Unix epoch seconds (NOT monotonic millis_since_start).
    DiscoveryAnnounceReceived {
        record_bytes: Vec<u8>,
        unix_now: u64,
    },
    /// Local tunnel endpoint info became available.
    LocalTunnelInfo {
        node_id: [u8; 32],
        relay_url: Option<String>,
    },
    /// Peer lifecycle: a tunnel connection has been established to a peer.
    TunnelPeerEstablished {
        identity_hash: [u8; 16],
        node_id: [u8; 32],
        now: u64,
    },
    /// Peer lifecycle: a tunnel connection attempt failed.
    TunnelPeerFailed { identity_hash: [u8; 16] },
    /// Peer lifecycle: an established tunnel connection was dropped.
    TunnelPeerDropped { identity_hash: [u8; 16] },
    /// Peer lifecycle: a contact was added or modified.
    ContactChanged { identity_hash: [u8; 16] },
    /// Replication: encrypted book pushed by a peer for backup storage.
    ReplicaPushReceived {
        peer_identity: [u8; 16],
        cid: [u8; 32],
        data: Vec<u8>,
    },
    /// Replication: a peer requests content with a bearer token.
    ReplicaPullWithTokenReceived {
        peer_identity: [u8; 16],
        cid: [u8; 32],
        token_bytes: Vec<u8>,
        /// Unix epoch seconds — injected by the event loop for sans-I/O testability.
        unix_now: u64,
    },
    /// A peer's ML-DSA public key was learned (from handshake or discovery).
    PeerPublicKeyLearned {
        identity_hash: [u8; 16],
        dsa_pubkey: Vec<u8>,
    },

    /// Memo fetch: caller requests memos for an input CID.
    MemoFetchRequest { input: ContentId },

    /// Memo fetch: Zenoh query reply arrived with memo data.
    /// `unix_now` is Unix epoch seconds — injected by the event loop for sans-I/O testability.
    MemoFetchResponse {
        key_expr: String,
        payload: Vec<u8>,
        unix_now: u64,
    },

    /// A raw L2 interface is ready for Reticulum traffic.
    L2InterfaceReady { interface_name: String },
    /// A raw L2 interface has been shut down.
    L2InterfaceClosed { interface_name: String },
    /// DSD: verify response received from target node.
    VerifyResponse { payload: Vec<u8> },

    /// Disk read completed — content fetched from persistent storage.
    DiskReadComplete {
        cid: ContentId,
        query_id: u64,
        data: Vec<u8>,
    },
    /// Disk read failed — file missing or corrupted.
    DiskReadFailed { cid: ContentId, query_id: u64 },

    /// S3 read completed — content fetched from S3 fallback.
    S3ReadComplete {
        cid: ContentId,
        query_id: u64,
        data: Vec<u8>,
    },
    /// S3 read failed — object missing or fetch error.
    S3ReadFailed { cid: ContentId, query_id: u64 },
}

/// Outbound actions returned by the runtime for the caller to execute.
#[derive(Debug, Clone, PartialEq)]
pub enum RuntimeAction {
    /// Tier 1: Send raw packet on a network interface.
    SendOnInterface {
        interface_name: Arc<str>,
        raw: Vec<u8>,
        /// Cooperation weight for probabilistic broadcast selection.
        /// `None` = directed send (always deliver).
        /// `Some(score)` = broadcast send (caller may drop if random > score).
        weight: Option<f32>,
    },
    /// Tier 2: Reply to a content or stats query.
    SendReply { query_id: u64, payload: Vec<u8> },
    /// Tier 2: Publish a message (e.g., content availability announcement).
    Publish { key_expr: String, payload: Vec<u8> },
    /// Tier 2/3: Fetch content by CID from storage / network.
    FetchContent { cid: [u8; 32] },
    /// Tier 3: Fetch a WASM module by CID. The caller must route the response
    /// to `RuntimeEvent::ModuleFetchResponse` (not `ContentFetchResponse`).
    FetchModule { cid: [u8; 32] },
    /// Setup: Declare a queryable key expression.
    DeclareQueryable { key_expr: String },
    /// Setup: Subscribe to a key expression.
    Subscribe { key_expr: String },
    /// Peer lifecycle: initiate a tunnel connection to a peer via iroh-net.
    InitiateTunnel {
        identity_hash: [u8; 16],
        node_id: [u8; 32],
        relay_url: Option<String>,
        /// ML-DSA-65 public key from the peer's announce record (1952 bytes).
        peer_dsa_pubkey: Vec<u8>,
        /// ML-KEM-768 public key from the peer's announce record (1184 bytes).
        peer_kem_pubkey: Vec<u8>,
    },
    /// Peer lifecycle: send a path request (announce probe) for a peer.
    SendPathRequest { identity_hash: [u8; 16] },
    /// Peer lifecycle: close an active tunnel connection for a peer.
    CloseTunnel { identity_hash: [u8; 16] },
    /// Replication: push encrypted book data to a peer for storage.
    ReplicaPush {
        peer_identity: [u8; 16],
        cid: [u8; 32],
        data: Vec<u8>,
    },
    /// Replication: serve replicated content in response to a validated PullWithToken.
    ReplicaPullResponse {
        peer_identity: [u8; 16],
        cid: [u8; 32],
        data: Vec<u8>,
    },
    /// Memo fetch: emit a Zenoh session.get() query for memos.
    /// The event loop calls session.get(key_expr) and feeds each reply
    /// back as RuntimeEvent::MemoFetchResponse.
    QueryMemo { key_expr: String },
    /// DSD: send a verify query to the target node's verify queryable.
    /// The event loop sends a Zenoh query to `key_expr` with `payload`,
    /// and feeds the reply back as `RuntimeEvent::VerifyResponse`.
    SendVerifyQuery { key_expr: String, payload: Vec<u8> },
    /// Persist a CAS book to disk (spawned as blocking I/O by event loop).
    PersistToDisk { cid: ContentId, data: Vec<u8> },
    /// Read a CAS book from disk (spawned as blocking I/O by event loop).
    DiskLookup { cid: ContentId, query_id: u64 },
    /// Delete a book from disk (evicted by quota enforcement).
    RemoveFromDisk { cid: ContentId },
    /// Fetch a CAS book from S3 fallback (spawned as async I/O by event loop).
    S3Lookup { cid: ContentId, query_id: u64 },
    /// Spawn a streaming inference task.
    ///
    /// Sampling parameters are stored as raw 20-byte LE wire format because
    /// `harmony_inference::SamplingParams` does not derive `PartialEq` (needed
    /// by `RuntimeAction`). The event loop decodes them before spawning.
    #[cfg(feature = "inference")]
    RunInference {
        query_id: u64,
        task_id: String,
        input: crate::inference_types::InferenceInput,
        sampling_params_raw: [u8; 20],
    },
}

/// Filter state received from a peer, with metadata.
///
/// Each filter type tracks its own received tick so that staleness detection
/// is independent — a fresh Bloom filter doesn't mask a stale Cuckoo filter.
struct PeerFilter {
    content_filter: Option<BloomFilter>,
    flatpack_filter: Option<CuckooFilter>,
    memo_filter: Option<BloomFilter>,
    page_filter: Option<BloomFilter>,
    content_received_tick: u64,
    flatpack_received_tick: u64,
    memo_received_tick: u64,
    page_received_tick: u64,
}

/// Per-peer Bloom filter table for content query routing.
struct PeerFilterTable {
    filters: HashMap<String, PeerFilter>,
    staleness_ticks: u64,
    /// Count of malformed filter payloads that failed deserialization.
    parse_errors: u64,
}

impl PeerFilterTable {
    fn new(staleness_ticks: u64) -> Self {
        Self {
            filters: HashMap::new(),
            staleness_ticks,
            parse_errors: 0,
        }
    }

    fn record_parse_error(&mut self) {
        self.parse_errors += 1;
    }

    fn parse_errors(&self) -> u64 {
        self.parse_errors
    }

    fn upsert_content(&mut self, peer_addr: String, filter: BloomFilter, tick: u64) {
        let entry = self.filters.entry(peer_addr).or_insert(PeerFilter {
            content_filter: None,
            flatpack_filter: None,
            memo_filter: None,
            page_filter: None,
            content_received_tick: tick,
            flatpack_received_tick: 0,
            memo_received_tick: 0,
            page_received_tick: 0,
        });
        entry.content_filter = Some(filter);
        entry.content_received_tick = tick;
    }

    fn upsert_flatpack(&mut self, peer_addr: String, filter: CuckooFilter, tick: u64) {
        let entry = self.filters.entry(peer_addr).or_insert(PeerFilter {
            content_filter: None,
            flatpack_filter: None,
            memo_filter: None,
            page_filter: None,
            content_received_tick: 0,
            flatpack_received_tick: tick,
            memo_received_tick: 0,
            page_received_tick: 0,
        });
        entry.flatpack_filter = Some(filter);
        entry.flatpack_received_tick = tick;
    }

    fn upsert_memo(&mut self, peer_addr: String, filter: BloomFilter, tick: u64) {
        let entry = self.filters.entry(peer_addr).or_insert(PeerFilter {
            content_filter: None,
            flatpack_filter: None,
            memo_filter: None,
            page_filter: None,
            content_received_tick: 0,
            flatpack_received_tick: 0,
            memo_received_tick: tick,
            page_received_tick: 0,
        });
        entry.memo_filter = Some(filter);
        entry.memo_received_tick = tick;
    }

    fn upsert_page(&mut self, peer_addr: String, filter: BloomFilter, tick: u64) {
        let entry = self.filters.entry(peer_addr).or_insert(PeerFilter {
            content_filter: None,
            flatpack_filter: None,
            memo_filter: None,
            page_filter: None,
            content_received_tick: 0,
            flatpack_received_tick: 0,
            memo_received_tick: 0,
            page_received_tick: tick,
        });
        entry.page_filter = Some(filter);
        entry.page_received_tick = tick;
    }

    /// Returns true if the peer should be queried (no filter, stale filter, or filter says "maybe").
    /// Returns false only if the filter definitively says the CID is absent.
    fn should_query(&self, peer_addr: &str, cid: &ContentId, current_tick: u64) -> bool {
        match self.filters.get(peer_addr) {
            None => true,
            Some(pf) => {
                if current_tick.saturating_sub(pf.content_received_tick) > self.staleness_ticks {
                    true
                } else {
                    match &pf.content_filter {
                        Some(bf) => bf.may_contain(cid),
                        None => true,
                    }
                }
            }
        }
    }

    /// Returns true if the peer should be queried for flatpack reverse-lookup entries.
    fn should_query_flatpack(
        &self,
        peer_addr: &str,
        child_cid: &ContentId,
        current_tick: u64,
    ) -> bool {
        match self.filters.get(peer_addr) {
            None => true,
            Some(pf) => {
                if current_tick.saturating_sub(pf.flatpack_received_tick) > self.staleness_ticks {
                    true
                } else {
                    match &pf.flatpack_filter {
                        Some(cf) => cf.may_contain(child_cid),
                        None => true,
                    }
                }
            }
        }
    }

    /// Returns true if the peer should be queried for memo input CIDs.
    fn should_query_memo(&self, peer_addr: &str, input_cid: &ContentId, current_tick: u64) -> bool {
        match self.filters.get(peer_addr) {
            None => true,
            Some(pf) => {
                if current_tick.saturating_sub(pf.memo_received_tick) > self.staleness_ticks {
                    true
                } else {
                    match &pf.memo_filter {
                        Some(bf) => bf.may_contain(input_cid),
                        None => true,
                    }
                }
            }
        }
    }

    /// Returns true if the peer should be queried for page addresses.
    fn should_query_page(
        &self,
        peer_addr: &str,
        page_addr: &harmony_athenaeum::PageAddr,
        current_tick: u64,
    ) -> bool {
        match self.filters.get(peer_addr) {
            None => true,
            Some(pf) => {
                if current_tick.saturating_sub(pf.page_received_tick) > self.staleness_ticks {
                    true
                } else {
                    match &pf.page_filter {
                        Some(bf) => bf.may_contain_bytes(&page_addr.to_bytes()),
                        None => true,
                    }
                }
            }
        }
    }

    /// Evict peers where all filters are stale.
    fn evict_stale(&mut self, current_tick: u64) {
        self.filters.retain(|_, pf| {
            let content_fresh =
                current_tick.saturating_sub(pf.content_received_tick) <= self.staleness_ticks;
            let flatpack_fresh =
                current_tick.saturating_sub(pf.flatpack_received_tick) <= self.staleness_ticks;
            let memo_fresh =
                current_tick.saturating_sub(pf.memo_received_tick) <= self.staleness_ticks;
            let page_fresh =
                current_tick.saturating_sub(pf.page_received_tick) <= self.staleness_ticks;
            content_fresh || flatpack_fresh || memo_fresh || page_fresh
        });
    }
}

/// Active DSD session state — tracks an in-progress speculative decoding loop.
#[cfg(feature = "inference")]
struct DsdSession {
    /// Query ID of the original speculative request (to reply when done).
    query_id: u64,
    /// Accepted token sequence so far (prompt + accepted draft + bonus tokens).
    accepted_tokens: Vec<u32>,
    /// Target node's verify key expression.
    target_verify_key: String,
    /// Maximum tokens to generate.
    max_tokens: u32,
    /// EOS token ID (from tokenizer, if available).
    eos_token_id: Option<u32>,
    /// Token IDs of the pending draft tokens (awaiting verification).
    pending_drafts: Vec<u32>,
    /// Length of the original prompt in tokens (for slicing generated output).
    prompt_len: usize,
}

/// Sans-I/O node runtime wiring Tier 1 (Router), Tier 2 (Storage), and Tier 3 (Compute).
///
/// Events are pushed via [`push_event`](Self::push_event) into internal
/// priority queues. Each [`tick`](Self::tick) processes events according to
/// the [`TierSchedule`] configuration:
///
/// - **Router/Storage:** drain up to `max_per_tick` events (default: drain all).
/// - **Compute:** one execution slice with fuel scaled by data-plane queue depth
///   (see [`AdaptiveCompute`]).
/// - **Starvation protection:** tiers idle beyond `starvation_threshold` ticks
///   are promoted in tick order.
///
/// With default configuration, behavior is: drain all router events, drain all
/// storage events, then run one compute slice — information flow is never
/// starved and compute gets whatever budget remains.
pub struct NodeRuntime<B: BookStore> {
    // Tier 1: Reticulum packet router
    router: Node,
    // Tier 1/2: Zenoh query dispatch
    queryable_router: QueryableRouter,
    // Tier 2: Content storage
    storage: StorageTier<B>,
    // Tier 3: Compute (workflow engine)
    workflow: WorkflowEngine,
    // Peer lifecycle manager (sans-I/O)
    peer_manager: PeerManager,
    // Contact store — intentional peer relationships
    contact_store: ContactStore,
    // Internal priority queues
    router_queue: VecDeque<NodeEvent>,
    storage_queue: VecDeque<StorageTierEvent>,
    // Queryable IDs belonging to the storage tier
    storage_queryable_ids: HashSet<QueryableId>,
    // Queryable IDs belonging to the compute tier
    compute_queryable_ids: HashSet<QueryableId>,
    // Pending workflow actions buffered between push_event and tick
    pending_workflow_actions: Vec<WorkflowAction>,
    // Direct runtime actions buffered from push_event (error replies, module fetches, etc.)
    pending_direct_actions: Vec<RuntimeAction>,
    // Maps WorkflowId -> query_ids for reply routing (multiple callers may
    // submit the same module+input; the engine deduplicates but all callers
    // need a reply).
    workflow_to_query: HashMap<WorkflowId, Vec<u64>>,
    // Maps module CID -> Vec<(query_id, input)> for CID-based requests pending module fetch
    cid_to_query: HashMap<[u8; 32], Vec<(u64, Vec<u8>)>>,
    // Tier scheduling configuration
    schedule: TierSchedule,
    // Starvation counters: incremented when a tier has no events in a tick, reset on processing
    router_starved: u32,
    storage_starved: u32,
    compute_starved: u32,
    // Per-peer Bloom filter table for content query routing
    peer_filters: PeerFilterTable,
    // Monotonically increasing tick counter
    tick_count: u64,
    // This node's address (hex-encoded), used for filter broadcast keys
    node_addr: String,
    // Ticks since last FilterTimerTick was injected
    ticks_since_filter_broadcast: u64,
    // How many ticks between filter broadcasts
    filter_broadcast_interval_ticks: u64,
    // Filter broadcast configuration (expected_items, fp_rate) for memo Bloom filter sizing.
    filter_broadcast_config: FilterBroadcastConfig,
    // Coalesces multiple BroadcastFilter actions within a single tick into one publish.
    pending_filter_broadcast: Option<Vec<u8>>,
    // Coalesces cuckoo filter broadcasts (flatpack reverse-index) within a single tick.
    pending_cuckoo_broadcast: Option<Vec<u8>>,
    // Coalesces memo Bloom filter broadcasts within a single tick.
    pending_memo_broadcast: Option<Vec<u8>>,
    // Coalesces page Bloom filter broadcasts within a single tick.
    pending_page_broadcast: Option<Vec<u8>>,
    // Memo attestation store (input CID → signed memos)
    memo_store: MemoStore,
    // Identity discovery manager (sans-I/O)
    discovery: DiscoveryManager,
    // Local tunnel routing hint (populated when iroh endpoint binds).
    // TODO: Wire into AnnounceBuilder when outgoing announce publishing
    // is implemented via DiscoveryAction::PublishAnnounce + Zenoh put.
    local_tunnel_hint: Option<harmony_discovery::RoutingHint>,
    // Most recent monotonic timestamp from TimerTick (used for PeerManager ticks).
    last_now: u64,
    // Replication: stores encrypted books on behalf of other peers
    replica_store: MemoryReplicaStore,
    // Replication: ticks since last replication scan
    ticks_since_replica_scan: u32,
    // Page index: maps page addresses to book pages for queryable lookup
    page_index: crate::page_index::PageIndex,
    // CIDs already indexed — prevents duplicate entries on re-announce
    indexed_book_cids: HashSet<ContentId>,
    // Queryable ID for the page namespace (harmony/page/**)
    page_queryable_id: QueryableId,
    // Cache of ML-DSA public keys by identity hash.
    // Populated from HandshakeComplete and AnnounceRecord events.
    // Capped at MAX_PUBKEY_CACHE_SIZE entries — evicts an arbitrary entry
    // on overflow (not LRU). A future improvement could use an LRU structure.
    // Announce records are verified for pubkey→hash binding by
    // verify_announce() before reaching this point — forged announces
    // are rejected with DiscoveryError::AddressMismatch.
    pubkey_cache: HashMap<[u8; 16], Vec<u8>>,
    // Queryable ID for the memo namespace (harmony/memo/**)
    memo_queryable_id: QueryableId,
    // In-flight memo fetches: input CID → tick when fetch was started.
    // Prevents re-querying for the same input while a fetch is in-flight.
    pending_memo_fetches: HashMap<ContentId, u64>,
    // This node's Ed25519 identity hash (copied from config for direct access).
    local_identity_hash: harmony_identity::IdentityHash,
    // This node's PQ identity hash (ML-KEM + ML-DSA derived).
    // Used for Discovery UCAN token validation (issuer check).
    local_pq_identity_hash: harmony_identity::IdentityHash,
    // Queryable ID for the discover namespace (harmony/discover/**)
    discover_queryable_id: QueryableId,
    // Pre-serialized public announce record (Reticulum-only hints).
    // Populated when outbound announce publishing is wired (existing TODO).
    // Until then, the discover queryable returns no reply (harmless — the
    // feature becomes active once announce building is implemented).
    local_public_announce: Option<Vec<u8>>,
    // Pre-serialized full announce record (all hints including tunnel).
    local_full_announce: Option<Vec<u8>>,
    // Unix epoch seconds, updated on each TimerTick for token time-bound checks.
    last_unix_now: u64,
    // Parsed ML-DSA-65 verifying key for self-issued Discovery tokens.
    // Parsed once at construction; None if local_dsa_pubkey is empty/invalid.
    local_dsa_verifying_key: Option<harmony_crypto::ml_dsa::MlDsaPublicKey>,
    /// Queryable ID for the inference service, if model is loaded.
    inference_queryable_id: Option<QueryableId>,
    /// GGUF model CID for inference runner input construction.
    inference_model_cid: Option<[u8; 32]>,
    /// Tokenizer CID for inference runner input construction.
    inference_tokenizer_cid: Option<[u8; 32]>,
    /// Whether the GGUF model data has been fetched from CAS.
    inference_gguf_data: Option<Vec<u8>>,
    /// Whether the tokenizer data has been fetched from CAS.
    inference_tokenizer_data: Option<Vec<u8>>,
    /// Monotonic nonce for inference requests — ensures each request gets a
    /// unique WorkflowId, preventing dedup of non-deterministic results.
    inference_request_nonce: u64,
    /// QwenEngine for native verification (DSD target side).
    /// Separate from the WasmiRuntime's persisted engine used by WASM workflows.
    #[cfg(feature = "inference")]
    verification_engine: Option<harmony_inference::QwenEngine>,
    /// Whether the inference engine is currently running a streaming task.
    /// Used by DSD paths to distinguish "engine busy" from "engine not loaded".
    #[cfg(feature = "inference")]
    inference_running: bool,
    /// Queryable ID for the DSD verify endpoint.
    #[cfg(feature = "inference")]
    verify_queryable_id: Option<QueryableId>,
    /// Active DSD session (one at a time for simplicity).
    #[cfg(feature = "inference")]
    dsd_session: Option<DsdSession>,
    /// Tick when the DSD session last had activity (set on start, reset on verify response).
    #[cfg(feature = "inference")]
    dsd_session_last_activity_tick: u64,
    /// Discovered target node address for DSD (differs from our model CID).
    #[cfg(feature = "inference")]
    dsd_target_addr: Option<String>,
    /// Queryable ID for the speculative inference endpoint.
    #[cfg(feature = "inference")]
    speculative_queryable_id: Option<QueryableId>,
}

/// Adapts the runtime's pubkey_cache to the CredentialKeyResolver trait
/// for memo signature verification.
struct PubkeyCacheKeyResolver<'a> {
    cache: &'a HashMap<[u8; 16], Vec<u8>>,
}

impl<'a> harmony_credential::CredentialKeyResolver for PubkeyCacheKeyResolver<'a> {
    fn resolve(&self, issuer: &harmony_identity::IdentityRef, _issued_at: u64) -> Option<Vec<u8>> {
        // Key rotation is not supported; each identity has a single lifetime key.
        self.cache.get(&issuer.hash).cloned()
    }
}

impl<B: BookStore> NodeRuntime<B> {
    /// Construct a new node runtime, returning startup actions the caller
    /// must execute (queryable declarations, subscriptions).
    pub fn new(config: NodeConfig, store: B) -> (Self, Vec<RuntimeAction>) {
        assert!(
            !matches!(config.schedule.router_max_per_tick, Some(0)),
            "router_max_per_tick must be None or > 0"
        );
        assert!(
            !matches!(config.schedule.storage_max_per_tick, Some(0)),
            "storage_max_per_tick must be None or > 0"
        );

        let router = Node::new();
        let mut queryable_router = QueryableRouter::new();

        let filter_broadcast_interval_ticks =
            config.filter_broadcast_config.max_interval_ticks as u64;
        assert!(
            filter_broadcast_interval_ticks >= 2,
            "filter_broadcast_interval_ticks must be >= 2; \
             with interval=1 the timer fires every tick"
        );

        let filter_broadcast_config = config.filter_broadcast_config.clone();
        let (storage, storage_startup) = StorageTier::new(
            store,
            config.storage_budget,
            config.content_policy,
            config.filter_broadcast_config,
        );

        let mut actions = Vec::new();
        let mut storage_queryable_ids = HashSet::new();

        // Process storage startup actions: register queryables and subscriptions
        for action in storage_startup {
            match action {
                StorageTierAction::DeclareQueryables { key_exprs } => {
                    for key_expr in &key_exprs {
                        let (qid, _qactions) = queryable_router
                            .declare(key_expr)
                            .expect("static key expression must be valid");
                        storage_queryable_ids.insert(qid);
                        actions.push(RuntimeAction::DeclareQueryable {
                            key_expr: key_expr.clone(),
                        });
                    }
                }
                StorageTierAction::DeclareSubscribers { key_exprs } => {
                    for key_expr in key_exprs {
                        actions.push(RuntimeAction::Subscribe { key_expr });
                    }
                }
                _ => {} // other startup actions not expected
            }
        }

        // Tier 3: Compute — register activity queryable
        let workflow = WorkflowEngine::new(
            Box::new(harmony_compute::WasmiRuntime::new()),
            config.compute_budget,
        );
        let mut compute_queryable_ids = HashSet::new();

        let (qid, _) = queryable_router
            .declare(harmony_zenoh::namespace::compute::ACTIVITY_SUB)
            .expect("static key expression must be valid");
        compute_queryable_ids.insert(qid);
        actions.push(RuntimeAction::DeclareQueryable {
            key_expr: harmony_zenoh::namespace::compute::ACTIVITY_SUB.to_string(),
        });

        // Page namespace: register queryable for page lookups.
        let (page_qid, _) = queryable_router
            .declare(page_ns::SUB)
            .expect("static key expression must be valid");
        actions.push(RuntimeAction::DeclareQueryable {
            key_expr: page_ns::SUB.to_string(),
        });

        // Memo namespace: register queryable for memo lookups.
        let (memo_qid, _) = queryable_router
            .declare(harmony_zenoh::namespace::memo::SUB)
            .expect("static key expression must be valid");
        actions.push(RuntimeAction::DeclareQueryable {
            key_expr: harmony_zenoh::namespace::memo::SUB.to_string(),
        });

        // Discover namespace: register queryable for authenticated routing hint queries.
        let (discover_qid, _) = queryable_router
            .declare(harmony_zenoh::namespace::discover::SUB)
            .expect("static key expression must be valid");
        actions.push(RuntimeAction::DeclareQueryable {
            key_expr: harmony_zenoh::namespace::discover::SUB.to_string(),
        });

        // Subscribe to peer filter broadcasts.
        actions.push(RuntimeAction::Subscribe {
            key_expr: harmony_zenoh::namespace::filters::CONTENT_SUB.to_string(),
        });
        actions.push(RuntimeAction::Subscribe {
            key_expr: harmony_zenoh::namespace::filters::FLATPACK_SUB.to_string(),
        });
        actions.push(RuntimeAction::Subscribe {
            key_expr: harmony_zenoh::namespace::filters::MEMO_SUB.to_string(),
        });
        actions.push(RuntimeAction::Subscribe {
            key_expr: harmony_zenoh::namespace::filters::PAGE_SUB.to_string(),
        });

        // Subscribe to capacity advertisements for DSD target discovery.
        #[cfg(feature = "inference")]
        actions.push(RuntimeAction::Subscribe {
            key_expr: harmony_zenoh::namespace::compute::CAPACITY_SUB.to_string(),
        });

        let mut rt = Self {
            router,
            queryable_router,
            storage,
            workflow,
            peer_manager: PeerManager::with_local_identity(config.local_identity_hash),
            contact_store: ContactStore::new(),
            router_queue: VecDeque::new(),
            storage_queue: VecDeque::new(),
            storage_queryable_ids,
            compute_queryable_ids,
            pending_workflow_actions: Vec::new(),
            pending_direct_actions: Vec::new(),
            workflow_to_query: HashMap::new(),
            cid_to_query: HashMap::new(),
            schedule: config.schedule.clone(),
            router_starved: 0,
            storage_starved: 0,
            compute_starved: 0,
            peer_filters: PeerFilterTable::new(filter_broadcast_interval_ticks * 3),
            tick_count: 0,
            node_addr: config.node_addr,
            ticks_since_filter_broadcast: 0,
            filter_broadcast_interval_ticks,
            filter_broadcast_config,
            pending_filter_broadcast: None,
            pending_cuckoo_broadcast: None,
            pending_memo_broadcast: None,
            pending_page_broadcast: None,
            memo_store: MemoStore::new(),
            discovery: DiscoveryManager::new(),
            local_tunnel_hint: None,
            last_now: 0,
            replica_store: MemoryReplicaStore::new(),
            ticks_since_replica_scan: 0,
            page_index: crate::page_index::PageIndex::new(),
            indexed_book_cids: HashSet::new(),
            page_queryable_id: page_qid,
            pubkey_cache: HashMap::new(),
            memo_queryable_id: memo_qid,
            pending_memo_fetches: HashMap::new(),
            local_identity_hash: config.local_identity_hash,
            local_pq_identity_hash: config.local_pq_identity_hash,
            discover_queryable_id: discover_qid,
            local_public_announce: None,
            local_full_announce: None,
            last_unix_now: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            local_dsa_verifying_key: harmony_crypto::ml_dsa::MlDsaPublicKey::from_bytes(
                &config.local_dsa_pubkey,
            )
            .ok(),
            inference_queryable_id: None,
            inference_model_cid: config.inference_gguf_cid,
            inference_tokenizer_cid: config.inference_tokenizer_cid,
            inference_gguf_data: None,
            inference_tokenizer_data: None,
            inference_request_nonce: 0,
            #[cfg(feature = "inference")]
            verification_engine: None,
            #[cfg(feature = "inference")]
            inference_running: false,
            #[cfg(feature = "inference")]
            verify_queryable_id: None,
            #[cfg(feature = "inference")]
            dsd_session: None,
            #[cfg(feature = "inference")]
            dsd_session_last_activity_tick: 0,
            #[cfg(feature = "inference")]
            dsd_target_addr: None,
            #[cfg(feature = "inference")]
            speculative_queryable_id: None,
        };

        // Activate disk tier when data_dir is configured (even if empty on first boot).
        if config.disk_enabled {
            rt.storage.enable_disk(config.disk_entries);
            if let Some(quota) = config.disk_quota {
                let eviction_actions = rt.storage.set_disk_quota(quota);
                // Startup eviction: if existing data exceeds the new quota,
                // emit RemoveFromDisk actions so the event loop cleans up
                // immediately rather than waiting for the first new persist.
                for action in eviction_actions {
                    if let StorageTierAction::RemoveFromDisk { cid } = action {
                        actions.push(RuntimeAction::RemoveFromDisk { cid });
                    }
                }
            }
        }

        // Activate S3 fallback tier for durable content.
        if config.s3_enabled {
            rt.storage.enable_s3();
        }

        // If inference model CIDs are configured AND the inference feature is
        // enabled, fetch both from CAS at startup. Without the feature, the
        // wasmi host functions (model_load etc.) aren't registered and WASM
        // instantiation would fail.
        #[cfg(feature = "inference")]
        {
            if let Some(gguf_cid) = rt.inference_model_cid {
                actions.push(RuntimeAction::FetchContent { cid: gguf_cid });
            }
            if let Some(tok_cid) = rt.inference_tokenizer_cid {
                actions.push(RuntimeAction::FetchContent { cid: tok_cid });
            }
        }
        #[cfg(not(feature = "inference"))]
        if config.inference_gguf_cid.is_some() || config.inference_tokenizer_cid.is_some() {
            tracing::warn!(
                "inference CIDs configured but 'inference' feature not enabled; inference disabled"
            );
        }

        (rt, actions)
    }

    /// Read-only access to the tier schedule configuration.
    pub fn schedule(&self) -> &TierSchedule {
        &self.schedule
    }

    /// Disable S3 fallback at runtime (e.g., when S3Library init fails).
    /// Prevents StorageTier from emitting S3Lookup actions.
    pub fn disable_s3(&mut self) {
        self.storage.disable_s3();
    }

    /// Read-only access to storage metrics.
    pub fn metrics(&self) -> &StorageMetrics {
        self.storage.metrics()
    }

    /// Number of pending Tier 1 (router) events.
    pub fn router_queue_len(&self) -> usize {
        self.router_queue.len()
    }

    /// Number of pending Tier 2 (storage) events.
    pub fn storage_queue_len(&self) -> usize {
        self.storage_queue.len()
    }

    /// Number of workflows tracked by the Tier 3 (compute) engine.
    pub fn compute_queue_len(&self) -> usize {
        self.workflow.workflow_count()
    }

    /// Current starvation counters (router, storage, compute).
    pub fn starvation_counters(&self) -> (u32, u32, u32) {
        (
            self.router_starved,
            self.storage_starved,
            self.compute_starved,
        )
    }

    /// Check if a peer should be queried for a given CID.
    ///
    /// Returns `false` only if the peer's Bloom filter definitively says
    /// the CID is absent. Returns `true` for unknown peers, stale filters,
    /// or filter matches.
    pub fn should_query_peer(&self, peer_addr: &str, cid: &ContentId) -> bool {
        self.peer_filters
            .should_query(peer_addr, cid, self.tick_count)
    }

    /// Check if a peer should be queried for flatpack reverse-lookup entries.
    pub fn should_query_peer_flatpack(&self, peer_addr: &str, child_cid: &ContentId) -> bool {
        self.peer_filters
            .should_query_flatpack(peer_addr, child_cid, self.tick_count)
    }

    /// Check if a peer should be queried for memo attestations about an input CID.
    ///
    /// Returns `false` only if the peer's memo Bloom filter definitively says
    /// the input CID is absent. Returns `true` for unknown peers, stale filters,
    /// or filter matches.
    pub fn should_query_memo_peer(&self, peer_addr: &str, input_cid: &ContentId) -> bool {
        self.peer_filters
            .should_query_memo(peer_addr, input_cid, self.tick_count)
    }

    /// Check if a peer should be queried for page addresses.
    ///
    /// Returns `false` only if the peer's page Bloom filter definitively says
    /// the page address is absent. Returns `true` for unknown peers, stale
    /// filters, or filter matches.
    pub fn should_query_page_peer(
        &self,
        peer_addr: &str,
        page_addr: &harmony_athenaeum::PageAddr,
    ) -> bool {
        self.peer_filters
            .should_query_page(peer_addr, page_addr, self.tick_count)
    }

    /// Read-only access to the contact store.
    pub fn contact_store(&self) -> &ContactStore {
        &self.contact_store
    }

    /// Mutable access to the contact store.
    pub fn contact_store_mut(&mut self) -> &mut ContactStore {
        &mut self.contact_store
    }

    /// Maximum number of cached public keys. Each ML-DSA-65 pubkey is ~1952 bytes,
    /// so 4096 entries ≈ 8 MB — acceptable for long-running nodes.
    const MAX_PUBKEY_CACHE_SIZE: usize = 4096;

    /// Handle a memo fetch request: check local store, dedup, issue Zenoh query.
    fn handle_memo_fetch_request(&mut self, input: ContentId) {
        // 1. Local check — short-circuit if we already have memos
        if !self.memo_store.get_by_input(&input).is_empty() {
            return;
        }

        // 2. Dedup check — skip if already in-flight
        if self.pending_memo_fetches.contains_key(&input) {
            return;
        }

        // 3. Issue Zenoh query
        let input_hex = hex::encode(input.to_bytes());
        let key_expr = harmony_zenoh::namespace::memo::input_query(&input_hex);
        self.pending_direct_actions
            .push(RuntimeAction::QueryMemo { key_expr });

        // 4. Track in-flight
        self.pending_memo_fetches.insert(input, self.tick_count);
    }

    /// Handle a memo fetch response: parse, verify, insert into MemoStore.
    ///
    /// Each Zenoh reply becomes a separate MemoFetchResponse event.
    /// Multiple peers may respond — all are processed independently.
    fn handle_memo_fetch_response(&mut self, key_expr: &str, payload: &[u8], unix_now: u64) {
        // Note: we intentionally do NOT check pending_memo_fetches here.
        // Responses may arrive after timeout expiry, and accepting any
        // verified memo is harmless (memos are self-certifying). The dedup
        // map only prevents re-issuing QueryMemo, not response processing.
        //
        // 1. Parse input_cid from key_expr
        let input_hex = match key_expr
            .strip_prefix(harmony_zenoh::namespace::memo::PREFIX)
            .and_then(|s| s.strip_prefix('/'))
        {
            Some(rest) => rest.split('/').next().unwrap_or(""),
            None => return,
        };

        let cid_bytes = match hex::decode(input_hex) {
            Ok(b) if b.len() == 32 => b,
            _ => return,
        };
        let mut arr = [0u8; 32];
        arr.copy_from_slice(&cid_bytes);
        let input_cid = ContentId::from_bytes(arr);

        // 2. Size guard
        if payload.len() > MAX_MEMO_RESPONSE_BYTES {
            return;
        }

        // 3. Decode response: [u16 LE count][u32 LE len][bytes]...
        if payload.len() < 2 {
            return;
        }
        let count = u16::from_le_bytes([payload[0], payload[1]]) as usize;
        let mut offset = 2;

        let resolver = PubkeyCacheKeyResolver {
            cache: &self.pubkey_cache,
        };

        for _ in 0..count {
            if offset + 4 > payload.len() {
                break;
            }
            let memo_len = u32::from_le_bytes([
                payload[offset],
                payload[offset + 1],
                payload[offset + 2],
                payload[offset + 3],
            ]) as usize;
            offset += 4;

            if offset + memo_len > payload.len() {
                break;
            }
            let memo_bytes = &payload[offset..offset + memo_len];
            offset += memo_len;

            // Deserialize
            let memo = match harmony_memo::deserialize(memo_bytes) {
                Ok(m) => m,
                Err(_) => continue,
            };

            // Verify input CID matches the requested input
            if memo.input != input_cid {
                continue;
            }

            // Verify credential signature and time bounds
            if harmony_memo::verify::verify_memo(&memo, unix_now, &resolver).is_err() {
                continue;
            }

            // Insert
            self.memo_store.insert(memo);
        }
    }

    /// Insert a public key into the cache, evicting a random entry if at capacity.
    fn insert_pubkey_capped(&mut self, identity_hash: [u8; 16], pubkey: Vec<u8>) {
        if self.pubkey_cache.len() >= Self::MAX_PUBKEY_CACHE_SIZE
            && !self.pubkey_cache.contains_key(&identity_hash)
        {
            // Evict one entry — take the first key from the iterator (arbitrary).
            if let Some(&evict_key) = self.pubkey_cache.keys().next() {
                self.pubkey_cache.remove(&evict_key);
            }
        }
        self.pubkey_cache.insert(identity_hash, pubkey);
    }

    /// Read-only access to the memo store.
    pub fn memo_store(&self) -> &MemoStore {
        &self.memo_store
    }

    /// Mutable access to the memo store.
    pub fn memo_store_mut(&mut self) -> &mut MemoStore {
        &mut self.memo_store
    }

    /// This node's Ed25519 identity hash (Reticulum-compatible).
    pub fn local_identity_hash(&self) -> [u8; 16] {
        self.local_identity_hash
    }

    /// This node's PQ identity hash (ML-KEM + ML-DSA derived).
    /// Used for discover query key expressions and Discovery token validation.
    pub fn local_pq_identity_hash(&self) -> [u8; 16] {
        self.local_pq_identity_hash
    }

    /// Set the pre-serialized public announce record (Reticulum-only hints).
    pub fn set_local_public_announce(&mut self, data: Vec<u8>) {
        self.local_public_announce = Some(data);
    }

    /// Set the pre-serialized full announce record (all hints including tunnel).
    pub fn set_local_full_announce(&mut self, data: Vec<u8>) {
        self.local_full_announce = Some(data);
    }

    /// Read-only access to the page index.
    pub fn page_index(&self) -> &crate::page_index::PageIndex {
        &self.page_index
    }

    /// Number of malformed filter payloads that failed deserialization.
    pub fn peer_filter_parse_errors(&self) -> u64 {
        self.peer_filters.parse_errors()
    }

    /// Calculate the effective compute fuel budget based on data-plane queue depth.
    ///
    /// At zero depth, returns the full base budget. As combined queue depth
    /// approaches `high_water`, fuel shrinks linearly toward `floor_fraction * base`.
    /// Above high_water, fuel stays at the floor.
    ///
    /// Special case: if `high_water` is 0, always returns `floor_fraction * base`
    /// regardless of queue depth (adaptive scaling is effectively maximally aggressive).
    pub fn effective_fuel(&self) -> u64 {
        let base = self.workflow.budget().fuel;
        let ac = &self.schedule.adaptive_compute;
        if ac.high_water == 0 {
            return (base as f64 * ac.floor_fraction.clamp(0.0, 1.0)).round() as u64;
        }
        let combined = self.router_queue.len() + self.storage_queue.len();
        let load_factor = (combined as f64 / ac.high_water as f64).min(1.0);
        let floor = ac.floor_fraction.clamp(0.0, 1.0);
        let scale = 1.0 - load_factor * (1.0 - floor);
        (base as f64 * scale).round() as u64
    }

    /// Push an event into the runtime's internal priority queues.
    pub fn push_event(&mut self, event: RuntimeEvent) {
        match event {
            RuntimeEvent::InboundPacket {
                interface_name,
                raw,
                now,
            } => {
                self.router_queue.push_back(NodeEvent::InboundPacket {
                    interface_name,
                    raw,
                    now,
                });
            }
            RuntimeEvent::TimerTick { now, unix_now } => {
                self.last_now = now;
                self.last_unix_now = unix_now;
                self.router_queue.push_back(NodeEvent::TimerTick { now });
            }
            RuntimeEvent::QueryReceived {
                query_id,
                key_expr,
                payload,
            } => {
                self.route_query(query_id, key_expr, payload);
            }
            RuntimeEvent::SubscriptionMessage { key_expr, payload } => {
                self.route_subscription(key_expr, payload);
            }
            RuntimeEvent::ComputeQuery {
                query_id,
                key_expr,
                payload,
            } => {
                let workflow_actions = self.route_compute_query(query_id, key_expr, payload);
                self.pending_workflow_actions.extend(workflow_actions);
            }
            RuntimeEvent::ModuleFetchResponse { cid, result } => {
                match result {
                    Ok(module) => {
                        // Convert pending CID-based requests to inline Submit now
                        // that we have the module bytes. SubmitByCid is stubbed in
                        // the workflow engine, so we go through Submit instead.
                        // TODO: Remove this manual handling when the engine's
                        // SubmitByCid/ModuleFetched stubs are implemented.
                        // See engine.rs WorkflowEvent::SubmitByCid stub comment.
                        // TODO: blake3_hash is computed once here but also re-computed
                        // inside handle_submit for each call. For large WASM modules
                        // with many pending queries, consider passing pre-computed hash.
                        if let Some(pending) = self.cid_to_query.remove(&cid) {
                            let module_hash = harmony_crypto::hash::blake3_hash(&module);
                            for (query_id, input) in pending {
                                let wf_id = WorkflowId::new(&module_hash, &input);
                                self.workflow_to_query
                                    .entry(wf_id)
                                    .or_default()
                                    .push(query_id);
                                let actions = self.workflow.handle(WorkflowEvent::Submit {
                                    module: module.clone(),
                                    input,
                                    hint: ComputeHint::PreferLocal,
                                });
                                self.pending_workflow_actions.extend(actions);
                            }
                        }
                    }
                    Err(_) => {
                        // Fail all pending CID-based requests for this module.
                        if let Some(pending) = self.cid_to_query.remove(&cid) {
                            for (query_id, _input) in pending {
                                let error_msg = format!("module not found: {}", hex::encode(cid));
                                let mut payload = vec![0x01];
                                payload.extend_from_slice(error_msg.as_bytes());
                                self.pending_direct_actions
                                    .push(RuntimeAction::SendReply { query_id, payload });
                            }
                        }
                    }
                }
            }
            RuntimeEvent::ContentFetchResponse { cid, result } => {
                // Check if this is inference model data arriving from startup fetch.
                let is_inference_gguf =
                    Some(cid) == self.inference_model_cid && self.inference_gguf_data.is_none();
                let is_inference_tok = Some(cid) == self.inference_tokenizer_cid
                    && self.inference_tokenizer_data.is_none();

                if is_inference_gguf || is_inference_tok {
                    match result {
                        Ok(data) => {
                            // Clone for workflow engine in case a workflow also needs
                            // this CID (unlikely but possible CID collision).
                            let wf_event = WorkflowEvent::ContentFetched {
                                cid,
                                data: data.clone(),
                            };
                            if is_inference_gguf {
                                self.inference_gguf_data = Some(data);
                            } else {
                                self.inference_tokenizer_data = Some(data);
                            }
                            self.check_inference_model_ready();
                            let actions = self.workflow.handle(wf_event);
                            self.pending_workflow_actions.extend(actions);
                        }
                        Err(e) => {
                            let label = if is_inference_gguf {
                                "GGUF model"
                            } else {
                                "tokenizer"
                            };
                            tracing::error!(
                                "failed to fetch inference {label} (CID {}): {e}; inference disabled",
                                hex::encode(cid)
                            );
                            self.inference_model_cid = None;
                            self.inference_tokenizer_cid = None;
                            self.inference_gguf_data = None;
                            self.inference_tokenizer_data = None;
                            let actions = self
                                .workflow
                                .handle(WorkflowEvent::ContentFetchFailed { cid });
                            self.pending_workflow_actions.extend(actions);
                        }
                    }
                } else {
                    // Non-inference content — forward to workflow engine
                    let event = match result {
                        Ok(data) => WorkflowEvent::ContentFetched { cid, data },
                        Err(_) => WorkflowEvent::ContentFetchFailed { cid },
                    };
                    let actions = self.workflow.handle(event);
                    self.pending_workflow_actions.extend(actions);
                }
            }
            RuntimeEvent::TunnelHandshakeComplete {
                interface_name,
                peer_node_id,
            } => {
                tracing::info!(
                    %interface_name,
                    peer = %hex::encode(&peer_node_id[..4]),
                    "tunnel handshake complete"
                );
                self.router.register_interface(
                    interface_name,
                    harmony_reticulum::InterfaceMode::PointToPoint,
                    None,
                );
            }
            RuntimeEvent::TunnelReticulumReceived {
                interface_name,
                packet,
                now,
            } => {
                self.router_queue.push_back(NodeEvent::InboundPacket {
                    interface_name,
                    raw: packet,
                    now,
                });
            }
            RuntimeEvent::TunnelClosed { interface_name } => {
                tracing::info!(%interface_name, "tunnel closed — interface unregistered");
                self.router.unregister_interface(&interface_name);
            }
            RuntimeEvent::DiscoveryAnnounceReceived {
                record_bytes,
                unix_now,
            } => {
                // unix_now is Unix epoch seconds, provided by the caller for
                // sans-I/O testability. Discovery timestamps (published_at,
                // expires_at) are Unix epoch seconds.
                let record = match harmony_discovery::AnnounceRecord::deserialize(&record_bytes) {
                    Ok(r) => r,
                    Err(e) => {
                        tracing::debug!("invalid announce record: {e:?}");
                        return;
                    }
                };
                // Pre-reject before touching the manager's cache. DiscoveryManager
                // also verifies internally, but this avoids unnecessary state mutations.
                if let Err(e) = harmony_discovery::verify_announce(&record, unix_now) {
                    tracing::debug!("announce verification failed: {e:?}");
                    return;
                }
                let identity_hash = record.identity_ref.hash;

                // Snapshot the cached published_at BEFORE feeding to the manager,
                // so we can detect whether the record was actually accepted
                // (not rejected as stale duplicate).
                let old_published_at = self
                    .discovery
                    .get_record(&identity_hash, unix_now)
                    .map(|r| r.published_at);

                // Feed to DiscoveryManager — it performs staleness checks
                // (rejects records with published_at <= cached published_at).
                let actions =
                    self.discovery
                        .on_event(harmony_discovery::DiscoveryEvent::AnnounceReceived {
                            record,
                            now: unix_now,
                        });
                self.dispatch_discovery_actions(actions);

                // Only process tunnel hints if the manager actually updated
                // the cached record (strictly newer published_at, or first seen).
                let new_published_at = self
                    .discovery
                    .get_record(&identity_hash, unix_now)
                    .map(|r| r.published_at);
                let was_updated = new_published_at != old_published_at;
                if was_updated {
                    // DiscoveryManager.IdentityDiscovered requires the peer to
                    // be in the `online` set (via LivelinessChange), which needs
                    // Zenoh liveliness token wiring — not yet implemented.
                    // Process hints directly from the accepted record.
                    if let Some(cached) = self.discovery.get_record(&identity_hash, unix_now) {
                        let cached_clone = cached.clone();
                        self.process_discovered_tunnel_hints(&cached_clone);
                    }
                }
            }
            RuntimeEvent::LocalTunnelInfo { node_id, relay_url } => {
                self.local_tunnel_hint = Some(harmony_discovery::RoutingHint::Tunnel {
                    node_id,
                    relay_url,
                    direct_addrs: vec![],
                });
            }
            RuntimeEvent::TunnelPeerEstablished {
                identity_hash,
                node_id,
                now,
            } => {
                // PeerManager uses seconds; event loop provides milliseconds.
                let peer_actions = self.peer_manager.on_event(
                    PeerEvent::TunnelEstablished {
                        identity_hash,
                        node_id,
                        now: now / 1000,
                    },
                    &self.contact_store,
                );
                self.translate_peer_actions(peer_actions);
            }
            RuntimeEvent::TunnelPeerFailed { identity_hash } => {
                let peer_actions = self.peer_manager.on_event(
                    PeerEvent::TunnelFailed { identity_hash },
                    &self.contact_store,
                );
                self.translate_peer_actions(peer_actions);
            }
            RuntimeEvent::TunnelPeerDropped { identity_hash } => {
                let peer_actions = self.peer_manager.on_event(
                    PeerEvent::TunnelDropped { identity_hash },
                    &self.contact_store,
                );
                self.translate_peer_actions(peer_actions);
            }
            RuntimeEvent::ContactChanged { identity_hash } => {
                let peer_actions = self.peer_manager.on_event(
                    PeerEvent::ContactChanged { identity_hash },
                    &self.contact_store,
                );
                self.translate_peer_actions(peer_actions);
            }
            RuntimeEvent::ReplicaPushReceived {
                peer_identity,
                cid,
                data,
            } => {
                // Store the replica if the peer has quota.
                let quota = self
                    .contact_store
                    .get(&peer_identity)
                    .and_then(|c| c.replication.as_ref())
                    .map(|r| r.quota_bytes)
                    .unwrap_or(0);
                if quota > 0 {
                    let _ = self.replica_store.store(peer_identity, cid, data, quota);
                }
            }
            RuntimeEvent::ReplicaPullWithTokenReceived {
                peer_identity,
                cid,
                token_bytes,
                unix_now,
            } => {
                if let Some(action) =
                    self.handle_pull_with_token(peer_identity, cid, &token_bytes, unix_now)
                {
                    self.pending_direct_actions.push(action);
                }
            }
            RuntimeEvent::PeerPublicKeyLearned {
                identity_hash,
                dsa_pubkey,
            } => {
                // Guard against empty keys from degraded handshakes —
                // don't overwrite a valid cached key with an empty one.
                if !dsa_pubkey.is_empty() {
                    self.insert_pubkey_capped(identity_hash, dsa_pubkey);
                }
            }
            RuntimeEvent::MemoFetchRequest { input } => {
                self.handle_memo_fetch_request(input);
            }
            RuntimeEvent::MemoFetchResponse {
                key_expr,
                payload,
                unix_now,
            } => {
                self.handle_memo_fetch_response(&key_expr, &payload, unix_now);
            }
            RuntimeEvent::L2InterfaceReady { interface_name } => {
                tracing::info!(%interface_name, "L2 interface registered");
                self.router.register_interface(
                    interface_name,
                    harmony_reticulum::InterfaceMode::Full,
                    None,
                );
            }
            RuntimeEvent::L2InterfaceClosed { interface_name } => {
                tracing::info!(%interface_name, "L2 interface closed — unregistered");
                self.router.unregister_interface(&interface_name);
            }
            RuntimeEvent::VerifyResponse { payload } => {
                #[cfg(feature = "inference")]
                self.handle_verify_response(payload);
                #[cfg(not(feature = "inference"))]
                {
                    let _ = payload;
                }
            }
            RuntimeEvent::DiskReadComplete {
                cid,
                query_id,
                data,
            } => {
                let storage_actions = self.storage.handle(StorageTierEvent::DiskReadComplete {
                    cid,
                    query_id,
                    data,
                });
                self.dispatch_storage_actions_inline(storage_actions);
            }
            RuntimeEvent::DiskReadFailed { cid, query_id } => {
                let storage_actions = self
                    .storage
                    .handle(StorageTierEvent::DiskReadFailed { cid, query_id });
                self.dispatch_storage_actions_inline(storage_actions);
            }
            RuntimeEvent::S3ReadComplete {
                cid,
                query_id,
                data,
            } => {
                let storage_actions = self.storage.handle(StorageTierEvent::S3ReadComplete {
                    cid,
                    query_id,
                    data,
                });
                self.dispatch_storage_actions_inline(storage_actions);
            }
            RuntimeEvent::S3ReadFailed { cid, query_id } => {
                let storage_actions = self
                    .storage
                    .handle(StorageTierEvent::S3ReadFailed { cid, query_id });
                self.dispatch_storage_actions_inline(storage_actions);
            }
        }
    }

    /// Run one iteration of the priority event loop.
    ///
    /// Default order: Router → Storage → Compute. When a tier's starvation
    /// counter reaches the configured threshold, it is promoted to the front
    /// of the processing order for that tick.
    ///
    /// Within each tier, at most `max_per_tick` events are drained (if set).
    /// When limits are `None` (the default), all queued events are drained.
    pub fn tick(&mut self) -> Vec<RuntimeAction> {
        self.tick_count += 1;

        // Increment the filter timer counter. The actual FilterTimerTick is
        // injected after the storage drain (inside tier 1 processing), so we
        // can skip it when a threshold-triggered broadcast already fired.
        self.ticks_since_filter_broadcast += 1;

        // Evict stale peer filters.
        self.peer_filters.evict_stale(self.tick_count);

        // Expire in-flight memo fetches that have timed out.
        self.pending_memo_fetches.retain(|_, started| {
            self.tick_count.saturating_sub(*started) < MEMO_FETCH_TIMEOUT_TICKS
        });

        // Evict stale DSD sessions that never received a VerifyResponse.
        #[cfg(feature = "inference")]
        if let Some(ref session) = self.dsd_session {
            if self
                .tick_count
                .saturating_sub(self.dsd_session_last_activity_tick)
                >= DSD_SESSION_TIMEOUT_TICKS
            {
                let query_id = session.query_id;
                tracing::warn!(
                    query_id,
                    "DSD session timed out after {DSD_SESSION_TIMEOUT_TICKS} ticks"
                );
                let payload =
                    harmony_speculative::VerifyResponse::serialize_error("DSD session timed out");
                self.pending_direct_actions
                    .push(RuntimeAction::SendReply { query_id, payload });
                self.dsd_session = None;
            }
        }

        let mut actions = Vec::new();
        // Note: fuel is captured once before any tier executes. When Compute is
        // promoted via starvation, it still runs with this pre-tick fuel value —
        // promotion reorders execution but does not bypass adaptive scaling.
        // Operators tuning both starvation_threshold and adaptive_compute should
        // be aware that promotion does not grant additional fuel budget.
        let effective_fuel = self.effective_fuel();
        let threshold = self.schedule.starvation_threshold;

        // Determine tier order: promote starved tiers to front.
        // Default order: [0=Router, 1=Storage, 2=Compute]
        let mut order = [0u8, 1, 2];
        let starved = [
            self.router_starved,
            self.storage_starved,
            self.compute_starved,
        ];
        // Stable sort: starved tiers (>= threshold) move to front, preserving original priority order
        order.sort_by_key(|&tier| {
            if starved[tier as usize] >= threshold {
                0
            } else {
                1
            }
        });

        // Emit direct replies (error responses, module fetch requests) buffered from
        // push_event. These are cross-tier concerns and should not move with tier reordering.
        actions.append(&mut self.pending_direct_actions);

        for &tier in &order {
            match tier {
                0 => {
                    // Tier 1: Router
                    let limit = self.schedule.router_max_per_tick.unwrap_or(usize::MAX);
                    let mut processed = 0;
                    while processed < limit {
                        match self.router_queue.pop_front() {
                            Some(event) => {
                                let node_actions = self.router.handle_event(event);
                                self.dispatch_router_actions(node_actions, &mut actions);
                                processed += 1;
                            }
                            None => break,
                        }
                    }
                    if processed > 0 {
                        self.router_starved = 0;
                    } else {
                        self.router_starved = self.router_starved.saturating_add(1);
                    }
                }
                1 => {
                    // Tier 2: Storage
                    // Snapshot the timer state BEFORE processing events — a
                    // threshold-triggered BroadcastFilter resets the counter,
                    // but should not suppress the timer if it was due to fire.
                    let timer_due =
                        self.ticks_since_filter_broadcast >= self.filter_broadcast_interval_ticks;

                    let limit = self.schedule.storage_max_per_tick.unwrap_or(usize::MAX);
                    let mut processed = 0;
                    while processed < limit {
                        match self.storage_queue.pop_front() {
                            Some(event) => {
                                let storage_actions = self.storage.handle(event);
                                self.dispatch_storage_actions(storage_actions, &mut actions);
                                processed += 1;
                            }
                            None => break,
                        }
                    }
                    // Timer-triggered filter rebuild: fires when the interval
                    // has elapsed. Both bloom and cuckoo filters are rebuilt
                    // together. The pending_filter_broadcast / pending_cuckoo_broadcast
                    // buffers coalesce multiple rebuilds within a tick, so
                    // firing alongside a threshold-triggered bloom broadcast
                    // is harmless — the latest snapshot wins at flush time.
                    if timer_due {
                        self.ticks_since_filter_broadcast = 0;
                        let timer_actions = self.storage.handle(StorageTierEvent::FilterTimerTick);
                        self.dispatch_storage_actions(timer_actions, &mut actions);

                        // Rebuild memo Bloom filter from MemoStore input CIDs.
                        if !self.memo_store.is_empty() {
                            let mut memo_filter = BloomFilter::new(
                                self.filter_broadcast_config.expected_items,
                                self.filter_broadcast_config.fp_rate,
                            );
                            for cid in self.memo_store.input_cids() {
                                memo_filter.insert(cid);
                            }
                            self.pending_memo_broadcast = Some(memo_filter.to_bytes());
                        }

                        // Rebuild page Bloom filter from PageIndex mode-00 addresses.
                        if !self.page_index.is_empty() {
                            let mut page_filter = BloomFilter::new(
                                self.filter_broadcast_config.expected_items,
                                self.filter_broadcast_config.fp_rate,
                            );
                            for addr in self.page_index.addr00_iter() {
                                page_filter.insert_bytes(&addr.to_bytes());
                            }
                            self.pending_page_broadcast = Some(page_filter.to_bytes());
                        }
                    }

                    // Flush coalesced filter broadcast — multiple threshold crossings
                    // within one tick produce only one publish (the latest snapshot).
                    if let Some(payload) = self.pending_filter_broadcast.take() {
                        let key_expr =
                            harmony_zenoh::namespace::filters::content_key(&self.node_addr);
                        actions.push(RuntimeAction::Publish { key_expr, payload });
                    }
                    if let Some(payload) = self.pending_cuckoo_broadcast.take() {
                        let key_expr =
                            harmony_zenoh::namespace::filters::flatpack_key(&self.node_addr);
                        actions.push(RuntimeAction::Publish { key_expr, payload });
                    }
                    if let Some(payload) = self.pending_memo_broadcast.take() {
                        let key_expr = harmony_zenoh::namespace::filters::memo_key(&self.node_addr);
                        actions.push(RuntimeAction::Publish { key_expr, payload });
                    }
                    if let Some(payload) = self.pending_page_broadcast.take() {
                        let key_expr = harmony_zenoh::namespace::filters::page_key(&self.node_addr);
                        actions.push(RuntimeAction::Publish { key_expr, payload });
                    }
                    if processed > 0 {
                        self.storage_starved = 0;
                    } else {
                        self.storage_starved = self.storage_starved.saturating_add(1);
                    }
                }
                2 => {
                    // Tier 3: Compute — dispatch pending workflow actions, then one slice
                    let pending = std::mem::take(&mut self.pending_workflow_actions);
                    let had_pending = !pending.is_empty();
                    self.dispatch_workflow_actions(pending, &mut actions);
                    let effective_budget = InstructionBudget {
                        fuel: effective_fuel,
                    };
                    let workflow_actions = self.workflow.tick_with_budget(effective_budget);
                    let had_work = had_pending || !workflow_actions.is_empty();
                    self.dispatch_workflow_actions(workflow_actions, &mut actions);
                    if had_work {
                        self.compute_starved = 0;
                    } else {
                        self.compute_starved = self.compute_starved.saturating_add(1);
                    }
                }
                _ => unreachable!(),
            }
        }

        // Feed discovery tick — uses Unix epoch seconds (not monotonic)
        // because AnnounceRecord.published_at/expires_at are Unix epoch.
        // NOTE: This is the one place where tick() calls SystemTime::now()
        // rather than using a passed-in timestamp. Discovery expiry needs
        // wall-clock time; the caller (event loop) only provides monotonic
        // millis. If tick() is later refactored to accept a unix_now param,
        // this should use it instead.
        let disc_unix_now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let disc_actions = self
            .discovery
            .on_event(harmony_discovery::DiscoveryEvent::Tick { now: disc_unix_now });
        self.dispatch_discovery_actions(disc_actions);

        // Peer lifecycle tick — process probe timers, connecting timeouts.
        // PeerManager constants are in seconds; the event loop provides
        // milliseconds via millis_since_start(). Convert at the boundary.
        let peer_actions = self.peer_manager.on_event(
            PeerEvent::Tick {
                now: self.last_now / 1000,
            },
            &self.contact_store,
        );
        self.translate_peer_actions_out(peer_actions, &mut actions);

        // Replication scan: every 240 ticks (~60s at 250ms tick interval),
        // scan for encrypted-durable books not yet replicated to peers.
        self.ticks_since_replica_scan += 1;
        if self.ticks_since_replica_scan >= 240 {
            self.ticks_since_replica_scan = 0;
            self.scan_unreplicated_books(&mut actions);
        }

        actions
    }

    /// Scan for encrypted-durable books that haven't been replicated to peers
    /// and emit `ReplicaPush` actions.
    ///
    /// Currently a no-op: the `BookStore` trait doesn't expose content iteration
    /// yet. When `BookStore::iter()` is added (future bead), this method will
    /// walk local encrypted-durable books and emit one `ReplicaPush` per peer
    /// per scan cycle. A `replicated_set` dedup structure should be re-added
    /// at that point with proper cleanup logic.
    fn scan_unreplicated_books(&mut self, _out: &mut Vec<RuntimeAction>) {
        // Collect replication-eligible peers — used to gate the scan.
        let has_replication_peers = self
            .contact_store
            .iter()
            .any(|(_, c)| c.replication.is_some() && c.peering.enabled);

        if !has_replication_peers {
            return;
        }

        // TODO: When BookStore gains an iter() method, iterate local
        // encrypted-durable books and emit ReplicaPush for each peer
        // with an active replication policy.
    }

    fn dispatch_router_actions(
        &mut self,
        node_actions: Vec<NodeAction>,
        out: &mut Vec<RuntimeAction>,
    ) {
        for action in node_actions {
            match action {
                NodeAction::SendOnInterface {
                    interface_name,
                    raw,
                    weight,
                } => {
                    out.push(RuntimeAction::SendOnInterface {
                        interface_name,
                        raw,
                        weight,
                    });
                }
                NodeAction::AnnounceReceived {
                    validated_announce, ..
                } => {
                    // Feed announce into PeerManager so it can trigger
                    // link/tunnel initiation for known contacts.
                    // Use identity.address_hash (the raw identity hash), NOT
                    // destination_hash (which is SHA256(name_hash || address_hash)[:16]).
                    let identity_hash = validated_announce.identity.address_hash;
                    let peer_actions = self.peer_manager.on_event(
                        PeerEvent::AnnounceReceived { identity_hash },
                        &self.contact_store,
                    );
                    self.translate_peer_actions_out(peer_actions, out);
                }
                // Other router actions are diagnostics — drop for now.
                _ => {}
            }
        }
    }

    fn dispatch_storage_actions(
        &mut self,
        storage_actions: Vec<StorageTierAction>,
        out: &mut Vec<RuntimeAction>,
    ) {
        for action in storage_actions {
            match action {
                StorageTierAction::SendReply { query_id, payload } => {
                    out.push(RuntimeAction::SendReply { query_id, payload });
                }
                StorageTierAction::AnnounceContent { key_expr, payload } => {
                    out.push(RuntimeAction::Publish {
                        key_expr: key_expr.clone(),
                        payload,
                    });

                    // If the announced CID is a depth-0 book, index its pages
                    // and publish page metadata to the page namespace.
                    if let Some(cid_hex) = key_expr
                        .strip_prefix(announce_ns::PREFIX)
                        .and_then(|s| s.strip_prefix('/'))
                    {
                        if let Ok(cid_bytes) = hex::decode(cid_hex) {
                            if let Ok(cid_arr) = <[u8; 32]>::try_from(cid_bytes) {
                                let cid = ContentId::from_bytes(cid_arr);
                                if cid.cid_type() == harmony_content::cid::CidType::Book
                                    && !self.indexed_book_cids.contains(&cid)
                                {
                                    if let Some(data) = self.storage.get(&cid) {
                                        if let Ok(book) =
                                            harmony_athenaeum::Book::from_book(cid.to_bytes(), data)
                                        {
                                            self.indexed_book_cids.insert(cid);
                                            self.page_index.insert_book(cid, &book);
                                            for (i, addrs) in book.data_pages().iter().enumerate() {
                                                let a00 = hex::encode(addrs[0].to_bytes());
                                                let a01 = hex::encode(addrs[1].to_bytes());
                                                let a10 = hex::encode(addrs[2].to_bytes());
                                                let a11 = hex::encode(addrs[3].to_bytes());
                                                let book_cid_hex = hex::encode(cid.to_bytes());
                                                let pk = page_ns::page_key(
                                                    &a00,
                                                    &a01,
                                                    &a10,
                                                    &a11,
                                                    &book_cid_hex,
                                                    i as u8,
                                                );
                                                out.push(RuntimeAction::Publish {
                                                    key_expr: pk,
                                                    payload: vec![i as u8],
                                                });
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                StorageTierAction::SendStatsReply { query_id, payload } => {
                    out.push(RuntimeAction::SendReply { query_id, payload });
                }
                // Startup-only declarations — already processed in new().
                StorageTierAction::DeclareQueryables { .. }
                | StorageTierAction::DeclareSubscribers { .. } => {}
                StorageTierAction::PersistToDisk { cid, data } => {
                    out.push(RuntimeAction::PersistToDisk { cid, data });
                }
                StorageTierAction::DiskLookup { cid, query_id } => {
                    out.push(RuntimeAction::DiskLookup { cid, query_id });
                }
                StorageTierAction::RemoveFromDisk { cid } => {
                    out.push(RuntimeAction::RemoveFromDisk { cid });
                }
                StorageTierAction::S3Lookup { cid, query_id } => {
                    out.push(RuntimeAction::S3Lookup { cid, query_id });
                }
                StorageTierAction::BroadcastFilter { payload } => {
                    // Buffer the latest filter payload — multiple threshold crossings
                    // within one tick are coalesced into a single publish at flush time.
                    self.pending_filter_broadcast = Some(payload);
                    // Reset timer so threshold-triggered broadcasts defer the next
                    // timer-triggered one, avoiding redundant back-to-back rebuilds.
                    self.ticks_since_filter_broadcast = 0;
                }
                StorageTierAction::BroadcastCuckooFilter { payload } => {
                    self.pending_cuckoo_broadcast = Some(payload);
                }
            }
        }
    }

    /// Variant of `dispatch_storage_actions` used from `push_event` where no
    /// output vec is available. Buffers actions into `pending_direct_actions`
    /// so they are emitted at the start of the next `tick()`.
    fn dispatch_storage_actions_inline(&mut self, storage_actions: Vec<StorageTierAction>) {
        let mut out = std::mem::take(&mut self.pending_direct_actions);
        self.dispatch_storage_actions(storage_actions, &mut out);
        self.pending_direct_actions = out;
    }

    fn dispatch_discovery_actions(&mut self, _actions: Vec<harmony_discovery::DiscoveryAction>) {
        // Tunnel hint processing is done at the call site (after staleness
        // check) rather than here via IdentityDiscovered, because
        // IdentityDiscovered requires Zenoh liveliness tokens (not wired).
        // Other actions (PublishAnnounce, SetLiveliness, etc.) deferred to
        // full Zenoh wiring.
    }

    /// Validate a PullWithToken request and emit a PullResponse if authorized.
    ///
    /// 9-step validation: (0) reject oversized payload, (1) deserialize token,
    /// (2) check capability==Content, (3) check resource==cid, (4) check expiry,
    /// (5) check not-before, (6) verify ML-DSA-65 signature,
    /// (7) check audience==peer_identity, (8) retrieve replica.
    /// `unix_now` is Unix epoch seconds, injected by the caller for sans-I/O
    /// testability. The event loop computes this from `SystemTime::now()`.
    fn handle_pull_with_token(
        &self,
        peer_identity: [u8; 16],
        cid: [u8; 32],
        token_bytes: &[u8],
        unix_now: u64,
    ) -> Option<RuntimeAction> {
        use harmony_content::replica::ReplicaStore;

        // 0. Reject oversized payloads before allocating for deserialization.
        //    A PqUcanToken with a 32-byte resource + ML-DSA-65 signature (3309B)
        //    serializes to ~3.4 KB. 8 KB is generous headroom.
        const MAX_TOKEN_BYTES: usize = 8 * 1024;
        if token_bytes.len() > MAX_TOKEN_BYTES {
            tracing::debug!(len = token_bytes.len(), "token payload too large");
            return None;
        }

        // 1. Deserialize token
        let token = match harmony_identity::PqUcanToken::from_bytes(token_bytes) {
            Ok(t) => t,
            Err(e) => {
                tracing::debug!("token parse failed: {e:?}");
                return None;
            }
        };

        // 2. Check capability
        if token.capability != harmony_identity::CapabilityType::Content {
            tracing::debug!("token capability is not Content");
            return None;
        }

        // 3. Check resource matches CID
        if token.resource.len() != 32 || token.resource[..] != cid[..] {
            tracing::debug!("token resource doesn't match requested CID");
            return None;
        }

        // 4. Check expiry — use `>` (not `>=`) to match verify_token in ucan.rs,
        //    which accepts the token at the exact expires_at timestamp.
        if token.expires_at != 0 && unix_now > token.expires_at {
            tracing::debug!("token expired");
            return None;
        }

        // 5. Check not-before
        if token.not_before > unix_now {
            tracing::debug!("token not yet valid");
            return None;
        }

        // 6. Verify ML-DSA signature BEFORE any state-dependent lookups.
        //    Checking the replica store first would create a timing oracle:
        //    an attacker could probe which (issuer, cid) pairs exist by
        //    observing fast-reject (no replica) vs slow-reject (sig verify).
        let pubkey_bytes = match self.pubkey_cache.get(&token.issuer) {
            Some(pk) => pk,
            None => {
                tracing::debug!("issuer public key not cached");
                return None;
            }
        };
        let pubkey = match harmony_crypto::ml_dsa::MlDsaPublicKey::from_bytes(pubkey_bytes) {
            Ok(pk) => pk,
            Err(_) => {
                tracing::debug!("cached pubkey invalid");
                return None;
            }
        };
        if token.verify_signature(&pubkey).is_err() {
            tracing::debug!("token signature verification failed");
            return None;
        }

        // 7. Check audience — token must be scoped to this requester.
        //    The audience is part of the signed payload, so signature
        //    verification already authenticated it — we just enforce it.
        if token.audience != peer_identity {
            tracing::debug!("token audience mismatch");
            return None;
        }

        // 8. Signature verified, audience matched — now safe to probe replica store.
        //    Single retrieve() call serves both existence check and data fetch.
        let data = match self.replica_store.retrieve(&token.issuer, &cid) {
            Some(d) => d,
            None => {
                tracing::debug!("no replica from token issuer for this CID");
                return None;
            }
        };
        Some(RuntimeAction::ReplicaPullResponse {
            peer_identity,
            cid,
            data,
        })
    }

    /// Extract `RoutingHint::Tunnel` from a discovery record and auto-populate
    /// `ContactAddress::Tunnel` in the contact store.
    pub fn process_discovered_tunnel_hints(&mut self, record: &harmony_discovery::AnnounceRecord) {
        use harmony_contacts::{Contact, ContactAddress, PeeringPolicy, PeeringPriority};
        use harmony_discovery::RoutingHint;

        let identity_hash = record.identity_ref.hash;

        // Cache the peer's signing public key for token verification.
        // For MlDsa65 suites, record.public_key is the ML-DSA-65 verifying key
        // (1952 bytes), NOT the combined PQ key (3136 bytes). This matches the
        // format expected by MlDsaPublicKey::from_bytes() in handle_pull_with_token.
        // verify_announce() in the discovery crate uses the same key format.
        //
        // Pubkey→hash binding is verified by verify_announce() before this
        // point — forged announces are rejected with AddressMismatch.
        if !record.public_key.is_empty() {
            self.insert_pubkey_capped(identity_hash, record.public_key.clone());
        }

        let tunnel_hints: Vec<_> = record
            .routing_hints
            .iter()
            .filter_map(|hint| {
                if let RoutingHint::Tunnel {
                    node_id,
                    relay_url,
                    direct_addrs,
                } = hint
                {
                    Some(ContactAddress::Tunnel {
                        node_id: *node_id,
                        relay_url: relay_url.clone(),
                        direct_addrs: direct_addrs.clone(),
                    })
                } else {
                    None
                }
            })
            .collect();

        if tunnel_hints.is_empty() {
            return;
        }

        if let Some(contact) = self.contact_store.get_mut(&identity_hash) {
            // Replace stale tunnel addresses with fresh ones.
            contact
                .addresses
                .retain(|a| !matches!(a, ContactAddress::Tunnel { .. }));
            contact.addresses.extend(tunnel_hints);
            let peer_actions = self.peer_manager.on_event(
                harmony_peers::PeerEvent::ContactChanged { identity_hash },
                &self.contact_store,
            );
            self.translate_peer_actions(peer_actions);
        } else {
            // Auto-create contact for discovered peer.
            let unix_now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let contact = Contact {
                identity_hash,
                display_name: None,
                peering: PeeringPolicy {
                    enabled: true,
                    priority: PeeringPriority::Normal,
                },
                added_at: unix_now,
                last_seen: None,
                notes: None,
                addresses: tunnel_hints,
                replication: None,
            };
            if self.contact_store.add(contact).is_ok() {
                let peer_actions = self.peer_manager.on_event(
                    harmony_peers::PeerEvent::ContactChanged { identity_hash },
                    &self.contact_store,
                );
                self.translate_peer_actions(peer_actions);
            }
        }
    }

    /// Shared guard logic and action emission for both text and token inference requests.
    ///
    /// Checks that a model is loaded and the engine is not busy, then emits
    /// `RuntimeAction::RunInference`. Returns an empty `Vec` — all actions are
    /// pushed into `pending_direct_actions`.
    #[cfg(feature = "inference")]
    fn emit_run_inference(
        &mut self,
        query_id: u64,
        input: crate::inference_types::InferenceInput,
        sampling_params: [u8; 20],
    ) -> Vec<WorkflowAction> {
        // Guard: model must be loaded.
        match (self.inference_model_cid, self.inference_tokenizer_cid) {
            (Some(_), Some(_)) => {}
            _ => {
                let result = harmony_agent::AgentResult {
                    task_id: String::new(),
                    status: harmony_agent::TaskStatus::Rejected,
                    output: None,
                    error: Some("no inference model loaded".into()),
                };
                let payload = harmony_agent::encode_result(&result)
                    .unwrap_or_else(|_| b"no inference model loaded".to_vec());
                self.pending_direct_actions
                    .push(RuntimeAction::SendReply { query_id, payload });
                return Vec::new();
            }
        };

        // Guard: DSD session must not be active.
        if self.dsd_session.is_some() {
            let result = harmony_agent::AgentResult {
                task_id: String::new(),
                status: harmony_agent::TaskStatus::Rejected,
                output: None,
                error: Some("engine busy with DSD session".into()),
            };
            let payload =
                harmony_agent::encode_result(&result).unwrap_or_else(|_| b"engine busy".to_vec());
            self.pending_direct_actions
                .push(RuntimeAction::SendReply { query_id, payload });
            return Vec::new();
        }

        self.inference_request_nonce += 1;
        let task_id = format!(
            "{:016x}-{}",
            self.inference_request_nonce,
            hex::encode(self.local_identity_hash)
        );
        self.pending_direct_actions
            .push(RuntimeAction::RunInference {
                query_id,
                task_id,
                input,
                sampling_params_raw: sampling_params,
            });
        Vec::new()
    }

    /// Route a compute query by parsing its payload and submitting to the workflow engine.
    ///
    /// Returns `WorkflowAction`s to be buffered as pending. For CID-based requests,
    /// also emits `FetchModule` as a direct action (stored in `pending_direct_actions`).
    fn route_compute_query(
        &mut self,
        query_id: u64,
        key_expr: String,
        payload: Vec<u8>,
    ) -> Vec<WorkflowAction> {
        // Check if this is a speculative query (key_expr match, not payload tag).
        #[cfg(feature = "inference")]
        if key_expr == harmony_zenoh::namespace::compute::SPECULATIVE_ACTIVITY {
            self.handle_speculative_query(query_id, payload);
            return Vec::new();
        }
        #[cfg(not(feature = "inference"))]
        let _ = &key_expr; // Suppress unused warning

        let parsed = match self.parse_compute_payload(payload) {
            Some(p) => p,
            None => {
                let mut payload = vec![0x01];
                payload.extend_from_slice(b"malformed compute payload");
                self.pending_direct_actions
                    .push(RuntimeAction::SendReply { query_id, payload });
                return Vec::new();
            }
        };

        match parsed {
            ParsedCompute::Inline { module, input } => {
                // Compute workflow ID and store the query_id mapping.
                // Multiple callers may submit the same module+input; the engine
                // deduplicates but we track all query_ids so every caller gets a reply.
                let module_hash = harmony_crypto::hash::blake3_hash(&module);
                let wf_id = WorkflowId::new(&module_hash, &input);
                self.workflow_to_query
                    .entry(wf_id)
                    .or_default()
                    .push(query_id);
                self.workflow.handle(WorkflowEvent::Submit {
                    module,
                    input,
                    hint: ComputeHint::PreferLocal,
                })
            }
            ParsedCompute::ByCid { module_cid, input } => {
                // Store the pending mapping and emit FetchModule.
                // TODO: When module fetch completes, each pending query is
                // converted to a Submit. If the engine dedup-returns immediately
                // (workflow already complete), the reply is sent right away. But
                // the first Submit's completion action also removes from
                // workflow_to_query, potentially missing later callers. Consider
                // accumulating all query_ids before dispatching any actions.
                let already_pending = self.cid_to_query.contains_key(&module_cid);
                self.cid_to_query
                    .entry(module_cid)
                    .or_default()
                    .push((query_id, input));
                if !already_pending {
                    self.pending_direct_actions
                        .push(RuntimeAction::FetchModule { cid: module_cid });
                }
                Vec::new()
            }
            ParsedCompute::Inference { request } => {
                #[cfg(feature = "inference")]
                {
                    return self.emit_run_inference(
                        query_id,
                        crate::inference_types::InferenceInput::Text(request.prompt),
                        request.sampling_params,
                    );
                }
                #[cfg(not(feature = "inference"))]
                {
                    let _ = &request;
                    let mut payload = vec![0x01];
                    payload.extend_from_slice(b"inference feature not enabled");
                    self.pending_direct_actions
                        .push(RuntimeAction::SendReply { query_id, payload });
                    Vec::new()
                }
            }
            ParsedCompute::TokenInference { request } => {
                #[cfg(feature = "inference")]
                {
                    return self.emit_run_inference(
                        query_id,
                        crate::inference_types::InferenceInput::TokenIds(request.token_ids),
                        request.sampling_params,
                    );
                }
                #[cfg(not(feature = "inference"))]
                {
                    let _ = &request;
                    let mut payload = vec![0x01];
                    payload.extend_from_slice(b"inference feature not enabled");
                    self.pending_direct_actions
                        .push(RuntimeAction::SendReply { query_id, payload });
                    Vec::new()
                }
            }
            ParsedCompute::Verify { request } => {
                // DSD target verification — run natively, NOT through WASM.
                // Reject if a DSD session is active — the engine is shared and
                // concurrent verification could interfere with the draft session.
                #[cfg(feature = "inference")]
                {
                    if self.inference_running {
                        let payload = harmony_speculative::VerifyResponse::serialize_error(
                            "engine busy running inference",
                        );
                        self.pending_direct_actions
                            .push(RuntimeAction::SendReply { query_id, payload });
                        return Vec::new();
                    }
                    if self.dsd_session.is_some() {
                        let payload = harmony_speculative::VerifyResponse::serialize_error(
                            "busy: DSD session active on this node",
                        );
                        self.pending_direct_actions
                            .push(RuntimeAction::SendReply { query_id, payload });
                        return Vec::new();
                    }
                    if let Some(engine) = &self.verification_engine {
                        let response = Self::run_verification(engine, &request);
                        let payload = match response {
                            Ok(resp) => resp.serialize(),
                            Err(e) => harmony_speculative::VerifyResponse::serialize_error(&e),
                        };
                        self.pending_direct_actions
                            .push(RuntimeAction::SendReply { query_id, payload });
                    } else {
                        let payload = harmony_speculative::VerifyResponse::serialize_error(
                            "no verification engine",
                        );
                        self.pending_direct_actions
                            .push(RuntimeAction::SendReply { query_id, payload });
                    }
                }
                #[cfg(not(feature = "inference"))]
                {
                    let _ = &request;
                    let payload = harmony_speculative::VerifyResponse::serialize_error(
                        "inference feature not enabled",
                    );
                    self.pending_direct_actions
                        .push(RuntimeAction::SendReply { query_id, payload });
                }
                Vec::new()
            }
        }
    }

    /// Translate workflow actions into runtime actions.
    ///
    /// Wire format encoding (`[0x00]` success / `[0x01]` error tags) is handled
    /// here — the workflow engine emits semantic actions (`WorkflowComplete` with
    /// raw output) and the node encodes them for the wire.
    fn dispatch_workflow_actions(
        &mut self,
        workflow_actions: Vec<WorkflowAction>,
        out: &mut Vec<RuntimeAction>,
    ) {
        // Use an iterative work queue instead of recursion to avoid
        // unbounded stack depth when LoadModel produces further actions.
        let mut queue = std::collections::VecDeque::from(workflow_actions);
        while let Some(action) = queue.pop_front() {
            match action {
                WorkflowAction::WorkflowComplete {
                    workflow_id,
                    output,
                } => {
                    if let Some(query_ids) = self.workflow_to_query.remove(&workflow_id) {
                        let mut payload = vec![0x00];
                        payload.extend_from_slice(&output);
                        for query_id in query_ids {
                            out.push(RuntimeAction::SendReply {
                                query_id,
                                payload: payload.clone(),
                            });
                        }
                    }
                }
                WorkflowAction::WorkflowFailed { workflow_id, error } => {
                    if let Some(query_ids) = self.workflow_to_query.remove(&workflow_id) {
                        let mut payload = vec![0x01];
                        payload.extend_from_slice(error.as_bytes());
                        for query_id in query_ids {
                            out.push(RuntimeAction::SendReply {
                                query_id,
                                payload: payload.clone(),
                            });
                        }
                    }
                }
                WorkflowAction::FetchContent { cid, .. } => {
                    out.push(RuntimeAction::FetchContent { cid });
                }
                WorkflowAction::FetchModule { cid } => {
                    out.push(RuntimeAction::FetchModule { cid });
                }
                WorkflowAction::LoadModel {
                    workflow_id: _workflow_id,
                    gguf_cid,
                    tokenizer_cid,
                } => {
                    // Consume raw model data via take() — the persisted engine
                    // in WasmiRuntime handles all subsequent requests, so this
                    // data is never needed again. Avoids retaining multi-GB.
                    if self.inference_gguf_data.is_some()
                        && self.inference_tokenizer_data.is_some()
                        && Some(gguf_cid) == self.inference_model_cid
                        && Some(tokenizer_cid) == self.inference_tokenizer_cid
                    {
                        let model_event = WorkflowEvent::ModelLoaded {
                            gguf_cid,
                            tokenizer_cid,
                            gguf_data: self.inference_gguf_data.take().unwrap(),
                            tokenizer_data: self.inference_tokenizer_data.take().unwrap(),
                        };
                        let new_actions = self.workflow.handle(model_event);
                        queue.extend(new_actions);
                        continue;
                    }
                    // Model not available — fail the workflow.
                    let fail_event = WorkflowEvent::ModelLoadFailed {
                        gguf_cid,
                        tokenizer_cid,
                    };
                    let new_actions = self.workflow.handle(fail_event);
                    queue.extend(new_actions);
                }
                WorkflowAction::PersistHistory {
                    workflow_id: _workflow_id,
                } => {
                    // TODO: Persist history (e.g., via Zenoh PUT or write-ahead log)
                    // BEFORE compacting. compact_workflow() clears replay_cache,
                    // saved_session, and module_bytes — after compaction, crash
                    // recovery via deterministic replay is impossible.
                    //
                    // Compaction is intentionally deferred until persistence is
                    // implemented. Until then, workflows retain their full
                    // in-memory state at the cost of higher memory usage.
                }
            }
        }
    }

    /// Take the inference engine for an async streaming task.
    /// Returns `None` if the engine is already in use or not loaded.
    #[cfg(feature = "inference")]
    pub fn take_inference_engine(&mut self) -> Option<harmony_inference::QwenEngine> {
        let engine = self.verification_engine.take();
        if engine.is_some() {
            self.inference_running = true;
        }
        engine
    }

    /// Return the inference engine after an async streaming task completes.
    #[cfg(feature = "inference")]
    pub fn return_inference_engine(&mut self, engine: harmony_inference::QwenEngine) {
        self.verification_engine = Some(engine);
        self.inference_running = false;
    }

    /// Reset inference state after a panic destroyed the engine.
    ///
    /// Clears `inference_running` so future requests aren't permanently rejected.
    /// The engine is gone — `check_inference_model_ready` will recreate it from
    /// cached GGUF/tokenizer data on the next tick if they're still available.
    #[cfg(feature = "inference")]
    pub fn reset_inference_after_panic(&mut self) {
        self.inference_running = false;
        // verification_engine is already None (it was moved into the panicked task).
        // Force re-creation by clearing the queryable so check_inference_model_ready
        // will re-run.
        self.inference_queryable_id = None;
    }

    /// Check if both GGUF and tokenizer data have arrived, and if so,
    /// declare the inference queryable and publish capacity.
    fn check_inference_model_ready(&mut self) {
        if self.inference_queryable_id.is_some() {
            return; // Already declared
        }
        if self.inference_gguf_data.is_some() && self.inference_tokenizer_data.is_some() {
            // Build the DSD verification engine from the same model data.
            #[cfg(feature = "inference")]
            if self.verification_engine.is_none() {
                if let (Some(gguf_data), Some(tok_data)) = (
                    self.inference_gguf_data.as_ref(),
                    self.inference_tokenizer_data.as_ref(),
                ) {
                    use harmony_inference::InferenceEngine;
                    let mut engine = harmony_inference::QwenEngine::new(candle_core::Device::Cpu);
                    match engine.load_gguf(gguf_data) {
                        Ok(()) => match engine.load_tokenizer(tok_data) {
                            Ok(()) => {
                                self.verification_engine = Some(engine);
                                tracing::info!("DSD verification engine loaded");
                            }
                            Err(e) => {
                                tracing::error!(
                                    "failed to load tokenizer for verification engine: {e}"
                                );
                            }
                        },
                        Err(e) => {
                            tracing::error!("failed to load GGUF for verification engine: {e}");
                        }
                    }
                }
            }

            // Both pieces arrived — declare the inference queryable.
            let key_expr = harmony_zenoh::namespace::compute::INFERENCE_ACTIVITY;
            match self.queryable_router.declare(key_expr) {
                Ok((qid, actions)) => {
                    self.inference_queryable_id = Some(qid);
                    self.compute_queryable_ids.insert(qid);
                    for action in actions {
                        if let QueryableAction::SendQueryableDeclare { key_expr } = action {
                            self.pending_direct_actions
                                .push(RuntimeAction::DeclareQueryable { key_expr });
                        }
                    }
                    // Publish capacity advertisement.
                    if let Some(model_cid) = &self.inference_model_cid {
                        let payload = crate::inference_types::build_capacity_payload(model_cid, true);
                        let cap_key =
                            harmony_zenoh::namespace::compute::capacity_key(&self.node_addr);
                        self.pending_direct_actions.push(RuntimeAction::Publish {
                            key_expr: cap_key,
                            payload,
                        });
                    }
                    tracing::info!("inference queryable declared on {}", key_expr);
                }
                Err(e) => {
                    tracing::error!("failed to declare inference queryable: {e}");
                }
            }

            // Also declare the verify queryable for DSD target verification.
            // Use a per-node key so edges can route queries to a specific target.
            #[cfg(feature = "inference")]
            if self.verify_queryable_id.is_none() {
                let verify_key = harmony_zenoh::namespace::compute::verify_key(&self.node_addr);
                match self.queryable_router.declare(&verify_key) {
                    Ok((qid, actions)) => {
                        self.verify_queryable_id = Some(qid);
                        self.compute_queryable_ids.insert(qid);
                        for action in actions {
                            if let QueryableAction::SendQueryableDeclare { key_expr } = action {
                                self.pending_direct_actions
                                    .push(RuntimeAction::DeclareQueryable { key_expr });
                            }
                        }
                        tracing::info!("DSD verify queryable declared on {}", verify_key);
                    }
                    Err(e) => {
                        tracing::error!("failed to declare verify queryable: {e}");
                    }
                }
            }

            // Check if speculative queryable should be declared now that the engine is ready.
            #[cfg(feature = "inference")]
            self.check_speculative_ready();
        }
    }

    /// Run DSD sequential verification using the target's inference engine.
    ///
    /// Algorithm:
    /// 1. Create a fresh inference cache for this verification pass
    /// 2. `engine.forward(&context_tokens, &mut cache)` — prefill, returns logits for verifying draft[0]
    /// 3. For each draft token i:
    ///    - Check if draft[i] is acceptable given current logits
    ///    - If rejected: sample bonus token from current distribution, return
    ///    - If accepted: `engine.forward(&[draft_token[i]], &mut cache)` to advance KV cache
    /// 4. If all accepted: sample bonus token from last logits (the "free" extra token)
    /// 5. Return `VerifyResponse` with `accepted_count` + `bonus_token` + `bonus_logprob`
    #[cfg(feature = "inference")]
    fn run_verification(
        engine: &harmony_inference::QwenEngine,
        request: &harmony_speculative::VerifyRequest,
    ) -> Result<harmony_speculative::VerifyResponse, String> {
        use harmony_inference::InferenceEngine;
        use harmony_speculative::verify::{sample_greedy_with_logprob, should_accept_draft};

        if request.drafts.is_empty() {
            return Err("empty drafts".into());
        }
        debug_assert!(
            request.drafts.len() <= u8::MAX as usize,
            "draft count {} exceeds u8::MAX",
            request.drafts.len()
        );

        // 1. Create fresh cache
        let mut cache = engine.new_cache().map_err(|e| e.to_string())?;

        // 2. Prefill with context tokens — returns logits for verifying draft[0]
        let mut current_logits = if request.context_tokens.is_empty() {
            return Err("empty context".into());
        } else {
            let logits = engine
                .forward(&request.context_tokens, &mut cache)
                .map_err(|e| format!("prefill failed: {e}"))?;
            if logits.is_empty() {
                return Err("prefill returned empty logits".into());
            }
            logits
        };

        // 3. Verify each draft token sequentially
        for (i, draft) in request.drafts.iter().enumerate() {
            // Check if this draft token is acceptable given current_logits
            if !should_accept_draft(&current_logits, draft.token_id, draft.logprob) {
                // Rejected — sample bonus token from current distribution
                let (bonus_token, bonus_logprob) = sample_greedy_with_logprob(&current_logits);
                return Ok(harmony_speculative::VerifyResponse {
                    accepted_count: i as u8,
                    bonus_token,
                    bonus_logprob,
                });
            }

            // Accepted — feed this token to advance KV cache
            current_logits = engine
                .forward(&[draft.token_id], &mut cache)
                .map_err(|e| format!("forward failed at draft {i}: {e}"))?;
            if current_logits.is_empty() {
                return Err(format!("forward returned empty logits at draft {i}"));
            }
        }

        // 4. All accepted — sample bonus token from last logits (the "free" extra token)
        let (bonus_token, bonus_logprob) = sample_greedy_with_logprob(&current_logits);

        Ok(harmony_speculative::VerifyResponse {
            accepted_count: request.drafts.len() as u8,
            bonus_token,
            bonus_logprob,
        })
    }

    /// Declare the speculative queryable if we have both a draft model and a target node.
    #[cfg(feature = "inference")]
    fn check_speculative_ready(&mut self) {
        if self.speculative_queryable_id.is_some() {
            return; // Already declared
        }
        if self.verification_engine.is_none() || self.dsd_target_addr.is_none() {
            return; // Not ready yet
        }
        let key_expr = harmony_zenoh::namespace::compute::SPECULATIVE_ACTIVITY;
        match self.queryable_router.declare(key_expr) {
            Ok((qid, actions)) => {
                self.speculative_queryable_id = Some(qid);
                self.compute_queryable_ids.insert(qid);
                for action in actions {
                    if let QueryableAction::SendQueryableDeclare { key_expr } = action {
                        self.pending_direct_actions
                            .push(RuntimeAction::DeclareQueryable { key_expr });
                    }
                }
                tracing::info!("DSD speculative queryable declared on {}", key_expr);
            }
            Err(e) => {
                tracing::error!("failed to declare speculative queryable: {e}");
            }
        }
    }

    /// Handle a speculative inference query (DSD edge side).
    ///
    /// Parses the text prompt, tokenizes, drafts gamma tokens, and sends
    /// a verify request to the target node. Stores DSD session state.
    #[cfg(feature = "inference")]
    fn handle_speculative_query(&mut self, query_id: u64, payload: Vec<u8>) {
        use harmony_inference::InferenceEngine;

        // Reject if already in a DSD session (one at a time).
        if self.dsd_session.is_some() {
            let err =
                harmony_speculative::VerifyResponse::serialize_error("busy: DSD session active");
            self.pending_direct_actions.push(RuntimeAction::SendReply {
                query_id,
                payload: err,
            });
            return;
        }

        // Parse the prompt (reuse inference request format: 0x02 tag).
        let request = match crate::inference_types::InferenceRequest::parse(&payload) {
            Ok(r) => r,
            Err(e) => {
                let err = harmony_speculative::VerifyResponse::serialize_error(&e);
                self.pending_direct_actions.push(RuntimeAction::SendReply {
                    query_id,
                    payload: err,
                });
                return;
            }
        };

        if self.inference_running {
            let err = harmony_speculative::VerifyResponse::serialize_error(
                "engine busy running inference",
            );
            self.pending_direct_actions.push(RuntimeAction::SendReply {
                query_id,
                payload: err,
            });
            return;
        }

        let engine = match &self.verification_engine {
            Some(e) => e,
            None => {
                let err =
                    harmony_speculative::VerifyResponse::serialize_error("no draft model loaded");
                self.pending_direct_actions.push(RuntimeAction::SendReply {
                    query_id,
                    payload: err,
                });
                return;
            }
        };

        let target_addr = match &self.dsd_target_addr {
            Some(a) => a.clone(),
            None => {
                let err = harmony_speculative::VerifyResponse::serialize_error(
                    "no DSD target discovered",
                );
                self.pending_direct_actions.push(RuntimeAction::SendReply {
                    query_id,
                    payload: err,
                });
                return;
            }
        };

        // Tokenize the prompt.
        let prompt_tokens = match engine.tokenize(&request.prompt) {
            Ok(tokens) => tokens,
            Err(e) => {
                let err = harmony_speculative::VerifyResponse::serialize_error(&format!(
                    "tokenize failed: {e}"
                ));
                self.pending_direct_actions.push(RuntimeAction::SendReply {
                    query_id,
                    payload: err,
                });
                return;
            }
        };

        // Get EOS token ID from tokenizer.
        let eos_token_id = engine.eos_token_id();
        if eos_token_id.is_none() {
            tracing::warn!("tokenizer has no EOS token; DSD session will run to max_tokens");
        }
        let prompt_len = prompt_tokens.len();

        // Create fresh cache and prefill draft model.
        let mut cache = match engine.new_cache() {
            Ok(c) => c,
            Err(e) => {
                let err = harmony_speculative::VerifyResponse::serialize_error(&format!(
                    "cache creation failed: {e}"
                ));
                self.pending_direct_actions.push(RuntimeAction::SendReply {
                    query_id,
                    payload: err,
                });
                return;
            }
        };
        let mut last_logits = match engine.forward(&prompt_tokens, &mut cache) {
            Ok(l) if !l.is_empty() => l,
            Ok(_) => {
                let err = harmony_speculative::VerifyResponse::serialize_error(
                    "prefill returned empty logits",
                );
                self.pending_direct_actions.push(RuntimeAction::SendReply {
                    query_id,
                    payload: err,
                });
                return;
            }
            Err(e) => {
                let err = harmony_speculative::VerifyResponse::serialize_error(&format!(
                    "prefill failed: {e}"
                ));
                self.pending_direct_actions.push(RuntimeAction::SendReply {
                    query_id,
                    payload: err,
                });
                return;
            }
        };

        // Draft gamma tokens.
        let gamma = harmony_speculative::DEFAULT_DRAFT_GAMMA as usize;
        let mut drafts = Vec::with_capacity(gamma);
        for _ in 0..gamma {
            let (token_id, logprob) =
                harmony_speculative::verify::sample_greedy_with_logprob(&last_logits);
            drafts.push(harmony_speculative::DraftEntry { token_id, logprob });

            // Check for EOS.
            if eos_token_id == Some(token_id) {
                break;
            }

            // Advance KV cache.
            match engine.forward(&[token_id], &mut cache) {
                Ok(l) if !l.is_empty() => last_logits = l,
                Ok(_) => {
                    let err = harmony_speculative::VerifyResponse::serialize_error(
                        "draft forward returned empty logits",
                    );
                    self.pending_direct_actions.push(RuntimeAction::SendReply {
                        query_id,
                        payload: err,
                    });
                    return;
                }
                Err(e) => {
                    let err = harmony_speculative::VerifyResponse::serialize_error(&format!(
                        "draft forward failed: {e}"
                    ));
                    self.pending_direct_actions.push(RuntimeAction::SendReply {
                        query_id,
                        payload: err,
                    });
                    return;
                }
            }
        }

        // Build and send verify request.
        let draft_token_ids: Vec<u32> = drafts.iter().map(|d| d.token_id).collect();
        let verify_request = harmony_speculative::VerifyRequest {
            context_tokens: prompt_tokens.clone(),
            drafts,
        };
        let verify_payload = verify_request.serialize();
        let target_key = harmony_zenoh::namespace::compute::verify_key(&target_addr);

        self.pending_direct_actions
            .push(RuntimeAction::SendVerifyQuery {
                key_expr: target_key.clone(),
                payload: verify_payload,
            });

        // Store session state and start the timeout timer.
        self.dsd_session = Some(DsdSession {
            query_id,
            accepted_tokens: prompt_tokens,
            target_verify_key: target_key,
            max_tokens: crate::inference_types::DEFAULT_MAX_INFERENCE_TOKENS,
            eos_token_id,
            pending_drafts: draft_token_ids,
            prompt_len,
        });
        self.dsd_session_last_activity_tick = self.tick_count;
    }

    /// Handle a verify response from the target node (DSD edge side).
    ///
    /// Appends accepted tokens + bonus, checks for completion, or drafts more.
    #[cfg(feature = "inference")]
    fn handle_verify_response(&mut self, payload: Vec<u8>) {
        use harmony_inference::InferenceEngine;

        let session = match &mut self.dsd_session {
            Some(s) => s,
            None => {
                tracing::warn!("received VerifyResponse but no active DSD session");
                return;
            }
        };

        // Reset timeout timer — the target is alive and responding.
        self.dsd_session_last_activity_tick = self.tick_count;

        let response = match harmony_speculative::VerifyResponse::parse(&payload) {
            Ok(r) => r,
            Err(e) => {
                // Verification failed — reply with error to original requester.
                let query_id = session.query_id;
                let err_payload = harmony_speculative::VerifyResponse::serialize_error(&e);
                self.pending_direct_actions.push(RuntimeAction::SendReply {
                    query_id,
                    payload: err_payload,
                });
                self.dsd_session = None;
                return;
            }
        };

        let query_id = session.query_id;
        let eos_token_id = session.eos_token_id;
        let max_tokens = session.max_tokens as usize;

        // Append accepted draft tokens.
        let accepted = response.accepted_count as usize;
        let pending = session.pending_drafts.clone();
        if accepted > pending.len() {
            let err_payload = harmony_speculative::VerifyResponse::serialize_error(&format!(
                "accepted_count {} exceeds pending drafts {}",
                accepted,
                pending.len()
            ));
            self.pending_direct_actions.push(RuntimeAction::SendReply {
                query_id,
                payload: err_payload,
            });
            self.dsd_session = None;
            return;
        }
        // Append accepted draft tokens, stopping early if EOS is encountered.
        let mut hit_eos = false;
        for &token_id in pending.iter().take(accepted) {
            if eos_token_id == Some(token_id) {
                hit_eos = true;
                break;
            }
            session.accepted_tokens.push(token_id);
        }
        // Append bonus token only if we haven't already hit EOS.
        if !hit_eos {
            if eos_token_id == Some(response.bonus_token) {
                hit_eos = true;
            } else {
                session.accepted_tokens.push(response.bonus_token);
            }
        }

        // Count generated tokens (total - prompt length).
        let prompt_len = session.prompt_len;
        let generated = session.accepted_tokens.len() - prompt_len;

        if hit_eos || generated >= max_tokens {
            // Done — detokenize and reply.
            let engine = match &self.verification_engine {
                Some(e) => e,
                None => {
                    let err = harmony_speculative::VerifyResponse::serialize_error("engine lost");
                    self.pending_direct_actions.push(RuntimeAction::SendReply {
                        query_id,
                        payload: err,
                    });
                    self.dsd_session = None;
                    return;
                }
            };

            // Detokenize only the generated tokens (after prompt).
            let session = self.dsd_session.as_ref().unwrap();
            let generated_tokens = &session.accepted_tokens[prompt_len..];
            match engine.detokenize(generated_tokens) {
                Ok(text) => {
                    let mut payload = vec![0x00]; // success tag
                    payload.extend_from_slice(text.as_bytes());
                    self.pending_direct_actions
                        .push(RuntimeAction::SendReply { query_id, payload });
                }
                Err(e) => {
                    let err = harmony_speculative::VerifyResponse::serialize_error(&format!(
                        "detokenize failed: {e}"
                    ));
                    self.pending_direct_actions.push(RuntimeAction::SendReply {
                        query_id,
                        payload: err,
                    });
                }
            }
            self.dsd_session = None;
            return;
        }

        // Not done — draft more tokens and send another verify request.
        let engine = match &self.verification_engine {
            Some(e) => e,
            None => {
                let err = harmony_speculative::VerifyResponse::serialize_error("engine lost");
                self.pending_direct_actions.push(RuntimeAction::SendReply {
                    query_id,
                    payload: err,
                });
                self.dsd_session = None;
                return;
            }
        };

        // Create fresh cache and prefill with full accepted sequence.
        let mut cache = match engine.new_cache() {
            Ok(c) => c,
            Err(e) => {
                let err = harmony_speculative::VerifyResponse::serialize_error(&format!(
                    "cache creation failed: {e}"
                ));
                self.pending_direct_actions.push(RuntimeAction::SendReply {
                    query_id,
                    payload: err,
                });
                self.dsd_session = None;
                return;
            }
        };
        let session = self.dsd_session.as_ref().unwrap();
        let full_sequence = session.accepted_tokens.clone();
        let mut last_logits = match engine.forward(&full_sequence, &mut cache) {
            Ok(l) if !l.is_empty() => l,
            Ok(_) => {
                let err = harmony_speculative::VerifyResponse::serialize_error(
                    "reprefill returned empty logits",
                );
                self.pending_direct_actions.push(RuntimeAction::SendReply {
                    query_id,
                    payload: err,
                });
                self.dsd_session = None;
                return;
            }
            Err(e) => {
                let err = harmony_speculative::VerifyResponse::serialize_error(&format!(
                    "reprefill failed: {e}"
                ));
                self.pending_direct_actions.push(RuntimeAction::SendReply {
                    query_id,
                    payload: err,
                });
                self.dsd_session = None;
                return;
            }
        };

        // Draft gamma more tokens.
        let gamma = harmony_speculative::DEFAULT_DRAFT_GAMMA as usize;
        let remaining = max_tokens - generated;
        let draft_count = gamma.min(remaining);
        let mut drafts = Vec::with_capacity(draft_count);
        let mut draft_token_ids = Vec::with_capacity(draft_count);

        for _ in 0..draft_count {
            let (token_id, logprob) =
                harmony_speculative::verify::sample_greedy_with_logprob(&last_logits);
            drafts.push(harmony_speculative::DraftEntry { token_id, logprob });
            draft_token_ids.push(token_id);

            if eos_token_id == Some(token_id) {
                break;
            }

            match engine.forward(&[token_id], &mut cache) {
                Ok(l) if !l.is_empty() => last_logits = l,
                Ok(_) => {
                    let err = harmony_speculative::VerifyResponse::serialize_error(
                        "draft forward returned empty logits",
                    );
                    self.pending_direct_actions.push(RuntimeAction::SendReply {
                        query_id,
                        payload: err,
                    });
                    self.dsd_session = None;
                    return;
                }
                Err(e) => {
                    let err = harmony_speculative::VerifyResponse::serialize_error(&format!(
                        "draft forward failed: {e}"
                    ));
                    self.pending_direct_actions.push(RuntimeAction::SendReply {
                        query_id,
                        payload: err,
                    });
                    self.dsd_session = None;
                    return;
                }
            }
        }

        // Send verify request.
        let session = self.dsd_session.as_ref().unwrap();
        let verify_request = harmony_speculative::VerifyRequest {
            context_tokens: session.accepted_tokens.clone(),
            drafts,
        };
        let target_key = session.target_verify_key.clone();
        self.pending_direct_actions
            .push(RuntimeAction::SendVerifyQuery {
                key_expr: target_key,
                payload: verify_request.serialize(),
            });

        // Update session with new pending drafts.
        if let Some(session) = &mut self.dsd_session {
            session.pending_drafts = draft_token_ids;
        }
    }

    fn parse_compute_payload(&self, payload: Vec<u8>) -> Option<ParsedCompute> {
        if payload.is_empty() {
            return None;
        }
        match payload[0] {
            // Inline: [0x00] [module_len: u32 LE] [module_bytes] [input_bytes]
            0x00 => {
                if payload.len() < 5 {
                    return None;
                }
                let module_len = u32::from_le_bytes(payload[1..5].try_into().ok()?) as usize;
                const MAX_MODULE_BYTES: usize = 10 * 1024 * 1024; // 10 MB
                if module_len > MAX_MODULE_BYTES || payload.len() < 5 + module_len {
                    return None;
                }
                let module = payload[5..5 + module_len].to_vec();
                let input = payload[5 + module_len..].to_vec();
                Some(ParsedCompute::Inline { module, input })
            }
            // CID reference: [0x01] [cid: 32 bytes] [input_bytes]
            0x01 => {
                if payload.len() < 33 {
                    return None;
                }
                let cid: [u8; 32] = payload[1..33].try_into().ok()?;
                let input = payload[33..].to_vec();
                Some(ParsedCompute::ByCid {
                    module_cid: cid,
                    input,
                })
            }
            // Inference: [0x02] [prompt_len: u32 LE] [prompt_utf8] [sampling_params: 20 bytes (optional)]
            0x02 => match crate::inference_types::InferenceRequest::parse(&payload) {
                Ok(request) => Some(ParsedCompute::Inference { request }),
                Err(_) => None,
            },
            // Token inference: [0x03] [token_count: u32 LE] [token_ids: u32 LE × count] [sampling_params: 20 bytes (optional)]
            crate::inference_types::TOKEN_INFERENCE_TAG => {
                match crate::inference_types::TokenInferenceRequest::parse(&payload) {
                    Ok(request) => Some(ParsedCompute::TokenInference { request }),
                    Err(_) => None,
                }
            }
            // Verify: [0x04] [context_len: u32 LE] [context_tokens...] [draft_count: u8] [draft_entries...]
            0x04 => match harmony_speculative::VerifyRequest::parse(&payload) {
                Ok(request) => Some(ParsedCompute::Verify { request }),
                Err(_) => None,
            },
            _ => None,
        }
    }

    /// Translate PeerActions into RuntimeActions, buffering into pending_direct_actions.
    /// Used from push_event where no output vec is available.
    fn translate_peer_actions(&mut self, peer_actions: Vec<PeerAction>) {
        let mut out = std::mem::take(&mut self.pending_direct_actions);
        self.translate_peer_actions_out(peer_actions, &mut out);
        self.pending_direct_actions = out;
    }

    /// Translate PeerActions into RuntimeActions, pushing directly to an output vec.
    /// Used from tick() and dispatch_router_actions where an output vec is available.
    fn translate_peer_actions_out(
        &mut self,
        peer_actions: Vec<PeerAction>,
        out: &mut Vec<RuntimeAction>,
    ) {
        for action in peer_actions {
            match action {
                PeerAction::InitiateTunnel {
                    identity_hash,
                    node_id,
                    relay_url,
                } => {
                    let (peer_dsa_pubkey, peer_kem_pubkey) = match self
                        .discovery
                        .get_record(&identity_hash, self.last_unix_now)
                    {
                        Some(record) => (record.public_key.clone(), record.encryption_key.clone()),
                        None => {
                            tracing::warn!(
                                identity = %hex::encode(identity_hash),
                                "no announce record for tunnel peer — dial will fail at handshake"
                            );
                            (Vec::new(), Vec::new())
                        }
                    };
                    out.push(RuntimeAction::InitiateTunnel {
                        identity_hash,
                        node_id,
                        relay_url,
                        peer_dsa_pubkey,
                        peer_kem_pubkey,
                    });
                }
                PeerAction::SendPathRequest { identity_hash } => {
                    out.push(RuntimeAction::SendPathRequest { identity_hash });
                }
                PeerAction::UpdateLastSeen {
                    identity_hash,
                    timestamp: _, // PeerManager provides monotonic seconds; ignore it
                } => {
                    // Contact.last_seen uses Unix epoch seconds (like added_at).
                    // PeerManager operates on monotonic timestamps internally,
                    // so we use wall-clock time here at the boundary.
                    let unix_now = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_secs())
                        .unwrap_or(0);
                    self.contact_store
                        .update_last_seen(&identity_hash, unix_now);
                }
                PeerAction::InitiateLink { .. } | PeerAction::CloseLink { .. } => {
                    // Reticulum link initiation/close — stub for now.
                }
                PeerAction::CloseTunnel { identity_hash } => {
                    out.push(RuntimeAction::CloseTunnel { identity_hash });
                }
            }
        }
    }

    fn route_query(&mut self, query_id: u64, key_expr: String, payload: Vec<u8>) {
        let event = QueryableEvent::QueryReceived {
            query_id,
            key_expr: key_expr.clone(),
            payload,
        };
        let Ok(actions) = self.queryable_router.handle_event(event) else {
            return;
        };
        for action in actions {
            if let QueryableAction::DeliverQuery {
                queryable_id,
                query_id,
                key_expr,
                payload,
            } = action
            {
                if self.storage_queryable_ids.contains(&queryable_id) {
                    if let Some(event) = self.parse_storage_query(query_id, &key_expr) {
                        self.storage_queue.push_back(event);
                    }
                } else if self.compute_queryable_ids.contains(&queryable_id) {
                    let workflow_actions = self.route_compute_query(query_id, key_expr, payload);
                    self.pending_workflow_actions.extend(workflow_actions);
                } else if queryable_id == self.page_queryable_id {
                    let page_actions = self.handle_page_query(query_id, &key_expr);
                    self.pending_direct_actions.extend(page_actions);
                } else if queryable_id == self.memo_queryable_id {
                    let memo_actions = self.handle_memo_query(query_id, &key_expr);
                    self.pending_direct_actions.extend(memo_actions);
                } else if queryable_id == self.discover_queryable_id {
                    let actions = self.handle_discover_query(query_id, &key_expr, &payload);
                    self.pending_direct_actions.extend(actions);
                }
            }
        }
    }

    /// Handle a memo query: look up MemoStore by input CID from key expression.
    ///
    /// Key expression: `harmony/memo/{input_hex}/**`
    /// Response: `[u16 LE count][u32 LE len][memo_bytes]...`
    /// Returns empty vec (no reply) if no memos found or key malformed.
    fn handle_memo_query(&self, query_id: u64, key_expr: &str) -> Vec<RuntimeAction> {
        // Strip prefix and trailing wildcard to extract input_hex.
        let input_hex = match key_expr
            .strip_prefix(harmony_zenoh::namespace::memo::PREFIX)
            .and_then(|s| s.strip_prefix('/'))
        {
            Some(rest) => {
                // rest is e.g. "aabb.../**" — strip trailing /**
                rest.split('/').next().unwrap_or("")
            }
            None => return Vec::new(),
        };

        // Decode hex to ContentId
        let cid_bytes = match hex::decode(input_hex) {
            Ok(b) if b.len() == 32 => b,
            _ => return Vec::new(),
        };
        let mut arr = [0u8; 32];
        arr.copy_from_slice(&cid_bytes);
        let input_cid = ContentId::from_bytes(arr);

        let memos = self.memo_store.get_by_input(&input_cid);
        if memos.is_empty() {
            return Vec::new();
        }

        // Serialize up to MAX_MEMO_RESPONSE_COUNT memos into a temp buffer,
        // then write the actual count. This avoids a count mismatch if any
        // serialize() call fails.
        let mut memo_buf = Vec::new();
        let mut actual_count: u16 = 0;

        for memo in memos.iter().take(MAX_MEMO_RESPONSE_COUNT) {
            if let Ok(bytes) = harmony_memo::serialize(memo) {
                let entry_len = 4 + bytes.len();
                // Stop before exceeding the byte limit (leave room for the 2-byte count header).
                if 2 + memo_buf.len() + entry_len > MAX_MEMO_RESPONSE_BYTES {
                    break;
                }
                memo_buf.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
                memo_buf.extend_from_slice(&bytes);
                actual_count += 1;
            }
        }

        if actual_count == 0 {
            return Vec::new();
        }

        let mut payload = Vec::with_capacity(2 + memo_buf.len());
        payload.extend_from_slice(&actual_count.to_le_bytes());
        payload.extend_from_slice(&memo_buf);

        vec![RuntimeAction::SendReply { query_id, payload }]
    }

    /// Handle a discover query: validate Discovery UCAN token, respond with
    /// full or public announce record.
    ///
    /// Key expression: `harmony/discover/{identity_hash_hex}`
    /// Payload: serialized PqUcanToken (or empty for public hints)
    /// Response: serialized AnnounceRecord bytes
    fn handle_discover_query(
        &self,
        query_id: u64,
        key_expr: &str,
        payload: &[u8],
    ) -> Vec<RuntimeAction> {
        // 1. Parse identity_hash from key expression
        let identity_hex = match key_expr
            .strip_prefix(harmony_zenoh::namespace::discover::PREFIX)
            .and_then(|s| s.strip_prefix('/'))
        {
            Some(hex) => hex.split('/').next().unwrap_or(""),
            None => return Vec::new(),
        };

        let id_bytes = match hex::decode(identity_hex) {
            Ok(b) if b.len() == 16 => b,
            _ => return Vec::new(),
        };
        let mut queried_hash = [0u8; 16];
        queried_hash.copy_from_slice(&id_bytes);

        // 2. Check if this is our PQ identity (Discovery tokens use PQ address hash)
        let local_hash = self.local_pq_identity_hash;
        if queried_hash != local_hash {
            return Vec::new(); // Not our identity — no reply
        }

        // 3. If no records available, no reply
        let public_bytes = match &self.local_public_announce {
            Some(b) => b,
            None => return Vec::new(),
        };

        // 4. Empty payload → public hints
        if payload.is_empty() {
            return vec![RuntimeAction::SendReply {
                query_id,
                payload: public_bytes.clone(),
            }];
        }

        // 5. Parse and validate Discovery UCAN token
        let full_bytes = match &self.local_full_announce {
            Some(b) => b,
            None => {
                return vec![RuntimeAction::SendReply {
                    query_id,
                    payload: public_bytes.clone(),
                }];
            }
        };

        // Size guard
        const MAX_TOKEN_BYTES: usize = 8 * 1024;
        if payload.len() > MAX_TOKEN_BYTES {
            return vec![RuntimeAction::SendReply {
                query_id,
                payload: public_bytes.clone(),
            }];
        }

        let token = match harmony_identity::PqUcanToken::from_bytes(payload) {
            Ok(t) => t,
            Err(_) => {
                return vec![RuntimeAction::SendReply {
                    query_id,
                    payload: public_bytes.clone(),
                }];
            }
        };

        // Check capability
        if token.capability != harmony_identity::CapabilityType::Discovery {
            return vec![RuntimeAction::SendReply {
                query_id,
                payload: public_bytes.clone(),
            }];
        }

        // Check resource matches our identity hash
        if token.resource.len() != 16 || token.resource[..] != local_hash[..] {
            return vec![RuntimeAction::SendReply {
                query_id,
                payload: public_bytes.clone(),
            }];
        }

        // Check expiry (expires_at == 0 means no expiry / indefinite grant).
        if token.expires_at != 0 && self.last_unix_now > token.expires_at {
            return vec![RuntimeAction::SendReply {
                query_id,
                payload: public_bytes.clone(),
            }];
        }

        // Check not-before (not_before == 0 means valid immediately / no activation delay).
        if token.not_before > self.last_unix_now {
            return vec![RuntimeAction::SendReply {
                query_id,
                payload: public_bytes.clone(),
            }];
        }

        // Check issuer matches our identity (token was issued by us).
        // Cheap byte comparison before expensive signature verification.
        if token.issuer != local_hash {
            return vec![RuntimeAction::SendReply {
                query_id,
                payload: public_bytes.clone(),
            }];
        }

        // Verify ML-DSA signature with LOCAL key (we issued this token).
        // The verifying key is parsed once at construction time.
        let pubkey = match &self.local_dsa_verifying_key {
            Some(pk) => pk,
            None => {
                tracing::warn!(
                    "local DSA verifying key not available — discovery token verification skipped"
                );
                return vec![RuntimeAction::SendReply {
                    query_id,
                    payload: public_bytes.clone(),
                }];
            }
        };
        if token.verify_signature(pubkey).is_err() {
            return vec![RuntimeAction::SendReply {
                query_id,
                payload: public_bytes.clone(),
            }];
        }

        // Note: `audience` is intentionally not checked. Zenoh queries carry no
        // authenticated sender identity, so audience enforcement would require an
        // extra round-trip. Discovery tokens operate as bearer credentials —
        // issuers should distribute them only through encrypted channels (e.g.,
        // tunnel-encrypted messages or out-of-band secure transfer).

        // All checks passed — serve full record
        vec![RuntimeAction::SendReply {
            query_id,
            payload: full_bytes.clone(),
        }]
    }

    /// Handle a page query against the page index.
    ///
    /// Key expression: `harmony/page/{addr_00}/{addr_01}/{addr_10}/{addr_11}/{book_cid}/{page_num}`
    /// Segments that are `*` are treated as wildcards (None).
    ///
    /// Response protocol:
    /// - 0 matches → empty vec (no reply)
    /// - 1 match → `[0x01] + 4KB_page_data`
    /// - N matches → one `[0x00] + key_expression_bytes` per match
    fn handle_page_query(&self, query_id: u64, key_expr: &str) -> Vec<RuntimeAction> {
        let after_prefix = match key_expr
            .strip_prefix(page_ns::PREFIX)
            .and_then(|s| s.strip_prefix('/'))
        {
            Some(s) => s,
            None => return Vec::new(),
        };

        let segments: Vec<&str> = after_prefix.split('/').collect();
        if segments.len() != 6 {
            return Vec::new();
        }

        // Parse each segment: "*" → Ok(None) (wildcard), concrete → Ok(Some(val)),
        // malformed → Err (reject entire query).
        let parse_addr = |s: &str| -> Result<Option<harmony_athenaeum::PageAddr>, ()> {
            if s == "*" {
                return Ok(None);
            }
            let bytes = hex::decode(s).map_err(|_| ())?;
            let arr: [u8; 4] = bytes.try_into().map_err(|_| ())?;
            Ok(Some(harmony_athenaeum::PageAddr::from_bytes(arr)))
        };

        let parse_cid = |s: &str| -> Result<Option<ContentId>, ()> {
            if s == "*" {
                return Ok(None);
            }
            let bytes = hex::decode(s).map_err(|_| ())?;
            let arr: [u8; 32] = bytes.try_into().map_err(|_| ())?;
            Ok(Some(ContentId::from_bytes(arr)))
        };

        let parse_page_num = |s: &str| -> Result<Option<u8>, ()> {
            if s == "*" {
                return Ok(None);
            }
            s.parse::<u8>().map(Some).map_err(|_| ())
        };

        let Ok(addr_00) = parse_addr(segments[0]) else {
            return Vec::new();
        };
        let Ok(addr_01) = parse_addr(segments[1]) else {
            return Vec::new();
        };
        let Ok(addr_10) = parse_addr(segments[2]) else {
            return Vec::new();
        };
        let Ok(addr_11) = parse_addr(segments[3]) else {
            return Vec::new();
        };
        let Ok(book_cid) = parse_cid(segments[4]) else {
            return Vec::new();
        };
        let Ok(page_num) = parse_page_num(segments[5]) else {
            return Vec::new();
        };

        let matches = self.page_index.match_query(
            addr_00.as_ref(),
            addr_01.as_ref(),
            addr_10.as_ref(),
            addr_11.as_ref(),
            book_cid.as_ref(),
            page_num,
        );

        if matches.is_empty() {
            return Vec::new();
        }

        if matches.len() == 1 {
            let entry = &matches[0];
            // Fetch the 4KB page data by direct slice — no Book::from_book needed.
            // We only index depth-0 (raw) books, so page data is at a fixed offset.
            const PAGE_SIZE: usize = 4096;
            if let Some(data) = self.storage.get(&entry.book_cid) {
                let start = entry.page_num as usize * PAGE_SIZE;
                if start < data.len() {
                    let end = (start + PAGE_SIZE).min(data.len());
                    let mut payload = Vec::with_capacity(1 + PAGE_SIZE);
                    payload.push(0x01);
                    payload.extend_from_slice(&data[start..end]);
                    // Zero-pad if last page is shorter than PAGE_SIZE
                    if payload.len() < 1 + PAGE_SIZE {
                        payload.resize(1 + PAGE_SIZE, 0);
                    }
                    return vec![RuntimeAction::SendReply { query_id, payload }];
                }
            }
            // Couldn't fetch page data — no reply
            return Vec::new();
        }

        // Multiple matches: return metadata for each.
        let mut actions = Vec::new();
        for entry in &matches {
            let a00 = hex::encode(entry.addrs[0].to_bytes());
            let a01 = hex::encode(entry.addrs[1].to_bytes());
            let a10 = hex::encode(entry.addrs[2].to_bytes());
            let a11 = hex::encode(entry.addrs[3].to_bytes());
            let cid_hex = hex::encode(entry.book_cid.to_bytes());
            let pk = page_ns::page_key(&a00, &a01, &a10, &a11, &cid_hex, entry.page_num);
            let mut payload = vec![0x00];
            payload.extend_from_slice(pk.as_bytes());
            actions.push(RuntimeAction::SendReply { query_id, payload });
        }
        actions
    }

    fn route_subscription(&mut self, key_expr: String, payload: Vec<u8>) {
        // Check if this is a content filter broadcast.
        if let Some(peer_addr) = key_expr
            .strip_prefix(harmony_zenoh::namespace::filters::CONTENT_PREFIX)
            .and_then(|s| s.strip_prefix('/'))
        {
            // Don't process our own filter broadcasts.
            if peer_addr != self.node_addr {
                match BloomFilter::from_bytes(&payload) {
                    Ok(filter) => {
                        self.peer_filters.upsert_content(
                            peer_addr.to_string(),
                            filter,
                            self.tick_count,
                        );
                    }
                    Err(_) => {
                        self.peer_filters.record_parse_error();
                    }
                }
            }
            return;
        }

        // Check if this is a flatpack filter broadcast.
        if let Some(peer_addr) = key_expr
            .strip_prefix(harmony_zenoh::namespace::filters::FLATPACK_PREFIX)
            .and_then(|s| s.strip_prefix('/'))
        {
            if peer_addr != self.node_addr {
                match CuckooFilter::from_bytes(&payload) {
                    Ok(filter) => {
                        self.peer_filters.upsert_flatpack(
                            peer_addr.to_string(),
                            filter,
                            self.tick_count,
                        );
                    }
                    Err(_) => {
                        self.peer_filters.record_parse_error();
                    }
                }
            }
            return;
        }

        // Check if this is a memo filter broadcast.
        if let Some(peer_addr) = key_expr
            .strip_prefix(harmony_zenoh::namespace::filters::MEMO_PREFIX)
            .and_then(|s| s.strip_prefix('/'))
        {
            if peer_addr != self.node_addr {
                match BloomFilter::from_bytes(&payload) {
                    Ok(filter) => {
                        self.peer_filters.upsert_memo(
                            peer_addr.to_string(),
                            filter,
                            self.tick_count,
                        );
                    }
                    Err(_) => {
                        self.peer_filters.record_parse_error();
                    }
                }
            }
            return;
        }

        // Check if this is a page filter broadcast.
        if let Some(peer_addr) = key_expr
            .strip_prefix(harmony_zenoh::namespace::filters::PAGE_PREFIX)
            .and_then(|s| s.strip_prefix('/'))
        {
            if peer_addr != self.node_addr {
                match BloomFilter::from_bytes(&payload) {
                    Ok(filter) => {
                        self.peer_filters.upsert_page(
                            peer_addr.to_string(),
                            filter,
                            self.tick_count,
                        );
                    }
                    Err(_) => {
                        self.peer_filters.record_parse_error();
                    }
                }
            }
            return;
        }

        // Check if this is a capacity advertisement for DSD target discovery.
        // Only consume messages that match; let non-matching fall through so
        // future handlers below can see capacity-prefixed keys if needed.
        #[cfg(feature = "inference")]
        if let Some(peer_addr) = key_expr
            .strip_prefix(harmony_zenoh::namespace::compute::CAPACITY)
            .and_then(|s| s.strip_prefix('/'))
        {
            if peer_addr != self.node_addr && payload.len() >= 33 {
                let mut peer_model_cid = [0u8; 32];
                peer_model_cid.copy_from_slice(&payload[..32]);
                let status = payload[32];
                // A target qualifies if it's ready and its model CID differs from ours.
                if status == crate::inference_types::CAPACITY_READY
                    && self
                        .inference_model_cid
                        .map_or(true, |our_cid| peer_model_cid != our_cid)
                {
                    // Don't overwrite existing target (use first discovered).
                    if self.dsd_target_addr.is_none() {
                        self.dsd_target_addr = Some(peer_addr.to_string());
                        tracing::info!(
                            peer_addr,
                            peer_model = hex::encode(&peer_model_cid[..4]),
                            "DSD target discovered"
                        );
                        self.check_speculative_ready();
                    }
                }
                return;
            }
        }

        if let Some(event) = self.parse_subscription_event(&key_expr, payload) {
            self.storage_queue.push_back(event);
        }
    }

    fn parse_storage_query(&self, query_id: u64, key_expr: &str) -> Option<StorageTierEvent> {
        // Stats query: harmony/content/stats/{node_addr}
        if key_expr.starts_with(content_ns::STATS) {
            return Some(StorageTierEvent::StatsQuery { query_id });
        }
        // Content query: harmony/content/{prefix}/{cid_hex}
        let after_prefix = key_expr
            .strip_prefix(content_ns::PREFIX)?
            .strip_prefix('/')?;
        let (_shard, cid_hex) = after_prefix.split_once('/')?;
        let cid_bytes: [u8; 32] = hex::decode(cid_hex).ok()?.try_into().ok()?;
        let cid = ContentId::from_bytes(cid_bytes);
        Some(StorageTierEvent::ContentQuery { query_id, cid })
    }

    fn parse_subscription_event(
        &self,
        key_expr: &str,
        payload: Vec<u8>,
    ) -> Option<StorageTierEvent> {
        if let Some(cid_hex) = key_expr
            .strip_prefix(content_ns::TRANSIT)
            .and_then(|s| s.strip_prefix('/'))
        {
            let cid_bytes: [u8; 32] = hex::decode(cid_hex).ok()?.try_into().ok()?;
            let cid = ContentId::from_bytes(cid_bytes);
            return Some(StorageTierEvent::TransitContent { cid, data: payload });
        }
        if let Some(cid_hex) = key_expr
            .strip_prefix(content_ns::PUBLISH)
            .and_then(|s| s.strip_prefix('/'))
        {
            let cid_bytes: [u8; 32] = hex::decode(cid_hex).ok()?.try_into().ok()?;
            let cid = ContentId::from_bytes(cid_bytes);
            return Some(StorageTierEvent::PublishContent { cid, data: payload });
        }
        None
    }
}

/// Parsed compute payload variants (internal).
enum ParsedCompute {
    Inline {
        module: Vec<u8>,
        input: Vec<u8>,
    },
    ByCid {
        module_cid: [u8; 32],
        input: Vec<u8>,
    },
    Inference {
        request: crate::inference_types::InferenceRequest,
    },
    TokenInference {
        request: crate::inference_types::TokenInferenceRequest,
    },
    /// DSD verify request: [0x04] payload
    Verify {
        request: harmony_speculative::VerifyRequest,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    const ADD_WAT: &str = r#"
    (module
      (memory (export "memory") 1)
      (func (export "compute") (param $input_ptr i32) (param $input_len i32) (result i32)
        (i32.store
          (i32.add (local.get $input_ptr) (local.get $input_len))
          (i32.add
            (i32.load (local.get $input_ptr))
            (i32.load (i32.add (local.get $input_ptr) (i32.const 4)))))
        (i32.const 4)))
"#;

    const FETCH_WAT: &str = r#"
    (module
      (import "harmony" "fetch_content" (func $fetch (param i32 i32 i32) (result i32)))
      (memory (export "memory") 1)
      (func (export "compute") (param $input_ptr i32) (param $input_len i32) (result i32)
        (local $result i32)
        (local $out_ptr i32)
        (local.set $out_ptr (i32.add (local.get $input_ptr) (local.get $input_len)))
        (local.set $result
          (call $fetch
            (local.get $input_ptr)
            (i32.add (local.get $out_ptr) (i32.const 4))
            (i32.const 1024)))
        (i32.store (local.get $out_ptr) (local.get $result))
        (if (result i32) (i32.gt_s (local.get $result) (i32.const 0))
          (then (i32.add (i32.const 4) (local.get $result)))
          (else (i32.const 4)))))
"#;

    #[test]
    fn runtime_exposes_schedule() {
        use harmony_content::book::MemoryBookStore;

        let mut config = NodeConfig::default();
        config.schedule.router_max_per_tick = Some(5);
        let (rt, _) = NodeRuntime::new(config, MemoryBookStore::new());
        assert_eq!(rt.schedule().router_max_per_tick, Some(5));
    }

    #[test]
    fn runtime_event_variants_exist() {
        let _e1 = RuntimeEvent::InboundPacket {
            interface_name: "lo".into(),
            raw: vec![0u8; 20],
            now: 1000,
        };
        let _e2 = RuntimeEvent::TimerTick {
            now: 1000,
            unix_now: 0,
        };
        let _e3 = RuntimeEvent::QueryReceived {
            query_id: 1,
            key_expr: "harmony/content/a/abc".into(),
            payload: vec![],
        };
        let _e4 = RuntimeEvent::SubscriptionMessage {
            key_expr: "harmony/content/transit/abc".into(),
            payload: vec![1, 2, 3],
        };
        let _e5 = RuntimeEvent::ComputeQuery {
            query_id: 1,
            key_expr: "harmony/compute/activity/test".into(),
            payload: vec![0x00],
        };
        let _e6 = RuntimeEvent::ModuleFetchResponse {
            cid: [0u8; 32],
            result: Ok(vec![1, 2, 3]),
        };
        let _e7 = RuntimeEvent::DiscoveryAnnounceReceived {
            record_bytes: vec![0u8; 10],
            unix_now: 1000,
        };
        let _e8 = RuntimeEvent::LocalTunnelInfo {
            node_id: [0u8; 32],
            relay_url: Some("https://iroh.q8.fyi".into()),
        };
        let _e_mfr = RuntimeEvent::MemoFetchRequest {
            input: ContentId::from_bytes([0u8; 32]),
        };
        let _e_mfp = RuntimeEvent::MemoFetchResponse {
            key_expr: "harmony/memo/aa/**".into(),
            payload: vec![],
            unix_now: 1000,
        };
        let _e_l2 = RuntimeEvent::L2InterfaceReady {
            interface_name: "l2:wlan0".into(),
        };
        let _e_l2c = RuntimeEvent::L2InterfaceClosed {
            interface_name: "l2:wlan0".into(),
        };
        let _e_vr = RuntimeEvent::VerifyResponse {
            payload: vec![0x00, 1, 0, 0, 0, 42, 0, 0, 0, 0],
        };
        let _e_drc = RuntimeEvent::DiskReadComplete {
            cid: ContentId::from_bytes([0u8; 32]),
            query_id: 1,
            data: vec![1, 2, 3],
        };
        let _e_drf = RuntimeEvent::DiskReadFailed {
            cid: ContentId::from_bytes([0u8; 32]),
            query_id: 1,
        };
        let _e_s3rc = RuntimeEvent::S3ReadComplete {
            cid: ContentId::from_bytes([0u8; 32]),
            query_id: 1,
            data: vec![1, 2, 3],
        };
        let _e_s3rf = RuntimeEvent::S3ReadFailed {
            cid: ContentId::from_bytes([0u8; 32]),
            query_id: 1,
        };
    }

    #[test]
    fn runtime_action_variants_exist() {
        let _a1 = RuntimeAction::SendOnInterface {
            interface_name: "lo".into(),
            raw: vec![0u8; 20],
            weight: None,
        };
        let _a2 = RuntimeAction::SendReply {
            query_id: 1,
            payload: vec![1, 2, 3],
        };
        let _a3 = RuntimeAction::Publish {
            key_expr: "harmony/announce/abc".into(),
            payload: vec![],
        };
        let _a4 = RuntimeAction::FetchContent { cid: [0u8; 32] };
        let _a4b = RuntimeAction::FetchModule { cid: [0u8; 32] };
        let _a5 = RuntimeAction::DeclareQueryable {
            key_expr: "harmony/content/a/**".into(),
        };
        let _a6 = RuntimeAction::Subscribe {
            key_expr: "harmony/content/transit/**".into(),
        };
        let _a_qm = RuntimeAction::QueryMemo {
            key_expr: "harmony/memo/aa/**".into(),
        };
        let _a_vq = RuntimeAction::SendVerifyQuery {
            key_expr: "harmony/compute/activity/verify".into(),
            payload: vec![0x04],
        };
        let _a_ptd = RuntimeAction::PersistToDisk {
            cid: ContentId::from_bytes([0u8; 32]),
            data: vec![1, 2, 3],
        };
        let _a_dl = RuntimeAction::DiskLookup {
            cid: ContentId::from_bytes([0u8; 32]),
            query_id: 1,
        };
        let _a_s3 = RuntimeAction::S3Lookup {
            cid: ContentId::from_bytes([0u8; 32]),
            query_id: 1,
        };
    }

    #[test]
    fn node_config_defaults() {
        let config = NodeConfig::default();
        assert_eq!(config.storage_budget.cache_capacity, 1024);
        assert_eq!(config.compute_budget.fuel, 100_000);
    }

    #[test]
    fn tier_schedule_defaults() {
        let schedule = TierSchedule::default();
        assert!(schedule.router_max_per_tick.is_none());
        assert!(schedule.storage_max_per_tick.is_none());
        assert_eq!(schedule.starvation_threshold, 10);
        assert_eq!(schedule.adaptive_compute.high_water, 50);
        assert!((schedule.adaptive_compute.floor_fraction - 0.1).abs() < f64::EPSILON);
    }

    use harmony_content::book::MemoryBookStore;

    fn make_runtime() -> (NodeRuntime<MemoryBookStore>, Vec<RuntimeAction>) {
        let config = NodeConfig::default();
        NodeRuntime::new(config, MemoryBookStore::new())
    }

    #[test]
    fn constructor_returns_startup_actions() {
        let (_, actions) = make_runtime();

        let queryable_count = actions
            .iter()
            .filter(|a| matches!(a, RuntimeAction::DeclareQueryable { .. }))
            .count();
        let subscribe_count = actions
            .iter()
            .filter(|a| matches!(a, RuntimeAction::Subscribe { .. }))
            .count();

        // 16 shard + 1 stats + 1 compute + 1 page + 1 memo + 1 discover = 21
        assert_eq!(queryable_count, 21);
        // transit + publish + content filter + flatpack filter + memo filter + page filter subscriptions = 6
        assert_eq!(subscribe_count, 6);
    }

    #[test]
    fn metrics_start_at_zero() {
        let (rt, _) = make_runtime();
        let m = rt.metrics();
        assert_eq!(m.queries_served, 0);
        assert_eq!(m.cache_hits, 0);
    }

    #[test]
    fn queues_start_empty() {
        let (rt, _) = make_runtime();
        assert_eq!(rt.router_queue_len(), 0);
        assert_eq!(rt.storage_queue_len(), 0);
        assert_eq!(rt.compute_queue_len(), 0);
    }

    #[test]
    fn push_event_classifies_router_events() {
        let (mut rt, _) = make_runtime();
        rt.push_event(RuntimeEvent::InboundPacket {
            interface_name: "lo".into(),
            raw: vec![0u8; 20],
            now: 1000,
        });
        rt.push_event(RuntimeEvent::TimerTick {
            now: 1001,
            unix_now: 0,
        });
        assert_eq!(rt.router_queue_len(), 2);
        assert_eq!(rt.storage_queue_len(), 0);
    }

    #[test]
    fn push_event_routes_query_to_storage() {
        let (mut rt, _) = make_runtime();
        rt.push_event(RuntimeEvent::QueryReceived {
            query_id: 1,
            key_expr: "harmony/content/a/abc123".into(),
            payload: vec![],
        });
        // Key expr "harmony/content/a/abc123" matches shard queryable "harmony/content/a/**",
        // but the CID hex "c123" is too short to decode as 32 bytes, so
        // parse_storage_query drops it. Use a stats query instead -- always matches.
        rt.push_event(RuntimeEvent::QueryReceived {
            query_id: 2,
            key_expr: "harmony/content/stats".into(),
            payload: vec![],
        });
        assert_eq!(rt.storage_queue_len(), 1); // only stats query matched
    }

    #[test]
    fn tick_drains_all_router_and_storage_events() {
        let (mut rt, _) = make_runtime();

        for i in 0..3 {
            rt.push_event(RuntimeEvent::TimerTick {
                now: 1000 + i,
                unix_now: 0,
            });
        }
        rt.push_event(RuntimeEvent::QueryReceived {
            query_id: 10,
            key_expr: "harmony/content/stats".into(),
            payload: vec![],
        });
        rt.push_event(RuntimeEvent::QueryReceived {
            query_id: 11,
            key_expr: "harmony/content/stats".into(),
            payload: vec![],
        });

        assert_eq!(rt.router_queue_len(), 3);
        assert_eq!(rt.storage_queue_len(), 2);

        // One tick should drain ALL router and ALL storage events.
        let actions = rt.tick();
        assert_eq!(rt.router_queue_len(), 0);
        assert_eq!(rt.storage_queue_len(), 0);

        let reply_count = actions
            .iter()
            .filter(|a| matches!(a, RuntimeAction::SendReply { .. }))
            .count();
        assert_eq!(reply_count, 2);
    }

    #[test]
    fn tick_with_empty_queues_returns_nothing() {
        let (mut rt, _) = make_runtime();
        let actions = rt.tick();
        assert!(actions.is_empty());
    }

    use harmony_content::cid::ContentId;

    /// Helper: build a valid CID hex string for test data.
    fn cid_hex_for(data: &[u8]) -> (ContentId, String) {
        let cid = ContentId::for_book(data, harmony_content::cid::ContentFlags::default()).unwrap();
        let hex = hex::encode(cid.to_bytes());
        (cid, hex)
    }

    #[test]
    fn publish_then_query_round_trip() {
        let (mut rt, _) = make_runtime();
        let data = b"round trip test data";
        let (_cid, cid_hex) = cid_hex_for(data);

        // Publish via subscription message
        let publish_key = format!("harmony/content/publish/{cid_hex}");
        rt.push_event(RuntimeEvent::SubscriptionMessage {
            key_expr: publish_key,
            payload: data.to_vec(),
        });
        let publish_actions = rt.tick();

        // Should get an AnnounceContent → Publish action
        assert!(
            publish_actions
                .iter()
                .any(|a| matches!(a, RuntimeAction::Publish { .. })),
            "expected Publish action from publish event"
        );

        // Query the same CID
        let first_char = &cid_hex[..1];
        let query_key = format!("harmony/content/{first_char}/{cid_hex}");
        rt.push_event(RuntimeEvent::QueryReceived {
            query_id: 42,
            key_expr: query_key,
            payload: vec![],
        });
        let query_actions = rt.tick();

        // Should get a SendReply with the original data
        let reply = query_actions
            .iter()
            .find(|a| matches!(a, RuntimeAction::SendReply { query_id: 42, .. }));
        assert!(reply.is_some(), "expected SendReply for query_id 42");
        if let Some(RuntimeAction::SendReply { payload, .. }) = reply {
            assert_eq!(payload.as_slice(), data);
        }
    }

    #[test]
    fn transit_content_admitted_and_queryable() {
        let (mut rt, _) = make_runtime();
        let data = b"transit data";
        let (_cid, cid_hex) = cid_hex_for(data);

        // Transit via subscription
        let transit_key = format!("harmony/content/transit/{cid_hex}");
        rt.push_event(RuntimeEvent::SubscriptionMessage {
            key_expr: transit_key,
            payload: data.to_vec(),
        });
        let transit_actions = rt.tick();
        assert!(
            transit_actions
                .iter()
                .any(|a| matches!(a, RuntimeAction::Publish { .. })),
            "admitted transit should produce announcement"
        );

        // Query it back
        let first_char = &cid_hex[..1];
        let query_key = format!("harmony/content/{first_char}/{cid_hex}");
        rt.push_event(RuntimeEvent::QueryReceived {
            query_id: 99,
            key_expr: query_key,
            payload: vec![],
        });
        let query_actions = rt.tick();
        assert!(
            query_actions
                .iter()
                .any(|a| matches!(a, RuntimeAction::SendReply { query_id: 99, .. })),
            "cached transit content should be queryable"
        );
    }

    #[test]
    fn stats_query_returns_metrics() {
        let (mut rt, _) = make_runtime();

        // Do a content query first (to bump metrics)
        let (_, cid_hex) = cid_hex_for(b"anything");
        let first_char = &cid_hex[..1];
        rt.push_event(RuntimeEvent::QueryReceived {
            query_id: 1,
            key_expr: format!("harmony/content/{first_char}/{cid_hex}"),
            payload: vec![],
        });
        rt.tick();

        // Now query stats
        rt.push_event(RuntimeEvent::QueryReceived {
            query_id: 50,
            key_expr: "harmony/content/stats".into(),
            payload: vec![],
        });
        let actions = rt.tick();

        let reply = actions
            .iter()
            .find(|a| matches!(a, RuntimeAction::SendReply { query_id: 50, .. }));
        assert!(reply.is_some(), "stats query should produce reply");
        if let Some(RuntimeAction::SendReply { payload, .. }) = reply {
            // 11 metrics × 8 bytes = 88 bytes
            assert_eq!(payload.len(), 88);
            // First metric is queries_served, should be 1
            let queries = u64::from_be_bytes(payload[0..8].try_into().unwrap());
            assert_eq!(queries, 1);
        }
    }

    #[test]
    fn malformed_query_key_silently_dropped() {
        let (mut rt, _) = make_runtime();
        // Invalid CID hex (not 64 chars)
        rt.push_event(RuntimeEvent::QueryReceived {
            query_id: 1,
            key_expr: "harmony/content/a/not_valid_hex".into(),
            payload: vec![],
        });
        // Should not panic, event is silently dropped
        assert_eq!(rt.storage_queue_len(), 0);
    }

    #[test]
    fn malformed_subscription_key_silently_dropped() {
        let (mut rt, _) = make_runtime();
        rt.push_event(RuntimeEvent::SubscriptionMessage {
            key_expr: "harmony/content/transit/bad_hex".into(),
            payload: vec![1, 2, 3],
        });
        assert_eq!(rt.storage_queue_len(), 0);
    }

    // ── Tier 3: Compute integration tests ────────────────────────────

    #[test]
    fn startup_declares_compute_queryable() {
        let (_, actions) = make_runtime();
        let compute_queryables: Vec<_> = actions
            .iter()
            .filter(|a| {
                matches!(
                    a,
                    RuntimeAction::DeclareQueryable { key_expr } if key_expr.starts_with("harmony/compute")
                )
            })
            .collect();
        assert_eq!(compute_queryables.len(), 1);
    }

    #[test]
    fn compute_inline_round_trip() {
        let (mut rt, _) = make_runtime();

        let module = ADD_WAT.as_bytes();
        let mut input = Vec::new();
        input.extend_from_slice(&5i32.to_le_bytes());
        input.extend_from_slice(&3i32.to_le_bytes());

        let mut payload = vec![0x00];
        payload.extend_from_slice(&(module.len() as u32).to_le_bytes());
        payload.extend_from_slice(module);
        payload.extend_from_slice(&input);

        rt.push_event(RuntimeEvent::ComputeQuery {
            query_id: 99,
            key_expr: "harmony/compute/activity/test".into(),
            payload,
        });

        let actions = rt.tick();
        let reply = actions
            .iter()
            .find(|a| matches!(a, RuntimeAction::SendReply { query_id: 99, .. }));
        assert!(reply.is_some(), "compute query should produce a reply");
        if let Some(RuntimeAction::SendReply { payload, .. }) = reply {
            assert_eq!(payload[0], 0x00); // success tag
            let sum = i32::from_le_bytes(payload[1..5].try_into().unwrap());
            assert_eq!(sum, 8);
        }
    }

    #[test]
    fn tick_priority_order_with_compute() {
        let (mut rt, _) = make_runtime();

        // Queue a router event, a storage event, and a compute event
        rt.push_event(RuntimeEvent::TimerTick {
            now: 1000,
            unix_now: 0,
        });
        rt.push_event(RuntimeEvent::QueryReceived {
            query_id: 10,
            key_expr: "harmony/content/stats".into(),
            payload: vec![],
        });

        let module = ADD_WAT.as_bytes();
        let mut input = Vec::new();
        input.extend_from_slice(&1i32.to_le_bytes());
        input.extend_from_slice(&2i32.to_le_bytes());
        let mut compute_payload = vec![0x00];
        compute_payload.extend_from_slice(&(module.len() as u32).to_le_bytes());
        compute_payload.extend_from_slice(module);
        compute_payload.extend_from_slice(&input);

        rt.push_event(RuntimeEvent::ComputeQuery {
            query_id: 20,
            key_expr: "harmony/compute/activity/test".into(),
            payload: compute_payload,
        });

        let actions = rt.tick();
        let storage_reply = actions
            .iter()
            .any(|a| matches!(a, RuntimeAction::SendReply { query_id: 10, .. }));
        let compute_reply = actions
            .iter()
            .any(|a| matches!(a, RuntimeAction::SendReply { query_id: 20, .. }));
        assert!(storage_reply, "should have storage reply");
        assert!(compute_reply, "should have compute reply");
    }

    #[test]
    fn oversized_module_len_rejected() {
        let (mut rt, _) = make_runtime();

        // Build a payload with module_len = 0xFFFFFFFF (way over 10MB cap).
        let mut payload = vec![0x00];
        payload.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
        payload.extend_from_slice(&[0u8; 10]); // dummy bytes

        rt.push_event(RuntimeEvent::ComputeQuery {
            query_id: 200,
            key_expr: "harmony/compute/activity/test".into(),
            payload,
        });

        // Malformed payload: no compute task queued, but error reply emitted.
        assert_eq!(rt.compute_queue_len(), 0);

        let actions = rt.tick();
        let reply = actions.iter().find_map(|a| match a {
            RuntimeAction::SendReply {
                query_id: 200,
                payload,
            } => Some(payload),
            _ => None,
        });
        let payload = reply.expect("should emit error reply for oversized module");
        assert_eq!(payload[0], 0x01, "error tag");
        let msg = std::str::from_utf8(&payload[1..]).unwrap();
        assert!(
            msg.contains("malformed"),
            "error message should mention malformed payload"
        );
    }

    #[test]
    fn content_fetch_response_routes_to_compute() {
        let (mut rt, _) = make_runtime();
        let cid = [0xAB; 32];

        // Submit a compute query with a fetch module.
        let module = FETCH_WAT.as_bytes();
        let mut payload = vec![0x00];
        payload.extend_from_slice(&(module.len() as u32).to_le_bytes());
        payload.extend_from_slice(module);
        payload.extend_from_slice(&cid); // CID as input

        rt.push_event(RuntimeEvent::ComputeQuery {
            query_id: 300,
            key_expr: "harmony/compute/activity/test".into(),
            payload,
        });

        // tick → NeedsIO → FetchContent action
        let actions = rt.tick();
        let fetch = actions
            .iter()
            .find(|a| matches!(a, RuntimeAction::FetchContent { .. }));
        assert!(
            fetch.is_some(),
            "should emit FetchContent, got: {actions:?}"
        );

        // Deliver content via ContentFetchResponse.
        let content = b"resolved content";
        rt.push_event(RuntimeEvent::ContentFetchResponse {
            cid,
            result: Ok(content.to_vec()),
        });

        // tick → dispatches pending compute actions → SendReply
        let actions = rt.tick();
        let reply = actions
            .iter()
            .find(|a| matches!(a, RuntimeAction::SendReply { query_id: 300, .. }));
        assert!(
            reply.is_some(),
            "should emit SendReply for query 300, got: {actions:?}"
        );

        if let Some(RuntimeAction::SendReply { payload, .. }) = reply {
            assert_eq!(payload[0], 0x00, "success tag");
            let result_code = i32::from_le_bytes(payload[1..5].try_into().unwrap());
            assert_eq!(result_code, content.len() as i32);
        }
    }

    #[test]
    fn compute_content_read_round_trip() {
        let (mut rt, _) = make_runtime();
        let cid = [0x42; 32];
        let content = vec![10, 20, 30, 40, 50];

        // Submit inline compute query with fetch module.
        let module = FETCH_WAT.as_bytes();
        let mut payload = vec![0x00];
        payload.extend_from_slice(&(module.len() as u32).to_le_bytes());
        payload.extend_from_slice(module);
        payload.extend_from_slice(&cid);

        rt.push_event(RuntimeEvent::ComputeQuery {
            query_id: 400,
            key_expr: "harmony/compute/activity/fetch".into(),
            payload,
        });

        // Tick 1: execute → NeedsIO → FetchContent
        let actions = rt.tick();
        assert!(
            actions
                .iter()
                .any(|a| matches!(a, RuntimeAction::FetchContent { cid: c } if *c == cid)),
            "tick 1 should emit FetchContent"
        );

        // External IO resolves the content.
        rt.push_event(RuntimeEvent::ContentFetchResponse {
            cid,
            result: Ok(content.clone()),
        });

        // Tick 2: resume compute → SendReply
        let actions = rt.tick();
        let reply = actions
            .iter()
            .find(|a| matches!(a, RuntimeAction::SendReply { query_id: 400, .. }));
        assert!(reply.is_some(), "tick 2 should emit SendReply");

        if let Some(RuntimeAction::SendReply { payload, .. }) = reply {
            assert_eq!(payload[0], 0x00);
            let result_code = i32::from_le_bytes(payload[1..5].try_into().unwrap());
            assert_eq!(result_code, 5);
            assert_eq!(&payload[5..], &content);
        }
    }

    // ── Adaptive compute fuel tests ─────────────────────────────────

    #[test]
    fn effective_fuel_scales_with_queue_depth() {
        use harmony_content::book::MemoryBookStore;

        let mut config = NodeConfig::default();
        config.compute_budget = InstructionBudget { fuel: 1000 };
        config.schedule.adaptive_compute.high_water = 10;
        config.schedule.adaptive_compute.floor_fraction = 0.1;
        let (mut rt, _) = NodeRuntime::new(config, MemoryBookStore::new());

        // Empty queues → full budget
        assert_eq!(rt.effective_fuel(), 1000);

        // Push 5 router events (half of high_water=10)
        for i in 0..5 {
            rt.push_event(RuntimeEvent::TimerTick {
                now: 1000 + i,
                unix_now: 0,
            });
        }
        // load_factor = 5/10 = 0.5
        // effective = 1000 * (1.0 - 0.5 * 0.9) = 1000 * 0.55 = 550
        assert_eq!(rt.effective_fuel(), 550);

        // Push 5 more (at high_water)
        for i in 5..10 {
            rt.push_event(RuntimeEvent::TimerTick {
                now: 1000 + i,
                unix_now: 0,
            });
        }
        // load_factor = 10/10 = 1.0 → floor
        // effective = 1000 * 0.1 = 100
        assert_eq!(rt.effective_fuel(), 100);

        // Push beyond high_water — stays at floor
        for i in 10..20 {
            rt.push_event(RuntimeEvent::TimerTick {
                now: 1000 + i,
                unix_now: 0,
            });
        }
        assert_eq!(rt.effective_fuel(), 100);
    }

    #[test]
    fn router_max_per_tick_caps_drain() {
        let mut config = NodeConfig::default();
        config.schedule.router_max_per_tick = Some(2);
        let (mut rt, _) = NodeRuntime::new(config, MemoryBookStore::new());

        // Push 5 router events
        for i in 0..5 {
            rt.push_event(RuntimeEvent::TimerTick {
                now: 1000 + i,
                unix_now: 0,
            });
        }
        assert_eq!(rt.router_queue_len(), 5);

        // Tick should drain only 2
        rt.tick();
        assert_eq!(rt.router_queue_len(), 3);

        // Next tick drains 2 more
        rt.tick();
        assert_eq!(rt.router_queue_len(), 1);

        // Final tick drains the last one
        rt.tick();
        assert_eq!(rt.router_queue_len(), 0);
    }

    #[test]
    fn storage_max_per_tick_caps_drain() {
        let mut config = NodeConfig::default();
        config.schedule.storage_max_per_tick = Some(1);
        let (mut rt, _) = NodeRuntime::new(config, MemoryBookStore::new());

        // Push 3 storage events (stats queries)
        for i in 0..3 {
            rt.push_event(RuntimeEvent::QueryReceived {
                query_id: 100 + i,
                key_expr: "harmony/content/stats".into(),
                payload: vec![],
            });
        }
        assert_eq!(rt.storage_queue_len(), 3);

        // Tick should drain only 1
        let actions = rt.tick();
        assert_eq!(rt.storage_queue_len(), 2);
        let reply_count = actions
            .iter()
            .filter(|a| matches!(a, RuntimeAction::SendReply { .. }))
            .count();
        assert_eq!(reply_count, 1);
    }

    // ── Starvation tracking tests ─────────────────────────────────

    #[test]
    fn starvation_counters_track_idle_ticks() {
        let mut config = NodeConfig::default();
        config.schedule.router_max_per_tick = Some(1);
        let (mut rt, _) = NodeRuntime::new(config, MemoryBookStore::new());

        // No events at all → all tiers idle
        rt.tick();
        assert_eq!(rt.starvation_counters(), (1, 1, 1));

        // Push router event only
        rt.push_event(RuntimeEvent::TimerTick {
            now: 1000,
            unix_now: 0,
        });
        rt.tick();
        // Router processed → reset to 0. Storage/compute still idle → increment.
        assert_eq!(rt.starvation_counters(), (0, 2, 2));

        // Push storage event only
        rt.push_event(RuntimeEvent::QueryReceived {
            query_id: 1,
            key_expr: "harmony/content/stats".into(),
            payload: vec![],
        });
        rt.tick();
        // Router idle → 1. Storage processed → 0. Compute still idle → 3.
        assert_eq!(rt.starvation_counters(), (1, 0, 3));
    }

    #[test]
    fn starvation_promotes_starved_tier() {
        let mut config = NodeConfig::default();
        config.schedule.router_max_per_tick = Some(1);
        config.schedule.starvation_threshold = 3;
        let (mut rt, _) = NodeRuntime::new(config, MemoryBookStore::new());

        // Push many router events but no storage events
        for i in 0..10 {
            rt.push_event(RuntimeEvent::TimerTick {
                now: 1000 + i,
                unix_now: 0,
            });
        }

        // Also push 1 storage event
        rt.push_event(RuntimeEvent::QueryReceived {
            query_id: 50,
            key_expr: "harmony/content/stats".into(),
            payload: vec![],
        });

        // Tick 1: router=1, storage=1 (processes the stats query)
        let actions = rt.tick();
        assert!(actions
            .iter()
            .any(|a| matches!(a, RuntimeAction::SendReply { query_id: 50, .. })));

        // Verify default order: router actions appear before storage actions.
        // SendOnInterface (router) should precede SendReply (storage) if any router events produce output.
        // At minimum, starvation counters confirm storage is not yet starved.
        assert_eq!(
            rt.starvation_counters().1,
            0,
            "storage processed, counter should be 0"
        );

        rt.tick(); // tick 2: router=1, storage=0 (starved=1)
        assert_eq!(rt.starvation_counters().1, 1);
        rt.tick(); // tick 3: router=1, storage=0 (starved=2)
        assert_eq!(rt.starvation_counters().1, 2);
        rt.tick(); // tick 4: router=1, storage=0 (starved=3 → threshold hit)
        assert_eq!(
            rt.starvation_counters().1,
            3,
            "storage should hit starvation threshold"
        );

        // Now push both a storage event AND a router event.
        // Without promotion, default order is [Router, Storage, Compute].
        // With promotion, storage (starved=3 >= threshold=3) moves to front: [Storage, Router, Compute].
        rt.push_event(RuntimeEvent::QueryReceived {
            query_id: 60,
            key_expr: "harmony/content/stats".into(),
            payload: vec![],
        });
        rt.push_event(RuntimeEvent::TimerTick {
            now: 2000,
            unix_now: 0,
        });

        // Tick 5: storage promoted → its actions should appear before router actions in output
        let actions = rt.tick();
        assert!(
            actions
                .iter()
                .any(|a| matches!(a, RuntimeAction::SendReply { query_id: 60, .. })),
            "promoted storage tier should process its event"
        );

        // Verify promotion resets the starvation counter
        assert_eq!(
            rt.starvation_counters().1,
            0,
            "storage counter should reset after processing"
        );

        // Verify ordering: SendReply (storage) should appear before SendOnInterface (router)
        // because storage was promoted to process first.
        let reply_pos = actions
            .iter()
            .position(|a| matches!(a, RuntimeAction::SendReply { query_id: 60, .. }));
        let send_pos = actions
            .iter()
            .position(|a| matches!(a, RuntimeAction::SendOnInterface { .. }));
        if let (Some(r), Some(s)) = (reply_pos, send_pos) {
            assert!(
                r < s,
                "promoted storage actions should appear before router actions"
            );
        }
    }

    #[test]
    #[should_panic(expected = "router_max_per_tick must be None or > 0")]
    fn zero_router_limit_panics() {
        let mut config = NodeConfig::default();
        config.schedule.router_max_per_tick = Some(0);
        let _ = NodeRuntime::new(config, MemoryBookStore::new());
    }

    #[test]
    #[should_panic(expected = "storage_max_per_tick must be None or > 0")]
    fn zero_storage_limit_panics() {
        let mut config = NodeConfig::default();
        config.schedule.storage_max_per_tick = Some(0);
        let _ = NodeRuntime::new(config, MemoryBookStore::new());
    }

    #[test]
    fn adaptive_fuel_reduces_compute_under_load() {
        let mut config = NodeConfig::default();
        config.compute_budget = InstructionBudget { fuel: 1000 };
        config.schedule.adaptive_compute.high_water = 10;
        config.schedule.adaptive_compute.floor_fraction = 0.1;
        let (mut rt, _) = NodeRuntime::new(config, MemoryBookStore::new());

        // Verify full fuel with empty queues
        assert_eq!(rt.effective_fuel(), 1000);

        // Push 10 router events (= high_water) → fuel at floor
        for i in 0..10 {
            rt.push_event(RuntimeEvent::TimerTick {
                now: 1000 + i,
                unix_now: 0,
            });
        }
        assert_eq!(rt.effective_fuel(), 100);

        // After tick drains them all, fuel should recover
        rt.tick();
        assert_eq!(rt.effective_fuel(), 1000);
    }

    #[test]
    fn runtime_uses_content_policy() {
        use harmony_content::book::MemoryBookStore;

        let config = NodeConfig {
            storage_budget: StorageBudget {
                cache_capacity: 100,
                max_pinned_bytes: 1_000_000,
            },
            compute_budget: InstructionBudget { fuel: 1000 },
            schedule: Default::default(),
            content_policy: ContentPolicy {
                encrypted_durable_persist: true,
                encrypted_durable_announce: true,
                public_ephemeral_announce: false,
            },
            filter_broadcast_config: FilterBroadcastConfig::default(),
            node_addr: "test".to_string(),
            local_identity_hash: [0u8; 16],
            local_pq_identity_hash: [0u8; 16],
            local_dsa_pubkey: Vec::new(),
            inference_gguf_cid: None,
            inference_tokenizer_cid: None,
            disk_enabled: false,
            disk_entries: Vec::new(),
            disk_quota: None,
            s3_enabled: false,
        };
        let (rt, _) = NodeRuntime::new(config, MemoryBookStore::new());
        assert_eq!(rt.storage_queue_len(), 0);
    }

    // ── PeerFilterTable tests ───────────────────────────────────────

    use harmony_content::cid::ContentFlags;

    #[test]
    fn peer_filter_table_skips_definite_miss() {
        let mut table = PeerFilterTable::new(100);
        let mut filter = BloomFilter::new(1000, 0.01);
        let cid_in = ContentId::for_book(b"present", ContentFlags::default()).unwrap();
        let cid_out = ContentId::for_book(b"absent", ContentFlags::default()).unwrap();
        filter.insert(&cid_in);
        table.upsert_content("peer-1".into(), filter, 10);

        assert!(table.should_query("peer-1", &cid_in, 10));
        assert!(!table.should_query("peer-1", &cid_out, 10));
    }

    #[test]
    fn peer_filter_table_queries_unknown_peer() {
        let table = PeerFilterTable::new(100);
        let cid = ContentId::for_book(b"test", ContentFlags::default()).unwrap();
        assert!(table.should_query("unknown", &cid, 10));
    }

    #[test]
    fn peer_filter_table_queries_stale_filter() {
        let mut table = PeerFilterTable::new(100);
        let filter = BloomFilter::new(1000, 0.01);
        table.upsert_content("peer-1".into(), filter, 10);

        let cid = ContentId::for_book(b"test", ContentFlags::default()).unwrap();
        // Fresh filter with no items => definite miss, should NOT query
        assert!(!table.should_query("peer-1", &cid, 10));
        // Stale filter (current_tick - received_tick > staleness_ticks) => should query
        assert!(table.should_query("peer-1", &cid, 200));
    }

    #[test]
    fn peer_filter_table_tracks_parse_errors() {
        let mut table = PeerFilterTable::new(100);
        assert_eq!(table.parse_errors(), 0);
        table.record_parse_error();
        table.record_parse_error();
        assert_eq!(table.parse_errors(), 2);
    }

    #[test]
    fn route_subscription_counts_malformed_filter() {
        use harmony_content::book::MemoryBookStore;

        let config = NodeConfig {
            storage_budget: StorageBudget {
                cache_capacity: 100,
                max_pinned_bytes: 1_000_000,
            },
            compute_budget: InstructionBudget { fuel: 1000 },
            schedule: Default::default(),
            content_policy: ContentPolicy::default(),
            filter_broadcast_config: FilterBroadcastConfig::default(),
            node_addr: "self-node".to_string(),
            local_identity_hash: [0u8; 16],
            local_pq_identity_hash: [0u8; 16],
            local_dsa_pubkey: Vec::new(),
            inference_gguf_cid: None,
            inference_tokenizer_cid: None,
            disk_enabled: false,
            disk_entries: Vec::new(),
            disk_quota: None,
            s3_enabled: false,
        };
        let (mut rt, _) = NodeRuntime::new(config, MemoryBookStore::new());

        // Send a malformed filter payload from a peer.
        rt.push_event(RuntimeEvent::SubscriptionMessage {
            key_expr: "harmony/filters/content/peer-abc".to_string(),
            payload: vec![0xFF, 0x01], // too short to parse
        });
        rt.tick();

        assert_eq!(rt.peer_filter_parse_errors(), 1);
    }

    #[test]
    #[should_panic(expected = "filter_broadcast_interval_ticks must be >= 2")]
    fn filter_interval_rejects_less_than_two() {
        use harmony_content::book::MemoryBookStore;

        let config = NodeConfig {
            storage_budget: StorageBudget {
                cache_capacity: 100,
                max_pinned_bytes: 1_000_000,
            },
            compute_budget: InstructionBudget { fuel: 1000 },
            schedule: Default::default(),
            content_policy: ContentPolicy::default(),
            filter_broadcast_config: FilterBroadcastConfig {
                max_interval_ticks: 1,
                ..FilterBroadcastConfig::default()
            },
            node_addr: "reject-test".to_string(),
            local_identity_hash: [0u8; 16],
            local_pq_identity_hash: [0u8; 16],
            local_dsa_pubkey: Vec::new(),
            inference_gguf_cid: None,
            inference_tokenizer_cid: None,
            disk_enabled: false,
            disk_entries: Vec::new(),
            disk_quota: None,
            s3_enabled: false,
        };
        let _ = NodeRuntime::new(config, MemoryBookStore::new());
    }

    #[test]
    fn timer_skipped_when_threshold_broadcast_pending() {
        use harmony_content::book::MemoryBookStore;
        use harmony_content::cid::ContentFlags;

        // Set mutation_threshold=2 and max_interval_ticks=2.
        // Queue 2 transit events (crosses threshold) then call tick().
        // The timer also fires on tick 2, producing both bloom and cuckoo
        // broadcasts. The bloom broadcast from threshold and timer are
        // coalesced into one publish; the cuckoo broadcast adds a second.
        let config = NodeConfig {
            storage_budget: StorageBudget {
                cache_capacity: 100,
                max_pinned_bytes: 1_000_000,
            },
            compute_budget: InstructionBudget { fuel: 1000 },
            schedule: Default::default(),
            content_policy: ContentPolicy::default(),
            filter_broadcast_config: FilterBroadcastConfig {
                mutation_threshold: 2,
                max_interval_ticks: 2,
                ..FilterBroadcastConfig::default()
            },
            node_addr: "skip-timer-test".to_string(),
            local_identity_hash: [0u8; 16],
            local_pq_identity_hash: [0u8; 16],
            local_dsa_pubkey: Vec::new(),
            inference_gguf_cid: None,
            inference_tokenizer_cid: None,
            disk_enabled: false,
            disk_entries: Vec::new(),
            disk_quota: None,
            s3_enabled: false,
        };
        let (mut rt, _) = NodeRuntime::new(config, MemoryBookStore::new());

        // Tick 1: no events, no broadcast.
        rt.tick();

        // Queue 2 transit events to cross the mutation threshold.
        for i in 0..2u8 {
            let data = [i; 16];
            let cid = ContentId::for_book(&data, ContentFlags::default()).unwrap();
            let cid_hex = hex::encode(cid.to_bytes());
            let key_expr = format!("harmony/content/transit/{cid_hex}");
            rt.push_event(RuntimeEvent::SubscriptionMessage {
                key_expr,
                payload: data.to_vec(),
            });
        }

        // Tick 2: timer interval also fires (tick 2 >= interval 2).
        // Bloom coalesces (threshold + timer → 1 publish), cuckoo adds 1.
        // The transit data are Book CIDs that get indexed into page_index,
        // so the timer also produces 1 page filter broadcast.
        let actions = rt.tick();
        let filter_publishes: Vec<_> = actions
            .iter()
            .filter(|a| {
                matches!(a, RuntimeAction::Publish { key_expr, .. }
                    if key_expr.starts_with("harmony/filters/"))
            })
            .collect();
        assert_eq!(
            filter_publishes.len(),
            3,
            "expected 3 filter broadcasts (1 coalesced bloom + 1 cuckoo + 1 page)"
        );
    }

    #[test]
    fn filter_broadcasts_coalesced_within_tick() {
        use harmony_content::book::MemoryBookStore;
        use harmony_content::cid::ContentFlags;

        // mutation_threshold=2: every 2 transit events triggers a rebuild.
        // We'll queue 6 events → would be 3 rebuilds without coalescing.
        let config = NodeConfig {
            storage_budget: StorageBudget {
                cache_capacity: 100,
                max_pinned_bytes: 1_000_000,
            },
            compute_budget: InstructionBudget { fuel: 1000 },
            schedule: Default::default(),
            content_policy: ContentPolicy::default(),
            filter_broadcast_config: FilterBroadcastConfig {
                mutation_threshold: 2,
                ..FilterBroadcastConfig::default()
            },
            node_addr: "coalesce-test".to_string(),
            local_identity_hash: [0u8; 16],
            local_pq_identity_hash: [0u8; 16],
            local_dsa_pubkey: Vec::new(),
            inference_gguf_cid: None,
            inference_tokenizer_cid: None,
            disk_enabled: false,
            disk_entries: Vec::new(),
            disk_quota: None,
            s3_enabled: false,
        };
        let (mut rt, _) = NodeRuntime::new(config, MemoryBookStore::new());

        // Queue 6 transit events (each unique CID).
        for i in 0..6u8 {
            let data = [i; 16];
            let cid = ContentId::for_book(&data, ContentFlags::default()).unwrap();
            let cid_hex = hex::encode(cid.to_bytes());
            let key_expr = format!("harmony/content/transit/{cid_hex}");
            rt.push_event(RuntimeEvent::SubscriptionMessage {
                key_expr,
                payload: data.to_vec(),
            });
        }

        let actions = rt.tick();
        // Only ONE Publish with the filter key should appear, not three.
        let filter_publishes: Vec<_> = actions
            .iter()
            .filter(|a| {
                matches!(a, RuntimeAction::Publish { key_expr, .. }
                if key_expr.starts_with("harmony/filters/"))
            })
            .collect();
        assert_eq!(
            filter_publishes.len(),
            1,
            "expected 1 coalesced filter broadcast, got {}",
            filter_publishes.len()
        );
    }

    #[test]
    fn cuckoo_filter_broadcast_dispatched() {
        use harmony_content::book::MemoryBookStore;

        let config = NodeConfig {
            storage_budget: StorageBudget {
                cache_capacity: 100,
                max_pinned_bytes: 1_000_000,
            },
            compute_budget: InstructionBudget { fuel: 1000 },
            schedule: Default::default(),
            content_policy: ContentPolicy::default(),
            filter_broadcast_config: FilterBroadcastConfig::default(),
            node_addr: "cuckoo-test".to_string(),
            local_identity_hash: [0u8; 16],
            local_pq_identity_hash: [0u8; 16],
            local_dsa_pubkey: Vec::new(),
            inference_gguf_cid: None,
            inference_tokenizer_cid: None,
            disk_enabled: false,
            disk_entries: Vec::new(),
            disk_quota: None,
            s3_enabled: false,
        };
        let (mut rt, _) = NodeRuntime::new(config, MemoryBookStore::new());

        // Tick enough times to trigger the timer (default interval is 30).
        // The timer fires on the 30th tick.
        for _ in 0..29 {
            rt.tick();
        }
        let actions = rt.tick();

        let filter_publishes: Vec<_> = actions
            .iter()
            .filter(|a| {
                matches!(a, RuntimeAction::Publish { key_expr, .. }
                if key_expr.starts_with("harmony/filters/"))
            })
            .collect();
        assert!(
            filter_publishes.len() >= 2,
            "expected bloom + cuckoo filter broadcasts, got {}",
            filter_publishes.len()
        );
    }

    #[test]
    fn route_subscription_parses_cuckoo_filter() {
        use harmony_content::book::MemoryBookStore;
        use harmony_content::cuckoo::CuckooFilter;

        let config = NodeConfig {
            storage_budget: StorageBudget {
                cache_capacity: 100,
                max_pinned_bytes: 1_000_000,
            },
            compute_budget: InstructionBudget { fuel: 1000 },
            schedule: Default::default(),
            content_policy: ContentPolicy::default(),
            filter_broadcast_config: FilterBroadcastConfig::default(),
            node_addr: "self-node".to_string(),
            local_identity_hash: [0u8; 16],
            local_pq_identity_hash: [0u8; 16],
            local_dsa_pubkey: Vec::new(),
            inference_gguf_cid: None,
            inference_tokenizer_cid: None,
            disk_enabled: false,
            disk_entries: Vec::new(),
            disk_quota: None,
            s3_enabled: false,
        };
        let (mut rt, _) = NodeRuntime::new(config, MemoryBookStore::new());

        let cf = CuckooFilter::new(100);
        let payload = cf.to_bytes();

        rt.push_event(RuntimeEvent::SubscriptionMessage {
            key_expr: "harmony/filters/flatpack/peer-xyz".to_string(),
            payload,
        });
        rt.tick();

        assert_eq!(rt.peer_filter_parse_errors(), 0);
    }

    #[test]
    fn staleness_tracking_independent_per_filter_type() {
        let mut table = PeerFilterTable::new(10);
        let cid = ContentId::for_book(b"staleness-test", ContentFlags::default()).unwrap();

        // Insert content filter at tick 5.
        let bf = BloomFilter::new(100, 0.01);
        table.upsert_content("peer-a".to_string(), bf, 5);

        // At tick 20 (15 ticks later, > staleness_ticks=10), content filter
        // should be stale, so should_query returns true (fall back to querying).
        assert!(
            table.should_query("peer-a", &cid, 20),
            "stale content filter should fall back to querying"
        );

        // But if we also insert a flatpack filter at tick 18...
        let cf = CuckooFilter::new(100);
        table.upsert_flatpack("peer-a".to_string(), cf, 18);

        // Content filter should STILL be considered stale at tick 20.
        // (With the old shared received_tick, tick 18 would mask content staleness.)
        assert!(
            table.should_query("peer-a", &cid, 20),
            "content filter must be stale independently of flatpack freshness"
        );

        // Flatpack filter should be fresh at tick 20 (20-18=2 <= 10).
        assert!(
            !table.should_query_flatpack("peer-a", &cid, 20),
            "fresh flatpack filter with empty cuckoo should say 'definitely no'"
        );
    }

    // ── Discovery integration tests ─────────────────────────────────

    #[test]
    fn discovery_tunnel_hint_creates_contact() {
        let (mut rt, _) = make_runtime();
        let record = harmony_discovery::AnnounceRecord {
            identity_ref: harmony_identity::IdentityRef {
                hash: [0xDD; 16],
                suite: harmony_identity::CryptoSuite::Ed25519,
            },
            public_key: vec![0u8; 32],
            encryption_key: vec![],
            routing_hints: vec![harmony_discovery::RoutingHint::Tunnel {
                node_id: [0xEE; 32],
                relay_url: Some("https://iroh.q8.fyi".into()),
                direct_addrs: vec![],
            }],
            published_at: 1000,
            expires_at: 2000,
            nonce: [0u8; 16],
            signature: vec![0u8; 64],
        };
        rt.process_discovered_tunnel_hints(&record);
        let contact = rt.contact_store().get(&[0xDD; 16]).unwrap();
        assert_eq!(contact.addresses.len(), 1);
        assert!(matches!(
            &contact.addresses[0],
            harmony_contacts::ContactAddress::Tunnel { node_id, .. } if *node_id == [0xEE; 32]
        ));
    }

    #[test]
    fn discovery_tunnel_hint_updates_existing_contact() {
        let (mut rt, _) = make_runtime();
        // Pre-add contact
        let contact = harmony_contacts::Contact {
            identity_hash: [0xFF; 16],
            display_name: None,
            peering: harmony_contacts::PeeringPolicy {
                enabled: true,
                priority: harmony_contacts::PeeringPriority::High,
            },
            added_at: 1000,
            last_seen: None,
            notes: None,
            addresses: vec![harmony_contacts::ContactAddress::Tunnel {
                node_id: [0x11; 32],
                relay_url: None,
                direct_addrs: vec![],
            }],
            replication: None,
        };
        rt.contact_store_mut().add(contact).unwrap();

        let record = harmony_discovery::AnnounceRecord {
            identity_ref: harmony_identity::IdentityRef {
                hash: [0xFF; 16],
                suite: harmony_identity::CryptoSuite::Ed25519,
            },
            public_key: vec![0u8; 32],
            encryption_key: vec![],
            routing_hints: vec![harmony_discovery::RoutingHint::Tunnel {
                node_id: [0x22; 32],
                relay_url: Some("https://new-relay.example.com".into()),
                direct_addrs: vec![],
            }],
            published_at: 2000,
            expires_at: 3000,
            nonce: [0u8; 16],
            signature: vec![0u8; 64],
        };
        rt.process_discovered_tunnel_hints(&record);

        let contact = rt.contact_store().get(&[0xFF; 16]).unwrap();
        assert_eq!(contact.addresses.len(), 1);
        assert!(matches!(
            &contact.addresses[0],
            harmony_contacts::ContactAddress::Tunnel { node_id, .. } if *node_id == [0x22; 32]
        ));
    }

    #[test]
    fn discovery_no_tunnel_hints_is_noop() {
        let (mut rt, _) = make_runtime();
        let record = harmony_discovery::AnnounceRecord {
            identity_ref: harmony_identity::IdentityRef {
                hash: [0xCC; 16],
                suite: harmony_identity::CryptoSuite::Ed25519,
            },
            public_key: vec![0u8; 32],
            encryption_key: vec![],
            routing_hints: vec![harmony_discovery::RoutingHint::Reticulum {
                destination_hash: [0xAA; 16],
            }],
            published_at: 1000,
            expires_at: 2000,
            nonce: [0u8; 16],
            signature: vec![0u8; 64],
        };
        rt.process_discovered_tunnel_hints(&record);
        assert!(rt.contact_store().get(&[0xCC; 16]).is_none());
    }

    #[test]
    fn local_tunnel_info_stores_hint() {
        let (mut rt, _) = make_runtime();
        assert!(rt.local_tunnel_hint.is_none());
        rt.push_event(RuntimeEvent::LocalTunnelInfo {
            node_id: [0xAA; 32],
            relay_url: Some("https://iroh.q8.fyi".into()),
        });
        assert!(matches!(
            &rt.local_tunnel_hint,
            Some(harmony_discovery::RoutingHint::Tunnel { node_id, relay_url, .. })
            if *node_id == [0xAA; 32] && relay_url.as_deref() == Some("https://iroh.q8.fyi")
        ));
    }

    // ── Peer lifecycle tests (from PR 89) ─────────────────────────────

    #[test]
    fn runtime_event_peer_variants_exist() {
        let _e1 = RuntimeEvent::TunnelPeerEstablished {
            identity_hash: [0xAA; 16],
            node_id: [0xBB; 32],
            now: 5000,
        };
        let _e2 = RuntimeEvent::TunnelPeerFailed {
            identity_hash: [0xCC; 16],
        };
        let _e3 = RuntimeEvent::TunnelPeerDropped {
            identity_hash: [0xDD; 16],
        };
        let _e4 = RuntimeEvent::ContactChanged {
            identity_hash: [0xEE; 16],
        };
    }

    #[test]
    fn runtime_action_peer_variants_exist() {
        let _a1 = RuntimeAction::InitiateTunnel {
            identity_hash: [0xAA; 16],
            node_id: [0xBB; 32],
            relay_url: Some("https://relay.example.com".into()),
            peer_dsa_pubkey: vec![0xDD; 1952],
            peer_kem_pubkey: vec![0xEE; 1184],
        };
        let _a2 = RuntimeAction::SendPathRequest {
            identity_hash: [0xCC; 16],
        };
    }

    #[test]
    fn contact_store_accessible() {
        let (rt, _) = make_runtime();
        assert!(rt.contact_store().is_empty());
    }

    #[test]
    fn contact_store_mut_accessible() {
        let (mut rt, _) = make_runtime();
        let contact = harmony_contacts::Contact {
            identity_hash: [0xAA; 16],
            display_name: None,
            peering: harmony_contacts::PeeringPolicy {
                enabled: true,
                priority: harmony_contacts::PeeringPriority::Normal,
            },
            added_at: 1000,
            last_seen: None,
            notes: None,
            addresses: vec![],
            replication: None,
        };
        rt.contact_store_mut().add(contact).unwrap();
        assert_eq!(rt.contact_store().len(), 1);
    }

    #[test]
    fn contact_changed_event_registers_peer() {
        let (mut rt, _) = make_runtime();
        let contact = harmony_contacts::Contact {
            identity_hash: [0xAA; 16],
            display_name: None,
            peering: harmony_contacts::PeeringPolicy {
                enabled: true,
                priority: harmony_contacts::PeeringPriority::High,
            },
            added_at: 1000,
            last_seen: None,
            notes: None,
            addresses: vec![],
            replication: None,
        };
        rt.contact_store_mut().add(contact).unwrap();
        rt.push_event(RuntimeEvent::ContactChanged {
            identity_hash: [0xAA; 16],
        });
        // Tick with a timer to trigger PeerManager tick, which should produce
        // a SendPathRequest for the High-priority peer.
        rt.push_event(RuntimeEvent::TimerTick {
            now: 1000,
            unix_now: 0,
        });
        let actions = rt.tick();
        assert!(
            actions.iter().any(|a| matches!(
                a,
                RuntimeAction::SendPathRequest { identity_hash }
                if *identity_hash == [0xAA; 16]
            )),
            "expected SendPathRequest for [0xAA;16], got: {actions:?}"
        );
    }

    #[test]
    fn tunnel_established_event_updates_contact_last_seen() {
        let (mut rt, _) = make_runtime();
        let contact = harmony_contacts::Contact {
            identity_hash: [0xBB; 16],
            display_name: None,
            peering: harmony_contacts::PeeringPolicy {
                enabled: true,
                priority: harmony_contacts::PeeringPriority::High,
            },
            added_at: 1000,
            last_seen: None,
            notes: None,
            addresses: vec![harmony_contacts::ContactAddress::Tunnel {
                node_id: [0xCC; 32],
                relay_url: None,
                direct_addrs: vec![],
            }],
            replication: None,
        };
        rt.contact_store_mut().add(contact).unwrap();
        rt.push_event(RuntimeEvent::ContactChanged {
            identity_hash: [0xBB; 16],
        });
        // Event loop provides milliseconds; PeerManager converts to seconds.
        rt.push_event(RuntimeEvent::TunnelPeerEstablished {
            identity_hash: [0xBB; 16],
            node_id: [0xCC; 32],
            now: 5000, // 5000ms from event loop
        });
        // UpdateLastSeen stores wall-clock Unix epoch seconds (not monotonic).
        let last_seen = rt.contact_store().get(&[0xBB; 16]).unwrap().last_seen;
        assert!(last_seen.is_some());
        // Should be a plausible Unix timestamp (after 2024-01-01).
        assert!(last_seen.unwrap() > 1_700_000_000);
    }

    // ── Replication tests ─────────────────────────────────────────────────

    #[test]
    fn replica_received_stores_within_quota() {
        let (mut rt, _) = make_runtime();
        let contact = harmony_contacts::Contact {
            identity_hash: [0xAA; 16],
            display_name: None,
            peering: harmony_contacts::PeeringPolicy {
                enabled: true,
                priority: harmony_contacts::PeeringPriority::Normal,
            },
            added_at: 1000,
            last_seen: None,
            notes: None,
            addresses: vec![],
            replication: Some(harmony_contacts::ReplicationPolicy {
                quota_bytes: 10_000,
            }),
        };
        rt.contact_store_mut().add(contact).unwrap();

        rt.push_event(RuntimeEvent::ReplicaPushReceived {
            peer_identity: [0xAA; 16],
            cid: [0xBB; 32],
            data: vec![1, 2, 3, 4, 5],
        });

        // Verify replica store has the data.
        assert_eq!(rt.replica_store.usage(&[0xAA; 16]), 5);
        assert_eq!(
            rt.replica_store.retrieve(&[0xAA; 16], &[0xBB; 32]),
            Some(vec![1, 2, 3, 4, 5])
        );
    }

    #[test]
    fn replica_received_rejected_without_policy() {
        let (mut rt, _) = make_runtime();
        // Add contact WITHOUT replication policy
        let contact = harmony_contacts::Contact {
            identity_hash: [0xAA; 16],
            display_name: None,
            peering: harmony_contacts::PeeringPolicy {
                enabled: true,
                priority: harmony_contacts::PeeringPriority::Normal,
            },
            added_at: 1000,
            last_seen: None,
            notes: None,
            addresses: vec![],
            replication: None,
        };
        rt.contact_store_mut().add(contact).unwrap();

        rt.push_event(RuntimeEvent::ReplicaPushReceived {
            peer_identity: [0xAA; 16],
            cid: [0xBB; 32],
            data: vec![1, 2, 3, 4, 5],
        });

        // Nothing stored — no replication policy.
        assert_eq!(rt.replica_store.usage(&[0xAA; 16]), 0);
        assert_eq!(rt.replica_store.retrieve(&[0xAA; 16], &[0xBB; 32]), None);
    }

    #[test]
    fn replica_received_rejected_unknown_peer() {
        let (mut rt, _) = make_runtime();
        // No contact at all for this peer
        rt.push_event(RuntimeEvent::ReplicaPushReceived {
            peer_identity: [0xFF; 16],
            cid: [0xBB; 32],
            data: vec![1, 2, 3],
        });
        assert_eq!(rt.replica_store.usage(&[0xFF; 16]), 0);
    }

    #[test]
    fn runtime_event_replica_variant_exists() {
        let _e = RuntimeEvent::ReplicaPushReceived {
            peer_identity: [0xAA; 16],
            cid: [0xBB; 32],
            data: vec![1, 2, 3],
        };
    }

    #[test]
    fn runtime_action_replica_push_variant_exists() {
        let _a = RuntimeAction::ReplicaPush {
            peer_identity: [0xAA; 16],
            cid: [0xBB; 32],
            data: vec![1, 2, 3],
        };
    }

    #[test]
    fn runtime_action_replica_pull_response_variant_exists() {
        let _a = RuntimeAction::ReplicaPullResponse {
            peer_identity: [0xAA; 16],
            cid: [0xBB; 32],
            data: vec![1, 2, 3],
        };
    }

    // ── PullWithToken tests ─────────────────────────────────────────────

    /// Helper: set up a runtime with a contact, replica, and cached pubkey
    /// for PullWithToken testing. Returns (runtime, identity, cid).
    fn setup_pull_with_token_runtime() -> (
        NodeRuntime<MemoryBookStore>,
        harmony_identity::PqPrivateIdentity,
        [u8; 32],
    ) {
        use harmony_content::replica::ReplicaStore;
        use rand::rngs::OsRng;

        let (mut rt, _) = make_runtime();

        // Generate a PQ identity for the content owner (issuer).
        let owner = harmony_identity::PqPrivateIdentity::generate(&mut OsRng);
        let owner_identity = owner.public_identity();
        let issuer_hash = owner_identity.address_hash;

        // Add contact with replication quota so we can store a replica.
        let contact = harmony_contacts::Contact {
            identity_hash: issuer_hash,
            display_name: None,
            peering: harmony_contacts::PeeringPolicy {
                enabled: true,
                priority: harmony_contacts::PeeringPriority::Normal,
            },
            added_at: 1000,
            last_seen: None,
            notes: None,
            addresses: vec![],
            replication: Some(harmony_contacts::ReplicationPolicy {
                quota_bytes: 100_000,
            }),
        };
        rt.contact_store_mut().add(contact).unwrap();

        // Store a replica via the event-driven path (exercises quota check).
        let cid = [0xCC; 32];
        let content = vec![0xDE, 0xAD, 0xBE, 0xEF];
        rt.push_event(RuntimeEvent::ReplicaPushReceived {
            peer_identity: issuer_hash,
            cid,
            data: content,
        });

        // Cache the owner's ML-DSA public key via the event-driven path,
        // exercising insert_pubkey_capped the same way production code does.
        let pubkey_bytes = owner_identity.verifying_key.as_bytes();
        rt.push_event(RuntimeEvent::PeerPublicKeyLearned {
            identity_hash: issuer_hash,
            dsa_pubkey: pubkey_bytes,
        });

        (rt, owner, cid)
    }

    #[test]
    fn pull_with_token_valid_serves_content() {
        use rand::rngs::OsRng;

        let (rt, owner, cid) = setup_pull_with_token_runtime();

        // Issue a valid token granting Content access to this CID.
        let requester_hash = [0x42; 16];
        let token = owner
            .issue_pq_root_token(
                &mut OsRng,
                &requester_hash,
                harmony_identity::CapabilityType::Content,
                &cid,
                0, // not_before
                0, // expires_at (0 = no expiry)
            )
            .unwrap();
        let token_bytes = token.to_bytes();

        let result = rt.handle_pull_with_token(requester_hash, cid, &token_bytes, 1_700_000_000);
        assert!(result.is_some(), "valid token should produce a response");
        match result.unwrap() {
            RuntimeAction::ReplicaPullResponse {
                peer_identity,
                cid: resp_cid,
                data,
            } => {
                assert_eq!(peer_identity, requester_hash);
                assert_eq!(resp_cid, cid);
                assert_eq!(data, vec![0xDE, 0xAD, 0xBE, 0xEF]);
            }
            other => panic!("expected ReplicaPullResponse, got {other:?}"),
        }
    }

    #[test]
    fn pull_with_token_expired_rejected() {
        use rand::rngs::OsRng;

        let (rt, owner, cid) = setup_pull_with_token_runtime();

        let requester_hash = [0x42; 16];
        // Token that expired in the past.
        let token = owner
            .issue_pq_root_token(
                &mut OsRng,
                &requester_hash,
                harmony_identity::CapabilityType::Content,
                &cid,
                0, // not_before
                1, // expires_at = 1 second after epoch (long expired)
            )
            .unwrap();
        let token_bytes = token.to_bytes();

        let result = rt.handle_pull_with_token(requester_hash, cid, &token_bytes, 1_700_000_000);
        assert!(result.is_none(), "expired token should be rejected");
    }

    #[test]
    fn pull_with_token_wrong_cid_rejected() {
        use rand::rngs::OsRng;

        let (rt, owner, cid) = setup_pull_with_token_runtime();

        let requester_hash = [0x42; 16];
        // Token for a different CID.
        let wrong_cid = [0xFF; 32];
        let token = owner
            .issue_pq_root_token(
                &mut OsRng,
                &requester_hash,
                harmony_identity::CapabilityType::Content,
                &wrong_cid,
                0,
                0,
            )
            .unwrap();
        let token_bytes = token.to_bytes();

        // Request with the actual CID, but token says different CID.
        let result = rt.handle_pull_with_token(requester_hash, cid, &token_bytes, 1_700_000_000);
        assert!(result.is_none(), "wrong CID should be rejected");
    }

    #[test]
    fn pull_with_token_unknown_issuer_rejected() {
        use rand::rngs::OsRng;

        let (mut rt, _owner, cid) = setup_pull_with_token_runtime();

        // Create a different identity not known to the runtime.
        let stranger = harmony_identity::PqPrivateIdentity::generate(&mut OsRng);
        let stranger_hash = stranger.public_identity().address_hash;

        let requester_hash = [0x42; 16];
        let token = stranger
            .issue_pq_root_token(
                &mut OsRng,
                &requester_hash,
                harmony_identity::CapabilityType::Content,
                &cid,
                0,
                0,
            )
            .unwrap();
        let token_bytes = token.to_bytes();

        // The stranger's pubkey was never cached (only the owner's was).
        // The token issuer is the stranger → unknown issuer → rejection.
        let result = rt.handle_pull_with_token(requester_hash, cid, &token_bytes, 1_700_000_000);
        assert!(result.is_none(), "unknown issuer should be rejected");
    }

    #[test]
    fn pull_with_token_no_replica_rejected() {
        use rand::rngs::OsRng;

        let (rt, owner, _cid) = setup_pull_with_token_runtime();

        // Use a CID that has no replica stored.
        let missing_cid = [0xDD; 32];

        let requester_hash = [0x42; 16];
        let token = owner
            .issue_pq_root_token(
                &mut OsRng,
                &requester_hash,
                harmony_identity::CapabilityType::Content,
                &missing_cid,
                0,
                0,
            )
            .unwrap();
        let token_bytes = token.to_bytes();

        // Even though the owner is known and pubkey is cached, no replica for this CID.
        let result =
            rt.handle_pull_with_token(requester_hash, missing_cid, &token_bytes, 1_700_000_000);
        assert!(result.is_none(), "missing replica should be rejected");
    }

    #[test]
    fn pull_with_token_wrong_audience_rejected() {
        use rand::rngs::OsRng;

        let (rt, owner, cid) = setup_pull_with_token_runtime();

        // Token issued to [0x42; 16] but presented by [0x99; 16].
        let intended_audience = [0x42u8; 16];
        let actual_requester = [0x99u8; 16];
        let token = owner
            .issue_pq_root_token(
                &mut OsRng,
                &intended_audience,
                harmony_identity::CapabilityType::Content,
                &cid,
                0,
                0,
            )
            .unwrap();
        let token_bytes = token.to_bytes();

        let result = rt.handle_pull_with_token(actual_requester, cid, &token_bytes, 1_700_000_000);
        assert!(
            result.is_none(),
            "token with wrong audience should be rejected"
        );
    }

    // ── Memo filter tests ───────────────────────────────────────────────

    #[test]
    fn peer_memo_filter_upsert_and_query() {
        let mut table = PeerFilterTable::new(100);
        let mut filter = BloomFilter::new(1000, 0.01);
        let cid_in = ContentId::for_book(b"memo-present", ContentFlags::default()).unwrap();
        let cid_out = ContentId::for_book(b"memo-absent", ContentFlags::default()).unwrap();
        filter.insert(&cid_in);
        table.upsert_memo("peer-m".into(), filter, 10);

        assert!(
            table.should_query_memo("peer-m", &cid_in, 10),
            "filter should say 'maybe' for inserted CID"
        );
        assert!(
            !table.should_query_memo("peer-m", &cid_out, 10),
            "filter should say 'definitely not' for absent CID"
        );
    }

    #[test]
    fn pull_with_token_bad_signature_rejected() {
        use rand::rngs::OsRng;

        let (rt, owner, cid) = setup_pull_with_token_runtime();

        let requester_hash = [0x42; 16];
        let token = owner
            .issue_pq_root_token(
                &mut OsRng,
                &requester_hash,
                harmony_identity::CapabilityType::Content,
                &cid,
                0,
                0,
            )
            .unwrap();
        let mut token_bytes = token.to_bytes();

        // Corrupt a byte in the signature portion (last bytes of the token).
        // ML-DSA-65 signatures are 3309 bytes — flipping a bit in the
        // signature makes it invalid without affecting deserialization.
        let len = token_bytes.len();
        token_bytes[len - 1] ^= 0xFF;

        let result = rt.handle_pull_with_token(requester_hash, cid, &token_bytes, 1_700_000_000);
        assert!(
            result.is_none(),
            "token with corrupted signature should be rejected"
        );
    }

    #[test]
    fn pull_with_token_not_yet_valid_rejected() {
        use rand::rngs::OsRng;

        let (rt, owner, cid) = setup_pull_with_token_runtime();
        let requester_hash = [0x42; 16];
        // not_before is far in the future — token is not yet valid.
        let token = owner
            .issue_pq_root_token(
                &mut OsRng,
                &requester_hash,
                harmony_identity::CapabilityType::Content,
                &cid,
                u64::MAX, // not_before — will never be valid
                0,
            )
            .unwrap();
        let token_bytes = token.to_bytes();
        let result = rt.handle_pull_with_token(requester_hash, cid, &token_bytes, 1_700_000_000);
        assert!(
            result.is_none(),
            "token with future not_before should be rejected"
        );
    }

    #[test]
    fn peer_memo_filter_staleness() {
        let mut table = PeerFilterTable::new(10);
        let filter = BloomFilter::new(1000, 0.01);
        // Insert an empty filter at tick 5.
        table.upsert_memo("peer-s".into(), filter, 5);

        let cid = ContentId::for_book(b"stale-test", ContentFlags::default()).unwrap();

        // Fresh filter with no items => definite miss, should NOT query.
        assert!(
            !table.should_query_memo("peer-s", &cid, 5),
            "fresh empty memo filter should say 'definitely not'"
        );

        // Stale filter (current_tick - received_tick > staleness_ticks) => should query.
        assert!(
            table.should_query_memo("peer-s", &cid, 20),
            "stale memo filter should fall back to querying"
        );
    }

    // ── Page filter tests ────────────────────────────────────────────────

    #[test]
    fn peer_page_filter_upsert_and_query() {
        let mut table = PeerFilterTable::new(100);
        let mut bf = BloomFilter::new(100, 0.01);
        // hash_bits must fit in 28 bits (≤ 0x0FFF_FFFF)
        let addr =
            harmony_athenaeum::PageAddr::new(0x0EADBEE, harmony_athenaeum::Algorithm::Sha256Msb);
        bf.insert_bytes(&addr.to_bytes());

        table.upsert_page("peer-a".to_string(), bf, 10);

        // Should find the inserted address
        assert!(table.should_query_page("peer-a", &addr, 10));
        // Should NOT find a different address
        let other =
            harmony_athenaeum::PageAddr::new(0x0AFEBAB, harmony_athenaeum::Algorithm::Sha256Msb);
        assert!(!table.should_query_page("peer-a", &other, 10));
        // Unknown peer → always query
        assert!(table.should_query_page("peer-b", &addr, 10));
    }

    #[test]
    fn peer_page_filter_staleness() {
        let mut table = PeerFilterTable::new(100);
        let mut bf = BloomFilter::new(100, 0.01);
        // hash_bits must fit in 28 bits (≤ 0x0FFF_FFFF)
        let addr =
            harmony_athenaeum::PageAddr::new(0x0EADBEE, harmony_athenaeum::Algorithm::Sha256Msb);
        bf.insert_bytes(&addr.to_bytes());
        table.upsert_page("peer-a".to_string(), bf, 10);

        // At tick 10 + 101 = 111, the filter is stale → should query even for absent items
        let absent =
            harmony_athenaeum::PageAddr::new(0x00000000, harmony_athenaeum::Algorithm::Sha256Msb);
        assert!(table.should_query_page("peer-a", &absent, 111));
    }

    #[test]
    fn memo_queryable_serves_local_memos() {
        use harmony_identity::PqPrivateIdentity;
        use rand::rngs::OsRng;

        let (mut rt, _) = make_runtime();
        let identity = PqPrivateIdentity::generate(&mut OsRng);
        let input = ContentId::from_bytes([0x11; 32]);
        let output = ContentId::from_bytes([0x22; 32]);

        let memo =
            harmony_memo::create::create_memo(input, output, &identity, &mut OsRng, 1000, 2000)
                .unwrap();
        rt.memo_store_mut().insert(memo);

        // Simulate a query for this input
        let input_hex = hex::encode(input.to_bytes());
        let key_expr = format!("harmony/memo/{input_hex}/**");
        rt.push_event(RuntimeEvent::QueryReceived {
            query_id: 42,
            key_expr,
            payload: vec![],
        });

        let actions = rt.tick();
        let reply = actions
            .iter()
            .find(|a| matches!(a, RuntimeAction::SendReply { query_id: 42, .. }));
        assert!(reply.is_some(), "should reply to memo query");

        // Parse response format: [u16 LE count][u32 LE len][bytes]...
        if let Some(RuntimeAction::SendReply { payload, .. }) = reply {
            assert!(payload.len() >= 2, "response too short");
            let count = u16::from_le_bytes([payload[0], payload[1]]) as usize;
            assert_eq!(count, 1, "should have 1 memo");
            // Verify the memo can be deserialized
            let memo_len =
                u32::from_le_bytes([payload[2], payload[3], payload[4], payload[5]]) as usize;
            let memo_bytes = &payload[6..6 + memo_len];
            let restored = harmony_memo::deserialize(memo_bytes).expect("should deserialize");
            assert_eq!(restored.input, input);
            assert_eq!(restored.output, output);
        }
    }

    #[test]
    fn memo_queryable_empty_no_reply() {
        let (mut rt, _) = make_runtime();

        let input_hex = hex::encode([0xFF; 32]);
        let key_expr = format!("harmony/memo/{input_hex}/**");
        rt.push_event(RuntimeEvent::QueryReceived {
            query_id: 99,
            key_expr,
            payload: vec![],
        });

        let actions = rt.tick();
        let reply = actions
            .iter()
            .find(|a| matches!(a, RuntimeAction::SendReply { query_id: 99, .. }));
        assert!(reply.is_none(), "should not reply when no memos exist");
    }

    #[test]
    fn memo_queryable_caps_response_count() {
        use harmony_identity::PqPrivateIdentity;
        use rand::rngs::OsRng;

        let (mut rt, _) = make_runtime();
        let input = ContentId::from_bytes([0x11; 32]);

        // Insert MAX_MEMO_RESPONSE_COUNT + 10 memos with different signers
        for _ in 0..(MAX_MEMO_RESPONSE_COUNT + 10) {
            let identity = PqPrivateIdentity::generate(&mut OsRng);
            let output = ContentId::from_bytes([0x22; 32]);
            let memo =
                harmony_memo::create::create_memo(input, output, &identity, &mut OsRng, 1000, 2000)
                    .unwrap();
            rt.memo_store_mut().insert(memo);
        }

        let input_hex = hex::encode(input.to_bytes());
        let key_expr = format!("harmony/memo/{input_hex}/**");
        rt.push_event(RuntimeEvent::QueryReceived {
            query_id: 50,
            key_expr,
            payload: vec![],
        });

        let actions = rt.tick();
        let reply = actions
            .iter()
            .find(|a| matches!(a, RuntimeAction::SendReply { query_id: 50, .. }));
        assert!(reply.is_some(), "should reply");

        if let Some(RuntimeAction::SendReply { payload, .. }) = reply {
            let count = u16::from_le_bytes([payload[0], payload[1]]) as usize;
            assert_eq!(
                count, MAX_MEMO_RESPONSE_COUNT,
                "should cap at MAX_MEMO_RESPONSE_COUNT"
            );
        }
    }

    #[test]
    fn page_filter_broadcast_on_timer() {
        use harmony_athenaeum::{Book, PAGE_SIZE};
        use harmony_content::book::MemoryBookStore;

        // Use a small interval so the timer fires on tick 2.
        let config = NodeConfig {
            storage_budget: StorageBudget {
                cache_capacity: 100,
                max_pinned_bytes: 1_000_000,
            },
            compute_budget: InstructionBudget { fuel: 1000 },
            schedule: Default::default(),
            content_policy: ContentPolicy::default(),
            filter_broadcast_config: FilterBroadcastConfig {
                max_interval_ticks: 2,
                ..FilterBroadcastConfig::default()
            },
            node_addr: "page-filter-timer-test".to_string(),
            local_identity_hash: [0u8; 16],
            local_pq_identity_hash: [0u8; 16],
            local_dsa_pubkey: Vec::new(),
            inference_gguf_cid: None,
            inference_tokenizer_cid: None,
            disk_enabled: false,
            disk_entries: Vec::new(),
            disk_quota: None,
            s3_enabled: false,
        };
        let (mut rt, _) = NodeRuntime::new(config, MemoryBookStore::new());

        // Build a book and index it into the page index.
        let data = vec![0xAAu8; PAGE_SIZE];
        let cid_bytes = [0x11u8; 32];
        let book = Book::from_book(cid_bytes, &data).unwrap();
        let cid = harmony_content::ContentId::from_bytes(cid_bytes);
        rt.page_index.insert_book(cid, &book);

        // Tick 1: timer not yet due.
        rt.tick();

        // Tick 2: timer fires (ticks_since >= interval 2).
        let actions = rt.tick();

        // Expect exactly one page filter Publish action.
        let page_publishes: Vec<_> = actions
            .iter()
            .filter(|a| {
                matches!(a, RuntimeAction::Publish { key_expr, .. }
                    if key_expr.starts_with("harmony/filters/page/"))
            })
            .collect();
        assert_eq!(
            page_publishes.len(),
            1,
            "expected one page filter broadcast"
        );
    }

    // ── Memo fetch request tests ──────────────────────────────────────

    #[test]
    fn memo_fetch_request_local_short_circuit() {
        use harmony_identity::PqPrivateIdentity;
        use rand::rngs::OsRng;

        let (mut rt, _) = make_runtime();
        let identity = PqPrivateIdentity::generate(&mut OsRng);
        let input = ContentId::from_bytes([0x11; 32]);
        let output = ContentId::from_bytes([0x22; 32]);

        let memo =
            harmony_memo::create::create_memo(input, output, &identity, &mut OsRng, 1000, 2000)
                .unwrap();
        rt.memo_store_mut().insert(memo);

        // Request memos for an input we already have locally
        rt.push_event(RuntimeEvent::MemoFetchRequest { input });
        let actions = rt.tick();

        // Should NOT emit QueryMemo — local data is sufficient
        assert!(
            !actions
                .iter()
                .any(|a| matches!(a, RuntimeAction::QueryMemo { .. })),
            "should not query when memos exist locally"
        );
    }

    #[test]
    fn memo_fetch_request_emits_query() {
        let (mut rt, _) = make_runtime();
        let input = ContentId::from_bytes([0x11; 32]);

        rt.push_event(RuntimeEvent::MemoFetchRequest { input });
        let actions = rt.tick();

        let query = actions
            .iter()
            .find(|a| matches!(a, RuntimeAction::QueryMemo { .. }));
        assert!(query.is_some(), "should emit QueryMemo for unknown input");

        if let Some(RuntimeAction::QueryMemo { key_expr }) = query {
            let expected = format!("harmony/memo/{}/**", hex::encode(input.to_bytes()));
            assert_eq!(key_expr, &expected);
        }
    }

    #[test]
    fn memo_fetch_request_dedup() {
        let (mut rt, _) = make_runtime();
        let input = ContentId::from_bytes([0x11; 32]);

        // First request → should emit query
        rt.push_event(RuntimeEvent::MemoFetchRequest { input });
        let actions1 = rt.tick();
        assert!(
            actions1
                .iter()
                .any(|a| matches!(a, RuntimeAction::QueryMemo { .. })),
            "first request should emit query"
        );

        // Second request for same input → should NOT emit query (in-flight)
        rt.push_event(RuntimeEvent::MemoFetchRequest { input });
        let actions2 = rt.tick();
        assert!(
            !actions2
                .iter()
                .any(|a| matches!(a, RuntimeAction::QueryMemo { .. })),
            "duplicate request should be suppressed"
        );
    }

    #[test]
    fn memo_fetch_timeout_clears_pending() {
        let (mut rt, _) = make_runtime();
        let input = ContentId::from_bytes([0x11; 32]);

        // Issue a fetch
        rt.push_event(RuntimeEvent::MemoFetchRequest { input });
        rt.tick();

        // Advance past timeout
        for _ in 0..MEMO_FETCH_TIMEOUT_TICKS {
            rt.push_event(RuntimeEvent::TimerTick {
                now: 0,
                unix_now: 0,
            });
            rt.tick();
        }

        // Should be able to re-fetch now
        rt.push_event(RuntimeEvent::MemoFetchRequest { input });
        let actions = rt.tick();
        assert!(
            actions
                .iter()
                .any(|a| matches!(a, RuntimeAction::QueryMemo { .. })),
            "should re-emit query after timeout"
        );
    }

    // ── Memo fetch response tests ─────────────────────────────────────

    #[test]
    fn memo_fetch_response_inserts_verified_memos() {
        use harmony_identity::PqPrivateIdentity;
        use rand::rngs::OsRng;

        let (mut rt, _) = make_runtime();
        let identity = PqPrivateIdentity::generate(&mut OsRng);
        let pub_id = identity.public_identity();
        let input = ContentId::from_bytes([0x11; 32]);
        let output = ContentId::from_bytes([0x22; 32]);

        // Cache the signer's public key via the production code path
        let id_ref = harmony_identity::IdentityRef::from(pub_id);
        rt.push_event(RuntimeEvent::PeerPublicKeyLearned {
            identity_hash: id_ref.hash,
            dsa_pubkey: pub_id.verifying_key.as_bytes().to_vec(),
        });

        // Build a valid response payload
        let memo =
            harmony_memo::create::create_memo(input, output, &identity, &mut OsRng, 1000, 2000)
                .unwrap();
        let memo_bytes = harmony_memo::serialize(&memo).unwrap();

        let mut payload = Vec::new();
        payload.extend_from_slice(&1u16.to_le_bytes()); // count = 1
        payload.extend_from_slice(&(memo_bytes.len() as u32).to_le_bytes());
        payload.extend_from_slice(&memo_bytes);

        let input_hex = hex::encode(input.to_bytes());
        let key_expr = format!("harmony/memo/{input_hex}/**");
        rt.push_event(RuntimeEvent::MemoFetchResponse {
            key_expr,
            payload,
            unix_now: 1500,
        });
        rt.tick();

        // Memo should now be in the store
        let stored = rt.memo_store().get_by_input(&input);
        assert_eq!(stored.len(), 1);
        assert_eq!(stored[0].output, output);
    }

    #[test]
    fn memo_fetch_response_rejects_invalid_memos() {
        let (mut rt, _) = make_runtime();
        let input = ContentId::from_bytes([0x11; 32]);

        // Build a payload with garbage memo bytes
        let garbage = vec![0xFF; 100];
        let mut payload = Vec::new();
        payload.extend_from_slice(&1u16.to_le_bytes()); // count = 1
        payload.extend_from_slice(&(garbage.len() as u32).to_le_bytes());
        payload.extend_from_slice(&garbage);

        let input_hex = hex::encode(input.to_bytes());
        let key_expr = format!("harmony/memo/{input_hex}/**");
        rt.push_event(RuntimeEvent::MemoFetchResponse {
            key_expr,
            payload,
            unix_now: 1500,
        });
        rt.tick();

        // No memo should be stored
        assert!(rt.memo_store().get_by_input(&input).is_empty());
    }

    #[test]
    fn memo_fetch_response_oversized_rejected() {
        let (mut rt, _) = make_runtime();
        let input = ContentId::from_bytes([0x11; 32]);

        // Build a payload that exceeds MAX_MEMO_RESPONSE_BYTES
        let payload = vec![0u8; MAX_MEMO_RESPONSE_BYTES + 1];

        let input_hex = hex::encode(input.to_bytes());
        let key_expr = format!("harmony/memo/{input_hex}/**");
        rt.push_event(RuntimeEvent::MemoFetchResponse {
            key_expr,
            payload,
            unix_now: 1500,
        });
        rt.tick();

        assert!(rt.memo_store().get_by_input(&input).is_empty());
    }

    #[test]
    fn memo_fetch_multiple_responses_all_inserted() {
        use harmony_identity::PqPrivateIdentity;
        use rand::rngs::OsRng;

        let (mut rt, _) = make_runtime();
        let input = ContentId::from_bytes([0x11; 32]);

        // Two different signers
        let alice = PqPrivateIdentity::generate(&mut OsRng);
        let bob = PqPrivateIdentity::generate(&mut OsRng);

        let alice_pub = alice.public_identity();
        let bob_pub = bob.public_identity();

        // Cache both public keys via the production code path
        let alice_ref = harmony_identity::IdentityRef::from(alice_pub);
        rt.push_event(RuntimeEvent::PeerPublicKeyLearned {
            identity_hash: alice_ref.hash,
            dsa_pubkey: alice_pub.verifying_key.as_bytes().to_vec(),
        });
        let bob_ref = harmony_identity::IdentityRef::from(bob_pub);
        rt.push_event(RuntimeEvent::PeerPublicKeyLearned {
            identity_hash: bob_ref.hash,
            dsa_pubkey: bob_pub.verifying_key.as_bytes().to_vec(),
        });

        let output = ContentId::from_bytes([0x22; 32]);
        let input_hex = hex::encode(input.to_bytes());
        let key_expr = format!("harmony/memo/{input_hex}/**");

        // First response: Alice's memo
        let memo_alice =
            harmony_memo::create::create_memo(input, output, &alice, &mut OsRng, 1000, 2000)
                .unwrap();
        let alice_bytes = harmony_memo::serialize(&memo_alice).unwrap();
        let mut payload1 = Vec::new();
        payload1.extend_from_slice(&1u16.to_le_bytes());
        payload1.extend_from_slice(&(alice_bytes.len() as u32).to_le_bytes());
        payload1.extend_from_slice(&alice_bytes);

        rt.push_event(RuntimeEvent::MemoFetchResponse {
            key_expr: key_expr.clone(),
            payload: payload1,
            unix_now: 1500,
        });
        rt.tick();

        // Second response: Bob's memo
        let memo_bob =
            harmony_memo::create::create_memo(input, output, &bob, &mut OsRng, 1000, 2000).unwrap();
        let bob_bytes = harmony_memo::serialize(&memo_bob).unwrap();
        let mut payload2 = Vec::new();
        payload2.extend_from_slice(&1u16.to_le_bytes());
        payload2.extend_from_slice(&(bob_bytes.len() as u32).to_le_bytes());
        payload2.extend_from_slice(&bob_bytes);

        rt.push_event(RuntimeEvent::MemoFetchResponse {
            key_expr,
            payload: payload2,
            unix_now: 1500,
        });
        rt.tick();

        // Both memos should be in the store
        let stored = rt.memo_store().get_by_input(&input);
        assert_eq!(
            stored.len(),
            2,
            "both Alice's and Bob's memos should be stored"
        );
    }

    #[test]
    fn discover_query_missing_token_returns_public_hints() {
        let (mut rt, _) = make_runtime();
        let public_data = b"public-announce-record".to_vec();
        let full_data = b"full-announce-record-with-tunnel".to_vec();
        rt.set_local_public_announce(public_data.clone());
        rt.set_local_full_announce(full_data.clone());

        let identity_hex = hex::encode(rt.local_pq_identity_hash());
        let key_expr = harmony_zenoh::namespace::discover::key(&identity_hex);
        rt.push_event(RuntimeEvent::QueryReceived {
            query_id: 42,
            key_expr,
            payload: vec![],
        });
        let actions = rt.tick();
        let reply = actions
            .iter()
            .find(|a| matches!(a, RuntimeAction::SendReply { query_id: 42, .. }));
        assert!(reply.is_some(), "should reply with public hints");
        if let Some(RuntimeAction::SendReply { payload, .. }) = reply {
            assert_eq!(payload, &public_data, "should return public record");
        }
    }

    #[test]
    fn discover_query_valid_token_returns_full_hints() {
        use harmony_identity::PqPrivateIdentity;
        use rand::rngs::OsRng;

        let identity = PqPrivateIdentity::generate(&mut OsRng);
        let pub_id = identity.public_identity();
        let id_ref = harmony_identity::IdentityRef::from(pub_id);

        let mut config = NodeConfig::default();
        config.local_identity_hash = id_ref.hash;
        config.local_pq_identity_hash = id_ref.hash;
        config.local_dsa_pubkey = pub_id.verifying_key.as_bytes();
        config.node_addr = hex::encode(id_ref.hash);
        let (mut rt, _) = NodeRuntime::new(config, MemoryBookStore::new());

        let public_data = b"public-only".to_vec();
        let full_data = b"full-with-tunnel".to_vec();
        rt.set_local_public_announce(public_data.clone());
        rt.set_local_full_announce(full_data.clone());

        rt.push_event(RuntimeEvent::TimerTick {
            now: 1000,
            unix_now: 1500,
        });
        rt.tick();

        let peer_hash = [0xBBu8; 16];
        let token = identity
            .issue_pq_root_token(
                &mut OsRng,
                &peer_hash,
                harmony_identity::CapabilityType::Discovery,
                &id_ref.hash,
                1000,
                2000,
            )
            .unwrap();
        let token_bytes = token.to_bytes();

        let identity_hex = hex::encode(id_ref.hash);
        let key_expr = harmony_zenoh::namespace::discover::key(&identity_hex);
        rt.push_event(RuntimeEvent::QueryReceived {
            query_id: 43,
            key_expr,
            payload: token_bytes,
        });
        let actions = rt.tick();
        let reply = actions
            .iter()
            .find(|a| matches!(a, RuntimeAction::SendReply { query_id: 43, .. }));
        assert!(reply.is_some(), "should reply with full hints");
        if let Some(RuntimeAction::SendReply { payload, .. }) = reply {
            assert_eq!(payload, &full_data, "should return full record");
        }
    }

    #[test]
    fn discover_query_expired_token_returns_public_hints() {
        use harmony_identity::PqPrivateIdentity;
        use rand::rngs::OsRng;

        let identity = PqPrivateIdentity::generate(&mut OsRng);
        let pub_id = identity.public_identity();
        let id_ref = harmony_identity::IdentityRef::from(pub_id);

        let mut config = NodeConfig::default();
        config.local_identity_hash = id_ref.hash;
        config.local_pq_identity_hash = id_ref.hash;
        config.local_dsa_pubkey = pub_id.verifying_key.as_bytes();
        config.node_addr = hex::encode(id_ref.hash);
        let (mut rt, _) = NodeRuntime::new(config, MemoryBookStore::new());

        let public_data = b"public-only".to_vec();
        rt.set_local_public_announce(public_data.clone());
        rt.set_local_full_announce(b"full".to_vec());

        rt.push_event(RuntimeEvent::TimerTick {
            now: 1000,
            unix_now: 3000,
        });
        rt.tick();

        let peer_hash = [0xBBu8; 16];
        let token = identity
            .issue_pq_root_token(
                &mut OsRng,
                &peer_hash,
                harmony_identity::CapabilityType::Discovery,
                &id_ref.hash,
                1000,
                2000,
            )
            .unwrap();
        let token_bytes = token.to_bytes();

        let identity_hex = hex::encode(id_ref.hash);
        let key_expr = harmony_zenoh::namespace::discover::key(&identity_hex);
        rt.push_event(RuntimeEvent::QueryReceived {
            query_id: 44,
            key_expr,
            payload: token_bytes,
        });
        let actions = rt.tick();
        let reply = actions
            .iter()
            .find(|a| matches!(a, RuntimeAction::SendReply { query_id: 44, .. }));
        assert!(reply.is_some());
        if let Some(RuntimeAction::SendReply { payload, .. }) = reply {
            assert_eq!(payload, &public_data, "expired token → public record");
        }
    }

    #[test]
    fn discover_query_wrong_capability_returns_public_hints() {
        use harmony_identity::PqPrivateIdentity;
        use rand::rngs::OsRng;

        let identity = PqPrivateIdentity::generate(&mut OsRng);
        let pub_id = identity.public_identity();
        let id_ref = harmony_identity::IdentityRef::from(pub_id);

        let mut config = NodeConfig::default();
        config.local_identity_hash = id_ref.hash;
        config.local_pq_identity_hash = id_ref.hash;
        config.local_dsa_pubkey = pub_id.verifying_key.as_bytes();
        config.node_addr = hex::encode(id_ref.hash);
        let (mut rt, _) = NodeRuntime::new(config, MemoryBookStore::new());

        let public_data = b"public-only".to_vec();
        rt.set_local_public_announce(public_data.clone());
        rt.set_local_full_announce(b"full".to_vec());

        rt.push_event(RuntimeEvent::TimerTick {
            now: 1000,
            unix_now: 1500,
        });
        rt.tick();

        let peer_hash = [0xBBu8; 16];
        let token = identity
            .issue_pq_root_token(
                &mut OsRng,
                &peer_hash,
                harmony_identity::CapabilityType::Content, // WRONG
                &id_ref.hash,
                1000,
                2000,
            )
            .unwrap();
        let token_bytes = token.to_bytes();

        let identity_hex = hex::encode(id_ref.hash);
        let key_expr = harmony_zenoh::namespace::discover::key(&identity_hex);
        rt.push_event(RuntimeEvent::QueryReceived {
            query_id: 45,
            key_expr,
            payload: token_bytes,
        });
        let actions = rt.tick();
        let reply = actions
            .iter()
            .find(|a| matches!(a, RuntimeAction::SendReply { query_id: 45, .. }));
        assert!(reply.is_some());
        if let Some(RuntimeAction::SendReply { payload, .. }) = reply {
            assert_eq!(payload, &public_data, "wrong capability → public record");
        }
    }

    #[test]
    fn discover_query_wrong_identity_no_reply() {
        let (mut rt, _) = make_runtime();
        rt.set_local_public_announce(b"public".to_vec());
        rt.set_local_full_announce(b"full".to_vec());

        let wrong_hex = hex::encode([0xFFu8; 16]);
        let key_expr = harmony_zenoh::namespace::discover::key(&wrong_hex);
        rt.push_event(RuntimeEvent::QueryReceived {
            query_id: 46,
            key_expr,
            payload: vec![],
        });
        let actions = rt.tick();
        let reply = actions
            .iter()
            .find(|a| matches!(a, RuntimeAction::SendReply { query_id: 46, .. }));
        assert!(reply.is_none(), "should not reply for unknown identity");
    }

    #[test]
    fn discover_query_invalid_token_returns_public_hints() {
        let (mut rt, _) = make_runtime();
        let public_data = b"public-announce-record".to_vec();
        rt.set_local_public_announce(public_data.clone());
        rt.set_local_full_announce(b"full".to_vec());

        let identity_hex = hex::encode(rt.local_pq_identity_hash());
        let key_expr = harmony_zenoh::namespace::discover::key(&identity_hex);
        rt.push_event(RuntimeEvent::QueryReceived {
            query_id: 47,
            key_expr,
            payload: vec![0xFF; 100], // garbage
        });
        let actions = rt.tick();
        let reply = actions
            .iter()
            .find(|a| matches!(a, RuntimeAction::SendReply { query_id: 47, .. }));
        assert!(reply.is_some());
        if let Some(RuntimeAction::SendReply { payload, .. }) = reply {
            assert_eq!(payload, &public_data, "invalid token → public record");
        }
    }

    // ── Tier 3: Inference integration tests ─────────────────────────

    #[test]
    fn inference_query_no_model_returns_error() {
        let (mut rt, _) = make_runtime();

        // Build an inference query payload: [0x02] [prompt_len: u32 LE] [prompt_utf8]
        let prompt = b"Hello";
        let mut payload = vec![crate::inference_types::INFERENCE_TAG];
        payload.extend_from_slice(&(prompt.len() as u32).to_le_bytes());
        payload.extend_from_slice(prompt);

        rt.push_event(RuntimeEvent::ComputeQuery {
            query_id: 500,
            key_expr: "harmony/compute/activity/inference".into(),
            payload,
        });

        let actions = rt.tick();
        let reply = actions
            .iter()
            .find(|a| matches!(a, RuntimeAction::SendReply { query_id: 500, .. }));
        assert!(
            reply.is_some(),
            "inference query without model should produce error reply"
        );
        if let Some(RuntimeAction::SendReply { payload, .. }) = reply {
            // With inference feature: AgentResult wire format (0x00 + JSON) with "no inference model loaded".
            // Without inference feature: "inference feature not enabled" message.
            // In both cases, verify that an error is surfaced.
            let payload_str = String::from_utf8_lossy(payload);
            assert!(
                payload_str.contains("no inference model loaded")
                    || payload_str.contains("inference feature not enabled")
                    || payload.is_empty(),
                "error reply should mention no model or feature not enabled, got: {payload_str}"
            );
        }
    }

    #[test]
    #[cfg(feature = "inference")]
    fn inference_startup_emits_fetch_for_model_cids() {
        let mut config = NodeConfig::default();
        config.inference_gguf_cid = Some([0xAA; 32]);
        config.inference_tokenizer_cid = Some([0xBB; 32]);

        let (_rt, actions) = NodeRuntime::new(config, MemoryBookStore::new());

        let fetch_cids: Vec<[u8; 32]> = actions
            .iter()
            .filter_map(|a| match a {
                RuntimeAction::FetchContent { cid } => Some(*cid),
                _ => None,
            })
            .collect();

        assert!(
            fetch_cids.contains(&[0xAA; 32]),
            "should fetch GGUF CID at startup"
        );
        assert!(
            fetch_cids.contains(&[0xBB; 32]),
            "should fetch tokenizer CID at startup"
        );
    }

    #[test]
    fn inference_startup_no_fetch_without_cids() {
        let (_rt, actions) = make_runtime();

        let fetch_count = actions
            .iter()
            .filter(|a| matches!(a, RuntimeAction::FetchContent { .. }))
            .count();

        assert_eq!(
            fetch_count, 0,
            "default config should not emit FetchContent for inference"
        );
    }

    #[test]
    fn parse_compute_payload_inference() {
        let (rt, _) = make_runtime();

        let prompt = b"test prompt";
        let mut payload = vec![crate::inference_types::INFERENCE_TAG];
        payload.extend_from_slice(&(prompt.len() as u32).to_le_bytes());
        payload.extend_from_slice(prompt);

        let parsed = rt.parse_compute_payload(payload);
        assert!(
            matches!(parsed, Some(ParsedCompute::Inference { .. })),
            "should parse inference payload"
        );
    }

    #[test]
    fn parse_compute_payload_inference_invalid() {
        let (rt, _) = make_runtime();

        // Tag 0x02 but too short for prompt length
        let payload = vec![crate::inference_types::INFERENCE_TAG, 0xFF, 0, 0, 0];
        let parsed = rt.parse_compute_payload(payload);
        assert!(
            parsed.is_none(),
            "truncated inference payload should return None"
        );
    }

    #[test]
    fn parse_compute_payload_verify() {
        let (rt, _) = make_runtime();

        let request = harmony_speculative::VerifyRequest {
            context_tokens: vec![1, 2, 3],
            drafts: vec![
                harmony_speculative::DraftEntry {
                    token_id: 10,
                    logprob: -0.5,
                },
                harmony_speculative::DraftEntry {
                    token_id: 20,
                    logprob: -1.0,
                },
            ],
        };
        let payload = request.serialize();
        let parsed = rt.parse_compute_payload(payload);
        assert!(
            matches!(parsed, Some(ParsedCompute::Verify { .. })),
            "should parse verify payload"
        );
    }

    #[test]
    fn parse_compute_payload_verify_invalid() {
        let (rt, _) = make_runtime();
        // Tag 0x04 but truncated
        let payload = vec![0x04, 0x01];
        let parsed = rt.parse_compute_payload(payload);
        assert!(
            parsed.is_none(),
            "truncated verify payload should return None"
        );
    }

    #[test]
    fn verify_query_without_engine_returns_error() {
        let (mut rt, _) = make_runtime();

        let request = harmony_speculative::VerifyRequest {
            context_tokens: vec![1, 2, 3],
            drafts: vec![harmony_speculative::DraftEntry {
                token_id: 10,
                logprob: -0.5,
            }],
        };
        let payload = request.serialize();

        rt.push_event(RuntimeEvent::ComputeQuery {
            query_id: 600,
            key_expr: "harmony/compute/activity/verify".into(),
            payload,
        });

        let actions = rt.tick();
        let reply = actions
            .iter()
            .find(|a| matches!(a, RuntimeAction::SendReply { query_id: 600, .. }));
        assert!(
            reply.is_some(),
            "verify query without engine should produce error reply"
        );
        if let Some(RuntimeAction::SendReply { payload, .. }) = reply {
            // Error response starts with 0x01 tag
            assert_eq!(payload[0], 0x01, "error tag");
        }
    }

    #[test]
    fn runtime_event_verify_response_exists() {
        let _e = RuntimeEvent::VerifyResponse {
            payload: vec![0x00, 1, 0, 0, 0, 42, 0, 0, 0, 0],
        };
    }

    #[test]
    fn runtime_action_send_verify_query_exists() {
        let _a = RuntimeAction::SendVerifyQuery {
            key_expr: "harmony/compute/activity/verify".into(),
            payload: vec![0x04],
        };
    }

    #[test]
    fn capacity_sub_constant_matches_prefix() {
        assert!(
            harmony_zenoh::namespace::compute::CAPACITY_SUB
                .starts_with(harmony_zenoh::namespace::compute::CAPACITY),
            "CAPACITY_SUB should start with CAPACITY prefix"
        );
    }
}
