//! Node runtime: priority event loop wiring Tier 1 (Router) + Tier 2 (Storage) + Tier 3 (Compute).

// Remaining dead code (RuntimeEvent, push_event, tick, metrics, etc.) is consumed
// only by tests until the async event loop is wired.
#![allow(dead_code)]

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use harmony_compute::InstructionBudget;
use harmony_content::blob::BlobStore;
use harmony_content::bloom::BloomFilter;
use harmony_content::cid::ContentId;
use harmony_content::cuckoo::CuckooFilter;
use harmony_content::storage_tier::{
    ContentPolicy, FilterBroadcastConfig, StorageBudget, StorageMetrics, StorageTier,
    StorageTierAction, StorageTierEvent,
};
use harmony_reticulum::node::{Node, NodeAction, NodeEvent};
use harmony_workflow::{ComputeHint, WorkflowAction, WorkflowEngine, WorkflowEvent, WorkflowId};
use harmony_zenoh::namespace::content as content_ns;
use harmony_zenoh::queryable::{QueryableAction, QueryableEvent, QueryableId, QueryableRouter};

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
    TimerTick { now: u64 },
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
}

/// Outbound actions returned by the runtime for the caller to execute.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeAction {
    /// Tier 1: Send raw packet on a network interface.
    SendOnInterface {
        interface_name: Arc<str>,
        raw: Vec<u8>,
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
}

/// Filter state received from a peer, with metadata.
struct PeerFilter {
    content_filter: Option<BloomFilter>,
    flatpack_filter: Option<CuckooFilter>,
    received_tick: u64,
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
            received_tick: tick,
        });
        entry.content_filter = Some(filter);
        entry.received_tick = tick;
    }

    fn upsert_flatpack(&mut self, peer_addr: String, filter: CuckooFilter, tick: u64) {
        let entry = self.filters.entry(peer_addr).or_insert(PeerFilter {
            content_filter: None,
            flatpack_filter: None,
            received_tick: tick,
        });
        entry.flatpack_filter = Some(filter);
        entry.received_tick = tick;
    }

    /// Returns true if the peer should be queried (no filter, stale filter, or filter says "maybe").
    /// Returns false only if the filter definitively says the CID is absent.
    fn should_query(&self, peer_addr: &str, cid: &ContentId, current_tick: u64) -> bool {
        match self.filters.get(peer_addr) {
            None => true,
            Some(pf) => {
                if current_tick.saturating_sub(pf.received_tick) > self.staleness_ticks {
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
    fn should_query_flatpack(&self, peer_addr: &str, child_cid: &ContentId, current_tick: u64) -> bool {
        match self.filters.get(peer_addr) {
            None => true,
            Some(pf) => {
                if current_tick.saturating_sub(pf.received_tick) > self.staleness_ticks {
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

    fn evict_stale(&mut self, current_tick: u64) {
        self.filters
            .retain(|_, pf| current_tick.saturating_sub(pf.received_tick) <= self.staleness_ticks);
    }
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
pub struct NodeRuntime<B: BlobStore> {
    // Tier 1: Reticulum packet router
    router: Node,
    // Tier 1/2: Zenoh query dispatch
    queryable_router: QueryableRouter,
    // Tier 2: Content storage
    storage: StorageTier<B>,
    // Tier 3: Compute (workflow engine)
    workflow: WorkflowEngine,
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
    // Coalesces multiple BroadcastFilter actions within a single tick into one publish.
    pending_filter_broadcast: Option<Vec<u8>>,
    // Coalesces cuckoo filter broadcasts (flatpack reverse-index) within a single tick.
    pending_cuckoo_broadcast: Option<Vec<u8>>,
}

impl<B: BlobStore> NodeRuntime<B> {
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

        let filter_broadcast_interval_ticks = config.filter_broadcast_config.max_interval_ticks as u64;
        assert!(
            filter_broadcast_interval_ticks >= 2,
            "filter_broadcast_interval_ticks must be >= 2; \
             with interval=1 the timer fires every tick"
        );

        let (storage, storage_startup) =
            StorageTier::new(store, config.storage_budget, config.content_policy, config.filter_broadcast_config);

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

        // Subscribe to peer filter broadcasts.
        actions.push(RuntimeAction::Subscribe {
            key_expr: harmony_zenoh::namespace::filters::CONTENT_SUB.to_string(),
        });
        actions.push(RuntimeAction::Subscribe {
            key_expr: harmony_zenoh::namespace::filters::FLATPACK_SUB.to_string(),
        });

        let rt = Self {
            router,
            queryable_router,
            storage,
            workflow,
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
            pending_filter_broadcast: None,
            pending_cuckoo_broadcast: None,
        };

        (rt, actions)
    }

    /// Read-only access to the tier schedule configuration.
    pub fn schedule(&self) -> &TierSchedule {
        &self.schedule
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
            RuntimeEvent::TimerTick { now } => {
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
                let event = match result {
                    Ok(data) => WorkflowEvent::ContentFetched { cid, data },
                    Err(_) => WorkflowEvent::ContentFetchFailed { cid },
                };
                let actions = self.workflow.handle(event);
                self.pending_workflow_actions.extend(actions);
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
                    // Timer-triggered filter rebuild: fires only if the interval
                    // has elapsed AND no threshold-triggered broadcast already
                    // occurred during this tick's drain (which would have set
                    // pending_filter_broadcast and reset ticks_since_filter_broadcast
                    // via dispatch_storage_actions).
                    if self.ticks_since_filter_broadcast >= self.filter_broadcast_interval_ticks
                        && self.pending_filter_broadcast.is_none()
                    {
                        self.ticks_since_filter_broadcast = 0;
                        let timer_actions =
                            self.storage.handle(StorageTierEvent::FilterTimerTick);
                        self.dispatch_storage_actions(timer_actions, &mut actions);
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

        actions
    }

    fn dispatch_router_actions(
        &mut self,
        node_actions: Vec<NodeAction>,
        out: &mut Vec<RuntimeAction>,
    ) {
        for action in node_actions {
            if let NodeAction::SendOnInterface {
                interface_name,
                raw,
            } = action
            {
                out.push(RuntimeAction::SendOnInterface {
                    interface_name,
                    raw,
                });
            }
            // Other router actions are diagnostics -- drop for now.
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
                    out.push(RuntimeAction::Publish { key_expr, payload });
                }
                StorageTierAction::SendStatsReply { query_id, payload } => {
                    out.push(RuntimeAction::SendReply { query_id, payload });
                }
                // Startup-only declarations — already processed in new().
                StorageTierAction::DeclareQueryables { .. }
                | StorageTierAction::DeclareSubscribers { .. } => {}
                // Disk I/O actions — sans-I/O boundary. The runtime must wire
                // these to actual filesystem operations (not yet implemented).
                StorageTierAction::PersistToDisk { .. }
                | StorageTierAction::RemoveFromDisk { .. }
                | StorageTierAction::DiskLookup { .. } => {}
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

    /// Route a compute query by parsing its payload and submitting to the workflow engine.
    ///
    /// Returns `WorkflowAction`s to be buffered as pending. For CID-based requests,
    /// also emits `FetchModule` as a direct action (stored in `pending_direct_actions`).
    fn route_compute_query(
        &mut self,
        query_id: u64,
        _key_expr: String,
        payload: Vec<u8>,
    ) -> Vec<WorkflowAction> {
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
        for action in workflow_actions {
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
            _ => None,
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
                }
            }
        }
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_exposes_schedule() {
        use harmony_content::blob::MemoryBlobStore;

        let mut config = NodeConfig::default();
        config.schedule.router_max_per_tick = Some(5);
        let (rt, _) = NodeRuntime::new(config, MemoryBlobStore::new());
        assert_eq!(rt.schedule().router_max_per_tick, Some(5));
    }

    #[test]
    fn runtime_event_variants_exist() {
        let _e1 = RuntimeEvent::InboundPacket {
            interface_name: "lo".into(),
            raw: vec![0u8; 20],
            now: 1000,
        };
        let _e2 = RuntimeEvent::TimerTick { now: 1000 };
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
    }

    #[test]
    fn runtime_action_variants_exist() {
        let _a1 = RuntimeAction::SendOnInterface {
            interface_name: "lo".into(),
            raw: vec![0u8; 20],
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

    use harmony_content::blob::MemoryBlobStore;

    fn make_runtime() -> (NodeRuntime<MemoryBlobStore>, Vec<RuntimeAction>) {
        let config = NodeConfig::default();
        NodeRuntime::new(config, MemoryBlobStore::new())
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

        // 16 shard queryables + 1 stats queryable + 1 compute activity queryable = 18
        assert_eq!(queryable_count, 18);
        // transit + publish + content filter + flatpack filter subscriptions = 4
        assert_eq!(subscribe_count, 4);
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
        rt.push_event(RuntimeEvent::TimerTick { now: 1001 });
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
            rt.push_event(RuntimeEvent::TimerTick { now: 1000 + i });
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
        let cid = ContentId::for_blob(data, harmony_content::cid::ContentFlags::default()).unwrap();
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
            // 9 metrics × 8 bytes = 72 bytes
            assert_eq!(payload.len(), 72);
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

        let module = crate::compute::tests::ADD_WAT.as_bytes();
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
        rt.push_event(RuntimeEvent::TimerTick { now: 1000 });
        rt.push_event(RuntimeEvent::QueryReceived {
            query_id: 10,
            key_expr: "harmony/content/stats".into(),
            payload: vec![],
        });

        let module = crate::compute::tests::ADD_WAT.as_bytes();
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
        let module = crate::compute::tests::FETCH_WAT.as_bytes();
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
        let module = crate::compute::tests::FETCH_WAT.as_bytes();
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
        use harmony_content::blob::MemoryBlobStore;

        let mut config = NodeConfig::default();
        config.compute_budget = InstructionBudget { fuel: 1000 };
        config.schedule.adaptive_compute.high_water = 10;
        config.schedule.adaptive_compute.floor_fraction = 0.1;
        let (mut rt, _) = NodeRuntime::new(config, MemoryBlobStore::new());

        // Empty queues → full budget
        assert_eq!(rt.effective_fuel(), 1000);

        // Push 5 router events (half of high_water=10)
        for i in 0..5 {
            rt.push_event(RuntimeEvent::TimerTick { now: 1000 + i });
        }
        // load_factor = 5/10 = 0.5
        // effective = 1000 * (1.0 - 0.5 * 0.9) = 1000 * 0.55 = 550
        assert_eq!(rt.effective_fuel(), 550);

        // Push 5 more (at high_water)
        for i in 5..10 {
            rt.push_event(RuntimeEvent::TimerTick { now: 1000 + i });
        }
        // load_factor = 10/10 = 1.0 → floor
        // effective = 1000 * 0.1 = 100
        assert_eq!(rt.effective_fuel(), 100);

        // Push beyond high_water — stays at floor
        for i in 10..20 {
            rt.push_event(RuntimeEvent::TimerTick { now: 1000 + i });
        }
        assert_eq!(rt.effective_fuel(), 100);
    }

    #[test]
    fn router_max_per_tick_caps_drain() {
        let mut config = NodeConfig::default();
        config.schedule.router_max_per_tick = Some(2);
        let (mut rt, _) = NodeRuntime::new(config, MemoryBlobStore::new());

        // Push 5 router events
        for i in 0..5 {
            rt.push_event(RuntimeEvent::TimerTick { now: 1000 + i });
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
        let (mut rt, _) = NodeRuntime::new(config, MemoryBlobStore::new());

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
        let (mut rt, _) = NodeRuntime::new(config, MemoryBlobStore::new());

        // No events at all → all tiers idle
        rt.tick();
        assert_eq!(rt.starvation_counters(), (1, 1, 1));

        // Push router event only
        rt.push_event(RuntimeEvent::TimerTick { now: 1000 });
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
        let (mut rt, _) = NodeRuntime::new(config, MemoryBlobStore::new());

        // Push many router events but no storage events
        for i in 0..10 {
            rt.push_event(RuntimeEvent::TimerTick { now: 1000 + i });
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
        rt.push_event(RuntimeEvent::TimerTick { now: 2000 });

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
        let _ = NodeRuntime::new(config, MemoryBlobStore::new());
    }

    #[test]
    #[should_panic(expected = "storage_max_per_tick must be None or > 0")]
    fn zero_storage_limit_panics() {
        let mut config = NodeConfig::default();
        config.schedule.storage_max_per_tick = Some(0);
        let _ = NodeRuntime::new(config, MemoryBlobStore::new());
    }

    #[test]
    fn adaptive_fuel_reduces_compute_under_load() {
        let mut config = NodeConfig::default();
        config.compute_budget = InstructionBudget { fuel: 1000 };
        config.schedule.adaptive_compute.high_water = 10;
        config.schedule.adaptive_compute.floor_fraction = 0.1;
        let (mut rt, _) = NodeRuntime::new(config, MemoryBlobStore::new());

        // Verify full fuel with empty queues
        assert_eq!(rt.effective_fuel(), 1000);

        // Push 10 router events (= high_water) → fuel at floor
        for i in 0..10 {
            rt.push_event(RuntimeEvent::TimerTick { now: 1000 + i });
        }
        assert_eq!(rt.effective_fuel(), 100);

        // After tick drains them all, fuel should recover
        rt.tick();
        assert_eq!(rt.effective_fuel(), 1000);
    }

    #[test]
    fn runtime_uses_content_policy() {
        use harmony_content::blob::MemoryBlobStore;

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
        };
        let (rt, _) = NodeRuntime::new(config, MemoryBlobStore::new());
        assert_eq!(rt.storage_queue_len(), 0);
    }

    // ── PeerFilterTable tests ───────────────────────────────────────

    use harmony_content::cid::ContentFlags;

    #[test]
    fn peer_filter_table_skips_definite_miss() {
        let mut table = PeerFilterTable::new(100);
        let mut filter = BloomFilter::new(1000, 0.01);
        let cid_in = ContentId::for_blob(b"present", ContentFlags::default()).unwrap();
        let cid_out = ContentId::for_blob(b"absent", ContentFlags::default()).unwrap();
        filter.insert(&cid_in);
        table.upsert_content("peer-1".into(), filter, 10);

        assert!(table.should_query("peer-1", &cid_in, 10));
        assert!(!table.should_query("peer-1", &cid_out, 10));
    }

    #[test]
    fn peer_filter_table_queries_unknown_peer() {
        let table = PeerFilterTable::new(100);
        let cid = ContentId::for_blob(b"test", ContentFlags::default()).unwrap();
        assert!(table.should_query("unknown", &cid, 10));
    }

    #[test]
    fn peer_filter_table_queries_stale_filter() {
        let mut table = PeerFilterTable::new(100);
        let filter = BloomFilter::new(1000, 0.01);
        table.upsert_content("peer-1".into(), filter, 10);

        let cid = ContentId::for_blob(b"test", ContentFlags::default()).unwrap();
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
        use harmony_content::blob::MemoryBlobStore;

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
        };
        let (mut rt, _) = NodeRuntime::new(config, MemoryBlobStore::new());

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
        use harmony_content::blob::MemoryBlobStore;

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
        };
        let _ = NodeRuntime::new(config, MemoryBlobStore::new());
    }

    #[test]
    fn timer_skipped_when_threshold_broadcast_pending() {
        use harmony_content::blob::MemoryBlobStore;
        use harmony_content::cid::ContentFlags;

        // Set mutation_threshold=2 and max_interval_ticks=2.
        // Queue 2 transit events (crosses threshold) then call tick().
        // The timer fires on tick 2, but should be skipped because
        // a threshold broadcast is already pending.
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
        };
        let (mut rt, _) = NodeRuntime::new(config, MemoryBlobStore::new());

        // Tick 1: no events, no broadcast.
        rt.tick();

        // Queue 2 transit events to cross the mutation threshold.
        for i in 0..2u8 {
            let data = [i; 16];
            let cid = ContentId::for_blob(&data, ContentFlags::default()).unwrap();
            let cid_hex = hex::encode(cid.to_bytes());
            let key_expr = format!("harmony/content/transit/{cid_hex}");
            rt.push_event(RuntimeEvent::SubscriptionMessage {
                key_expr,
                payload: data.to_vec(),
            });
        }

        // Tick 2: timer interval also fires (tick 2 >= interval 2),
        // but should be skipped because threshold broadcast is pending.
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
            1,
            "expected exactly 1 filter broadcast (timer should be skipped when threshold pending)"
        );
    }

    #[test]
    fn filter_broadcasts_coalesced_within_tick() {
        use harmony_content::blob::MemoryBlobStore;
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
        };
        let (mut rt, _) = NodeRuntime::new(config, MemoryBlobStore::new());

        // Queue 6 transit events (each unique CID).
        for i in 0..6u8 {
            let data = [i; 16];
            let cid = ContentId::for_blob(&data, ContentFlags::default()).unwrap();
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
            .filter(|a| matches!(a, RuntimeAction::Publish { key_expr, .. }
                if key_expr.starts_with("harmony/filters/")))
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
        use harmony_content::blob::MemoryBlobStore;

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
        };
        let (mut rt, _) = NodeRuntime::new(config, MemoryBlobStore::new());

        // Tick enough times to trigger the timer (default interval is 30).
        // The timer fires on the 30th tick.
        for _ in 0..29 {
            rt.tick();
        }
        let actions = rt.tick();

        let filter_publishes: Vec<_> = actions
            .iter()
            .filter(|a| matches!(a, RuntimeAction::Publish { key_expr, .. }
                if key_expr.starts_with("harmony/filters/")))
            .collect();
        assert!(
            filter_publishes.len() >= 2,
            "expected bloom + cuckoo filter broadcasts, got {}",
            filter_publishes.len()
        );
    }

    #[test]
    fn route_subscription_parses_cuckoo_filter() {
        use harmony_content::blob::MemoryBlobStore;
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
        };
        let (mut rt, _) = NodeRuntime::new(config, MemoryBlobStore::new());

        let cf = CuckooFilter::new(100);
        let payload = cf.to_bytes();

        rt.push_event(RuntimeEvent::SubscriptionMessage {
            key_expr: "harmony/filters/flatpack/peer-xyz".to_string(),
            payload,
        });
        rt.tick();

        assert_eq!(rt.peer_filter_parse_errors(), 0);
    }
}
