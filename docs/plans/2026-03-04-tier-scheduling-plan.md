# Configurable Tier Scheduling Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the NodeRuntime's per-tick event processing strategy configurable with max-per-tick limits, adaptive compute fuel, and starvation-based priority promotion.

**Architecture:** Add `TierSchedule` and `AdaptiveCompute` config types to `NodeConfig`. Store per-tier starvation counters in `NodeRuntime`. Refactor `tick()` to calculate effective fuel, determine tier order from starvation state, and respect max-per-tick limits. All existing behavior preserved under default config.

**Tech Stack:** Rust, harmony-node crate, pure sans-I/O (no async)

---

### Task 1: Add TierSchedule and AdaptiveCompute types

**Files:**
- Modify: `crates/harmony-node/src/runtime.rs:21-40`

**Step 1: Write the failing test**

Add to the test module at the bottom of `runtime.rs`:

```rust
#[test]
fn tier_schedule_defaults() {
    let schedule = TierSchedule::default();
    assert!(schedule.router_max_per_tick.is_none());
    assert!(schedule.storage_max_per_tick.is_none());
    assert_eq!(schedule.starvation_threshold, 10);
    assert_eq!(schedule.adaptive_compute.high_water, 50);
    assert!((schedule.adaptive_compute.floor_fraction - 0.1).abs() < f64::EPSILON);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p harmony-node tier_schedule_defaults`
Expected: FAIL — `TierSchedule` not found

**Step 3: Write minimal implementation**

Add after line 28 (after `NodeConfig` fields, before its `Default` impl):

```rust
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
```

Add `schedule: TierSchedule` field to `NodeConfig` and include `schedule: TierSchedule::default()` in `NodeConfig::default()`.

**Step 4: Run test to verify it passes**

Run: `cargo test -p harmony-node tier_schedule_defaults`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/harmony-node/src/runtime.rs
git commit -m "feat(node): add TierSchedule and AdaptiveCompute config types"
```

---

### Task 2: Wire TierSchedule into NodeRuntime and add starvation counters

**Files:**
- Modify: `crates/harmony-node/src/runtime.rs:109-200` (NodeRuntime struct + new())

**Step 1: Write the failing test**

```rust
#[test]
fn runtime_exposes_schedule() {
    let mut config = NodeConfig::default();
    config.schedule.router_max_per_tick = Some(5);
    let (rt, _) = NodeRuntime::new(config, MemoryBlobStore::new());
    assert_eq!(rt.schedule().router_max_per_tick, Some(5));
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p harmony-node runtime_exposes_schedule`
Expected: FAIL — `schedule()` method not found

**Step 3: Write minimal implementation**

Add to `NodeRuntime` struct:

```rust
    // Tier scheduling configuration
    schedule: TierSchedule,
    // Starvation counters: incremented when a tier has no events in a tick, reset on processing
    router_starved: u32,
    storage_starved: u32,
    compute_starved: u32,
```

Initialize in `new()`:

```rust
    schedule: config.schedule.clone(),
    router_starved: 0,
    storage_starved: 0,
    compute_starved: 0,
```

Add accessor:

```rust
    /// Read-only access to the tier schedule configuration.
    pub fn schedule(&self) -> &TierSchedule {
        &self.schedule
    }
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p harmony-node runtime_exposes_schedule`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/harmony-node/src/runtime.rs
git commit -m "feat(node): wire TierSchedule into NodeRuntime with starvation counters"
```

---

### Task 3: Add effective_fuel calculation

**Files:**
- Modify: `crates/harmony-node/src/runtime.rs`

**Step 1: Write the failing test**

```rust
#[test]
fn effective_fuel_scales_with_queue_depth() {
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
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p harmony-node effective_fuel_scales`
Expected: FAIL — `effective_fuel()` not found

**Step 3: Write minimal implementation**

Add method to `NodeRuntime`:

```rust
    /// Calculate the effective compute fuel budget based on data-plane queue depth.
    ///
    /// At zero depth, returns the full base budget. As combined queue depth
    /// approaches `high_water`, fuel shrinks linearly toward `floor_fraction * base`.
    /// Above high_water, fuel stays at the floor.
    pub fn effective_fuel(&self) -> u64 {
        let base = self.workflow.budget().fuel;
        let ac = &self.schedule.adaptive_compute;
        if ac.high_water == 0 {
            return (base as f64 * ac.floor_fraction) as u64;
        }
        let combined = self.router_queue.len() + self.storage_queue.len();
        let load_factor = (combined as f64 / ac.high_water as f64).min(1.0);
        let scale = 1.0 - load_factor * (1.0 - ac.floor_fraction);
        (base as f64 * scale) as u64
    }
```

This requires exposing `budget()` from `WorkflowEngine`. Check if it exists; if not, add:

```rust
// In harmony-workflow/src/engine.rs
pub fn budget(&self) -> InstructionBudget {
    self.budget
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p harmony-node effective_fuel_scales`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/harmony-node/src/runtime.rs crates/harmony-workflow/src/engine.rs
git commit -m "feat(node): add effective_fuel() with linear adaptive scaling"
```

---

### Task 4: Refactor tick() to respect max-per-tick limits

**Files:**
- Modify: `crates/harmony-node/src/runtime.rs:311-344` (tick method)

**Step 1: Write the failing test**

```rust
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
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-node max_per_tick_caps`
Expected: FAIL — all 5/3 events drained (current drain-all)

**Step 3: Refactor tick()**

Replace the router and storage drain loops to respect limits:

```rust
pub fn tick(&mut self) -> Vec<RuntimeAction> {
    let mut actions = Vec::new();

    // Tier 1: drain router events (up to max_per_tick if set)
    let router_limit = self.schedule.router_max_per_tick.unwrap_or(usize::MAX);
    let mut router_processed = 0;
    while router_processed < router_limit {
        match self.router_queue.pop_front() {
            Some(event) => {
                let node_actions = self.router.handle_event(event);
                self.dispatch_router_actions(node_actions, &mut actions);
                router_processed += 1;
            }
            None => break,
        }
    }

    // Tier 2: drain storage events (up to max_per_tick if set)
    let storage_limit = self.schedule.storage_max_per_tick.unwrap_or(usize::MAX);
    let mut storage_processed = 0;
    while storage_processed < storage_limit {
        match self.storage_queue.pop_front() {
            Some(event) => {
                let storage_actions = self.storage.handle(event);
                self.dispatch_storage_actions(storage_actions, &mut actions);
                storage_processed += 1;
            }
            None => break,
        }
    }

    // Tier 3: emit any direct replies buffered from push_event
    actions.append(&mut self.pending_direct_actions);

    // Tier 3: dispatch any pending workflow actions, then one compute slice
    let pending = std::mem::take(&mut self.pending_workflow_actions);
    self.dispatch_workflow_actions(pending, &mut actions);
    let workflow_actions = self.workflow.tick();
    self.dispatch_workflow_actions(workflow_actions, &mut actions);

    actions
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-node`
Expected: ALL pass (new tests + existing tests unchanged because default is `None` = drain-all)

**Step 5: Commit**

```bash
git add crates/harmony-node/src/runtime.rs
git commit -m "feat(node): respect max-per-tick limits in tick()"
```

---

### Task 5: Add starvation tracking and tier reordering

**Files:**
- Modify: `crates/harmony-node/src/runtime.rs` (tick method)

**Step 1: Write the failing test**

```rust
#[test]
fn starvation_promotes_starved_tier() {
    let mut config = NodeConfig::default();
    // Cap router to 1 per tick so storage can be starved
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

    // Tick 1-3: router drains 1 per tick, storage processes its 1 event on tick 1
    let actions = rt.tick(); // tick 1: router=1, storage=1 (processes the stats query)
    assert!(actions.iter().any(|a| matches!(a, RuntimeAction::SendReply { query_id: 50, .. })));

    rt.tick(); // tick 2: router=1, storage=0 (empty → starved=1)
    rt.tick(); // tick 3: router=1, storage=0 (starved=2)
    rt.tick(); // tick 4: router=1, storage=0 (starved=3 → threshold hit)

    // Now push a storage event — it should be promoted ahead of router
    rt.push_event(RuntimeEvent::QueryReceived {
        query_id: 60,
        key_expr: "harmony/content/stats".into(),
        payload: vec![],
    });

    // Tick 5: storage should be promoted (processed first)
    // Verify storage event was processed (starvation counter resets)
    let actions = rt.tick();
    assert!(
        actions.iter().any(|a| matches!(a, RuntimeAction::SendReply { query_id: 60, .. })),
        "starved storage tier should be promoted and process its event"
    );
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p harmony-node starvation_promotes`
Expected: FAIL or PASS depending on current ordering — but the test is structured to verify the promotion actually happened. If storage always runs after router anyway, we need a more specific test. The key test is that promotion *occurs* and the counter *resets*.

Add a more targeted test for counter state via accessor:

```rust
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
```

**Step 3: Write implementation**

Update `tick()` to track starvation and reorder tiers:

```rust
pub fn tick(&mut self) -> Vec<RuntimeAction> {
    let mut actions = Vec::new();
    let threshold = self.schedule.starvation_threshold;

    // Determine tier order: promote starved tiers to front
    let mut order = [0u8, 1, 2]; // Router=0, Storage=1, Compute=2
    let starved = [self.router_starved, self.storage_starved, self.compute_starved];
    // Stable sort: starved tiers (>= threshold) move to front, preserving original order
    order.sort_by_key(|&tier| if starved[tier as usize] >= threshold { 0 } else { 1 });

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
                if processed > 0 { self.router_starved = 0; } else { self.router_starved += 1; }
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
                if processed > 0 { self.storage_starved = 0; } else { self.storage_starved += 1; }
            }
            2 => {
                // Tier 3: Compute
                actions.append(&mut self.pending_direct_actions);
                let pending = std::mem::take(&mut self.pending_workflow_actions);
                self.dispatch_workflow_actions(pending, &mut actions);
                let workflow_actions = self.workflow.tick();
                let had_work = !workflow_actions.is_empty();
                self.dispatch_workflow_actions(workflow_actions, &mut actions);
                if had_work { self.compute_starved = 0; } else { self.compute_starved += 1; }
            }
            _ => unreachable!(),
        }
    }

    actions
}
```

Add accessor:

```rust
    /// Current starvation counters (router, storage, compute).
    pub fn starvation_counters(&self) -> (u32, u32, u32) {
        (self.router_starved, self.storage_starved, self.compute_starved)
    }
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-node`
Expected: ALL pass

**Step 5: Commit**

```bash
git add crates/harmony-node/src/runtime.rs
git commit -m "feat(node): add starvation tracking and tier reordering"
```

---

### Task 6: Wire adaptive fuel into tick's compute slice

**Files:**
- Modify: `crates/harmony-node/src/runtime.rs`
- Modify: `crates/harmony-workflow/src/engine.rs` (add `tick_with_fuel` or `set_budget`)

**Step 1: Write the failing test**

```rust
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
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p harmony-node adaptive_fuel_reduces`
Expected: PASS (effective_fuel already works from Task 3, this is a sanity check)

**Step 3: Wire effective_fuel into the compute slice in tick()**

In the Tier 3 (compute) branch of the tick loop, replace `self.workflow.tick()` with a fuel-aware version:

```rust
    // Use effective fuel based on current queue depth
    let fuel = self.effective_fuel();
    let workflow_actions = self.workflow.tick_with_budget(InstructionBudget { fuel });
```

Add `tick_with_budget` to `WorkflowEngine`:

```rust
    /// Run one compute slice with a specific fuel budget (for adaptive scheduling).
    pub fn tick_with_budget(&mut self, budget: InstructionBudget) -> Vec<WorkflowAction> {
        let saved = self.budget;
        self.budget = budget;
        let result = self.tick();
        self.budget = saved;
        result
    }
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-node && cargo test -p harmony-workflow`
Expected: ALL pass

**Step 5: Commit**

```bash
git add crates/harmony-node/src/runtime.rs crates/harmony-workflow/src/engine.rs
git commit -m "feat(node): wire adaptive fuel into compute slice"
```

---

### Task 7: Update NodeRuntime doc comment and run full workspace tests

**Files:**
- Modify: `crates/harmony-node/src/runtime.rs:102-108` (doc comment)

**Step 1: Update doc comment**

Replace the existing `NodeRuntime` doc comment:

```rust
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
```

**Step 2: Run full workspace tests and clippy**

Run: `cargo test --workspace && cargo clippy --workspace`
Expected: ALL tests pass, zero clippy warnings

**Step 3: Commit**

```bash
git add crates/harmony-node/src/runtime.rs
git commit -m "docs(node): update NodeRuntime doc comment for configurable scheduling"
```
