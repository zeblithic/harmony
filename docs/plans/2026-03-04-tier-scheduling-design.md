# Configurable Tier Scheduling for Node Event Loop

**Bead:** harmony-e4e
**Date:** 2026-03-04
**Status:** Approved

## Problem

The `NodeRuntime::tick()` hard-codes its per-tick event processing strategy: drain-all router, drain-all storage, one compute slice. This is correct for current workloads but provides no knobs for tuning under different load profiles.

## Design

### TierSchedule Config

Add `TierSchedule` to `NodeConfig`:

```rust
pub struct TierSchedule {
    /// Max events to drain per tier per tick. None = drain-all.
    pub router_max_per_tick: Option<usize>,
    pub storage_max_per_tick: Option<usize>,

    /// Adaptive compute: shrink fuel when data-plane queues are deep.
    pub adaptive_compute: AdaptiveCompute,

    /// Ticks without processing before a tier is promoted in tick order.
    pub starvation_threshold: u32,
}

pub struct AdaptiveCompute {
    /// Queue depth at which compute fuel starts shrinking.
    pub high_water: usize,
    /// Minimum fuel as fraction of base budget (0.0..=1.0).
    pub floor_fraction: f64,
}
```

**Defaults:** `router_max_per_tick: None`, `storage_max_per_tick: None`, `starvation_threshold: 10`, `high_water: 50`, `floor_fraction: 0.1`. Existing behavior is unchanged unless the caller opts into limits.

### Tick Loop

1. **Calculate effective compute fuel:**
   - `combined_depth = router_queue.len() + storage_queue.len()`
   - `load_factor = (combined_depth / high_water).min(1.0)`
   - `effective_fuel = base_fuel * (1.0 - load_factor * (1.0 - floor_fraction))`
   - Clamp to at least `floor_fraction * base_fuel`

2. **Track starvation counters** per tier (increment when tier has zero events in a tick, reset on processing).

3. **Determine tick order:**
   - Default: [Router, Storage, Compute]
   - Promote any tier whose starvation counter >= threshold to front
   - Multiple starved tiers: promote in original priority order

4. **Process each tier** in determined order, respecting max-per-tick limits and effective fuel.

**Key invariant:** With default config, behavior is identical to today. Starvation counters only matter when max-per-tick limits are set.

### Adaptive Compute Detail

Linear scaling: at zero queue depth, compute gets full budget. As combined queue depth approaches `high_water`, fuel shrinks linearly to `floor_fraction * base_fuel`. Above high_water, fuel stays at the floor.

### Testing

1. **Default behavior preserved** — drain-all + one slice, identical to current.
2. **Max-per-tick capping** — verify partial drain with remaining events on next tick.
3. **Adaptive compute fuel** — verify linear scaling at various queue depths.
4. **Starvation promotion** — verify tier reordering after threshold exceeded.

All tests are pure sans-I/O.
