use alloc::{sync::Arc, vec::Vec};
#[cfg(not(feature = "std"))]
use hashbrown::HashMap;
#[cfg(feature = "std")]
use std::collections::HashMap;

// ── Types ────────────────────────────────────────────────────────────────

/// Per-interface cooperation score tracking announce forwarding and proof delivery.
#[derive(Debug, Clone)]
pub struct CooperationScore {
    /// EMA score for announce forwarding behavior (0.0–1.0).
    pub announce_score: f32,
    /// EMA score for proof delivery behavior (0.0–1.0).
    pub proof_score: f32,
    /// Combined weighted score: 0.7 * announce_score + 0.3 * proof_score.
    pub combined: f32,
    /// Total number of observations recorded.
    pub observation_count: u64,
    /// Monotonic timestamp (ms) of the last observation.
    pub last_observed: u64,
}

/// Tracks per-interface cooperation scores using exponential moving averages.
///
/// Scores reflect two signals:
/// - **Announce forwarding** (weight 70%): did the interface forward announces onward?
/// - **Proof delivery** (weight 30%): did the interface deliver proof packets reliably?
///
/// The combined score is used for probabilistic weighted broadcast selection.
/// The highest-scoring interface is always forced to weight 1.0 to guarantee
/// at least one interface always receives every broadcast.
pub struct CooperationTable {
    scores: HashMap<Arc<str>, CooperationScore>,
    /// EMA smoothing factor α (default 0.1).
    alpha: f32,
    /// Minimum score floor — prevents an interface from dropping to zero (default 0.05).
    floor: f32,
    /// Starting score for newly registered interfaces (default 0.5).
    initial: f32,
    /// Milliseconds of inactivity before staleness decay kicks in (default 300_000 = 5 min).
    staleness_window: u64,
}

impl Default for CooperationTable {
    fn default() -> Self {
        Self::new(0.1, 0.05, 0.5, 300_000)
    }
}

impl CooperationTable {
    /// Create a new CooperationTable with explicit parameters.
    ///
    /// - `alpha`: EMA smoothing factor (0 < alpha ≤ 1). Smaller = slower adaptation.
    /// - `floor`: minimum score floor after decay.
    /// - `initial`: starting score for new interfaces.
    /// - `staleness_window`: milliseconds after which an interface is considered stale.
    pub fn new(alpha: f32, floor: f32, initial: f32, staleness_window: u64) -> Self {
        Self {
            scores: HashMap::new(),
            alpha,
            floor,
            initial,
            staleness_window,
        }
    }

    /// Register an interface. If it is already present, this is a no-op.
    pub fn register_interface(&mut self, name: Arc<str>) {
        self.scores.entry(name).or_insert_with(|| CooperationScore {
            announce_score: self.initial,
            proof_score: self.initial,
            combined: self.initial,
            observation_count: 0,
            last_observed: 0,
        });
    }

    /// Return the combined cooperation weight for `interface`.
    ///
    /// - If only one interface is registered, always returns 1.0.
    /// - If the interface is unknown, returns `self.initial`.
    pub fn get_weight(&self, interface: &str) -> f32 {
        if self.scores.len() == 1 {
            return 1.0;
        }
        self.scores
            .get(interface)
            .map(|s| s.combined)
            .unwrap_or(self.initial)
    }

    /// Return weights for all registered interfaces.
    ///
    /// The highest-scored interface is forced to 1.0 to guarantee at least one
    /// interface always participates in every broadcast.
    pub fn get_broadcast_weights(&self) -> Vec<(Arc<str>, f32)> {
        if self.scores.is_empty() {
            return Vec::new();
        }

        // Find the maximum combined score.
        let max_score = self
            .scores
            .values()
            .map(|s| s.combined)
            .fold(f32::NEG_INFINITY, f32::max);

        self.scores
            .iter()
            .map(|(name, score)| {
                let weight = if score.combined >= max_score {
                    1.0
                } else {
                    score.combined
                };
                (Arc::clone(name), weight)
            })
            .collect()
    }

    /// Record a positive announce-forwarding signal for `interface`.
    pub fn observe_announce_forwarded(&mut self, interface: &str, now: u64) {
        let (alpha, floor) = (self.alpha, self.floor);
        if let Some(score) = self.scores.get_mut(interface) {
            score.announce_score = Self::ema_update_params(alpha, floor, score.announce_score, true);
            score.observation_count += 1;
            score.last_observed = now;
            Self::recombine(score);
        }
    }

    /// Record a negative announce-forwarding signal for `interface`.
    pub fn observe_announce_timeout(&mut self, interface: &str, now: u64) {
        let (alpha, floor) = (self.alpha, self.floor);
        if let Some(score) = self.scores.get_mut(interface) {
            score.announce_score =
                Self::ema_update_params(alpha, floor, score.announce_score, false);
            score.observation_count += 1;
            score.last_observed = now;
            Self::recombine(score);
        }
    }

    /// Record a positive proof-delivery signal for `interface`.
    pub fn observe_proof_delivered(&mut self, interface: &str, now: u64) {
        let (alpha, floor) = (self.alpha, self.floor);
        if let Some(score) = self.scores.get_mut(interface) {
            score.proof_score = Self::ema_update_params(alpha, floor, score.proof_score, true);
            score.observation_count += 1;
            score.last_observed = now;
            Self::recombine(score);
        }
    }

    /// Record a negative proof-delivery signal for `interface`.
    pub fn observe_proof_timeout(&mut self, interface: &str, now: u64) {
        let (alpha, floor) = (self.alpha, self.floor);
        if let Some(score) = self.scores.get_mut(interface) {
            score.proof_score = Self::ema_update_params(alpha, floor, score.proof_score, false);
            score.observation_count += 1;
            score.last_observed = now;
            Self::recombine(score);
        }
    }

    /// Move stale interfaces 10% of the way back toward `self.initial`.
    ///
    /// An interface is considered stale if `now - last_observed > staleness_window`.
    /// Interfaces with `last_observed == 0` (never seen) are skipped.
    pub fn decay_stale(&mut self, now: u64) {
        let staleness_window = self.staleness_window;
        let initial = self.initial;
        let floor = self.floor;

        for score in self.scores.values_mut() {
            // Skip interfaces that have never been observed.
            if score.last_observed == 0 {
                continue;
            }
            let elapsed = now.saturating_sub(score.last_observed);
            if elapsed > staleness_window {
                // Move 10% toward initial.
                score.announce_score =
                    (score.announce_score + 0.1 * (initial - score.announce_score)).max(floor);
                score.proof_score =
                    (score.proof_score + 0.1 * (initial - score.proof_score)).max(floor);
                Self::recombine(score);
                // Reset last_observed so decay fires once per staleness window,
                // not on every tick after the window passes.
                score.last_observed = now;
            }
        }
    }

    /// Remove an interface from the table.
    pub fn remove_interface(&mut self, interface: &str) {
        self.scores.remove(interface);
    }

    /// Return the number of registered interfaces.
    pub fn interface_count(&self) -> usize {
        self.scores.len()
    }

    /// Iterate over all (name, score) pairs for diagnostics.
    pub fn iter(&self) -> impl Iterator<Item = (&Arc<str>, &CooperationScore)> {
        self.scores.iter()
    }

    // ── Private helpers ──────────────────────────────────────────────────

    fn ema_update_params(alpha: f32, floor: f32, current: f32, success: bool) -> f32 {
        let observation = if success { 1.0_f32 } else { 0.0_f32 };
        let updated = alpha * observation + (1.0 - alpha) * current;
        updated.max(floor)
    }

    fn recombine(score: &mut CooperationScore) {
        score.combined = 0.7 * score.announce_score + 0.3 * score.proof_score;
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a table with default settings and one registered interface.
    fn table_with(name: &str) -> (CooperationTable, Arc<str>) {
        let mut table = CooperationTable::default();
        let iface: Arc<str> = Arc::from(name);
        table.register_interface(Arc::clone(&iface));
        (table, iface)
    }

    #[test]
    fn ema_converges_positive() {
        let (mut table, _) = table_with("eth0");
        // Second interface so single-interface short-circuit doesn't fire.
        table.register_interface(Arc::from("eth1"));

        for _ in 0..10 {
            table.observe_announce_forwarded("eth0", 1000);
        }
        let score = table.scores["eth0"].announce_score;
        assert!(
            score > 0.8,
            "expected announce_score > 0.8 after 10 positive observations, got {score}"
        );
    }

    #[test]
    fn ema_converges_negative() {
        let (mut table, _) = table_with("eth0");
        table.register_interface(Arc::from("eth1"));

        // With α=0.1 starting from 0.5, need ~25 negative observations to reach below 0.1.
        for _ in 0..25 {
            table.observe_announce_timeout("eth0", 1000);
        }
        let score = table.scores["eth0"].announce_score;
        assert!(
            score < 0.1,
            "expected announce_score < 0.1 after 25 negative observations, got {score}"
        );
    }

    #[test]
    fn floor_enforced() {
        let (mut table, _) = table_with("eth0");
        table.register_interface(Arc::from("eth1"));

        for _ in 0..100 {
            table.observe_announce_timeout("eth0", 1000);
        }
        let score = table.scores["eth0"].announce_score;
        assert!(
            score >= 0.05,
            "score dropped below floor: {score}"
        );
    }

    #[test]
    fn new_interface_starts_at_initial() {
        let mut table = CooperationTable::default();
        // Register a second interface so we don't hit the single-interface path.
        table.register_interface(Arc::from("eth0"));
        table.register_interface(Arc::from("eth1"));

        // eth2 is not registered at all.
        let weight = table.get_weight("eth2");
        assert!(
            (weight - 0.5).abs() < 1e-6,
            "expected initial weight 0.5 for unknown interface, got {weight}"
        );
    }

    #[test]
    fn combined_weighting() {
        let (mut table, _) = table_with("eth0");
        table.register_interface(Arc::from("eth1"));

        // Force announce_score=1.0, proof_score=0.0 by manipulating the score directly.
        {
            let score = table.scores.get_mut("eth0").unwrap();
            score.announce_score = 1.0;
            score.proof_score = 0.0;
            CooperationTable::recombine(score);
        }

        let combined = table.scores["eth0"].combined;
        assert!(
            (combined - 0.7).abs() < 1e-5,
            "expected combined ~0.7, got {combined}"
        );
    }

    #[test]
    fn remove_interface_cleans_up() {
        let (mut table, _) = table_with("eth0");
        table.register_interface(Arc::from("eth1"));

        table.remove_interface("eth1");

        let names: Vec<_> = table.iter().map(|(n, _)| n.as_ref()).collect();
        assert!(
            !names.contains(&"eth1"),
            "eth1 should have been removed, but iter still shows it"
        );
        assert_eq!(table.interface_count(), 1);
    }

    #[test]
    fn single_interface_always_one() {
        let (mut table, name) = table_with("eth0");

        // Drive score way down.
        for _ in 0..100 {
            table.observe_announce_timeout("eth0", 1000);
        }

        let weight = table.get_weight(&name);
        assert!(
            (weight - 1.0).abs() < 1e-6,
            "single registered interface should always return weight 1.0, got {weight}"
        );
    }

    #[test]
    fn broadcast_weights_highest_forced_one() {
        let mut table = CooperationTable::default();
        let a: Arc<str> = Arc::from("eth0");
        let b: Arc<str> = Arc::from("eth1");
        table.register_interface(Arc::clone(&a));
        table.register_interface(Arc::clone(&b));

        // Push eth0 higher by giving it positive observations.
        for _ in 0..20 {
            table.observe_announce_forwarded("eth0", 1000);
        }
        // Push eth1 lower with negative observations.
        for _ in 0..10 {
            table.observe_announce_timeout("eth1", 1000);
        }

        let weights = table.get_broadcast_weights();
        let eth0_weight = weights
            .iter()
            .find(|(n, _)| n.as_ref() == "eth0")
            .map(|(_, w)| *w)
            .expect("eth0 missing from broadcast weights");
        let eth1_weight = weights
            .iter()
            .find(|(n, _)| n.as_ref() == "eth1")
            .map(|(_, w)| *w)
            .expect("eth1 missing from broadcast weights");

        assert!(
            (eth0_weight - 1.0).abs() < 1e-6,
            "highest-scored interface eth0 should have weight 1.0, got {eth0_weight}"
        );
        assert!(
            eth1_weight < 1.0,
            "lower-scored interface eth1 should have weight < 1.0, got {eth1_weight}"
        );
    }

    #[test]
    fn staleness_decay() {
        let (mut table, _) = table_with("eth0");
        table.register_interface(Arc::from("eth1"));

        // Drive eth0 score high so decay is clearly visible.
        // Observe at t=1 so last_observed != 0 (0 is the sentinel for "never observed").
        for _ in 0..20 {
            table.observe_announce_forwarded("eth0", 1);
        }
        let before = table.scores["eth0"].announce_score;

        // Decay at staleness_window + 2 ms (well past the window since last_observed=1).
        let now = table.staleness_window + 2;
        table.decay_stale(now);

        let after = table.scores["eth0"].announce_score;

        // Score should have moved toward initial (0.5), so it should be lower than before.
        assert!(
            after < before,
            "staleness decay should move score toward initial; before={before}, after={after}"
        );
    }
}
