//! Week-aligned epoch math.
//!
//! Epoch identifier = `floor(unix_ms / EPOCH_DURATION_MS)`.
//! Resolver queries `[epoch-1, epoch, epoch+1]` in parallel for ±1-day
//! clock-skew tolerance across the 7-day boundary.

/// 1 week in milliseconds.
pub const EPOCH_DURATION_MS: u64 = 7 * 86_400_000;

/// Returns the epoch identifier containing `now_ms`.
pub fn current_epoch_id(now_ms: u64) -> u64 {
    now_ms / EPOCH_DURATION_MS
}

/// Returns the tolerance window `[prev, current, next]` epoch IDs.
/// Resolver queries all three keys in parallel.
pub fn epoch_tolerance_window(now_ms: u64) -> [u64; 3] {
    let e = current_epoch_id(now_ms);
    [e.saturating_sub(1), e, e.saturating_add(1)]
}

/// Wall-clock time (ms) when the given epoch starts.
pub fn epoch_start_ms(epoch_id: u64) -> u64 {
    epoch_id * EPOCH_DURATION_MS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epoch_zero_starts_at_unix_epoch() {
        assert_eq!(current_epoch_id(0), 0);
    }

    #[test]
    fn epoch_advances_at_week_boundary() {
        let just_before = EPOCH_DURATION_MS - 1;
        let just_after = EPOCH_DURATION_MS;
        assert_eq!(current_epoch_id(just_before), 0);
        assert_eq!(current_epoch_id(just_after), 1);
    }

    #[test]
    fn tolerance_window_returns_prev_current_next() {
        let now = 5 * EPOCH_DURATION_MS + EPOCH_DURATION_MS / 2; // mid-epoch 5
        let window = epoch_tolerance_window(now);
        assert_eq!(window, [4, 5, 6]);
    }

    #[test]
    fn tolerance_window_saturates_at_zero() {
        // Inside epoch 0: prev should saturate to 0, not underflow.
        let now = EPOCH_DURATION_MS / 4; // ~1.75 days in
        let window = epoch_tolerance_window(now);
        assert_eq!(window, [0, 0, 1]);
    }

    #[test]
    fn epoch_start_is_inverse_of_current_epoch_id() {
        let now = 12345 * EPOCH_DURATION_MS + 1000;
        let e = current_epoch_id(now);
        let start = epoch_start_ms(e);
        assert_eq!(start, 12345 * EPOCH_DURATION_MS);
        assert!(now - start < EPOCH_DURATION_MS);
    }
}
