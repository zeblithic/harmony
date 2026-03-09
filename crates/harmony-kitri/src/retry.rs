// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Failure classification and retry policy.

use core::cmp;

/// Classification of a workflow failure.
///
/// Determines whether automatic retry is appropriate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailureKind {
    /// Network timeout, node crash, temporary unavailability.
    Transient,
    /// Fuel budget or memory limit exceeded.
    ResourceExhausted,
    /// Panic, assertion failure, unrecoverable logic error.
    LogicError,
    /// Capability denied — missing or invalid UCAN.
    Unauthorized,
    /// Lyll/Nakaiah detected memory corruption.
    IntegrityViolation,
}

impl FailureKind {
    /// Whether this failure kind should trigger automatic retry.
    pub fn is_retryable(self) -> bool {
        matches!(self, Self::Transient | Self::ResourceExhausted)
    }
}

/// Backoff strategy between retries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackoffStrategy {
    /// Fixed delay between retries.
    Fixed { interval_ms: u64 },
    /// Exponential backoff: `initial_ms * 2^attempt`, capped at `max_ms`.
    Exponential { initial_ms: u64, max_ms: u64 },
}

impl BackoffStrategy {
    /// Compute the delay in milliseconds for a given retry attempt (0-indexed).
    pub fn delay_ms(&self, attempt: u32) -> u64 {
        match self {
            Self::Fixed { interval_ms } => *interval_ms,
            Self::Exponential { initial_ms, max_ms } => {
                let factor = 1u64.checked_shl(attempt).unwrap_or(u64::MAX);
                let delay = initial_ms.saturating_mul(factor);
                cmp::min(delay, *max_ms)
            }
        }
    }
}

/// Retry policy for a Kitri workflow.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetryPolicy {
    /// Maximum number of retry attempts (0 = no retries).
    pub max_retries: u32,
    /// Backoff strategy between retries.
    pub backoff: BackoffStrategy,
    /// Total timeout in milliseconds (0 = no timeout).
    pub timeout_ms: u64,
}

impl RetryPolicy {
    /// No retries — fail immediately on any error.
    pub fn none() -> Self {
        Self {
            max_retries: 0,
            backoff: BackoffStrategy::Fixed { interval_ms: 0 },
            timeout_ms: 0,
        }
    }
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            backoff: BackoffStrategy::Exponential {
                initial_ms: 100,
                max_ms: 30_000,
            },
            timeout_ms: 300_000, // 5 minutes
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failure_kind_is_retryable() {
        assert!(FailureKind::Transient.is_retryable());
        assert!(FailureKind::ResourceExhausted.is_retryable());
        assert!(!FailureKind::LogicError.is_retryable());
        assert!(!FailureKind::Unauthorized.is_retryable());
        assert!(!FailureKind::IntegrityViolation.is_retryable());
    }

    #[test]
    fn retry_policy_default() {
        let policy = RetryPolicy::default();
        assert_eq!(policy.max_retries, 3);
        assert!(matches!(
            policy.backoff,
            BackoffStrategy::Exponential { .. }
        ));
    }

    #[test]
    fn retry_policy_no_retries() {
        let policy = RetryPolicy::none();
        assert_eq!(policy.max_retries, 0);
    }

    #[test]
    fn backoff_delay_exponential() {
        let backoff = BackoffStrategy::Exponential {
            initial_ms: 100,
            max_ms: 10_000,
        };
        assert_eq!(backoff.delay_ms(0), 100); // 100 * 2^0
        assert_eq!(backoff.delay_ms(1), 200); // 100 * 2^1
        assert_eq!(backoff.delay_ms(2), 400); // 100 * 2^2
        assert_eq!(backoff.delay_ms(10), 10_000); // capped at max
    }

    #[test]
    fn backoff_delay_fixed() {
        let backoff = BackoffStrategy::Fixed { interval_ms: 500 };
        assert_eq!(backoff.delay_ms(0), 500);
        assert_eq!(backoff.delay_ms(5), 500);
    }
}
