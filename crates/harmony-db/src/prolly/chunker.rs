// CDF boundary decision for Prolly Tree chunking.

/// Configuration for the CDF-based chunking algorithm.
#[derive(Debug, Clone)]
pub(crate) struct ChunkerConfig {
    /// Target average chunk size in bytes.
    pub target_size: usize,
    /// Hard minimum: never split below this size.
    pub min_size: usize,
    /// Hard maximum: always split at this size.
    pub max_size: usize,
    /// Standard deviation for the normal CDF approximation.
    pub std_dev: f64,
}

impl ChunkerConfig {
    /// Default configuration targeting ~4 KiB chunks.
    pub fn default_4k() -> Self {
        ChunkerConfig {
            target_size: 4096,
            min_size: 512,
            max_size: 16384,
            std_dev: 1024.0,
        }
    }

    /// Decide whether the current position is a chunk boundary.
    ///
    /// Uses a conditional probability derived from the normal CDF (logistic
    /// approximation): P = (F(s+a) - F(s)) / (1 - F(s)), where s is the
    /// current accumulated size and a is the next entry size. The key bytes
    /// are hashed with BLAKE3 to produce a uniform random variable in [0, 1).
    ///
    /// Hard boundaries: returns false below `min_size`, returns true at or
    /// above `max_size`.
    pub fn is_boundary(&self, current_size: usize, entry_size: usize, key: &[u8]) -> bool {
        // Hard minimum: never split below min_size.
        // When is_boundary returns true, the finalized chunk has size
        // `current_size` (the new entry starts the next chunk), so we
        // check current_size, not current_size + entry_size.
        if current_size < self.min_size {
            return false;
        }
        // Hard maximum: always split at or above max_size.
        if current_size + entry_size >= self.max_size {
            return true;
        }

        let s = current_size as f64;
        let a = entry_size as f64;
        let mu = self.target_size as f64;
        let sigma = self.std_dev;

        // Normal CDF logistic approximation: Φ(z) ≈ 0.5*(1 + tanh(√(2/π)*z))
        // which equals logistic(2*√(2/π)*z). So k = 2*√(2/π)/σ ≈ 1.5957691/σ.
        let k = 2.0 * 0.7978845608028654_f64 / sigma;

        let f_s = logistic(k * (s - mu));
        let f_s_a = logistic(k * (s + a - mu));

        // Conditional probability: P = (F(s+a) - F(s)) / (1 - F(s))
        let denom = 1.0 - f_s;
        let p = if denom <= 1e-15 {
            1.0
        } else {
            ((f_s_a - f_s) / denom).clamp(0.0, 1.0)
        };

        // Hash ONLY the key with BLAKE3 to get a uniform random var in [0, 1).
        let hash = blake3::hash(key);
        let hash_bytes = hash.as_bytes();
        let hash_u64 = u64::from_le_bytes(hash_bytes[..8].try_into().unwrap());
        let random_var = (hash_u64 as f64) / (u64::MAX as f64);

        random_var <= p
    }
}

/// Logistic function: 1 / (1 + exp(-x)).
#[inline]
fn logistic(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Split a sorted slice of items into chunks using the CDF boundary algorithm.
///
/// `size_fn` returns the approximate byte size of an item.
/// `key_fn` returns the key bytes used for the BLAKE3 boundary hash.
pub(crate) fn chunk_items<T: Clone>(
    items: &[T],
    config: &ChunkerConfig,
    size_fn: impl Fn(&T) -> usize,
    key_fn: impl Fn(&T) -> &[u8],
) -> Vec<Vec<T>> {
    if items.is_empty() {
        return vec![];
    }

    let mut chunks: Vec<Vec<T>> = Vec::new();
    let mut current_chunk: Vec<T> = Vec::new();
    let mut current_size: usize = 0;

    for item in items {
        let entry_size = size_fn(item);
        let key = key_fn(item);

        if !current_chunk.is_empty() && config.is_boundary(current_size, entry_size, key) {
            chunks.push(std::mem::take(&mut current_chunk));
            current_size = 0;
        }

        current_chunk.push(item.clone());
        current_size += entry_size;
    }

    if !current_chunk.is_empty() {
        chunks.push(current_chunk);
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boundary_below_min_never_splits() {
        let config = ChunkerConfig::default_4k();
        // 100 iterations at 100 bytes total (well below min_size of 512).
        for i in 0..100 {
            let key = format!("key-{i}");
            assert!(
                !config.is_boundary(50, 50, key.as_bytes()),
                "should never split at size 100, below min_size 512"
            );
        }
    }

    #[test]
    fn boundary_above_max_always_splits() {
        let config = ChunkerConfig::default_4k();
        // 100 iterations at max_size (should always split).
        for i in 0..100 {
            let key = format!("key-{i}");
            assert!(
                config.is_boundary(config.max_size, 100, key.as_bytes()),
                "should always split at or above max_size"
            );
        }
    }

    #[test]
    fn boundary_deterministic() {
        let config = ChunkerConfig::default_4k();
        let key = b"deterministic-key";
        let result1 = config.is_boundary(3000, 200, key);
        let result2 = config.is_boundary(3000, 200, key);
        assert_eq!(result1, result2, "same key+size must give same result");
    }

    #[test]
    fn boundary_key_only() {
        // Verify that BLAKE3 hash uses only key bytes — different keys at
        // the same size can produce different results. We search for a pair
        // of keys that diverge.
        let config = ChunkerConfig::default_4k();
        let current_size = 4000; // near target, non-trivial probability
        let entry_size = 200;
        let mut found_true = false;
        let mut found_false = false;

        for i in 0..1000 {
            let key = format!("test-key-{i}");
            let result = config.is_boundary(current_size, entry_size, key.as_bytes());
            if result {
                found_true = true;
            } else {
                found_false = true;
            }
            if found_true && found_false {
                break;
            }
        }

        assert!(
            found_true && found_false,
            "different keys should produce different boundary decisions near the target size"
        );
    }

    #[test]
    fn chunk_size_distribution() {
        let config = ChunkerConfig::default_4k();

        // Create 1000 items, each ~40 bytes of key+size.
        let items: Vec<(Vec<u8>, usize)> = (0..1000)
            .map(|i| {
                let key = format!("item-{i:06}").into_bytes();
                let size = 40;
                (key, size)
            })
            .collect();

        let chunks = chunk_items(
            &items,
            &config,
            |item| item.1,
            |item| &item.0,
        );

        // Verify we got multiple chunks (1000 items * 40 bytes = 40000 bytes,
        // target 4096, so we should get roughly 10 chunks).
        assert!(
            chunks.len() > 1,
            "should produce multiple chunks, got {}",
            chunks.len()
        );

        // Verify average chunk size is in the right ballpark.
        let total_items: usize = chunks.iter().map(|c| c.len()).sum();
        assert_eq!(total_items, 1000, "all items must be accounted for");

        let chunk_sizes: Vec<usize> = chunks
            .iter()
            .map(|c| c.iter().map(|item| item.1).sum::<usize>())
            .collect();
        let avg_size: f64 =
            chunk_sizes.iter().sum::<usize>() as f64 / chunk_sizes.len() as f64;

        // Average should be within a reasonable range of the target.
        assert!(
            avg_size > (config.min_size as f64),
            "average chunk size {} should be above min_size {}",
            avg_size,
            config.min_size
        );

        // Verify min_size is respected (except possibly the last chunk).
        for (i, size) in chunk_sizes.iter().enumerate() {
            if i < chunk_sizes.len() - 1 {
                assert!(
                    *size >= config.min_size,
                    "chunk {} has size {} below min_size {}",
                    i,
                    size,
                    config.min_size
                );
            }
        }
    }
}
