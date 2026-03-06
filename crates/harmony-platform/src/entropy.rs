/// Platform-provided cryptographic randomness.
///
/// Ring 0 state machines accept `&mut impl EntropySource` wherever they need
/// randomness. Ring 1+ implements this against hardware RNG, OS getrandom,
/// or a seeded CSPRNG.
///
/// A blanket implementation is provided for all types implementing
/// [`rand_core::CryptoRngCore`], so existing callers passing `&mut OsRng`
/// or `&mut StdRng` work unchanged.
pub trait EntropySource {
    /// Fill `buf` with cryptographically secure random bytes.
    fn fill_bytes(&mut self, buf: &mut [u8]);
}

impl<T: rand_core::CryptoRngCore> EntropySource for T {
    fn fill_bytes(&mut self, buf: &mut [u8]) {
        rand_core::RngCore::fill_bytes(self, buf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blanket_impl_works_with_std_rng() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut buf = [0u8; 32];
        EntropySource::fill_bytes(&mut rng, &mut buf);
        assert_ne!(buf, [0u8; 32]);
    }

    #[test]
    fn deterministic_with_same_seed() {
        use rand::SeedableRng;
        let mut rng1 = rand::rngs::StdRng::seed_from_u64(99);
        let mut rng2 = rand::rngs::StdRng::seed_from_u64(99);
        let mut buf1 = [0u8; 16];
        let mut buf2 = [0u8; 16];
        EntropySource::fill_bytes(&mut rng1, &mut buf1);
        EntropySource::fill_bytes(&mut rng2, &mut buf2);
        assert_eq!(buf1, buf2);
    }

    #[test]
    fn accepts_entropy_source_trait_object() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let source: &mut dyn EntropySource = &mut rng;
        let mut buf = [0u8; 8];
        source.fill_bytes(&mut buf);
        assert_ne!(buf, [0u8; 8]);
    }
}
