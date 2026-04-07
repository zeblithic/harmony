// Argon2id Hashcash puzzle for Sybil-resistant identity generation.
//
// Creating a network-usable identity requires solving a proof-of-work
// puzzle: find a nonce such that Argon2id(nonce, public_key_bytes)
// produces a hash with `difficulty` leading zero bits. This takes
// ~6.4 seconds at production difficulty (d=7) but only ~50ms to verify.

use argon2::{Algorithm, Argon2, Params, Version};
use rand_core::CryptoRngCore;
use serde::{Deserialize, Serialize};

use crate::IdentityError;

// ── Parameters ──────────────────────────────────────────────────────

/// Tunable parameters for the identity proof-of-work puzzle.
///
/// Production defaults: m=64MB, t=3, p=1, difficulty=7.
/// These yield ~50ms per single Argon2id evaluation on modern hardware,
/// requiring ~128 attempts (2^7) for an expected total of ~6.4 seconds.
///
/// Use [`PRODUCTION`](Self::PRODUCTION) or [`TEST`](Self::TEST) for
/// well-known configurations. For custom params, use [`new()`](Self::new)
/// which validates against Argon2 constraints.
#[derive(Debug, Clone, Copy)]
pub struct PuzzleParams {
    /// Memory cost in KiB (Argon2 `m` parameter).
    pub memory_kib: u32,
    /// Time cost / iteration count (Argon2 `t` parameter).
    pub time_cost: u32,
    /// Parallelism (Argon2 `p` parameter).
    pub parallelism: u32,
    /// Required number of leading zero bits in the hash output.
    pub difficulty: u8,
    /// Version tag embedded in proofs for forward compatibility.
    /// Verifiers reject proofs whose version doesn't match.
    pub params_version: u8,
}

impl PuzzleParams {
    /// Production parameters: ~6.4 seconds expected generation time.
    pub const PRODUCTION: Self = Self {
        memory_kib: 64 * 1024, // 64 MB
        time_cost: 3,
        parallelism: 1,
        difficulty: 7,
        params_version: 1,
    };

    /// Fast parameters for unit tests.
    pub const TEST: Self = Self {
        memory_kib: 256, // 256 KB
        time_cost: 1,
        parallelism: 1,
        difficulty: 1,
        params_version: 0,
    };

    /// Construct and validate custom puzzle parameters.
    ///
    /// Returns an error if the Argon2 parameters are out of range
    /// (e.g. memory_kib < 8, time_cost < 1, parallelism < 1).
    pub fn new(
        memory_kib: u32,
        time_cost: u32,
        parallelism: u32,
        difficulty: u8,
        params_version: u8,
    ) -> Result<Self, IdentityError> {
        Params::new(memory_kib, time_cost, parallelism, Some(32))
            .map_err(|_| IdentityError::InvalidPuzzleParams)?;
        Ok(Self {
            memory_kib,
            time_cost,
            parallelism,
            difficulty,
            params_version,
        })
    }

    fn argon2(&self) -> Argon2<'_> {
        let params = Params::new(self.memory_kib, self.time_cost, self.parallelism, Some(32))
            .expect("puzzle params are valid — use PuzzleParams::new() for custom values");
        Argon2::new(Algorithm::Argon2id, Version::V0x13, params)
    }
}

// ── Proof ───────────────────────────────────────────────────────────

/// Proof-of-work for a generated identity.
///
/// The nonce, combined with the identity's public key bytes as the
/// Argon2id salt, produces a hash with at least `difficulty` leading
/// zero bits. The `params_version` identifies which Argon2 parameters
/// were used, preventing silent verification failures on param mismatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct IdentityProof {
    /// The nonce that solves the puzzle.
    pub nonce: u64,
    /// The difficulty (leading zero bits) that was required.
    pub difficulty: u8,
    /// Identifies the Argon2 parameters used (must match verifier's params).
    pub params_version: u8,
}

// ── Core functions ──────────────────────────────────────────────────

/// Find a nonce such that `Argon2id(nonce, public_key_bytes)` has
/// `params.difficulty` leading zero bits.
///
/// The RNG provides a random starting nonce so concurrent solvers
/// don't collide. The search is sequential from that point.
pub fn solve(
    public_key_bytes: &[u8],
    params: &PuzzleParams,
    rng: &mut impl CryptoRngCore,
) -> IdentityProof {
    let argon2 = params.argon2();
    let mut nonce: u64 = {
        let mut buf = [0u8; 8];
        rng.fill_bytes(&mut buf);
        u64::from_le_bytes(buf)
    };

    loop {
        let hash = argon2_hash(&argon2, nonce, public_key_bytes);
        if has_leading_zero_bits(&hash, params.difficulty) {
            return IdentityProof {
                nonce,
                difficulty: params.difficulty,
                params_version: params.params_version,
            };
        }
        nonce = nonce.wrapping_add(1);
    }
}

/// Verify that a proof is valid for the given public key bytes.
///
/// Performs a single Argon2id evaluation (~50ms at production params).
/// Returns `false` if the proof's params_version doesn't match, the
/// difficulty is insufficient, or the hash doesn't have enough leading zeros.
pub fn verify(public_key_bytes: &[u8], proof: &IdentityProof, params: &PuzzleParams) -> bool {
    if proof.params_version != params.params_version {
        return false;
    }
    if proof.difficulty < params.difficulty {
        return false;
    }
    let argon2 = params.argon2();
    let hash = argon2_hash(&argon2, proof.nonce, public_key_bytes);
    has_leading_zero_bits(&hash, proof.difficulty)
}

// ── Helpers ─────────────────────────────────────────────────────────

/// Compute Argon2id(nonce_bytes, public_key_bytes) → 32-byte hash.
fn argon2_hash(argon2: &Argon2<'_>, nonce: u64, public_key_bytes: &[u8]) -> [u8; 32] {
    let mut output = [0u8; 32];
    argon2
        .hash_password_into(&nonce.to_le_bytes(), public_key_bytes, &mut output)
        .expect("output length 32 is always valid for Argon2");
    output
}

/// Check whether the first `n` bits of `hash` are zero.
fn has_leading_zero_bits(hash: &[u8], n: u8) -> bool {
    let full_bytes = (n / 8) as usize;
    let remaining_bits = n % 8;

    let needed = full_bytes + if remaining_bits > 0 { 1 } else { 0 };
    if hash.len() < needed {
        return false;
    }

    for byte in &hash[..full_bytes] {
        if *byte != 0 {
            return false;
        }
    }

    if remaining_bits > 0 {
        let mask = 0xFF << (8 - remaining_bits);
        if hash[full_bytes] & mask != 0 {
            return false;
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::OsRng;

    // ── has_leading_zero_bits ──────────────────────────────────────

    #[test]
    fn zero_difficulty_always_passes() {
        assert!(has_leading_zero_bits(&[0xFF; 32], 0));
    }

    #[test]
    fn one_leading_zero_bit() {
        assert!(has_leading_zero_bits(&[0x7F, 0xFF], 1));
        assert!(!has_leading_zero_bits(&[0x80, 0xFF], 1));
    }

    #[test]
    fn seven_leading_zero_bits() {
        assert!(has_leading_zero_bits(&[0x01, 0xFF], 7));
        assert!(!has_leading_zero_bits(&[0x02, 0xFF], 7));
    }

    #[test]
    fn eight_leading_zero_bits() {
        assert!(has_leading_zero_bits(&[0x00, 0xFF], 8));
        assert!(!has_leading_zero_bits(&[0x00, 0x80], 9));
        assert!(has_leading_zero_bits(&[0x00, 0x7F], 9));
    }

    #[test]
    fn sixteen_leading_zero_bits() {
        assert!(has_leading_zero_bits(&[0x00, 0x00, 0xFF], 16));
        assert!(!has_leading_zero_bits(&[0x00, 0x01, 0xFF], 16));
    }

    #[test]
    fn hash_too_short_for_difficulty() {
        assert!(!has_leading_zero_bits(&[0x00], 9));
    }

    // ── solve + verify ────────────────────────────────────────────

    #[test]
    fn solve_produces_valid_proof() {
        let pub_key = [0xAA; 64];
        let proof = solve(&pub_key, &PuzzleParams::TEST, &mut OsRng);
        assert!(verify(&pub_key, &proof, &PuzzleParams::TEST));
    }

    #[test]
    fn verify_rejects_wrong_public_key() {
        // d=8 → 1/256 false-pass probability, safe for CI.
        let params = PuzzleParams {
            difficulty: 8,
            ..PuzzleParams::TEST
        };
        let pub_key = [0xAA; 64];
        let proof = solve(&pub_key, &params, &mut OsRng);
        let wrong_key = [0xBB; 64];
        assert!(!verify(&wrong_key, &proof, &params));
    }

    #[test]
    fn verify_rejects_insufficient_difficulty() {
        let pub_key = [0xAA; 64];
        let params_easy = PuzzleParams {
            difficulty: 1,
            ..PuzzleParams::TEST
        };
        let proof = solve(&pub_key, &params_easy, &mut OsRng);
        let params_hard = PuzzleParams {
            difficulty: 4,
            ..PuzzleParams::TEST
        };
        // proof.difficulty is 1, requirement is 4 → rejected
        assert!(!verify(&pub_key, &proof, &params_hard));
    }

    #[test]
    fn harder_proof_accepted_at_easier_difficulty() {
        let pub_key = [0xAA; 64];
        let params_hard = PuzzleParams {
            difficulty: 4,
            ..PuzzleParams::TEST
        };
        let proof = solve(&pub_key, &params_hard, &mut OsRng);
        // d=4 proof should satisfy d=1 requirement
        assert!(verify(&pub_key, &proof, &PuzzleParams::TEST));
    }

    #[test]
    fn proof_serde_round_trip() {
        let proof = IdentityProof {
            nonce: 0xDEAD_BEEF_CAFE_BABE,
            difficulty: 7,
            params_version: 1,
        };
        let bytes = postcard::to_allocvec(&proof).unwrap();
        let decoded: IdentityProof = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(decoded, proof);
    }

    #[test]
    fn proof_deterministic_verify() {
        let pub_key = [0xCC; 64];
        let proof = solve(&pub_key, &PuzzleParams::TEST, &mut OsRng);
        let r1 = verify(&pub_key, &proof, &PuzzleParams::TEST);
        let r2 = verify(&pub_key, &proof, &PuzzleParams::TEST);
        assert_eq!(r1, r2);
        assert!(r1);
    }

    #[test]
    fn large_salt_works() {
        // PQ keys are ~3136 bytes — verify the puzzle works with large salts
        let pub_key = vec![0x55u8; 3136];
        let proof = solve(&pub_key, &PuzzleParams::TEST, &mut OsRng);
        assert!(verify(&pub_key, &proof, &PuzzleParams::TEST));
    }

    // ── params_version ────────────────────────────────────────────

    #[test]
    fn verify_rejects_params_version_mismatch() {
        let pub_key = [0xAA; 64];
        let proof = solve(&pub_key, &PuzzleParams::TEST, &mut OsRng);
        // Verify with PRODUCTION params (version 1 vs proof's version 0)
        assert!(!verify(&pub_key, &proof, &PuzzleParams::PRODUCTION));
    }

    // ── PuzzleParams::new ─────────────────────────────────────────

    #[test]
    fn new_with_valid_params_succeeds() {
        let params = PuzzleParams::new(256, 1, 1, 1, 0);
        assert!(params.is_ok());
    }

    #[test]
    fn new_with_zero_time_cost_fails() {
        let params = PuzzleParams::new(256, 0, 1, 1, 0);
        assert!(params.is_err());
    }

    #[test]
    fn new_with_zero_parallelism_fails() {
        let params = PuzzleParams::new(256, 1, 0, 1, 0);
        assert!(params.is_err());
    }
}
