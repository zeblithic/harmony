# ZEB-411 — Expiry-check quorum signer enrollments (close local↔peer divergence)

**Status:** design approved 2026-07-20 (Jake). **Crate:** `harmony-owner`.
**Origin:** CodeAnt finding during ZEB-378 (harmony#276) review, filed as a follow-up.

## Problem

`OwnerState`'s quorum-issued cert acceptance walks back over each signer's
enrollment and hand-rolls its validity checks. Those walk-backs check the signer
for **presence / non-revocation / Master-issuer / `issued_at` ordering / signature**
— but **not** whether the signer's own enrollment cert has **expired** (`expires_at < now`).

This is not merely theoretical defense-in-depth. The **peer-side** verifiers —
`EnrollmentCert::verify_quorum_with_signers` (`certs/enrollment.rs:269`) and
`RevocationCert::verify_quorum_with_signers` (`certs/revocation.rs:322`) — each call
`signer_cert.verify(now)`, which **does** enforce `expires_at`. So the *local* apply
path admits a quorum cert that *peer* verification rejects: a **local↔peer
divergence**. The `add_enrollment` walk-back's own comment states its purpose is
"or state would admit certs that peer verification rejects" — the expiry gate is the
one guard missing from a walk-back explicitly built to mirror peer verification.

The existing **active-signer** check (`active_devices`) does not cover this: it
filters on *liveness recency + non-revocation only*, never enrollment `expires_at`.
A signer with a fresh liveness cert but an expired enrollment passes it.

### Why it was latent (and still low-urgency)

Quorum certs are unused in alpha (Master-only mints), and every production Master
enrollment is minted with `expires_at: None` (`mint`, `enroll_via_master`). But
`EnrollmentCert::sign_master` accepts `expires_at: Option<u64>`, so the type system
permits expiring Master certs, and the local↔peer split is a real correctness gap
regardless of current call sites.

## The fix

At each local walk-back, after resolving `signer_enrollment`, add — adjacent to the
existing `issued_at` "Backdated-Signer" check — the same expiry block
`EnrollmentCert::verify` uses:

```rust
if let Some(exp) = signer_enrollment.expires_at {
    if now > exp {
        return Err(OwnerError::EnrollmentCertExpired { expires_at: exp, now_secs: now });
    }
}
```

**Error choice:** return `OwnerError::EnrollmentCertExpired { .. }` — exactly what the
peer-side `signer_cert.verify(now)` returns for the same condition — so local and peer
paths surface the *same* error. That is the point of the fix (kill the divergence),
and it matches how the other walk-back guards already mirror peer error types 1:1
(`Signer-Not-Master`, `Backdated-Signer`).

## Sites (all three — one PR)

| # | Site | Location | Peer verifier it must mirror |
|---|------|----------|------------------------------|
| 1 | `OwnerState::add_enrollment` quorum walk-back | `state.rs` ~255–289 | `enrollment.rs:269` |
| 2 | `OwnerState::add_revocation` quorum walk-back | `state.rs` ~463–497 | `revocation.rs:322` |
| 3 | `enroll_via_quorum` fail-fast pre-check | `lifecycle/enroll_quorum.rs` ~42–64 | (mirrors #1 at construction) |

Site #3 is a construction-time fail-fast that intentionally surfaces the same errors
`add_enrollment` would raise at apply time, so callers get an error before signing —
it must gate expiry to stay consistent with #1. Site #2 was not named in the original
ticket because quorum *revocation* (ZEB-677 S1) postdates the ticket; it has the
identical gap and the identical peer divergence, so it is in scope.

## Testing (TDD)

Per CodeAnt's repro, one failing test + one positive guard per site:

- **Expired signer rejected:** mint owner; add a Master-issued signer `B` with
  `expires_at: Some(T)` (accepted at a `now < T`); publish fresh liveness for `B` so it
  passes the active-signer check; advance to `now2 > T`; attempt the quorum operation
  with `B` as a signer → assert `OwnerError::EnrollmentCertExpired`.
- **Non-expired signer still accepted:** same setup at `now2 < T` → succeeds
  (guards against over-rejection).
- Site #3 additionally: the fail-fast rejects before signing.

No wall-clock sleeps — `now` is a plain `u64` parameter throughout.

## Rollout (two PRs, in order)

1. **`zeblithic/harmony`** — this fix + tests, green CI, merge.
2. **`zeblithic/harmony-client`** — bump the `harmony-owner` git `rev`
   (`1ecb4160` → new merged commit) in `src-tauri/Cargo.toml`, green client CI.

The client rev-bump must point at a merged harmony commit, so it strictly follows (1).
ZEB-411 is labeled `harmony-client` but the substance lands in `harmony`.
