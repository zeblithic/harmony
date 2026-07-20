# ZEB-411 — Quorum signer cert-expiry gate: Implementation Plan

**Goal:** Reject expired signer enrollments in all three local quorum walk-backs, matching the peer verifiers.

**Architecture:** Add the `EnrollmentCert::verify`-style `expires_at` block (returning `OwnerError::EnrollmentCertExpired`) at each hand-rolled signer walk-back. No API/signature changes; purely an added rejection path.

**Tech Stack:** Rust, `harmony-owner` crate. Time is a `u64` `now` parameter (no clocks/sleeps).

## Global Constraints

- Error must be `OwnerError::EnrollmentCertExpired { expires_at, now_secs }` (mirror peer `verify`).
- Place the new check adjacent to the existing `issued_at` "Backdated-Signer" guard at each site.
- Gates (CI parity): `cargo fmt --all -- --check`; `cargo clippy --locked -p harmony-owner --all-targets -- -D warnings`; `cargo nextest run --locked -p harmony-owner --all-targets`.
- One commit per task (TDD: test → run-fail → impl → run-pass → commit).

---

### Task 1: `add_enrollment` quorum walk-back (`state.rs`)

**Files:** Modify `crates/harmony-owner/src/state.rs` (loop at ~255–289); Test: same file `#[cfg(test)] mod tests`.

- [ ] **Step 1 — failing test** `add_enrollment_quorum_rejects_expired_signer`: mint owner (device A, Master, non-expiring). Build a second Master signer B via `EnrollmentCert::sign_master(.., expires_at = Some(T))`; `add_enrollment(B, now=T-…, window)`; publish fresh liveness for A and B at `now2 = T+…`. Attempt a quorum enroll of device C signed by [A, B] applied at `now2 > T` → assert `Err(OwnerError::EnrollmentCertExpired { .. })`.
- [ ] **Step 2 — run, expect FAIL** (`cargo nextest run -p harmony-owner add_enrollment_quorum_rejects_expired_signer`): currently accepts (or fails on a later guard), not `EnrollmentCertExpired`.
- [ ] **Step 3 — implement:** in the `for (signer_id, sig) in signers.iter().zip(signatures.iter())` loop, after the `issued_at > cert.issued_at` "Backdated-Signer" check, insert:

```rust
if let Some(exp) = signer_enrollment.expires_at {
    if now > exp {
        return Err(OwnerError::EnrollmentCertExpired { expires_at: exp, now_secs: now });
    }
}
```

- [ ] **Step 4 — run, expect PASS.** Add a positive guard `add_enrollment_quorum_accepts_unexpired_signer` (apply at `now2 < T`) → `Ok(())`.
- [ ] **Step 5 — commit** `ZEB-411: expiry-check quorum signers in add_enrollment walk-back`.

### Task 2: `add_revocation` quorum walk-back (`state.rs`)

**Files:** Modify `crates/harmony-owner/src/state.rs` (loop at ~463–497); Test: same module.

- [ ] **Step 1 — failing test** `add_revocation_quorum_rejects_expired_signer`: same signer setup (A + expiring B, both live); assemble a quorum `RevocationCert` targeting some device, signed by [A, B]; `add_revocation(cert, now2 > T, window)` → assert `Err(OwnerError::EnrollmentCertExpired { .. })`.
- [ ] **Step 2 — run, expect FAIL.**
- [ ] **Step 3 — implement:** in the revocation quorum `for (signer_id, sig) …` loop, after the `issued_at > cert.issued_at` "Revocation-Quorum-Backdated-Signer" check, insert the identical `expires_at` block (uses `now`).
- [ ] **Step 4 — run, expect PASS.** Positive guard at `now2 < T`.
- [ ] **Step 5 — commit** `ZEB-411: expiry-check quorum signers in add_revocation walk-back`.

### Task 3: `enroll_via_quorum` fail-fast (`lifecycle/enroll_quorum.rs`)

**Files:** Modify `crates/harmony-owner/src/lifecycle/enroll_quorum.rs` (loop at ~42–64); Test: same module.

- [ ] **Step 1 — failing test** `enroll_via_quorum_rejects_expired_signer`: same setup; call `enroll_via_quorum(state, [(a_sk, A), (b_sk, B)], …, now2 > T, window)` → assert `Err(OwnerError::EnrollmentCertExpired { .. })` (fail-fast, before signing).
- [ ] **Step 2 — run, expect FAIL** (currently discards `_enrollment`).
- [ ] **Step 3 — implement:** rename `let _enrollment = state.enrollments.get(id)…?` to `let enrollment = …?` and, after the active-signer check, insert the `expires_at` block (uses `now`).
- [ ] **Step 4 — run, expect PASS.** Positive guard at `now2 < T`.
- [ ] **Step 5 — commit** `ZEB-411: expiry-check quorum signers in enroll_via_quorum fail-fast`.

### Task 4: Full gates + docs + PR

- [ ] `cargo fmt --all -- --check` → clean.
- [ ] `cargo clippy --locked -p harmony-owner --all-targets -- -D warnings` → clean.
- [ ] `cargo nextest run --locked -p harmony-owner --all-targets` → green.
- [ ] Commit spec+plan docs (this file + the design doc) if not already committed.
- [ ] Push; open PR on `zeblithic/harmony`; trigger review; converge to CI-green + threads-resolved. Do **not** auto-merge.
