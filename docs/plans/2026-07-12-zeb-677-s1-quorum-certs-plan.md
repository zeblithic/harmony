# ZEB-677 S1: Quorum cert primitives (harmony-owner) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `harmony-owner` the distributed-quorum cert primitives (detached-signature payload exposure, part signing, assembly, and signer-cert-backed verification for both cert types, plus `OwnerState::add_revocation` quorum acceptance) that ZEB-677's client slices S2–S5 build on.

**Architecture:** Depth-1 chain carriage per `harmony-client/docs/specs/2026-07-12-zeb-677-quorum-wiring-design.md` §2: a quorum cert verifies against its K signer enrollment certs, each of which must be Master-issued. Signing payloads become public associated functions so signatures can be collected across devices; the existing private payload constructions in `state.rs` and `lifecycle/enroll_quorum.rs` are consolidated onto them so signing and verification cannot drift.

**Tech Stack:** Rust, ed25519-dalek (`verify_strict` via `signing::verify_with_tag`), canonical CBOR (`crate::cbor::to_canonical`), thiserror.

## Global Constraints

- All signatures are `tag || canonical_cbor(payload)` through `signing::{sign_with_tag, verify_with_tag}`; enrollment uses `tags::ENROLLMENT`, revocation uses `tags::REVOCATION` — never raw signing.
- Quorum minimum is 2 (`OwnerError::InsufficientQuorum { min: 2, .. }`), signers distinct, signers/signatures parity — same checks everywhere quorum structure is touched.
- Depth-1: every quorum **signer's** enrollment cert must be `EnrollmentIssuer::Master` — reject `Quorum`-issued signer certs.
- Timestamps (`issued_at`, `expires_at`, `now_secs`, `active_window_secs`) are Unix **seconds** (doc comment on `EnrollmentCert::verify`).
- No new dependencies; errors are existing-or-new `OwnerError` variants; no `unwrap()` outside tests.
- Repo gates (CI parity, `.github/workflows/ci.yml`): `cargo fmt --all -- --check`; `cargo clippy --locked --workspace --all-targets -- -D warnings`; `cargo nextest run --locked --workspace`.
- Work on branch `zeb-677-s1-quorum-certs` off `origin/main` (`25641d8`); repo root `/Users/zeblith/work/zeblithic/harmony`.

---

### Task 1: Public quorum payload + part-sign + assemble for `EnrollmentCert` (and DRY the two private copies onto it)

**Files:**
- Modify: `crates/harmony-owner/src/certs/enrollment.rs`
- Modify: `crates/harmony-owner/src/state.rs:418-447` (private `quorum_signing_payload` becomes a thin call)
- Modify: `crates/harmony-owner/src/lifecycle/enroll_quorum.rs:77-94` (inline payload construction becomes a call)

**Interfaces:**
- Produces (Tasks 2, 4, and client S3/S4 rely on these exact signatures):
  - `EnrollmentCert::quorum_signing_payload_bytes(owner_id: [u8;16], device_id: [u8;16], device_pubkeys: &PubKeyBundle, issued_at: u64, expires_at: Option<u64>, signers: &[[u8;16]]) -> Result<Vec<u8>, OwnerError>`
  - `EnrollmentCert::sign_quorum_part(sk: &SigningKey, payload_bytes: &[u8]) -> Vec<u8>`
  - `EnrollmentCert::assemble_quorum(owner_id: [u8;16], device_id: [u8;16], device_pubkeys: PubKeyBundle, issued_at: u64, expires_at: Option<u64>, parts: Vec<([u8;16], Vec<u8>)>) -> Result<EnrollmentCert, OwnerError>`

- [ ] **Step 1: Write the failing tests** (append to `mod tests` in `enrollment.rs`)

```rust
    #[test]
    fn assembled_quorum_cert_matches_shape_and_passes_structural_verify() {
        let (a_sk, a_bundle) = fresh_pubkey_bundle(1, 2);
        let (b_sk, b_bundle) = fresh_pubkey_bundle(3, 4);
        let (_new_sk, new_bundle) = fresh_pubkey_bundle(5, 6);
        let owner_id = [7u8; 16];
        let new_id = new_bundle.identity_hash();
        let signers = [a_bundle.identity_hash(), b_bundle.identity_hash()];

        let payload = EnrollmentCert::quorum_signing_payload_bytes(
            owner_id, new_id, &new_bundle, 1_000, None, &signers,
        )
        .unwrap();
        let parts = vec![
            (signers[0], EnrollmentCert::sign_quorum_part(&a_sk, &payload)),
            (signers[1], EnrollmentCert::sign_quorum_part(&b_sk, &payload)),
        ];
        let cert =
            EnrollmentCert::assemble_quorum(owner_id, new_id, new_bundle, 1_000, None, parts)
                .unwrap();
        assert!(matches!(&cert.issuer, EnrollmentIssuer::Quorum { signers: s, signatures } if s.len() == 2 && signatures.len() == 2));
        assert!(cert.signature.is_empty()); // quorum sigs live in the issuer
        cert.verify(2_000).unwrap(); // structural verify passes
    }

    #[test]
    fn assemble_quorum_rejects_single_part_and_duplicate_signers() {
        let (a_sk, a_bundle) = fresh_pubkey_bundle(1, 2);
        let (_new_sk, new_bundle) = fresh_pubkey_bundle(5, 6);
        let owner_id = [7u8; 16];
        let new_id = new_bundle.identity_hash();
        let a_id = a_bundle.identity_hash();

        let payload = EnrollmentCert::quorum_signing_payload_bytes(
            owner_id, new_id, &new_bundle, 1_000, None, &[a_id],
        )
        .unwrap();
        let one = vec![(a_id, EnrollmentCert::sign_quorum_part(&a_sk, &payload))];
        assert!(matches!(
            EnrollmentCert::assemble_quorum(owner_id, new_id, new_bundle.clone(), 1_000, None, one),
            Err(OwnerError::InsufficientQuorum { min: 2, got: 1 })
        ));

        let payload2 = EnrollmentCert::quorum_signing_payload_bytes(
            owner_id, new_id, &new_bundle, 1_000, None, &[a_id, a_id],
        )
        .unwrap();
        let dup = vec![
            (a_id, EnrollmentCert::sign_quorum_part(&a_sk, &payload2)),
            (a_id, EnrollmentCert::sign_quorum_part(&a_sk, &payload2)),
        ];
        assert!(EnrollmentCert::assemble_quorum(owner_id, new_id, new_bundle, 1_000, None, dup)
            .is_err());
    }
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p harmony-owner --lib certs::enrollment -- assembled_quorum assemble_quorum 2>&1 | tail -5`
Expected: compile FAIL — `quorum_signing_payload_bytes` not found.

- [ ] **Step 3: Implement** (in `impl EnrollmentCert`, after `verify`)

```rust
    /// Canonical detached-signature payload for a Quorum-issued enrollment
    /// cert. The signer set is part of the payload, so it must be fixed
    /// before any part is signed. Public so the ceremony can collect
    /// signatures across devices (ZEB-677); `OwnerState` and
    /// `enroll_via_quorum` use the same function, so signing and
    /// verification cannot drift.
    pub fn quorum_signing_payload_bytes(
        owner_id: [u8; 16],
        device_id: [u8; 16],
        device_pubkeys: &PubKeyBundle,
        issued_at: u64,
        expires_at: Option<u64>,
        signers: &[[u8; 16]],
    ) -> Result<Vec<u8>, OwnerError> {
        let issuer_data = cbor::to_canonical(&signers.to_vec())?;
        cbor::to_canonical(&EnrollmentSigningPayload {
            version: ENROLLMENT_VERSION,
            owner_id,
            device_id,
            device_pubkeys,
            issued_at,
            expires_at,
            issuer_kind: 1, // Quorum
            issuer_data,
        })
    }

    /// Sign one quorum part over `quorum_signing_payload_bytes` output with
    /// the correct domain tag. Thin wrapper so callers never touch raw tags.
    pub fn sign_quorum_part(sk: &SigningKey, payload_bytes: &[u8]) -> Vec<u8> {
        sign_with_tag(sk, tags::ENROLLMENT, payload_bytes)
    }

    /// Assemble a Quorum-issued cert from independently collected
    /// `(signer_device_id, detached_signature)` parts. Performs the
    /// structural quorum checks; full signature verification requires the
    /// signer certs (`verify_quorum_with_signers`) or `OwnerState`.
    pub fn assemble_quorum(
        owner_id: [u8; 16],
        device_id: [u8; 16],
        device_pubkeys: PubKeyBundle,
        issued_at: u64,
        expires_at: Option<u64>,
        parts: Vec<([u8; 16], Vec<u8>)>,
    ) -> Result<Self, OwnerError> {
        if parts.len() < 2 {
            return Err(OwnerError::InsufficientQuorum {
                min: 2,
                got: parts.len(),
            });
        }
        let (signers, signatures): (Vec<[u8; 16]>, Vec<Vec<u8>>) = parts.into_iter().unzip();
        let cert = EnrollmentCert {
            version: ENROLLMENT_VERSION,
            owner_id,
            device_id,
            device_pubkeys,
            issued_at,
            expires_at,
            issuer: EnrollmentIssuer::Quorum {
                signers,
                signatures,
            },
            signature: Vec::new(),
        };
        // Structural checks (distinctness, parity, device-id binding) live in
        // verify(); expiry is irrelevant at issuance time, so verify at
        // issued_at.
        cert.verify(issued_at)?;
        Ok(cert)
    }
```

Note: `cbor::to_canonical(&signers.to_vec())` — the existing payload sites serialize `Vec<[u8;16]>`/`&Vec` with the `arr16_vec` field attr on the issuer, but here the standalone value is CBOR-encoded exactly as `state.rs:435` (`cbor::to_canonical(signers)` where `signers: &Vec<[u8;16]>`) and `enroll_quorum.rs:78` (`&signers` where `signers: Vec<[u8;16]>`) do today. Serializing `[[u8;16]]` as a slice of arrays must produce identical bytes to `Vec<[u8;16]>` — if serde's default array-of-16 encoding differs from the `arr16_vec` helper used inside structs, match the existing sites exactly by converting to the same type first (`signers.to_vec()`), which this code does.

- [ ] **Step 4: DRY the two private copies.** In `state.rs`, replace the body of `fn quorum_signing_payload` (`:421-447`) with:

```rust
fn quorum_signing_payload(cert: &EnrollmentCert) -> Result<Vec<u8>, OwnerError> {
    let signers = match &cert.issuer {
        EnrollmentIssuer::Quorum { signers, .. } => signers,
        _ => {
            return Err(OwnerError::InvalidSignature {
                cert_type: "Enrollment-Quorum-Member",
            })
        }
    };
    EnrollmentCert::quorum_signing_payload_bytes(
        cert.owner_id,
        cert.device_id,
        &cert.device_pubkeys,
        cert.issued_at,
        cert.expires_at,
        signers,
    )
}
```

In `enroll_quorum.rs`, replace lines 77-89 (the `issuer_data` + `payload_bytes` construction) with:

```rust
    let payload_bytes = EnrollmentCert::quorum_signing_payload_bytes(
        state.owner_id,
        device_id,
        &new_device_pubkey,
        now,
        None,
        &signers,
    )?;
```

and drop the now-unused `cbor` / `EnrollmentSigningPayload` imports from that file. Remove unused imports flagged by clippy in `state.rs` too (the `use crate::cbor;`/`EnrollmentSigningPayload` inside the old fn body).

- [ ] **Step 5: Run the crate tests**

Run: `cargo nextest run --locked -p harmony-owner 2>&1 | tail -5`
Expected: PASS — all existing quorum tests (payload refactor is byte-identical) + the 2 new tests.

- [ ] **Step 6: Commit**

```bash
git add -A crates/harmony-owner && git commit -m "ZEB-677 S1: public quorum payload/part-sign/assemble for EnrollmentCert

Consolidates the two private payload constructions (state.rs,
enroll_quorum.rs) onto the new public function so distributed ceremony
signing (client S3/S4) uses byte-identical payloads."
```

---

### Task 2: `EnrollmentCert::verify_quorum_with_signers` (peer-side chain verification)

**Files:**
- Modify: `crates/harmony-owner/src/certs/enrollment.rs`

**Interfaces:**
- Consumes: Task 1's `quorum_signing_payload_bytes`.
- Produces (client S2's `verify_enrollment_any_issuer` chokepoint calls this):
  - `EnrollmentCert::verify_quorum_with_signers(&self, signer_certs: &[EnrollmentCert], now_secs: u64) -> Result<(), OwnerError>`

- [ ] **Step 1: Write the failing tests** (append to `mod tests`; the helper builds a valid 2-signer world)

```rust
    /// Master-enrolls devices A and B under one owner, quorum-enrolls a new
    /// device C signed by A+B, and returns everything a chain verifier needs.
    fn quorum_world() -> (
        [u8; 16],            // owner_id
        EnrollmentCert,      // quorum cert for C
        Vec<EnrollmentCert>, // signer certs [A, B] (Master-issued)
        SigningKey,          // A's device sk (for tamper tests)
    ) {
        let (master_sk, master_bundle) = fresh_pubkey_bundle(1, 2);
        let owner_id = master_bundle.identity_hash();
        let (a_sk, a_bundle) = fresh_pubkey_bundle(3, 4);
        let (b_sk, b_bundle) = fresh_pubkey_bundle(5, 6);
        let (_c_sk, c_bundle) = fresh_pubkey_bundle(7, 8);
        let a_cert = EnrollmentCert::sign_master(
            &master_sk, master_bundle.clone(), a_bundle.identity_hash(), a_bundle.clone(),
            100, None,
        )
        .unwrap();
        let b_cert = EnrollmentCert::sign_master(
            &master_sk, master_bundle, b_bundle.identity_hash(), b_bundle.clone(), 100, None,
        )
        .unwrap();
        let signers = [a_bundle.identity_hash(), b_bundle.identity_hash()];
        let c_id = c_bundle.identity_hash();
        let payload = EnrollmentCert::quorum_signing_payload_bytes(
            owner_id, c_id, &c_bundle, 1_000, None, &signers,
        )
        .unwrap();
        let parts = vec![
            (signers[0], EnrollmentCert::sign_quorum_part(&a_sk, &payload)),
            (signers[1], EnrollmentCert::sign_quorum_part(&b_sk, &payload)),
        ];
        let c_cert =
            EnrollmentCert::assemble_quorum(owner_id, c_id, c_bundle, 1_000, None, parts).unwrap();
        (owner_id, c_cert, vec![a_cert, b_cert], a_sk)
    }

    #[test]
    fn quorum_chain_verifies_with_signer_certs() {
        let (_owner, cert, signer_certs, _a_sk) = quorum_world();
        cert.verify_quorum_with_signers(&signer_certs, 2_000).unwrap();
    }

    #[test]
    fn quorum_chain_rejects_missing_signer_cert() {
        let (_owner, cert, signer_certs, _a_sk) = quorum_world();
        let only_a = &signer_certs[..1];
        assert!(matches!(
            cert.verify_quorum_with_signers(only_a, 2_000),
            Err(OwnerError::NotEnrolled { .. })
        ));
    }

    #[test]
    fn quorum_chain_rejects_tampered_signature() {
        let (_owner, mut cert, signer_certs, _a_sk) = quorum_world();
        if let EnrollmentIssuer::Quorum { signatures, .. } = &mut cert.issuer {
            signatures[0][0] ^= 0xFF;
        }
        assert!(matches!(
            cert.verify_quorum_with_signers(&signer_certs, 2_000),
            Err(OwnerError::InvalidSignature { .. })
        ));
    }

    #[test]
    fn quorum_chain_rejects_wrong_owner_signer_cert() {
        let (_owner, cert, mut signer_certs, _a_sk) = quorum_world();
        // Re-issue A's cert under a different owner (breaks its own verify
        // via identity-hash mismatch is not enough — construct a different
        // master so the cert itself is valid but foreign).
        let (other_master_sk, other_master_bundle) = fresh_pubkey_bundle(9, 10);
        let a_pub = signer_certs[0].device_pubkeys.clone();
        let a_id = signer_certs[0].device_id;
        signer_certs[0] = EnrollmentCert::sign_master(
            &other_master_sk, other_master_bundle, a_id, a_pub, 100, None,
        )
        .unwrap();
        assert!(matches!(
            cert.verify_quorum_with_signers(&signer_certs, 2_000),
            Err(OwnerError::WrongOwner { .. })
        ));
    }

    #[test]
    fn quorum_chain_rejects_quorum_issued_signer_cert_depth1() {
        let (owner_id, cert, mut signer_certs, a_sk) = quorum_world();
        // Replace A's Master cert with a (structurally valid) Quorum-issued
        // one for the same device: depth-1 violation.
        let a = signer_certs[0].clone();
        let b_id = signer_certs[1].device_id;
        let payload = EnrollmentCert::quorum_signing_payload_bytes(
            owner_id, a.device_id, &a.device_pubkeys, 100, None, &[a.device_id, b_id],
        )
        .unwrap();
        let parts = vec![
            (a.device_id, EnrollmentCert::sign_quorum_part(&a_sk, &payload)),
            (b_id, EnrollmentCert::sign_quorum_part(&a_sk, &payload)),
        ];
        signer_certs[0] = EnrollmentCert::assemble_quorum(
            owner_id, a.device_id, a.device_pubkeys.clone(), 100, None, parts,
        )
        .unwrap();
        let err = cert.verify_quorum_with_signers(&signer_certs, 2_000);
        assert!(
            matches!(err, Err(OwnerError::InvalidSignature { cert_type }) if cert_type == "Enrollment-Quorum-Signer-Not-Master")
        );
    }

    #[test]
    fn quorum_chain_rejects_expired_signer_cert() {
        let (_owner, cert, signer_certs, _a_sk) = quorum_world();
        // signer certs issued at 100 with expires_at None are fine; rebuild
        // world manually is heavy — instead verify at a time where a
        // synthetic expiring signer cert is expired.
        let (master_sk, master_bundle) = fresh_pubkey_bundle(11, 12);
        let _ = (master_sk, master_bundle);
        // Simpler: mutate a copy's expires_at — signature breaks, but the
        // expiry check must fire FIRST for the deterministic error.
        let mut certs = signer_certs.clone();
        certs[0].expires_at = Some(150);
        assert!(matches!(
            cert.verify_quorum_with_signers(&certs, 2_000),
            Err(OwnerError::EnrollmentCertExpired { .. })
        ));
    }

    #[test]
    fn quorum_chain_rejects_backdated_signer_and_master_cert_input() {
        let (_owner, cert, mut signer_certs, _a_sk) = quorum_world();
        // Master-issued cert may not be passed to the quorum verifier.
        assert!(signer_certs[0]
            .verify_quorum_with_signers(&signer_certs, 2_000)
            .is_err());
        // Backdated: signer enrolled AFTER the quorum cert was issued.
        // quorum_world issues signers at 100 and the cert at 1_000; forge a
        // later-issued signer cert (re-sign under the same master is not
        // available here, so assert via issued_at mutation → expiry-ordering
        // guard must reject before signature validation).
        signer_certs[0].issued_at = 5_000;
        let err = cert.verify_quorum_with_signers(&signer_certs, 6_000);
        assert!(err.is_err());
    }
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p harmony-owner --lib certs::enrollment -- quorum_chain 2>&1 | tail -5`
Expected: compile FAIL — `verify_quorum_with_signers` not found.

- [ ] **Step 3: Implement** (in `impl EnrollmentCert`)

```rust
    /// Peer-side full verification of a Quorum-issued cert using presented
    /// signer enrollment certs (depth-1 chain carriage, ZEB-677 §2). Checks,
    /// in order: structural validity + own expiry (`verify`); Quorum issuer
    /// required; then per signer — a matching presented cert exists, is
    /// bound to the same owner, is Master-issued (depth-1), passes its own
    /// full `verify(now_secs)` (expiry checked before signatures so the
    /// error is deterministic), was not issued after this cert (backdating),
    /// and its enrolled ed25519 key validates the quorum part signature.
    /// Extra presented certs are ignored.
    pub fn verify_quorum_with_signers(
        &self,
        signer_certs: &[EnrollmentCert],
        now_secs: u64,
    ) -> Result<(), OwnerError> {
        self.verify(now_secs)?;
        let (signers, signatures) = match &self.issuer {
            EnrollmentIssuer::Quorum {
                signers,
                signatures,
            } => (signers, signatures),
            EnrollmentIssuer::Master { .. } => {
                return Err(OwnerError::InvalidSignature {
                    cert_type: "Enrollment-Quorum-Expected",
                })
            }
        };
        let payload_bytes = Self::quorum_signing_payload_bytes(
            self.owner_id,
            self.device_id,
            &self.device_pubkeys,
            self.issued_at,
            self.expires_at,
            signers,
        )?;
        for (signer_id, sig) in signers.iter().zip(signatures.iter()) {
            let signer_cert = signer_certs
                .iter()
                .find(|c| c.device_id == *signer_id)
                .ok_or(OwnerError::NotEnrolled {
                    owner: self.owner_id,
                    device: *signer_id,
                })?;
            if signer_cert.owner_id != self.owner_id {
                return Err(OwnerError::WrongOwner {
                    expected: self.owner_id,
                    got: signer_cert.owner_id,
                });
            }
            if !matches!(signer_cert.issuer, EnrollmentIssuer::Master { .. }) {
                return Err(OwnerError::InvalidSignature {
                    cert_type: "Enrollment-Quorum-Signer-Not-Master",
                });
            }
            signer_cert.verify(now_secs)?;
            if signer_cert.issued_at > self.issued_at {
                return Err(OwnerError::InvalidSignature {
                    cert_type: "Enrollment-Quorum-Backdated-Signer",
                });
            }
            let vk = VerifyingKey::from_bytes(&signer_cert.device_pubkeys.classical.ed25519_verify)
                .map_err(|_| OwnerError::InvalidSignature {
                    cert_type: "Enrollment-Quorum-Member",
                })?;
            verify_with_tag(
                &vk,
                tags::ENROLLMENT,
                &payload_bytes,
                sig,
                "Enrollment-Quorum-Member",
            )?;
        }
        Ok(())
    }
```

Ordering note: the expiry check inside `signer_cert.verify(now_secs)` runs before the signature check in that function, so `quorum_chain_rejects_expired_signer_cert`'s mutated cert deterministically yields `EnrollmentCertExpired` (the mutation also breaks the signature, but expiry fires first).

- [ ] **Step 4: Run tests**

Run: `cargo nextest run --locked -p harmony-owner 2>&1 | tail -5`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A crates/harmony-owner && git commit -m "ZEB-677 S1: EnrollmentCert::verify_quorum_with_signers (depth-1 chain verification)"
```

---

### Task 3: `RevocationCert` quorum: payload, part-sign, assemble, chain verify; retire the `NotImplemented` error

**Files:**
- Modify: `crates/harmony-owner/src/certs/revocation.rs`
- Modify: `crates/harmony-owner/src/error.rs:32-33` (variant rename)
- Modify: `crates/harmony-owner/src/state.rs:389-391` (compile fix only here; full quorum arm is Task 4)

**Interfaces:**
- Consumes: `signing::{sign_with_tag, verify_with_tag}`, `tags::REVOCATION`; Task 2's error-shape conventions.
- Produces (Task 4 and client S2/S3 rely on):
  - `RevocationCert::quorum_signing_payload_bytes(owner_id: [u8;16], target: [u8;16], issued_at: u64, reason: &RevocationReason, signers: &[[u8;16]]) -> Result<Vec<u8>, OwnerError>`
  - `RevocationCert::sign_quorum_part(sk: &SigningKey, payload_bytes: &[u8]) -> Vec<u8>`
  - `RevocationCert::assemble_quorum(owner_id: [u8;16], target: [u8;16], issued_at: u64, reason: RevocationReason, parts: Vec<([u8;16], Vec<u8>)>) -> Result<RevocationCert, OwnerError>`
  - `RevocationCert::verify_quorum_with_signers(&self, signer_certs: &[EnrollmentCert], now_secs: u64) -> Result<(), OwnerError>`
  - `OwnerError::QuorumRequiresSignerCerts` (replaces `QuorumRevocationNotImplemented`)

- [ ] **Step 1: Rename the error variant.** In `error.rs` replace:

```rust
    #[error("quorum revocation not yet implemented (use SelfDevice or Master variants)")]
    QuorumRevocationNotImplemented,
```

with:

```rust
    #[error(
        "quorum cert cannot be verified standalone — signer certs required (verify_quorum_with_signers)"
    )]
    QuorumRequiresSignerCerts,
```

Then `grep -rn QuorumRevocationNotImplemented crates/` and update the two production references (`revocation.rs:169`, `state.rs:390`) and the `state.rs` test `quorum_revocation_rejected_until_implemented` (rename its assertion to the new variant for now; Task 4 rewrites the test).

- [ ] **Step 2: Write the failing tests** (append to `mod tests` in `revocation.rs`)

```rust
    fn signer_world() -> (
        [u8; 16],
        Vec<crate::certs::EnrollmentCert>, // Master-issued certs for A, B
        SigningKey,                        // A sk
        SigningKey,                        // B sk
    ) {
        use crate::certs::EnrollmentCert;
        let master_sk = SigningKey::generate(&mut OsRng);
        let master_bundle = PubKeyBundle {
            classical: ClassicalKeys {
                ed25519_verify: master_sk.verifying_key().to_bytes(),
                x25519_pub: [0u8; 32],
            },
            post_quantum: None,
        };
        let owner_id = master_bundle.identity_hash();
        let mk = |seed: u8| -> (SigningKey, PubKeyBundle) {
            let sk = SigningKey::generate(&mut OsRng);
            let b = PubKeyBundle {
                classical: ClassicalKeys {
                    ed25519_verify: sk.verifying_key().to_bytes(),
                    x25519_pub: [seed; 32],
                },
                post_quantum: None,
            };
            (sk, b)
        };
        let (a_sk, a_bundle) = mk(1);
        let (b_sk, b_bundle) = mk(2);
        let a_cert = EnrollmentCert::sign_master(
            &master_sk, master_bundle.clone(), a_bundle.identity_hash(), a_bundle, 100, None,
        )
        .unwrap();
        let b_cert = EnrollmentCert::sign_master(
            &master_sk, master_bundle, b_bundle.identity_hash(), b_bundle, 100, None,
        )
        .unwrap();
        (owner_id, vec![a_cert, b_cert], a_sk, b_sk)
    }

    fn assemble_quorum_revocation(
        owner_id: [u8; 16],
        signer_certs: &[crate::certs::EnrollmentCert],
        a_sk: &SigningKey,
        b_sk: &SigningKey,
        target: [u8; 16],
    ) -> RevocationCert {
        let signers = [signer_certs[0].device_id, signer_certs[1].device_id];
        let payload = RevocationCert::quorum_signing_payload_bytes(
            owner_id, target, 1_000, &RevocationReason::Lost, &signers,
        )
        .unwrap();
        RevocationCert::assemble_quorum(
            owner_id,
            target,
            1_000,
            RevocationReason::Lost,
            vec![
                (signers[0], RevocationCert::sign_quorum_part(a_sk, &payload)),
                (signers[1], RevocationCert::sign_quorum_part(b_sk, &payload)),
            ],
        )
        .unwrap()
    }

    #[test]
    fn quorum_revocation_assembles_and_chain_verifies() {
        let (owner_id, signer_certs, a_sk, b_sk) = signer_world();
        let cert = assemble_quorum_revocation(owner_id, &signer_certs, &a_sk, &b_sk, [9u8; 16]);
        cert.verify_quorum_with_signers(&signer_certs, 2_000).unwrap();
        // Standalone verify still fails closed.
        assert!(matches!(
            cert.verify(None),
            Err(OwnerError::QuorumRequiresSignerCerts)
        ));
    }

    #[test]
    fn quorum_revocation_rejects_tamper_and_reason_binding() {
        let (owner_id, signer_certs, a_sk, b_sk) = signer_world();
        let mut cert = assemble_quorum_revocation(owner_id, &signer_certs, &a_sk, &b_sk, [9u8; 16]);
        cert.reason = RevocationReason::Compromised; // reason is in the payload
        assert!(cert.verify_quorum_with_signers(&signer_certs, 2_000).is_err());
    }

    #[test]
    fn quorum_revocation_rejects_cross_domain_signature() {
        // A signature made over the ENROLLMENT payload/tag must not verify as
        // a REVOCATION quorum part (domain separation: tag + issuer_kind).
        let (owner_id, signer_certs, a_sk, b_sk) = signer_world();
        let signers = [signer_certs[0].device_id, signer_certs[1].device_id];
        let target = [9u8; 16];
        let rev_payload = RevocationCert::quorum_signing_payload_bytes(
            owner_id, target, 1_000, &RevocationReason::Lost, &signers,
        )
        .unwrap();
        // Enrollment-domain signatures over the same bytes:
        let wrong = vec![
            (
                signers[0],
                crate::certs::EnrollmentCert::sign_quorum_part(&a_sk, &rev_payload),
            ),
            (
                signers[1],
                crate::certs::EnrollmentCert::sign_quorum_part(&b_sk, &rev_payload),
            ),
        ];
        let cert = RevocationCert::assemble_quorum(
            owner_id, target, 1_000, RevocationReason::Lost, wrong,
        )
        .unwrap();
        assert!(matches!(
            cert.verify_quorum_with_signers(&signer_certs, 2_000),
            Err(OwnerError::InvalidSignature { .. })
        ));
    }

    #[test]
    fn quorum_revocation_rejects_non_master_and_missing_signer_certs() {
        let (owner_id, signer_certs, a_sk, b_sk) = signer_world();
        let cert = assemble_quorum_revocation(owner_id, &signer_certs, &a_sk, &b_sk, [9u8; 16]);
        assert!(matches!(
            cert.verify_quorum_with_signers(&signer_certs[..1], 2_000),
            Err(OwnerError::NotEnrolled { .. })
        ));
    }
```

- [ ] **Step 3: Run to verify failure**

Run: `cargo test -p harmony-owner --lib certs::revocation 2>&1 | tail -5`
Expected: compile FAIL — new functions not found.

- [ ] **Step 4: Implement** (in `impl RevocationCert`; imports: add `use crate::certs::enrollment::EnrollmentCert;` and `EnrollmentIssuer` where needed)

```rust
    /// Canonical detached-signature payload for a Quorum-issued revocation
    /// (ZEB-677 §2). Signer set is part of the payload — fix it before
    /// signing. issuer_kind = 2 (Quorum) and tags::REVOCATION give domain
    /// separation from enrollment quorum parts.
    pub fn quorum_signing_payload_bytes(
        owner_id: [u8; 16],
        target: [u8; 16],
        issued_at: u64,
        reason: &RevocationReason,
        signers: &[[u8; 16]],
    ) -> Result<Vec<u8>, OwnerError> {
        let issuer_data = cbor::to_canonical(&signers.to_vec())?;
        cbor::to_canonical(&RevocationSigningPayload {
            version: REVOCATION_VERSION,
            owner_id,
            target,
            issued_at,
            issuer_kind: 2, // Quorum
            issuer_data,
            reason,
        })
    }

    /// Sign one quorum part with the revocation domain tag.
    pub fn sign_quorum_part(sk: &SigningKey, payload_bytes: &[u8]) -> Vec<u8> {
        sign_with_tag(sk, tags::REVOCATION, payload_bytes)
    }

    /// Assemble a Quorum-issued revocation from collected parts. Structural
    /// checks only (≥2, distinct); full verification needs signer certs.
    pub fn assemble_quorum(
        owner_id: [u8; 16],
        target: [u8; 16],
        issued_at: u64,
        reason: RevocationReason,
        parts: Vec<([u8; 16], Vec<u8>)>,
    ) -> Result<Self, OwnerError> {
        if parts.len() < 2 {
            return Err(OwnerError::InsufficientQuorum {
                min: 2,
                got: parts.len(),
            });
        }
        let unique: std::collections::HashSet<[u8; 16]> =
            parts.iter().map(|(id, _)| *id).collect();
        if unique.len() != parts.len() {
            return Err(OwnerError::InvalidSignature {
                cert_type: "Revocation-Quorum-Duplicate-Signer",
            });
        }
        let (signers, signatures): (Vec<[u8; 16]>, Vec<Vec<u8>>) = parts.into_iter().unzip();
        Ok(RevocationCert {
            version: REVOCATION_VERSION,
            owner_id,
            target,
            issued_at,
            issuer: RevocationIssuer::Quorum {
                signers,
                signatures,
            },
            reason,
            signature: Vec::new(),
        })
    }

    /// Peer-side full verification of a Quorum-issued revocation against
    /// presented signer enrollment certs. Same depth-1 policy as
    /// `EnrollmentCert::verify_quorum_with_signers`: each signer cert must
    /// be present, same-owner, Master-issued, valid at `now_secs`, and not
    /// issued after this revocation.
    pub fn verify_quorum_with_signers(
        &self,
        signer_certs: &[EnrollmentCert],
        now_secs: u64,
    ) -> Result<(), OwnerError> {
        if self.version != REVOCATION_VERSION {
            return Err(OwnerError::UnknownVersion(self.version));
        }
        let (signers, signatures) = match &self.issuer {
            RevocationIssuer::Quorum {
                signers,
                signatures,
            } => (signers, signatures),
            _ => {
                return Err(OwnerError::InvalidSignature {
                    cert_type: "Revocation-Quorum-Expected",
                })
            }
        };
        if signers.len() < 2 {
            return Err(OwnerError::InsufficientQuorum {
                min: 2,
                got: signers.len(),
            });
        }
        if signers.len() != signatures.len() {
            return Err(OwnerError::InvalidSignature {
                cert_type: "Revocation-Quorum-Length-Mismatch",
            });
        }
        let unique: std::collections::HashSet<[u8; 16]> = signers.iter().copied().collect();
        if unique.len() != signers.len() {
            return Err(OwnerError::InvalidSignature {
                cert_type: "Revocation-Quorum-Duplicate-Signer",
            });
        }
        let payload_bytes = Self::quorum_signing_payload_bytes(
            self.owner_id,
            self.target,
            self.issued_at,
            &self.reason,
            signers,
        )?;
        for (signer_id, sig) in signers.iter().zip(signatures.iter()) {
            let signer_cert = signer_certs
                .iter()
                .find(|c| c.device_id == *signer_id)
                .ok_or(OwnerError::NotEnrolled {
                    owner: self.owner_id,
                    device: *signer_id,
                })?;
            if signer_cert.owner_id != self.owner_id {
                return Err(OwnerError::WrongOwner {
                    expected: self.owner_id,
                    got: signer_cert.owner_id,
                });
            }
            if !matches!(
                signer_cert.issuer,
                crate::certs::EnrollmentIssuer::Master { .. }
            ) {
                return Err(OwnerError::InvalidSignature {
                    cert_type: "Revocation-Quorum-Signer-Not-Master",
                });
            }
            signer_cert.verify(now_secs)?;
            if signer_cert.issued_at > self.issued_at {
                return Err(OwnerError::InvalidSignature {
                    cert_type: "Revocation-Quorum-Backdated-Signer",
                });
            }
            let vk = VerifyingKey::from_bytes(&signer_cert.device_pubkeys.classical.ed25519_verify)
                .map_err(|_| OwnerError::InvalidSignature {
                    cert_type: "Revocation-Quorum-Member",
                })?;
            verify_with_tag(
                &vk,
                tags::REVOCATION,
                &payload_bytes,
                sig,
                "Revocation-Quorum-Member",
            )?;
        }
        Ok(())
    }
```

Also change `verify()`'s Quorum arm (`revocation.rs:169`) to the renamed variant:

```rust
            (RevocationIssuer::Quorum { .. }, _) => Err(OwnerError::QuorumRequiresSignerCerts),
```

`RevocationSigningPayload` stays private (same file). The struct's `reason: &'a RevocationReason` field already fits the payload fn's `&RevocationReason` parameter.

- [ ] **Step 5: Run tests**

Run: `cargo nextest run --locked -p harmony-owner 2>&1 | tail -5`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add -A crates/harmony-owner && git commit -m "ZEB-677 S1: RevocationCert quorum sign/assemble/verify; QuorumRequiresSignerCerts

Standalone verify() still fails closed on Quorum certs; full verification
requires the signer enrollment certs (depth-1 chain carriage)."
```

---

### Task 4: `OwnerState::add_revocation` accepts quorum (fleet-internal policy checks)

**Files:**
- Modify: `crates/harmony-owner/src/state.rs:365-398` (+ its tests around `:660`)
- Modify: `crates/harmony-owner/tests/e2e_threats.rs:116,128`, `crates/harmony-owner/tests/e2e_lifecycle.rs:167` (signature change at call sites)

**Interfaces:**
- Consumes: Task 3's `RevocationCert::quorum_signing_payload_bytes` / part verification pieces.
- Produces (client S2/S3 call this through trust-sync merge and `revoke_device_inner`):
  - `OwnerState::add_revocation(&mut self, cert: RevocationCert, now: u64, active_window_secs: u64) -> Result<(), OwnerError>` — **signature change**; `now`/`active_window_secs` are used only by the Quorum arm (SelfDevice/Master behavior unchanged).

- [ ] **Step 1: Update the signature and all existing call sites mechanically** (compiler-driven): `state.rs` tests, `e2e_threats.rs`, `e2e_lifecycle.rs` — pass `(cert, 0, u64::MAX)` where the old behavior must be preserved exactly (SelfDevice/Master arms ignore the params; `0`/`u64::MAX` makes that explicit at test sites that don't exercise quorum).

- [ ] **Step 2: Write the failing tests** (in `state.rs` `mod tests`; reuse the existing test helpers `keypair_and_bundle` / liveness-cert construction visible around `state.rs:449-575`, and mirror the arrangement of the existing `quorum_with_inactive_signer_rejected` test at `:704` for the world setup — master + A + B enrolled, liveness minted for A and B at `now`)

```rust
    #[test]
    fn quorum_revocation_accepted_with_valid_parts() {
        // World: master-enrolled A and B, both active (liveness at t=1000).
        // A+B quorum-revoke device C (also enrolled). Mirrors the setup in
        // quorum_with_inactive_signer_rejected but drives add_revocation.
        let (mut state, master_sk, master_bundle, a, b) = quorum_state_world();
        let (c_sk, c_bundle) = keypair_and_bundle();
        let _ = c_sk;
        let c_cert = EnrollmentCert::sign_master(
            &master_sk, master_bundle, c_bundle.identity_hash(), c_bundle.clone(), 500, None,
        )
        .unwrap();
        let c_id = c_cert.device_id;
        state.add_enrollment(c_cert, 1_000, 90 * 24 * 3600).unwrap();

        let signers = [a.id, b.id];
        let payload = crate::certs::RevocationCert::quorum_signing_payload_bytes(
            state.owner_id, c_id, 1_000, &crate::certs::RevocationReason::Lost, &signers,
        )
        .unwrap();
        let cert = crate::certs::RevocationCert::assemble_quorum(
            state.owner_id, c_id, 1_000, crate::certs::RevocationReason::Lost,
            vec![
                (a.id, crate::certs::RevocationCert::sign_quorum_part(&a.sk, &payload)),
                (b.id, crate::certs::RevocationCert::sign_quorum_part(&b.sk, &payload)),
            ],
        )
        .unwrap();
        state.add_revocation(cert, 1_000, 90 * 24 * 3600).unwrap();
        assert!(state.is_revoked(c_id));
    }

    #[test]
    fn quorum_revocation_rejects_inactive_revoked_unenrolled_and_backdated_signers() {
        let (mut state, _master_sk, _master_bundle, a, b) = quorum_state_world();
        let target = [9u8; 16];
        let mk = |state: &OwnerState, a_id, b_id, a_sk: &SigningKey, b_sk: &SigningKey| {
            let signers = [a_id, b_id];
            let payload = crate::certs::RevocationCert::quorum_signing_payload_bytes(
                state.owner_id, target, 1_000, &crate::certs::RevocationReason::Lost, &signers,
            )
            .unwrap();
            crate::certs::RevocationCert::assemble_quorum(
                state.owner_id, target, 1_000, crate::certs::RevocationReason::Lost,
                vec![
                    (a_id, crate::certs::RevocationCert::sign_quorum_part(a_sk, &payload)),
                    (b_id, crate::certs::RevocationCert::sign_quorum_part(b_sk, &payload)),
                ],
            )
            .unwrap()
        };
        // Inactive signer: verify far in the future so liveness lapses.
        let cert = mk(&state, a.id, b.id, &a.sk, &b.sk);
        assert!(state
            .add_revocation(cert, 1_000 + 91 * 24 * 3600, 90 * 24 * 3600)
            .is_err());
        // Unenrolled signer: unknown id in the signer set.
        let ghost = [0xEEu8; 16];
        let signers = [a.id, ghost];
        let payload = crate::certs::RevocationCert::quorum_signing_payload_bytes(
            state.owner_id, target, 1_000, &crate::certs::RevocationReason::Lost, &signers,
        )
        .unwrap();
        let cert = crate::certs::RevocationCert::assemble_quorum(
            state.owner_id, target, 1_000, crate::certs::RevocationReason::Lost,
            vec![
                (a.id, crate::certs::RevocationCert::sign_quorum_part(&a.sk, &payload)),
                (ghost, vec![0u8; 64]),
            ],
        )
        .unwrap();
        assert!(matches!(
            state.add_revocation(cert, 1_000, 90 * 24 * 3600),
            Err(OwnerError::NotEnrolled { .. })
        ));
        // Revoked signer: revoke B (via master path), then B co-signs.
        let cert_revoke_b = mk(&state, a.id, b.id, &a.sk, &b.sk); // placeholder quorum against B unused below
        let _ = cert_revoke_b;
        state
            .add_revocation(
                crate::certs::RevocationCert::sign_self(
                    &b.sk, state.owner_id, b.id, 1_001, crate::certs::RevocationReason::Lost,
                )
                .unwrap(),
                1_001,
                90 * 24 * 3600,
            )
            .unwrap();
        let cert = mk(&state, a.id, b.id, &a.sk, &b.sk);
        assert!(matches!(
            state.add_revocation(cert, 1_001, 90 * 24 * 3600),
            Err(OwnerError::Revoked { .. })
        ));
    }
```

(`quorum_state_world()` is a small test helper to add alongside the existing helpers: mint master, enroll A and B via `EnrollmentCert::sign_master` at `issued_at=100`, `add_liveness` for both at `timestamp=1_000`, return `(state, master_sk, master_bundle, DeviceFixture{id, sk} for A and B)`. Model it on the existing quorum enrollment tests around `state.rs:576-750` — reuse their construction idioms exactly; if an equivalent helper already exists there, use it instead of adding a new one.)

- [ ] **Step 3: Run to verify failure**

Run: `cargo test -p harmony-owner --lib state -- quorum_revocation 2>&1 | tail -5`
Expected: compile FAIL (signature) then assertion FAIL (`QuorumRequiresSignerCerts` path still rejects).

- [ ] **Step 4: Implement the Quorum arm.** Replace `state.rs:389-391` with:

```rust
            crate::certs::RevocationIssuer::Quorum {
                signers,
                signatures,
            } => {
                if signers.len() < 2 {
                    return Err(OwnerError::InsufficientQuorum {
                        min: 2,
                        got: signers.len(),
                    });
                }
                if signers.len() != signatures.len() {
                    return Err(OwnerError::InvalidSignature {
                        cert_type: "Revocation-Quorum-Length-Mismatch",
                    });
                }
                let unique: std::collections::HashSet<[u8; 16]> =
                    signers.iter().copied().collect();
                if unique.len() != signers.len() {
                    return Err(OwnerError::InvalidSignature {
                        cert_type: "Revocation-Quorum-Duplicate-Signer",
                    });
                }
                for signer_id in signers {
                    if self.is_revoked(*signer_id) {
                        return Err(OwnerError::Revoked { device: *signer_id });
                    }
                }
                // Same active-signer policy as add_enrollment's quorum path:
                // every signer must have fresh liveness. Fleet-internal only —
                // peer verification (verify_quorum_with_signers) cannot and
                // does not check liveness.
                let active = self.active_devices(now, active_window_secs);
                let active_set: std::collections::HashSet<[u8; 16]> =
                    active.into_iter().collect();
                for signer_id in signers {
                    if !active_set.contains(signer_id) {
                        return Err(OwnerError::InvalidSignature {
                            cert_type: "Revocation-Quorum-Inactive-Signer",
                        });
                    }
                }
                let payload_bytes = crate::certs::RevocationCert::quorum_signing_payload_bytes(
                    cert.owner_id,
                    cert.target,
                    cert.issued_at,
                    &cert.reason,
                    signers,
                )?;
                for (signer_id, sig) in signers.iter().zip(signatures.iter()) {
                    let signer_enrollment =
                        self.enrollments
                            .get(signer_id)
                            .ok_or(OwnerError::NotEnrolled {
                                owner: self.owner_id,
                                device: *signer_id,
                            })?;
                    if signer_enrollment.issued_at > cert.issued_at {
                        return Err(OwnerError::InvalidSignature {
                            cert_type: "Revocation-Quorum-Backdated-Signer",
                        });
                    }
                    let vk = VerifyingKey::from_bytes(
                        &signer_enrollment.device_pubkeys.classical.ed25519_verify,
                    )
                    .map_err(|_| OwnerError::InvalidSignature {
                        cert_type: "Revocation-Quorum-Member",
                    })?;
                    crate::signing::verify_with_tag(
                        &vk,
                        crate::signing::tags::REVOCATION,
                        &payload_bytes,
                        sig,
                        "Revocation-Quorum-Member",
                    )?;
                }
            }
```

and change the function signature to `pub fn add_revocation(&mut self, cert: crate::certs::RevocationCert, now: u64, active_window_secs: u64) -> Result<(), OwnerError>`. Note the borrow order: destructure `signers`/`signatures` from `&cert.issuer` before `self.revocations.insert(cert)` at the end (the existing insert stays the final statement).

Rewrite the old `quorum_revocation_rejected_until_implemented` test (`state.rs:660` area) as `quorum_revocation_with_garbage_signatures_rejected`: same construction it has today (hand-built `RevocationIssuer::Quorum` with junk signatures) but now expecting `InvalidSignature`/`NotEnrolled` from the real verification path instead of the retired variant.

- [ ] **Step 5: Run the full crate suite**

Run: `cargo nextest run --locked -p harmony-owner 2>&1 | tail -5`
Expected: PASS, including the two e2e integration test files with updated call sites.

- [ ] **Step 6: Commit**

```bash
git add -A crates/harmony-owner && git commit -m "ZEB-677 S1: OwnerState::add_revocation accepts quorum certs

Signature gains (now, active_window_secs) for the quorum arm's
active-signer policy — same checks as add_enrollment's quorum path.
SelfDevice/Master behavior unchanged."
```

---

### Task 5: Workspace gates + PR

**Files:**
- No new code; whole-workspace validation + `docs/plans/2026-07-12-zeb-677-s1-quorum-certs-plan.md` (this file, committed for the record).

- [ ] **Step 1: Full gates (CI parity)**

```bash
cargo fmt --all -- --check
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo nextest run --locked --workspace
grep -rn "QuorumRevocationNotImplemented" crates/ && echo "STALE REFERENCE" || echo clean
```

Expected: all clean; grep prints `clean`.

- [ ] **Step 2: Push branch + open PR against zeblithic/harmony main**

```bash
git push -u origin zeb-677-s1-quorum-certs
gh pr create --repo zeblithic/harmony --title "ZEB-677 S1: quorum cert primitives — distributed sign/assemble/verify + OwnerState quorum revocation" --body-file <prepared body>
```

PR body covers: spec link (`harmony-client` `docs/specs/2026-07-12-zeb-677-quorum-wiring-design.md` §2/§9), the depth-1 chain-carriage rationale, the `add_revocation` signature change and why (quorum active-signer policy needs time), the error-variant rename, and the gate evidence. Then fire the CodeRabbit review (once, at open), post the PR link + claim note on ZEB-677, and converge per the standing loop rules.
