// SPDX-License-Identifier: Apache-2.0 OR MIT
//! `harmony-crdt-sync` — reusable substrate for verified-CRDT-over-pubsub
//! synchronization.
//!
//! Core Harmony has no generic engine for the "replicate a verified,
//! CRDT-merged dataset over best-effort pubsub, catching up on the history
//! you missed" shape that apps built on the platform need. This crate is
//! that substrate.
//!
//! It exposes:
//!
//! - the **backfill/backoff latches** ([`backfill_latch`]) — the pure,
//!   runtime-free decision cores for paginated catch-up over an unreliable
//!   request/reply transport (ZEB-571 item 3), over the shared escalating
//!   [`retry_backoff`] schedule that any retrying driver can compose
//!   (ZEB-761);
//! - the **verified-event-log engine** ([`verified_log`]) — a log that
//!   dedups, verifies each event against its materialized prior state, and
//!   re-materializes on demand (ZEB-571 item 6a);
//! - the **snapshot-replication kernels** — [`replay_admission`] (per-source
//!   replay protection with apply-before-advance enforced by the types),
//!   [`hlc`] (the hybrid-logical-clock tick rule), and [`debounce_latch`]
//!   (sliding-window publish scheduling with claim/settle) — ZEB-571 item 6b.
//!
//! Everything here is sans-I/O: the types decide *what* to do (request,
//! wait, idle, admit, publish) and leave *doing* it (transport, timers,
//! persistence, crypto) to a caller-supplied driver.
//!
//! # Scope boundary
//!
//! Harmony-client's `FleetSyncEngine` — the snapshot-replication *engine*
//! these kernels were extracted from — deliberately did **not** move here,
//! and no engine of that shape should. Its job is sequencing content-store
//! fetches, key-epoch crypto, transport channels, and an owned async task;
//! that is orchestration against concrete I/O, not a decision core, and
//! hosting it would require dragging a CAS trait, a wire clock type, and a
//! key set into this crate along with a runtime dependency (ZEB-759). What
//! belongs here is what an engine *decides*; how it talks to the world stays
//! with the engine.
//!
//! The same boundary governs *state*, not just I/O: a kernel here owns only
//! state its callers never write behind its back. A dirty flag poked by
//! every mutation site in an application is shared, concurrently-written
//! caller state, so [`debounce_latch`] holds the publish window and takes
//! the flag's value as an argument rather than owning it — a kernel that
//! kept its own copy would force callers to mirror theirs into it, and two
//! copies of one signal drift (ZEB-759).

// no_std when the `std` feature is off — except under `test`, where the
// test harness always links std (so the unit tests may use `Vec`/`vec!`
// even in a `--no-default-features` test build). The library's own no_std
// cleanliness is still enforced by a non-test `--no-default-features` build.
#![cfg_attr(all(not(feature = "std"), not(test)), no_std)]

extern crate alloc;

pub mod backfill_latch;
pub mod debounce_latch;
pub mod hlc;
pub mod replay_admission;
pub mod retry_backoff;
pub mod verified_log;

pub use backfill_latch::{
    BackfillAction, BackfillLatch, PageOutcome, RootFetchAction, RootFetchLatch,
    BACKFILL_RETRY_BASE_MS, BACKFILL_RETRY_CAP_MS,
};
pub use debounce_latch::{DebounceLatch, DirtySignal, PublishClaim, PublishOutcome};
pub use hlc::HlcTick;
pub use replay_admission::{Admission, CommitTicket, MonotoneMap, ReplayTracker};
pub use retry_backoff::{RetryBackoff, RETRY_BASE_MS, RETRY_CAP_MS};
pub use verified_log::{InsertOutcome, LogPolicy, VerifiedLog};
