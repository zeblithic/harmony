// SPDX-License-Identifier: Apache-2.0 OR MIT
//! `harmony-crdt-sync` — reusable substrate for verified-CRDT-over-pubsub
//! synchronization.
//!
//! Core Harmony has no generic engine for the "replicate a verified,
//! CRDT-merged dataset over best-effort pubsub, catching up on the history
//! you missed" shape that apps built on the platform need. This crate is
//! that substrate.
//!
//! It currently exposes the **backfill/backoff latches** ([`backfill_latch`])
//! — the pure, runtime-free decision cores for paginated catch-up over an
//! unreliable request/reply transport, extracted from harmony-client
//! (ZEB-571 item 3). It is designed to grow to host the verified-event-log
//! and snapshot sync engines (`VerifiedLog` / `FleetSyncEngine`; ZEB-571
//! item 6), which the latches compose with.
//!
//! Everything here is sans-I/O: the types decide *what* to do (request,
//! wait, idle) and leave *doing* it (transport, timers, persistence) to a
//! caller-supplied driver.

// no_std when the `std` feature is off — except under `test`, where the
// test harness always links std (so the unit tests may use `Vec`/`vec!`
// even in a `--no-default-features` test build). The library's own no_std
// cleanliness is still enforced by a non-test `--no-default-features` build.
#![cfg_attr(all(not(feature = "std"), not(test)), no_std)]

pub mod backfill_latch;

pub use backfill_latch::{
    BackfillAction, BackfillLatch, PageOutcome, RootFetchAction, RootFetchLatch,
    BACKFILL_RETRY_BASE_MS, BACKFILL_RETRY_CAP_MS,
};
