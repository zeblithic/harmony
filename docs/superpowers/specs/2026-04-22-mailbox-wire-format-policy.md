# Mailbox Wire-Format Evolution Policy

**Status:** Adopted
**Scope:** `harmony-mailbox` crate — `MailRoot`, `MailFolder`, `MailPage`,
`MessageEntry`, and `HarmonyMessage` wire formats.
**Resolves:** ZEB-99, ZEB-100, ZEB-101.

## Why a policy exists

Mailbox and message blobs are written to the content-addressed store and
referenced by CID. Every byte of the wire format is hashed into that CID, so
format decisions directly control which logical data points to which physical
blob. Two independent consumers parse these bytes — the `harmony-mail` gateway
and the `harmony-client` desktop app — so we cannot diverge them quietly.
Email is also inherently long-lived data; a mailbox written today may be
read years from now. This policy pins down how the format is allowed to
evolve, so future changes don't silently break existing data or split the
decoder population.

## Versioning rules

Each top-level blob begins with magic bytes followed by a one-byte
`MAILBOX_VERSION` (for mailbox types) or `VERSION` (for `HarmonyMessage`).
The current value is `0x01` for all formats in this crate.

1. **Version bumps are breaking.** A bump means "old readers MUST reject
   the new blob, and new readers MUST reject old blobs unless an explicit
   migration path is provided." There is no partial compatibility.
2. **Readers reject unknown versions.** Any version byte other than the
   one the decoder was built for returns `MailboxError::UnsupportedVersion`.
   Silent best-effort decoding is not permitted — it produces data-loss
   bugs at higher layers.
3. **Writers pin to the current version.** `to_bytes` validates
   `self.version == VERSION` and errors on mismatch. This prevents a
   caller from accidentally emitting legacy or forward-version bytes.
4. **Byte layout is frozen within a major version.** We do not add
   "optional fields at the end" inside v1. If a field needs to be added,
   removed, or reshaped, bump the version.

## Forward compatibility: none, by design

Some binary formats (Protocol Buffers, TLV) are engineered so old readers
can skip unknown fields. We explicitly do not want that here:

- Unknown fields change the CID anyway, so the main benefit (bytes
  round-tripping through old decoders) does not apply to a CAS format.
- Fixed-layout decoders are faster and simpler to audit.
- One set of bytes per version gives a crisp mental model: the decoder
  either fully understands the blob or fails loudly.

When in doubt, prefer a new version over a best-effort extension.

## CID stability

CIDs are a hash of the wire bytes, so:

- Two logically equivalent messages under different versions hash to
  different CIDs. This is by design — the CID commits to the exact bytes,
  not to any higher-level semantic equivalence.
- A version bump is therefore a data-migration event: blobs written at
  version N become addressable only via their v-N CID, and readers upgraded
  to v-(N+1) will refuse them unless a migration path rewrites them.
- Pre-alpha: when we bump versions, existing CIDs in test data become
  invalid. That is acceptable today because no real user data exists.
- Post-launch: a version bump requires a coordinated rewrite of all
  persisted blobs, or a multi-version decoder with an explicit migration
  path kept alive for the transition window.

## The stability boundary

Three external constants are load-bearing on the v1 wire format:

- `CID_LEN = 32` (content-addressed hash width)
- `MESSAGE_ID_LEN = 16` (internal message identifier width)
- `ADDRESS_HASH_LEN = 16` (Harmony address hash width, re-exported from
  `harmony-identity`)

If any of these change, every blob serialized under v1 becomes
unparseable. To catch this at the earliest possible moment, `mailbox.rs`
enforces all three with `const _: () = assert!(...)` guards that fail
compilation. These guards are the stability boundary between this crate
and its upstream dependencies. Removing them without a matching version
bump is forbidden.

## Recipe: evolving the format

To add, remove, or reshape a field on an existing struct:

1. Bump `MAILBOX_VERSION` (or the per-struct `VERSION`).
2. Update `to_bytes` to emit the new layout only at the new version.
3. Update `from_bytes` to reject old bytes, unless a migration is provided.
4. If a migration is needed, add a `from_bytes_v1` → v2 converter path
   with explicit tests covering both versions.
5. Update both `harmony-mail` (gateway) and `harmony-client` (desktop) in
   the same release cycle — any skew breaks the mesh.
6. Add a changelog entry here (below) describing the change and its
   motivation.

To change the *meaning* of a field without changing the bytes:

- If decoded semantics differ for any consumer, it still requires a
  version bump. A silent re-interpretation of existing bytes is the
  worst failure mode this policy exists to prevent.
- If the change is purely documentation (naming, comments), no bump.

## Canonical vs delivery serialization (ZEB-100 resolution)

`HarmonyMessage::to_bytes` strips BCC recipients from the wire format,
and `HarmonyMessage::from_bytes` rejects any blob that claims a BCC
recipient on decode. This is deliberate and adopted as policy:

- **The wire format is the *delivery* form.** There is no canonical
  wire form that preserves BCC — no "sender's sent folder" blob type
  retains hidden recipient lists.
- **BCC information lives out of band.** Sender-side resend lists,
  auditing metadata, or client-side "this is who I sent it to" views
  must be kept in non-CAS state (e.g., local databases) and never hit
  the content-addressed store.
- **The decoder is hardened.** Rejecting BCC on decode prevents a
  forged, replayed, or corrupted blob from reintroducing hidden
  recipients into client-visible state. This is a security invariant,
  not a performance choice.

If future requirements demand canonical-with-BCC storage (e.g.,
server-side audit logs), a *separate* format with its own magic bytes
must be introduced. We will not extend `HarmonyMessage` to smuggle BCC
through the existing wire format.

## Dual page-linkage resolution (ZEB-101 resolution)

Prior to this policy, `MailPage` carried a `next_page: Option<CID>` field
redundant with `MailFolder.page_cids`. Two sources of truth for the same
navigation invites drift: a page's linked-list pointer could disagree
with the folder's index, and readers would have no principled way to
resolve the conflict.

Policy decision: `MailFolder.page_cids` is canonical. `MailPage.next_page`
is removed from the wire format. Readers walk pages via the folder's
index; individual pages are self-contained data blobs with no navigation
state. Because this repo is pre-alpha and the only consumers (this repo
and `harmony-client`) are updated in lockstep, the removal ships as a
direct change to the v1 wire format rather than a version bump. The
version byte is preserved for future evolution; this one-time rework
consumes the "we can still break v1" budget.

## Invariants to preserve across all future versions

- Magic bytes (`ROOT_MAGIC` / `FOLDER_MAGIC` / `PAGE_MAGIC`) identify the
  format *across* versions and must not change. The version byte
  disambiguates layout; the magic disambiguates format.
- The version byte always occupies offset 4 (immediately after the
  4-byte magic) for every top-level blob.
- The `const_assert` stability-boundary guards on `CID_LEN`,
  `MESSAGE_ID_LEN`, and `ADDRESS_HASH_LEN` are load-bearing and must not
  be relaxed without a matching version bump and policy revision.

## Changelog

- **2026-04-22 — v1 (this document):** Policy adopted. ZEB-100 settled in
  favor of delivery-only serialization. ZEB-101 settled by removing
  `MailPage.next_page` from the v1 wire format (pre-alpha break).
