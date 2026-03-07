# Trust Network Design

Local-first trust scoring for the Harmony client. Each user expresses
trust in peers as a single byte encoding four dimensions. Published to
the network (future work); this bead covers the data model, mock service,
and client UI.

---

## Data Model

### Trust Score (8 bits)

A single byte encodes four dimensions, each with four levels (2 bits,
`00` low to `11` high). Ordered most-significant to least-significant:

| Bits | Dimension | Question | Nature |
|------|-----------|----------|--------|
| 0-1 | **Identity** | Do you believe this person is who they say they are? | Verifiable (key-signing) |
| 2-3 | **Compliance** | Do you believe this person follows the rules? | Semi-verifiable (cryptographic evidence) |
| 4-5 | **Association** | Would you vouch for this person? | Subjective |
| 6-7 | **Endorsement** | Would you stake your reputation on this person? | Social capital exchange |

The 8-bit value can be treated as a 0-255 scalar for most aggregation
and ranking purposes (sorting, thresholds, color mapping), even though
the individual dimensions carry specific meaning.

### Types

```typescript
type TrustScore = number; // 0-255, uint8

interface TrustEdge {
  source: string;   // SHA-256 address of the scorer
  target: string;   // SHA-256 address of the scored peer
  score: TrustScore;
  timestamp: number; // when this score was last set
}
```

### Dimension Helpers

```typescript
function getIdentity(score: TrustScore): number    // bits 0-1, returns 0-3
function getCompliance(score: TrustScore): number   // bits 2-3, returns 0-3
function getAssociation(score: TrustScore): number  // bits 4-5, returns 0-3
function getEndorsement(score: TrustScore): number  // bits 6-7, returns 0-3

function buildScore(
  identity: number,    // 0-3
  compliance: number,  // 0-3
  association: number, // 0-3
  endorsement: number, // 0-3
): TrustScore
```

### Trust is directional and global

- Trust is directional: Alice's score for Bob != Bob's score for Alice.
- Trust is global: one score per directed pair, not per-community.
- Trust changes over time: re-score at any point, timestamp records when.
- Trust is public: scores are published to the network (future work).

---

## Trust Graph

```typescript
interface TrustGraph {
  edges: TrustEdge[];
  edgesFrom(address: string): TrustEdge[];
  edgesTo(address: string): TrustEdge[];
  directScore(source: string, target: string): TrustScore | null;
}
```

### Transitive trust (future work)

EigenTrust-style propagation over the full graph to compute derived
reputation scores. Not in this bead -- the network is small and mock
data doesn't benefit from multi-hop derivation. The data model supports
it: the graph of signed edges is everything EigenTrust needs as input.

---

## Mock Service

```typescript
class MockTrustGraphService {
  edges: TrustEdge[];
  localAddress: string;

  setScore(target: string, score: TrustScore): void;
  clearScore(target: string): void;
  getEdges(): TrustEdge[];
  edgesFrom(address: string): TrustEdge[];
  edgesTo(address: string): TrustEdge[];
  directScore(source: string, target: string): TrustScore | null;
  resolveMediaTrust(peerAddress: string): TrustLevel;
}
```

On construction, populates the graph with randomized trust edges between
mock peers (reusing NATO-named nodes from the network data service). The
local user has a few pre-set scores; other peers have randomized scores
for each other.

### Media trust derivation

`resolveMediaTrust()` maps the identity dimension (bits 0-1) to the
existing `TrustLevel` type:

| Identity bits | TrustLevel |
|---------------|------------|
| `00` or `01` | `untrusted` |
| `10` | `preview` |
| `11` | `trusted` |

### Integration with existing TrustService

The existing `TrustService` keeps its override API (`setPeerTrust`,
`setCommunityTrust`, `setGlobalTrust`). Resolution chain becomes:

```
1. Per-peer override          -> use it
2. Per-community override     -> use it
3. Trust graph derivation     -> resolveMediaTrust(peerAddress)
4. Global default             -> untrusted
```

Step 3 is new. User overrides still win.

---

## Client UI

### 1. Trust Score Editor (`TrustEditor.svelte`)

Displayed in peer detail/profile views. Four dimensions, each a row of
4 radio-style buttons (0-3):

```
Trust
  Identity      [  0  |  1  | *2* |  3  ]
  Compliance    [ *0* |  1  |  2  |  3  ]
  Association   [  0  | *1* |  2  |  3  ]
  Endorsement   [  0  | *1* |  2  |  3  ]

  Overall: 0x69 (105/255)          [Clear]
```

- Each dimension is a `role="radiogroup"` with `aria-label`.
- Overall score shown as hex and fraction.
- "Clear" removes the local user's score for this peer.
- Keyboard navigable (arrow keys within group, tab between groups).

### 2. Trust Badge (`TrustBadge.svelte`)

Small colored dot for inline trust indication wherever a peer name
appears:

| Score range | Color | Meaning |
|-------------|-------|---------|
| unscored | `#72767d` (gray) | No opinion |
| 0-63 | `#ed4245` (red) | Low trust |
| 64-127 | `#faa61a` (amber) | Cautious |
| 128-191 | `#43b581` (green) | Trusted |
| 192-255 | `#5865f2` (accent blue) | Highly trusted |

Includes `aria-label` describing the trust level.

### 3. Trust Overview (`TrustOverview.svelte`)

Sortable table of all peers with trust scores. Reuses the `DataTable`
pattern (sortable column headers with `aria-sort`, keyed `{#each}`,
`role="grid"`).

Columns: Name, My Score, Their Score, Identity, Compliance, Association,
Endorsement.

Sortable by any column. Shows both "my score for them" and "their score
for me" when available.

---

## File Plan

### New files

| File | Purpose |
|---|---|
| `src/lib/trust-score.ts` | TrustScore type, dimension helpers, TrustEdge interface |
| `src/lib/trust-score.test.ts` | Bit manipulation tests, buildScore round-trips, edge cases |
| `src/lib/trust-graph-service.ts` | MockTrustGraphService with mock data generation |
| `src/lib/trust-graph-service.test.ts` | setScore, getEdges, resolveMediaTrust threshold tests |
| `src/lib/components/TrustEditor.svelte` | 4-dimension radiogroup editor |
| `src/lib/components/__tests__/TrustEditor.test.ts` | Renders dimensions, emits changes, clear works |
| `src/lib/components/TrustBadge.svelte` | Colored dot for inline trust indication |
| `src/lib/components/__tests__/TrustBadge.test.ts` | Color thresholds, unscored state, aria-label |
| `src/lib/components/TrustOverview.svelte` | Sortable peer trust table |
| `src/lib/components/__tests__/TrustOverview.test.ts` | Renders rows, sorts by dimension |

### Modified files

| File | Change |
|---|---|
| `src/lib/trust-service.ts` | Add fallback to resolveMediaTrust when no override exists |
| `src/lib/trust-service.test.ts` | Test the new fallback path |
| `src/lib/types.ts` | Re-export TrustScore, TrustEdge types |

---

## Out of Scope (YAGNI)

- Network publishing / signed blobs (future bead).
- EigenTrust computation (future bead -- data model supports it).
- Trust graph visualization (reuse network viz infrastructure later).
- Per-community trust scores (global only for now).
- Trust history / changelog.
- Anonymous scoring / ZK proofs.

---

## Accessibility

All trust UI follows Harmony's accessibility requirements:

- `TrustEditor` dimensions use `role="radiogroup"` with labeled options.
- `TrustBadge` includes `aria-label` describing the trust level.
- `TrustOverview` uses `role="grid"` with `aria-sort` on sortable headers.
- All interactive elements are keyboard-operable (Enter, Space, arrow keys).
- Color is never the sole indicator -- text labels accompany all color coding.
