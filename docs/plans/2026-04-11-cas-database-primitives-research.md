# Content-Addressed Database Primitives for Distributed Systems

An Architectural Evaluation for the Harmony Mesh

Research conducted April 2026. Evaluates Prolly Trees, Merkle Search Trees,
CRDT-backed Merkle DAGs, and IPLD ADLs for building a distributed, syncable,
and verifiable database layer over Harmony's BLAKE3 CAS + Zenoh mesh.

## Key Findings

- **Prolly Trees are the optimal long-term primitive** for harmony-db
  - CDF-based content-defined chunking over key-only hashes
  - History independence, tight chunk-size distribution, O(d) diffing
  - No hash-mining DoS vulnerability (unlike MSTs)
  - Better range scan performance than MSTs

- **Phase 1 (v0.1): Hybrid architecture** — CAS truth + local index cache (sled/redb)
- **Phase 2 (v1.0): Native Prolly Tree engine** in harmony-db crate
- **Zenoh anti-entropy couples natively** with Merkle tree diffing
- **Beelay Sediment Trees** solve E2EE sync without plaintext access

## Introduction: Decentralized State and the Harmony Architecture

The transition from monolithic, host-addressed database architectures to distributed,
content-addressed storage (CAS) paradigms represents a fundamental shift in how networks
negotiate truth, state, and identity. Traditional database systems rely on a trusted
authority and location-based addressing (e.g., an IP address and a connection string) to
maintain state. In contrast, modern distributed systems -- such as the Harmony ecosystem,
the AT Protocol, and decentralized peer-to-peer networks -- utilize cryptographic hashing
to create self-certifying data structures where the content itself dictates its physical
and logical address. This paradigm is characterized by structural sharing, mathematical
immutability, and inherently verifiable data histories.

Within the context of the Harmony system, which leverages BLAKE3 Content Identifiers (CIDs)
over a Zenoh publish-subscribe and query mesh network, establishing a robust database-like
query layer over CAS is a complex architectural requirement. The motivating use case -- a
distributed email architecture conceptualized as a "Merkle mailbox" -- illustrates this
necessity perfectly.

To immediately address the foundational architectural questions regarding this system: the
utilization of pure CAS over embedded relational databases like SQLite is the correct
architectural vector. SQLite, while highly optimized for local block storage, fundamentally
lacks the native deduplication, history independence, and cryptographic verifiability required
for a mesh-native application. Furthermore, for the initial v0.1 implementation, wire formats
should prioritize plain text and markdown bytes directly within the payload, as rich-text
abstraction layers introduce unnecessary serialization complexity before the core CAS
synchronization logic is proven.

Similarly, the address book functionality should be entirely deferred to the Zenoh network's
native announce system, querying the mesh for known cryptographic registrations rather than
maintaining a duplicated local ledger. Finally, the development methodology must strictly
prioritize a backend-first approach. Given the intricate nature of CAS design, constructing
the Tauri commands for Zenoh subscriptions and CAS storage persistence must precede any
frontend work; mocking user interface data over an unproven Merkle DAG synchronization layer
will invariably result in discarded code.

## The Merkle Mailbox: Concrete Design and State Transitions

To ground the theoretical analysis of CAS database primitives, it is essential to define the
concrete data structures and state transitions of the proposed Merkle mailbox. In this
architecture, the mailbox operates similarly to Git: a root CID points to folder nodes, which
point to message pages, which ultimately point to individual message blobs and their associated
attachments. The root CID serves as the solitary mutable pointer representing the user's
current mail state.

The precise schema of this CAS structure begins with a mail_root node containing metadata
(version, owner address hash, and a timestamp) and a map of standard folders such as "inbox",
"sent", "drafts", and "trash". Each folder is represented by its own CID. An "inbox" node
contains aggregate metadata, such as the total message count and the number of unread messages,
alongside an array of page CIDs. These pages act as pagination boundaries, each holding an
array of individual message CIDs and a pointer to the subsequent page.

The lifecycle of an incoming email demonstrates the inherent immutability and cascading hash
mechanics of the system. When a new message arrives, the client first deserializes the current
inbox page from the local or network CAS. The new message CID is appended to this page, or,
if the page has reached its maximum defined capacity, a new page is initialized. Because the
content of the page has been mathematically altered, hashing the new page yields a completely
new CID. Consequently, the inbox folder node must be updated to reference this new page CID,
which generates a new inbox CID. Finally, the mail_root is updated to reference the new inbox
CID, generating a new root hash. This new root CID is then published to the network via a
well-known Zenoh key expression, such as `harmony/mail/v1/{address_hash}/root`.

This pure CAS approach guarantees O(1) fetch times by CID, immediate verification of
mathematical integrity through hash chain validation, and the preservation of an immutable,
auditable history where old roots remain cryptographically valid. However, it completely lacks
native indexing.

## Foundational Mechanics of Merkle DAGs in Database Architecture

A Merkle DAG is a directed acyclic graph where every node is identified by the cryptographic
hash of its opaque payload and the CIDs of its distinct children. Because cryptographic hash
functions, such as BLAKE3 or SHA-256, are practically collision-resistant one-way functions,
cycles cannot mathematically exist within the structure.

The primary advantage of Merkle DAGs in database applications is structural sharing, also
known as data deduplication. If two different database versions, or even two different users
on the network, share identical rows or chunks of data, those data chunks evaluate to the
exact same CID. A storage engine backing the CAS will therefore persist the physical bytes
only once.

However, flat Merkle DAGs and standard Merkle trees mapped to traditional database structures
(like standard B-trees) suffer from severe limitations when representing dynamic, ordered, and
frequently mutating datasets. A standard B-tree mapped directly to a Merkle structure is
highly sensitive to the order of operations. This means nodes inserting the same records in
different orders get different root hashes.

To serve as a distributed, content-addressed database, a tree structure must possess **history
independence** (also called "unicity"): a specific set of keys and values will definitively
yield one strictly unique tree topology, regardless of insertion order.

## Prolly Trees: Probabilistic B-Trees and Content-Defined Chunking

Prolly Trees emerged specifically to solve the structural volatility of standard Merkle trees
in database applications. Originally conceptualized in the Noms project, and later heavily
refined in Dolt, Prolly trees combine B-tree performance with strict history independence.

### Content-Defined Chunking Algorithm

The defining innovation is "content-defined chunking," which dictates how key-value pairs are
grouped into nodes. Unlike standard B-trees (fixed capacity splits), Prolly trees calculate
boundaries probabilistically based on the data itself.

Dolt's critical refinement: restrict hash input exclusively to keys, ignoring values. This
means value updates don't trigger cascading chunk boundary shifts.

Chunk boundaries use a CDF-based probability function targeting ~4KB chunks (aligned with OS
page sizes). This produces a tight normal distribution of chunk sizes, avoiding the geometric
distribution problems of early implementations.

### Performance Characteristics

| Operation      | Standard B-Tree | Prolly Tree              | Notes |
|----------------|-----------------|--------------------------|-------|
| Random Read    | O(log_k(n))     | O(log_k(n))              | Identical |
| Random Write   | O(log_k(n))     | O((1+k/w) * log_k(n))   | Minor hash overhead |
| Ordered Scan   | O(z/k)          | O(z/k)                   | Efficient range queries |
| Calculate Diff | O(n)            | **O(d)**                 | Proportional to diff size |
| Structural Sharing | None        | **High**                 | Native deduplication |

## Merkle Search Trees (MSTs)

MSTs are deployed as the core repository architecture of the AT Protocol (Bluesky). They use
leading zeros of cryptographic hashes to assign vertical node levels, functioning like a
deterministic Radix tree or Skip List.

### Vulnerabilities

- **Hash-mining DoS:** Attackers can craft keys with many leading zeros to force pathological
  tree height, degrading O(log n) to O(n).
- **Range scan overhead:** Data keys distributed vertically require up-and-down traversal for
  ordered scans, unlike Prolly trees' linear leaf-level iteration.

## Merkle-DAG CRDTs and E2EE Synchronization

### The Beelay Algorithm and Sediment Trees

For E2EE contexts (where sync servers can't read plaintext), the Beelay algorithm introduces:

- **Auth Graph + Commit Graph** hybrid architecture
- **RIBLT** (Rateless Invertible Bloom Lookup Tables) for efficient key reconciliation
- **Sediment Trees:** Deterministic chunking of encrypted ciphertext using hash leading zeros.
  All peers independently calculate identical chunk boundaries without decryption, enabling
  single-round-trip E2EE sync.

## IPLD Advanced Data Layouts

### HAMT (Hash Array Mapped Tries)

Deterministic, unordered key-value maps. Excellent for exact-match point queries. No key
ordering = no range queries. Uses bitWidth + bitmap + popcount for compact routing.

### Flexible Byte Layout (FBL)

Recursive multi-block schema for large binary assets. **Security note:** The internal length
parameter is untrusted and unverified by CID hash -- must be treated as a hint only.

## Secondary Indexing in CAS

Secondary indexes must be derived Merkle Trees (separate Prolly Trees) mapping index values
to primary keys. Key design considerations:

- Keys must be globally unique (concatenate index value + primary CID)
- Normalize keys for consistent lexicographic sorting
- Trade-off: compute-heavy sync (rebuild indexes locally) vs bandwidth-heavy sync (sync all
  index trees)

## Integration with Zenoh

### Storages and Key Expression Routing

Zenoh Storages act as subscriber + queryable: persist publications, respond to queries.
CIDs map directly to key expressions, with pluggable backends (RocksDB, filesystem, S3).

### Anti-Entropy Alignment

Zenoh's anti-entropy protocol uses temporal Merkle trees (eras, intervals, subintervals)
to align divergent storages -- structurally identical to Prolly Tree diffing. Coupling
the database's CAS logic to Zenoh's anti-entropy engine approaches the theoretical minimum
synchronization overhead.

### Dynamic Merkle Proofs via Queryables

Thin clients dispatch queries; compute nodes walk local indexes and return cryptographically
verifiable Merkle Proofs. Client verifies against trusted root CID without syncing full dataset.

## Architectural Recommendation

### Comparative Matrix

| Dimension            | Prolly Trees | MSTs       | Merkle-DAG CRDTs | IPLD HAMT |
|----------------------|-------------|------------|-------------------|-----------|
| Use Case             | Relational  | Key-value  | Collaborative     | Unordered maps |
| Key Ordering         | Lexicographic | Lexicographic | Causal        | Unordered |
| History Independence | Strict      | Strict     | Eventually consistent | Strict |
| E2EE Compatibility   | Low         | Low        | **High (Sediment Trees)** | Moderate |

### Phased Implementation

**Phase 1 (v0.1): Hybrid Architecture**
- Source of truth: Raw Merkle DAG via harmony-content
- State gossiping: Root CID published to Zenoh
- Local query cache: sled or redb (derived, rebuildable)

**Phase 2 (v1.0): Native CAS Database (harmony-db)**
- Prolly Tree implementation with CDF chunking
- Primary table + secondary index trees under atomic commits
- O(d) diff/sync over Zenoh anti-entropy
- Queryable Merkle proofs for thin clients
- Beelay-inspired Sediment Trees for E2EE sync

## References

1. AT Protocol Repository Spec - https://atproto.com/specs/repository
2. Zenoh Documentation - https://zenoh.io/
3. Dolt Prolly Tree Architecture - https://docs.dolthub.com/architecture/storage-engine/prolly-tree
4. IPLD Specifications - https://ipld.io/
5. Beelay Protocol (Alex Good) - https://www.youtube.com/watch?v=neRuBAPAsE0
6. Keyhive: Local-first access control - https://www.inkandswitch.com/keyhive/notebook/
7. CRDTs Turned Inside Out - https://interjectedfuture.com/crdts-turned-inside-out/
8. Zenoh Storage Alignment - https://zenoh.io/blog/2022-11-29-zenoh-alignment/
9. Accelerating Prolly Trees - https://ceur-ws.org/Vol-3791/paper8.pdf
10. IPLD HAMT Specification - https://ipld.io/specs/advanced-data-layouts/hamt/spec/
