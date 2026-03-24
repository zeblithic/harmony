# S3 Great Library Design

## Goal

Provide a global, permanent backing store for Harmony's content-addressed storage layer. Durable books (immutable, BLAKE3-addressed) are archived to S3 once and never rewritten. S3 serves as the "Great Library" — a last-resort source for any content that no live mesh node currently holds.

## Architecture

Two new components:

- **harmony-s3 crate** — Thin async S3 client wrapper. PUT/GET/HEAD books keyed by hex-encoded ContentId. No protocol logic.
- **Archivist service** — Background task in harmony-node that subscribes to Zenoh content publish events, filters for durable content, and uploads to S3.

The archivist is stateless — it tracks no local record of what it has uploaded. Dedup relies on S3 HEAD checks before upload. Idempotency is guaranteed by CAS: writing the same CID twice produces the same object.

## S3 Client (harmony-s3)

### S3Library struct

```rust
pub struct S3Library {
    client: aws_sdk_s3::Client,
    bucket: String,
    prefix: String,
}
```

### Object key format

```
{prefix}book/{cid_hex_64chars}
```

Example: `harmony-zenoh/book/00a1b2c3d4e5f60708091011121314151617181920212223242526272829303132`

The full 32-byte ContentId is hex-encoded (64 chars). This includes the 4-byte header (flags, depth, size, checksum) and the 28-byte hash. Using the full CID as the key (not just the hash) preserves content metadata in the path and ensures different content classes with the same hash don't collide.

### Operations

- **`put_book(cid, data)`** — `PutObject` with `StorageClass::IntelligentTiering`. Idempotent.
- **`get_book(cid)`** — `GetObject`, returns `Option<Vec<u8>>`. `None` if `NoSuchKey`.
- **`exists(cid)`** — `HeadObject`, returns `bool`. Used for dedup before upload.

### Storage class

All PUTs use `INTELLIGENT_TIERING`. S3 automatically transitions objects:
- Frequent Access → Infrequent Access (30 days)
- → Archive Instant Access (90 days)
- → Archive Access (180 days, optional)
- → Deep Archive (270+ days, optional)

No application-level tiering logic needed. Configure Archive/Deep Archive tiers via the S3 bucket's Intelligent-Tiering configuration.

### Credentials

Resolved via the standard AWS SDK credential chain: environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`), `~/.aws/credentials`, EC2 instance profile, or ECS task role. No credentials in the Harmony config file.

### Dependencies

```toml
aws-config = "1"
aws-sdk-s3 = "1"
tokio = { workspace = true }
```

## Archivist Service

### Subscription

The archivist subscribes to `harmony/content/publish/*` — the existing Zenoh key expression for newly published content. When a book is published to the mesh, the archivist receives it.

### Content class filtering

Check the ContentId's flags to determine durability:

| encrypted | ephemeral | Class | Archive to S3? |
|-----------|-----------|-------|---------------|
| false | false | PublicDurable (00) | Yes |
| true | false | EncryptedDurable (10) | Yes |
| false | true | PublicEphemeral (01) | No |
| true | true | EncryptedEphemeral (11) | No |

Only durable content (ephemeral=false) is archived. Encrypted durable content is stored as opaque ciphertext — S3 never sees plaintext. Security relies on the client-side encryption (ChaCha20-Poly1305), not S3.

### Upload flow

1. Receive published book via Zenoh subscriber
2. Extract ContentId from the publication key expression
3. Check content flags — skip if ephemeral
4. Call `s3.exists(cid)` — skip if already archived (HEAD request, ~$0.0004/1000)
5. Call `s3.put_book(cid, data)` — upload with Intelligent-Tiering
6. Log success via `tracing::info!`

### Idempotency

- No local state. Every decision is based on content flags (from CID) and S3 existence (HEAD).
- Multiple archivist nodes can run simultaneously — racing PUTs produce identical objects.
- Restart-safe — the archivist re-subscribes and continues archiving new content.

### Configuration

```rust
pub struct ArchivistConfig {
    pub bucket: String,        // e.g., "zeblithic"
    pub prefix: String,        // e.g., "harmony-zenoh/"
    pub region: Option<String>, // defaults to AWS_DEFAULT_REGION env
}
```

Added to `ConfigFile`:
```rust
pub archivist: Option<ArchivistConfig>,
```

If `None`, no archiving happens. The node operates normally without S3.

### Event loop integration

The archivist runs as an independent tokio task, communicating with the rest of the system only through Zenoh pub/sub. No changes to the runtime or StorageTier — the archivist is a consumer of Zenoh events, not a protocol participant.

```rust
if let Some(ref archivist_config) = config.archivist {
    let s3 = harmony_s3::S3Library::new(archivist_config).await?;
    let session = session.clone();
    tokio::spawn(async move {
        harmony_s3::archivist::run(s3, session).await;
    });
}
```

## Integrity and Trust Model

- **CID verification:** Any consumer fetching a book from S3 (future: harmony-aa7 fallback resolver) hashes the downloaded bytes and compares to the CID. Mismatch → reject. S3 is untrusted storage.
- **No versioning:** S3 object versioning is unnecessary — content is immutable by definition.
- **No deletion:** The archivist never deletes from S3. Books are permanent. Cost management is handled by Intelligent-Tiering lifecycle policies at the bucket level.
- **No dedup across encryption:** Two nodes encrypting the same plaintext with different keys produce different ciphertexts and different CIDs. Both are stored as separate S3 objects. This is correct — they ARE different content from the CAS perspective.

## Cost Model

Assuming 1TB of durable content archived over time:

| Time since write | Storage class | Cost/GB/mo | 1TB/mo |
|-----------------|--------------|-----------|--------|
| 0-30 days | Frequent Access | $0.023 | $23 |
| 30-90 days | Infrequent Access | $0.0125 | $12.50 |
| 90-180 days | Archive Instant | $0.004 | $4 |
| 180+ days | Archive Access | $0.0036 | $3.60 |
| 270+ days | Deep Archive | $0.00099 | $0.99 |

Steady-state cost for 1TB of cold library: **~$1/month**. PUT costs: $0.005/1000 requests. HEAD costs: $0.0004/1000. Reads (future fallback): $0.0004/1000 GET + $0.0007/GB transfer.

The WORM model means no rewrite costs ever. A book written once in 2026 costs $0.00099/GB/month forever in Deep Archive.

## Testing Strategy

### Unit tests (harmony-s3)

- Object key formatting: CID → S3 key for various flag combinations
- Content flag filtering: durability check logic
- Storage class specification in PUT requests

### Integration tests (feature-gated)

- `s3-integration` feature flag, requires AWS credentials + test bucket
- PUT/GET round-trip with byte verification
- HEAD existence check (positive and negative)
- CID integrity verification on GET

### Archivist tests (harmony-node, mock S3)

- MockS3Library backed by HashMap
- Verify durable content uploaded, ephemeral content skipped
- Verify HEAD-based dedup (second publish of same CID skips PUT)

## Scope Exclusions

- **S3 fallback resolver** — separate bead (harmony-aa7, P3)
- **Accounting/quota** — future work for multi-user cost sharing
- **StorageTier disk action wiring** — future enhancement (wire PersistToDisk → S3)
- **CDN/CloudFront** — future optimization for read performance
- **Cross-region replication** — future disaster recovery
- **Bundle/Vine upload** — archivist stores individual books; DAG structure is preserved because each book's CID is independent
