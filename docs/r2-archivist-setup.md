# Cloudflare R2 Content Archivist

Harmony uses Cloudflare R2 as a last-resort content archive for durable CAS
blobs that don't fit in local storage. R2 is S3-compatible with zero egress
fees, making it ideal for WORM (write once, read many) archival where reads
are infrequent but must be free when they happen.

## How It Works

```
harmony-node (cache full)
  └── eviction push ──→ harmony/archive/{cid} (Zenoh pub/sub)
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        LAN archivist    S3 archivist    other subscribers
        (disk storage)   (R2 upload)
```

The S3 archivist subscribes to content publish events, filters for durable
(non-ephemeral) CIDs, checks if the object already exists in R2 (HEAD
request), and uploads new content. On cache miss + disk miss, the read
fallback path fetches from R2 and re-injects into the local cache.

## Cost Estimate

| Resource | Free tier | Overage | Notes |
|----------|-----------|---------|-------|
| Class A ops (PUT) | 1M/month | $4.50/M | ~1M puts = ~1TB at 1MB avg |
| Class B ops (GET/HEAD) | 10M/month | $0.36/M | HEAD for dedup + GET for fallback |
| Storage | 10GB/month | $0.015/GB | ~$15/TB/month |
| Egress | Unlimited | $0 | Zero egress fees (R2's main advantage) |

For a last-resort archive with moderate write volume, expect $0-15/month.

## Setup

### 1. Create the R2 bucket

In the Cloudflare dashboard: **R2 > Create bucket**

- **Bucket name:** `harmony-content`
- **Location:** Auto (or choose a region close to your nodes)

### 2. Create API credentials

In the Cloudflare dashboard: **R2 > Manage R2 API tokens > Create API token**

- **Permissions:** Object Read & Write
- **Scope:** Apply to specific bucket: `harmony-content`
- Save the **Access Key ID** and **Secret Access Key**

If you already have R2 credentials from sccache setup (`harmony-sccache`
bucket), you can reuse the same token if its scope covers both buckets.
Otherwise, create a new token scoped to `harmony-content`.

### 3. Configure credentials

The AWS SDK reads credentials from standard locations. Set them via
environment variables or `~/.aws/credentials`:

**Option A — Environment variables:**

```bash
export AWS_ACCESS_KEY_ID="<R2_ACCESS_KEY_ID>"
export AWS_SECRET_ACCESS_KEY="<R2_SECRET_ACCESS_KEY>"
```

**Option B — AWS credentials file (`~/.aws/credentials`):**

```ini
[default]
aws_access_key_id = <R2_ACCESS_KEY_ID>
aws_secret_access_key = <R2_SECRET_ACCESS_KEY>
```

### 4. Configure harmony-node

Add the `[archivist]` section to your node's TOML config:

```toml
[archivist]
bucket = "harmony-content"
prefix = "v1/"
region = "auto"
endpoint = "https://<ACCOUNT_ID>.r2.cloudflarestorage.com"
```

Replace `<ACCOUNT_ID>` with your Cloudflare account ID (visible in the R2
dashboard URL or under **R2 > Overview**).

The `prefix` field namespaces objects within the bucket (`v1/book/{cid_hex}`).
This allows future schema migrations without bucket recreation.

### 5. Build with archivist feature

The archivist is behind a feature flag:

```bash
cargo build -p harmony-node --features archivist
```

Or in your node's Cargo config:

```toml
[features]
default = ["archivist"]
```

### 6. Verify

Start the node and check logs for:

```
S3 archivist + read fallback enabled  bucket=harmony-content prefix=v1/
```

To test the upload path, publish durable content and check the R2 dashboard
for new objects under `v1/book/`.

## Using with AWS S3

The same configuration works with AWS S3 — omit the `endpoint` field:

```toml
[archivist]
bucket = "my-harmony-bucket"
prefix = "v1/"
region = "us-west-2"
# endpoint omitted — uses AWS S3 endpoints with INTELLIGENT_TIERING storage class
```

## Related

- [sccache setup](sccache-setup.md) — Shared Rust compilation cache (also uses R2)
- [S3 Great Library design](superpowers/specs/2026-03-24-s3-great-library-design.md) — Architecture spec
- [S3 fallback design](superpowers/specs/2026-03-26-s3-fallback-design.md) — Read fallback on cache miss
