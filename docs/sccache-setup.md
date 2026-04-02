# sccache: Shared Rust Compilation Cache

Harmony uses [sccache](https://github.com/mozilla/sccache) with Cloudflare R2 as a shared compilation cache across all development machines. Cache hits are keyed by target triple, so all machines targeting the same triple share entries.

## Quick Start

If you use `nix develop` (or direnv), sccache is already in your PATH and `RUSTC_WRAPPER=sccache` is set automatically. You just need the credentials config.

### 1. Create sccache config

Create `~/.config/sccache/config` with your R2 credentials:

```toml
[cache.s3]
bucket = "harmony-sccache"
endpoint = "https://<ACCOUNT_ID>.r2.cloudflarestorage.com"
use_ssl = true
region = "auto"
key_prefix = ""
server_side_encryption = false

[cache.s3.credentials]
aws_access_key_id = "<R2_ACCESS_KEY_ID>"
aws_secret_access_key = "<R2_SECRET_ACCESS_KEY>"
```

This keeps credentials scoped to sccache only — they are not exposed to unrelated processes. Ask a team member for the R2 credentials or find them in the Cloudflare dashboard under R2 > API Tokens.

### 2. Verify

```bash
# Enter the dev shell (if not using direnv)
nix develop

# Check sccache is active
sccache --show-stats

# Run a build
cargo check -p harmony-node

# Verify cache hits
sccache --show-stats
```

On a warm cache, you should see a high cache hit rate. First build on a new machine may show mostly misses (compiles are uploaded to R2 for next time).

## How It Works

- `RUSTC_WRAPPER=sccache` is set in the Nix devShell
- `CARGO_INCREMENTAL=0` is required (incremental builds are host-specific, incompatible with shared caching)
- sccache intercepts each `rustc` invocation, hashes the inputs, and checks R2
- Cache hits download the compiled artifact instead of recompiling
- Cache misses compile normally and upload the result to R2

## Without Nix

If you're not using `nix develop`, install sccache manually and set the environment:

```bash
cargo install sccache

export RUSTC_WRAPPER=sccache
export CARGO_INCREMENTAL=0
```

The `~/.config/sccache/config` file works the same way regardless of how sccache is installed.

## Cache Key Details

sccache keys by: rustc version + target triple + crate source hash + compiler flags. This means:
- Same rustc version + same target → cache hit (even across machines)
- Different rustc versions → cache miss (expected on version bumps)
- x86_64 host building for aarch64-musl shares cache with RPi5 building natively for aarch64-musl
