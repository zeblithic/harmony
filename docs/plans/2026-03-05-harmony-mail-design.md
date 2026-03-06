# Harmony Mail Gateway Design

**Date:** 2026-03-05
**Status:** Approved

## Summary

`harmony-mail` is a new crate in the Harmony workspace — a single Rust binary that
serves as a bidirectional bridge between SMTP (legacy internet email) and the Harmony
network. Any domain operator deploys it on their VPS, points DNS at it, and their
domain becomes a Harmony email gateway.

The first instance is `q8.fyi`, but the software is domain-agnostic. Victoria at
`vrk.dev`, or anyone with a domain and a VPS, can run the same binary and offer
Harmony-backed email to their users.

When both sender and recipient are on Harmony, SMTP is bypassed entirely — messages
travel as E2EE Harmony-native envelopes over Zenoh pub/sub. The gateway only handles
translation at the boundary with legacy internet email.

## Decision Record

- **Single custom binary (Approach B)** over Stalwart-as-frontend: Single binary is
  critical for operator adoption. `harmony-mail --domain vrk.dev` is infinitely more
  approachable than configuring Stalwart + a bridge service. Full control over every
  byte. No AGPL concerns.

- **Stalwart utility crates as dependencies** (not Stalwart server): `mail-parser`,
  `mail-auth`, `mail-builder`, `mail-send` are all Apache-2.0/MIT dual-licensed.
  They handle RFC compliance (DKIM/SPF/DMARC/ARC, RFC 5322 parsing, message
  construction) without pulling in the AGPL server. Stalwart server source is
  reference-only — no code imported.

- **Harmony-native message format** over RFC 5322 internally: Messages between
  Harmony users use a lean binary format (~139 bytes for a short email vs ~1000+
  bytes for RFC 5322). RFC 5322 only exists at the gateway boundary. Export-to-RFC5322
  capability provided for interop/portability.

- **Network-wide name registry via announces** over gateway-local databases:
  Name-to-identity mappings are published as Harmony announces and cached across the
  network. Any node can resolve `jake_z@q8.fyi` without contacting the q8.fyi
  gateway. Harmony's announce system keeps the cache fresh.

- **v1 scope: inbound SMTP + outbound SMTP + attachments + spam filtering.** IMAP
  server, Gmail import, JMAP, and SMS bridging are documented as future roadmap items.

## v1 Scope

| In Scope | Out of Scope (documented in roadmap) |
|----------|--------------------------------------|
| Inbound SMTP (receive from internet) | IMAP server (v1.1) |
| Outbound SMTP (send to internet) | Gmail/Outlook import (v1.2) |
| MIME attachment handling | JMAP API (v1.3) |
| Spam filtering (DNSBL + auth + heuristics + Harmony trust) | Bayesian/ML spam (v2) |
| DKIM signing/verification | Mailing lists (v2.1) |
| SPF/DMARC checking | PGP gateway encryption (v2.2) |
| TLS (rustls + ACME) | SMS bridge (v3+) |
| Harmony-native E2EE messaging | WASM plugin migration (v3+) |
| Cross-gateway trust coordination | CalDAV/CardDAV |
| Name registration via announces | |
| Outbound queue with retry | |

## Architecture

### Component Diagram

```
                        +-------------------------------------+
                        |          harmony-mail binary         |
                        |                                      |
  Internet <----------->|  SMTP Listener (inbound, port 25)   |
  (Gmail, Outlook,      |  SMTP Submitter (outbound, lettre)  |
   other gateways)      |  DKIM/SPF/DMARC (mail-auth)         |
                        |  Message Parser (mail-parser)        |
                        |  Message Builder (mail-builder)      |
                        |  Spam Filter (DNSBL + rate-limit)    |
                        |  TLS (rustls + ACME)                 |
                        |                                      |
                        |  -- Harmony Bridge Layer --          |
                        |  Address Resolver (announce cache)   |
                        |  Format Translator (RFC5322 <> native)|
                        |  Attachment Mapper (MIME <> blobs)   |
                        |  Trust Coordinator (cross-gateway)   |
                        |                                      |
                        |  -- Harmony Core (library deps) --   |
                        |  harmony-identity (keypairs, ECDH)   |
                        |  harmony-crypto (E2EE, HKDF)         |
                        |  harmony-content (CIDs, blobs)       |
                        |  harmony-reticulum (announces)       |
                        |  harmony-zenoh (pub/sub, envelopes)  |
                        +-------------------------------------+
                                        ^
                                        | Harmony Network
                                        v
                              Other Harmony Nodes
                              Other Mail Gateways
                              Harmony-native clients
```

### Crate Dependencies

```
harmony-mail (new, binary crate)
  +-- harmony-identity  (Ed25519/X25519 keypairs, address derivation)
  +-- harmony-crypto    (ChaCha20-Poly1305, HKDF, hashing)
  +-- harmony-content   (CIDs, blobs, chunking)
  +-- harmony-reticulum (announces, packet format)
  +-- harmony-zenoh     (pub/sub, E2EE envelopes)
  +-- mail-parser       (RFC 5322 parsing, MIME, 41 charsets)  [Apache-2.0/MIT]
  +-- mail-auth         (DKIM/SPF/DMARC/ARC verification)      [Apache-2.0/MIT]
  +-- mail-builder      (RFC 5322 message construction)         [Apache-2.0/MIT]
  +-- lettre            (async SMTP client for outbound)        [MIT]
  +-- rustls            (TLS, pure Rust)
  +-- instant-acme      (ACME/Let's Encrypt client)             [Apache-2.0]
  +-- tokio             (async runtime)
```

## Message Flows

### Inbound: Internet to Harmony

1. External sender (e.g., Gmail) connects to port 25
2. TLS negotiated (STARTTLS or implicit TLS on 465)
3. SMTP handshake: EHLO -> MAIL FROM -> RCPT TO -> DATA
4. At RCPT TO: resolve local part -> Harmony identity via announce cache
5. At DATA: `mail-auth` verifies DKIM/SPF/DMARC. Spam filter scores.
6. `mail-parser` extracts headers, body, attachments
7. Format translation: RFC 5322 -> HarmonyMessage
8. Attachments -> `harmony-content` blobs (chunked, CID-addressed)
9. Message encrypted with recipient's X25519 public key (ChaCha20-Poly1305)
10. Published to recipient's Zenoh key expression

### Outbound: Harmony to Internet

1. Harmony user publishes outbound message to their gateway's key expression
2. Gateway receives it, decrypts with relay key
3. Format translation: HarmonyMessage -> RFC 5322 via `mail-builder`
4. Harmony content blob CIDs -> MIME attachments
5. DKIM-sign with gateway's domain key via `mail-auth`
6. Send via `lettre` to destination MX (with queue + retry)
7. Bounce handling: delivery failures -> HarmonyMessage bounce back to sender

### Harmony-to-Harmony (SMTP bypassed)

1. Sender's client resolves recipient identity from announce cache
2. Message sent directly as Harmony E2EE envelope over Zenoh
3. No SMTP involved. No gateway touched. No format translation.
4. Recipient's client receives and renders natively
5. If both users are on the same gateway domain, the gateway isn't even involved

## Address Scheme

### Address Formats

Every gateway domain supports three address patterns, checked in this order:

| Pattern | Example | Resolution |
|---------|---------|------------|
| Hex identity | `a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6@q8.fyi` | Direct: hex IS the 128-bit address hash |
| Named identity | `jake_z@q8.fyi` | Lookup: name=jake, namespace=z -> address hash via announce |
| Vanity alias | `victoria@vrk.dev` | Operator-defined: local alias -> address hash |

The hex format is the universal fallback — it always works, requires no registration,
and is how machines address each other. The named format is for humans. Vanity aliases
are a convenience layer the domain operator controls freely.

### Name Registration via Announces

When a user registers `jake` under namespace `z` on the `q8.fyi` gateway, the gateway
publishes a Harmony announce containing:

```
NameRegistration {
    name: "jake",
    namespace: "z",
    domain: "q8.fyi",
    identity: Identity (X25519 pub + Ed25519 pub),
    registered_at: timestamp,
    signature: Ed25519 signature over all above fields  // USER signs
    domain_signature: Ed25519 signature                 // GATEWAY co-signs
}
```

The user signs with their own Ed25519 key. The gateway co-signs with its domain key
to attest authorization. This dual-signature means:

- Anyone can verify the user owns the name (user's signature)
- Anyone can verify the domain operator authorized it (domain's signature)
- A rogue gateway can't reassign names without detection — the user's signature
  would be missing

The announce propagates through the Harmony network. Any node can resolve
`jake_z@q8.fyi` -> identity without contacting the q8.fyi gateway.

### Namespace Conventions

| Namespace | Meaning | Managed by |
|-----------|---------|------------|
| `z` | Zeblithic namespace on q8.fyi | Zeblithic |
| `vrk` | Victoria's namespace on vrk.dev | Victoria |
| `harmony` | Official Harmony project names | Project governance |
| `ens` | Bridged from Ethereum Name Service | Automated bridge |
| `nostr` | Bridged from Nostr NIP-05 | Automated bridge |

Namespaces are just strings — no protocol-level enforcement of meaning. Domain
operators decide which namespaces they host and what the registration policy is.

### Conflict Resolution

Domain scoping eliminates conflicts. `jake_z@q8.fyi` and `jake_z@vrk.dev` are
entirely different identities. Within a single domain, the gateway operator is
authoritative (they co-sign registrations). Conflicting registrations for the same
name+namespace under the same domain: nodes keep the one with the earlier
`registered_at` timestamp and valid dual-signature chain.

### Hex Address Spam Scoring Mitigation

Hex local parts trigger SpamAssassin's `FROM_LOCAL_HEX` rule. Mitigations:

- **Inbound hex**: Fine. The receiving gateway owns its scoring config.
- **Outbound from hex**: Gateway rewrites `From:` to human-readable name if
  registered (`jake_z@q8.fyi`), falling back to `user-a1b2c3d4@q8.fyi`
  (truncated, with prefix). Full hex goes in `X-Harmony-Identity:` header.
- **Between Harmony gateways**: Irrelevant — SMTP is bypassed.

## Harmony-Native Message Format

### Design Principles

- Lean over compatible — NOT RFC 5322. Purpose-built for E2EE transport.
- Content-addressed attachments — CID pointers, not inlined MIME base64.
- Binary encoding — fast to parse, small on the wire.
- Fits inside existing E2EE envelope — the message is the plaintext payload
  inside the 33-byte header ChaCha20-Poly1305 envelope.

### Wire Format

```
HarmonyMessage {
    version: u8,                          // 1 byte  -- format version (0x01)
    message_type: u8,                     // 1 byte  -- 0x00=email, 0x01=receipt, 0x02=bounce
    flags: u8,                            // 1 byte  -- bit 0: has_attachments, bit 1: is_reply, bit 2: is_forward
    timestamp: u64,                       // 8 bytes -- unix millis, sender's clock
    message_id: [u8; 16],                 // 16 bytes -- random, unique per message
    in_reply_to: Option<[u8; 16]>,        // 0 or 16 bytes -- threading
    sender_address: [u8; 16],             // 16 bytes -- sender's address hash
    recipient_count: u8,                  // 1 byte
    recipients: Vec<Recipient>,           // variable
    subject_len: u16,                     // 2 bytes -- big-endian
    subject: Vec<u8>,                     // variable -- UTF-8
    body_len: u32,                        // 4 bytes -- big-endian
    body: Vec<u8>,                        // variable -- UTF-8 plaintext or markdown
    attachment_count: u8,                 // 1 byte
    attachments: Vec<AttachmentRef>,      // variable
}

Recipient {
    address_hash: [u8; 16],              // 16 bytes
    recipient_type: u8,                   // 1 byte -- 0x00=to, 0x01=cc, 0x02=bcc
}

AttachmentRef {
    cid: [u8; 32],                        // 32 bytes -- content ID (BLAKE3 hash)
    filename_len: u8,                     // 1 byte
    filename: Vec<u8>,                    // variable -- UTF-8
    mime_type_len: u8,                    // 1 byte
    mime_type: Vec<u8>,                   // variable -- ASCII
    size: u64,                            // 8 bytes -- original file size
}
```

### Size Comparison

Typical short email ("Hey, meeting at 3pm tomorrow?"), one recipient, no attachments:

| Format | Size |
|--------|------|
| RFC 5322 (with headers) | ~800-1200 bytes |
| HarmonyMessage | ~90 bytes |
| + E2EE envelope overhead | +33 bytes header + 16 bytes poly1305 tag |
| **Total on wire** | **~139 bytes** |

Message with a 50MB PDF attachment:

| Format | Size on wire |
|--------|-------------|
| RFC 5322 + MIME base64 | ~67MB (base64 bloat) + headers |
| HarmonyMessage | ~180 bytes (message) + 32-byte CID pointer |
| Attachment blob | 50MB transferred separately via harmony-content |

### Body Format

UTF-8 text. Rich content uses Markdown (CommonMark). No HTML email, no CSS, no
tracking pixels, no remote image loads. Deliberate anti-spam and anti-surveillance
choice.

HTML email received from the internet is converted to Markdown (stripping scripts,
styles, tracking) during format translation. Original HTML optionally preserved as
an attachment blob.

### Threading

`message_id` (16 random bytes) + `in_reply_to` (parent's message_id) form a simple
thread chain. Client builds the tree. No `References:` header explosion.

### Delivery Receipts and Bounces

- `message_type = 0x01` (receipt): Sent by recipient's client to confirm delivery.
  Contains original `message_id` + timestamp. Optional per user preference.
- `message_type = 0x02` (bounce): Sent by gateway when SMTP outbound delivery fails.
  Contains original `message_id` + UTF-8 reason in body.

### RFC 5322 Translation (gateway boundary)

**Inbound (RFC 5322 -> HarmonyMessage):**

- `From:` -> look up sender in announce cache, or hash raw email as sender_address
- `To:/Cc:/Bcc:` -> resolve each to Harmony identity
- `Subject:` -> subject field
- `Date:` -> timestamp
- `Message-ID:` -> hash to 16 bytes (or generate new)
- `In-Reply-To:` -> map if known, else None
- Body: extract text/plain (prefer) or convert text/html -> markdown
- MIME attachments: each part -> `harmony-content` blob -> AttachmentRef

**Outbound (HarmonyMessage -> RFC 5322):**

- sender_address -> look up registered name -> `From: jake_z@q8.fyi`
- recipients -> `To:/Cc:` headers (BCC stripped, delivered separately per SMTP)
- subject -> `Subject:`
- timestamp -> `Date:`
- message_id -> `Message-ID: <hex(message_id)@q8.fyi>`
- body -> text/plain part (markdown rendered to HTML for text/html alternative)
- AttachmentRef CIDs -> fetch blobs -> MIME parts with Content-Type/Disposition

## SMTP State Machine

### Listener Ports

| Port | Purpose | TLS Mode |
|------|---------|----------|
| 25 | Inbound relay (other MTAs deliver here) | STARTTLS (opportunistic -> required) |
| 465 | Submission (implicit TLS) | Implicit TLS from connection start |
| 587 | Submission (STARTTLS) | STARTTLS required before AUTH |

Ports 465/587 require authentication (Harmony identity signature challenge) before
accepting mail for outbound relay. Port 25 accepts unauthenticated inbound.

### Sans-I/O Design

Consistent with Harmony's architectural pattern (Reticulum Node, session state
machine). The SMTP handler is a pure function: `(State, Event) -> (State, Vec<Action>)`.

```
          CONNECTED
              |
              v
         GREETING_SENT -- client EHLO --> EHLO_RECEIVED
                                               |
                                       verify FCrDNS, check DNSBL
                                               |
                                               v
                                         EHLO_ACCEPTED
                                               |
                                     +--- STARTTLS? ---+
                                     v                  v
                               TLS_UPGRADE        (already TLS)
                                     |                  |
                                     v                  |
                               re-EHLO after TLS        |
                                     |                  |
                                     v                  v
                                  READY <--------------+
                                     |
                               MAIL FROM
                                     |
                                     v
                              MAIL_FROM_RECEIVED
                              (extract sender domain, begin SPF check)
                                     |
                               RCPT TO
                                     |
                                     v
                              RCPT_TO_RECEIVED
                              (resolve local part -> Harmony identity,
                               reject if unknown)
                                     |
                              (may repeat RCPT TO)
                                     |
                                DATA
                                     |
                                     v
                              DATA_RECEIVING
                              (stream body until lone ".\r\n")
                                     |
                                     v
                              MESSAGE_COMPLETE
                              +------+------+
                              v             v
                        DKIM/DMARC     spam score
                        verify         calculate
                              |             |
                              v             v
                           VERDICT
                         accept / reject / defer
                              |
                        +-----+------+
                        v            v
                   ACCEPTED     REJECTED
                   (translate    (550 response)
                    & deliver
                    to Harmony)
                        |
                        v
                   READY (await next MAIL FROM or QUIT)
```

### Event/Action Types

```rust
enum SmtpEvent {
    Connected { peer_ip: IpAddr, tls: bool },
    DataReceived(Vec<u8>),
    TlsCompleted,
    DnsblResult { listed: bool, list_name: String },
    SpfResult(SpfOutput),
    DkimResult(DkimOutput),
    DmarcResult(DmarcOutput),
    HarmonyResolved { local_part: String, identity: Option<Identity> },
}

enum SmtpAction {
    SendResponse(u16, String),
    StartTls,
    QueryDnsbl(IpAddr),
    CheckSpf { sender_domain: String, peer_ip: IpAddr },
    VerifyDkim(Vec<u8>),
    CheckDmarc { from_domain: String, spf: SpfOutput, dkim: DkimOutput },
    ResolveHarmonyAddress { local_part: String, domain: String },
    DeliverToHarmony { recipient: Identity, message: HarmonyMessage },
    Reject { code: u16, reason: String },
    Close,
}
```

### Connection Limits

| Limit | Value | Rationale |
|-------|-------|-----------|
| Max concurrent connections per IP | 5 | Prevents single-IP flood |
| Max connections per IP per hour | 50 | Sustained abuse prevention |
| Max RCPT TO per message | 100 | RFC 5321 recommendation |
| Max message size | 25 MB | Matches Gmail; large files go via content blobs |
| Connection timeout (idle) | 5 minutes | Free stale resources |
| DATA timeout | 10 minutes | Large messages on slow links |

## Spam Filtering

Layered defense. Each layer produces a score contribution. Cumulative score above
threshold triggers rejection.

### Layer 1: Connection-level (before any mail data)

- **DNSBL lookup**: Peer IP against Spamhaus ZEN, Barracuda, SpamCop.
  Listed -> +10 (instant reject, threshold is 5).
- **FCrDNS**: Forward-confirmed reverse DNS. Fail -> +3.
- **Rate limit exceeded**: -> immediate 421 (temp reject).

### Layer 2: Envelope-level (MAIL FROM / RCPT TO)

- **SPF check** via mail-auth: Fail -> +3. SoftFail -> +1. None -> +1.
- **Unknown recipient**: RCPT TO doesn't resolve -> 550 reject (no backscatter).

### Layer 3: Content-level (after DATA)

- **DKIM verification**: Fail -> +3. Missing -> +1.
- **DMARC evaluation**: Fail -> act per published policy (none/quarantine/reject).
- **ARC validation**: Valid ARC chain from trusted intermediary -> reduce DKIM penalty.
- **Basic heuristics**: Executable attachments (.exe, .scr, .bat) -> +5.
  Excessive URLs -> +1. Empty subject -> +1.

### Layer 4: Harmony-native trust

- **Known Harmony sender**: SMTP sender maps to known Harmony identity -> -3 (trust bonus).
- **Gateway reputation**: Cross-gateway trust score. High trust -> -2. Low trust -> +2.
- **First-contact penalty**: Sender never mailed this recipient before -> +1.

### Score Thresholds

```
<= 0 : Deliver (trusted)
1-4  : Deliver with X-Spam-Score header (borderline)
>= 5 : Reject at SMTP level (550)
```

## Outbound Queue & Retry

Persistent queue (SQLite or sled) for Harmony -> Internet outbound messages.

- **Immediate attempt**: Deliver via `lettre` to destination MX.
- **Retry schedule**: 5min, 15min, 30min, 1h, 2h, 4h, 8h, 24h, 48h, 72h
  (per RFC 5321 section 4.5.4.1 — retry for at least 4-5 days).
- **Bounce generation**: After final retry fails -> HarmonyMessage bounce
  (message_type = 0x02) back to sender.
- **Queue persistence**: Messages survive gateway restarts.

## TLS & Certificate Management

- **rustls** for all TLS (no OpenSSL dependency).
- **instant-acme** for automatic Let's Encrypt certificates.
- DNS-01 challenge preferred (works without port 80). HTTP-01 as fallback.
- Certificate auto-renewal 30 days before expiry.
- DANE/TLSA records documented as recommended, not required for v1.

## DKIM Signing (Outbound)

- Gateway generates Ed25519 key pair on first run.
- Public key published as DNS TXT: `harmony._domainkey.<domain>`.
- `mail-auth` signs outbound messages. Signed headers: From, To, Subject, Date,
  Message-ID, MIME-Version, Content-Type.
- Ed25519 preferred (RFC 8463). RSA-2048 as fallback for receivers that don't
  support Ed25519 — gateway can dual-sign.

## Cross-Gateway Trust Coordination

### Gateway Identity

Every gateway is a Harmony identity. On first run, generates Ed25519/X25519 keypair.
Publishes a Gateway Announce:

```
GatewayAnnounce {
    domain: "q8.fyi",
    gateway_identity: Identity,
    smtp_host: "mail.q8.fyi",
    capabilities: u16,             // bitfield: inbound, outbound, relay
    user_count: u32,               // approximate
    uptime_since: u64,
    signature: Ed25519 signature
}
```

### Trust Score System

Gateways share reputation scores over Harmony pub/sub on a well-known Zenoh key
expression: `harmony/mail/trust/v1`.

```
TrustReport {
    reporter: GatewayIdentity,
    subject: GatewayIdentity,
    domain: String,
    period: (u64, u64),            // start/end timestamps
    metrics: TrustMetrics,
    signature: Ed25519 signature
}

TrustMetrics {
    messages_received: u32,
    spam_ratio: f32,               // 0.0-1.0
    bounce_ratio: f32,
    dkim_pass_ratio: f32,
    spf_pass_ratio: f32,
    availability: f32,
}
```

**Aggregation:** Weighted median of all TrustReports, weighted by reporter's own
trust score (recursive, capped at depth 2). New gateways start neutral.

**Blacklisting:** Trust below threshold -> auto-reject. Harmony-native equivalent
of DNSBLs, managed by the operator community rather than centralized list operators.

## Operator Deployment Model

### Setup Workflow

```bash
# 1. Install
cargo install harmony-mail

# 2. Initialize (generates keys, creates config, prints DNS records needed)
harmony-mail init --domain vrk.dev --admin-email victoria@vrk.dev

# 3. Configure TLS
harmony-mail tls --acme --dns-challenge

# 4. Register users
harmony-mail user add --name victoria --namespace vrk --identity <harmony_pubkey>

# 5. Run
harmony-mail run
```

### Configuration

```toml
# /etc/harmony-mail/config.toml

[domain]
name = "vrk.dev"
mx_host = "mail.vrk.dev"

[gateway]
identity_key = "/etc/harmony-mail/gateway.key"
listen_smtp = "0.0.0.0:25"
listen_submission = "0.0.0.0:465"
listen_submission_starttls = "0.0.0.0:587"

[tls]
mode = "acme"
acme_email = "victoria@vrk.dev"
acme_challenge = "dns-01"

[dkim]
selector = "harmony"
algorithm = "ed25519"
key = "/etc/harmony-mail/dkim.key"

[spam]
dnsbl = ["zen.spamhaus.org", "b.barracudacentral.org"]
reject_threshold = 5
max_connections_per_ip = 5
max_message_size = "25MB"

[outbound]
queue_path = "/var/lib/harmony-mail/queue"
max_retries = 10
retry_schedule = [300, 900, 1800, 3600, 7200, 14400, 28800, 86400, 172800, 259200]

[harmony]
node_config = "/etc/harmony/node.toml"
trust_topic = "harmony/mail/trust/v1"
announce_interval = 3600
```

### DNS Records Required

```dns
; MX
<domain>.                          IN  MX   10  mail.<domain>.
mail.<domain>.                     IN  A        <IP>
mail.<domain>.                     IN  AAAA     <IPv6>

; SPF
<domain>.                          IN  TXT  "v=spf1 ip4:<IP> ip6:<IPv6> -all"

; DKIM
harmony._domainkey.<domain>.       IN  TXT  "v=DKIM1; k=ed25519; p=<PUBKEY>"

; DMARC (start monitoring, progress to reject)
_dmarc.<domain>.                   IN  TXT  "v=DMARC1; p=none; rua=mailto:dmarc@<domain>; fo=1"

; MTA-STS
_mta-sts.<domain>.                 IN  TXT  "v=STSv1; id=<timestamp>"

; TLS reporting
_smtp._tls.<domain>.               IN  TXT  "v=TLSRPTv1; rua=mailto:tls@<domain>"

; Autodiscovery
_imaps._tcp.<domain>.              IN  SRV  0 1 993 mail.<domain>.
_submission._tcp.<domain>.         IN  SRV  0 1 587 mail.<domain>.

; Reverse DNS (configured at hosting provider)
PTR <IP> -> mail.<domain>.
```

### Domain Warming

Built-in warming mode for new deployments:

```bash
harmony-mail warm --target 500 --days 42
```

Throttles outbound volume (5/day -> 500+/day over 6 weeks). Alerts if bounce rates
exceed 2% or complaint rates exceed 0.1%. DMARC policy progression: `p=none` ->
`p=quarantine; pct=25` (week 5) -> `p=quarantine; pct=100` (week 7) ->
`p=reject` (week 9+).

### Vanity Alias Management

```bash
harmony-mail alias add victoria victoria@vrk.dev
harmony-mail alias add support support-team-identity@vrk.dev
harmony-mail alias add ceo victoria@vrk.dev
```

Aliases are gateway-local (not announced to network). Operator convenience only.

## Future Roadmap

### v1.1: IMAP Server

IMAP4rev2 (RFC 9051) listener in the gateway binary. Virtual mailbox view:
Harmony messages rendered as RFC 5322 on demand when IMAP client requests them.
IMAP IDLE via Zenoh subscription for real-time notifications. The gateway does
not store RFC 5322 — source of truth is always Harmony-native.

### v1.2: Gmail Import / External Account Bridging

OAuth2 IMAP pull to migrate existing Gmail inbox into Harmony. `async-imap` with
XOAUTH2 authentication. Incremental sync via IMAP IDLE + polling fallback.
Each imported message parsed -> HarmonyMessage -> Harmony content system.
Attachments deduplicated by CID. Same architecture works for Outlook
(Microsoft Entra OAuth2). Google CASA assessment required for >100 users.

### v1.3: JMAP API

JMAP Core (RFC 8620) + JMAP Mail (RFC 8621). Natural fit since HarmonyMessage
is already structured data. Enables fast web/mobile clients without IMAP complexity.

### v2: Advanced Spam Intelligence

Bayesian classifier trained per-user. Collaborative filtering via anonymized spam
signals shared across gateways. URL reputation checking. Optional rspamd sidecar
for operators wanting industrial-strength filtering.

### v2.1: Mailing Lists / Group Addresses

Gateway-local group definitions expanding to N identities. ARC signing for
redistributed SMTP-origin messages. Harmony-native groups via shared Zenoh key
expression with group key agreement.

### v2.2: PGP/S/MIME Gateway Encryption

WKD and keyserver lookup for recipient PGP keys. Gateway encrypts outbound
RFC 5322 with recipient's PGP key before SMTP delivery. Opt-in per user.
Documented limitation: gateway-level encryption, not true E2E from sender's key.

### v3+: SMS Bridge

Conceptually similar bridge pattern but fundamentally different: requires telco
integration (Twilio/Vonage), per-message costs, heavy regulation (TCPA, GDPR).
Best as separate `harmony-sms` crate sharing identity infrastructure. Overlap
with email: OTP identity linking, announce system, trust coordination all
share infrastructure.

### v3+: WASM Plugin Migration

WASI Preview 2 can't support full mail server today (no threading, no TLS).
Path: native binary handles I/O, wasmtime plugin runtime handles message
processing (parsing, translation, spam scoring — all pure computation).
As WASI matures, more migrates to WASM.

### Feature Priority Matrix

| Feature | Version | Depends On | Value |
|---------|---------|------------|-------|
| Inbound SMTP | v1 | -- | Can receive email |
| Outbound SMTP | v1 | -- | Can send email |
| Attachments (MIME <> blobs) | v1 | harmony-content | Real email |
| Spam filtering | v1 | -- | Survivable inbox |
| IMAP server | v1.1 | v1 | Existing client support |
| Gmail import | v1.2 | v1.1 | Migration story |
| JMAP API | v1.3 | v1 | Modern client API |
| Bayesian spam | v2 | v1 spam | Adaptive filtering |
| Mailing lists | v2.1 | v1 | Group communication |
| PGP gateway | v2.2 | v1 outbound | Enhanced privacy |
| SMS bridge | v3+ | identity system | Second identity anchor |
| WASM plugins | v3+ | wasmtime runtime | Decentralized processing |

## Key RFCs

The gateway must comply with or implement these core RFCs:

| RFC | Subject | How Used |
|-----|---------|----------|
| 5321 | SMTP | Core protocol, state machine |
| 5322 | Internet Message Format | Parsing/building at gateway boundary |
| 6376 | DKIM | Signing outbound, verifying inbound |
| 7208 | SPF | Verifying inbound sender authorization |
| 7489 | DMARC | Policy enforcement |
| 8617 | ARC | Forwarding chain preservation |
| 3207 | STARTTLS | Transport encryption upgrade |
| 8314 | Implicit TLS | Port 465 |
| 8461 | MTA-STS | Mandatory TLS policy |
| 8463 | Ed25519 DKIM | Preferred signing algorithm |
| 2045-2049 | MIME | Attachment handling at boundary |
| 5228 | Sieve | Future: user-defined filtering |
| 9051 | IMAP4rev2 | Future: v1.1 |
| 8620 | JMAP Core | Future: v1.3 |
| 8621 | JMAP Mail | Future: v1.3 |
