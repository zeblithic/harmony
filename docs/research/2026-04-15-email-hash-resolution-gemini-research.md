# Email-Style Identifier to Cryptographic Identity Resolution in Decentralized Messaging: Design Patterns, Attacks, and Trade-offs

**Date:** 2026-04-15
**Source:** Gemini Deep Research
**Commissioned for:** ZEB-120 design (`docs/superpowers/specs/2026-04-15-smtp-rcpt-admission-design.md`)
**Status:** Archival reference — this is the verbatim research output used to inform the ZEB-120 design decisions

---

## 1. Architectural Context and the Resolution Dilemma

The transition from centralized, siloed communication networks to decentralized, peer-to-peer messaging infrastructures introduces a fundamental challenge in identifier resolution. Users expect to interact using human-readable, domain-bound identifiers — specifically, the standard SMTP-style email address (`local_part@domain`). However, modern decentralized routing and encryption protocols, including the Harmony peer-to-peer email gateway, operate exclusively on cryptographic primitives.

Within the Harmony architecture, the discovery layer is keyed strictly by a 16-byte identity hash (`harmony/identity/{hash_hex}/{announce,resolve,alive}`). The `AnnounceRecord` is purely cryptographic; it intentionally omits metadata such as email addresses, display names, handles, or domain fields to preserve network-level privacy. Verification procedures validate only that a truncated SHA-256 hash of the encryption key concatenated with the public key matches the identity reference. Consequently, the network has no innate scaffolding for email resolution, leaving a critical void in the recipient resolution phase: transforming an SMTP-style email address into the requisite 16-byte `IdentityHash` before message sealing can occur.

The absence of an email → hash abstraction forces a complex architectural decision regarding trust locality, latency, and privacy. An analysis of the design space reveals four primary trajectories, each presenting severe trade-offs:

- **Public Email Announcements (Option A):** Broadcasting the `local_part@domain` directly within the Zenoh announce record. While this enables zero-RTT (Round Trip Time), network-wide resolution, it transforms the decentralized network into a globally visible, easily enumeratable directory of email addresses, presenting a catastrophic privacy failure and an optimal target for spam harvesting. It also introduces critical authorization questions regarding who is permitted to sign and broadcast a claim for a specific domain.

- **Gateway-Mediated Zenoh Queries (Option B):** Establishing a queryable Zenoh path (`harmony/mail/v1/resolve/{domain_hash}`) answered exclusively by the authoritative gateway for that domain. This preserves announce privacy but recreates the systemic flaws of SMTP: if the recipient's gateway is temporarily unavailable, resolution fails, creating a hard dependency that undermines the resilient, asynchronous nature of peer-to-peer messaging. Furthermore, Zenoh traffic analysis could still map resolution queries to specific domains.

- **HTTPS Domain Resolution (Option C):** Adopting a `.well-known/harmony-users` HTTPS endpoint architecture, heavily mirroring standard protocols like Nostr's NIP-05 and the W3C `did:web` standard. This is the simplest approach, requiring zero modifications to the Harmony wire protocol, and seamlessly aligns with existing DNS and Certificate Authority (CA) trust hierarchies. However, it mandates synchronous HTTPS requests on the critical send path and centralizes trust in legacy DNS/CA infrastructure, deviating from pure peer-to-peer principles.

- **Private Information Retrieval (Option D):** Utilizing advanced cryptographic techniques like PIR or ZipPIR to allow nodes to query a global registry without revealing the query target. While mathematically ideal, this approach relies heavily on hardware crypto-acceleration maturity (e.g., AVX-512) and imposes immense computational and infrastructural overhead. It has been flagged as ZEB-47 (Low priority) and is currently out of scope for immediate implementation.

The root design tension is the conflict between trust locality and privacy. To forge an optimal path forward for the Harmony gateway, this report conducts an exhaustive comparative survey of how established decentralized messaging ecosystems have navigated this exact design space. It systematically dissects threat models, evaluates state-of-the-art private lookup and verifiable binding protocols, and synthesizes an explicit, phased architectural recommendation.

---

## 2. Comparative Survey of Identity Resolution Architectures

The challenge of translating a human-readable namespace into a cryptographic namespace is a well-trodden design space. Decentralized, federated, and privacy-preserving messaging protocols have spent the last decade engineering solutions to the identifier → identity mapping problem. Evaluating these architectures exposes the diverse strategies used to balance privacy, usability, and verifiability.

### Nostr (NIP-05)

The Nostr protocol addresses user identifier resolution via NIP-05, a specification that establishes an HTTPS-based query mechanism closely mirroring the mechanics of WebFinger. In the Nostr ecosystem, users construct a Kind 0 (user metadata) event containing a `nip05` field populated with an internet identifier in the format `<local-part>@<domain>`.

To verify this claim and resolve the identifier, a client parses the string and executes an HTTP GET request to `https://<domain>/.well-known/nostr.json?name=<local-part>`. The domain's web server responds with a JSON document mapping the requested name to a hex-formatted public key.

**Concrete wire-format example:**

```json
{
  "names": {
    "bob": "b0635d6a9851d3aed0cd6c495b282167acf761729078d975fc341b22650b07b9"
  },
  "relays": {
    "b0635d6a9851d3aed0cd6c495b282167acf761729078d975fc341b22650b07b9": [
      "wss://relay.example.com"
    ]
  }
}
```

If the public key returned by the `.well-known` endpoint matches the public key that signed the user metadata event, the client concludes that the cryptographic identity is legitimately authorized by the domain. While NIP-05 is universally adopted due to its implementation simplicity, it is burdened by significant privacy and security trade-offs. It inherently centralizes trust in the domain owner and the underlying DNS/CA infrastructure. A compromised domain allows an attacker to silently alter the `nostr.json` file, mapping valid user identifiers to attacker-controlled keys. Furthermore, standard NIP-05 implementations lack enumeration protections; attackers can easily scrape the endpoint to harvest user lists.

### Matrix (MSC2134 and Identity Servers)

The Matrix protocol utilizes federated Identity Servers (IS) to map Third-Party Identifiers (3PIDs), such as email addresses and MSISDN phone numbers, to cryptographic Matrix User IDs (MXIDs). Historically, Matrix permitted plaintext lookups, which posed severe privacy risks by allowing identity servers to harvest complete social graphs from users' contact lists. To mitigate this vulnerability, Matrix introduced MSC2134, an identity hash lookup specification.

Under MSC2134, a client wishing to resolve an email address must first request a cryptographic "pepper" and a list of supported hashing algorithms from the Identity Server via a `GET /_matrix/identity/v2/hash_details` request. The server typically returns a pepper string and indicates support for `sha256`.

The client then formats the query by concatenating the identifier, the medium, and the pepper (e.g., `alice@example.com email matrixrocks`). This concatenated string is hashed using SHA-256, and the resulting bytes are encoded using URL-safe unpadded Base64. The client submits these hashes in a `POST /_matrix/identity/v2/lookup` request.

**Concrete wire-format example (MSC2134 request):**

```json
{
  "addresses": ["..."],
  "algorithm": "sha256",
  "pepper": "matrixrocks"
}
```

While peppered hashing prevents trivial plaintext scraping, it remains vulnerable to offline dictionary attacks if the pepper is ever compromised or leaked, prompting Matrix specifications to advise frequent pepper rotation. Recognizing that hashing all directory entries is computationally prohibitive for identity servers backed by large external systems (such as enterprise LDAP directories), Matrix reluctantly retains a fallback `none` algorithm for plaintext resolution, highlighting the practical difficulties of retrofitting privacy onto existing centralized directories.

### Signal (Phone Numbers, SGX, and ORAM)

Signal provides the industry's gold standard for privacy-preserving identifier resolution and sealed sender communication. Because Signal uses MSISDN phone numbers as the primary user-facing identifier, a naive discovery mechanism would allow the central Signal servers to construct a complete, plaintext social graph of global communications. To circumvent this, Signal engineered a Contact Discovery Service (CDS) built upon Intel Software Guard Extensions (SGX) secure enclaves.

In the Signal architecture, a client establishes a mutually authenticated, encrypted TLS connection that terminates directly inside the SGX enclave, entirely bypassing the host operating system's visibility. The client performs remote attestation to mathematically verify the enclave's code signature against published open-source measurements. Once attested, the client transmits their encrypted address book to the enclave.

However, executing within an enclave is insufficient; the host operating system can still deduce relationships by monitoring the enclave's RAM access patterns. To thwart side-channel memory analysis, Signal layers Path Oblivious RAM (Path ORAM) over the user database. Path ORAM structures memory as a tree. When the enclave queries a contact, it reads a random path from the root to a leaf, remaps the accessed item to a new random path, and writes the path back. This ensures that reading the exact same contact twice produces completely disparate memory access patterns, completely obfuscating the query from the host machine.

Once resolution is complete, Signal employs a "Sealed Sender" protocol to conceal routing metadata. Senders obtain a short-lived sender certificate from the service. The message payload and the sender certificate are encrypted to the recipient's identity key. The Signal server routes this opaque envelope using only a delivery token, ensuring the server learns the destination but remains entirely blind to the sender's identity.

### Session, Briar, and Scuttlebutt (Pubkey-Only and Derived Names)

Conversely, ultra-private networks like Session, Briar, and Secure Scuttlebutt (SSB) largely reject the premise of central authoritative resolution, opting instead to use cryptographic public keys directly as the user-facing identifier, or relying entirely on localized gossip for naming.

- **Session and Briar:** Users share long alphanumeric strings (derived directly from their public keys or Tor onion addresses) out-of-band. There is no network-wide registry mapping a human-readable name to these strings. While providing maximum decentralization and immunity to domain takeover, the user experience suffers significantly, as users cannot naturally discover each other without prior physical or out-of-band digital contact.

- **Secure Scuttlebutt (SSB):** Utilizes a decentralized "petname" system. Identities are fundamentally cryptographic feed hashes. Users locally assign human-readable names to these hashes within their own clients, and these assignments are gossiped across the network. While highly resilient and devoid of central authority, petname systems fail to provide globally canonical identifiers; "Alice" on one user's device may resolve to a completely different cryptographic hash than "Alice" on another user's device, enabling localized impersonation if the Web of Trust is weak.

### Farcaster and ActivityPub (WebFinger)

Federated social networks operating on ActivityPub (e.g., Mastodon) rely heavily on the WebFinger protocol (RFC 7033) for resolving cross-instance identifiers. An identifier formatted as `@user@domain.tld` is resolved by querying the authoritative domain's `.well-known/webfinger` endpoint.

**Concrete wire-format example:**

```
GET https://jambor.dev/.well-known/webfinger?resource=acct:seb@jambor.dev
```

```json
{
  "subject": "acct:seb@jambor.dev",
  "aliases": ["https://mastodon.social/@crepels"],
  "links": [
    {
      "rel": "self",
      "type": "application/activity+json",
      "href": "https://mastodon.social/users/crepels"
    }
  ]
}
```

The endpoint returns a JSON-LD ActivityStreams representation detailing the actor's canonical HTTPS URI, which serves as the actor's ID and links to their cryptographic keys. Much like Nostr's NIP-05, WebFinger prioritizes ease of deployment and broad interoperability but exposes the user directory to unauthenticated public scraping and centralizes absolute trust in domain hosting.

Farcaster takes a hybrid approach, utilizing FNames which are managed by a centralized username registrar mapping human-readable names to underlying cryptographic Farcaster IDs (FIDs), though the protocol also supports integration with decentralized Ethereum Name Service (ENS) resolution.

### ENS, Handshake, and Unstoppable Domains (Blockchain Registrars)

Blockchain-based name systems map human-readable identifiers (e.g., `alice.eth`) to cryptographic addresses using distributed ledgers. These systems provide absolute censorship resistance, cryptographic immutability, and protection against arbitrary domain seizure. However, they introduce severe operational constraints: mapping updates incur financial transaction costs (gas fees), resolution requires querying a blockchain node (introducing latency), and the public nature of the ledger results in maximum metadata exposure, permanently publicizing the linkage between an identifier and a cryptographic key.

### DIDComm and DID Resolution

The W3C Decentralized Identifier (DID) specification provides a formalized, URI-based framework for resolving identifiers into DID Documents, which contain the cryptographic keys necessary for DIDComm messaging. The resolution mechanism varies radically based on the DID method:

- **`did:web`:** Operates identically to NIP-05 and WebFinger, utilizing standard HTTPS requests to retrieve a `did.json` file from a domain. While highly attractive due to its low cost and simplicity, it is widely considered a "honey trap" in high-security environments because it lacks cryptographic tamper resistance, historical traversability, and relies entirely on centralized DNS security.
- **`did:ion` and `did:dht`:** `did:ion` utilizes the Sidetree protocol anchored to the Bitcoin blockchain, batching operations to reduce costs. `did:dht` leverages the BitTorrent Distributed Hash Table (DHT) for permissionless, lightweight identity resolution without a blockchain. Both prioritize decentralization but can suffer from state synchronization delays.
- **`did:webplus`:** Extends `did:web` by introducing Verifiable Data Gateways (VDGs) and requiring self-hashing DID documents to ensure tamper evidence.
- **`did:webvh` (Verifiable History):** A significant evolution of `did:web` that introduces a Self-Certifying Identifier (SCID). The DID Document is maintained as an append-only JSON Lines (`.jsonl`) file, containing a cryptographic hash chain of all historical updates. This ensures that a compromised DNS server cannot silently alter the key history without invalidating the cryptographic proofs. `did:webvh` also supports optional "witnesses" — third-party nodes that validate updates before publication, storing their cryptographic proofs in a separate `did-witness.json` file to minimize processing overhead during resolution.

### XMPP JID and Server-Bound Identity

In the XMPP ecosystem, end-to-end encryption is facilitated by OMEMO, an extension based on the Double Ratchet algorithm. Resolving a Jabber ID (JID) to an identity key relies on XEP-0384. Clients independently publish their device identity keys and prekeys as "bundles" to the server using the Personal Eventing Protocol (PEP).

When a user wishes to establish a secure session with a JID, their client queries the server for these bundles. Critically, XMPP does not intrinsically bind the JID to the keys via server-side cryptography; it relies on Trust on First Use (TOFU) and out-of-band manual fingerprint verification. If an attacker compromises the XMPP server, they can silently inject rogue device keys into a user's PEP node, facilitating seamless man-in-the-middle attacks unless the communicating parties manually audit their key fingerprints.

### Email Baseline: DKIM, MTA-STS, and DANE

Traditional SMTP infrastructure secures identity bindings using DomainKeys Identified Mail (DKIM). The authoritative domain owner publishes a public cryptographic key in DNS via a standard TXT record. Outgoing emails are processed by the Mail Transfer Agent (MTA); a cryptographic hash of the message content and headers (including the From address) is generated and signed with the corresponding private key. The receiving MTA fetches the public key from DNS to verify the signature, proving that the domain owner authorized the message.

**Concrete wire-format example (DKIM header):**

```
DKIM-Signature: v=1; a=rsa-sha256; c=relaxed/relaxed; d=example.com; s=selector1;
 h=from:to:subject:date:message-id;
 bh=xyz123HashValue=;
 b=abc987SignatureValue=;
```

While DKIM provides robust origin authentication, it is an infrastructure-level signature applied by the domain gateway, not an end-to-end cryptographic identity belonging to the individual user. DKIM is frequently paired with DANE (DNS-Based Authentication of Named Entities) and MTA-STS to mandate encrypted Transport Layer Security (TLS) connections between routing gateways, preventing downgrade attacks.

### Comparative summary

| Protocol | Transport | Cryptographic Binding Authority | Privacy Posture | Documented Failure Modes |
|---|---|---|---|---|
| Nostr (NIP-05) | HTTPS | DNS / Domain Owner | Public / Scrapable | Domain hijacking enables silent key replacement |
| Matrix (MSC2134) | HTTPS | Identity Server (Federated) | Peppered Hashes | Offline dictionary attacks if pepper leaks |
| Signal (CDS) | SGX / ORAM | Centralized Enclave | Highly Private | Hardware side-channel attacks on SGX |
| WebFinger | HTTPS | DNS / Domain Owner | Public / Scrapable | Enumeration, centralizes trust in domain hosting |
| did:webvh | HTTPS | Hash Chain + Witnesses | Public History | Key rotation loss, complex witness threshold logic |
| XMPP (OMEMO) | XMPP (XML) | Trust on First Use (TOFU) | Public Bundles | Server injection of rogue device keys |
| ENS / Handshake | Blockchain | Smart Contract Ledger | Public Ledger | High latency, on-chain metadata exposure, gas fees |
| DKIM | DNS TXT | Domain Owner (MTA) | Public | Secures transport infrastructure, not end-to-end user data |

---

## 3. Attack and Threat Analysis for Email → Hash Resolution

Mapping a mutable, human-readable email address to an immutable 16-byte identity hash introduces a massive attack surface. If the resolution mechanism is compromised, all subsequent cryptographic sealing is fundamentally invalidated.

### Impersonation and Name Squatting

The foundational vulnerability of any naming system is authorization: who possesses the cryptographic authority to bind `local_part@domain.tld` to a specific identity hash? In purely decentralized, unauthenticated systems — such as Option A (public email claims in Zenoh announces) or unmanaged DHTs — any malicious node can broadcast a fraudulent binding. An attacker could announce themselves as `admin@q8.fyi`. Without a hierarchical verification mechanism, identity resolution becomes a race to squat on high-value identifiers, completely eroding network trust. DNS-backed systems solve this by defining the domain owner as the absolute authority, but in doing so, they inherit the vulnerabilities of the centralized ICANN ecosystem.

### Enumeration and Data Scraping

If a network provides unauthenticated resolution queries or broadcasts bindings openly, it inadvertently functions as an automated directory for spam harvesters and surveillance entities. Option A turns every Zenoh announce into a public email broadcast, effectively gifting the entire network topology and user base to scrapers. Similarly, `.well-known` HTTP endpoints that lack strict rate-limiting allow attackers to iterate through dictionaries, dumping all `*@domain.tld` bindings. Matrix's MSC2134 attempts to thwart this by salting lookups with a server-rotated pepper, ensuring that attackers cannot pre-compute hash tables (rainbow tables). However, this assumes the pepper itself remains secret, which is inherently flawed given that the pepper must be served to any legitimate client upon request via `hash_details`.

### Denial of Service via Resolution Paths

Placing the resolution mapping on the critical path of message transmission introduces severe latency and availability risks. In Option C (HTTPS `.well-known` resolution) or Option B (Gateway-mediated Zenoh queries), every initial message sent to a new recipient requires a synchronous network round-trip. If the recipient's identity server or domain gateway is offline, misconfigured, or under a Distributed Denial of Service (DDoS) attack, the sender's gateway cannot resolve the identity hash. Consequently, the gateway cannot seal the message, resulting in delivery failure. This hard dependency violates the asynchronous, highly available ethos of peer-to-peer networks.

### Sybil Attacks and Reputation Undermining

In completely decentralized environments lacking a central registrar or a financial cost to identifier creation, attackers can generate millions of synthetic email-to-hash bindings. If a gateway relies on gossip or unauthenticated distributed hash tables without requiring proof-of-work, proof-of-stake, or strict domain-authority attestation, the network will be overwhelmed by Sybil nodes. Reputation systems require scarcity to function; the traditional DNS system inherently enforces this scarcity via domain registration and hosting costs.

### Equivocation (Split-View Attacks)

A compromised or malicious domain authority could selectively return different identity hashes to different querying gateways for the identical email address. For example, a compromised identity server might return Alice's true cryptographic key to Bob, but an attacker-controlled key to Charlie. This allows the attacker to silently intercept and decrypt Charlie's messages intended for Alice. Standard HTTPS lookups (Option C) provide absolutely no mechanism to detect this equivocation. Detecting split-view attacks requires advanced cryptographic data structures, such as Certificate Transparency logs or verifiable gossip-based consistency checks among clients.

### Offline-Recipient Race Conditions

In highly asynchronous networks, state synchronization delays can lead to dangerous race conditions. Consider a scenario where Alice's device is compromised, and she rapidly updates her identity hash. If her domain's resolution server temporarily serves a cached or stale hash, a sender's gateway might resolve the old identifier and seal a highly sensitive message using the compromised key. Protocol designs must define explicit Time-To-Live (TTL) parameters for bindings and ensure that key rotation events propagate through the network faster than the maximum expected message latency.

### Domain Takeover Attacks

Because systems like `did:web` and NIP-05 inextricably link cryptographic identity to domain ownership, a lapse in domain registration or a DNS hijacking event yields catastrophic results. The new domain controller instantly assumes control over all associated email identifiers. The attacker can simply overwrite the `.well-known` file on the web server, routing all future encrypted communications to their own identity hashes. Mitigating domain takeover requires integrating historical cryptographic chaining. For instance, the SCID mechanisms in `did:webvh` mandate that any new domain owner must also possess the previous cryptographic signing keys to validly append new bindings to the identity log, breaking the reliance on pure DNS control.

---

## 4. Private-Lookup Techniques: Cost and Practicality

### Private Information Retrieval (PIR) and ZipPIR

Private Information Retrieval (PIR) is a cryptographic protocol allowing a client to retrieve a record from an untrusted database server without the server learning which record was accessed. Modern single-server PIR protocols leverage Learning with Errors (LWE) and homomorphic encryption.

- **SimplePIR:** Represents the current state-of-the-art in computational throughput for single-server PIR, achieving up to 10 GB/s per core. The server performs fewer than one 32-bit multiplication and addition per database byte. However, SimplePIR requires the client to download and store a massive "hint" during an offline preprocessing phase. For a 1 GB database, the client must store a 121 MB hint. This massive client-side storage and bandwidth overhead renders SimplePIR completely unsuitable for lightweight mobile clients or federated gateways operating over constrained networks.
- **ZipPIR:** Specifically designed to solve the client-hint bottleneck of SimplePIR. ZipPIR achieves this by compressing LWE ciphertexts into significantly smaller Paillier ciphertexts during the offline phase. ZipPIR achieves over 2 GB/s of throughput while completely eliminating the need for client-side hint storage, reducing the 121 MB requirement to zero, and shrinking offline communication overhead to mere kilobytes. Furthermore, under certain computational assumptions, ZipPIR features a "hintless" offline phase where the server independently generates and updates hints during idle times without requiring any client interaction.

| PIR Protocol | Throughput (1 GB DB) | Client-Side Storage | Offline Communication | Server-Side Storage |
|---|---|---|---|---|
| SimplePIR | 10 GB/s / core | 121 MB | 121 MB | High |
| ZipPIR | > 2 GB/s | 0 MB | ~600 KB | < 200 KB per client |

While ZipPIR makes single-server PIR mathematically feasible for identity resolution, it necessitates encoding the entire network's identifier space into a matrix. Continuously re-computing homomorphic properties upon every identity update introduces immense operational complexity. Furthermore, the performance of these protocols is heavily reliant on hardware crypto-acceleration instructions, specifically AVX-512 VNNI, which optimizes multiply-accumulate operations. As noted in the Harmony project constraints (ZEB-47), assuming AVX-512 hardware acceleration maturity across a decentralized network of heterogeneous gateways is premature.

### Hardware Enclaves and Oblivious RAM (ORAM)

As an alternative to pure cryptography, hardware-based Trusted Execution Environments (TEEs), such as Intel SGX, provide secure computing enclaves. However, as demonstrated by Signal, TEEs alone cannot prevent the host OS from deducing queries by analyzing RAM access patterns. Path Oblivious RAM (Path ORAM) arranges memory into a specialized tree structure. When an item is requested, the enclave reads a random path from the root, remaps the accessed item to a new random path, and writes it back. This guarantees that accessing the same record twice produces completely different memory traces. While effective (requiring roughly 1,800 memory accesses compared to a linear scan), deploying SGX+ORAM infrastructure requires specialized hardware provisioning and rigid vendor lock-in, directly contradicting the hardware-agnostic ethos of decentralized P2P networks.

### Anonymous Credentials and Blind Signatures

Techniques like blind signatures allow a server to sign an identity claim without knowing the content of the claim, useful for anonymous authorization. However, these techniques solve the problem of proving authorization anonymously, not the problem of discovering an unknown hash from a known email address. Thus, they are misaligned with the primary resolution requirement.

---

## 5. Key-Transparency and Verifiable Binding Protocols

To systematically prevent equivocation (split-view attacks) and mitigate the fallout of domain compromise, Identity Providers (IdPs) must maintain public, cryptographically verifiable logs of all email → hash bindings. Key Transparency (KT) adapts the concepts of Certificate Transparency to end-user identities.

### Evolution of Key Transparency

- **CONIKS:** One of the earliest KT protocols, CONIKS utilizes a specialized Merkle tree to commit to the state of all user keys at regular epochs. To prevent the server from presenting different Merkle roots to different users, CONIKS relies on a synchronous gossip protocol among clients to compare Signed Tree Roots (STRs). Unfortunately, maintaining global gossip is unscalable and easily disrupted, making the protocol fragile in practice.
- **SEEMless and Parakeet:** Addressed the shortcomings of CONIKS by formalizing the Verifiable Key Directory (VKD). SEEMless provides mathematically rigorous security proofs, allowing clients to efficiently query their own key histories and verify non-equivocation without relying on global gossip. Parakeet further optimized the VKD data structures, achieving planetary-scale throughput by ensuring that storage costs remain independent of the number of epochs, scaling purely with the number of key updates.
- **OPTIKS:** Focused explicitly on scalability, achieving smaller storage overheads than SEEMless or Parakeet while supporting complex, real-world edge cases such as account decommissioning and multi-device mapping.
- **Zoom End-to-End KT:** Zoom acquired Keybase and implemented an enterprise-grade KT system to bind SSO identities to cryptographic device keys. Zoom utilizes "sigchains" to bind keys from a user's devices to their account, allowing clients to independently monitor their own identities and detect impersonation attempts without trusting the Zoom server.

### Implementation Complexity for Small Teams (The AKD Library)

Implementing a Verifiable Key Directory from scratch requires maintaining an append-only tree structure, handling epoch commitments, and generating complex inclusion and consistency proofs. For a small team building the Harmony gateway, manually implementing SEEMless or Parakeet is fraught with severe cryptographic pitfalls.

However, Meta has open-sourced an implementation of KT called the Auditable Key Directory (`akd`), which is currently utilized in production by WhatsApp. Evaluated by the NCC Group, `akd` is a highly performant Rust crate based on SEEMless and Parakeet. It provides a stateless API, requiring the consumer to implement the storage backend. `akd` uses Verifiable Random Functions (VRFs) to map user identifiers to random paths in the tree, preventing enumeration attacks. Leveraging an audited crate like `akd` reduces the cryptographic implementation risk to near zero. Nevertheless, operating a KT server imposes strict append-only storage requirements; because obsolete data cannot be trivially purged to maintain cryptographic proofs, storage costs scale perpetually.

---

## 6. Binding Authority Models

The ultimate arbiter of the email → hash mapping defines the network's foundational trust model.

### 1. DNS + TLS (The Classical Authority)

Utilized by Option C, NIP-05, and `did:web`. The domain's DNS records and the CA-issued TLS certificate serve as the absolute root of trust.

- **Advantage:** Zero implementation friction. It natively understands and enforces the `@domain` boundary.
- **Disadvantage:** High centralization. It is subject to state-level censorship, ICANN administrative takedowns, and BGP/DNS hijacking. It provides no cryptographic tamper resistance.

### 2. Blockchain Registrars

Utilized by ENS, Handshake, and `did:ion`. Identifiers are bound to keys via smart contracts or immutable ledger entries.

- **Advantage:** Cryptographically immutable, globally consistent, and completely decentralized.
- **Disadvantage:** High latency, requires native cryptocurrency tokens to pay for state changes (gas fees), and forces total privacy leakage through public on-chain metadata analysis.

### 3. Domain-Scoped Signing Keys (DKIM-Style for Identity)

In this hybrid approach, the authoritative domain owner publishes a long-lived, cryptographic public key in DNS (conceptually identical to a DKIM TXT record). Instead of the gateway making an HTTP request for every single identity lookup, the domain controller cryptographically signs an assertion (e.g., "I, `domain.tld`, attest that `local_part@domain.tld` is bound to IdentityHash X"). This signed claim is then propagated.

- **Advantage:** Decouples resolution from live HTTP servers. The signed claim can be cached indefinitely, gossiped, or distributed via pub/sub (Zenoh) securely, as any client can independently verify the domain's signature using the DNS root.
- **Disadvantage:** Requires custom verification logic and strict key rotation specifications to handle compromised domain keys.

### 4. Gossip and Reputation (No Authority)

Utilized by purely peer-to-peer systems like Secure Scuttlebutt. Users locally assign names to hashes and gossip these mappings.

- **Advantage:** Maximum resilience; impossible to censor centrally.
- **Disadvantage:** Does not scale to establish global consensus for canonical identifiers. It is highly susceptible to massive impersonation unless paired with a dense, manually curated Web of Trust.

### 5. Hybrid: DNS Bootstrap + In-Network Sigchain

Combines DNS for initial discovery with a verifiable history log (like `did:webvh` or Keybase sigchains) for ongoing security. DNS is only trusted for the first lookup; subsequent updates must be signed by the cryptographic key established in the initial log entry.

---

## 7. Explicit Recommendation for the Harmony Gateway

The architectural constraints for the Harmony PR are explicitly defined: hash-keyed Zenoh discovery already exists (`harmony/identity/{hash}/announce`), announces are currently public, no blockchain integration is permitted, PIR infrastructure is out of scope (ZEB-47), interoperability with standard SMTP/IMAP is desired, enumeration must be minimized, and requiring a gateway per domain is acceptable.

Given these constraints, the root design tension is balancing the desire for peer-to-peer privacy against the reality of SMTP's inherently centralized, domain-centric routing.

### Evaluation of Proposed Options

- **Option A (Email in Announce):** Rejected. Broadcasting plaintext emails in Zenoh announces converts the decentralized network into a globally accessible spam harvesting list, violating core privacy tenets.
- **Option B (Gateway-mediated Zenoh Query):** Rejected for initial implementation. It re-creates the SMTP availability problem; if the destination gateway is temporarily offline, message sealing fails synchronously. Furthermore, Zenoh traffic analysis could still map social graphs based on query frequency.
- **Option D (PIR/ZipPIR):** Rejected. Hardware acceleration (AVX-512 VNNI) and the immense cryptographic scaffolding required make this infeasible for an immediate PR implementation by a small team.

### Recommended Architecture: Phased Hybrid Implementation

**Phase 1 (Immediate PR Implementation): Option C (NIP-05 Style HTTPS)**

For the immediate codebase update, Harmony should implement Option C: a `.well-known/harmony-users` endpoint via HTTPS.

- **Protocol Flow:** Upon receiving an email for `alice@q8.fyi`, the Harmony gateway executes a GET request to `https://q8.fyi/.well-known/harmony-users?name=alice`. The web server returns a simple JSON object containing Alice's 16-byte identity hash.
- **Rationale:** This is the lowest-complexity, most interoperable path. It requires absolutely zero modifications to the existing Zenoh wire protocol or the cryptographic `AnnounceRecord`. It directly mirrors Nostr's NIP-05 and the W3C `did:web` standard, both of which are highly successful in production environments.
- **Security Mitigation:** To prevent the enumeration vulnerabilities inherent in Option A, the HTTPS endpoint must be strictly configured to only answer exact-match queries and apply aggressive IP-based rate limiting to prevent dictionary scraping.

**Phase 2 (Target Architecture): Domain-Scoped Signed Claims over Zenoh**

While Option C solves the immediate engineering bottleneck, its reliance on synchronous HTTP calls violates the resilient, asynchronous nature of peer-to-peer messaging. Therefore, Phase 2 should transition to Domain-Scoped Signed Identity Claims, bridging the gap between Option C and Option A without sacrificing privacy.

1. **Bootstrap Trust (DNS):** The domain owner publishes an Ed25519 "Identity Master Key" as a DNS TXT record, conceptually identical to the DKIM deployment process.
2. **Generate Claims:** The domain's gateway generates a cryptographic claim: `Sign(Domain_Master_Key, "alice@q8.fyi -> IdentityHash_X + Timestamp")`.
3. **Private Distribution (Zenoh):** Instead of broadcasting this claim in the public announce (which causes enumeration), the claim is distributed only to nodes that explicitly request it via a targeted Zenoh query, or it is embedded as an encrypted preamble in the payload of the first message sent from Alice to a new contact.
4. **Asynchronous Verification:** When a remote gateway receives this signed claim, it queries the DNS TXT record for `q8.fyi`, verifies the signature, and permanently caches the `alice@q8.fyi -> IdentityHash_X` mapping.

**Rationale:** This establishes the "network-wide name registry" hinted at in the design docs without exposing an enumeration vector. Because the claim is signed by the domain key, it cannot be forged (preventing impersonation). Because it is verified and cached asynchronously, it eliminates the strict uptime dependency of Option B or Option C.

---

## 8. Open Research Questions

To ensure the Harmony gateway architecture remains robust as the network scales, several open research vectors must be continuously monitored to avoid naively re-implementing broken patterns:

1. **Equivocation Detection without Heavy Gossip:** If Harmony adopts Domain-Scoped Signed Claims (Phase 2), how can a client detect if a compromised domain owner is signing different identity hashes for the same email address without relying on fragile global gossip protocols (like CONIKS) or forcing the domain to implement a full, storage-heavy Key Transparency server (like AKD)? Exploring lightweight, probabilistic witness sampling — akin to `did:webvh` watcher nodes — is a critical area for future specification.
2. **Privacy-Preserving Key Revocation:** When an identity hash is compromised, how is the revocation signal propagated through the Zenoh network without leaking the identifier mapping to the broader network? Standard Certificate Revocation Lists (CRLs) suffer from severe privacy leaks, and bloom-filter-based revocation lists often suffer from unacceptable false-positive rates.
3. **Hardware-Agnostic PIR at Edge Scale:** As ZipPIR matures and eliminates the client-hint storage bottleneck, identifying the exact CPU (AVX-512) threshold required to run homomorphic matrix multiplication on a decentralized, low-power edge gateway will determine when Option D transitions from theoretical to highly practical. Continuous monitoring of Rust-based ZipPIR implementations is recommended.

---

## Citations (from original Gemini report)

Full citation list preserved in the original Gemini-generated report. Key references by specification/identifier:

- **Nostr NIP-05:** https://github.com/nostr-protocol/nips/blob/master/05.md
- **Matrix MSC2134:** https://github.com/matrix-org/matrix-spec-proposals/blob/main/proposals/2134-identity-hash-lookup.md
- **Signal sealed sender / CDS:** https://signal.org/blog/sealed-sender/ , https://signal.org/blog/private-contact-discovery/
- **Signal Path ORAM:** https://signal.org/blog/building-faster-oram/
- **ActivityPub / WebFinger:** https://www.w3.org/community/reports/socialcg/CG-FINAL-apwf-20240608/
- **W3C DID Resolution v0.3:** https://www.w3.org/TR/did-resolution/
- **did:webvh:** https://identity.foundation/didwebvh/
- **did:webplus:** https://ledgerdomain.github.io/did-webplus-spec/
- **OMEMO (XMPP):** https://xmpp.org/extensions/xep-0384.html
- **DKIM (RFC 6376):** https://datatracker.ietf.org/doc/html/rfc6376
- **DANE / MTA-STS overview:** https://learn.microsoft.com/en-us/purview/how-smtp-dane-works
- **SimplePIR:** https://github.com/ahenzinger/simplepir , https://people.eecs.berkeley.edu/~henrycg/files/academic/pres/simplepir-slides.pdf
- **ZipPIR:** https://arxiv.org/abs/2603.09190 , https://arxiv.org/pdf/2603.09190
- **CONIKS:** https://www.usenix.org/system/files/conference/usenixsecurity15/sec15-paper-melara.pdf
- **SEEMless / Parakeet:** https://www.ndss-symposium.org/wp-content/uploads/2023-545-paper.pdf
- **OPTIKS:** https://www.usenix.org/system/files/sec24summer-prepub-814-len.pdf
- **Zoom E2E KT:** https://css.csail.mit.edu/6.858/2023/readings/zoom_e2e_v4.pdf
- **Meta AKD:** https://github.com/facebook/akd , https://docs.rs/akd , NCC Group review: https://www.nccgroup.com/media/phzpm0qv/_ncc_group_metaplatforms_e008327_report_2023-11-14_v10.pdf
