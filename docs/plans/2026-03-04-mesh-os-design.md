# Harmony Mesh Operating System — Architectural Vision

**Date:** 2026-03-04
**Status:** Approved

## Problem

The Harmony protocol stack (Reticulum routing, Zenoh pub/sub, CID-addressed content, WASM compute, durable workflows) currently runs as a library atop a host operating system. But the stack's sans-I/O architecture — pure state machines with no runtime dependencies — makes it uniquely suited to serve as the native substrate of an operating system purpose-built for decentralized mesh computing.

Traditional operating systems bolt distributed capabilities onto a monolithic kernel designed for a single machine. The Harmony mesh OS inverts this: the distributed mesh protocol is the foundation, and local hardware management is a service layered on top.

## Architecture: Concentric Rings

The OS is structured as concentric rings around a shared protocol core. Each ring adds capabilities for a specific deployment tier. The innermost ring is a `no_std` Rust library; the outermost is a full operating system with Linux ABI compatibility.

```
Ring 0: harmony-core (no_std) — crypto, identity, packets, CIDs, state machines
Ring 1: harmony-unikernel   — Ring 0 + bare-metal driver + event loop = bootable single-purpose node
Ring 2: harmony-microkernel  — Ring 1 + IPC (9P), capability enforcement, process isolation
Ring 3: harmony-os           — Ring 2 + Linuxulator, DDE, declarative config, hot-swap
```

Each ring is independently useful. The unikernel runs SUNDRAGON appliances. The microkernel runs edge nodes. The full OS runs infrastructure nodes. All share the same protocol stack. The sans-I/O pattern makes this possible — Ring 0 works identically at every tier because it has zero runtime assumptions.

---

## Ring 0: Harmony Protocol Core (no_std)

The shared substrate that every deployment tier builds on.

### Subsystems

| Subsystem | Crate | Role |
|-----------|-------|------|
| Cryptographic primitives | `harmony-crypto` | SHA-256, BLAKE3, HKDF, Fernet, ChaCha20-Poly1305 |
| Self-sovereign identity | `harmony-identity` | Ed25519/X25519 keypairs, address derivation, sign/verify, ECDH |
| Mesh routing | `harmony-reticulum` | Packet format, path table, announces, links, node state machine |
| Pub/sub namespace | `harmony-zenoh` | Key expressions, sessions, queryables, liveliness |
| Content addressing | `harmony-content` | CIDs, blob store, W-TinyLFU cache, bundles, DAGs, delta-sync |
| Deterministic compute | `harmony-compute` | WASM runtime, fuel budgeting, checkpoint/restore |
| Durable workflows | `harmony-workflow` | Event-sourced execution, replay recovery |
| Priority scheduler | `harmony-node` | Three-tier event loop, adaptive fuel, starvation prevention |

### Design Constraints

1. **no_std compatible** — All crates compile without the standard library (with `alloc` for heap). `std` feature flags enable richer runtimes where available. This gate enables bare-metal unikernels.

2. **Zero runtime assumptions** — No threads, no system calls, no timers, no I/O. The caller provides everything: RNG via trait, timestamps as arguments, raw bytes in / action enums out.

3. **Platform-abstract traits** — Where Ring 0 needs platform services, it defines traits:
   - `BlobStore` — content storage backend (exists)
   - `NetworkInterface` — send/receive raw bytes (new)
   - `PersistentState` — save/load node state across restarts (new)
   - `EntropySource` — cryptographic randomness (new)

4. **Identity as trust root** — `PrivateIdentity` (Ed25519/X25519 keypair) is the fundamental credential. All higher-ring capabilities, binary signatures, and resource authorization chain back to this key. Address = `SHA256(X25519_pub || Ed25519_pub)[:16]`.

---

## Ring 1: Harmony Unikernel

A unikernel compiles the Harmony protocol stack, a network driver, and an event loop into a single binary that boots directly on bare metal or a hypervisor. No OS underneath.

### Architecture

```
+-------------------------------------+
|         Harmony Node Runtime        |  <- Ring 0 (sans-I/O state machines)
+-------------------------------------+
|        Unikernel Runtime Layer      |
|  +----------+----------+----------+ |
|  | Event    | Platform | Network  | |
|  | Loop     | Traits   | Drivers  | |
|  | (async)  | (impl)   | (virtio/ | |
|  |          |          |  raw UDP) | |
|  +----------+----------+----------+ |
+-------------------------------------+
|     Minimal Hardware Abstraction    |
|  +----------+----------+----------+ |
|  | Memory   | Interrupt| Timer    | |
|  | (bump/   | Handler  | (PIT/   | |
|  |  slab)   |          |  HPET)  | |
|  +----------+----------+----------+ |
+-------------------------------------+
|     Boot (multiboot2 / UEFI)        |
+-------------------------------------+
```

### Design Decisions

1. **Single address space, no processes** — One application, no isolation, no syscall boundary. Ring 0 state machines called directly as Rust function calls. Zero context-switch overhead.

2. **Two target platforms:**
   - **Bare-metal x86_64/aarch64** — Boots via multiboot2 or UEFI. For dedicated SUNDRAGON appliances.
   - **VirtIO on hypervisor** — Runs under QEMU/KVM or Firecracker. VirtIO-net for networking. For cloud edge or development.

3. **Network drivers:**
   - VirtIO-net for hypervisor deployments (~2000 lines of Rust)
   - Raw UDP over minimal IP stack (`smoltcp`) for mesh-over-internet
   - Raw serial/LoRa for RF mesh links (SUNDRAGON off-grid)

4. **Async executor** — Minimal single-threaded executor drives the event loop:
   ```
   loop {
       poll network interfaces for raw bytes
       feed events into NodeRuntime::push_event()
       let actions = runtime.tick()
       execute actions (send bytes, fetch content, etc.)
       yield to timer / wait for interrupt
   }
   ```

5. **No filesystem** — Content stored in-memory `BlobStore`. Persistent state (identity, path table) optionally on raw block/flash. Declarative config compiled into the binary at build time.

6. **Identity** — Either baked into the image at build time (dedicated appliances) or generated on first boot and persisted to flash.

### Capabilities and Limitations

**Can do:** Route packets, announce identity, discover peers, store/serve CID-addressed content, execute WASM workflows, participate in Zenoh pub/sub.

**Cannot do:** Run arbitrary applications, isolate tenants, hot-swap components, provide shell or package manager.

---

## Ring 2: Harmony Microkernel

Adds the three capabilities a unikernel lacks: **process isolation**, **inter-process communication**, and **capability-based authorization**.

### Three Kernel Responsibilities

The microkernel does exactly three things in privileged mode. Everything else runs in unprivileged userspace.

#### 1. Memory Management

- **Per-process virtual address spaces** — No shared memory unless explicitly mediated by capability exchange.
- **Memory capabilities (MemoryCap)** — Grant read, write, or execute permission on specific virtual ranges. Delegatable but unforgeable.
- **Allocators** — Slab for kernel objects, buddy for physical pages. Simple and auditable for formal verification readiness.

#### 2. Scheduling and IPC

- **9P as native IPC** — The key architectural decision. Every kernel object (memory region, interrupt line, device register) is exposed as a file in a 9P namespace. Processes interact via reading/writing 9P file descriptors. Not 9P-over-TCP — the kernel dispatches 9P messages directly in the syscall path. Target: ~100 cycles for a small message.

- **Per-process namespaces** — From Plan 9. Each process has a private namespace tree of 9P mount points. A process's view of the system is entirely determined by what's mounted into its namespace.

- **Thread scheduling** — Priority-based preemptive, with real-time bands for interrupt handlers. Per-core schedulers (multikernel style) with cross-core communication via explicit messages, never shared-memory locks.

#### 3. Capability Enforcement

No ambient authority. Every syscall takes a capability argument.

| Capability | Grants |
|-----------|--------|
| `MemoryCap` | Access to a memory region |
| `EndpointCap` | Permission to send 9P messages to a server |
| `InterruptCap` | Permission to receive a hardware interrupt |
| `IOPortCap` | Access to a hardware I/O port range |
| `IdentityCap` | Right to sign with a specific Harmony identity |
| `ContentCap` | Right to read/write a CID range in the blob store |
| `ComputeCap` | Right to execute WASM with a fuel budget |

- **Delegation** — A parent can grant capability subsets to children. The kernel tracks the delegation tree. Revoking a capability revokes all downstream delegations.
- **Harmony identity as root capability** — All capabilities derive from the `PrivateIdentity` via UCAN-style delegation chains signed by Ed25519. The kernel verifies these signatures.

### Ring 0 Integration

Ring 0's protocol stack runs as **userspace 9P server processes**, not in the kernel:

```
+-----------------------------------------------------+
|                    Userspace                         |
|  +---------+  +----------+  +----------+  +------+  |
|  |Reticulum|  | Storage  |  | Workflow |  | Apps |  |
|  | Router  |  |  Tier    |  | Engine   |  |      |  |
|  |(9P svr) |  |(9P svr)  |  |(9P svr)  |  |(9P)  |  |
|  +----+----+  +----+-----+  +----+-----+  +--+---+  |
|       | 9P         | 9P         | 9P        | 9P    |
+-------+------------+------------+-----------+--------+
|                   Microkernel                        |
|  +----------+-----------+----------------------+     |
|  | Memory   | Scheduler | Capability Enforcer  |     |
|  | Manager  | + 9P IPC  | (UCAN verification)  |     |
|  +----------+-----------+----------------------+     |
+------------------------------------------------------+
|                   Hardware                           |
+------------------------------------------------------+
```

If the storage tier crashes, the kernel restarts it. The router keeps routing. Microkernel resilience.

### 9P Namespace (Typical Node)

```
/
+-- dev/
|   +-- net/
|   |   +-- eth0              (network driver)
|   |   +-- lora0             (LoRa radio)
|   +-- block/
|   |   +-- nvme0             (block device)
|   +-- entropy               (hardware RNG)
+-- harmony/
|   +-- identity              (read: public bytes; write: sign request)
|   +-- router/
|   |   +-- ctl               (add interface, read stats)
|   |   +-- path_table        (known routes)
|   |   +-- announces/        (received announces)
|   +-- content/
|   |   +-- ctl               (cache stats, budget)
|   |   +-- {cid_hex}         (read: fetch by CID; write: store)
|   +-- compute/
|   |   +-- ctl               (fuel budget, active workflows)
|   |   +-- submit            (write: module_cid + input)
|   +-- mesh/
|       +-- peers/            (one file per known peer)
|       +-- pubsub/
|           +-- {key_expr}    (read: subscribe; write: publish)
+-- proc/{pid}/               (per-process info)
|   +-- status
|   +-- ns                    (namespace view)
|   +-- caps                  (capability set)
+-- env/
    +-- config                (node config, derivation hash)
    +-- modules/              (available WASM modules by CID)
```

Applications interact with the full Harmony stack by reading and writing files. No SDK, no library, no API versioning.

---

## Ring 3: Harmony OS (Full Operating System)

Adds four capabilities: **Linux compatibility**, **driver ecosystem**, **declarative configuration**, and **live component hot-swap**.

### 1. Linux ABI Compatibility (Linuxulator)

Translates Linux syscalls into 9P operations at the kernel boundary.

```
Linux binary:    open("/etc/hosts", O_RDONLY)
       |
Linuxulator:     9P Twalk + Topen on /compat/linux/etc/hosts
       |
9P server:       Serves from content-addressed blob (CID-backed)
       |
Linux binary:    Gets valid fd, unaware of non-Linux kernel
```

- **Syscall translation** — ~400 Linux syscalls mapped to 9P + capabilities. Day-one focus: ~80 syscalls covering 99% of applications (open, read, write, mmap, ioctl, socket, epoll, futex, clone).
- **Capability wrapping** — Every translated syscall is capability-checked. The binary sees POSIX errors; enforcement is capability-based underneath.
- **Linux personality** — Activated per-process at exec(). Provides /proc/self/, /sys/, /dev/shm/ as 9P mounts. Non-Linux processes don't see these.
- **ELF loader** — Dynamic linking against bundled musl libc. Libraries resolved from the content-addressed package store.
- **Container support** — OCI containers run natively. Each container gets its own namespace. Image layers are CID-addressed blobs with automatic deduplication.

### 2. Device Driver Environment (DDE)

Reuses Linux drivers in isolated userspace processes.

```
+------------------------------------+
|  Linux Driver (e.g., iwlwifi)      |  <- Unmodified C code
+------------------------------------+
|         DDE Shim Layer             |  <- Translates Linux kernel API
|  (kmalloc, pci_, irq_, skb_ shims) |     into 9P requests
+------------------------------------+
|       9P client (to kernel)        |
+------------------------------------+
```

- Fake "Linux kernel API" translates to 9P messages to the microkernel.
- Driver runs unprivileged with minimal capabilities (specific PCI BARs, specific IRQs).
- Driver crash cannot propagate — kernel restarts it.
- Priority: WiFi (iwlwifi, ath9k), USB, NVMe, GPU (basic framebuffer). LoRa drivers written natively in Rust.

### 3. Declarative Configuration

Every node's configuration is a **Harmony content bundle** — a DAG of CIDs:

```
NodeConfig (Bundle CID)
+-- kernel: CID
+-- drivers: [CID]
+-- services: [CID]
+-- identity: CID (encrypted PrivateIdentity)
+-- schedule: TierSchedule config
+-- network:
|   +-- interfaces: [{name, driver_cid, config}]
|   +-- mesh_seeds: [address_hash]
+-- storage:
|   +-- budget: StorageBudget
|   +-- pins: [CID]
+-- compute:
    +-- budget: InstructionBudget
    +-- allowed_modules: [CID]
```

- **Atomic upgrades** — New config = new bundle CID. Activate by pointer swap. Rollback by repointing.
- **Mesh-wide consistency** — Verify 1000 nodes match by comparing one 32-byte CID.
- **Cryptographic provenance** — Config signed by operator's Harmony identity. Kernel rejects unsigned configs.
- **Reproducibility** — Same CID = same running system, anywhere.
- **Distribution** — Configs propagate over the Harmony content layer via Zenoh pub/sub.

### 4. Live Component Hot-Swap

Adapted from K42, simplified by 9P architecture. Because every service is a 9P server, hot-swap is mount-point redirection:

1. New binary arrives (signed, CID-verified)
2. Kernel starts new process with new binary
3. New process signals ready
4. Kernel queues new 9P requests, lets outstanding requests complete (quiescence)
5. State transfer: Tread /state from old process, Twrite /state to new process
6. Kernel atomically rebinds 9P mount point to new process
7. Queued requests released to new process; old process terminated

**Hot-swappable:** Network drivers, Reticulum router, storage tier, workflow engine, Linuxulator.
**Not hot-swappable:** The microkernel itself (requires reboot, but it's tiny and changes rarely).

### Cryptographic Binary Validation

Stronger than IMA/EVM because validation is structural:

1. Every executable is a CID — the content hash IS its identity.
2. The signed config bundle lists authorized CIDs.
3. The kernel verifies BLAKE3 hash at exec() against the config's authorized list.
4. Content-addressed storage means verify-once, trust-the-hash-forever.

No unauthorized binary can load — the loader cannot find it if it's not in the content store with a valid CID.

---

## Cross-Cutting Concerns

### Mesh Resource Federation

9P extends transparently across the mesh via Reticulum links:

- **Discovery** — Zenoh liveliness tokens announce exported 9P services.
- **Binding** — Processes bind remote 9P namespaces into local tree.
- **Transport** — 9P messages serialized into Reticulum packets with link encryption.
- **Failover** — On link drop, kernel re-binds to alternate peer. Durable workflows replay from event log on new node.
- **Capability gating** — Remote binds require UCAN exchange. No unauthorized cross-mesh resource access.

### Process Migration

Simpler than MOSIX because of architectural properties:

- **Memory** — Serialized via content store as CID-addressed blobs.
- **File descriptors** — All fids are 9P; re-bind to remote mounts on target.
- **Libraries** — Declarative config guarantees identical environment (same CIDs).
- **Capabilities** — Serialized as UCAN delegation chains, verified by target kernel.
- **Workflows** — Checkpoint on source, ship history, resume from checkpoint on target. Deterministic replay guarantees identical behavior.

### Security Model

```
Layer              Mechanism                    Enforces
----------------------------------------------------------------------
Boot               Signed config bundle          Only authorized configs boot
Binary loading     CID verification              Only signed binaries execute
Kernel IPC         Capability-gated 9P           No ambient authority
Process isolation  Separate address spaces        Memory safety between components
Driver sandboxing  Userspace + limited caps       Driver crash != system crash
Mesh transport     Reticulum link encryption      Confidentiality between nodes
Identity           Ed25519/X25519                 Self-sovereign, no central authority
Authorization      UCAN delegation chains         Verifiable, revocable, delegatable
Content integrity  CID hash verification          Tamper-evident storage
Compute            WASM sandbox + fuel budget     Untrusted code with resource limits
```

Zero-trust: every binary verified, every IPC capability-checked, every connection authenticated, every blob hash-verified, every WASM module sandboxed.

---

## Phased Roadmap

```
Phase 1: Ring 0 hardening
  - Audit no_std compatibility across crates
  - Define platform-abstract traits (NetworkInterface, PersistentState, EntropySource)
  - Implement UCAN capability tokens in harmony-identity
  - Target: Protocol stack compiles for bare-metal targets

Phase 2: Ring 1 unikernel prototype
  - Minimal bootable image (x86_64, multiboot2)
  - VirtIO-net driver for QEMU
  - Harmony node boots, announces, routes, serves content
  - Target: Two unikernel nodes in QEMU exchange messages

Phase 3: Ring 2 microkernel foundation
  - Memory manager with capability-gated regions
  - 9P IPC in the syscall path
  - Per-process namespaces
  - Process lifecycle (spawn + capability delegation)
  - Ring 0 services as isolated 9P servers
  - Target: Multi-process Harmony node with crash isolation

Phase 4: Ring 3 full OS
  - Linuxulator (core ~80 syscalls)
  - DDE for WiFi and NVMe drivers
  - Declarative config via content-addressed bundles
  - Hot-swap mechanism for 9P services
  - Target: Linux terminal emulator and curl on HarmonyOS

Phase 5: Mesh federation
  - Remote 9P over Reticulum links
  - Peer discovery via Zenoh liveliness
  - Process migration prototype
  - Fleet management via config bundle distribution
  - Target: Multi-node mesh as single coherent system
```
