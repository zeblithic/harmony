# Ring 2 Microkernel Design

**Status:** Approved design
**Date:** 2026-03-06
**Scope:** harmony-os (crates/harmony-microkernel)

## Goal

Add process isolation, 9P-inspired IPC, and capability enforcement to the
Ring 1 unikernel. Ring 2 layers on top of Ring 1 — the existing unikernel
runtime becomes "process 0" (the kernel's built-in Harmony node). Other
processes are trait objects that serve files through a shared IPC mechanism,
with every cross-process call gated by Ring 0's UCAN capability tokens.

## Milestones

### Milestone A — Two Processes Communicating

Two userspace processes talking over IPC: process 0 (the Harmony node
runtime) writes to process 1 (an echo server), reads back the result,
and logs everything to serial. Proves the IPC trait, namespace
resolution, and capability enforcement work end-to-end.

### Milestone B — Reticulum Router as 9P Server (future)

The Harmony Reticulum router running as an isolated 9P server process,
with the kernel mediating all network I/O through capabilities. Proves
the full "everything is a 9P server" architecture.

## Key Design Decisions

1. **9P-inspired, not 9P-compatible** — Rust traits internally (`walk`,
   `read`, `write`, etc.) with the same semantics as 9P2000 but a
   custom encoding optimized for in-kernel dispatch. Wire-compatible
   9P2000 can be added as a translation layer later for external tooling.
   Rationale: interop matters at the network boundary (Reticulum), not
   inside the kernel.

2. **Cooperative isolation first** — Processes are `Box<dyn FileServer>`
   trait objects polled round-robin in a single address space. Isolation
   is enforced by Rust's type system and UCAN capability checks, not
   hardware page tables. Hardware paging (x86_64 page tables, separate
   address spaces) is a follow-up.

3. **Ring 2 layers on Ring 1** — The microkernel imports
   `harmony-unikernel` and wraps the existing runtime as process 0. Ring
   1's hardware drivers (VirtIO, PIT, serial) stay in-kernel. Extracting
   them into userspace 9P servers is milestone B.

4. **Trait objects for processes** — Each process implements
   `trait FileServer`. The kernel calls into processes synchronously
   during IPC dispatch. Simpler than async tasks, matches the sans-I/O
   pattern, and is trivially testable on the host. Async can be added
   later when real concurrency is needed.

5. **Sans-I/O all the way** — The entire `Kernel` struct is generic over
   `EntropySource` and `PersistentState`, just like `UnikernelRuntime`.
   All IPC, namespace resolution, and capability enforcement is testable
   on the host with `cargo test`. No QEMU needed for correctness testing.

## Architecture

```
+-------------------------------------------------------+
|                    "Processes"                         |
|  +-----------+  +-----------+  +-------------------+  |
|  | Process 1 |  | Process 2 |  | Process 0         |  |
|  | (echo     |  | (future:  |  | (UnikernelRuntime |  |
|  |  server)  |  |  content) |  |  = Harmony node)  |  |
|  +-----+-----+  +-----+-----+  +---------+---------+  |
|        |              |                  |             |
|  All implement trait FileServer          |             |
+--------+--------------+---------+--------+-------------+
|              Microkernel                               |
|  +------------+  +-------------+  +-----------------+  |
|  | Namespace  |  | IPC         |  | Capability      |  |
|  | Resolution |  | Dispatch    |  | Enforcement     |  |
|  +------------+  +-------------+  +-----------------+  |
+--------------------------------------------------------+
|         Ring 1 (unchanged)                             |
|  VirtIO-net, PIT timer, serial, heap, event loop       |
+--------------------------------------------------------+
```

## Core Types

### FileServer Trait

The heart of Ring 2. Every process implements this:

```rust
pub type Fid = u32;
pub type QPath = u64;

pub trait FileServer {
    fn walk(&mut self, fid: Fid, new_fid: Fid, name: &str) -> Result<QPath, IpcError>;
    fn open(&mut self, fid: Fid, mode: OpenMode) -> Result<(), IpcError>;
    fn read(&mut self, fid: Fid, offset: u64, count: u32) -> Result<Vec<u8>, IpcError>;
    fn write(&mut self, fid: Fid, offset: u64, data: &[u8]) -> Result<u32, IpcError>;
    fn clunk(&mut self, fid: Fid) -> Result<(), IpcError>;
    fn stat(&mut self, fid: Fid) -> Result<FileStat, IpcError>;
}
```

### Supporting Types

```rust
pub enum OpenMode { Read, Write, ReadWrite }

pub struct FileStat {
    pub qpath: QPath,
    pub name: Arc<str>,
    pub size: u64,
    pub file_type: FileType,
}

pub enum FileType { Regular, Directory }

pub enum IpcError {
    NotFound,
    PermissionDenied,
    NotOpen,
    InvalidFid,
    NotDirectory,
    ReadOnly,
}
```

### Process

```rust
pub struct Process {
    pub pid: u32,
    pub name: Arc<str>,
    pub namespace: Namespace,
    pub capabilities: Vec<UcanToken>,
    pub server: Box<dyn FileServer>,
}
```

Each process is both a FileServer (serves files to others) and a client
(accesses other servers through its namespace).

## Namespace Model

Each process has a private namespace — a mount table that determines
what it can see. From Plan 9: your view of the system is entirely
defined by what's mounted into your namespace.

```rust
pub struct Namespace {
    mounts: BTreeMap<Arc<str>, MountPoint>,
}

pub struct MountPoint {
    pub target_pid: u32,
    pub root_fid: Fid,
}
```

### Path Resolution

A process with this namespace:

```
/dev/serial    -> pid 0 (kernel serial server)
/harmony/node  -> pid 0 (unikernel runtime)
/echo          -> pid 1 (echo server)
```

When process 2 reads `/echo/hello`, the kernel:

1. Finds longest matching mount prefix: `/echo` -> pid 1
2. Strips the prefix, walks the remainder: `hello`
3. Checks process 2 holds an `EndpointCap` for pid 1
4. Dispatches `walk` + `open` + `read` to pid 1's `FileServer`
5. Returns the result to process 2

### Kernel-Provided Servers

The kernel itself acts as process 0 and serves built-in namespaces:

- `/dev/serial` — wraps Ring 1's SerialWriter
- `/dev/net/virtio0` — wraps Ring 1's VirtIO driver
- `/harmony/node` — the UnikernelRuntime

Mounted into each process's namespace at creation time, gated by
capabilities. A process without the right capability simply won't have
the mount point in its namespace.

### Deferred

- Union mounts (multiple servers overlaid on one path)
- Bind mounts (aliasing paths within a namespace)
- Remote mounts over Reticulum (Phase 5 mesh federation)

## Capability Enforcement

Every IPC dispatch goes through a capability check using Ring 0's
existing UCAN token verification.

### IPC Dispatch Flow

```
Process 2 wants to read /echo/hello
    |
    v
1. Kernel resolves namespace: /echo -> pid 1
    |
    v
2. Kernel searches process 2's capabilities for an EndpointCap
   where: audience = process 2's address_hash,
          capability = Endpoint, resource encodes "pid:1"
    |
    v
3. verify_token(cap, now, proofs, identities, revocations, max_depth)
   (Ring 0's existing sans-I/O UCAN verification)
    |
    v
4. Valid: dispatch walk/read to pid 1's FileServer
   Invalid: return IpcError::PermissionDenied
```

### Capability Granting

When the kernel spawns a process, the parent delegates capabilities:

```rust
let cap = kernel_identity.delegate(
    &mut entropy,
    &parent_cap,
    &process2_address,
    CapabilityType::Endpoint,
    b"pid:1",             // resource: access to pid 1
    now, 0,               // no expiry
)?;
process2.capabilities.push(cap);
```

### Resource Encoding (Milestone A)

| Capability | Resource encoding |
|---|---|
| `Endpoint` | `pid:<n>` or `*` (all servers) |
| `Memory` | deferred (no hardware isolation yet) |
| `IOPort` | deferred |
| `Interrupt` | deferred |

Resource-level path attenuation (restricting access to specific files
within a server) is future work. For milestone A, holding an EndpointCap
for a pid grants full access to that server.

### What Ring 0 Provides (no new code needed)

- Delegation chains (parent -> child attenuation)
- Revocation (revoking parent revokes all children)
- Time-bounded capabilities
- Ed25519 signature verification
- Sans-I/O verification traits (ProofResolver, IdentityResolver, RevocationSet)

## Kernel Structure

```rust
pub struct Kernel<E: EntropySource, P: PersistentState> {
    runtime: UnikernelRuntime<E, P>,
    processes: BTreeMap<u32, Process>,
    next_pid: u32,
    identity: PrivateIdentity,
    proof_store: MemoryProofStore,
    identity_store: MemoryIdentityStore,
    revocations: MemoryRevocationSet,
    ipc_queue: VecDeque<IpcMessage>,
}

pub struct IpcMessage {
    pub from_pid: u32,
    pub to_pid: u32,
    pub operation: IpcOperation,
}

pub enum IpcOperation {
    Walk { fid: Fid, new_fid: Fid, name: Arc<str> },
    Open { fid: Fid, mode: OpenMode },
    Read { fid: Fid, offset: u64, count: u32 },
    Write { fid: Fid, offset: u64, data: Vec<u8> },
    Clunk { fid: Fid },
    Stat { fid: Fid },
}
```

### Event Loop Integration

The main loop extends Ring 1's spin loop:

```rust
loop {
    let now = pit.now_ms();

    // 1. Poll network (unchanged from Ring 1)
    poll_virtio_net(&mut virtio_net, &mut kernel.runtime, now);

    // 2. Ring 1 tick (announces, heartbeats, peer timeout)
    let actions = kernel.runtime.tick(now);
    dispatch_runtime_actions(&actions, &mut virtio_net, &mut serial);

    // 3. Drain IPC queue — dispatch pending messages
    kernel.dispatch_ipc(now);

    // 4. Poll each process (cooperative scheduling)
    kernel.poll_processes(now);

    core::hint::spin_loop();
}
```

Step 3: kernel pops messages, verifies capabilities, calls target's
FileServer methods, queues responses.

Step 4: each process gets a chance to do work (generate outbound IPC).

## Milestone A Deliverables

### What Gets Built

1. **`harmony-microkernel` crate** — Kernel struct, FileServer trait,
   Namespace, Process, IpcMessage, IpcOperation, IpcError, capability
   verification in the dispatch path.

2. **`EchoServer`** — Trivial FileServer for testing. Exposes `hello`
   (returns a greeting) and `echo` (write bytes in, read them back).
   ~50 lines, proves the full IPC path.

3. **`KernelSerialServer`** — Wraps Ring 1's SerialWriter as a
   FileServer. Write to it, bytes go to COM1. Makes serial output
   accessible through the namespace as `/dev/serial`.

4. **Boot integration** — `harmony-boot/src/main.rs` gets a
   `--features ring2` flag. Without it: boots Ring 1 unikernel.
   With it: boots Ring 2 microkernel, spawns processes, runs IPC demo.

### Expected Serial Output

```
[BOOT]  Harmony microkernel v0.1.0
[HEAP]  4194304
[PIT]   timer initialized
[KERN]  identity a3b7...
[PROC]  pid=0 name=harmony-node
[PROC]  pid=1 name=echo-server
[CAP]   pid=0 granted Endpoint access to pid=1
[IPC]   pid=0 -> pid=1 walk "hello"
[IPC]   pid=0 -> pid=1 open fid=1 Read
[IPC]   pid=0 -> pid=1 read fid=1 offset=0 count=256
[IPC]   result: "Hello from process 1!"
[READY] entering event loop
```

### What's NOT in Milestone A

- No hardware page tables (cooperative isolation only)
- No dynamic process spawning (processes hardcoded at boot)
- No resource-level path attenuation in capabilities
- No Athenaeum / content-addressed storage (separate bead)
- No Linuxulator (Ring 3)
- No remote 9P over Reticulum (Phase 5)

## Testing Strategy

### Unit Tests (host-side, `cargo test -p harmony-microkernel`)

All sans-I/O, all on the host. No QEMU needed.

- **FileServer dispatch** — create EchoServer, call walk/open/read/write,
  verify responses
- **Namespace resolution** — mount servers at paths, resolve paths to
  correct pid + remainder
- **Capability enforcement** — valid cap dispatches; missing/expired/revoked
  cap returns PermissionDenied
- **IPC queue** — enqueue messages, dispatch, verify responses reach sender
- **Capability delegation** — parent delegates to child, child can access;
  revoke parent, child loses access
- **Full integration** — create Kernel with two processes, send IPC
  end-to-end, verify correct behavior

### QEMU Smoke Test

Boot with `ring2` feature, verify serial output shows IPC exchange
completing. Same pattern as Ring 1's `just test-qemu`.

## Future Work

- **Hardware page tables** — x86_64 paging, separate address spaces per
  process. The FileServer trait interface doesn't change — only the
  enforcement mechanism underneath.
- **Async processes** — replace synchronous trait objects with async
  futures driven by a minimal executor. Enables I/O-heavy servers.
- **Milestone B** — Reticulum router as isolated 9P server process.
- **Athenaeum integration** — content-addressed chunk store as a
  FileServer at `/harmony/content/`.
- **9P2000 wire protocol** — translation layer for external debugging
  tools (9pfuse, etc.).
- **Union mounts and bind mounts** — richer namespace composition.
- **Remote mounts over Reticulum** — Phase 5 mesh federation.
- **Linuxulator** — Ring 3, translates Linux syscalls to FileServer calls.
