# Ring 1: VirtIO-net Driver for QEMU — Design

**Date:** 2026-03-06
**Status:** Approved
**Bead:** harmony-bpo

## Goal

Give the Harmony unikernel network I/O in QEMU. After this, the boot stub can send and receive raw Reticulum packets over a virtual Ethernet link via the `NetworkInterface` trait.

## Architecture

```
NetworkInterface::send(reticulum_bytes)
        |
        v
+-------------------------+
|   VirtioNet driver      |
|  +-------------------+  |
|  | Ethernet framing  |  |  <- add/strip 14-byte header
|  | (src MAC, dst MAC,|  |
|  |  EtherType 0x88B5)|  |  <- 0x88B5 = Local Experimental
|  +-------------------+  |
|  | TX virtqueue      |  |  <- static buffer pool (32x2KB)
|  | RX virtqueue      |  |  <- static buffer pool (32x2KB)
|  +-------------------+  |
|  | VirtIO 1.0+ PCI   |  |  <- modern MMIO-style via BAR
|  | device negotiation|  |
|  +-------------------+  |
+-------------------------+
        |
        v
   QEMU VirtIO-net-pci
   (socket/mcast backend)
```

## Key Decisions

### VirtIO Transport: PCI Modern (1.0+)

QEMU's x86_64 default is VirtIO-PCI. The modern (1.0+) variant maps virtqueue registers into PCI BARs as MMIO, avoiding legacy I/O port access. Requires PCI bus enumeration (scan bus 0 for vendor 0x1AF4, device 0x1041) but the actual driver logic is cleaner than legacy.

Alternative considered: VirtIO-MMIO is simpler (~500 lines) but requires non-default QEMU flags (`-machine microvm`), creating friction for anyone running `just run`.

### DMA Memory: Heap Allocation

Allocate virtqueue rings and packet buffers from the existing heap (`linked_list_allocator`). Convert virtual to physical addresses by subtracting the bootloader's `physical_memory_offset`.

Rationale: Single address space, small allocations (~16KB for rings, ~128KB for buffers), no fragmentation risk at this scale. A dedicated DMA region is the right call for Ring 2 (capability-gated memory, process isolation) but over-engineering for Ring 1.

Alternative considered: Carving a dedicated DMA region at boot. Lower switching cost (A to B is ~50 lines inside one module) doesn't justify the complexity now. The `NetworkInterface` trait boundary isolates all DMA details from the rest of the system.

### Packet Buffers: Static Pool

Pre-allocate 32 x 2KB buffers per virtqueue (RX and TX) at init time. Zero allocation on the hot path. Buffers are recycled when the device signals completion via the used ring.

Rationale: The event loop runs `poll -> process -> send -> hlt` thousands of times per second. Reticulum MTU is 500 bytes, so 2KB is generous. 32 buffers per direction provides sufficient depth before backpressure.

### Ethernet Framing: Driver-Internal

The driver adds a 14-byte Ethernet header on send and strips it on receive. `NetworkInterface::send()` and `receive()` deal in raw Reticulum packet bytes, not Ethernet frames.

- EtherType: 0x88B5 (IEEE Local Experimental Ethertype 1). Reserved for private protocols, won't collide with IP, ARP, etc.
- Broadcast MAC (ff:ff:ff:ff:ff:ff) for initial mesh discovery. Unicast MAC optimization is future work.

### I/O Model: Poll-Based

The event loop polls `receive()` on each iteration (between `hlt` instructions). No interrupt handler, no IDT setup, no synchronization. Sufficient for a single-threaded unikernel.

Interrupt-driven I/O (MSI-X) is a Ring 2 concern — it requires IDT management, interrupt routing, and handler synchronization that add complexity without benefit in a single-purpose event loop.

### QEMU Backend: Socket/Multicast

```
-device virtio-net-pci,netdev=n0 -netdev socket,id=n0,mcast=230.0.0.1:1234
```

Multiple QEMU instances join the same multicast group for instant virtual LAN. Zero host configuration, no root/sudo, no TAP setup. Perfect for the next bead (harmony-44x: two unikernels discover each other via Reticulum announces).

The VirtIO-net device looks identical to the guest regardless of backend — switching to TAP for production-like testing is a QEMU flag change, not a driver change.

## Components

### 1. PCI Bus Scanner (~100 lines)

Scan bus 0 via x86 I/O ports 0xCF8 (CONFIG_ADDRESS) / 0xCFC (CONFIG_DATA). For each device/function slot:
- Read vendor ID (0x1AF4 = VirtIO)
- Read device ID (0x1041 = modern network device)
- Parse capability list to find VIRTIO_PCI_CAP_COMMON_CFG, VIRTIO_PCI_CAP_NOTIFY_CFG, VIRTIO_PCI_CAP_DEVICE_CFG
- Read BARs for MMIO base addresses

### 2. VirtIO Device Initialization (~150 lines)

Follow VirtIO 1.0 spec initialization sequence:
1. Reset device (write 0 to status)
2. Set ACKNOWLEDGE status bit
3. Set DRIVER status bit
4. Read/negotiate feature bits (VIRTIO_NET_F_MAC for device-assigned MAC)
5. Set FEATURES_OK
6. Allocate and configure virtqueues (RX = queue 0, TX = queue 1)
7. Set DRIVER_OK — device is live

### 3. Virtqueue Implementation (~200 lines)

VirtIO 1.0 split virtqueue with:
- **Descriptor table:** 32 entries, each pointing to a buffer with addr/len/flags/next
- **Available ring:** Guest-written ring of descriptor indices for the device to consume
- **Used ring:** Device-written ring of completed descriptor indices

Operations:
- `post_receive(buf_idx)` — post a device-writable descriptor for incoming packets
- `poll_used()` — check used ring head vs last-seen index, return completed descriptors
- `submit_send(data)` — copy data to buffer, post device-readable descriptor, notify device

Static buffer pool: `[[u8; 2048]; 32]` per queue, allocated once at init.

### 4. VirtioNet: NetworkInterface Implementation (~100 lines)

```rust
pub struct VirtioNet {
    common_cfg: *mut VirtioCommonCfg,  // MMIO registers
    notify_base: *mut u16,
    rx_queue: Virtqueue,
    tx_queue: Virtqueue,
    mac: [u8; 6],
}

impl NetworkInterface for VirtioNet {
    fn name(&self) -> &str { "virtio0" }
    fn mtu(&self) -> usize { 1500 }  // Ethernet payload MTU
    fn send(&mut self, data: &[u8]) -> Result<(), PlatformError>;
    fn receive(&mut self) -> Option<Vec<u8>>;
}
```

- `send()`: prepend 14-byte Ethernet header (src=self.mac, dst=broadcast, type=0x88B5) -> submit to TX queue -> notify
- `receive()`: poll RX used ring -> if packet available, strip Ethernet header -> return payload bytes

### 5. Event Loop Integration (~30 lines)

In `kernel_main()`:
1. After heap + entropy init, scan PCI for VirtIO-net
2. Initialize `VirtioNet`
3. In the event loop:
   - `virtio.receive()` -> if Some(data), feed `NodeEvent::InboundPacket` to runtime
   - `runtime.tick(now)` -> for each `NodeAction::SendOnInterface`, call `virtio.send()`

The `UnikernelRuntime` will need a `NetworkInterface` parameter or method to register interfaces.

### 6. Justfile Updates

```just
# Run with VirtIO-net (multicast LAN)
run: build
    qemu-system-x86_64 \
        -drive format=raw,file=target/harmony-boot-bios.img \
        -serial stdio -display none \
        -device isa-debug-exit,iobase=0xf4,iosize=0x04 \
        -cpu qemu64,+rdrand \
        -device virtio-net-pci,netdev=n0 \
        -netdev socket,id=n0,mcast=230.0.0.1:1234

# Run a second peer on the same virtual LAN
run-peer: build
    qemu-system-x86_64 \
        -drive format=raw,file=target/harmony-boot-bios.img \
        -serial stdio -display none \
        -device isa-debug-exit,iobase=0xf4,iosize=0x04 \
        -cpu qemu64,+rdrand \
        -device virtio-net-pci,netdev=n0 \
        -netdev socket,id=n0,mcast=230.0.0.1:1234
```

## Repo Split

| Repo | Changes |
|------|---------|
| harmony-os | All driver code: `crates/harmony-boot/src/virtio/`, `crates/harmony-boot/src/pci.rs`, event loop changes in `main.rs`, justfile updates |
| harmony | Design doc, implementation plan doc only |

## Testing

- **Unit tests (host-native):** Virtqueue descriptor chaining, ring wraparound, buffer indexing. These test the data structure logic without hardware.
- **QEMU boot test:** Boot -> PCI scan -> VirtIO init -> log `[VIRTIO] virtio0 ready, MAC=xx:xx:xx:xx:xx:xx` -> exit. Verifies the full init sequence works on real (emulated) hardware.
- **Two-node test (manual, documented):** Launch two QEMU instances on the same multicast group. Visual verification that packets flow. Automated version is harmony-44x scope.

## Out of Scope

- Interrupt-driven I/O / MSI-X (Ring 2)
- Multiqueue / RSS
- Checksum offload, TSO, GRO
- TAP backend support
- VLAN tagging
- Athenaeum integration (Ring 2+, tracked in harmony-ihy)
- smoltcp / IP stack (not needed — Reticulum operates at L2)

## Content-Addressing Boundary

The `NetworkInterface` trait is the boundary between location-addressed and content-addressed paradigms:

- **Below** (this driver): Physical DMA buffers, PCI BAR registers, virtqueue rings. Fundamentally location-addressed — the device does DMA to specific physical memory locations.
- **Above** (protocol stack): Reticulum packets, Zenoh references, CID-addressed content, and eventually the Athenaeum 32-bit chunk system (harmony-ihy). Content-addressed.

This boundary is clean: swapping the DMA allocation strategy (e.g., to capability-gated regions in Ring 2) requires changes only inside the driver module. Nothing above `NetworkInterface` is affected.
