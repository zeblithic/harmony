# WASM Content Read Access via NeedsIO — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Allow WASM compute tasks to read immutable content by CID, turning the NeedsIO path from an error into a working content resolution pipeline.

**Architecture:** WASM modules import a `harmony.fetch_content` host function that traps via wasmi's resumable HostTrap API. The ComputeTier suspends the task (non-blocking), emits a FetchContent action, and resumes when content arrives. Data flows: WASM trap → ComputeResult::NeedsIO → ComputeTierAction::FetchContent → external I/O → ContentFetched event → resume_with_io → WASM continues.

**Tech Stack:** Rust, wasmi 1.x (fuel metering + resumable calls), harmony-compute, harmony-node

---

### Task 1: Add IOResponse type and extend ComputeRuntime trait

**Files:**
- Modify: `crates/harmony-compute/src/types.rs:25-26`
- Modify: `crates/harmony-compute/src/runtime.rs:14-26`
- Modify: `crates/harmony-compute/src/lib.rs:8`

**Step 1: Write the failing test**

In `crates/harmony-compute/src/types.rs`, add to the `#[cfg(test)] mod tests` block (after the `compute_result_failed` test, line ~107):

```rust
#[test]
fn io_response_content_ready() {
    let resp = IOResponse::ContentReady {
        data: vec![1, 2, 3],
    };
    assert!(matches!(resp, IOResponse::ContentReady { data } if data == vec![1, 2, 3]));
}

#[test]
fn io_response_content_not_found() {
    let resp = IOResponse::ContentNotFound;
    assert!(matches!(resp, IOResponse::ContentNotFound));
}
```

In `crates/harmony-compute/src/runtime.rs`, update the `MockRuntime` impl to add `resume_with_io`:

```rust
fn resume_with_io(
    &mut self,
    _response: crate::types::IOResponse,
    _budget: InstructionBudget,
) -> ComputeResult {
    ComputeResult::Failed {
        error: ComputeError::NoPendingExecution,
    }
}
```

And add a test:

```rust
#[test]
fn mock_runtime_resume_with_io() {
    let mut rt = MockRuntime;
    let result = rt.resume_with_io(
        crate::types::IOResponse::ContentNotFound,
        InstructionBudget { fuel: 1000 },
    );
    assert!(matches!(
        result,
        ComputeResult::Failed {
            error: ComputeError::NoPendingExecution
        }
    ));
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p harmony-compute`
Expected: FAIL — `IOResponse` type doesn't exist, `resume_with_io` not in trait

**Step 3: Write minimal implementation**

In `crates/harmony-compute/src/types.rs`, after the `IORequest` enum (line 25), add:

```rust
/// Response to an I/O request, provided by the caller after resolving externally.
#[derive(Debug, Clone)]
pub enum IOResponse {
    /// Content was found and is ready.
    ContentReady { data: Vec<u8> },
    /// Content was not found.
    ContentNotFound,
}
```

In `crates/harmony-compute/src/runtime.rs`, add to the `ComputeRuntime` trait (after `resume`, line 19):

```rust
/// Resume a suspended execution that was waiting for I/O, providing the response.
fn resume_with_io(&mut self, response: crate::types::IOResponse, budget: InstructionBudget) -> ComputeResult;
```

In `crates/harmony-compute/src/lib.rs`, update the re-export on line 8:

```rust
pub use types::{Checkpoint, ComputeResult, IORequest, IOResponse, InstructionBudget};
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p harmony-compute`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/harmony-compute/src/types.rs crates/harmony-compute/src/runtime.rs crates/harmony-compute/src/lib.rs
git commit -m "feat(compute): add IOResponse type and resume_with_io to ComputeRuntime trait"
```

---

### Task 2: Register host function and implement NeedsIO in WasmiRuntime

This is the most complex task — it wires up the wasmi Linker with a host import, changes HostState to track IO requests, introduces `PendingResumable` to store both OutOfFuel and HostTrap continuations, and modifies `handle_call_result` to return `NeedsIO` on host trap.

**Files:**
- Modify: `crates/harmony-compute/src/wasmi_runtime.rs:1-163`

**Context:** wasmi's `TypedResumableCall::HostTrap(TypedResumableCallHostTrap<i32>)` is returned when a host function traps. It has `fn resume(self, ctx, inputs: &[Val]) -> Result<TypedResumableCall<Results>, Error>` which lets us provide the host function's return value and continue execution. The host function must return `Err(wasmi::Error)` to trigger a trap.

**Step 1: Write the failing test**

Add this WAT constant and test at the end of the `#[cfg(test)] mod tests` block in `crates/harmony-compute/src/wasmi_runtime.rs`:

```rust
/// WAT module that imports harmony.fetch_content and calls it.
/// Input: [cid: 32 bytes]
/// Output: [result_code: i32 LE] [fetched_data if result > 0]
///
/// Uses offset 4096 as scratch buffer for fetched content.
const FETCH_WAT: &str = r#"
    (module
      (import "harmony" "fetch_content" (func $fetch (param i32 i32 i32) (result i32)))
      (memory (export "memory") 1)
      (func (export "compute") (param $input_ptr i32) (param $input_len i32) (result i32)
        (local $result i32)
        (local $out_ptr i32)
        (local.set $out_ptr (i32.add (local.get $input_ptr) (local.get $input_len)))
        (local.set $result
          (call $fetch
            (local.get $input_ptr)
            (i32.add (local.get $out_ptr) (i32.const 4))
            (i32.const 1024)))
        (i32.store (local.get $out_ptr) (local.get $result))
        (if (result i32) (i32.gt_s (local.get $result) (i32.const 0))
          (then (i32.add (i32.const 4) (local.get $result)))
          (else (i32.const 4)))))
"#;

#[test]
fn host_function_triggers_needs_io() {
    use crate::types::IORequest;

    let mut rt = WasmiRuntime::new();
    let cid = [0xAB; 32];
    let result = rt.execute(
        FETCH_WAT.as_bytes(),
        &cid,
        InstructionBudget { fuel: 1_000_000 },
    );

    match result {
        ComputeResult::NeedsIO {
            request: IORequest::FetchContent { cid: got_cid },
        } => {
            assert_eq!(got_cid, cid);
        }
        other => panic!("expected NeedsIO with FetchContent, got: {other:?}"),
    }
    assert!(rt.has_pending(), "should have pending execution after NeedsIO");
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p harmony-compute host_function_triggers_needs_io`
Expected: FAIL — linker has no `harmony.fetch_content`, instantiation fails or HostTrap returns Failed

**Step 3: Write minimal implementation**

Replace `HostState`, add `PendingResumable`, update `WasmiSession`, modify `execute()` and `handle_call_result()`:

**a) Replace `HostState` (line 6):**

```rust
/// Host state stored in the wasmi `Store`.
///
/// When a WASM module calls `harmony.fetch_content`, the host function
/// stores the IO request and write target here before trapping.
struct HostState {
    /// IO request captured from the host function call.
    io_request: Option<crate::types::IORequest>,
    /// Where to write the IO response data: (out_ptr, out_cap).
    io_write_target: Option<(u32, u32)>,
}

impl HostState {
    fn new() -> Self {
        Self {
            io_request: None,
            io_write_target: None,
        }
    }
}
```

**b) Add `PendingResumable` enum (after HostState):**

```rust
/// A suspended WASM execution that can be resumed.
enum PendingResumable {
    /// Execution ran out of fuel (cooperative yield).
    OutOfFuel(wasmi::TypedResumableCallOutOfFuel<i32>),
    /// Execution trapped on a host function (IO request).
    HostTrap(wasmi::TypedResumableCallHostTrap<i32>),
}
```

**c) Update `WasmiSession` (line 12-19) to use `PendingResumable`:**

```rust
struct WasmiSession {
    store: wasmi::Store<HostState>,
    instance: wasmi::Instance,
    module_hash: [u8; 32],
    input_len: usize,
    total_fuel_consumed: u64,
    pending: Option<PendingResumable>,
}
```

**d) Update `ExecContext` store type (line 22-31):**

No change needed — it already uses `wasmi::Store<HostState>`.

**e) In `execute()` (lines 196-213), change the Linker setup:**

Replace:
```rust
let linker = <wasmi::Linker<HostState>>::new(&self.engine);
```

With:
```rust
let mut linker = <wasmi::Linker<HostState>>::new(&self.engine);

// Register harmony.fetch_content host import.
linker
    .define(
        "harmony",
        "fetch_content",
        wasmi::Func::wrap(
            &mut store,
            |mut caller: wasmi::Caller<'_, HostState>,
             cid_ptr: i32,
             out_ptr: i32,
             out_cap: i32|
             -> Result<i32, wasmi::Error> {
                // Read the 32-byte CID from WASM memory.
                let memory = caller
                    .get_export("memory")
                    .and_then(|e| e.into_memory())
                    .ok_or_else(|| {
                        wasmi::Error::new("fetch_content: no exported memory")
                    })?;

                let mut cid = [0u8; 32];
                memory
                    .read(&caller, cid_ptr as usize, &mut cid)
                    .map_err(|_| {
                        wasmi::Error::new("fetch_content: failed to read CID from memory")
                    })?;

                // Store the request in HostState for the runtime to read.
                let host = caller.data_mut();
                host.io_request = Some(crate::types::IORequest::FetchContent { cid });
                host.io_write_target = Some((out_ptr as u32, out_cap as u32));

                // Trap to suspend execution — the runtime will resume later.
                Err(wasmi::Error::new("harmony_io_trap"))
            },
        ),
    )
    .expect("failed to define harmony.fetch_content");
```

Also update the `Store::new` call to use `HostState::new()`:
```rust
let mut store = wasmi::Store::new(&self.engine, HostState::new());
```

**f) Update `handle_call_result` HostTrap arm (lines 149-156):**

Replace:
```rust
Ok(wasmi::TypedResumableCall::HostTrap(_)) => {
    // Unexpected host trap (we have no host functions).
    ComputeResult::Failed {
        error: ComputeError::Trap {
            reason: "unexpected host trap".into(),
        },
    }
}
```

With:
```rust
Ok(wasmi::TypedResumableCall::HostTrap(pending)) => {
    // Check if this is an IO request from our host function.
    let io_request = ctx.store.data().io_request.clone();
    match io_request {
        Some(request) => {
            // Compute fuel consumed so far.
            let remaining_fuel = ctx.store.get_fuel().unwrap_or(0);
            let fuel_this_slice = ctx.fuel_budget.saturating_sub(remaining_fuel);
            let total_fuel = ctx.fuel_before + fuel_this_slice;

            self.session = Some(WasmiSession {
                store: ctx.store,
                instance: ctx.instance,
                module_hash: ctx.module_hash,
                input_len: ctx.input_len,
                total_fuel_consumed: total_fuel,
                pending: Some(PendingResumable::HostTrap(pending)),
            });

            ComputeResult::NeedsIO { request }
        }
        None => {
            // Unexpected host trap — no IO request stored.
            ComputeResult::Failed {
                error: ComputeError::Trap {
                    reason: "unexpected host trap".into(),
                },
            }
        }
    }
}
```

**g) Update OutOfFuel arm (line 128-143) to use `PendingResumable::OutOfFuel`:**

Change line `pending: Some(pending),` to `pending: Some(PendingResumable::OutOfFuel(pending)),`

**h) Update `resume()` (lines 295-343) to handle `PendingResumable`:**

Replace the invocation extraction (lines 305-314):
```rust
let invocation = match session.pending.take() {
    Some(inv) => inv,
    None => {
        // Restore session so snapshot() still works.
        self.session = Some(session);
        return ComputeResult::Failed {
            error: ComputeError::NoPendingExecution,
        };
    }
};
```

With:
```rust
let invocation = match session.pending.take() {
    Some(PendingResumable::OutOfFuel(inv)) => inv,
    Some(PendingResumable::HostTrap(_)) => {
        // HostTrap requires resume_with_io, not resume.
        self.session = Some(session);
        return ComputeResult::Failed {
            error: ComputeError::NoPendingExecution,
        };
    }
    None => {
        self.session = Some(session);
        return ComputeResult::Failed {
            error: ComputeError::NoPendingExecution,
        };
    }
};
```

**i) Update `has_pending()` (line 345-347):**

```rust
fn has_pending(&self) -> bool {
    self.session.as_ref().is_some_and(|s| s.pending.is_some())
}
```

No change needed — `pending: Option<PendingResumable>` still works with `.is_some()`.

**j) Update `Completed` arm store to use `HostState::new()`-initialized sessions — actually, the completed arm stores the session as-is, which is fine because HostState fields will just be None.**

**Step 4: Run test to verify it passes**

Run: `cargo test -p harmony-compute host_function_triggers_needs_io`
Expected: PASS

Run: `cargo test -p harmony-compute`
Expected: ALL existing tests pass (OutOfFuel path unchanged, HostState::new() provides None fields)

**Step 5: Commit**

```bash
git add crates/harmony-compute/src/wasmi_runtime.rs
git commit -m "feat(compute): register harmony.fetch_content host import and handle HostTrap as NeedsIO"
```

---

### Task 3: Implement resume_with_io in WasmiRuntime

**Files:**
- Modify: `crates/harmony-compute/src/wasmi_runtime.rs`

**Step 1: Write the failing tests**

Add these tests to `#[cfg(test)] mod tests` in `wasmi_runtime.rs`:

```rust
#[test]
fn resume_with_content_ready() {
    use crate::types::{IORequest, IOResponse};

    let mut rt = WasmiRuntime::new();
    let cid = [0xAB; 32];
    let content = b"hello world";

    // Execute — should get NeedsIO
    let result = rt.execute(
        FETCH_WAT.as_bytes(),
        &cid,
        InstructionBudget { fuel: 1_000_000 },
    );
    assert!(matches!(result, ComputeResult::NeedsIO { .. }));

    // Resume with content
    let result = rt.resume_with_io(
        IOResponse::ContentReady {
            data: content.to_vec(),
        },
        InstructionBudget { fuel: 1_000_000 },
    );

    match result {
        ComputeResult::Complete { output } => {
            // Output: [result_code: i32 LE] [data]
            let result_code = i32::from_le_bytes(output[0..4].try_into().unwrap());
            assert_eq!(result_code, content.len() as i32);
            assert_eq!(&output[4..], content);
        }
        other => panic!("expected Complete, got: {other:?}"),
    }
}

#[test]
fn resume_with_content_not_found() {
    use crate::types::{IORequest, IOResponse};

    let mut rt = WasmiRuntime::new();
    let cid = [0xCD; 32];

    let result = rt.execute(
        FETCH_WAT.as_bytes(),
        &cid,
        InstructionBudget { fuel: 1_000_000 },
    );
    assert!(matches!(result, ComputeResult::NeedsIO { .. }));

    let result = rt.resume_with_io(
        IOResponse::ContentNotFound,
        InstructionBudget { fuel: 1_000_000 },
    );

    match result {
        ComputeResult::Complete { output } => {
            let result_code = i32::from_le_bytes(output[0..4].try_into().unwrap());
            assert_eq!(result_code, -1, "content not found should return -1");
        }
        other => panic!("expected Complete, got: {other:?}"),
    }
}

#[test]
fn resume_with_buffer_too_small() {
    use crate::types::{IORequest, IOResponse};

    // WAT with tiny buffer capacity (2 bytes)
    let small_buf_wat = r#"
        (module
          (import "harmony" "fetch_content" (func $fetch (param i32 i32 i32) (result i32)))
          (memory (export "memory") 1)
          (func (export "compute") (param $input_ptr i32) (param $input_len i32) (result i32)
            (local $result i32)
            (local $out_ptr i32)
            (local.set $out_ptr (i32.add (local.get $input_ptr) (local.get $input_len)))
            (local.set $result
              (call $fetch
                (local.get $input_ptr)
                (i32.add (local.get $out_ptr) (i32.const 4))
                (i32.const 2)))
            (i32.store (local.get $out_ptr) (local.get $result))
            (i32.const 4)))
    "#;

    let mut rt = WasmiRuntime::new();
    let cid = [0xEF; 32];

    let result = rt.execute(
        small_buf_wat.as_bytes(),
        &cid,
        InstructionBudget { fuel: 1_000_000 },
    );
    assert!(matches!(result, ComputeResult::NeedsIO { .. }));

    // Content is 10 bytes, but buffer is only 2
    let result = rt.resume_with_io(
        IOResponse::ContentReady {
            data: vec![0u8; 10],
        },
        InstructionBudget { fuel: 1_000_000 },
    );

    match result {
        ComputeResult::Complete { output } => {
            let result_code = i32::from_le_bytes(output[0..4].try_into().unwrap());
            assert_eq!(result_code, -2, "buffer too small should return -2");
        }
        other => panic!("expected Complete, got: {other:?}"),
    }
}

#[test]
fn fetch_content_then_complete() {
    use crate::types::{IORequest, IOResponse};

    let mut rt = WasmiRuntime::new();
    let cid = [0x42; 32];
    let content = vec![10, 20, 30, 40, 50];

    // Full cycle: execute → NeedsIO → resume_with_io → Complete
    let result = rt.execute(
        FETCH_WAT.as_bytes(),
        &cid,
        InstructionBudget { fuel: 1_000_000 },
    );
    assert!(matches!(result, ComputeResult::NeedsIO { request: IORequest::FetchContent { cid: c } } if c == cid));
    assert!(rt.has_pending());

    let result = rt.resume_with_io(
        IOResponse::ContentReady {
            data: content.clone(),
        },
        InstructionBudget { fuel: 1_000_000 },
    );
    match result {
        ComputeResult::Complete { output } => {
            let result_code = i32::from_le_bytes(output[0..4].try_into().unwrap());
            assert_eq!(result_code, 5);
            assert_eq!(&output[4..], &content);
        }
        other => panic!("expected Complete, got: {other:?}"),
    }
    assert!(!rt.has_pending());
}
```

**Step 2: Run test to verify they fail**

Run: `cargo test -p harmony-compute resume_with_content`
Expected: FAIL — `resume_with_io` not implemented on WasmiRuntime

**Step 3: Write minimal implementation**

Add `resume_with_io` method to the `impl ComputeRuntime for WasmiRuntime` block, after `resume()`:

```rust
fn resume_with_io(
    &mut self,
    response: crate::types::IOResponse,
    budget: InstructionBudget,
) -> ComputeResult {
    let mut session = match self.session.take() {
        Some(s) => s,
        None => {
            return ComputeResult::Failed {
                error: ComputeError::NoPendingExecution,
            };
        }
    };

    let invocation = match session.pending.take() {
        Some(PendingResumable::HostTrap(inv)) => inv,
        Some(PendingResumable::OutOfFuel(_)) => {
            self.session = Some(session);
            return ComputeResult::Failed {
                error: ComputeError::NoPendingExecution,
            };
        }
        None => {
            self.session = Some(session);
            return ComputeResult::Failed {
                error: ComputeError::NoPendingExecution,
            };
        }
    };

    // Read write target from HostState.
    let (out_ptr, out_cap) = match session.store.data().io_write_target {
        Some(target) => target,
        None => {
            return ComputeResult::Failed {
                error: ComputeError::Trap {
                    reason: "resume_with_io: no write target stored".into(),
                },
            };
        }
    };

    // Determine return value and optionally write data to WASM memory.
    let return_val: i32 = match &response {
        crate::types::IOResponse::ContentReady { data } => {
            if data.len() > out_cap as usize {
                -2 // buffer too small
            } else {
                // Write data into WASM memory at out_ptr.
                let memory = match session.instance.get_memory(&session.store, "memory") {
                    Some(mem) => mem,
                    None => {
                        return ComputeResult::Failed {
                            error: ComputeError::ExportNotFound {
                                name: "memory".into(),
                            },
                        };
                    }
                };
                if let Err(_) = memory.write(&mut session.store, out_ptr as usize, data) {
                    return ComputeResult::Failed {
                        error: ComputeError::MemoryTooSmall {
                            need: out_ptr as usize + data.len(),
                            have: memory.data_size(&session.store),
                        },
                    };
                }
                data.len() as i32
            }
        }
        crate::types::IOResponse::ContentNotFound => -1,
    };

    // Clear the IO state from HostState.
    let host = session.store.data_mut();
    host.io_request = None;
    host.io_write_target = None;

    // Set fuel budget for the resumed execution.
    if let Err(e) = session.store.set_fuel(budget.fuel) {
        self.session = Some(session);
        return ComputeResult::Failed {
            error: ComputeError::Trap {
                reason: format!("failed to set fuel: {e}"),
            },
        };
    }

    let fuel_before = session.total_fuel_consumed;
    let input_len = session.input_len;
    let module_hash = session.module_hash;
    let instance = session.instance;

    // Resume the host trap with the return value.
    let call_result = invocation.resume(&mut session.store, &[wasmi::Val::I32(return_val)]);

    self.handle_call_result(
        call_result,
        ExecContext {
            store: session.store,
            instance,
            module_hash,
            input_len,
            fuel_before,
            fuel_budget: budget.fuel,
        },
    )
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-compute`
Expected: ALL tests pass (existing + 5 new)

**Step 5: Commit**

```bash
git add crates/harmony-compute/src/wasmi_runtime.rs
git commit -m "feat(compute): implement resume_with_io for HostTrap content resolution"
```

---

### Task 4: Add WaitingForContent and content IO events to ComputeTier

**Files:**
- Modify: `crates/harmony-node/src/compute.rs`

**Step 1: Write the failing tests**

Add these tests to the `#[cfg(test)] mod tests` block in `crates/harmony-node/src/compute.rs`:

First, add a WAT constant that uses the host import (same as `FETCH_WAT` from Task 2 but defined here for the node crate's tests):

```rust
/// WAT module that calls harmony.fetch_content.
/// Input: [cid: 32 bytes]
/// Output: [result_code: i32 LE] [fetched_data if result > 0]
pub(crate) const FETCH_WAT: &str = r#"
    (module
      (import "harmony" "fetch_content" (func $fetch (param i32 i32 i32) (result i32)))
      (memory (export "memory") 1)
      (func (export "compute") (param $input_ptr i32) (param $input_len i32) (result i32)
        (local $result i32)
        (local $out_ptr i32)
        (local.set $out_ptr (i32.add (local.get $input_ptr) (local.get $input_len)))
        (local.set $result
          (call $fetch
            (local.get $input_ptr)
            (i32.add (local.get $out_ptr) (i32.const 4))
            (i32.const 1024)))
        (i32.store (local.get $out_ptr) (local.get $result))
        (if (result i32) (i32.gt_s (local.get $result) (i32.const 0))
          (then (i32.add (i32.const 4) (local.get $result)))
          (else (i32.const 4)))))
"#;

#[test]
fn needs_io_emits_fetch_content() {
    let mut tier = make_tier();
    let cid = [0xAB; 32];

    // Submit a module that will call fetch_content.
    tier.handle(ComputeTierEvent::ExecuteInline {
        query_id: 100,
        module: FETCH_WAT.as_bytes().to_vec(),
        input: cid.to_vec(), // CID as input
    });

    // tick() should execute, hit NeedsIO, emit FetchContent.
    let actions = tier.tick();
    assert_eq!(actions.len(), 1);
    assert!(
        matches!(&actions[0], ComputeTierAction::FetchContent { query_id: 100, cid: c } if *c == cid),
        "expected FetchContent for the CID, got: {actions:?}"
    );

    // Task should be in WaitingForContent, not active.
    assert!(!tier.has_active());
    assert_eq!(tier.queue_len(), 1); // WaitingForContent in queue
}

#[test]
fn content_fetched_resumes_task() {
    let mut tier = make_tier();
    let cid = [0xCD; 32];
    let content = b"fetched data";

    tier.handle(ComputeTierEvent::ExecuteInline {
        query_id: 101,
        module: FETCH_WAT.as_bytes().to_vec(),
        input: cid.to_vec(),
    });

    // tick → NeedsIO → FetchContent
    let actions = tier.tick();
    assert_eq!(actions.len(), 1);
    assert!(matches!(&actions[0], ComputeTierAction::FetchContent { .. }));

    // Deliver the content.
    let actions = tier.handle(ComputeTierEvent::ContentFetched {
        cid,
        data: content.to_vec(),
    });

    // Should get a SendReply with the result.
    assert_eq!(actions.len(), 1);
    if let ComputeTierAction::SendReply { query_id, payload } = &actions[0] {
        assert_eq!(*query_id, 101);
        assert_eq!(payload[0], 0x00, "success tag");
        let result_code = i32::from_le_bytes(payload[1..5].try_into().unwrap());
        assert_eq!(result_code, content.len() as i32);
        assert_eq!(&payload[5..], content);
    } else {
        panic!("expected SendReply, got: {actions:?}");
    }
}

#[test]
fn content_fetch_failed_returns_error_to_module() {
    let mut tier = make_tier();
    let cid = [0xEF; 32];

    tier.handle(ComputeTierEvent::ExecuteInline {
        query_id: 102,
        module: FETCH_WAT.as_bytes().to_vec(),
        input: cid.to_vec(),
    });

    // tick → NeedsIO → FetchContent
    tier.tick();

    // Content not found.
    let actions = tier.handle(ComputeTierEvent::ContentFetchFailed { cid });

    // Module should complete with result_code = -1.
    assert_eq!(actions.len(), 1);
    if let ComputeTierAction::SendReply { query_id, payload } = &actions[0] {
        assert_eq!(*query_id, 102);
        assert_eq!(payload[0], 0x00, "success tag — module completed normally");
        let result_code = i32::from_le_bytes(payload[1..5].try_into().unwrap());
        assert_eq!(result_code, -1, "not-found result code");
    } else {
        panic!("expected SendReply, got: {actions:?}");
    }
}

#[test]
fn waiting_for_content_does_not_block_queue() {
    let mut tier = make_tier();
    let cid = [0x11; 32];

    // First: module that calls fetch_content (will block on IO).
    tier.handle(ComputeTierEvent::ExecuteInline {
        query_id: 110,
        module: FETCH_WAT.as_bytes().to_vec(),
        input: cid.to_vec(),
    });

    // Second: simple add module (should not be blocked).
    let mut input = Vec::new();
    input.extend_from_slice(&4i32.to_le_bytes());
    input.extend_from_slice(&6i32.to_le_bytes());
    tier.handle(ComputeTierEvent::ExecuteInline {
        query_id: 111,
        module: ADD_WAT.as_bytes().to_vec(),
        input,
    });

    // First tick: executes fetch module → NeedsIO → WaitingForContent.
    let actions = tier.tick();
    assert!(matches!(&actions[0], ComputeTierAction::FetchContent { .. }));
    assert!(!tier.has_active());

    // Second tick: should skip WaitingForContent and execute the add module.
    let actions = tier.tick();
    assert_eq!(actions.len(), 1);
    if let ComputeTierAction::SendReply { query_id, payload } = &actions[0] {
        assert_eq!(*query_id, 111);
        assert_eq!(payload[0], 0x00); // success
        let sum = i32::from_le_bytes(payload[1..5].try_into().unwrap());
        assert_eq!(sum, 10);
    } else {
        panic!("expected add result, got: {actions:?}");
    }
}
```

**Step 2: Run test to verify they fail**

Run: `cargo test -p harmony-node needs_io_emits`
Expected: FAIL — `ComputeTierEvent::ContentFetched` doesn't exist, `ComputeTierAction::FetchContent` doesn't exist

**Step 3: Write minimal implementation**

**a) Add new events (after `ModuleFetchFailed` in `ComputeTierEvent`):**

```rust
/// Content data has been fetched for a suspended compute task.
ContentFetched { cid: [u8; 32], data: Vec<u8> },
/// Content fetch failed for a suspended compute task.
ContentFetchFailed { cid: [u8; 32] },
```

**b) Add new action (after `FetchModule` in `ComputeTierAction`):**

```rust
/// Request the storage/network layer to fetch content by CID for a suspended task.
FetchContent { query_id: u64, cid: [u8; 32] },
```

**c) Add new task state (after `WaitingForModule` in `ComputeTask`):**

```rust
/// Execution suspended waiting for content to be fetched.
/// The execution state lives in WasmiRuntime.session (HostTrap pending).
WaitingForContent { query_id: u64, cid: [u8; 32] },
```

**d) Update `handle_compute_result` NeedsIO arm (lines 257-262):**

```rust
ComputeResult::NeedsIO { request } => {
    match request {
        harmony_compute::IORequest::FetchContent { cid } => {
            self.queue.push_back(ComputeTask::WaitingForContent {
                query_id,
                cid,
            });
            self.active = None;
            vec![ComputeTierAction::FetchContent { query_id, cid }]
        }
    }
}
```

**e) Add `handle()` arms for `ContentFetched` and `ContentFetchFailed`:**

In `handle()`, after the `ModuleFetchFailed` arm:

```rust
ComputeTierEvent::ContentFetched { cid, data } => {
    // Find the WaitingForContent task matching this CID.
    let pos = self.queue.iter().position(|t| {
        matches!(t, ComputeTask::WaitingForContent { cid: c, .. } if *c == cid)
    });
    let Some(pos) = pos else {
        return Vec::new(); // No matching task — drop silently.
    };
    let Some(ComputeTask::WaitingForContent { query_id, .. }) = self.queue.remove(pos)
    else {
        return Vec::new();
    };

    // Resume the runtime with the fetched content.
    self.active = Some(ActiveExecution { query_id });
    let result = self.runtime.resume_with_io(
        harmony_compute::IOResponse::ContentReady { data },
        self.budget,
    );
    self.handle_compute_result(result)
}
ComputeTierEvent::ContentFetchFailed { cid } => {
    let pos = self.queue.iter().position(|t| {
        matches!(t, ComputeTask::WaitingForContent { cid: c, .. } if *c == cid)
    });
    let Some(pos) = pos else {
        return Vec::new();
    };
    let Some(ComputeTask::WaitingForContent { query_id, .. }) = self.queue.remove(pos)
    else {
        return Vec::new();
    };

    // Resume with ContentNotFound — the module will see -1.
    self.active = Some(ActiveExecution { query_id });
    let result = self.runtime.resume_with_io(
        harmony_compute::IOResponse::ContentNotFound,
        self.budget,
    );
    self.handle_compute_result(result)
}
```

**f) Update `tick()` ready_pos search to also skip `WaitingForContent`:**

The existing search already only matches `ComputeTask::Ready`, so `WaitingForContent` is naturally skipped — no change needed.

**g) Add import for `IOResponse` at the top of compute.rs:**

Update the import line to include `IOResponse`:
```rust
use harmony_compute::{ComputeRuntime, IOResponse, InstructionBudget, WasmiRuntime};
```

**Step 4: Run test to verify they pass**

Run: `cargo test -p harmony-node`
Expected: ALL tests pass (existing + 4 new)

Run: `cargo clippy -p harmony-node`
Expected: Zero warnings

**Step 5: Commit**

```bash
git add crates/harmony-node/src/compute.rs
git commit -m "feat(node): add WaitingForContent state and content IO events to ComputeTier"
```

---

### Task 5: Wire ContentFetchResponse into NodeRuntime

**Files:**
- Modify: `crates/harmony-node/src/runtime.rs`

**Step 1: Write the failing tests**

Add these tests at the end of the `#[cfg(test)] mod tests` block in `runtime.rs`:

```rust
#[test]
fn content_fetch_response_routes_to_compute() {
    let (mut rt, _) = make_runtime();
    let cid = [0xAB; 32];

    // First: submit a compute query with a fetch module.
    let module = crate::compute::tests::FETCH_WAT.as_bytes();
    let mut payload = vec![0x00];
    payload.extend_from_slice(&(module.len() as u32).to_le_bytes());
    payload.extend_from_slice(module);
    payload.extend_from_slice(&cid); // CID as input

    rt.push_event(RuntimeEvent::ComputeQuery {
        query_id: 300,
        key_expr: "harmony/compute/activity/test".into(),
        payload,
    });

    // tick → NeedsIO → FetchContent action
    let actions = rt.tick();
    let fetch = actions
        .iter()
        .find(|a| matches!(a, RuntimeAction::FetchContent { .. }));
    assert!(fetch.is_some(), "should emit FetchContent, got: {actions:?}");

    // Deliver content via ContentFetchResponse.
    let content = b"resolved content";
    rt.push_event(RuntimeEvent::ContentFetchResponse {
        cid,
        result: Ok(content.to_vec()),
    });

    // tick → dispatches pending compute actions → SendReply
    let actions = rt.tick();
    let reply = actions
        .iter()
        .find(|a| matches!(a, RuntimeAction::SendReply { query_id: 300, .. }));
    assert!(reply.is_some(), "should emit SendReply for query 300, got: {actions:?}");

    if let Some(RuntimeAction::SendReply { payload, .. }) = reply {
        assert_eq!(payload[0], 0x00, "success tag");
        let result_code = i32::from_le_bytes(payload[1..5].try_into().unwrap());
        assert_eq!(result_code, content.len() as i32);
    }
}

#[test]
fn compute_content_read_round_trip() {
    let (mut rt, _) = make_runtime();
    let cid = [0x42; 32];
    let content = vec![10, 20, 30, 40, 50];

    // Submit inline compute query with fetch module.
    let module = crate::compute::tests::FETCH_WAT.as_bytes();
    let mut payload = vec![0x00];
    payload.extend_from_slice(&(module.len() as u32).to_le_bytes());
    payload.extend_from_slice(module);
    payload.extend_from_slice(&cid);

    rt.push_event(RuntimeEvent::ComputeQuery {
        query_id: 400,
        key_expr: "harmony/compute/activity/fetch".into(),
        payload,
    });

    // Tick 1: execute → NeedsIO → FetchContent
    let actions = rt.tick();
    assert!(
        actions.iter().any(|a| matches!(a, RuntimeAction::FetchContent { cid: c } if *c == cid)),
        "tick 1 should emit FetchContent"
    );

    // External IO resolves the content.
    rt.push_event(RuntimeEvent::ContentFetchResponse {
        cid,
        result: Ok(content.clone()),
    });

    // Tick 2: resume compute → SendReply
    let actions = rt.tick();
    let reply = actions
        .iter()
        .find(|a| matches!(a, RuntimeAction::SendReply { query_id: 400, .. }));
    assert!(reply.is_some(), "tick 2 should emit SendReply");

    if let Some(RuntimeAction::SendReply { payload, .. }) = reply {
        assert_eq!(payload[0], 0x00);
        let result_code = i32::from_le_bytes(payload[1..5].try_into().unwrap());
        assert_eq!(result_code, 5);
        assert_eq!(&payload[5..], &content);
    }
}
```

**Step 2: Run test to verify they fail**

Run: `cargo test -p harmony-node content_fetch_response`
Expected: FAIL — `RuntimeEvent::ContentFetchResponse` doesn't exist

**Step 3: Write minimal implementation**

**a) Add new `RuntimeEvent` variant (after `ModuleFetchResponse`, line 72):**

```rust
/// Tier 3: Content fetch response for a suspended compute task.
ContentFetchResponse {
    cid: [u8; 32],
    result: Result<Vec<u8>, String>,
},
```

**b) Add `push_event` arm for `ContentFetchResponse` (after `ModuleFetchResponse` arm, ~line 246):**

```rust
RuntimeEvent::ContentFetchResponse { cid, result } => {
    let event = match result {
        Ok(data) => ComputeTierEvent::ContentFetched { cid, data },
        Err(_) => ComputeTierEvent::ContentFetchFailed { cid },
    };
    let actions = self.compute.handle(event);
    self.pending_compute_actions.extend(actions);
}
```

**c) Update `dispatch_compute_actions` to handle `FetchContent` (after `FetchModule` arm, ~line 393):**

```rust
ComputeTierAction::FetchContent { cid, .. } => {
    out.push(RuntimeAction::FetchContent { cid });
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-node`
Expected: ALL tests pass (existing + 2 new)

Run: `cargo clippy -p harmony-node`
Expected: Zero warnings

Run: `cargo test --workspace`
Expected: ALL workspace tests pass

**Step 5: Commit**

```bash
git add crates/harmony-node/src/runtime.rs
git commit -m "feat(node): wire ContentFetchResponse into NodeRuntime for compute content reads"
```
