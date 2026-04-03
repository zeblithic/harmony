# WASM Determinism Audit: wasmi Guarantees for Memo-Safe Execution

**Bead:** harmony-qok2
**Date:** 2026-04-02
**Status:** Complete
**Runtime:** wasmi 1.0.9 (resolved version pinned in `Cargo.lock`; `Cargo.toml` workspace constraint is `"1"`)

## Summary

The memo system caches computation results keyed by input CID. If two nodes
execute the same WASM module on the same input and get different outputs,
memo cache entries become inconsistent. This audit verifies whether wasmi
provides the determinism guarantees needed for memo-safe caching.

**Conclusion: The current system is safe.** Our WASM module
(`inference_runner.wat`) performs no float arithmetic inside the interpreter
— all float computation happens in host functions. wasmi's only
cross-platform non-determinism source (NaN bit-patterns) does not affect us
today. A follow-up bead tracks the mitigation needed if future WASM modules
introduce float math.

## Audit Areas

### 1. NaN Bit-Pattern Propagation

**Risk level: Low (not currently affected)**

wasmi 1.0.9 does NOT canonicalize NaN payloads. All float operations
(`f32.add`, `f32.mul`, etc.) delegate to Rust's native `f32`/`f64`
operators, which compile to hardware instructions (SSE/AVX on x86_64, NEON
on aarch64). NaN bit-patterns (sign bit, payload) are architecture-dependent
per Rust RFC 3514.

**Why we're safe today:** `inference_runner.wat` performs zero float
arithmetic. It only uses integer operations (`i32.load`, `i32.store`,
`i32.add`, `i32.shl`, `i32.gt_u`, etc.). The f32 sampling parameters
(temperature, top_p, repeat_penalty) are stored in WASM linear memory as
raw bytes and passed to host functions — the WASM module never interprets
them as floats.

All actual float computation (forward pass, softmax, sampling) happens in
host functions via candle-core, outside the wasmi interpreter.

**If a future WASM module does float math:** Any operation that produces a
NaN and observes its bit-pattern (e.g., via `i32.reinterpret_f32`) will
produce different results on x86_64 vs aarch64. Mitigations:

1. **Wasm-level NaN canonicalization** — instrument the module to
   canonicalize after every float op (what wasm-smith uses for fuzzing)
2. **Ban floats at validation time** — reject modules containing float
   instructions (what CosmWasm does)
3. **Softfloat library** — replace hardware floats with pure-software
   IEEE 754 (what EOS-VM does; requires forking wasmi)

Option 2 is simplest and sufficient unless we need float math in WASM.

**Implemented:** Option 2 was implemented in `harmony-fp4v` (see
`crates/harmony-compute/src/validate.rs`). The `reject_float_instructions()`
function uses `wasmparser` to scan all code section instructions before module
compilation. All 54 float-related opcodes are rejected (constants, arithmetic,
comparisons, conversions, reinterprets, and saturating truncations). The
validator is wired into both `WasmiRuntime::execute()` and
`WasmtimeRuntime::run_module()`, rejecting float modules before compilation
to avoid wasting CPU.

### 2. Fuel Metering Determinism

**Risk level: None**

Fuel costs are applied per wasmi IR instruction. The translation from WASM
bytecode to wasmi IR is deterministic for a given wasmi version, so fuel
consumption for the same module + inputs is identical across x86_64 and
aarch64.

**Caveat:** Fuel consumption is NOT stable across wasmi versions (see
[wasmi-labs/wasmi#1088](https://github.com/wasmi-labs/wasmi/issues/1088)).
Internal IR optimizations can change instruction counts. Since we pin wasmi
to 1.0.9, this doesn't affect us. If we upgrade wasmi, memo caches keyed by
fuel cost would need invalidation.

Our adaptive fuel scaling (`effective_fuel()` in runtime.rs) adjusts the
budget based on data-plane queue depth. This is intentionally
non-deterministic (it's a scheduling concern, not a memo concern) — the
budget determines how much work a node is *willing* to do, not the output.

### 3. Memory Growth

**Risk level: None**

`inference_runner.wat` declares a fixed 2-page (128KB) memory and never
calls `memory.grow`. All buffer offsets are statically known:

| Region | Offset | Size | Purpose |
|--------|--------|------|---------|
| Input metadata | 0 | 64B | Model CIDs |
| Prompt length | 64 | 4B | i32: byte length of prompt text |
| Prompt text | 68 | ~32KB | UTF-8 input |
| Scratch slot | 32764 | 4B | Single-token forward |
| Token buffer | 32768 | 8KB | Tokenized prompt |
| Generated tokens | 40960 | 8KB | Autoregressive output |
| Detokenized output | 49152 | 16KB | UTF-8 output |

If wasmi's `memory.grow` were used: growth within the declared maximum is
deterministic, but `OutOfSystemMemory` failures depend on host resources.
Not relevant to our current module.

### 4. WASI / System Access

**Risk level: None**

The module imports only `harmony.*` host functions:

- `harmony.model_load` — load GGUF model by CID
- `harmony.tokenize` / `harmony.detokenize` — text ↔ tokens
- `harmony.forward` — run inference forward pass
- `harmony.sample` — sample next token from logits

No WASI imports. No clocks, RNG, filesystem, environment variables, or
network access. The WASM sandbox is completely hermetic — all I/O goes
through explicit host function traps that we control.

### 5. Host Function Determinism

**Risk level: Acceptable (by design)**

The host functions themselves use non-deterministic operations:

- `forward()` runs candle-core matrix math (native floats). Cross-platform
  float differences in the forward pass could produce different logit
  vectors on x86_64 vs aarch64.
- `sample()` applies temperature scaling and top-p/top-k filtering to
  logits. With temperature > 0, sampling is inherently stochastic.

**Why this is fine:** The memo system caches results at the **input CID
level**. A memo says "I ran input X and got output Y." Different nodes
aren't expected to independently reproduce the same generative output —
they cache and share results. The non-determinism in sampling is
intentional and the memo system is designed around it:

1. Node A runs inference on input X, produces output Y, signs memo (X → Y)
2. Node B receives the memo, trusts Node A's attestation, caches Y
3. Node B never re-executes X — it uses the cached memo

Cross-platform float differences in `forward()` are also acceptable: if
Node A (x86_64) and Node B (aarch64) both independently run inference on
the same input, they may get slightly different logits and thus different
outputs. But both outputs are valid, and the memo system doesn't require
bitwise agreement — it only requires that each node's own output is
consistent with its own attestation.

### 6. Floating-Point Edge Cases (Cross-Architecture)

**Risk level: Low (host-side only)**

For non-NaN results, IEEE 754 basic operations (add, sub, mul, div, sqrt)
produce bit-identical results on all compliant hardware. Both x86_64 and
aarch64 are compliant. Differences only arise from:

- NaN bit-patterns (sign, payload) — architecture-dependent
- Fused multiply-add (FMA) — x86_64 with AVX2 may fuse where aarch64
  doesn't, or vice versa, producing slightly different rounding
- Extended precision — x86 historically used 80-bit x87 FPU, but SSE/AVX
  (which Rust targets) uses 32/64-bit, matching aarch64

Since our WASM module does no float math, these are host-side concerns
only. The candle-core inference engine may produce slightly different
logits across architectures due to FMA differences, but this is expected
and handled by the memo system's design (see section 5).

## References

- [Rust RFC 3514: Float Semantics](https://rust-lang.github.io/rfcs/3514-float-semantics.html) — NaN non-determinism in Rust
- [WebAssembly Nondeterminism spec](https://github.com/WebAssembly/design/blob/main/Nondeterminism.md) — official list of WASM non-determinism sources
- [wasmi-labs/wasmi#1088](https://github.com/wasmi-labs/wasmi/issues/1088) — unstable fuel metering across versions
- [Wasmtime: Deterministic Execution](https://docs.wasmtime.dev/examples-deterministic-wasm-execution.html) — comparison point
- [Wasmtime differential fuzzing with wasmi](https://github.com/bytecodealliance/wasmtime/issues/4818) — confirms wasmi lacks NaN canonicalization
- [CosmWasm float ban discussion](https://github.com/CosmWasm/CWIPs/issues/2) — precedent for rejecting float instructions
- [Polkadot PVF determinism](https://github.com/paritytech/polkadot/issues/1269) — blockchain WASM determinism approach
