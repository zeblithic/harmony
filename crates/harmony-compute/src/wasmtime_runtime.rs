//! WASM execution engine backed by wasmtime with JIT compilation.
//!
//! Alternative to `WasmiRuntime` for desktop/server nodes. Uses JIT compilation
//! for faster execution. Does not support cooperative yielding — modules run to
//! completion or fail on fuel exhaustion. NeedsIO is handled via deterministic
//! replay with a cached IO oracle.

/// Wasmtime-backed WASM execution engine with JIT compilation.
pub struct WasmtimeRuntime {
    _engine: wasmtime::Engine,
}
