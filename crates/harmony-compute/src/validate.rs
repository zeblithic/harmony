use wasmparser::{Operator, Parser, Payload};

/// Scans a WASM binary for floating-point instructions and rejects it if any
/// are found. This prevents non-deterministic execution: wasmi 1.x does not
/// canonicalize NaN bit-patterns, so float arithmetic produces different
/// results on x86_64 vs aarch64, breaking memo-safe caching.
///
/// Only validates WASM binary (starts with `\0asm`). Non-binary input (e.g.,
/// WAT text) is passed through — wasmparser will fail to parse it, producing
/// an error that still prevents execution. In production, modules are always
/// WASM binary loaded from CAS by hash; WAT is only used in tests (where
/// callers should convert via `wat::parse_str()` first).
///
/// Precedent: CosmWasm bans floats for the same reason (CWIPs #2).
pub fn reject_float_instructions(module_bytes: &[u8]) -> Result<(), String> {
    // Only validate WASM binary. Non-binary input (WAT text, garbage) is
    // passed through — the runtime's Module::new() will handle parsing or
    // error reporting. In production, modules are always WASM binary loaded
    // from CAS by content hash. WAT is only used in tests, where callers
    // should convert to WASM via wat::parse_str() before calling this
    // function if float validation is needed (see execute_rejects_float_module
    // test for the pattern).
    if !module_bytes.starts_with(b"\0asm") {
        return Ok(());
    }

    for payload in Parser::new(0).parse_all(module_bytes) {
        let payload = payload.map_err(|e| format!("failed to parse WASM module: {e}"))?;
        if let Payload::CodeSectionEntry(body) = payload {
            let mut reader = body
                .get_operators_reader()
                .map_err(|e| format!("failed to read WASM operators: {e}"))?;
            while !reader.eof() {
                let op = reader
                    .read()
                    .map_err(|e| format!("failed to decode WASM instruction: {e}"))?;
                if let Some(name) = float_op_name(&op) {
                    return Err(format!(
                        "module contains float instruction `{name}` — rejected for \
                         deterministic execution (wasmi does not canonicalize NaN \
                         bit-patterns across architectures)"
                    ));
                }
            }
        }
    }
    Ok(())
}

/// Returns the WAT name of a float instruction, or `None` for non-float ops.
fn float_op_name(op: &Operator) -> Option<&'static str> {
    match op {
        // f32/f64 constants
        Operator::F32Const { .. } => Some("f32.const"),
        Operator::F64Const { .. } => Some("f64.const"),

        // f32/f64 memory access — don't produce NaN non-determinism on
        // their own (just move bits), but a module using float load/store
        // is operating on float data and should be rejected for consistency.
        Operator::F32Load { .. } => Some("f32.load"),
        Operator::F64Load { .. } => Some("f64.load"),
        Operator::F32Store { .. } => Some("f32.store"),
        Operator::F64Store { .. } => Some("f64.store"),

        // f32 comparison
        Operator::F32Eq => Some("f32.eq"),
        Operator::F32Ne => Some("f32.ne"),
        Operator::F32Lt => Some("f32.lt"),
        Operator::F32Gt => Some("f32.gt"),
        Operator::F32Le => Some("f32.le"),
        Operator::F32Ge => Some("f32.ge"),

        // f64 comparison
        Operator::F64Eq => Some("f64.eq"),
        Operator::F64Ne => Some("f64.ne"),
        Operator::F64Lt => Some("f64.lt"),
        Operator::F64Gt => Some("f64.gt"),
        Operator::F64Le => Some("f64.le"),
        Operator::F64Ge => Some("f64.ge"),

        // f32 unary
        Operator::F32Abs => Some("f32.abs"),
        Operator::F32Neg => Some("f32.neg"),
        Operator::F32Ceil => Some("f32.ceil"),
        Operator::F32Floor => Some("f32.floor"),
        Operator::F32Trunc => Some("f32.trunc"),
        Operator::F32Nearest => Some("f32.nearest"),
        Operator::F32Sqrt => Some("f32.sqrt"),

        // f32 binary
        Operator::F32Add => Some("f32.add"),
        Operator::F32Sub => Some("f32.sub"),
        Operator::F32Mul => Some("f32.mul"),
        Operator::F32Div => Some("f32.div"),
        Operator::F32Min => Some("f32.min"),
        Operator::F32Max => Some("f32.max"),
        Operator::F32Copysign => Some("f32.copysign"),

        // f64 unary
        Operator::F64Abs => Some("f64.abs"),
        Operator::F64Neg => Some("f64.neg"),
        Operator::F64Ceil => Some("f64.ceil"),
        Operator::F64Floor => Some("f64.floor"),
        Operator::F64Trunc => Some("f64.trunc"),
        Operator::F64Nearest => Some("f64.nearest"),
        Operator::F64Sqrt => Some("f64.sqrt"),

        // f64 binary
        Operator::F64Add => Some("f64.add"),
        Operator::F64Sub => Some("f64.sub"),
        Operator::F64Mul => Some("f64.mul"),
        Operator::F64Div => Some("f64.div"),
        Operator::F64Min => Some("f64.min"),
        Operator::F64Max => Some("f64.max"),
        Operator::F64Copysign => Some("f64.copysign"),

        // Conversions FROM float to int
        Operator::I32TruncF32S => Some("i32.trunc_f32_s"),
        Operator::I32TruncF32U => Some("i32.trunc_f32_u"),
        Operator::I32TruncF64S => Some("i32.trunc_f64_s"),
        Operator::I32TruncF64U => Some("i32.trunc_f64_u"),
        Operator::I64TruncF32S => Some("i64.trunc_f32_s"),
        Operator::I64TruncF32U => Some("i64.trunc_f32_u"),
        Operator::I64TruncF64S => Some("i64.trunc_f64_s"),
        Operator::I64TruncF64U => Some("i64.trunc_f64_u"),

        // Conversions FROM int to float
        Operator::F32ConvertI32S => Some("f32.convert_i32_s"),
        Operator::F32ConvertI32U => Some("f32.convert_i32_u"),
        Operator::F32ConvertI64S => Some("f32.convert_i64_s"),
        Operator::F32ConvertI64U => Some("f32.convert_i64_u"),
        Operator::F64ConvertI32S => Some("f64.convert_i32_s"),
        Operator::F64ConvertI32U => Some("f64.convert_i32_u"),
        Operator::F64ConvertI64S => Some("f64.convert_i64_s"),
        Operator::F64ConvertI64U => Some("f64.convert_i64_u"),

        // Float width conversions
        Operator::F32DemoteF64 => Some("f32.demote_f64"),
        Operator::F64PromoteF32 => Some("f64.promote_f32"),

        // Reinterpret (bit-cast between int and float)
        Operator::I32ReinterpretF32 => Some("i32.reinterpret_f32"),
        Operator::I64ReinterpretF64 => Some("i64.reinterpret_f64"),
        Operator::F32ReinterpretI32 => Some("f32.reinterpret_i32"),
        Operator::F64ReinterpretI64 => Some("f64.reinterpret_i64"),

        // Saturating truncation (0xFC prefix)
        Operator::I32TruncSatF32S => Some("i32.trunc_sat_f32_s"),
        Operator::I32TruncSatF32U => Some("i32.trunc_sat_f32_u"),
        Operator::I32TruncSatF64S => Some("i32.trunc_sat_f64_s"),
        Operator::I32TruncSatF64U => Some("i32.trunc_sat_f64_u"),
        Operator::I64TruncSatF32S => Some("i64.trunc_sat_f32_s"),
        Operator::I64TruncSatF32U => Some("i64.trunc_sat_f32_u"),
        Operator::I64TruncSatF64S => Some("i64.trunc_sat_f64_s"),
        Operator::I64TruncSatF64U => Some("i64.trunc_sat_f64_u"),

        // ── SIMD float instructions (0xFD prefix) ──
        // These have identical NaN non-determinism: hardware SIMD lanes
        // produce architecture-dependent NaN payloads.

        // f32x4 splat/lane access
        Operator::F32x4Splat => Some("f32x4.splat"),
        Operator::F32x4ExtractLane { .. } => Some("f32x4.extract_lane"),
        Operator::F32x4ReplaceLane { .. } => Some("f32x4.replace_lane"),

        // f64x2 splat/lane access
        Operator::F64x2Splat => Some("f64x2.splat"),
        Operator::F64x2ExtractLane { .. } => Some("f64x2.extract_lane"),
        Operator::F64x2ReplaceLane { .. } => Some("f64x2.replace_lane"),

        // f32x4 comparison
        Operator::F32x4Eq => Some("f32x4.eq"),
        Operator::F32x4Ne => Some("f32x4.ne"),
        Operator::F32x4Lt => Some("f32x4.lt"),
        Operator::F32x4Gt => Some("f32x4.gt"),
        Operator::F32x4Le => Some("f32x4.le"),
        Operator::F32x4Ge => Some("f32x4.ge"),

        // f64x2 comparison
        Operator::F64x2Eq => Some("f64x2.eq"),
        Operator::F64x2Ne => Some("f64x2.ne"),
        Operator::F64x2Lt => Some("f64x2.lt"),
        Operator::F64x2Gt => Some("f64x2.gt"),
        Operator::F64x2Le => Some("f64x2.le"),
        Operator::F64x2Ge => Some("f64x2.ge"),

        // f32x4 unary
        Operator::F32x4Abs => Some("f32x4.abs"),
        Operator::F32x4Neg => Some("f32x4.neg"),
        Operator::F32x4Sqrt => Some("f32x4.sqrt"),
        Operator::F32x4Ceil => Some("f32x4.ceil"),
        Operator::F32x4Floor => Some("f32x4.floor"),
        Operator::F32x4Trunc => Some("f32x4.trunc"),
        Operator::F32x4Nearest => Some("f32x4.nearest"),

        // f32x4 binary
        Operator::F32x4Add => Some("f32x4.add"),
        Operator::F32x4Sub => Some("f32x4.sub"),
        Operator::F32x4Mul => Some("f32x4.mul"),
        Operator::F32x4Div => Some("f32x4.div"),
        Operator::F32x4Min => Some("f32x4.min"),
        Operator::F32x4Max => Some("f32x4.max"),
        Operator::F32x4PMin => Some("f32x4.pmin"),
        Operator::F32x4PMax => Some("f32x4.pmax"),

        // f64x2 unary
        Operator::F64x2Abs => Some("f64x2.abs"),
        Operator::F64x2Neg => Some("f64x2.neg"),
        Operator::F64x2Sqrt => Some("f64x2.sqrt"),
        Operator::F64x2Ceil => Some("f64x2.ceil"),
        Operator::F64x2Floor => Some("f64x2.floor"),
        Operator::F64x2Trunc => Some("f64x2.trunc"),
        Operator::F64x2Nearest => Some("f64x2.nearest"),

        // f64x2 binary
        Operator::F64x2Add => Some("f64x2.add"),
        Operator::F64x2Sub => Some("f64x2.sub"),
        Operator::F64x2Mul => Some("f64x2.mul"),
        Operator::F64x2Div => Some("f64x2.div"),
        Operator::F64x2Min => Some("f64x2.min"),
        Operator::F64x2Max => Some("f64x2.max"),
        Operator::F64x2PMin => Some("f64x2.pmin"),
        Operator::F64x2PMax => Some("f64x2.pmax"),

        // SIMD conversions involving floats
        Operator::F32x4ConvertI32x4S => Some("f32x4.convert_i32x4_s"),
        Operator::F32x4ConvertI32x4U => Some("f32x4.convert_i32x4_u"),
        Operator::F64x2ConvertLowI32x4S => Some("f64x2.convert_low_i32x4_s"),
        Operator::F64x2ConvertLowI32x4U => Some("f64x2.convert_low_i32x4_u"),
        Operator::I32x4TruncSatF32x4S => Some("i32x4.trunc_sat_f32x4_s"),
        Operator::I32x4TruncSatF32x4U => Some("i32x4.trunc_sat_f32x4_u"),
        Operator::I32x4TruncSatF64x2SZero => Some("i32x4.trunc_sat_f64x2_s_zero"),
        Operator::I32x4TruncSatF64x2UZero => Some("i32x4.trunc_sat_f64x2_u_zero"),
        Operator::F32x4DemoteF64x2Zero => Some("f32x4.demote_f64x2_zero"),
        Operator::F64x2PromoteLowF32x4 => Some("f64x2.promote_low_f32x4"),

        // Relaxed SIMD float instructions — explicitly non-deterministic by
        // spec (implementation may use FMA or separate mul+add).
        Operator::F32x4RelaxedMadd => Some("f32x4.relaxed_madd"),
        Operator::F32x4RelaxedNmadd => Some("f32x4.relaxed_nmadd"),
        Operator::F64x2RelaxedMadd => Some("f64x2.relaxed_madd"),
        Operator::F64x2RelaxedNmadd => Some("f64x2.relaxed_nmadd"),
        Operator::F32x4RelaxedMin => Some("f32x4.relaxed_min"),
        Operator::F32x4RelaxedMax => Some("f32x4.relaxed_max"),
        Operator::F64x2RelaxedMin => Some("f64x2.relaxed_min"),
        Operator::F64x2RelaxedMax => Some("f64x2.relaxed_max"),
        Operator::I32x4RelaxedTruncF32x4S => Some("i32x4.relaxed_trunc_f32x4_s"),
        Operator::I32x4RelaxedTruncF32x4U => Some("i32x4.relaxed_trunc_f32x4_u"),
        Operator::I32x4RelaxedTruncF64x2SZero => Some("i32x4.relaxed_trunc_f64x2_s_zero"),
        Operator::I32x4RelaxedTruncF64x2UZero => Some("i32x4.relaxed_trunc_f64x2_u_zero"),

        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wat_to_wasm(wat: &str) -> Vec<u8> {
        wat::parse_str(wat).expect("valid WAT")
    }

    #[test]
    fn accepts_integer_only_module() {
        let wasm = wat_to_wasm(
            r#"(module
              (memory (export "memory") 1)
              (func (export "compute") (param i32) (param i32) (result i32)
                (i32.add (local.get 0) (local.get 1))))"#,
        );
        assert!(reject_float_instructions(&wasm).is_ok());
    }

    #[test]
    fn rejects_f32_arithmetic() {
        let wasm = wat_to_wasm(
            r#"(module
              (memory (export "memory") 1)
              (func (export "compute") (param i32) (param i32) (result i32)
                (drop (f32.add (f32.const 1.0) (f32.const 2.0)))
                (i32.const 0)))"#,
        );
        let err = reject_float_instructions(&wasm).unwrap_err();
        // First float instruction encountered is f32.const (the first operand)
        assert!(err.contains("f32.const"), "error was: {err}");
    }

    #[test]
    fn rejects_f64_arithmetic() {
        let wasm = wat_to_wasm(
            r#"(module
              (memory (export "memory") 1)
              (func (export "compute") (param i32) (param i32) (result i32)
                (drop (f64.mul (f64.const 3.14) (f64.const 2.0)))
                (i32.const 0)))"#,
        );
        let err = reject_float_instructions(&wasm).unwrap_err();
        assert!(err.contains("f64.const"), "error was: {err}");
    }

    #[test]
    fn rejects_float_conversion() {
        let wasm = wat_to_wasm(
            r#"(module
              (memory (export "memory") 1)
              (func (export "compute") (param i32) (param i32) (result i32)
                (i32.trunc_f32_s (f32.const 42.0))))"#,
        );
        let err = reject_float_instructions(&wasm).unwrap_err();
        assert!(err.contains("f32.const"), "error was: {err}");
    }

    #[test]
    fn rejects_reinterpret() {
        let wasm = wat_to_wasm(
            r#"(module
              (memory (export "memory") 1)
              (func (export "compute") (param i32) (param i32) (result i32)
                (i32.reinterpret_f32 (f32.const 0.0))))"#,
        );
        let err = reject_float_instructions(&wasm).unwrap_err();
        assert!(err.contains("f32.const"), "error was: {err}");
    }

    #[test]
    fn rejects_simd_f32x4() {
        let wasm = wat_to_wasm(
            r#"(module
              (memory (export "memory") 1)
              (func (export "compute") (param i32) (param i32) (result i32)
                (drop (f32x4.add (v128.const f32x4 1.0 2.0 3.0 4.0)
                                 (v128.const f32x4 5.0 6.0 7.0 8.0)))
                (i32.const 0)))"#,
        );
        let err = reject_float_instructions(&wasm).unwrap_err();
        assert!(err.contains("f32x4"), "error was: {err}");
    }

    #[test]
    fn rejects_simd_f64x2() {
        let wasm = wat_to_wasm(
            r#"(module
              (memory (export "memory") 1)
              (func (export "compute") (param i32) (param i32) (result i32)
                (drop (f64x2.mul (v128.const f64x2 1.0 2.0)
                                 (v128.const f64x2 3.0 4.0)))
                (i32.const 0)))"#,
        );
        let err = reject_float_instructions(&wasm).unwrap_err();
        assert!(err.contains("f64x2"), "error was: {err}");
    }

    #[test]
    fn skips_non_wasm_binary_input() {
        // Non-binary input (WAT text, garbage) is passed through to the
        // runtime's Module::new(). In production, modules are always WASM
        // binary from CAS. The execute_rejects_float_module integration test
        // in wasmi_runtime.rs demonstrates the correct test pattern: convert
        // WAT to WASM binary first, then validate.
        assert!(reject_float_instructions(b"not wasm").is_ok());
    }

    #[test]
    fn rejects_float_in_second_function() {
        let wasm = wat_to_wasm(
            r#"(module
              (memory (export "memory") 1)
              (func $safe (param i32) (result i32) (local.get 0))
              (func (export "compute") (param i32) (param i32) (result i32)
                (drop (f32.neg (f32.const 1.0)))
                (i32.const 0)))"#,
        );
        let err = reject_float_instructions(&wasm).unwrap_err();
        assert!(err.contains("f32.const"), "error was: {err}");
    }

    #[test]
    fn rejects_float_load_store() {
        let wasm = wat_to_wasm(
            r#"(module
              (memory (export "memory") 1)
              (func (export "compute") (param i32) (param i32) (result i32)
                (f32.store (i32.const 0) (f32.load (i32.const 0)))
                (i32.const 0)))"#,
        );
        let err = reject_float_instructions(&wasm).unwrap_err();
        assert!(err.contains("f32.load"), "error was: {err}");
    }

    #[test]
    fn error_message_names_the_instruction() {
        let wasm = wat_to_wasm(
            r#"(module
              (func (result f64) (f64.sqrt (f64.const 4.0))))"#,
        );
        let err = reject_float_instructions(&wasm).unwrap_err();
        assert!(
            err.contains("NaN bit-patterns"),
            "error should explain why: {err}"
        );
    }
}
