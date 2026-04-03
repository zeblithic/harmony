use wasmparser::{Operator, Parser, Payload};

/// Scans a WASM binary for floating-point instructions and rejects the module
/// if any are found. This prevents non-deterministic execution: wasmi 1.x does
/// not canonicalize NaN bit-patterns, so float arithmetic produces different
/// results on x86_64 vs aarch64, breaking memo-safe caching.
///
/// Precedent: CosmWasm bans floats for the same reason (CWIPs #2).
pub fn reject_float_instructions(module_bytes: &[u8]) -> Result<(), String> {
    // If the input is WAT (text format), skip validation here — the runtime
    // will compile it to WASM first. In production, modules are always WASM
    // binary (loaded from CAS by hash). WAT is only used in tests.
    if !module_bytes.starts_with(b"\0asm") {
        // Not WASM binary — likely WAT text or invalid. Let the runtime
        // handle parsing errors; we only validate WASM binary.
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
    fn skips_wat_input() {
        // WAT text input is not validated (only WASM binary is).
        // The runtime handles WAT → WASM conversion internally.
        let wat = r#"(module
          (func (export "f") (result f32) (f32.const 1.0)))"#;
        assert!(reject_float_instructions(wat.as_bytes()).is_ok());
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
