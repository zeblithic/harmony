fn main() {
    let wat_path = std::path::Path::new("src/inference_runner.wat");
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let wasm_path = std::path::Path::new(&out_dir).join("inference_runner.wasm");

    if wat_path.exists() {
        let wat_source =
            std::fs::read_to_string(wat_path).expect("failed to read inference_runner.wat");
        let wasm_bytes =
            wat::parse_str(&wat_source).expect("failed to compile inference_runner.wat to WASM");
        std::fs::write(&wasm_path, wasm_bytes).expect("failed to write inference_runner.wasm");
    } else {
        panic!("src/inference_runner.wat not found — cannot build inference_runner.wasm");
    }

    println!("cargo:rerun-if-changed=src/inference_runner.wat");
}
