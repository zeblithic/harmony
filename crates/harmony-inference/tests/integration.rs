//! Integration tests for harmony-inference.
//!
//! These tests require real GGUF model files and are marked `#[ignore]`.
//! To run: `cargo test -p harmony-inference --test integration -- --ignored`
//!
//! Required environment variables:
//! - `HARMONY_TEST_GGUF`: path to a Qwen3 GGUF file (e.g. qwen3-0.8b-q4_k_m.gguf)
//! - `HARMONY_TEST_TOKENIZER`: path to the matching tokenizer.json

use candle_core::Device;
use harmony_inference::{InferenceEngine, QwenEngine, SamplingParams};

fn load_test_engine() -> QwenEngine {
    let gguf_path =
        std::env::var("HARMONY_TEST_GGUF").expect("set HARMONY_TEST_GGUF to a Qwen3 GGUF path");
    let tokenizer_path = std::env::var("HARMONY_TEST_TOKENIZER")
        .expect("set HARMONY_TEST_TOKENIZER to a tokenizer.json path");

    let gguf_bytes =
        std::fs::read(&gguf_path).unwrap_or_else(|e| panic!("failed to read {gguf_path}: {e}"));
    let tokenizer_bytes = std::fs::read(&tokenizer_path)
        .unwrap_or_else(|e| panic!("failed to read {tokenizer_path}: {e}"));

    let mut engine = QwenEngine::new(Device::Cpu);
    engine
        .load_gguf(&gguf_bytes)
        .unwrap_or_else(|e| panic!("failed to load GGUF: {e}"));
    engine
        .load_tokenizer(&tokenizer_bytes)
        .unwrap_or_else(|e| panic!("failed to load tokenizer: {e}"));
    engine
}

#[test]
#[ignore]
fn test_load_gguf_model() {
    let _engine = load_test_engine();
    // If we get here without panicking, the model loaded successfully
}

#[test]
#[ignore]
fn test_tokenize_and_detokenize() {
    let engine = load_test_engine();
    let text = "What is Harmony?";
    let tokens = engine.tokenize(text).expect("tokenize failed");
    assert!(!tokens.is_empty(), "tokenization produced no tokens");
    assert!(
        tokens.len() < 100,
        "unexpectedly many tokens: {}",
        tokens.len()
    );

    let decoded = engine.detokenize(&tokens).expect("detokenize failed");
    assert!(
        decoded.contains("Harmony"),
        "roundtrip lost content: {decoded:?}"
    );
}

#[test]
#[ignore]
fn test_forward_pass_returns_logits() {
    let mut engine = load_test_engine();
    let tokens = engine.tokenize("Hello").expect("tokenize failed");
    let logits = engine.forward(&tokens).expect("forward failed");

    // Logits should have vocab_size entries (Qwen3 vocab is ~151k)
    assert!(
        logits.len() > 1000,
        "logits too short: {} (expected vocab_size)",
        logits.len()
    );
    // All values should be finite
    assert!(
        logits.iter().all(|l| l.is_finite()),
        "logits contain non-finite values"
    );
}

#[test]
#[ignore]
fn test_sample_produces_valid_token() {
    let mut engine = load_test_engine();
    let tokens = engine
        .tokenize("The capital of France is")
        .expect("tokenize");
    let logits = engine.forward(&tokens).expect("forward");

    let token = engine
        .sample(&logits, &SamplingParams::greedy())
        .expect("sample");
    assert!(
        (token as usize) < logits.len(),
        "sampled token {token} >= vocab size {}",
        logits.len()
    );

    let text = engine.detokenize(&[token]).expect("detokenize");
    assert!(!text.is_empty(), "sampled token decoded to empty string");
}

#[test]
#[ignore]
fn test_generate_ten_tokens() {
    let mut engine = load_test_engine();
    let prompt_tokens = engine.tokenize("Once upon a time").expect("tokenize");
    let logits = engine.forward(&prompt_tokens).expect("prefill");
    let mut next = engine
        .sample(&logits, &SamplingParams::greedy())
        .expect("sample");

    let mut generated = vec![next];
    for _ in 0..9 {
        let logits = engine.forward(&[next]).expect("decode step");
        next = engine
            .sample(&logits, &SamplingParams::greedy())
            .expect("sample");
        generated.push(next);
    }

    let text = engine.detokenize(&generated).expect("detokenize");
    assert!(!text.is_empty(), "generated text is empty");
    eprintln!("Generated: {text}");
}

#[test]
#[ignore]
fn test_reset_allows_new_conversation() {
    let mut engine = load_test_engine();

    // First conversation
    let tokens = engine.tokenize("Hello").expect("tokenize");
    let _ = engine.forward(&tokens).expect("forward");

    // Reset
    engine.reset();

    // Second conversation should work from position 0
    let tokens = engine.tokenize("Goodbye").expect("tokenize");
    let logits = engine.forward(&tokens).expect("forward after reset");
    assert!(!logits.is_empty());
}
