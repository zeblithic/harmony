use std::process::Command;

fn harmony_cmd() -> Command {
    Command::new(env!("CARGO_BIN_EXE_harmony"))
}

#[test]
fn identity_new_prints_three_fields() {
    let output = harmony_cmd()
        .args(["identity", "new"])
        .output()
        .expect("failed to run harmony");
    assert!(output.status.success(), "exit code should be 0");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines.len(), 3, "expected 3 lines: {stdout}");

    // Address: 32 hex chars (16 bytes)
    assert!(lines[0].starts_with("Address:"), "line 0: {}", lines[0]);
    let addr_hex = lines[0].trim_start_matches("Address:").trim();
    assert_eq!(addr_hex.len(), 32, "address hex length");
    hex::decode(addr_hex).expect("address should be valid hex");

    // Public key: 128 hex chars (64 bytes)
    assert!(lines[1].starts_with("Public key:"), "line 1: {}", lines[1]);
    let pub_hex = lines[1].trim_start_matches("Public key:").trim();
    assert_eq!(pub_hex.len(), 128, "pubkey hex length");
    hex::decode(pub_hex).expect("pubkey should be valid hex");

    // Private key: 128 hex chars (64 bytes)
    assert!(lines[2].starts_with("Private key:"), "line 2: {}", lines[2]);
    let priv_hex = lines[2].trim_start_matches("Private key:").trim();
    assert_eq!(priv_hex.len(), 128, "privkey hex length");
    hex::decode(priv_hex).expect("privkey should be valid hex");
}

#[test]
fn identity_show_round_trips_from_new() {
    let new_output = harmony_cmd()
        .args(["identity", "new"])
        .output()
        .expect("failed to run identity new");
    let new_stdout = String::from_utf8_lossy(&new_output.stdout);
    let new_lines: Vec<&str> = new_stdout.lines().collect();
    let expected_addr = new_lines[0].trim_start_matches("Address:").trim();
    let expected_pub = new_lines[1].trim_start_matches("Public key:").trim();
    let priv_hex = new_lines[2].trim_start_matches("Private key:").trim();

    let show_output = harmony_cmd()
        .args(["identity", "show", priv_hex])
        .output()
        .expect("failed to run identity show");
    assert!(show_output.status.success());

    let show_stdout = String::from_utf8_lossy(&show_output.stdout);
    let show_lines: Vec<&str> = show_stdout.lines().collect();
    assert_eq!(show_lines.len(), 2, "show should print 2 lines");

    let show_addr = show_lines[0].trim_start_matches("Address:").trim();
    let show_pub = show_lines[1].trim_start_matches("Public key:").trim();
    assert_eq!(show_addr, expected_addr);
    assert_eq!(show_pub, expected_pub);
}

#[test]
fn identity_show_rejects_invalid_hex() {
    let output = harmony_cmd()
        .args(["identity", "show", "not-valid-hex"])
        .output()
        .expect("failed to run");
    assert!(!output.status.success(), "should fail on bad hex");
}

#[test]
fn identity_sign_produces_valid_signature() {
    let new_output = harmony_cmd().args(["identity", "new"]).output().unwrap();
    let new_stdout = String::from_utf8_lossy(&new_output.stdout);
    let new_lines: Vec<&str> = new_stdout.lines().collect();
    let priv_hex = new_lines[2].trim_start_matches("Private key:").trim();

    let sign_output = harmony_cmd()
        .args(["identity", "sign", priv_hex, "hello harmony"])
        .output()
        .unwrap();
    assert!(sign_output.status.success());

    let sign_stdout = String::from_utf8_lossy(&sign_output.stdout);
    let sig_line = sign_stdout.lines().next().expect("should have output");
    assert!(sig_line.starts_with("Signature:"), "line: {sig_line}");
    let sig_hex = sig_line.trim_start_matches("Signature:").trim();
    assert_eq!(
        sig_hex.len(),
        128,
        "signature should be 128 hex chars (64 bytes)"
    );
    hex::decode(sig_hex).expect("signature should be valid hex");
}

#[test]
fn sign_then_verify_round_trip() {
    let new_output = harmony_cmd().args(["identity", "new"]).output().unwrap();
    let new_stdout = String::from_utf8_lossy(&new_output.stdout);
    let new_lines: Vec<&str> = new_stdout.lines().collect();
    let pub_hex = new_lines[1].trim_start_matches("Public key:").trim();
    let priv_hex = new_lines[2].trim_start_matches("Private key:").trim();

    let message = "test message for verification";

    let sign_output = harmony_cmd()
        .args(["identity", "sign", priv_hex, message])
        .output()
        .unwrap();
    let sign_stdout = String::from_utf8_lossy(&sign_output.stdout);
    let sig_hex = sign_stdout
        .lines()
        .next()
        .unwrap()
        .trim_start_matches("Signature:")
        .trim();

    let verify_output = harmony_cmd()
        .args(["identity", "verify", pub_hex, message, sig_hex])
        .output()
        .unwrap();
    assert!(
        verify_output.status.success(),
        "valid signature should verify"
    );
    let verify_stdout = String::from_utf8_lossy(&verify_output.stdout);
    assert!(verify_stdout.contains("Valid"), "should print Valid");
}

#[test]
fn verify_rejects_corrupted_signature() {
    let new_output = harmony_cmd().args(["identity", "new"]).output().unwrap();
    let new_stdout = String::from_utf8_lossy(&new_output.stdout);
    let new_lines: Vec<&str> = new_stdout.lines().collect();
    let pub_hex = new_lines[1].trim_start_matches("Public key:").trim();
    let priv_hex = new_lines[2].trim_start_matches("Private key:").trim();

    let sign_output = harmony_cmd()
        .args(["identity", "sign", priv_hex, "original"])
        .output()
        .unwrap();
    let sign_stdout = String::from_utf8_lossy(&sign_output.stdout);
    let sig_hex = sign_stdout
        .lines()
        .next()
        .unwrap()
        .trim_start_matches("Signature:")
        .trim();

    let verify_output = harmony_cmd()
        .args(["identity", "verify", pub_hex, "tampered", sig_hex])
        .output()
        .unwrap();
    assert!(
        !verify_output.status.success(),
        "tampered message should fail verification"
    );
}

#[test]
fn identity_show_rejects_wrong_length_key() {
    let output = harmony_cmd()
        .args(["identity", "show", "aabbccdd"])
        .output()
        .expect("failed to run");
    assert!(!output.status.success(), "should fail on wrong-length key");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("expected 64 bytes"),
        "error message: {stderr}"
    );
}
