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
