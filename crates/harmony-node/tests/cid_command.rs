use std::io::Write;
use std::process::Command;

fn harmony_bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_harmony"))
}

#[test]
fn cid_from_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.bin");
    std::fs::write(&path, b"hello harmony cid test").unwrap();

    let output = harmony_bin()
        .args(["cid", "--file", path.to_str().unwrap()])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let hex_str = String::from_utf8(output.stdout).unwrap().trim().to_string();
    assert_eq!(
        hex_str.len(),
        64,
        "CID should be 64 hex chars, got {}",
        hex_str.len()
    );
    assert!(hex::decode(&hex_str).is_ok());
}

#[test]
fn cid_from_stdin() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.bin");
    let data = b"hello harmony cid test";
    std::fs::write(&path, data).unwrap();

    let file_output = harmony_bin()
        .args(["cid", "--file", path.to_str().unwrap()])
        .output()
        .unwrap();
    let file_cid = String::from_utf8(file_output.stdout)
        .unwrap()
        .trim()
        .to_string();

    let mut child = harmony_bin()
        .args(["cid"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();

    child.stdin.take().unwrap().write_all(data).unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdin_cid = String::from_utf8(output.stdout).unwrap().trim().to_string();

    assert_eq!(
        file_cid, stdin_cid,
        "file and stdin should produce same CID"
    );
}

#[test]
fn cid_deterministic() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.bin");
    std::fs::write(&path, b"determinism check").unwrap();

    let cid1 = harmony_bin()
        .args(["cid", "--file", path.to_str().unwrap()])
        .output()
        .unwrap();
    let cid2 = harmony_bin()
        .args(["cid", "--file", path.to_str().unwrap()])
        .output()
        .unwrap();

    assert_eq!(cid1.stdout, cid2.stdout, "same input must produce same CID");
}

#[test]
fn cid_empty_input_fails() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty.bin");
    std::fs::write(&path, b"").unwrap();

    let output = harmony_bin()
        .args(["cid", "--file", path.to_str().unwrap()])
        .output()
        .unwrap();

    assert!(!output.status.success(), "empty input should fail");
}

#[test]
fn cid_large_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("large.bin");
    // MAX_PAYLOAD_SIZE is 0xF_FFFF (1,048,575). Write one byte more.
    let data = vec![0xABu8; 0xF_FFFF + 1];
    std::fs::write(&path, &data).unwrap();

    let output = harmony_bin()
        .args(["cid", "--file", path.to_str().unwrap()])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let hex_str = String::from_utf8(output.stdout).unwrap().trim().to_string();
    assert_eq!(
        hex_str.len(),
        64,
        "CID should be 64 hex chars for large files, got {}",
        hex_str.len()
    );
    assert!(hex::decode(&hex_str).is_ok());
}
