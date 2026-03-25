//! End-to-end integration test with a tiny safetensors file.

use half::f16;
use harmony_content::cid::{ContentFlags, ContentId};

/// Create a minimal safetensors file with a single f16 tensor.
fn create_test_safetensors(name: &str, rows: usize, cols: usize) -> Vec<u8> {
    let mut f16_bytes = Vec::with_capacity(rows * cols * 2);
    for r in 0..rows {
        for c in 0..cols {
            let val = f16::from_f32((r * cols + c) as f32);
            f16_bytes.extend_from_slice(&val.to_le_bytes());
        }
    }

    let tensor_info = serde_json::json!({
        name: {
            "dtype": "F16",
            "shape": [rows, cols],
            "data_offsets": [0, f16_bytes.len()]
        }
    });
    let header_json = serde_json::to_string(&tensor_info).unwrap();
    let header_len = header_json.len() as u64;

    let mut buf = Vec::new();
    buf.extend_from_slice(&header_len.to_le_bytes());
    buf.extend_from_slice(header_json.as_bytes());
    buf.extend_from_slice(&f16_bytes);
    buf
}

#[test]
fn end_to_end_local_dir() {
    let dir = tempfile::tempdir().unwrap();
    let local_dir = dir.path().join("cache");
    let input_path = dir.path().join("test.safetensors");

    // 6 entries, dim=4, shard_size=3 → 2 shards
    let safetensors_bytes = create_test_safetensors("engram.weight", 6, 4);
    std::fs::write(&input_path, &safetensors_bytes).unwrap();

    let config_path = dir.path().join("config.toml");
    std::fs::write(
        &config_path,
        r#"
        version = "v1"
        shard_size = 3
        hash_seeds = [42, 99]
        tensor = "engram.weight"
        "#,
    )
    .unwrap();

    // Parse the safetensors to get raw tensor bytes for verification.
    let tensors = safetensors::SafeTensors::deserialize(&safetensors_bytes).unwrap();
    let tensor = tensors.tensor("engram.weight").unwrap();
    let tensor_bytes = tensor.data();
    let vector_bytes = 4 * 2; // dim=4, f16=2 bytes
    let shard_bytes = 3 * vector_bytes; // shard_size=3

    // Manually compute expected shards.
    let shard0_data = &tensor_bytes[0..shard_bytes];
    let shard1_data = &tensor_bytes[shard_bytes..2 * shard_bytes];
    let cid0 = ContentId::for_book(shard0_data, ContentFlags::default()).unwrap();
    let cid1 = ContentId::for_book(shard1_data, ContentFlags::default()).unwrap();

    // Run the CLI (local-dir only, no S3).
    let status = std::process::Command::new(env!("CARGO_BIN_EXE_harmony-ingest"))
        .args([
            "engram",
            "--config",
            config_path.to_str().unwrap(),
            "--input",
            input_path.to_str().unwrap(),
            "--local-dir",
            local_dir.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(status.success(), "harmony-ingest failed");

    // Verify shard files exist with correct content.
    let hex0 = hex::encode(cid0.to_bytes());
    let path0 = local_dir.join("book").join(&hex0[..2]).join(&hex0);
    assert!(path0.exists(), "shard 0 not found at {}", path0.display());
    assert_eq!(std::fs::read(&path0).unwrap(), shard0_data);

    let hex1 = hex::encode(cid1.to_bytes());
    let path1 = local_dir.join("book").join(&hex1[..2]).join(&hex1);
    assert!(path1.exists(), "shard 1 not found at {}", path1.display());
    assert_eq!(std::fs::read(&path1).unwrap(), shard1_data);

    // Verify journal has 2 entries.
    let journal_path = input_path.with_extension("journal");
    let journal_data = std::fs::read(&journal_path).unwrap();
    assert_eq!(journal_data.len(), 64); // 2 × 32 bytes
    assert_eq!(&journal_data[0..32], &cid0.to_bytes());
    assert_eq!(&journal_data[32..64], &cid1.to_bytes());

    // Verify manifest DAG books exist in local dir.
    let mut book_count = 0;
    for entry in walkdir::WalkDir::new(local_dir.join("book"))
        .into_iter()
        .filter_map(|e| e.ok())
    {
        if entry.file_type().is_file() {
            book_count += 1;
        }
    }
    // 2 shards + at least 1 manifest book
    assert!(book_count >= 3, "expected >= 3 books, found {book_count}");
}
