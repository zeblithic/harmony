use std::path::{Path, PathBuf};

use harmony_identity::{PqPrivateIdentity, PrivateIdentity};
use zeroize::Zeroizing;

const VERSION: u8 = 0x01;
const PQ_KEY_LEN: usize = 96;
const ED25519_KEY_LEN: usize = 64;
const FILE_LEN: usize = 1 + PQ_KEY_LEN + ED25519_KEY_LEN; // 161

pub struct NodeIdentity {
    pub pq: PqPrivateIdentity,
    pub ed25519: PrivateIdentity,
}

impl std::fmt::Debug for NodeIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NodeIdentity")
            .field("pq_address", &self.pq.public_identity().address_hash)
            .field("ed25519_address", &self.ed25519.public_identity().address_hash)
            .finish()
    }
}

pub fn resolve_path(cli_override: Option<&Path>) -> Result<PathBuf, String> {
    if let Some(p) = cli_override {
        return Ok(p.to_path_buf());
    }
    let home = std::env::var("HOME")
        .map_err(|_| "Cannot determine identity file path: $HOME not set. Use --identity-file.".to_string())?;
    Ok(PathBuf::from(home).join(".harmony").join("identity.key"))
}

pub fn load(path: &Path) -> Result<NodeIdentity, String> {
    let buf = Zeroizing::new(
        std::fs::read(path).map_err(|e| format!("Failed to read {}: {e}", path.display()))?,
    );
    if buf.len() != FILE_LEN {
        return Err(format!("Corrupt identity file: expected {FILE_LEN} bytes, got {}", buf.len()));
    }
    if buf[0] != VERSION {
        return Err(format!("Unsupported identity file version: {:#04x}", buf[0]));
    }
    let pq = PqPrivateIdentity::from_private_bytes(&buf[1..1 + PQ_KEY_LEN])
        .map_err(|e| format!("Corrupt PQ identity in key file: {e}"))?;
    let ed25519 = PrivateIdentity::from_private_bytes(&buf[1 + PQ_KEY_LEN..])
        .map_err(|e| format!("Corrupt Ed25519 identity in key file: {e}"))?;
    #[cfg(unix)]
    warn_permissions(path);
    Ok(NodeIdentity { pq, ed25519 })
}

pub fn save(path: &Path, identity: &NodeIdentity) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create {}: {e}", parent.display()))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(parent, std::fs::Permissions::from_mode(0o700));
        }
    }
    let pq_bytes = Zeroizing::new(identity.pq.to_private_bytes());
    let ed_bytes = identity.ed25519.to_private_bytes();
    let mut buf = Zeroizing::new(Vec::with_capacity(FILE_LEN));
    buf.push(VERSION);
    buf.extend_from_slice(&pq_bytes);
    buf.extend_from_slice(&ed_bytes);

    let tmp_path = path.with_extension("key.tmp");
    std::fs::write(&tmp_path, &*buf)
        .map_err(|e| format!("Failed to write {}: {e}", tmp_path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&tmp_path, std::fs::Permissions::from_mode(0o600))
            .map_err(|e| format!("Failed to set permissions on {}: {e}", tmp_path.display()))?;
    }
    let f = std::fs::File::open(&tmp_path)
        .map_err(|e| format!("Failed to open {}: {e}", tmp_path.display()))?;
    f.sync_all()
        .map_err(|e| format!("Failed to fsync {}: {e}", tmp_path.display()))?;
    std::fs::rename(&tmp_path, path)
        .map_err(|e| format!("Failed to rename {} → {}: {e}", tmp_path.display(), path.display()))?;
    Ok(())
}

pub fn load_or_generate(path: &Path) -> Result<NodeIdentity, String> {
    if path.exists() {
        return load(path);
    }
    let pq = PqPrivateIdentity::generate(&mut rand::rngs::OsRng);
    let ed25519 = PrivateIdentity::generate(&mut rand::rngs::OsRng);
    let identity = NodeIdentity { pq, ed25519 };
    save(path, &identity)?;
    Ok(identity)
}

#[cfg(unix)]
fn warn_permissions(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    if let Ok(meta) = std::fs::metadata(path) {
        let mode = meta.permissions().mode() & 0o777;
        if mode & 0o077 != 0 {
            eprintln!("Warning: {} has open permissions ({:#05o}), should be 0600", path.display(), mode);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn round_trip_save_and_load() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("identity.key");
        let pq = PqPrivateIdentity::generate(&mut rand::rngs::OsRng);
        let ed = PrivateIdentity::generate(&mut rand::rngs::OsRng);
        let pq_addr = pq.public_identity().address_hash;
        let ed_addr = ed.public_identity().address_hash;
        let identity = NodeIdentity { pq, ed25519: ed };
        save(&path, &identity).unwrap();
        let loaded = load(&path).unwrap();
        assert_eq!(loaded.pq.public_identity().address_hash, pq_addr);
        assert_eq!(loaded.ed25519.public_identity().address_hash, ed_addr);
    }

    #[test]
    fn load_wrong_version_fails() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("identity.key");
        let mut data = vec![0x99u8];
        data.extend_from_slice(&[0u8; PQ_KEY_LEN + ED25519_KEY_LEN]);
        std::fs::write(&path, &data).unwrap();
        let err = load(&path).unwrap_err();
        assert!(err.contains("version"), "expected version error, got: {err}");
    }

    #[test]
    fn load_truncated_file_fails() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("identity.key");
        std::fs::write(&path, &[VERSION; 10]).unwrap();
        let err = load(&path).unwrap_err();
        assert!(err.contains("161"), "expected length error, got: {err}");
    }

    #[test]
    fn load_empty_file_fails() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("identity.key");
        std::fs::write(&path, &[]).unwrap();
        let err = load(&path).unwrap_err();
        assert!(err.contains("161"), "expected length error, got: {err}");
    }

    #[test]
    fn load_oversized_file_fails() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("identity.key");
        std::fs::write(&path, &[VERSION; 200]).unwrap();
        let err = load(&path).unwrap_err();
        assert!(err.contains("161"), "expected length error, got: {err}");
    }

    #[test]
    fn load_or_generate_creates_new_file() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("subdir").join("identity.key");
        assert!(!path.exists());
        let identity = load_or_generate(&path).unwrap();
        assert!(path.exists());
        let reloaded = load(&path).unwrap();
        assert_eq!(identity.pq.public_identity().address_hash, reloaded.pq.public_identity().address_hash);
        assert_eq!(identity.ed25519.public_identity().address_hash, reloaded.ed25519.public_identity().address_hash);
    }

    #[test]
    fn load_or_generate_loads_existing() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("identity.key");
        let first = load_or_generate(&path).unwrap();
        let addr = first.pq.public_identity().address_hash;
        let second = load_or_generate(&path).unwrap();
        assert_eq!(second.pq.public_identity().address_hash, addr);
    }

    #[test]
    fn resolve_path_uses_override() {
        let p = resolve_path(Some(Path::new("/custom/path.key"))).unwrap();
        assert_eq!(p, PathBuf::from("/custom/path.key"));
    }

    #[test]
    fn resolve_path_uses_home() {
        if std::env::var("HOME").is_ok() {
            let p = resolve_path(None).unwrap();
            assert!(p.to_str().unwrap().contains(".harmony"));
            assert!(p.to_str().unwrap().ends_with("identity.key"));
        }
    }

    #[test]
    fn file_length_constant_is_correct() {
        assert_eq!(FILE_LEN, 161);
        assert_eq!(1 + PQ_KEY_LEN + ED25519_KEY_LEN, 161);
    }

    #[cfg(unix)]
    #[test]
    fn save_creates_directory_and_file_with_correct_permissions() {
        use std::os::unix::fs::PermissionsExt;
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("nested").join("identity.key");
        let identity = NodeIdentity {
            pq: PqPrivateIdentity::generate(&mut rand::rngs::OsRng),
            ed25519: PrivateIdentity::generate(&mut rand::rngs::OsRng),
        };
        save(&path, &identity).unwrap();
        let dir_perms = std::fs::metadata(path.parent().unwrap()).unwrap().permissions().mode() & 0o777;
        assert_eq!(dir_perms, 0o700, "directory should be 0700");
        let file_perms = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(file_perms, 0o600, "file should be 0600");
    }
}
