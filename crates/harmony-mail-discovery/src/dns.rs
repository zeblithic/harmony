//! DNS TXT fetch + parse for `_harmony.<domain>` records (spec §4.1, §5.3).

use async_trait::async_trait;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;

use crate::claim::{DomainRecord, MasterPubkey, SignatureAlg};

#[async_trait]
pub trait DnsClient: Send + Sync + 'static {
    /// Return all TXT strings for `name`. Empty Vec means NODATA /
    /// NOERROR-no-TXT. `Err` means transient network failure.
    async fn fetch_txt(&self, name: &str) -> Result<Vec<String>, DnsError>;
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DnsError {
    #[error("no such record (NXDOMAIN or NOERROR-no-TXT)")]
    NoRecord,
    #[error("transient DNS failure: {0}")]
    Transient(String),
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DnsFetchError {
    #[error("no _harmony TXT record found")]
    NoRecord,
    #[error("malformed TXT record: {0}")]
    Malformed(String),
    #[error("transient: {0}")]
    Transient(String),
    #[error("unsupported version byte: {0}")]
    UnsupportedVersion(u8),
    #[error("multiple v=harmony1 records present")]
    MultipleRecords,
}

pub async fn fetch_domain_record(
    dns: &dyn DnsClient,
    domain: &str,
) -> Result<DomainRecord, DnsFetchError> {
    let name = format!("_harmony.{}", domain.to_ascii_lowercase());
    let txts = match dns.fetch_txt(&name).await {
        Ok(t) => t,
        Err(DnsError::NoRecord) => return Err(DnsFetchError::NoRecord),
        Err(DnsError::Transient(m)) => return Err(DnsFetchError::Transient(m)),
    };
    parse_harmony_txts(&txts)
}

pub fn parse_harmony_txts(txts: &[String]) -> Result<DomainRecord, DnsFetchError> {
    // Filter any TXT that looks like a harmony record (starts with v=harmony),
    // regardless of version number, so unknown versions surface as
    // UnsupportedVersion rather than silently becoming NoRecord.
    let harmony_txts: Vec<&str> = txts
        .iter()
        .map(|s| s.as_str())
        .filter(|s| s.trim_start().starts_with("v=harmony"))
        .collect();
    match harmony_txts.len() {
        0 => Err(DnsFetchError::NoRecord),
        1 => parse_single_harmony_txt(harmony_txts[0]),
        _ => Err(DnsFetchError::MultipleRecords),
    }
}

fn parse_single_harmony_txt(txt: &str) -> Result<DomainRecord, DnsFetchError> {
    let mut version: Option<u8> = None;
    let mut k: Option<[u8; 32]> = None;
    let mut salt: Option<[u8; 16]> = None;
    let mut alg: Option<SignatureAlg> = None;
    for field in txt.split(';') {
        let field = field.trim();
        if field.is_empty() {
            continue;
        }
        let (name, value) = field.split_once('=').ok_or_else(|| {
            DnsFetchError::Malformed(format!("field missing '=': {field}"))
        })?;
        match name.trim() {
            "v" => {
                if value.trim() != "harmony1" {
                    return Err(DnsFetchError::UnsupportedVersion(0));
                }
                version = Some(1);
            }
            "k" => {
                let bytes = URL_SAFE_NO_PAD.decode(value.trim()).map_err(|e| {
                    DnsFetchError::Malformed(format!("k base64url: {e}"))
                })?;
                if bytes.len() != 32 {
                    return Err(DnsFetchError::Malformed(format!(
                        "k length {} != 32",
                        bytes.len()
                    )));
                }
                let mut arr = [0u8; 32];
                arr.copy_from_slice(&bytes);
                k = Some(arr);
            }
            "salt" => {
                let bytes = URL_SAFE_NO_PAD.decode(value.trim()).map_err(|e| {
                    DnsFetchError::Malformed(format!("salt base64url: {e}"))
                })?;
                if bytes.len() != 16 {
                    return Err(DnsFetchError::Malformed(format!(
                        "salt length {} != 16",
                        bytes.len()
                    )));
                }
                let mut arr = [0u8; 16];
                arr.copy_from_slice(&bytes);
                salt = Some(arr);
            }
            "alg" => match value.trim() {
                "ed25519" => alg = Some(SignatureAlg::Ed25519),
                other => {
                    return Err(DnsFetchError::Malformed(format!("unknown alg={other}")));
                }
            },
            _ => {} // unknown fields ignored (forward-compat)
        }
    }
    Ok(DomainRecord {
        version: version
            .ok_or_else(|| DnsFetchError::Malformed("missing v".into()))?,
        master_pubkey: MasterPubkey::Ed25519(
            k.ok_or_else(|| DnsFetchError::Malformed("missing k".into()))?,
        ),
        domain_salt: salt
            .ok_or_else(|| DnsFetchError::Malformed("missing salt".into()))?,
        alg: alg.ok_or_else(|| DnsFetchError::Malformed("missing alg".into()))?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_txt() -> String {
        let k = URL_SAFE_NO_PAD.encode([0xabu8; 32]);
        let salt = URL_SAFE_NO_PAD.encode([0xcd; 16]);
        format!("v=harmony1; k={k}; salt={salt}; alg=ed25519")
    }

    #[test]
    fn parses_well_formed_single_txt() {
        let rec = parse_harmony_txts(&[sample_txt()]).expect("parse");
        assert_eq!(rec.version, 1);
        assert_eq!(rec.alg, SignatureAlg::Ed25519);
        assert_eq!(rec.domain_salt, [0xcd; 16]);
    }

    #[test]
    fn rejects_multiple_harmony_records() {
        let err = parse_harmony_txts(&[sample_txt(), sample_txt()]).unwrap_err();
        assert_eq!(err, DnsFetchError::MultipleRecords);
    }

    #[test]
    fn ignores_unknown_fields_for_forward_compat() {
        let k = URL_SAFE_NO_PAD.encode([0u8; 32]);
        let salt = URL_SAFE_NO_PAD.encode([0u8; 16]);
        let txt = format!("v=harmony1; k={k}; salt={salt}; alg=ed25519; future=xyz");
        parse_harmony_txts(&[txt]).expect("must accept unknown field");
    }

    #[test]
    fn rejects_unknown_v_version() {
        let txt = "v=harmony2; k=AA; salt=AA; alg=ed25519".to_string();
        let err = parse_harmony_txts(&[txt]).unwrap_err();
        assert!(matches!(err, DnsFetchError::UnsupportedVersion(_)), "{err:?}");
    }

    #[test]
    fn rejects_missing_required_field() {
        let k = URL_SAFE_NO_PAD.encode([0u8; 32]);
        let txt = format!("v=harmony1; k={k}; alg=ed25519"); // no salt
        let err = parse_harmony_txts(&[txt]).unwrap_err();
        assert!(matches!(err, DnsFetchError::Malformed(_)), "{err:?}");
    }

    #[test]
    fn rejects_malformed_base64_k() {
        let salt = URL_SAFE_NO_PAD.encode([0u8; 16]);
        let txt = format!("v=harmony1; k=!!!bad!!!; salt={salt}; alg=ed25519");
        let err = parse_harmony_txts(&[txt]).unwrap_err();
        assert!(matches!(err, DnsFetchError::Malformed(_)), "{err:?}");
    }

    #[test]
    fn ignores_non_harmony_txt_records() {
        let txt_a = "spf1 include:_spf.google.com".to_string();
        let txt_b = sample_txt();
        let rec = parse_harmony_txts(&[txt_a, txt_b]).expect("parse");
        assert_eq!(rec.version, 1);
    }
}
