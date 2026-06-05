//! pkarr wire-format codec (ZEB-382).
//!
//! Encodes a harmony `PkarrRoutingRecord` as a real pkarr `SignedPacket`
//! relay payload (z-base-32 key + BEP44 big-endian + DNS-packet `v`) and back,
//! while `relay.rs` keeps doing the actual HTTP. The record's canonical CBOR
//! rides inside one `_r` TXT record as base64url.

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine as _;

use crate::error::PkarrError;
use crate::record::PkarrRoutingRecord;
use ed25519_dalek::{SigningKey, VerifyingKey};
use pkarr::dns::rdata::{RData, TXT};
use pkarr::dns::{CharacterString, Name};
use pkarr::errors::SignedPacketBuildError;
use pkarr::{Keypair, PublicKey, SignedPacket};

/// DNS label carrying the routing record's base64url payload.
const RECORD_LABEL: &str = "_r";
/// TTL (seconds) on the TXT record. Not load-bearing — harmony freshness is
/// governed by PkarrRoutingRecord::verify_skew + epoch rotation, not DNS TTL.
const RECORD_TTL: u32 = 300;

/// Encode a routing record's canonical CBOR as a base64url-unpadded string.
pub(crate) fn routing_record_to_txt_string(
    record: &PkarrRoutingRecord,
) -> Result<String, PkarrError> {
    let cbor = record.to_canonical_cbor()?;
    Ok(URL_SAFE_NO_PAD.encode(&cbor))
}

/// Decode a base64url-unpadded string back into a routing record.
pub(crate) fn txt_string_to_routing_record(s: &str) -> Result<PkarrRoutingRecord, PkarrError> {
    let cbor = URL_SAFE_NO_PAD
        .decode(s.as_bytes())
        .map_err(|_| PkarrError::InvalidRecord)?;
    PkarrRoutingRecord::from_canonical_cbor(&cbor)
}

/// Build the `(z-base-32 key, relay PUT payload)` for `record` signed under
/// the ephemeral `signing_key`. Returns `RecordTooLarge` if the packet would
/// exceed `SignedPacket::MAX_BYTES`.
pub(crate) fn build_relay_payload(
    signing_key: &SigningKey,
    record: &PkarrRoutingRecord,
) -> Result<(String, Vec<u8>), PkarrError> {
    let keypair = Keypair::from_secret_key(&signing_key.to_bytes());
    let b64 = routing_record_to_txt_string(record)?;

    // Split the base64 payload into <=255-byte DNS character-strings.
    let mut txt = TXT::new();
    for chunk in b64.as_bytes().chunks(255) {
        let cs = CharacterString::new(chunk)
            .map_err(|_| PkarrError::SerializeError("txt char-string"))?;
        txt.add_char_string(cs);
    }

    let name = Name::new(RECORD_LABEL).map_err(|_| PkarrError::SerializeError("txt name"))?;
    // Classify the build error honestly: an oversized routing_blob trips the
    // 1000-byte DNS-packet budget (PacketTooLarge → RecordTooLarge); any other
    // failure is a genuine encode bug, surfaced as SerializeError rather than
    // masquerading as a size problem.
    let signed = SignedPacket::builder()
        .txt(name, txt, RECORD_TTL)
        .sign(&keypair)
        .map_err(|e| match e {
            SignedPacketBuildError::PacketTooLarge(_) => PkarrError::RecordTooLarge,
            SignedPacketBuildError::FailedToWrite(_) => {
                PkarrError::SerializeError("pkarr signed-packet build")
            }
        })?;

    Ok((keypair.to_z32(), signed.to_relay_payload().to_vec()))
}

/// Parse + outer-verify a relay GET payload back into a routing record.
/// Verifies the BEP44 (ephemeral-key) signature against `expected_pk`; does
/// NOT verify the inner identity signature — the caller does that.
pub(crate) fn parse_relay_payload(
    expected_pk: &[u8; 32],
    payload: &[u8],
) -> Result<PkarrRoutingRecord, PkarrError> {
    let pk = PublicKey::try_from(expected_pk).map_err(|_| PkarrError::InvalidRecord)?;
    let bytes = bytes::Bytes::copy_from_slice(payload);
    let signed = SignedPacket::from_relay_payload(&pk, &bytes)
        .map_err(|_| PkarrError::OuterSignatureInvalid)?;
    let txt = read_record_txt(&signed)?;
    txt_string_to_routing_record(&txt)
}

/// z-base-32 string for an ed25519 verifying key (the relay GET URL key).
pub(crate) fn z32_for_verifying_key(pk: &VerifyingKey) -> Result<String, PkarrError> {
    let pk = PublicKey::try_from(&pk.to_bytes()).map_err(|_| PkarrError::InvalidRecord)?;
    Ok(pk.to_z32())
}

/// Read the concatenated base64 string from the single `_r` TXT record.
///
/// The wire format defines exactly one `_r` record; a packet carrying more
/// than one is rejected as malformed rather than silently resolving to the
/// first (which would make payload interpretation order-dependent).
fn read_record_txt(signed: &SignedPacket) -> Result<String, PkarrError> {
    let mut found: Option<String> = None;
    for rr in signed.resource_records(RECORD_LABEL) {
        // Any non-TXT `_r` record violates the contract (exactly one `_r` TXT).
        let txt = match &rr.rdata {
            RData::TXT(txt) => txt,
            _ => return Err(PkarrError::InvalidRecord),
        };
        let value = String::try_from(txt.clone()).map_err(|_| PkarrError::InvalidRecord)?;
        if found.replace(value).is_some() {
            return Err(PkarrError::InvalidRecord);
        }
    }
    found.ok_or(PkarrError::InvalidRecord)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;
    use rand::rngs::OsRng;

    fn sample_record() -> crate::record::PkarrRoutingRecord {
        let sk = SigningKey::generate(&mut OsRng);
        let mut id_pub = [0u8; 64];
        id_pub[32..].copy_from_slice(&sk.verifying_key().to_bytes());
        crate::record::PkarrRoutingRecord::sign_new(
            vec![0xCDu8; 120], // routing_blob big enough to force >255 base64
            id_pub,
            1_000_000,
            &sk,
        )
        .expect("sign record")
    }

    #[test]
    fn parse_rejects_duplicate_r_txt() {
        use pkarr::dns::rdata::TXT;
        use pkarr::dns::{CharacterString, Name};
        use pkarr::{Keypair, SignedPacket};

        // A packet carrying TWO `_r` TXT records is malformed for this format.
        let kp = Keypair::from_secret_key(&[8u8; 32]);
        let mut txt1 = TXT::new();
        txt1.add_char_string(CharacterString::new(b"AAAA").unwrap());
        let mut txt2 = TXT::new();
        txt2.add_char_string(CharacterString::new(b"BBBB").unwrap());
        let signed = SignedPacket::builder()
            .txt(Name::new("_r").unwrap(), txt1, 300)
            .txt(Name::new("_r").unwrap(), txt2, 300)
            .sign(&kp)
            .expect("sign");
        let payload = signed.to_relay_payload().to_vec();
        assert_eq!(
            parse_relay_payload(&kp.public_key().to_bytes(), &payload),
            Err(PkarrError::InvalidRecord)
        );
    }

    #[test]
    fn txt_string_round_trips() {
        let rec = sample_record();
        let s = routing_record_to_txt_string(&rec).expect("encode");
        assert!(
            s.len() > 255,
            "encoded payload should exceed one char-string"
        );
        let back = txt_string_to_routing_record(&s).expect("decode");
        assert_eq!(rec, back);
    }

    #[test]
    fn txt_string_rejects_garbage() {
        assert!(txt_string_to_routing_record("!!!not-base64!!!").is_err());
    }

    #[test]
    fn build_produces_z32_and_payload() {
        let sk = SigningKey::generate(&mut OsRng);
        let rec = sample_record();
        let (z32, payload) = build_relay_payload(&sk, &rec).expect("build");
        assert_eq!(z32.len(), 52, "z-base-32 ed25519 key is 52 chars");
        assert!(!payload.is_empty());
        assert!((payload.len() as u64) < pkarr::SignedPacket::MAX_BYTES);
    }

    #[test]
    fn build_rejects_oversize_record() {
        let sk = SigningKey::generate(&mut OsRng);
        let mut id_pub = [0u8; 64];
        id_pub[32..].copy_from_slice(&sk.verifying_key().to_bytes());
        // routing_blob far over the ~1000-byte v budget once base64'd.
        let big =
            crate::record::PkarrRoutingRecord::sign_new(vec![0u8; 3000], id_pub, 1_000_000, &sk)
                .expect("sign");
        assert_eq!(
            build_relay_payload(&sk, &big),
            Err(PkarrError::RecordTooLarge)
        );
    }

    #[test]
    fn build_then_parse_round_trips() {
        let sk = SigningKey::generate(&mut OsRng);
        let rec = sample_record();
        let (_z32, payload) = build_relay_payload(&sk, &rec).expect("build");
        let parsed = parse_relay_payload(&sk.verifying_key().to_bytes(), &payload).expect("parse");
        assert_eq!(parsed, rec);
    }

    #[test]
    fn parse_rejects_tampered_payload() {
        let sk = SigningKey::generate(&mut OsRng);
        let rec = sample_record();
        let (_z32, mut payload) = build_relay_payload(&sk, &rec).expect("build");
        payload[0] ^= 0xFF; // corrupt the signature
        assert_eq!(
            parse_relay_payload(&sk.verifying_key().to_bytes(), &payload),
            Err(PkarrError::OuterSignatureInvalid)
        );
    }

    #[test]
    fn parse_rejects_wrong_pubkey() {
        let sk = SigningKey::generate(&mut OsRng);
        let other = SigningKey::generate(&mut OsRng);
        let rec = sample_record();
        let (_z32, payload) = build_relay_payload(&sk, &rec).expect("build");
        assert_eq!(
            parse_relay_payload(&other.verifying_key().to_bytes(), &payload),
            Err(PkarrError::OuterSignatureInvalid)
        );
    }

    #[test]
    fn z32_matches_build() {
        let sk = SigningKey::generate(&mut OsRng);
        let rec = sample_record();
        let (z32, _payload) = build_relay_payload(&sk, &rec).expect("build");
        assert_eq!(z32_for_verifying_key(&sk.verifying_key()).unwrap(), z32);
    }

    /// Live round-trip against the public relay. Ignored by default; run with:
    ///   HARMONY_PKARR_LIVE_RELAY=1 cargo nextest run -p harmony-pkarr \
    ///     --run-ignored all wire::tests::live_relay_round_trip
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore = "hits relay.pkarr.org; set HARMONY_PKARR_LIVE_RELAY=1 to run"]
    async fn live_relay_round_trip() {
        if std::env::var("HARMONY_PKARR_LIVE_RELAY").is_err() {
            return;
        }
        use crate::relay::{RelayClient, RelayPool};

        let relay = RelayClient::new(RelayPool::new(vec!["https://relay.pkarr.org".to_string()]));
        let ephemeral = SigningKey::generate(&mut OsRng);
        let rec = sample_record();
        let (z32, payload) = build_relay_payload(&ephemeral, &rec).expect("build");

        relay
            .put(&z32, &payload)
            .await
            .expect("publish to live relay");
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;

        let got = relay
            .get(&z32)
            .await
            .expect("get from live relay")
            .expect("record present on relay");
        let parsed =
            parse_relay_payload(&ephemeral.verifying_key().to_bytes(), &got).expect("parse");
        assert_eq!(parsed, rec);
    }
}

#[cfg(test)]
mod spike_tests {
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use base64::Engine as _;
    use pkarr::dns::rdata::RData;
    use pkarr::dns::rdata::TXT;
    use pkarr::dns::CharacterString;
    use pkarr::dns::Name;
    use pkarr::{Keypair, PublicKey, SignedPacket};

    #[test]
    fn pkarr_codec_round_trips_in_memory() {
        let seed = [7u8; 32];
        let kp = Keypair::from_secret_key(&seed);

        // 450 bytes -> base64 > 255 -> requires multiple DNS character-strings.
        let raw = vec![0xABu8; 450];
        let b64 = URL_SAFE_NO_PAD.encode(&raw);
        assert!(
            b64.len() > 255,
            "payload must exercise multi-char-string path"
        );

        let mut txt = TXT::new();
        for chunk in b64.as_bytes().chunks(255) {
            let cs = CharacterString::new(chunk).expect("char-string <=255");
            // add_char_string returns () — no .expect() here.
            txt.add_char_string(cs);
        }

        let signed = SignedPacket::builder()
            .txt(Name::new("_r").expect("name"), txt, 300)
            .sign(&kp)
            .expect("sign");

        let payload = signed.to_relay_payload();

        let pk = PublicKey::try_from(&kp.public_key().to_bytes()).expect("pk");
        let parsed = SignedPacket::from_relay_payload(&pk, &payload).expect("verify");

        let mut found = None;
        for rr in parsed.resource_records("_r") {
            if let RData::TXT(t) = &rr.rdata {
                // String::try_from consumes a TXT — clone first.
                found = Some(String::try_from(t.clone()).expect("txt string"));
            }
        }
        assert_eq!(found.as_deref(), Some(b64.as_str()));

        // Key-consistency invariant: pkarr Keypair pubkey == ed25519_dalek's.
        let sk = ed25519_dalek::SigningKey::from_bytes(&seed);
        assert_eq!(kp.public_key().to_bytes(), sk.verifying_key().to_bytes());
    }
}
