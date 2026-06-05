//! pkarr wire-format codec (ZEB-382).
//!
//! Encodes a harmony `PkarrRoutingRecord` as a real pkarr `SignedPacket`
//! relay payload (z-base-32 key + BEP44 big-endian + DNS-packet `v`) and back,
//! while `relay.rs` keeps doing the actual HTTP. The record's canonical CBOR
//! rides inside one `_r` TXT record as base64url.

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
