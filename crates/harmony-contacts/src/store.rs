use alloc::vec::Vec;
use harmony_identity::IdentityHash;
use hashbrown::HashMap;
use serde::{Deserialize, Serialize};

use crate::contact::{Contact, PeeringPriority};
use crate::error::ContactError;

const FORMAT_VERSION: u8 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContactStore {
    contacts: HashMap<IdentityHash, Contact>,
}

impl ContactStore {
    pub fn new() -> Self {
        Self {
            contacts: HashMap::new(),
        }
    }

    pub fn add(&mut self, contact: Contact) -> Result<(), ContactError> {
        if self.contacts.contains_key(&contact.identity_hash) {
            return Err(ContactError::AlreadyExists(contact.identity_hash));
        }
        self.contacts.insert(contact.identity_hash, contact);
        Ok(())
    }

    pub fn remove(&mut self, id: &IdentityHash) -> Option<Contact> {
        self.contacts.remove(id)
    }

    pub fn get(&self, id: &IdentityHash) -> Option<&Contact> {
        self.contacts.get(id)
    }

    pub fn get_mut(&mut self, id: &IdentityHash) -> Option<&mut Contact> {
        self.contacts.get_mut(id)
    }

    pub fn contains(&self, id: &IdentityHash) -> bool {
        self.contacts.contains_key(id)
    }

    pub fn update_last_seen(&mut self, id: &IdentityHash, timestamp: u64) {
        if let Some(contact) = self.contacts.get_mut(id) {
            contact.last_seen = Some(timestamp);
        }
    }

    pub fn peers_with_peering_enabled(&self) -> impl Iterator<Item = &Contact> {
        self.contacts.values().filter(|c| c.peering.enabled)
    }

    pub fn peers_by_priority(&self, priority: PeeringPriority) -> impl Iterator<Item = &Contact> {
        self.contacts
            .values()
            .filter(move |c| c.peering.enabled && c.peering.priority == priority)
    }

    pub fn len(&self) -> usize {
        self.contacts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.contacts.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&IdentityHash, &Contact)> {
        self.contacts.iter()
    }

    pub fn serialize(&self) -> Result<Vec<u8>, ContactError> {
        let mut buf = Vec::new();
        buf.push(FORMAT_VERSION);
        let inner = postcard::to_allocvec(self)
            .map_err(|_| ContactError::SerializeError("postcard encode failed"))?;
        buf.extend_from_slice(&inner);
        Ok(buf)
    }

    pub fn deserialize(data: &[u8]) -> Result<Self, ContactError> {
        if data.is_empty() {
            return Err(ContactError::DeserializeError("empty data"));
        }
        if data[0] != FORMAT_VERSION {
            return Err(ContactError::DeserializeError("unsupported format version"));
        }
        postcard::from_bytes(&data[1..])
            .map_err(|_| ContactError::DeserializeError("postcard decode failed"))
    }
}

impl Default for ContactStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contact::{PeeringPolicy, PeeringPriority};

    fn make_contact(id_byte: u8, enabled: bool, priority: PeeringPriority) -> Contact {
        Contact {
            identity_hash: [id_byte; 16],
            display_name: None,
            peering: PeeringPolicy { enabled, priority },
            added_at: 1710000000,
            last_seen: None,
            notes: None,
        }
    }

    #[test]
    fn add_and_get() {
        let mut store = ContactStore::new();
        let contact = make_contact(0xAA, true, PeeringPriority::Normal);
        store.add(contact.clone()).unwrap();
        assert_eq!(store.len(), 1);
        assert!(store.contains(&[0xAA; 16]));
        assert_eq!(store.get(&[0xAA; 16]).unwrap().identity_hash, [0xAA; 16]);
    }

    #[test]
    fn add_duplicate_fails() {
        let mut store = ContactStore::new();
        store
            .add(make_contact(0xBB, true, PeeringPriority::Normal))
            .unwrap();
        let result = store.add(make_contact(0xBB, false, PeeringPriority::Low));
        assert!(matches!(result, Err(ContactError::AlreadyExists(_))));
    }

    #[test]
    fn remove_returns_contact() {
        let mut store = ContactStore::new();
        store
            .add(make_contact(0xCC, true, PeeringPriority::High))
            .unwrap();
        let removed = store.remove(&[0xCC; 16]);
        assert!(removed.is_some());
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn remove_nonexistent_returns_none() {
        let mut store = ContactStore::new();
        assert!(store.remove(&[0xFF; 16]).is_none());
    }

    #[test]
    fn update_last_seen() {
        let mut store = ContactStore::new();
        store
            .add(make_contact(0xDD, true, PeeringPriority::Normal))
            .unwrap();
        store.update_last_seen(&[0xDD; 16], 999);
        assert_eq!(store.get(&[0xDD; 16]).unwrap().last_seen, Some(999));
    }

    #[test]
    fn peers_with_peering_enabled() {
        let mut store = ContactStore::new();
        store
            .add(make_contact(0x01, true, PeeringPriority::Normal))
            .unwrap();
        store
            .add(make_contact(0x02, false, PeeringPriority::Normal))
            .unwrap();
        store
            .add(make_contact(0x03, true, PeeringPriority::High))
            .unwrap();
        let enabled: Vec<_> = store.peers_with_peering_enabled().collect();
        assert_eq!(enabled.len(), 2);
    }

    #[test]
    fn peers_by_priority() {
        let mut store = ContactStore::new();
        store
            .add(make_contact(0x01, true, PeeringPriority::Low))
            .unwrap();
        store
            .add(make_contact(0x02, true, PeeringPriority::High))
            .unwrap();
        store
            .add(make_contact(0x03, true, PeeringPriority::High))
            .unwrap();
        let high: Vec<_> = store.peers_by_priority(PeeringPriority::High).collect();
        assert_eq!(high.len(), 2);
        let low: Vec<_> = store.peers_by_priority(PeeringPriority::Low).collect();
        assert_eq!(low.len(), 1);
    }

    #[test]
    fn empty_store() {
        let store = ContactStore::new();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
        assert!(store.get(&[0x00; 16]).is_none());
    }

    #[test]
    fn serialize_round_trip() {
        let mut store = ContactStore::new();
        store
            .add(make_contact(0xAA, true, PeeringPriority::High))
            .unwrap();
        store
            .add(make_contact(0xBB, false, PeeringPriority::Low))
            .unwrap();
        let bytes = store.serialize().unwrap();
        let restored = ContactStore::deserialize(&bytes).unwrap();
        assert_eq!(restored.len(), 2);
        assert!(restored.contains(&[0xAA; 16]));
        assert!(restored.contains(&[0xBB; 16]));
    }

    #[test]
    fn deserialize_bad_data() {
        let result = ContactStore::deserialize(&[0xFF, 0xFF, 0xFF]);
        assert!(matches!(result, Err(ContactError::DeserializeError(_))));
    }
}
