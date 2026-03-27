//! Sans-I/O model registry state machine.
//!
//! Tracks local and remote model availability. Emits Zenoh publish actions
//! when local models are registered or unregistered. Receives remote
//! advertisements and answers queries about model availability.

use std::collections::{HashMap, HashSet};

use harmony_content::ContentId;

use crate::manifest::{ModelAdvertisement, ModelManifest, ModelTask};
use crate::wire::{self, ModelError};

/// Tracks local and remote model availability.
pub struct ModelRegistry {
    /// Models this node has locally (manifest CID → manifest).
    local_models: HashMap<ContentId, ModelManifest>,
    /// Remote advertisements (manifest CID → node address → advertisement).
    remote_models: HashMap<ContentId, HashMap<[u8; 16], ModelAdvertisement>>,
    /// This node's address (for constructing Zenoh keys).
    local_addr: [u8; 16],
}

/// Inbound events for the registry.
#[derive(Debug)]
pub enum ModelRegistryEvent {
    /// A local model was registered (e.g., after ingesting a GGUF file).
    RegisterLocal {
        manifest_cid: ContentId,
        manifest: ModelManifest,
    },
    /// A local model was removed.
    UnregisterLocal { manifest_cid: ContentId },
    /// A remote advertisement was received via Zenoh subscription.
    AdvertisementReceived {
        manifest_cid: ContentId,
        node_addr: [u8; 16],
        ad: ModelAdvertisement,
    },
    /// A remote node departed (all its advertisements should be removed).
    NodeDeparted { node_addr: [u8; 16] },
}

/// Outbound actions from the registry.
#[derive(Debug, PartialEq, Eq)]
pub enum ModelRegistryAction {
    /// Publish advertisement to Zenoh (maps to RuntimeAction::Publish).
    PublishAdvertisement { key_expr: String, payload: Vec<u8> },
    /// Retract advertisement: empty-payload publish (tombstone).
    RetractAdvertisement { key_expr: String },
}

/// Whether a model is available locally, remotely, or both.
#[derive(Debug, PartialEq, Eq)]
pub enum Source {
    Local,
    Remote(Vec<[u8; 16]>),
    Both(Vec<[u8; 16]>),
}

impl ModelRegistry {
    /// Create a new registry for the given node address.
    pub fn new(local_addr: [u8; 16]) -> Self {
        Self {
            local_models: HashMap::new(),
            remote_models: HashMap::new(),
            local_addr,
        }
    }

    /// Process an event and return any resulting actions.
    pub fn handle_event(
        &mut self,
        event: ModelRegistryEvent,
    ) -> Result<Vec<ModelRegistryAction>, ModelError> {
        match event {
            ModelRegistryEvent::RegisterLocal {
                manifest_cid,
                manifest,
            } => {
                let ad = ModelAdvertisement {
                    manifest_cid,
                    name: manifest.name.clone(),
                    family: manifest.family.clone(),
                    parameter_count: manifest.parameter_count,
                    quantization: manifest.quantization.clone(),
                    tasks: manifest.tasks.clone(),
                    memory_estimate: manifest.memory_estimate,
                };
                self.local_models.insert(manifest_cid, manifest);
                let key_expr = harmony_zenoh::namespace::model::advertisement_key(
                    &hex::encode(manifest_cid.to_bytes()),
                    &hex::encode(self.local_addr),
                );
                let payload = wire::encode_advertisement(&ad)?;
                Ok(vec![ModelRegistryAction::PublishAdvertisement {
                    key_expr,
                    payload,
                }])
            }
            ModelRegistryEvent::UnregisterLocal { manifest_cid } => {
                if self.local_models.remove(&manifest_cid).is_some() {
                    let key_expr = harmony_zenoh::namespace::model::advertisement_key(
                        &hex::encode(manifest_cid.to_bytes()),
                        &hex::encode(self.local_addr),
                    );
                    Ok(vec![ModelRegistryAction::RetractAdvertisement { key_expr }])
                } else {
                    Ok(vec![])
                }
            }
            ModelRegistryEvent::AdvertisementReceived {
                manifest_cid,
                node_addr,
                ad,
            } => {
                self.remote_models
                    .entry(manifest_cid)
                    .or_default()
                    .insert(node_addr, ad);
                Ok(vec![])
            }
            ModelRegistryEvent::NodeDeparted { node_addr } => {
                self.remote_models.retain(|_, nodes| {
                    nodes.remove(&node_addr);
                    !nodes.is_empty()
                });
                Ok(vec![])
            }
        }
    }

    /// Remove a specific node's advertisement for a model.
    ///
    /// Called by `route_subscription` when an empty-payload tombstone is received.
    pub fn remove_advertisement(&mut self, manifest_cid: &ContentId, node_addr: &[u8; 16]) {
        if let Some(nodes) = self.remote_models.get_mut(manifest_cid) {
            nodes.remove(node_addr);
            if nodes.is_empty() {
                self.remote_models.remove(manifest_cid);
            }
        }
    }

    /// All locally available models.
    pub fn local_models(&self) -> &HashMap<ContentId, ModelManifest> {
        &self.local_models
    }

    /// Find models matching a task type (searches both local and remote).
    pub fn find_by_task(&self, task: ModelTask) -> Vec<(ContentId, Source)> {
        let mut results = Vec::new();
        let mut seen = HashSet::new();

        // Local models matching the task.
        for (cid, manifest) in &self.local_models {
            if manifest.tasks.contains(&task) {
                let remote_nodes: Vec<[u8; 16]> = self
                    .remote_models
                    .get(cid)
                    .map(|nodes| nodes.keys().copied().collect())
                    .unwrap_or_default();
                let source = if remote_nodes.is_empty() {
                    Source::Local
                } else {
                    Source::Both(remote_nodes)
                };
                results.push((*cid, source));
                seen.insert(*cid);
            }
        }

        // Remote-only models matching the task.
        for (cid, nodes) in &self.remote_models {
            if seen.contains(cid) {
                continue;
            }
            let any_match = nodes.values().any(|ad| ad.tasks.contains(&task));
            if any_match {
                let node_addrs: Vec<[u8; 16]> = nodes.keys().copied().collect();
                results.push((*cid, Source::Remote(node_addrs)));
            }
        }

        results
    }

    /// Find models by family name (searches both local and remote).
    pub fn find_by_family(&self, family: &str) -> Vec<(ContentId, Source)> {
        let mut results = Vec::new();
        let mut seen = HashSet::new();

        for (cid, manifest) in &self.local_models {
            if manifest.family == family {
                let remote_nodes: Vec<[u8; 16]> = self
                    .remote_models
                    .get(cid)
                    .map(|nodes| nodes.keys().copied().collect())
                    .unwrap_or_default();
                let source = if remote_nodes.is_empty() {
                    Source::Local
                } else {
                    Source::Both(remote_nodes)
                };
                results.push((*cid, source));
                seen.insert(*cid);
            }
        }

        for (cid, nodes) in &self.remote_models {
            if seen.contains(cid) {
                continue;
            }
            let any_match = nodes.values().any(|ad| ad.family == family);
            if any_match {
                let node_addrs: Vec<[u8; 16]> = nodes.keys().copied().collect();
                results.push((*cid, Source::Remote(node_addrs)));
            }
        }

        results
    }

    /// Which remote nodes have a specific model?
    ///
    /// Returns `None` if the model is completely unknown (not local, not remote).
    /// Returns `Some(remote_nodes)` if the model is known — the vec may be empty
    /// if the model is only available locally. Check `local_models()` to determine
    /// if this node itself holds the model.
    pub fn nodes_for_model(&self, manifest_cid: &ContentId) -> Option<Vec<[u8; 16]>> {
        let local_has = self.local_models.contains_key(manifest_cid);
        let remote_nodes: Vec<[u8; 16]> = self
            .remote_models
            .get(manifest_cid)
            .map(|nodes| nodes.keys().copied().collect())
            .unwrap_or_default();

        if !local_has && remote_nodes.is_empty() {
            None
        } else {
            Some(remote_nodes)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::ModelFormat;
    use harmony_content::ContentFlags;

    const LOCAL_ADDR: [u8; 16] = [0xAA; 16];
    const REMOTE_ADDR_1: [u8; 16] = [0xBB; 16];
    const REMOTE_ADDR_2: [u8; 16] = [0xCC; 16];

    fn make_manifest(
        name: &str,
        family: &str,
        tasks: Vec<ModelTask>,
    ) -> (ContentId, ModelManifest) {
        let data_cid = ContentId::for_book(name.as_bytes(), ContentFlags::default()).unwrap();
        let manifest = ModelManifest {
            name: name.into(),
            family: family.into(),
            parameter_count: 600_000_000,
            format: ModelFormat::Gguf,
            quantization: Some("Q4_K_M".into()),
            context_length: 32768,
            vocab_size: 151936,
            memory_estimate: 512_000_000,
            tasks,
            data_cid,
            tokenizer_cid: None,
        };
        let encoded = wire::encode_manifest(&manifest).unwrap();
        let cid = wire::manifest_cid(&encoded).unwrap();
        (cid, manifest)
    }

    fn make_advertisement(manifest_cid: ContentId, manifest: &ModelManifest) -> ModelAdvertisement {
        ModelAdvertisement {
            manifest_cid,
            name: manifest.name.clone(),
            family: manifest.family.clone(),
            parameter_count: manifest.parameter_count,
            quantization: manifest.quantization.clone(),
            tasks: manifest.tasks.clone(),
            memory_estimate: manifest.memory_estimate,
        }
    }

    #[test]
    fn register_local_emits_publish() {
        let mut reg = ModelRegistry::new(LOCAL_ADDR);
        let (cid, manifest) = make_manifest("test-model", "qwen3", vec![ModelTask::TextGeneration]);
        let actions = reg
            .handle_event(ModelRegistryEvent::RegisterLocal {
                manifest_cid: cid,
                manifest: manifest.clone(),
            })
            .unwrap();
        assert_eq!(actions.len(), 1);
        match &actions[0] {
            ModelRegistryAction::PublishAdvertisement { key_expr, payload } => {
                assert!(key_expr.starts_with("harmony/model/"));
                assert!(!payload.is_empty());
                let ad = wire::decode_advertisement(payload).unwrap();
                assert_eq!(ad.name, "test-model");
                assert_eq!(ad.manifest_cid, cid);
            }
            other => panic!("expected PublishAdvertisement, got {:?}", other),
        }
        assert!(reg.local_models().contains_key(&cid));
    }

    #[test]
    fn unregister_local_emits_retract() {
        let mut reg = ModelRegistry::new(LOCAL_ADDR);
        let (cid, manifest) = make_manifest("test-model", "qwen3", vec![ModelTask::TextGeneration]);
        reg.handle_event(ModelRegistryEvent::RegisterLocal {
            manifest_cid: cid,
            manifest,
        })
        .unwrap();
        let actions = reg
            .handle_event(ModelRegistryEvent::UnregisterLocal { manifest_cid: cid })
            .unwrap();
        assert_eq!(actions.len(), 1);
        match &actions[0] {
            ModelRegistryAction::RetractAdvertisement { key_expr } => {
                assert!(key_expr.starts_with("harmony/model/"));
            }
            other => panic!("expected RetractAdvertisement, got {:?}", other),
        }
        assert!(!reg.local_models().contains_key(&cid));
    }

    #[test]
    fn advertisement_received_tracks_remote() {
        let mut reg = ModelRegistry::new(LOCAL_ADDR);
        let (cid, manifest) =
            make_manifest("remote-model", "llama", vec![ModelTask::TextGeneration]);
        let ad = make_advertisement(cid, &manifest);
        let actions = reg
            .handle_event(ModelRegistryEvent::AdvertisementReceived {
                manifest_cid: cid,
                node_addr: REMOTE_ADDR_1,
                ad,
            })
            .unwrap();
        assert!(
            actions.is_empty(),
            "receiving an ad should not emit actions"
        );
        let nodes = reg.nodes_for_model(&cid).unwrap();
        assert_eq!(nodes, vec![REMOTE_ADDR_1]);
    }

    #[test]
    fn node_departed_removes_all_entries() {
        let mut reg = ModelRegistry::new(LOCAL_ADDR);
        let (cid1, m1) = make_manifest("model-a", "qwen3", vec![ModelTask::TextGeneration]);
        let (cid2, m2) = make_manifest("model-b", "llama", vec![ModelTask::Embedding]);
        let ad1 = make_advertisement(cid1, &m1);
        let ad2 = make_advertisement(cid2, &m2);

        reg.handle_event(ModelRegistryEvent::AdvertisementReceived {
            manifest_cid: cid1,
            node_addr: REMOTE_ADDR_1,
            ad: ad1,
        })
        .unwrap();
        reg.handle_event(ModelRegistryEvent::AdvertisementReceived {
            manifest_cid: cid2,
            node_addr: REMOTE_ADDR_1,
            ad: ad2,
        })
        .unwrap();

        let actions = reg
            .handle_event(ModelRegistryEvent::NodeDeparted {
                node_addr: REMOTE_ADDR_1,
            })
            .unwrap();
        assert!(actions.is_empty());
        assert!(reg.nodes_for_model(&cid1).is_none());
        assert!(reg.nodes_for_model(&cid2).is_none());
    }

    #[test]
    fn find_by_task_local_and_remote() {
        let mut reg = ModelRegistry::new(LOCAL_ADDR);
        let (cid_local, manifest_local) =
            make_manifest("local-gen", "qwen3", vec![ModelTask::TextGeneration]);
        let (cid_remote, manifest_remote) =
            make_manifest("remote-embed", "bge", vec![ModelTask::Embedding]);

        reg.handle_event(ModelRegistryEvent::RegisterLocal {
            manifest_cid: cid_local,
            manifest: manifest_local,
        })
        .unwrap();
        let ad = make_advertisement(cid_remote, &manifest_remote);
        reg.handle_event(ModelRegistryEvent::AdvertisementReceived {
            manifest_cid: cid_remote,
            node_addr: REMOTE_ADDR_1,
            ad,
        })
        .unwrap();

        let text_gen = reg.find_by_task(ModelTask::TextGeneration);
        assert_eq!(text_gen.len(), 1);
        assert_eq!(text_gen[0].0, cid_local);
        assert_eq!(text_gen[0].1, Source::Local);

        let embed = reg.find_by_task(ModelTask::Embedding);
        assert_eq!(embed.len(), 1);
        assert_eq!(embed[0].0, cid_remote);
        assert!(matches!(embed[0].1, Source::Remote(_)));
    }

    #[test]
    fn find_by_family() {
        let mut reg = ModelRegistry::new(LOCAL_ADDR);
        let (cid, manifest) = make_manifest("qwen-model", "qwen3", vec![ModelTask::TextGeneration]);
        reg.handle_event(ModelRegistryEvent::RegisterLocal {
            manifest_cid: cid,
            manifest,
        })
        .unwrap();
        let results = reg.find_by_family("qwen3");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, cid);

        let no_results = reg.find_by_family("llama");
        assert!(no_results.is_empty());
    }

    #[test]
    fn source_both_when_local_and_remote() {
        let mut reg = ModelRegistry::new(LOCAL_ADDR);
        let (cid, manifest) =
            make_manifest("shared-model", "qwen3", vec![ModelTask::TextGeneration]);
        reg.handle_event(ModelRegistryEvent::RegisterLocal {
            manifest_cid: cid,
            manifest: manifest.clone(),
        })
        .unwrap();
        let ad = make_advertisement(cid, &manifest);
        reg.handle_event(ModelRegistryEvent::AdvertisementReceived {
            manifest_cid: cid,
            node_addr: REMOTE_ADDR_1,
            ad,
        })
        .unwrap();

        let results = reg.find_by_task(ModelTask::TextGeneration);
        assert_eq!(results.len(), 1);
        assert!(matches!(results[0].1, Source::Both(_)));
    }

    #[test]
    fn remove_advertisement_specific_node() {
        let mut reg = ModelRegistry::new(LOCAL_ADDR);
        let (cid, manifest) = make_manifest("multi-node", "llama", vec![ModelTask::TextGeneration]);
        let ad1 = make_advertisement(cid, &manifest);
        let ad2 = make_advertisement(cid, &manifest);

        reg.handle_event(ModelRegistryEvent::AdvertisementReceived {
            manifest_cid: cid,
            node_addr: REMOTE_ADDR_1,
            ad: ad1,
        })
        .unwrap();
        reg.handle_event(ModelRegistryEvent::AdvertisementReceived {
            manifest_cid: cid,
            node_addr: REMOTE_ADDR_2,
            ad: ad2,
        })
        .unwrap();

        reg.remove_advertisement(&cid, &REMOTE_ADDR_1);
        let nodes = reg.nodes_for_model(&cid).unwrap();
        assert_eq!(nodes, vec![REMOTE_ADDR_2]);

        reg.remove_advertisement(&cid, &REMOTE_ADDR_2);
        assert!(reg.nodes_for_model(&cid).is_none());
    }

    #[test]
    fn nodes_for_unknown_model_returns_none() {
        let reg = ModelRegistry::new(LOCAL_ADDR);
        let fake_cid = ContentId::for_book(b"nonexistent", ContentFlags::default()).unwrap();
        assert!(reg.nodes_for_model(&fake_cid).is_none());
    }

    #[test]
    fn nodes_for_local_only_model_returns_some_empty() {
        let mut reg = ModelRegistry::new(LOCAL_ADDR);
        let (cid, manifest) = make_manifest("local-only", "qwen3", vec![ModelTask::TextGeneration]);
        reg.handle_event(ModelRegistryEvent::RegisterLocal {
            manifest_cid: cid,
            manifest,
        })
        .unwrap();
        // Model is local-only — nodes_for_model returns Some(empty), not None.
        let nodes = reg.nodes_for_model(&cid);
        assert_eq!(nodes, Some(vec![]));
    }
}
