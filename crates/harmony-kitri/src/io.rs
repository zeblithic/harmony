// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Durable I/O operation types — the primitives that get event-sourced.

use alloc::string::String;
use alloc::vec::Vec;

/// A durable I/O operation requested by a Kitri workflow.
///
/// Each variant corresponds to one of the `kitri::*` SDK primitives.
/// The runtime intercepts these, logs them in the event log, and
/// replays cached responses on crash recovery.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum KitriIoOp {
    /// Retrieve content by CID from the mesh.
    Fetch { cid: [u8; 32] },
    /// Store content, returning its CID.
    Store { data: Vec<u8> },
    /// Invoke a Zenoh queryable.
    Query { topic: String, payload: Vec<u8> },
    /// Publish to a Zenoh topic (staged until audit passes).
    Publish { topic: String, payload: Vec<u8> },
    /// Invoke an AI model.
    Infer {
        prompt: String,
        context: Vec<u8>,
        model: Option<String>,
        max_tokens: Option<u32>,
    },
    /// Encrypt data via kernel (no key exposure to the workflow).
    Seal { data: Vec<u8>, recipient: [u8; 16] },
    /// Decrypt sealed data via kernel.
    Open { sealed: Vec<u8> },
    /// Spawn a child workflow by program CID.
    Spawn {
        program_cid: [u8; 32],
        input: Vec<u8>,
    },
}

impl KitriIoOp {
    /// Returns `true` if this operation has externally-visible side effects
    /// and must be staged before commit.
    pub fn is_side_effect(&self) -> bool {
        matches!(
            self,
            Self::Publish { .. } | Self::Store { .. } | Self::Seal { .. } | Self::Spawn { .. }
        )
    }
}

/// The result of a durable I/O operation, returned by the runtime.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum KitriIoResult {
    /// Content retrieved by CID.
    Fetched { data: Vec<u8> },
    /// Content stored; returns its CID.
    Stored { cid: [u8; 32] },
    /// Response from a Zenoh queryable.
    QueryReply { payload: Vec<u8> },
    /// Publish acknowledged (committed after audit).
    Published,
    /// AI model inference result.
    Inferred { output: String },
    /// Encrypted ciphertext from a Seal operation.
    Sealed { ciphertext: Vec<u8> },
    /// Decrypted plaintext from an Open operation.
    Opened { plaintext: Vec<u8> },
    /// Child workflow launched; returns its WorkflowId.
    Spawned { workflow_id: [u8; 32] },
    /// Any I/O operation failed (transient or permanent).
    Failed { error: String },
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;

    #[test]
    fn io_op_variants() {
        let fetch = KitriIoOp::Fetch { cid: [0xAA; 32] };
        assert!(matches!(fetch, KitriIoOp::Fetch { .. }));

        let store = KitriIoOp::Store {
            data: vec![1, 2, 3],
        };
        assert!(matches!(store, KitriIoOp::Store { .. }));

        let query = KitriIoOp::Query {
            topic: "blockchain/verify".into(),
            payload: vec![42],
        };
        assert!(matches!(query, KitriIoOp::Query { .. }));

        let publish = KitriIoOp::Publish {
            topic: "shipments/verified".into(),
            payload: vec![1],
        };
        assert!(matches!(publish, KitriIoOp::Publish { .. }));

        let infer = KitriIoOp::Infer {
            prompt: "summarize".into(),
            context: vec![],
            model: None,
            max_tokens: Some(1000),
        };
        assert!(matches!(infer, KitriIoOp::Infer { .. }));

        let seal = KitriIoOp::Seal {
            data: vec![1],
            recipient: [0xBB; 16],
        };
        assert!(matches!(seal, KitriIoOp::Seal { .. }));

        let open = KitriIoOp::Open { sealed: vec![1, 2] };
        assert!(matches!(open, KitriIoOp::Open { .. }));

        let spawn = KitriIoOp::Spawn {
            program_cid: [0xCC; 32],
            input: vec![1],
        };
        assert!(matches!(spawn, KitriIoOp::Spawn { .. }));
    }

    #[test]
    fn io_result_variants() {
        let fetched = KitriIoResult::Fetched { data: vec![1] };
        assert!(matches!(fetched, KitriIoResult::Fetched { .. }));

        let stored = KitriIoResult::Stored { cid: [0xAA; 32] };
        assert!(matches!(stored, KitriIoResult::Stored { .. }));

        let replied = KitriIoResult::QueryReply { payload: vec![42] };
        assert!(matches!(replied, KitriIoResult::QueryReply { .. }));

        let published = KitriIoResult::Published;
        assert!(matches!(published, KitriIoResult::Published));

        let inferred = KitriIoResult::Inferred {
            output: "result".into(),
        };
        assert!(matches!(inferred, KitriIoResult::Inferred { .. }));

        let sealed = KitriIoResult::Sealed {
            ciphertext: vec![1],
        };
        assert!(matches!(sealed, KitriIoResult::Sealed { .. }));

        let opened = KitriIoResult::Opened { plaintext: vec![1] };
        assert!(matches!(opened, KitriIoResult::Opened { .. }));

        let spawned = KitriIoResult::Spawned {
            workflow_id: [0xDD; 32],
        };
        assert!(matches!(spawned, KitriIoResult::Spawned { .. }));

        let failed = KitriIoResult::Failed {
            error: "timeout".into(),
        };
        assert!(matches!(failed, KitriIoResult::Failed { .. }));
    }

    #[test]
    fn io_op_is_side_effect() {
        // Publish, Store, Seal, Spawn are side effects (need staging).
        assert!(KitriIoOp::Publish {
            topic: "t".into(),
            payload: vec![]
        }
        .is_side_effect());
        assert!(KitriIoOp::Store { data: vec![] }.is_side_effect());
        assert!(KitriIoOp::Seal {
            data: vec![],
            recipient: [0; 16]
        }
        .is_side_effect());
        assert!(KitriIoOp::Spawn {
            program_cid: [0; 32],
            input: vec![]
        }
        .is_side_effect());

        // Fetch, Query, Infer, Open are reads (no staging needed).
        assert!(!KitriIoOp::Fetch { cid: [0; 32] }.is_side_effect());
        assert!(!KitriIoOp::Query {
            topic: "t".into(),
            payload: vec![]
        }
        .is_side_effect());
        assert!(!KitriIoOp::Infer {
            prompt: "p".into(),
            context: vec![],
            model: None,
            max_tokens: None
        }
        .is_side_effect());
        assert!(!KitriIoOp::Open { sealed: vec![] }.is_side_effect());
    }
}
