// SPDX-License-Identifier: Apache-2.0 OR MIT
//! DAG composition — explicit workflow topology declarations.

use alloc::collections::BTreeSet;
use alloc::string::String;
use alloc::vec::Vec;

use crate::error::KitriError;

/// A single step in a Kitri DAG.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DagStep {
    pub workflow: String,
    pub input_topic: Option<String>,
    pub output_topic: Option<String>,
}

/// A declared DAG of Kitri workflows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KitriDag {
    pub name: String,
    pub steps: Vec<DagStep>,
    pub retry_subgraph: bool,
}

impl KitriDag {
    /// Validate the DAG structure.
    pub fn validate(&self) -> Result<(), KitriError> {
        if self.steps.is_empty() {
            return Err(KitriError::DagInvalid {
                reason: "DAG has no steps".into(),
            });
        }

        let mut available_topics = BTreeSet::new();
        for step in &self.steps {
            if let Some(ref input) = step.input_topic {
                if !available_topics.contains(input) {
                    return Err(KitriError::DagInvalid {
                        reason: alloc::format!(
                            "step '{}' requires topic '{}' which is not produced by a preceding step",
                            step.workflow, input
                        ),
                    });
                }
            }
            if let Some(ref output) = step.output_topic {
                available_topics.insert(output.clone());
            }
        }

        Ok(())
    }

    pub fn entry_points(&self) -> Vec<&DagStep> {
        self.steps
            .iter()
            .filter(|s| s.input_topic.is_none())
            .collect()
    }

    pub fn terminal_steps(&self) -> Vec<&DagStep> {
        self.steps
            .iter()
            .filter(|s| s.output_topic.is_none())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dag_step_creation() {
        let step = DagStep {
            workflow: "scan-receiver".into(),
            input_topic: None,
            output_topic: Some("shipments/scanned".into()),
        };
        assert_eq!(step.workflow, "scan-receiver");
        assert!(step.input_topic.is_none());
        assert!(step.output_topic.is_some());
    }

    #[test]
    fn dag_creation_and_validation() {
        let dag = KitriDag {
            name: "supply-chain".into(),
            steps: vec![
                DagStep {
                    workflow: "scan-receiver".into(),
                    input_topic: None,
                    output_topic: Some("shipments/scanned".into()),
                },
                DagStep {
                    workflow: "shipment-verifier".into(),
                    input_topic: Some("shipments/scanned".into()),
                    output_topic: Some("shipments/verified".into()),
                },
            ],
            retry_subgraph: true,
        };
        assert_eq!(dag.steps.len(), 2);
        assert!(dag.validate().is_ok());
    }

    #[test]
    fn dag_empty_is_invalid() {
        let dag = KitriDag {
            name: "empty".into(),
            steps: vec![],
            retry_subgraph: false,
        };
        assert!(dag.validate().is_err());
    }

    #[test]
    fn dag_broken_chain_is_invalid() {
        let dag = KitriDag {
            name: "broken".into(),
            steps: vec![
                DagStep {
                    workflow: "step-a".into(),
                    input_topic: None,
                    output_topic: Some("topic-a".into()),
                },
                DagStep {
                    workflow: "step-b".into(),
                    input_topic: Some("topic-MISSING".into()),
                    output_topic: None,
                },
            ],
            retry_subgraph: false,
        };
        assert!(dag.validate().is_err());
    }
}
