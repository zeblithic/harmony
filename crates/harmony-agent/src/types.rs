//! Agent protocol message types.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentTask {
    pub task_id: String,
    pub task_type: String,
    pub params: serde_json::Value,
    pub context: Option<TaskContext>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskContext {
    pub parent_task_id: Option<String>,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentResult {
    pub task_id: String,
    pub status: TaskStatus,
    pub output: Option<serde_json::Value>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskStatus {
    Success,
    Failed,
    Rejected,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentCapacity {
    pub agent_id: String,
    pub task_types: Vec<String>,
    pub status: AgentStatus,
    pub max_concurrent: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentStatus {
    Ready,
    Busy,
    Draining,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn agent_task_round_trip() {
        let task = AgentTask {
            task_id: "task-001".to_string(),
            task_type: "summarize".to_string(),
            params: json!({"text": "hello world", "max_tokens": 100}),
            context: Some(TaskContext {
                parent_task_id: Some("parent-000".to_string()),
                metadata: Some(json!({"priority": "high"})),
            }),
        };
        let json = serde_json::to_string(&task).unwrap();
        let decoded: AgentTask = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.task_id, task.task_id);
        assert_eq!(decoded.task_type, task.task_type);
        assert_eq!(decoded.params, task.params);
        let ctx = decoded.context.unwrap();
        assert_eq!(ctx.parent_task_id.as_deref(), Some("parent-000"));
        assert_eq!(ctx.metadata, Some(json!({"priority": "high"})));
    }

    #[test]
    fn agent_task_without_context() {
        let task = AgentTask {
            task_id: "task-002".to_string(),
            task_type: "embed".to_string(),
            params: json!({"input": "foo"}),
            context: None,
        };
        let json = serde_json::to_string(&task).unwrap();
        let decoded: AgentTask = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.task_id, "task-002");
        assert!(decoded.context.is_none());
    }

    #[test]
    fn agent_result_success_round_trip() {
        let result = AgentResult {
            task_id: "task-001".to_string(),
            status: TaskStatus::Success,
            output: Some(json!({"summary": "a brief summary"})),
            error: None,
        };
        let json = serde_json::to_string(&result).unwrap();
        let decoded: AgentResult = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.task_id, "task-001");
        assert_eq!(decoded.status, TaskStatus::Success);
        assert_eq!(decoded.output, Some(json!({"summary": "a brief summary"})));
        assert!(decoded.error.is_none());
    }

    #[test]
    fn agent_result_failed_round_trip() {
        let result = AgentResult {
            task_id: "task-003".to_string(),
            status: TaskStatus::Failed,
            output: None,
            error: Some("timeout after 30s".to_string()),
        };
        let json = serde_json::to_string(&result).unwrap();
        let decoded: AgentResult = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.status, TaskStatus::Failed);
        assert!(decoded.output.is_none());
        assert_eq!(decoded.error.as_deref(), Some("timeout after 30s"));
    }

    #[test]
    fn agent_result_rejected() {
        let result = AgentResult {
            task_id: "task-004".to_string(),
            status: TaskStatus::Rejected,
            output: None,
            error: Some("unsupported task type".to_string()),
        };
        let json = serde_json::to_string(&result).unwrap();
        let decoded: AgentResult = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.status, TaskStatus::Rejected);
        assert_eq!(decoded.error.as_deref(), Some("unsupported task type"));
    }

    #[test]
    fn agent_capacity_round_trip() {
        let cap = AgentCapacity {
            agent_id: "agent-abc".to_string(),
            task_types: vec![
                "summarize".to_string(),
                "embed".to_string(),
                "classify".to_string(),
            ],
            status: AgentStatus::Ready,
            max_concurrent: 4,
        };
        let json = serde_json::to_string(&cap).unwrap();
        let decoded: AgentCapacity = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.agent_id, "agent-abc");
        assert_eq!(decoded.task_types.len(), 3);
        assert!(decoded.task_types.contains(&"embed".to_string()));
        assert_eq!(decoded.status, AgentStatus::Ready);
        assert_eq!(decoded.max_concurrent, 4);
    }

    #[test]
    fn agent_capacity_all_statuses() {
        for (status, expected_str) in [
            (AgentStatus::Ready, "\"Ready\""),
            (AgentStatus::Busy, "\"Busy\""),
            (AgentStatus::Draining, "\"Draining\""),
        ] {
            let cap = AgentCapacity {
                agent_id: "agent-x".to_string(),
                task_types: vec![],
                status,
                max_concurrent: 1,
            };
            let json = serde_json::to_string(&cap).unwrap();
            let decoded: AgentCapacity = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.status, status);
            // Verify the status serializes to the expected string variant
            let status_json = serde_json::to_string(&status).unwrap();
            assert_eq!(status_json, expected_str);
        }
    }

    #[test]
    fn snake_case_json_keys() {
        let task = AgentTask {
            task_id: "task-snake".to_string(),
            task_type: "classify".to_string(),
            params: json!({}),
            context: None,
        };
        let json = serde_json::to_string(&task).unwrap();
        // Must use snake_case keys, not camelCase
        assert!(json.contains("\"task_id\""), "expected snake_case key task_id, got: {json}");
        assert!(json.contains("\"task_type\""), "expected snake_case key task_type, got: {json}");
        assert!(!json.contains("\"taskId\""), "unexpected camelCase key taskId in: {json}");
        assert!(!json.contains("\"taskType\""), "unexpected camelCase key taskType in: {json}");
    }
}
