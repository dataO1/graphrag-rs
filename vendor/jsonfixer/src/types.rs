use serde::{Deserialize, Serialize};

/// Represents a JSON value - re-export of serde_json::Value
pub type JsonValue = serde_json::Value;

/// Configuration options for JSON repair
#[derive(Debug, Clone, Default)]
pub struct JsonRepairOptions {
    /// Return parsed objects instead of JSON string
    pub return_objects: bool,
    /// Skip calling standard JSON parser first
    pub skip_json_loads: bool,
    /// Enable repair logging
    pub logging: bool,
    /// Ensure ASCII output (escape non-ASCII characters)
    pub ensure_ascii: bool,
    /// Keep repair results stable for streaming JSON
    pub stream_stable: bool,
    /// Chunk length for file processing
    pub chunk_length: usize,
    /// Indentation for output
    pub indent: Option<usize>,
}

/// Represents a repair action log entry
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RepairLog {
    pub action: String,
    pub position: usize,
    pub description: String,
}

/// Result type for operations with logging
pub struct RepairResult<T> {
    pub value: T,
    pub logs: Vec<RepairLog>,
}

// For backward compatibility, implement From for tuple
impl<T> From<(T, Vec<RepairLog>)> for RepairResult<T> {
    fn from((value, logs): (T, Vec<RepairLog>)) -> Self {
        Self { value, logs }
    }
}

impl<T> From<RepairResult<T>> for (T, Vec<RepairLog>) {
    fn from(result: RepairResult<T>) -> Self {
        (result.value, result.logs)
    }
}