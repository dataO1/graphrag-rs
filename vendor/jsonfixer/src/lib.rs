pub mod constants;
pub mod context;
pub mod error;
pub mod json_parser;
pub mod object_comparer;
pub mod parsers;
pub mod string_file_wrapper;
pub mod types;

pub use error::{JsonRepairError, Result};
pub use json_parser::JsonParser;
pub use types::{JsonRepairOptions, JsonValue, RepairLog};

/// Repairs a JSON string and returns the repaired JSON as a string.
///
/// # Arguments
/// * `json_str` - The JSON string to repair
/// * `options` - Configuration options for the repair process
///
/// # Returns
/// A `Result` containing the repaired JSON string or an error
pub fn repair_json(json_str: &str, options: JsonRepairOptions) -> Result<String> {
    let mut parser = JsonParser::new(json_str, options);
    parser.parse_as_string()
}

/// Repairs a JSON string and returns it as a parsed `JsonValue`.
///
/// # Arguments
/// * `json_str` - The JSON string to repair
/// * `options` - Configuration options for the repair process
///
/// # Returns
/// A `Result` containing the parsed JSON value or an error
pub fn loads(json_str: &str, options: JsonRepairOptions) -> Result<JsonValue> {
    let mut parser = JsonParser::new(json_str, options);
    parser.parse()
}

/// Repairs JSON from a file and returns it as a parsed `JsonValue`.
///
/// # Arguments
/// * `path` - Path to the JSON file
/// * `options` - Configuration options for the repair process
///
/// # Returns
/// A `Result` containing the parsed JSON value or an error
pub fn from_file<P: AsRef<std::path::Path>>(
    path: P,
    options: JsonRepairOptions,
) -> Result<JsonValue> {
    let content = std::fs::read_to_string(path)?;
    loads(&content, options)
}

/// Repairs JSON from a reader and returns it as a parsed `JsonValue`.
///
/// # Arguments
/// * `reader` - A reader implementing `std::io::Read`
/// * `options` - Configuration options for the repair process
///
/// # Returns
/// A `Result` containing the parsed JSON value or an error
pub fn from_reader<R: std::io::Read>(
    mut reader: R,
    options: JsonRepairOptions,
) -> Result<JsonValue> {
    let mut content = String::new();
    reader.read_to_string(&mut content)?;
    loads(&content, options)
}
