use crate::error::{JsonRepairError, Result};
use crate::json_parser::JsonParser;
use crate::types::JsonValue;
use serde_json::Value;

pub fn parse_boolean_or_null(parser: &mut JsonParser) -> Result<JsonValue> {
    let start = parser.index();
    let c = parser.get_char_at().unwrap_or('\0').to_ascii_lowercase();
    
    match c {
        't' => {
            // Check for "true" character by character
            let mut chars = vec![];
            for _ in 0..4 {
                if let Some(ch) = parser.get_char_at_offset(chars.len()) {
                    chars.push(ch.to_ascii_lowercase());
                } else {
                    break;
                }
            }
            if chars.len() >= 4 && chars[0] == 't' && chars[1] == 'r' && chars[2] == 'u' && chars[3] == 'e' {
                parser.advance(4);
                Ok(Value::Bool(true))
            } else {
                parser.log("invalid_boolean", start, "Invalid boolean value");
                Err(JsonRepairError::InvalidBooleanOrNull)
            }
        }
        'f' => {
            // Check for "false" character by character
            let mut chars = vec![];
            for _ in 0..5 {
                if let Some(ch) = parser.get_char_at_offset(chars.len()) {
                    chars.push(ch.to_ascii_lowercase());
                } else {
                    break;
                }
            }
            if chars.len() >= 5 && chars[0] == 'f' && chars[1] == 'a' && chars[2] == 'l' && chars[3] == 's' && chars[4] == 'e' {
                parser.advance(5);
                Ok(Value::Bool(false))
            } else {
                parser.log("invalid_boolean", start, "Invalid boolean value");
                Err(JsonRepairError::InvalidBooleanOrNull)
            }
        }
        'n' => {
            // Check for "null" character by character
            let mut chars = vec![];
            for _ in 0..4 {
                if let Some(ch) = parser.get_char_at_offset(chars.len()) {
                    chars.push(ch.to_ascii_lowercase());
                } else {
                    break;
                }
            }
            if chars.len() >= 4 && chars[0] == 'n' && chars[1] == 'u' && chars[2] == 'l' && chars[3] == 'l' {
                parser.advance(4);
                Ok(Value::Null)
            } else {
                parser.log("invalid_null", start, "Invalid null value");
                Err(JsonRepairError::InvalidBooleanOrNull)
            }
        }
        _ => {
            parser.log("invalid_literal", start, "Expected boolean or null");
            Err(JsonRepairError::InvalidBooleanOrNull)
        }
    }
}