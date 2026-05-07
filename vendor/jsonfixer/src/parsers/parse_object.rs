use crate::context::ContextValue;
use crate::error::Result;
use crate::json_parser::JsonParser;
use crate::types::JsonValue;
use serde_json::Value;
use serde_json::Map;

pub fn parse_object(parser: &mut JsonParser) -> Result<JsonValue> {
    let mut obj = Map::new();
    
    // Expect opening brace
    if parser.get_char_at() == Some('{') {
        parser.advance(1);
    } else {
        parser.log("missing_open_brace", parser.index(), "Missing opening brace, adding it");
    }

    while parser.index() < parser.len() {
        // No-progress guard: if a full iteration completes without advancing
        // the cursor, a callee returned a fallback without consuming input.
        // Break to avoid an infinite loop that pushes fallback entries forever
        // and causes unbounded Map growth (graphrag-rs OOM 2026-05-07).
        let pre_iter_idx = parser.index();

        parser.skip_whitespace();

        let c = parser.get_char_at().unwrap_or('\0');
        if c == '}' {
            parser.advance(1);
            break;
        }

        // Handle unexpected colon before key
        if c == ':' {
            parser.log("unexpected_colon", parser.index(), "Found colon before key, ignoring");
            parser.advance(1);
            parser.skip_whitespace();
            continue;
        }

        // Parse key
        parser.context_mut().set(ContextValue::ObjectKey);
        let rollback_index = parser.index();
        
        let key = match parser.parse_string()? {
            Value::String(s) => s,
            _ => String::new(),
        };

        // Clean up quotes from the key if present
        let key = if key.ends_with('"') || key.ends_with('\'') {
            key[..key.len() - 1].trim().to_string()
        } else {
            key.clone()
        };

        // Handle case where key contains a colon (like "brokn:"borke")
        // We need to split this into proper key and value
        if !key.is_empty() && key.contains(':') {
            // Find the first colon that's not part of an escape sequence
            if let Some(colon_pos) = key.find(':') {
                let actual_key = key[..colon_pos].trim().to_string();
                let remaining = key[colon_pos + 1..].trim();
                
                // Move parser position to after the colon we found
                parser.set_index(rollback_index + colon_pos + 1);
                
                // Skip whitespace
                parser.skip_whitespace();
                
                // Now parse the value from the remaining part
                let value = if remaining.is_empty() {
                    // Look ahead to see if there's a value
                    if let Some(next_c) = parser.get_char_at() {
                        if next_c == '"' {
                            // Parse as string
                            parser.parse_string()?
                        } else {
                            Value::String(String::new())
                        }
                    } else {
                        Value::String(String::new())
                    }
                } else {
                    // The remaining part is the value
                    Value::String(remaining.to_string())
                };
                
                obj.insert(actual_key, value);
                
                // Skip to next entry
                parser.skip_whitespace();
                if parser.get_char_at() == Some(',') {
                    parser.advance(1);
                }
                continue;
            }
        }

        if key.is_empty() && parser.get_char_at() != Some(':') && parser.get_char_at() != Some('}') {
            // Bare `continue` here previously could spin without advancing.
            // Skip one char to make forward progress before retrying.
            if parser.index() < parser.len() {
                parser.advance(1);
            }
            continue;
        }

        // Check for duplicate key in array context
        if parser.context().contains(&ContextValue::Array) && obj.contains_key(&key) {
            parser.log("duplicate_key", parser.index(), "Found duplicate key, closing object");
            parser.set_index(rollback_index.saturating_sub(1));
            break;
        }

        parser.skip_whitespace();

        // Expect colon
        if parser.get_char_at() != Some(':') {
            parser.log("missing_colon", parser.index(), "Missing colon after key");
        } else {
            parser.advance(1);
        }

        parser.context_mut().reset();
        parser.context_mut().set(ContextValue::ObjectValue);
        parser.skip_whitespace();

        // Parse value
        let value = if parser.get_char_at().map_or(false, |c| [',', '}'].contains(&c)) {
            parser.log("empty_value", parser.index(), "Empty value, using empty string");
            Value::String(String::new())
        } else {
            parser.parse_json()?
        };

        obj.insert(key, value);
        
        parser.skip_whitespace();
        
        // Handle comma
        if parser.get_char_at() == Some(',') {
            parser.advance(1);
        } else if parser.get_char_at() != Some('}') {
            parser.log("missing_comma", parser.index(), "Missing comma between object entries");
        }

        if parser.index() == pre_iter_idx {
            parser.log("no_progress", parser.index(), "parse_object made no progress, breaking");
            break;
        }
    }

    // Ensure closing brace
    if parser.get_char_at() != Some('}') && parser.index() >= parser.len() {
        parser.log("missing_close_brace", parser.index(), "Missing closing brace, adding it");
    }

    Ok(Value::Object(obj))
}