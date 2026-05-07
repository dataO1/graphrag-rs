use crate::context::ContextValue;
use crate::error::Result;
use crate::json_parser::JsonParser;
use crate::types::JsonValue;
use serde_json::Value;

pub fn parse_array(parser: &mut JsonParser) -> Result<JsonValue> {
    let mut array = Vec::new();
    
    // Expect opening bracket
    if parser.get_char_at() == Some('[') {
        parser.advance(1);
    } else {
        parser.log("missing_open_bracket", parser.index(), "Missing opening bracket, adding it");
    }

    parser.context_mut().set(ContextValue::Array);

    while parser.index() < parser.len() {
        // No-progress guard: if a full iteration completes without advancing
        // the cursor, a callee returned a fallback without consuming input.
        // Break to avoid an infinite loop that pushes fallback elements forever
        // and causes unbounded Vec growth (graphrag-rs OOM 2026-05-07).
        let pre_iter_idx = parser.index();

        parser.skip_whitespace();

        let c = parser.get_char_at().unwrap_or('\0');
        if c == ']' {
            parser.advance(1);
            break;
        }

        // Parse array element
        let element = if c == ',' || c == ']' {
            parser.log("empty_element", parser.index(), "Empty array element, using null");
            Value::Null
        } else {
            parser.parse_json()?
        };

        array.push(element);

        parser.skip_whitespace();

        // Handle comma
        if parser.get_char_at() == Some(',') {
            parser.advance(1);

            // Handle trailing comma
            parser.skip_whitespace();
            if parser.get_char_at() == Some(']') {
                parser.log("trailing_comma", parser.index(), "Removing trailing comma");
                parser.advance(1);  // Consume the closing bracket
                break;
            }
        } else if parser.get_char_at() != Some(']') {
            parser.log("missing_comma", parser.index(), "Missing comma between array elements");
        }

        if parser.index() == pre_iter_idx {
            parser.log("no_progress", parser.index(), "parse_array made no progress, breaking");
            break;
        }
    }

    // Ensure closing bracket
    if parser.get_char_at() != Some(']') && parser.index() >= parser.len() {
        parser.log("missing_close_bracket", parser.index(), "Missing closing bracket, adding it");
    }

    parser.context_mut().reset();

    Ok(Value::Array(array))
}