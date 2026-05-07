use crate::error::Result;
use crate::json_parser::JsonParser;
use crate::types::JsonValue;
use serde_json::Value;

pub fn parse_comment(parser: &mut JsonParser) -> Result<JsonValue> {
    let c = parser.get_char_at().unwrap_or('\0');
    
    if c == '#' {
        // Single line comment
        parser.advance(1);
        while let Some(c) = parser.get_char_at() {
            if c == '\n' {
                parser.advance(1);
                break;
            }
            parser.advance(1);
        }
        parser.log("comment_removed", parser.index(), "Removed single-line comment");
    } else if c == '/' {
        parser.advance(1);
        
        if let Some(next_c) = parser.get_char_at() {
            if next_c == '/' {
                // Single line comment
                parser.advance(1);
                while let Some(c) = parser.get_char_at() {
                    if c == '\n' {
                        parser.advance(1);
                        break;
                    }
                    parser.advance(1);
                }
                parser.log("comment_removed", parser.index(), "Removed single-line comment");
            } else if next_c == '*' {
                // Multi-line comment
                parser.advance(1);
                let mut nesting = 1;
                
                while nesting > 0 && parser.index() < parser.len() {
                    if let Some(c) = parser.get_char_at() {
                        if c == '/' && parser.get_char_at_offset(1) == Some('*') {
                            nesting += 1;
                            parser.advance(2);
                        } else if c == '*' && parser.get_char_at_offset(1) == Some('/') {
                            nesting -= 1;
                            parser.advance(2);
                        } else {
                            parser.advance(1);
                        }
                    } else {
                        break;
                    }
                }
                parser.log("comment_removed", parser.index(), "Removed multi-line comment");
            } else {
                // Not a comment, might be a division operator or something else
                parser.set_index(parser.index() - 1);
            }
        }
    }

    // Return null and continue parsing
    Ok(Value::Null)
}