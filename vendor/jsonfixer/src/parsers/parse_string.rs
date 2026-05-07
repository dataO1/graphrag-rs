use crate::context::ContextValue;
use crate::error::Result;
use crate::json_parser::JsonParser;
use crate::types::JsonValue;
use crate::constants::STRING_DELIMITERS;
use serde_json::Value;

pub fn parse_string(parser: &mut JsonParser) -> Result<JsonValue> {
    let mut missing_quotes = false;
    let mut doubled_quotes = false;
    let mut lstring_delimiter = '"';
    let mut rstring_delimiter = '"';

    let char = parser.get_char_at().unwrap_or('\0');
    
    // Skip non-string characters until we find a delimiter or alphanumeric
    let mut current_char = char;
    while current_char != '\0' && !STRING_DELIMITERS.contains(&current_char) && !current_char.is_alphanumeric() {
        parser.advance(1);
        current_char = parser.get_char_at().unwrap_or('\0');
    }

    if current_char == '\0' {
        return Ok(Value::String(String::new()));
    }

    // Set delimiters based on quote type
    if current_char == '\'' {
        lstring_delimiter = '\'';
        rstring_delimiter = '\'';
    } else if current_char == '"' {
        lstring_delimiter = '"';
        rstring_delimiter = '"';
    } else if current_char.is_alphanumeric() {
        // Handle unquoted strings
        if let Some(ctx) = parser.context().current() {
            if *ctx != ContextValue::ObjectKey {
                // Try to parse as boolean or null first
                if let Ok(value) = parser.parse_boolean_or_null() {
                    return Ok(value);
                }
            }
        }
        parser.log("missing_quotes", parser.index(), "Found unquoted string");
        missing_quotes = true;
    }

    if !missing_quotes {
        parser.advance(1);
    }

    // Handle doubled quotes
    if let Some(next_char) = parser.get_char_at() {
        if STRING_DELIMITERS.contains(&next_char) && next_char == lstring_delimiter {
            // Check for empty string case
            let next_next_char = parser.get_char_at_offset(1).unwrap_or('\0');
            if (parser.context().current() == Some(&ContextValue::ObjectKey) && next_next_char == ':')
                || (parser.context().current() == Some(&ContextValue::ObjectValue) 
                    && [',', '}'].contains(&next_next_char)) {
                parser.advance(1);
                return Ok(Value::String(String::new()));
            } else if parser.get_char_at_offset(1).map_or(false, |c| c == lstring_delimiter) {
                parser.log("doubled_quotes", parser.index(), "Found doubled quotes");
                doubled_quotes = true;
                parser.advance(1);
            }
        }
    }

    let mut string_acc = String::new();
    let mut escape_next = false;

    // Parse the string content
    while let Some(c) = parser.get_char_at() {
        if escape_next {
            match c {
                'n' => string_acc.push('\n'),
                't' => string_acc.push('\t'),
                'r' => string_acc.push('\r'),
                'b' => string_acc.push('\x08'),
                'f' => string_acc.push('\x0c'),
                '\\' => string_acc.push('\\'),
                '/' => string_acc.push('/'),
                '"' => string_acc.push('"'),
                '\'' => string_acc.push('\''),
                'u' => {
                    parser.advance(1);
                    let hex_chars: String = (0..4)
                        .filter_map(|_| parser.get_char_at())
                        .take(4)
                        .collect();
                    if hex_chars.len() == 4 {
                        if let Ok(code_point) = u32::from_str_radix(&hex_chars, 16) {
                            if let Some(c) = char::from_u32(code_point) {
                                string_acc.push(c);
                            }
                        }
                    }
                }
                _ => {
                    parser.log("invalid_escape", parser.index(), "Invalid escape sequence");
                    string_acc.push(c);
                }
            }
            escape_next = false;
            parser.advance(1);
            continue;
        }

        if c == '\\' && !missing_quotes {
            escape_next = true;
            parser.advance(1);
            continue;
        }

        // Check for string terminator
        if !missing_quotes && c == rstring_delimiter {
            if !doubled_quotes || parser.get_char_at_offset(1).map_or(true, |c| c != rstring_delimiter) {
                parser.advance(1);
                break;
            } else if doubled_quotes {
                parser.advance(1);
            }
        }

        // Handle string termination conditions for both quoted and unquoted strings
        if let Some(ctx) = parser.context().current() {
            match ctx {
                // For object keys, always stop at colon
                ContextValue::ObjectKey if c == ':' => break,
                // For object values, stop at comma or closing brace
                ContextValue::ObjectValue if [',', '}'].contains(&c) => {
                    // For quoted strings, we need to be more careful
                    if !missing_quotes {
                        // Check if this is really the end of the string
                        // Look ahead to see if there's a pattern suggesting this is a structural delimiter
                        let mut lookahead = 1;
                        let mut is_structural = false;
                        
                        while let Some(next_c) = parser.get_char_at_offset(lookahead) {
                            if next_c.is_whitespace() {
                                lookahead += 1;
                                continue;
                            }
                            
                            // If we find a quote after whitespace, it might be a new key
                            if STRING_DELIMITERS.contains(&next_c) {
                                is_structural = true;
                                break;
                            }
                            // If we find another key-value pattern
                            else if next_c.is_alphanumeric() {
                                // Look further for a colon
                                let mut further_lookahead = lookahead + 1;
                                while let Some(further_c) = parser.get_char_at_offset(further_lookahead) {
                                    if further_c.is_whitespace() {
                                        further_lookahead += 1;
                                        continue;
                                    }
                                    if further_c == ':' {
                                        is_structural = true;
                                        break;
                                    }
                                    break;
                                }
                                break;
                            }
                            break;
                        }
                        
                        if is_structural || c == '}' {
                            break;
                        }
                    } else {
                        // For unquoted strings, comma and brace are always delimiters
                        break;
                    }
                },
                // For arrays, stop at comma or closing bracket
                ContextValue::Array if [',', ']'].contains(&c) => break,
                _ => {}
            }
        }
        
        // Additional check for missing quotes
        if missing_quotes && c.is_whitespace() {
            let next_char = parser.get_char_at_offset(1).unwrap_or('\0');
            if [':', ',', '}', ']', ' '].contains(&next_char) {
                if let Some(ContextValue::ObjectKey) = parser.context().current() {
                    if next_char == ':' {
                        break;
                    }
                } else if next_char == '}' || next_char == ']' {
                    break;
                }
            }
        }

        string_acc.push(c);
        parser.advance(1);
    }

    Ok(Value::String(string_acc))
}