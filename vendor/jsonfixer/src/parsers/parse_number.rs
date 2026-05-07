use crate::error::{JsonRepairError, Result};
use crate::json_parser::JsonParser;
use crate::types::JsonValue;
use serde_json::Value;

pub fn parse_number(parser: &mut JsonParser) -> Result<JsonValue> {
    let start = parser.index();
    let mut has_decimal = false;
    let mut has_exponent = false;
    let mut has_digits = false;

    // Handle negative sign
    if parser.get_char_at() == Some('-') {
        parser.advance(1);
    }

    // Parse integer part
    while let Some(c) = parser.get_char_at() {
        if c.is_ascii_digit() {
            has_digits = true;
            parser.advance(1);
        } else {
            break;
        }
    }

    // Parse decimal part
    if parser.get_char_at() == Some('.') {
        has_decimal = true;
        parser.advance(1);
        
        let mut decimal_digits = 0;
        while let Some(c) = parser.get_char_at() {
            if c.is_ascii_digit() {
                decimal_digits += 1;
                has_digits = true;
                parser.advance(1);
            } else {
                break;
            }
        }
        
        if decimal_digits == 0 {
            parser.log("invalid_number", parser.index(), "No digits after decimal point");
        }
    }

    // Parse exponent
    if parser.get_char_at().map_or(false, |c| c == 'e' || c == 'E') {
        has_exponent = true;
        parser.advance(1);
        
        // Handle exponent sign
        if parser.get_char_at().map_or(false, |c| c == '+' || c == '-') {
            parser.advance(1);
        }
        
        let mut exponent_digits = 0;
        while let Some(c) = parser.get_char_at() {
            if c.is_ascii_digit() {
                exponent_digits += 1;
                parser.advance(1);
            } else {
                break;
            }
        }
        
        if exponent_digits == 0 {
            parser.log("invalid_number", parser.index(), "No digits in exponent");
        }
    }

    if !has_digits {
        parser.set_index(start);
        return Err(JsonRepairError::InvalidNumber);
    }

    let number_str = &parser.json_str()[start..parser.index()];
    
    // Try to parse as integer first, then as float
    if !has_decimal && !has_exponent {
        if let Ok(int_val) = number_str.parse::<i64>() {
            return Ok(Value::Number(serde_json::Number::from(int_val)));
        }
    }
    
    match number_str.parse::<f64>() {
        Ok(num) => Ok(Value::Number(serde_json::Number::from_f64(num).unwrap())),
        Err(_) => {
            parser.log("invalid_number", start, "Failed to parse number");
            Err(JsonRepairError::InvalidNumber)
        }
    }
}