use thiserror::Error;

pub type Result<T> = std::result::Result<T, JsonRepairError>;

#[derive(Error, Debug)]
pub enum JsonRepairError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("JSON parsing error: {0}")]
    ParseError(String),
    
    #[error("Invalid JSON structure")]
    InvalidStructure,
    
    #[error("Unexpected end of input")]
    UnexpectedEndOfInput,
    
    #[error("Invalid escape sequence")]
    InvalidEscapeSequence,
    
    #[error("Invalid number format")]
    InvalidNumber,
    
    #[error("Invalid boolean or null value")]
    InvalidBooleanOrNull,
}