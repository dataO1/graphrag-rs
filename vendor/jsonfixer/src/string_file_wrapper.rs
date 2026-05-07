use std::collections::HashMap;
use std::io::{self, Read, Seek, SeekFrom};

pub struct StringFileWrapper<R: Read + Seek> {
    pub reader: R,
    pub length: Option<u64>,
    buffers: HashMap<usize, String>,
    pub buffer_length: usize,
    current_position: u64,
}

impl<R: Read + Seek> StringFileWrapper<R> {
    pub fn new(mut reader: R, buffer_length: usize) -> Self {
        let length = reader.seek(SeekFrom::End(0)).ok();
        reader.seek(SeekFrom::Start(0)).ok();
        
        let buffer_length = if buffer_length < 2 { 1_000_000 } else { buffer_length };
        
        Self {
            reader,
            length,
            buffers: HashMap::new(),
            buffer_length,
            current_position: 0,
        }
    }

    fn get_buffer(&mut self, index: usize) -> io::Result<&str> {
        if !self.buffers.contains_key(&index) {
            self.reader.seek(SeekFrom::Start((index * self.buffer_length) as u64))?;
            
            let mut buffer = String::with_capacity(self.buffer_length);
            self.reader.by_ref().take(self.buffer_length as u64).read_to_string(&mut buffer)?;
            
            // Keep memory usage low
            if self.buffers.len() > 2 {
                if let Some(&oldest_key) = self.buffers.keys().next() {
                    if oldest_key != index {
                        self.buffers.remove(&oldest_key);
                    }
                }
            }
            
            self.buffers.insert(index, buffer);
        }
        
        Ok(self.buffers.get(&index).unwrap().as_str())
    }

    pub fn get_char_at(&mut self, index: usize) -> Option<char> {
        if index >= self.len() {
            return None;
        }
        
        let buffer_index = index / self.buffer_length;
        let char_index = index % self.buffer_length;
        
        self.get_buffer(buffer_index)
            .ok()
            .and_then(|s| s[char_index..].chars().next())
    }

    pub fn len(&self) -> usize {
        self.length.unwrap_or(0) as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn advance(&mut self, count: usize) {
        self.current_position = self.current_position.saturating_add(count as u64);
    }

    pub fn position(&self) -> usize {
        self.current_position as usize
    }
}