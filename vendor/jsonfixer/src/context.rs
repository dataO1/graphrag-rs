#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextValue {
    ObjectKey,
    ObjectValue,
    Array,
}

#[derive(Debug)]
pub struct JsonContext {
    context: Vec<ContextValue>,
    current: Option<ContextValue>,
    pub empty: bool,
}

impl JsonContext {
    pub fn new() -> Self {
        Self {
            context: Vec::new(),
            current: None,
            empty: true,
        }
    }

    pub fn set(&mut self, value: ContextValue) {
        self.context.push(value);
        self.current = Some(value);
        self.empty = false;
    }

    pub fn reset(&mut self) {
        self.context.pop();
        self.current = self.context.last().cloned();
        self.empty = self.context.is_empty();
    }

    pub fn current(&self) -> Option<&ContextValue> {
        self.current.as_ref()
    }

    pub fn contains(&self, value: &ContextValue) -> bool {
        self.context.contains(value)
    }

    pub fn len(&self) -> usize {
        self.context.len()
    }
}

impl Default for JsonContext {
    fn default() -> Self {
        Self::new()
    }
}