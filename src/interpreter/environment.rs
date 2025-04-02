use std::collections::HashMap;
use crate::ast::expressions::Value;

#[derive(Clone)]
pub struct Environment {
    pub variables: HashMap<String, Value>,
}

impl Environment {
    pub fn new() -> Self {
        Environment {
            variables: HashMap::new(),
        }
    }
    
    pub fn set_variable(&mut self, name: String, value: Value) {
        self.variables.insert(name, value);
    }
    
    pub fn get_variable(&self, name: &str) -> Option<&Value> {
        self.variables.get(name)
    }
} 