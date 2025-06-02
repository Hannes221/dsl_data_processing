use std::collections::HashMap;
use crate::ast::expressions::Value;
use std::sync::Arc;

#[derive(Clone)]
pub struct Environment {
    variables: HashMap<String, Value>,
    parent: Option<Arc<Environment>>,
}

impl Environment {
    pub fn new() -> Self {
        Environment {
            variables: HashMap::new(),
            parent: None,
        }
    }
    
    pub fn with_parent(parent: Arc<Environment>) -> Self {
        Environment {
            variables: HashMap::new(),
            parent: Some(parent),
        }
    }
    
    pub fn set_variable(&mut self, name: String, value: Value) {
        self.variables.insert(name, value);
    }
    
    pub fn get_variable(&self, name: &str) -> Option<&Value> {
        // First check local variables
        if let Some(value) = self.variables.get(name) {
            return Some(value);
        }
        
        // Then check parent environment
        if let Some(parent) = &self.parent {
            return parent.get_variable(name);
        }
        
        None
    }

    pub fn get_variables(&self) -> &HashMap<String, Value> {
        &self.variables
    }
} 