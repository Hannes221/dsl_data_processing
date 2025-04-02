use std::collections::HashMap;
use std::fmt;

/// Represents a type in our DSL
#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    // Primitive types
    Int,
    Float,
    String,
    Boolean,
    
    // Complex types
    Array(Box<Type>),
    Record(HashMap<String, Type>),
    Function(Vec<Type>, Box<Type>), // Input types and return type
    
    // Type variables for inference
    TypeVar(usize),
    
    // Polymorphic types
    Generic(String, Vec<Type>),
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Int => write!(f, "Int"),
            Type::Float => write!(f, "Float"),
            Type::String => write!(f, "String"),
            Type::Boolean => write!(f, "Boolean"),
            Type::Array(elem_type) => write!(f, "Array<{}>", elem_type),
            Type::Record(fields) => {
                write!(f, "{{ ")?;
                for (i, (name, field_type)) in fields.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}: {}", name, field_type)?;
                }
                write!(f, " }}")
            },
            Type::Function(params, return_type) => {
                write!(f, "(")?;
                for (i, param_type) in params.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", param_type)?;
                }
                write!(f, ") -> {}", return_type)
            },
            Type::TypeVar(id) => write!(f, "T{}", id),
            Type::Generic(name, type_args) => {
                write!(f, "{}<", name)?;
                for (i, arg) in type_args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ">")
            },
        }
    }
}

/// Environment for storing type information
#[derive(Debug, Clone)]
pub struct TypeEnvironment {
    /// Maps variable names to their types
    pub variables: HashMap<String, Type>,
    
    /// Counter for generating unique type variables
    pub next_type_var: usize,
}

impl TypeEnvironment {
    /// Create a new empty type environment
    pub fn new() -> Self {
        TypeEnvironment {
            variables: HashMap::new(),
            next_type_var: 0,
        }
    }
    
    /// Generate a new type variable
    pub fn fresh_type_var(&mut self) -> Type {
        let id = self.next_type_var;
        self.next_type_var += 1;
        Type::TypeVar(id)
    }
    
    /// Add a variable to the environment
    pub fn add_variable(&mut self, name: String, ty: Type) {
        self.variables.insert(name, ty);
    }
    
    /// Look up a variable's type
    pub fn lookup_variable(&self, name: &str) -> Option<&Type> {
        self.variables.get(name)
    }
}