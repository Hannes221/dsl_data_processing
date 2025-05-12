use std::collections::HashMap;
use crate::type_system::types::Type;
use super::operations::*;

/// The core Expression enum that represents all possible expressions in our DSL
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum Expr {
    // Data sources
    DataSource(DataSourceExpr),
    
    // Operations
    Filter(Box<FilterExpr>),
    Map(Box<MapExpr>),
    GroupBy(Box<GroupByExpr>),
    Join(Box<JoinExpr>),
    Aggregate(Box<AggregateExpr>),
    BinaryOp(Box<BinaryOpExpr>),
    
    // Values
    Literal(LiteralExpr),
    RecordLiteral(RecordLiteralExpr),
    Variable(VariableExpr),
    
    // Computation
    FunctionCall(Box<FunctionCallExpr>),
    Lambda(Box<LambdaExpr>),
    
    // Object-oriented features
    FieldAccess(Box<FieldAccessExpr>),
    MethodCall(Box<MethodCallExpr>),
}

/// Represents a literal value
#[derive(Debug, Clone)]
pub struct LiteralExpr {
    pub value: Value,
    pub inferred_type: Option<Type>,
}

#[derive(Debug, Clone)]
pub struct RecordLiteralExpr {
    pub fields: HashMap<String, Expr>,
    pub inferred_type: Option<Type>,
}

/// Represents a variable reference
#[derive(Debug, Clone)]
pub struct VariableExpr {
    pub name: String,
    pub inferred_type: Option<Type>,
}

/// Represents a function call
#[derive(Debug, Clone)]
pub struct FunctionCallExpr {
    pub function: Box<Expr>,
    pub arguments: Vec<Expr>,
    pub inferred_type: Option<Type>,
}

/// Represents a lambda expression (anonymous function)
#[derive(Debug, Clone)]
pub struct LambdaExpr {
    pub parameters: Vec<String>,
    pub body: Box<Expr>,
    pub inferred_type: Option<Type>,
}

/// Represents field access on a record/object
#[derive(Debug, Clone)]
pub struct FieldAccessExpr {
    pub object: Box<Expr>,
    pub field: String,
    pub inferred_type: Option<Type>,
}

/// Represents a method call on an object
#[derive(Debug, Clone)]
pub struct MethodCallExpr {
    pub object: Box<Expr>,
    pub method: String,
    pub arguments: Vec<Expr>,
    pub inferred_type: Option<Type>,
}

/// Represents possible runtime values
#[derive(Debug, Clone)]
pub enum Value {
    Int(i64),
    Float(f64),
    String(String),
    Boolean(bool),
    Array(Vec<Value>),
    Record(HashMap<String, Value>),
    Function(Box<Function>),
    Null,
}

/// Represents a function value
#[derive(Debug, Clone)]
pub struct Function {
    pub parameters: Vec<String>,
    pub body: Box<Expr>,
    pub closure: HashMap<String, Value>,
}

impl Function {
    pub fn call(&self, args: Vec<Value>) -> Result<Value, crate::interpreter::runtime_error::RuntimeError> {
        use crate::interpreter::{Interpreter, RuntimeError};
        
        // Check if the number of arguments matches the number of parameters
        if args.len() != self.parameters.len() {
            return Err(RuntimeError::WrongNumberOfArguments(
                args.len(),
                self.parameters.len(),
            ));
        }
        
        // Create a new interpreter with the captured environment
        let mut interpreter = Interpreter::new();
        
        // Add captured variables from the closure
        for (name, value) in &self.closure {
            interpreter.env.set_variable(name.clone(), value.clone());
        }
        
        // Bind arguments to parameters
        for (i, param) in self.parameters.iter().enumerate() {
            interpreter.env.set_variable(param.clone(), args[i].clone());
        }
        
        // Evaluate the function body
        interpreter.evaluate(&self.body)
    }
}