use super::expressions::Expr;
use crate::type_system::types::Type;
use crate::parser::lexer::Token;
use std::collections::HashMap;

/// Represents a data source (e.g., CSV file, database table)
#[derive(Debug, Clone)]
pub struct DataSourceExpr {
    pub source: String,
    pub schema: HashMap<String, Type>,
    pub inferred_type: Option<Type>,
}

/// Represents a filter operation
#[derive(Debug, Clone)]
pub struct FilterExpr {
    pub input: Box<Expr>,
    pub predicate: Box<Expr>,
    pub inferred_type: Option<Type>,
}

/// Represents a map operation
#[derive(Debug, Clone)]
pub struct MapExpr {
    pub input: Box<Expr>,
    pub transform: Box<Expr>,
    pub inferred_type: Option<Type>,
}

/// Represents a group-by operation
#[derive(Debug, Clone)]
pub struct GroupByExpr {
    pub input: Box<Expr>,
    pub key_selector: Box<Expr>,
    pub inferred_type: Option<Type>,
}

/// Represents a join operation
#[derive(Debug, Clone)]
pub struct JoinExpr {
    pub left: Box<Expr>,
    pub right: Box<Expr>,
    pub left_key: Box<Expr>,
    pub right_key: Box<Expr>,
    pub result_selector: Box<Expr>,
    pub inferred_type: Option<Type>,
}

/// Represents an aggregate operation
#[derive(Debug, Clone)]
pub struct AggregateExpr {
    pub input: Box<Expr>,
    pub aggregator: Box<Expr>,
    pub inferred_type: Option<Type>,
}

/// Represents a binary operation
#[derive(Debug, Clone)]
pub struct BinaryOpExpr {
    pub left: Box<Expr>,
    pub operator: Token,
    pub right: Box<Expr>,
    pub inferred_type: Option<Type>,
}