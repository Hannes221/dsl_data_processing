# Type-Inferred DSL for Data Processing

This project implements a domain-specific language (DSL) for data processing with automatic type inference. The DSL allows users to express data transformation pipelines in a concise and type-safe manner.

## Overview

The DSL provides a set of composable operations for data manipulation, filtering, transformation, and aggregation. The type system automatically infers types throughout the pipeline, catching type errors at compile time while maintaining a clean syntax.

## Abstract Syntax Tree (AST)

The core of our DSL is defined by its Abstract Syntax Tree, which represents the structure of valid programs:

```rust
// Base expression type
pub enum Expr {
// Data sources
DataSource(DataSourceExpr),
// Operations
Filter(FilterExpr),
Map(MapExpr),
GroupBy(GroupByExpr),
Join(JoinExpr),
Aggregate(AggregateExpr),
// Values
Literal(LiteralExpr),
Variable(VariableExpr),
FunctionCall(FunctionCallExpr),
Lambda(LambdaExpr),
}
// Each expression type has its own struct with relevant fields
pub struct FilterExpr {
pub input: Box<Expr>,
pub predicate: Box<Expr>,
}
// And so on for other expression types...
```

Our DSL implements a static type system with inference:

```rust
// Type system definition
pub enum Type {
Int,
Float,
String,
Boolean,
Array(Box<Type>),
Record(HashMap<String, Type>),
Function(Vec<Type>, Box<Type>), // Input types and return type
TypeVar(usize), // For type inference
}
```

The type inference engine uses unification to determine types without requiring explicit annotations.

## Implementation Plan

### 1. Lexer and Parser

We'll use the `nom` crate to parse our DSL syntax into the AST:

```rust
// Lexer will convert input text into tokens
fn lex(input: &str) -> Result<Vec<Token>, LexError> {
// Implementation
}

// Parser converts tokens into AST
fn parse(tokens: Vec<Token>) -> Result<Expr, ParseError> {
// Implementation
}
```

### 2. Type Inference

The type inference system will implement Hindley-Milner type inference:

```rust
// Type inference engine
fn infer_types(expr: &mut Expr, env: &mut TypeEnv) -> Result<Type, TypeError> {
// Implementation
}
```

### 3. Interpreter

The interpreter will evaluate the AST:

```rust
// Interpreter
fn evaluate(expr: &Expr, env: &mut Environment) -> Result<Value, RuntimeError> {
match expr {
Expr::DataSource(ds) => evaluate_data_source(ds, env),
Expr::Filter(filter) => evaluate_filter(filter, env),
// Other cases...
}
}
```

## Features

This implementation will demonstrate:

### Type Systems

- Static typing with compile-time checks
- Type inference using Hindley-Milner algorithm
- Polymorphic types for generic operations

### Object-Oriented Concepts

- Classes and objects for data representation
- Inheritance through trait implementation
- Polymorphism via dynamic dispatch

### Functional Concepts

- Pure functions for data transformations
- Anonymous functions (lambdas) for predicates
- Higher-order functions (map, filter, reduce)

### Advanced Topics

- Memory management with ownership and borrowing
- Lexical analysis and parsing
- Semantic analysis with type checking
- Interpreter implementation

## Example Usage

// Define a data source
let users = data_source("users.csv", {
id: Int,
name: String,
age: Int,
department: String
});
// Create a pipeline
let result = users
.filter(|user| user.age > 30)
.map(|user| {
name: user.name,
department: user.department
})
.group_by(|user| user.department)
.map(|(key, group)| {
department: key,
count: group.len(),
names: group.map(|u| u.name)
});

## Development Phases

1. **Phase 1**: Define the AST and type system
2. **Phase 2**: Implement the lexer and parser
3. **Phase 3**: Build the type inference engine
4. **Phase 4**: Create the interpreter
5. **Phase 5**: Add standard library functions
6. **Phase 6**: Optimize and refine
