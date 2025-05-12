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
    BinaryOp(BinaryOpExpr),

    // Values
    Literal(LiteralExpr),
    RecordLiteral(RecordLiteralExpr),
    Variable(VariableExpr),

    // Computation
    FunctionCall(FunctionCallExpr),
    Lambda(LambdaExpr),

    // Object-oriented features
    FieldAccess(FieldAccessExpr),
    MethodCall(MethodCallExpr),
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

- Static typing with interpretation-time checks
- Type inference using Hindley-Milner algorithm

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

## Concepts covered

- Type Systems (Static Typing, Polymorphism, Compile-time checks)
- Type inference (Hindley-Milner Algorithm)
- Abstract Syntax Tree
- Object-Oriented Concepts (Classes, Traits, Inheritance)
- Functional Concepts (Pure functions, Higher-order functions, Lambdas/Anonymous Functions)
- Memory management (Ownership, Borrowing)
- Lexical Analysis (Tokenization)
- Semantic Analysis (Type checking)
- Parsing (Parsing Expressions, Parsing Statements, Parsing Declarations)
- Interpreter/Evaluator
- Introspection
- Data Operations (Filtering, Mapping, Grouping, Joining, Aggregating)
- Data Sources (CSV, more to be added...)

## Future Enhancements

### Performance Optimizations

#### Parallel Processing with Rayon

The interpreter could be enhanced with Rayon to parallelize data operations:

- Parallel execution of `filter`, `map`, and other transformations
- Work-stealing scheduler for efficient CPU utilization
- Simple API transition (changing `.iter()` to `.par_iter()`)
- Significant performance improvements for large datasets

#### Columnar Data Processing with Polars/Arrow

Integration with Polars or Apache Arrow would provide:

- High-performance columnar data processing
- Memory-efficient data representation
- Vectorized operations for faster computation
- Advanced analytical capabilities
- Interoperability with other data tools and formats

### Streaming Data Processing

#### Kafka Integration

The DSL could be extended to support streaming data via Kafka:

- Real-time data processing pipelines
- Continuous queries on streaming data
- Windowed operations (sliding, tumbling windows)
- Exactly-once processing semantics
- Integration with existing Kafka ecosystems

Implementation approach:

```rust
// Example of potential Kafka source syntax
let stream = kafka_source("topic_name", {
    server: "localhost:9092",
    group_id: "my_consumer_group",
    schema: {
        timestamp: Int,
        user_id: String,
        action: String
    }
})
.filter(|event| event.action == "purchase")
.window(sliding(minutes(5), seconds(30)))
.aggregate(|window| {
    count: window.count(),
    total: window.sum(|e| e.amount)
});
```

### Custom Transformers

The DSL could be extended with a plugin system for custom transformers:

- User-defined operations beyond the standard library
- Domain-specific transformations
- Integration with external libraries and tools
- Reusable transformation components

Implementation approach:

```rust
// Registering a custom transformer
register_transformer("sentiment_analysis", |text: String| -> Record {
    // Integration with external NLP library
    let sentiment = external_nlp::analyze_sentiment(&text);
    {
        score: sentiment.score,
        magnitude: sentiment.magnitude,
        classification: sentiment.classification
    }
});

// Using the custom transformer in a pipeline
let results = data_source("customer_reviews.csv")
    .map(|review| {
        review_text: review.text,
        sentiment: sentiment_analysis(review.text)
    })
    .filter(|r| r.sentiment.score > 0.7)
    .group_by(|r| r.sentiment.classification);
```

## Distribution Options

The DSL can be made available to users through multiple channels:

### Rust Crate

- Publish to crates.io for Rust developers
- Provide both library and binary interfaces

### Command-Line Interface

- Standalone executable for processing DSL scripts
- Support for file input/output and pipeline configuration

### Python Package

- PyO3 bindings to make the DSL accessible to Python users
- Integration with pandas and the Python data science ecosystem
- Maintain Rust performance while providing a Pythonic API

### Docker Container

- Consistent environment across platforms
- Easy integration with data pipelines and CI/CD workflows

## TODO:

- Work on TODOS in the codebase
- Lazy parsing
- Error handling
- Performance optimizations/Batch processing
- Streaming (Use in production ML pipelines)
- Documentation
- Tests
- Benchmarks
- Open Sourcing
