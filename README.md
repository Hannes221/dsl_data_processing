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

## Future Enhancements

### Performance Optimizations

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

## TODO:

- Work on TODOS in the codebase
- Lazy parsing
- Error handling
- Performance optimizations/Batch processing
- Streaming (Use in production ML pipelines)
- Documentation
- Tests
- Benchmarks
