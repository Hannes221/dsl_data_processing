#[derive(Debug)]
#[allow(dead_code)]
pub enum RuntimeError {
    UndefinedVariable(String),
    UndefinedField(String),
    NotARecord(String),
    NotAFunction(String),
    WrongNumberOfArguments(usize, usize),
    TypeMismatch(String, String, String),
    ExpectedArray(String),
    ExpectedLambda,
    UnsupportedOperator(String),
    Other(String),
    DivisionByZero,
    ExpectedFunction(String),
    DataSourceError(String),
    UnknownDataset(String),
} 