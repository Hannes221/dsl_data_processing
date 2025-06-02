use std::collections::HashMap;
use crate::ast::*;
use crate::ast::operations::*;
use crate::parser::lexer::Token;
use super::environment::Environment;
use super::runtime_error::RuntimeError;
use crate::ast::expressions::Function; 
use crate::data_sources::DataSourceFactory;
use rayon::prelude::*;
use std::sync::Arc;

pub struct Interpreter {
    pub env: Environment,
}

impl Interpreter {
    pub fn new() -> Self {
        Interpreter {
            env: Environment::new(),
        }
    }
    
    pub fn evaluate(&mut self, expr: &Expr) -> Result<Value, RuntimeError> {
        match expr {
            Expr::Literal(lit) => Ok(lit.value.clone()),
            Expr::Variable(var) => self.evaluate_variable(var),
            Expr::FunctionCall(call) => self.evaluate_function_call(call),
            Expr::Lambda(lambda) => self.evaluate_lambda(lambda),
            Expr::DataSource(ds) => self.evaluate_data_source(ds),
            Expr::Filter(filter) => self.evaluate_filter(filter),
            Expr::Map(map) => self.evaluate_map(map),
            Expr::GroupBy(group_by) => self.evaluate_group_by(group_by),
            Expr::Join(join) => self.evaluate_join(join),
            Expr::Aggregate(agg) => self.evaluate_aggregate(agg),
            Expr::FieldAccess(field) => self.evaluate_field_access(field),
            Expr::MethodCall(method) => self.evaluate_method_call(method),
            Expr::BinaryOp(binary_op) => self.evaluate_binary_op(binary_op),
            Expr::RecordLiteral(record) => self.evaluate_record_literal(record),
        }
    }
    
    fn evaluate_variable(&self, var: &VariableExpr) -> Result<Value, RuntimeError> {
        match self.env.get_variable(&var.name) {
            Some(value) => Ok(value.clone()),
            None => Err(RuntimeError::UndefinedVariable(var.name.clone())),
        }
    }

    fn evaluate_function_call(&mut self, call: &FunctionCallExpr) -> Result<Value, RuntimeError> {
        match &*call.function {
            Expr::Variable(var) => {
                // Get the function value first
                let func_value = match self.env.get_variable(&var.name) {
                    Some(value) => value.clone(),
                    None => return Err(RuntimeError::UndefinedVariable(var.name.clone())),
                };
                
                // Check if it's a function
                if let Value::Function(f) = func_value {
                    // Create a vector to hold evaluated arguments
                    let mut evaluated_args = Vec::new();
                    
                    // Evaluate each argument one by one
                    for i in 0..call.arguments.len() {
                        let arg_expr = &call.arguments[i];
                        let arg_value = self.evaluate(arg_expr)?;
                        evaluated_args.push(arg_value);
                    }
                    
                    // Call the function with the evaluated arguments
                    f.call(evaluated_args)
                } else {
                    Err(RuntimeError::NotAFunction(var.name.clone()))
                }
            },
            _ => Err(RuntimeError::ExpectedFunction(format!("{:?}", call.function))),
        }
    }

    fn evaluate_lambda(&mut self, lambda: &LambdaExpr) -> Result<Value, RuntimeError> {
        // Create a function value from the lambda expression
        let function = Function {
            parameters: lambda.parameters.clone(),
            body: lambda.body.clone(),
            closure: self.env.get_variables().clone(),
        };
        
        Ok(Value::Function(Box::new(function)))
    }
    
    fn evaluate_data_source(&mut self, ds: &DataSourceExpr) -> Result<Value, RuntimeError> {
        // Create the appropriate data source
        let data_source = match DataSourceFactory::create_data_source(&ds.source) {
            Ok(ds) => ds,
            Err(e) => return Err(RuntimeError::DataSourceError(format!("{:?}", e))),
        };
        
        // Load the data
        match data_source.load(&ds.source) {
            Ok(records) => Ok(Value::Array(records)),
            Err(e) => Err(RuntimeError::DataSourceError(format!("{:?}", e))),
        }
    }
    
    fn evaluate_filter(&mut self, filter: &FilterExpr) -> Result<Value, RuntimeError> {
        let input_value = self.evaluate(&filter.input)?;
        
        if let Value::Array(elements) = input_value {
            // Use Arc to share environment read-only
            let env = Arc::new(self.env.clone());
            
            // Pre-extract lambda information to avoid repeated pattern matching
            if let Expr::Lambda(lambda) = filter.predicate.as_ref() {
                if lambda.parameters.len() != 1 {
                    return Err(RuntimeError::Other("Filter predicate must have exactly one parameter".to_string()));
                }
                
                let param_name = Arc::new(lambda.parameters[0].clone());
                let lambda_body = Arc::new(lambda.body.clone());
                
                let result: Vec<Value> = elements.into_par_iter()
                    .filter_map(|element| {
                        // Create minimal scope with shared environment
                        let mut local_env = Environment::with_parent(env.clone());
                        local_env.set_variable(param_name.as_ref().clone(), element.clone());
                        
                        let mut local_interpreter = Interpreter { env: local_env };
                        
                        match local_interpreter.evaluate(&lambda_body) {
                            Ok(Value::Boolean(true)) => Some(element),
                            _ => None,
                        }
                    })
                    .collect();
                
                Ok(Value::Array(result))
            } else {
                Err(RuntimeError::ExpectedLambda)
            }
        } else {
            Err(RuntimeError::ExpectedArray(format!("{:?}", input_value)))
        }
    }
    
    fn evaluate_map(&mut self, map: &MapExpr) -> Result<Value, RuntimeError> {
        let input_value = self.evaluate(&map.input)?;
        
        if let Value::Array(elements) = input_value {
            let env = Arc::new(self.env.clone());
            
            if let Expr::Lambda(lambda) = map.transform.as_ref() {
                if lambda.parameters.len() != 1 {
                    return Err(RuntimeError::Other("Map transform must have exactly one parameter".to_string()));
                }
                
                let param_name = Arc::new(lambda.parameters[0].clone());
                let lambda_body = Arc::new(lambda.body.clone());
                
                let result: Vec<Value> = elements.into_par_iter()
                    .map(|element| {
                        let mut local_env = Environment::with_parent(env.clone());
                        local_env.set_variable(param_name.as_ref().clone(), element);
                        
                        let mut local_interpreter = Interpreter { env: local_env };
                        local_interpreter.evaluate(&lambda_body).unwrap_or(Value::Null)
                    })
                    .collect();
                
                Ok(Value::Array(result))
            } else {
                Err(RuntimeError::ExpectedLambda)
            }
        } else {
            Err(RuntimeError::ExpectedArray(format!("{:?}", input_value)))
        }
    }

    fn evaluate_binary_op(&mut self, binary_op: &BinaryOpExpr) -> Result<Value, RuntimeError> {
        let left = self.evaluate(&binary_op.left)?;
        let right = self.evaluate(&binary_op.right)?;
        
        match &binary_op.operator {
            Token::Plus => self.evaluate_plus(&left, &right),
            Token::Minus => self.evaluate_minus(&left, &right),
            Token::Multiply => self.evaluate_multiply(&left, &right),
            Token::Divide => self.evaluate_divide(&left, &right),
            Token::Equal => self.evaluate_equal(&left, &right),
            Token::NotEqual => self.evaluate_not_equal(&left, &right),
            Token::Greater => self.evaluate_greater(&left, &right),
            Token::Less => self.evaluate_less(&left, &right),
            Token::GreaterEqual => self.evaluate_greater_equal(&left, &right),
            Token::LessEqual => self.evaluate_less_equal(&left, &right),
            Token::And => self.evaluate_and(&left, &right),
            Token::Or => self.evaluate_or(&left, &right),
            _ => Err(RuntimeError::UnsupportedOperator(format!("{:?}", binary_op.operator))),
        }
    }
    
    fn evaluate_plus(&self, left: &Value, right: &Value) -> Result<Value, RuntimeError> {
        match (left, right) {
            (Value::Int(l), Value::Int(r)) => Ok(Value::Int(l + r)),
            (Value::Float(l), Value::Float(r)) => Ok(Value::Float(l + r)),
            (Value::Int(l), Value::Float(r)) => Ok(Value::Float(*l as f64 + r)),
            (Value::Float(l), Value::Int(r)) => Ok(Value::Float(l + *r as f64)),
            (Value::String(l), Value::String(r)) => Ok(Value::String(l.clone() + r)),
            _ => Err(RuntimeError::TypeMismatch(
                format!("{:?}", left),
                format!("{:?}", right),
                "addition".to_string(),
            )),
        }
    }

    fn evaluate_minus(&self, left: &Value, right: &Value) -> Result<Value, RuntimeError> {
        match (left, right) {
            (Value::Int(l), Value::Int(r)) => Ok(Value::Int(l - r)),
            (Value::Float(l), Value::Float(r)) => Ok(Value::Float(l - r)),
            (Value::Int(l), Value::Float(r)) => Ok(Value::Float(*l as f64 - r)),
            (Value::Float(l), Value::Int(r)) => Ok(Value::Float(l - *r as f64)),
            _ => Err(RuntimeError::TypeMismatch(
                format!("{:?}", left),
                format!("{:?}", right),
                "subtraction".to_string(),
            )),
        }
    }
    
    fn evaluate_multiply(&self, left: &Value, right: &Value) -> Result<Value, RuntimeError> {
        match (left, right) {
            (Value::Int(l), Value::Int(r)) => Ok(Value::Int(l * r)),
            (Value::Float(l), Value::Float(r)) => Ok(Value::Float(l * r)),
            (Value::Int(l), Value::Float(r)) => Ok(Value::Float(*l as f64 * r)),
            (Value::Float(l), Value::Int(r)) => Ok(Value::Float(l * *r as f64)),
            _ => Err(RuntimeError::TypeMismatch(
                format!("{:?}", left),
                format!("{:?}", right),
                "multiplication".to_string(),
            )),
        }
    }

    fn evaluate_divide(&self, left: &Value, right: &Value) -> Result<Value, RuntimeError> {
        match (left, right) {
            (Value::Int(l), Value::Int(r)) => {
                if *r == 0 {
                    return Err(RuntimeError::DivisionByZero);
                }
                Ok(Value::Float(*l as f64 / *r as f64))
            },
            (Value::Float(l), Value::Float(r)) => {
                if *r == 0.0 {
                    return Err(RuntimeError::DivisionByZero);
                }
                Ok(Value::Float(*l / *r))
            },
            (Value::Int(l), Value::Float(r)) => {
                if *r == 0.0 {
                    return Err(RuntimeError::DivisionByZero);
                }
                Ok(Value::Float(*l as f64 / *r))
            },
            (Value::Float(l), Value::Int(r)) => {
                if *r == 0 {
                    return Err(RuntimeError::DivisionByZero);
                }
                Ok(Value::Float(*l / *r as f64))
            },
            _ => Err(RuntimeError::TypeMismatch(
                format!("{:?}", left),
                format!("{:?}", right),
                "division".to_string(),
            )),
        }
    }
    
    fn evaluate_equal(&self, left: &Value, right: &Value) -> Result<Value, RuntimeError> {
        match (left, right) {
            (Value::Int(l), Value::Int(r)) => Ok(Value::Boolean(l == r)),
            (Value::Float(l), Value::Float(r)) => Ok(Value::Boolean(l == r)),
            (Value::String(l), Value::String(r)) => Ok(Value::Boolean(l == r)),
            (Value::Boolean(l), Value::Boolean(r)) => Ok(Value::Boolean(l == r)),
            _ => Err(RuntimeError::TypeMismatch(
                format!("{:?}", left),
                format!("{:?}", right),
                "equality comparison".to_string(),
            )),
        }
    }

    fn evaluate_not_equal(&self, left: &Value, right: &Value) -> Result<Value, RuntimeError> {
        match (left, right) {
            (Value::Int(l), Value::Int(r)) => Ok(Value::Boolean(l != r)),
            (Value::Float(l), Value::Float(r)) => Ok(Value::Boolean(l != r)),
            (Value::String(l), Value::String(r)) => Ok(Value::Boolean(l != r)),
            (Value::Boolean(l), Value::Boolean(r)) => Ok(Value::Boolean(l != r)),
            _ => Err(RuntimeError::TypeMismatch(
                format!("{:?}", left),
                format!("{:?}", right),
                "inequality comparison".to_string(),
            )),
        }
    }

    fn evaluate_greater(&self, left: &Value, right: &Value) -> Result<Value, RuntimeError> {
        match (left, right) {
            (Value::Int(l), Value::Int(r)) => Ok(Value::Boolean(*l > *r)),
            (Value::Float(l), Value::Float(r)) => Ok(Value::Boolean(*l > *r)),
            (Value::Int(l), Value::Float(r)) => Ok(Value::Boolean(*l as f64 > *r)),
            (Value::Float(l), Value::Int(r)) => Ok(Value::Boolean(*l > *r as f64)),
            _ => Err(RuntimeError::TypeMismatch(
                format!("{:?}", left),
                format!("{:?}", right),
                "greater than comparison".to_string(),
            )),
        }
    }

    fn evaluate_less(&self, left: &Value, right: &Value) -> Result<Value, RuntimeError> {
        match (left, right) {
            (Value::Int(l), Value::Int(r)) => Ok(Value::Boolean(l.lt(r))),
            (Value::Float(l), Value::Float(r)) => Ok(Value::Boolean(l.lt(r))),
            (Value::Int(l), Value::Float(r)) => Ok(Value::Boolean((*l as f64).lt(r))),
            (Value::Float(l), Value::Int(r)) => Ok(Value::Boolean(l.lt(&(*r as f64)))),
            _ => Err(RuntimeError::TypeMismatch(
                format!("{:?}", left),
                format!("{:?}", right),
                "less than comparison".to_string(),
            )),
        }
    }

    fn evaluate_greater_equal(&self, left: &Value, right: &Value) -> Result<Value, RuntimeError> {
        match (left, right) {
            (Value::Int(l), Value::Int(r)) => Ok(Value::Boolean(*l >= *r)),
            (Value::Float(l), Value::Float(r)) => Ok(Value::Boolean(*l >= *r)),
            (Value::Int(l), Value::Float(r)) => Ok(Value::Boolean(*l as f64 >= *r)),
            (Value::Float(l), Value::Int(r)) => Ok(Value::Boolean(*l >= *r as f64)),
            _ => Err(RuntimeError::TypeMismatch(
                format!("{:?}", left),
                format!("{:?}", right),
                "greater than or equal comparison".to_string(),
            )),
        }
    }
    
    fn evaluate_less_equal(&self, left: &Value, right: &Value) -> Result<Value, RuntimeError> {
        match (left, right) {
            (Value::Int(l), Value::Int(r)) => Ok(Value::Boolean(*l <= *r)),
            (Value::Float(l), Value::Float(r)) => Ok(Value::Boolean(*l <= *r)),
            (Value::Int(l), Value::Float(r)) => Ok(Value::Boolean(*l as f64 <= *r)),
            (Value::Float(l), Value::Int(r)) => Ok(Value::Boolean(*l <= *r as f64)),
            _ => Err(RuntimeError::TypeMismatch(
                format!("{:?}", left),
                format!("{:?}", right),
                "less than or equal comparison".to_string(),
            )),
        }
    }
    
    fn evaluate_and(&self, left: &Value, right: &Value) -> Result<Value, RuntimeError> {
        match (left, right) {
            (Value::Boolean(l), Value::Boolean(r)) => Ok(Value::Boolean(*l && *r)),
            _ => Err(RuntimeError::TypeMismatch(
                format!("{:?}", left),
                format!("{:?}", right),
                "logical AND".to_string(),
            )),
        }
    }
    
    fn evaluate_or(&self, left: &Value, right: &Value) -> Result<Value, RuntimeError> {
        match (left, right) {
            (Value::Boolean(l), Value::Boolean(r)) => Ok(Value::Boolean(*l || *r)),
            _ => Err(RuntimeError::TypeMismatch(
                format!("{:?}", left),
                format!("{:?}", right),
                "logical OR".to_string(),
            )),
        }
    }
    
    fn evaluate_record_literal(&mut self, record: &RecordLiteralExpr) -> Result<Value, RuntimeError> {
        let mut fields = HashMap::new();
        
        for (name, expr) in &record.fields {
            let value = self.evaluate(expr)?;
            fields.insert(name.clone(), value);
        }
        
        Ok(Value::Record(fields))
    }
    
    fn evaluate_field_access(&mut self, field: &FieldAccessExpr) -> Result<Value, RuntimeError> {
        let object = self.evaluate(&field.object)?;
        
        match object {
            Value::Record(fields) => {
                match fields.get(&field.field) {
                    Some(value) => Ok(value.clone()),
                    None => Err(RuntimeError::UndefinedField(field.field.clone())),
                }
            },
            _ => Err(RuntimeError::NotARecord(format!("{:?}", object))),
        }
    }

    fn evaluate_group_by(&mut self, group_by: &GroupByExpr) -> Result<Value, RuntimeError> {
        // Evaluate the input expression
        let input_value = self.evaluate(&group_by.input)?;
        
        // Input should be an array
        if let Value::Array(elements) = input_value {
            // Clone the environment for parallel processing
            let env = self.env.clone();
            
            // Create a parallel iterator over the elements
            let groups: HashMap<String, Vec<Value>> = elements.into_par_iter()
                .map(|element| {
                    // Create a new scope for the lambda parameter
                    let mut local_env = env.clone();
                    
                    // Bind the current element to the parameter name
                    match &*group_by.key_selector {
                        Expr::Lambda(lambda) => {
                            if lambda.parameters.len() != 1 {
                                return (String::new(), Vec::new());
                            }
                            
                            let param_name = &lambda.parameters[0];
                            local_env.set_variable(param_name.clone(), element.clone());
                            
                            // Create a temporary interpreter with the local environment
                            let mut local_interpreter = Interpreter { env: local_env };
                            
                            // Evaluate the key selector
                            let key_result = local_interpreter.evaluate(&lambda.body).unwrap_or(Value::Null);
                            
                            // Convert the key to a string for grouping
                            let key_string = match &key_result {
                                Value::String(s) => s.clone(),
                                Value::Int(i) => i.to_string(),
                                Value::Float(f) => f.to_string(),
                                Value::Boolean(b) => b.to_string(),
                                _ => format!("{:?}", key_result),
                            };
                            
                            (key_string, vec![element])
                        },
                        _ => (String::new(), Vec::new()),
                    }
                })
                .fold(
                    || HashMap::new(),
                    |mut acc, (key, mut values)| {
                        if !key.is_empty() {
                            acc.entry(key).or_insert_with(Vec::new).append(&mut values);
                        }
                        acc
                    }
                )
                .reduce(
                    || HashMap::new(),
                    |mut acc, map| {
                        for (key, mut values) in map {
                            acc.entry(key).or_insert_with(Vec::new).append(&mut values);
                        }
                        acc
                    }
                );
            
            // Convert the groups to an array of records
            let result: Vec<Value> = groups.into_par_iter()
                .map(|(key, group)| {
                    let mut record = HashMap::new();
                    record.insert("key".to_string(), Value::String(key));
                    record.insert("data".to_string(), Value::Array(group));
                    Value::Record(record)
                })
                .collect();
            
            Ok(Value::Array(result))
        } else {
            Err(RuntimeError::ExpectedArray(format!("{:?}", input_value)))
        }
    }

    fn evaluate_join(&mut self, join: &JoinExpr) -> Result<Value, RuntimeError> {
        // Evaluate the left and right inputs
        let left_value = self.evaluate(&join.left)?;
        let right_value = self.evaluate(&join.right)?;
        
        // Both inputs should be arrays
        match (&left_value, &right_value) {
            (Value::Array(left_elements), Value::Array(right_elements)) => {
                // Clone the environment for parallel processing
                let env = self.env.clone();
                
                // Extract keys from left elements in parallel
                let left_keys: HashMap<String, Vec<Value>> = left_elements.par_iter()
                    .map(|left_element| {
                        // Create a new scope for the lambda parameter
                        let mut local_env = env.clone();
                        
                        // Bind the current element to the parameter name
                        match &*join.left_key {
                            Expr::Lambda(lambda) => {
                                if lambda.parameters.len() != 1 {
                                    return (String::new(), Vec::new());
                                }
                                
                                let param_name = &lambda.parameters[0];
                                local_env.set_variable(param_name.clone(), left_element.clone());
                                
                                // Create a temporary interpreter with the local environment
                                let mut local_interpreter = Interpreter { env: local_env };
                                
                                // Evaluate the key selector
                                let key_result = local_interpreter.evaluate(&lambda.body).unwrap_or(Value::Null);
                                
                                // Convert the key to a string for joining
                                let key_string = match &key_result {
                                    Value::String(s) => s.clone(),
                                    Value::Int(i) => i.to_string(),
                                    Value::Float(f) => f.to_string(),
                                    Value::Boolean(b) => b.to_string(),
                                    _ => format!("{:?}", key_result),
                                };
                                
                                (key_string, vec![left_element.clone()])
                            },
                            _ => (String::new(), Vec::new()),
                        }
                    })
                    .fold(
                        || HashMap::new(),
                        |mut acc, (key, mut values)| {
                            if !key.is_empty() {
                                acc.entry(key).or_insert_with(Vec::new).append(&mut values);
                            }
                            acc
                        }
                    )
                    .reduce(
                        || HashMap::new(),
                        |mut acc, map| {
                            for (key, mut values) in map {
                                acc.entry(key).or_insert_with(Vec::new).append(&mut values);
                            }
                            acc
                        }
                    );
                
                // Join with right elements in parallel
                let result: Vec<Value> = right_elements.par_iter()
                    .flat_map(|right_element| {
                        // Create a new scope for the lambda parameter
                        let mut local_env = env.clone();
                        
                        // Bind the current element to the parameter name
                        match &*join.right_key {
                            Expr::Lambda(lambda) => {
                                if lambda.parameters.len() != 1 {
                                    return Vec::new();
                                }
                                
                                let param_name = &lambda.parameters[0];
                                local_env.set_variable(param_name.clone(), right_element.clone());
                                
                                // Create a temporary interpreter with the local environment
                                let mut local_interpreter = Interpreter { env: local_env };
                                
                                // Evaluate the key selector
                                let key_result = local_interpreter.evaluate(&lambda.body).unwrap_or(Value::Null);
                                
                                // Convert the key to a string for joining
                                let key_string = match &key_result {
                                    Value::String(s) => s.clone(),
                                    Value::Int(i) => i.to_string(),
                                    Value::Float(f) => f.to_string(),
                                    Value::Boolean(b) => b.to_string(),
                                    _ => format!("{:?}", key_result),
                                };
                                
                                // Find matching left elements
                                if let Some(matching_left) = left_keys.get(&key_string) {
                                    matching_left.iter()
                                        .map(|left_element| {
                                            // Create a new scope for the result selector
                                            let mut result_env = env.clone();
                                            
                                            // Apply the result selector to create the joined record
                                            match &*join.result_selector {
                                                Expr::Lambda(lambda) => {
                                                    if lambda.parameters.len() != 2 {
                                                        return Value::Null;
                                                    }
                                                    
                                                    // Bind the left and right elements to the parameter names
                                                    let left_param = &lambda.parameters[0];
                                                    let right_param = &lambda.parameters[1];
                                                    result_env.set_variable(left_param.clone(), left_element.clone());
                                                    result_env.set_variable(right_param.clone(), right_element.clone());
                                                    
                                                    // Create a temporary interpreter with the local environment
                                                    let mut result_interpreter = Interpreter { env: result_env };
                                                    
                                                    // Evaluate the result selector
                                                    result_interpreter.evaluate(&lambda.body).unwrap_or(Value::Null)
                                                },
                                                _ => {
                                                    // Default join behavior if no lambda is provided
                                                    let mut record = HashMap::new();
                                                    record.insert("key".to_string(), Value::String(key_string.clone()));
                                                    record.insert("left".to_string(), left_element.clone());
                                                    record.insert("right".to_string(), right_element.clone());
                                                    Value::Record(record)
                                                }
                                            }
                                        })
                                        .collect()
                                } else {
                                    Vec::new()
                                }
                            },
                            _ => Vec::new(),
                        }
                    })
                    .collect();
                
                Ok(Value::Array(result))
            },
            _ => Err(RuntimeError::ExpectedArray(format!(
                "Expected arrays for join, got {:?} and {:?}", 
                left_value, right_value
            ))),
        }
    }

    fn evaluate_aggregate(&mut self, agg: &AggregateExpr) -> Result<Value, RuntimeError> {
        // Evaluate the input expression
        let input_value = self.evaluate(&agg.input)?;
        
        // Input should be an array
        if let Value::Array(elements) = input_value {
            // If there are no elements, return null
            if elements.is_empty() {
                return Ok(Value::Null);
            }
            
            // Create a new scope for the aggregator
            let saved_env = self.env.clone();
            
            // Apply the aggregator to the elements
            match &*agg.aggregator {
                Expr::Lambda(lambda) => {
                    if lambda.parameters.len() != 1 {
                        return Err(RuntimeError::WrongNumberOfArguments(
                            lambda.parameters.len(),
                            1,
                        ));
                    }
                    
                    let param_name = &lambda.parameters[0];
                    self.env.set_variable(param_name.clone(), Value::Array(elements));
                    
                    // Evaluate the aggregator
                    let result = self.evaluate(&lambda.body)?;
                    
                    // Restore the environment
                    self.env = saved_env;
                    
                    Ok(result)
                },
                _ => {
                    // For non-lambda aggregators
                    Err(RuntimeError::ExpectedLambda)
                }
            }
        } else {
            Err(RuntimeError::ExpectedArray(format!("{:?}", input_value)))
        }
    }

    fn evaluate_method_call(&mut self, method: &MethodCallExpr) -> Result<Value, RuntimeError> {
        // Evaluate the object
        let object = self.evaluate(&method.object)?;
        
        // Evaluate the arguments
        let mut arg_values = Vec::new();
        for arg in &method.arguments {
            arg_values.push(self.evaluate(arg)?);
        }
        
        match object {
            Value::Record(fields) => {
                // Check if the method exists in the record
                match fields.get(&method.method) {
                    Some(Value::Function(func)) => {
                        // Call the method with the arguments
                        let mut args = vec![Value::Record(fields.clone())];
                        args.extend(arg_values);
                        
                        // Call the function using our custom call method
                        (**func).call(args)
                    },
                    Some(_) => Err(RuntimeError::NotAFunction(method.method.clone())),
                    None => Err(RuntimeError::UndefinedField(method.method.clone())),
                }
            },
            Value::Array(elements) => {
                // Built-in methods for arrays
                match method.method.as_str() {
                    "length" => {
                        if !arg_values.is_empty() {
                            return Err(RuntimeError::WrongNumberOfArguments(
                                arg_values.len(),
                                0,
                            ));
                        }
                        Ok(Value::Int(elements.len() as i64))
                    },
                    "get" => {
                        if arg_values.len() != 1 {
                            return Err(RuntimeError::WrongNumberOfArguments(
                                arg_values.len(),
                                1,
                            ));
                        }
                        
                        match &arg_values[0] {
                            Value::Int(index) => {
                                if *index < 0 || *index >= elements.len() as i64 {
                                    return Err(RuntimeError::Other(format!("Index out of bounds: {}", index)));
                                }
                                Ok(elements[*index as usize].clone())
                            },
                            _ => Err(RuntimeError::TypeMismatch(
                                format!("{:?}", arg_values[0]),
                                "Int".to_string(),
                                "array index".to_string(),
                            )),
                        }
                    },
                    _ => Err(RuntimeError::UndefinedField(method.method.clone())),
                    // TODO: Add more methods
                }
            },
            Value::String(s) => {
                // Built-in methods for strings
                match method.method.as_str() {
                    "length" => {
                        if !arg_values.is_empty() {
                            return Err(RuntimeError::WrongNumberOfArguments(
                                arg_values.len(),
                                0,
                            ));
                        }
                        Ok(Value::Int(s.len() as i64))
                    },
                    "substring" => {
                        if arg_values.len() != 2 {
                            return Err(RuntimeError::WrongNumberOfArguments(
                                arg_values.len(),
                                2,
                            ));
                        }
                        
                        match (&arg_values[0], &arg_values[1]) {
                            (Value::Int(start), Value::Int(end)) => {
                                if *start < 0 || *end < 0 || *start > *end || *end > s.len() as i64 {
                                    return Err(RuntimeError::Other(format!("Invalid substring range: {} to {}", start, end)));
                                }
                                Ok(Value::String(s[*start as usize..*end as usize].to_string()))
                            },
                            _ => Err(RuntimeError::TypeMismatch(
                                format!("{:?}, {:?}", arg_values[0], arg_values[1]),
                                "Int, Int".to_string(),
                                "substring indices".to_string(),
                            )),
                        }
                    },
                    _ => Err(RuntimeError::UndefinedField(method.method.clone())),
                }
            },
            _ => Err(RuntimeError::NotARecord(format!("{:?}", object))),
        }
    }
}