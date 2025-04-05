use std::collections::HashMap;
use crate::ast::*;
use crate::ast::operations::*;
use crate::parser::lexer::Token;
use super::environment::Environment;
use super::runtime_error::RuntimeError;
use crate::ast::expressions::Function; 
use crate::data_sources::DataSourceFactory;

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
        // Evaluate the input expression
        let input_value = self.evaluate(&filter.input)?;
        
        // Input should be an array
        if let Value::Array(elements) = input_value {
            let mut result = Vec::new();
            
            // Apply the predicate to each element
            for element in elements {
                // Create a new scope for the lambda parameter
                let saved_env = self.env.clone();
                
                // Bind the current element to the parameter name
                match &*filter.predicate {
                    Expr::Lambda(lambda) => {
                        if lambda.parameters.len() != 1 {
                            return Err(RuntimeError::WrongNumberOfArguments(
                                lambda.parameters.len(),
                                1,
                            ));
                        }
                        
                        let param_name = &lambda.parameters[0];
                        self.env.set_variable(param_name.clone(), element.clone());
                        
                        // Evaluate the predicate
                        let predicate_result = self.evaluate(&lambda.body)?;
                        
                        // Check if the predicate is true
                        if let Value::Boolean(true) = predicate_result {
                            result.push(element);
                        }
                        
                        // Restore the environment
                        self.env = saved_env;
                    },
                    _ => {
                        // For non-lambda predicates, we need to evaluate them in a different way
                        // This is a simplified implementation
                        // TODO: Implement this
                        return Err(RuntimeError::ExpectedLambda);
                    }
                }
            }
            
            Ok(Value::Array(result))
        } else {
            Err(RuntimeError::ExpectedArray(format!("{:?}", input_value)))
        }
    }
    
    fn evaluate_map(&mut self, map: &MapExpr) -> Result<Value, RuntimeError> {
        // Evaluate the input expression
        let input_value = self.evaluate(&map.input)?;
        
        // Input should be an array
        if let Value::Array(elements) = input_value {
            let mut result = Vec::new();
            
            // Apply the transform to each element
            for element in elements {
                // Create a new scope for the lambda parameter
                let saved_env = self.env.clone();
                
                // Bind the current element to the parameter name
                match &*map.transform {
                    Expr::Lambda(lambda) => {
                        if lambda.parameters.len() != 1 {
                            return Err(RuntimeError::WrongNumberOfArguments(
                                lambda.parameters.len(),
                                1,
                            ));
                        }
                        
                        let param_name = &lambda.parameters[0];
                        self.env.set_variable(param_name.clone(), element.clone());
                        
                        // Evaluate the transform
                        let transformed = self.evaluate(&lambda.body)?;
                        result.push(transformed);
                        
                        // Restore the environment
                        self.env = saved_env;
                    },
                    _ => {
                        // For non-lambda transforms, we need to evaluate them in a different way
                        // This is a simplified implementation
                        // TODO: Implement this
                        return Err(RuntimeError::ExpectedLambda);
                    }
                }
            }
            
            Ok(Value::Array(result))
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
            let mut groups: HashMap<String, Vec<Value>> = HashMap::new();
            
            // Group elements by key
            for element in elements {
                // Create a new scope for the lambda parameter
                let saved_env = self.env.clone();
                
                // Bind the current element to the parameter name
                match &*group_by.key_selector {
                    Expr::Lambda(lambda) => {
                        if lambda.parameters.len() != 1 {
                            return Err(RuntimeError::WrongNumberOfArguments(
                                lambda.parameters.len(),
                                1,
                            ));
                        }
                        
                        let param_name = &lambda.parameters[0];
                        self.env.set_variable(param_name.clone(), element.clone());
                        
                        // Evaluate the key selector
                        let key_result = self.evaluate(&lambda.body)?;
                        
                        // Convert the key to a string for grouping
                        let key_string = match &key_result {
                            Value::String(s) => s.clone(),
                            Value::Int(i) => i.to_string(),
                            Value::Float(f) => f.to_string(),
                            Value::Boolean(b) => b.to_string(),
                            _ => format!("{:?}", key_result),
                        };
                        
                        // Add the element to the appropriate group
                        groups.entry(key_string).or_insert_with(Vec::new).push(element);
                        
                        // Restore the environment
                        self.env = saved_env;
                    },
                    _ => {
                        // For non-lambda key selectors
                        return Err(RuntimeError::ExpectedLambda);
                    }
                }
            }
            
            // Convert the groups to an array of records
            let mut result = Vec::new();
            for (key, group) in groups {
                let mut record = HashMap::new();
                record.insert("key".to_string(), Value::String(key));
                record.insert("data".to_string(), Value::Array(group));
                result.push(Value::Record(record));
            }
            
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
                let mut result = Vec::new();
                
                // Extract keys from left elements
                let mut left_keys: HashMap<String, Vec<Value>> = HashMap::new();
                for left_element in left_elements.iter() {
                    // Create a new scope for the lambda parameter
                    let saved_env = self.env.clone();
                    
                    // Bind the current element to the parameter name
                    match &*join.left_key {
                        Expr::Lambda(lambda) => {
                            if lambda.parameters.len() != 1 {
                                return Err(RuntimeError::WrongNumberOfArguments(
                                    lambda.parameters.len(),
                                    1,
                                ));
                            }
                            
                            let param_name = &lambda.parameters[0];
                            self.env.set_variable(param_name.clone(), left_element.clone());
                            
                            // Evaluate the key selector
                            let key_result = self.evaluate(&lambda.body)?;
                            
                            // Convert the key to a string for joining
                            let key_string = match &key_result {
                                Value::String(s) => s.clone(),
                                Value::Int(i) => i.to_string(),
                                Value::Float(f) => f.to_string(),
                                Value::Boolean(b) => b.to_string(),
                                _ => format!("{:?}", key_result),
                            };
                            
                            // Add the element to the appropriate key group
                            left_keys.entry(key_string).or_insert_with(Vec::new).push(left_element.clone());
                            
                            // Restore the environment
                            self.env = saved_env;
                        },
                        _ => {
                            return Err(RuntimeError::ExpectedLambda);
                        }
                    }
                }
                
                // Join with right elements
                for right_element in right_elements.iter() {
                    // Create a new scope for the lambda parameter
                    let saved_env = self.env.clone();
                    
                    // Bind the current element to the parameter name
                    match &*join.right_key {
                        Expr::Lambda(lambda) => {
                            if lambda.parameters.len() != 1 {
                                return Err(RuntimeError::WrongNumberOfArguments(
                                    lambda.parameters.len(),
                                    1,
                                ));
                            }
                            
                            let param_name = &lambda.parameters[0];
                            self.env.set_variable(param_name.clone(), right_element.clone());
                            
                            // Evaluate the key selector
                            let key_result = self.evaluate(&lambda.body)?;
                            
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
                                for left_element in matching_left {
                                    // Create a new scope for the result selector
                                    let saved_env_inner = self.env.clone();
                                    
                                    // Apply the result selector to create the joined record
                                    match &*join.result_selector {
                                        Expr::Lambda(lambda) => {
                                            if lambda.parameters.len() != 2 {
                                                return Err(RuntimeError::WrongNumberOfArguments(
                                                    lambda.parameters.len(),
                                                    2,
                                                ));
                                            }
                                            
                                            // Bind the left and right elements to the parameter names
                                            let left_param = &lambda.parameters[0];
                                            let right_param = &lambda.parameters[1];
                                            self.env.set_variable(left_param.clone(), left_element.clone());
                                            self.env.set_variable(right_param.clone(), right_element.clone());
                                            
                                            // Evaluate the result selector
                                            let joined_result = self.evaluate(&lambda.body)?;
                                            result.push(joined_result);
                                            
                                            // Restore the environment
                                            self.env = saved_env_inner;
                                        },
                                        _ => {
                                            // Default join behavior if no lambda is provided
                                            let mut record = HashMap::new();
                                            record.insert("key".to_string(), Value::String(key_string.clone()));
                                            record.insert("left".to_string(), left_element.clone());
                                            record.insert("right".to_string(), right_element.clone());
                                            result.push(Value::Record(record));
                                        }
                                    }
                                }
                            }
                            
                            // Restore the environment
                            self.env = saved_env;
                        },
                        _ => {
                            return Err(RuntimeError::ExpectedLambda);
                        }
                    }
                }
                
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