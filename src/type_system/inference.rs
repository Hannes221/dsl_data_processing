use std::collections::HashMap;
use crate::ast::*;
use crate::ast::operations::*;
use super::types::{Type, TypeEnvironment};      
use crate::parser::lexer::Token;
use crate::data_sources::DataSourceFactory;
/// Error type for type inference
#[derive(Debug)]
pub enum TypeError {
    UnificationError(Type, Type),
    UndefinedVariable(String),
    UndefinedField(String),
    NotARecord(Type),
    NotAFunction(Type),
    WrongNumberOfArguments(usize, usize),
    Other(String),
}

/// Type inference engine
pub struct TypeInference {
    /// Type environment
    pub env: TypeEnvironment,
    
    /// Type substitutions found during unification
    pub substitutions: HashMap<usize, Type>,
}

impl TypeInference {
    /// Create a new type inference engine
    pub fn new() -> Self {
        TypeInference {
            env: TypeEnvironment::new(),
            substitutions: HashMap::new(),
        }
    }
    
    /// Infer the type of an expression
    pub fn infer(&mut self, expr: &mut Expr) -> Result<Type, TypeError> {
        match expr {
            Expr::Literal(lit) => self.infer_literal(lit),
            Expr::Variable(var) => self.infer_variable(var),
            Expr::FunctionCall(call) => self.infer_function_call(call),
            Expr::Lambda(lambda) => self.infer_lambda(lambda),
            Expr::DataSource(ds) => self.infer_data_source(ds),
            Expr::Filter(filter) => self.infer_filter(filter),
            Expr::Map(map) => self.infer_map(map),
            Expr::GroupBy(group_by) => self.infer_group_by(group_by),
            Expr::Join(join) => self.infer_join(join),
            Expr::Aggregate(agg) => self.infer_aggregate(agg),
            Expr::FieldAccess(field) => self.infer_field_access(field),
            Expr::MethodCall(method) => self.infer_method_call(method),
            Expr::BinaryOp(binary_op) => self.infer_binary_op(binary_op),
            Expr::RecordLiteral(record) => self.infer_record_literal(record),
        }
    }
    
    /// Unify two types, updating substitutions
    pub fn unify(&mut self, t1: &Type, t2: &Type) -> Result<(), TypeError> {
        let t1 = self.apply_substitutions(t1.clone());
        let t2 = self.apply_substitutions(t2.clone());
        
        match (t1, t2) {
            // Same types unify
            (t1, t2) if t1 == t2 => Ok(()),
            
            // Type variable unifies with any type (except itself in the type)
            (Type::TypeVar(id), t) => {
                // Occurs check (prevent infinite types)
                if self.occurs_check(id, &t) {
                    return Err(TypeError::UnificationError(Type::TypeVar(id), t));
                }
                self.substitutions.insert(id, t);
                Ok(())
            },
            (t, Type::TypeVar(id)) => {
                // Occurs check (prevent infinite types)
                if self.occurs_check(id, &t) {
                    return Err(TypeError::UnificationError(t, Type::TypeVar(id)));
                }
                self.substitutions.insert(id, t);
                Ok(())
            },
            
            // Arrays unify if their element types unify
            (Type::Array(t1), Type::Array(t2)) => {
                self.unify(&t1, &t2)
            },
            
            // Records unify if they have the same fields and those fields' types unify
            (Type::Record(fields1), Type::Record(fields2)) => {
                // Check that all fields in fields1 are in fields2 with compatible types
                for (name, ty1) in fields1.iter() {
                    match fields2.get(name) {
                        Some(ty2) => self.unify(ty1, ty2)?,
                        None => return Err(TypeError::UnificationError(
                            Type::Record(fields1.clone()), 
                            Type::Record(fields2.clone())
                        )),
                    }
                }
                
                Ok(())
            },
            
            // Functions unify if their parameter types and return types unify
            (Type::Function(params1, ret1), Type::Function(params2, ret2)) => {
                if params1.len() != params2.len() {
                    return Err(TypeError::UnificationError(
                        Type::Function(params1, ret1), 
                        Type::Function(params2, ret2)
                    ));
                }
                
                // Unify parameter types
                for (p1, p2) in params1.iter().zip(params2.iter()) {
                    self.unify(p1, p2)?;
                }
                
                // Unify return types
                self.unify(&ret1, &ret2)
            },
            
            // Generic types unify if they have the same name and their type arguments unify
            (Type::Generic(name1, args1), Type::Generic(name2, args2)) => {
                if name1 != name2 || args1.len() != args2.len() {
                    return Err(TypeError::UnificationError(
                        Type::Generic(name1, args1), 
                        Type::Generic(name2, args2)
                    ));
                }
                
                // Unify type arguments
                for (a1, a2) in args1.iter().zip(args2.iter()) {
                    self.unify(a1, a2)?;
                }
                
                Ok(())
            },
            
            // Other combinations don't unify
            (t1, t2) => Err(TypeError::UnificationError(t1, t2)),
        }
    }
    
    /// Apply current substitutions to a type
    pub fn apply_substitutions(&self, ty: Type) -> Type {
        match ty {
            Type::TypeVar(id) => {
                match self.substitutions.get(&id) {
                    Some(substituted) => self.apply_substitutions(substituted.clone()),
                    None => Type::TypeVar(id),
                }
            },
            Type::Array(elem_ty) => {
                Type::Array(Box::new(self.apply_substitutions(*elem_ty)))
            },
            Type::Record(fields) => {
                let mut new_fields = HashMap::new();
                for (name, field_ty) in fields {
                    new_fields.insert(name, self.apply_substitutions(field_ty));
                }
                Type::Record(new_fields)
            },
            Type::Function(params, ret_ty) => {
                let new_params = params.into_iter()
                    .map(|p| self.apply_substitutions(p))
                    .collect();
                let new_ret = Box::new(self.apply_substitutions(*ret_ty));
                Type::Function(new_params, new_ret)
            },
            Type::Generic(name, args) => {
                let new_args = args.into_iter()
                    .map(|a| self.apply_substitutions(a))
                    .collect();
                Type::Generic(name, new_args)
            },
            // Primitive types remain unchanged
            ty => ty,
        }
    }
    
    /// Check if a type variable occurs in a type (to prevent infinite types)
    fn occurs_check(&self, id: usize, ty: &Type) -> bool {
        match ty {
            Type::TypeVar(other_id) => {
                if *other_id == id {
                    return true;
                }
                // Check if there's a substitution for this variable
                match self.substitutions.get(other_id) {
                    Some(substituted) => self.occurs_check(id, substituted),
                    None => false,
                }
            },
            Type::Array(elem_ty) => self.occurs_check(id, elem_ty),
            Type::Record(fields) => {
                fields.values().any(|field_ty| self.occurs_check(id, field_ty))
            },
            Type::Function(params, ret_ty) => {
                params.iter().any(|p| self.occurs_check(id, p)) || 
                self.occurs_check(id, ret_ty)
            },
            Type::Generic(_, args) => {
                args.iter().any(|a| self.occurs_check(id, a))
            },
            // Primitive types don't contain type variables
            _ => false,
        }
    }
    
    fn infer_literal(&mut self, lit: &mut LiteralExpr) -> Result<Type, TypeError> {
        // Infer type based on the literal value
        let ty = match &lit.value {
            Value::Int(_) => Type::Int,
            Value::Float(_) => Type::Float,
            Value::String(_) => Type::String,
            Value::Boolean(_) => Type::Boolean,
            Value::Array(elements) => {
                if elements.is_empty() {
                    // For empty arrays, we need a type variable
                    let elem_ty = self.env.fresh_type_var();
                    Type::Array(Box::new(elem_ty))
                } else {
                    // Infer element type from the first element
                    // This is simplified; a real implementation would check all elements
                    // TODO: Implement this
                    let mut elem_expr = LiteralExpr {
                        value: elements[0].clone(),
                        inferred_type: None,
                    };
                    let elem_ty = self.infer_literal(&mut elem_expr)?;
                    Type::Array(Box::new(elem_ty))
                }
            },
            Value::Record(fields) => {
                let mut field_types = HashMap::new();
                for (name, value) in fields {
                    let mut field_expr = LiteralExpr {
                        value: value.clone(),
                        inferred_type: None,
                    };
                    let field_ty = self.infer_literal(&mut field_expr)?;
                    field_types.insert(name.clone(), field_ty);
                }
                Type::Record(field_types)
            },
            Value::Function(_) => {
                // For function literals, we need to infer the type from the function body
                // This is a simplified placeholder
                // TODO: Implement this
                let param_ty = self.env.fresh_type_var();
                let return_ty = self.env.fresh_type_var();
                Type::Function(vec![param_ty], Box::new(return_ty))
            },
            Value::Null => {
                // Null can be any type, so we use a type variable
                self.env.fresh_type_var()
            },
        };
        
        lit.inferred_type = Some(ty.clone());
        Ok(ty)
    }
    
    fn infer_variable(&mut self, var: &mut VariableExpr) -> Result<Type, TypeError> {
        // Look up the variable in the type environment
        match self.env.lookup_variable(&var.name) {
            Some(ty) => {
                let ty = ty.clone();
                var.inferred_type = Some(ty.clone());
                Ok(ty)
            },
            None => Err(TypeError::UndefinedVariable(var.name.clone())),
        }
    }

    fn infer_function_call(&mut self, call: &mut FunctionCallExpr) -> Result<Type, TypeError> {
        // Infer type based on the function call
        let func_ty = self.infer(&mut call.function)?;
        
        // Check if the function type is a function
        if let Type::Function(params, ret_ty) = func_ty {
            // Check if the number of arguments matches
            if call.arguments.len() != params.len() {
                return Err(TypeError::WrongNumberOfArguments(
                    call.arguments.len(),
                    params.len()
                ));
            }
            
            // Infer types for each argument
            for (i, param_ty) in params.iter().enumerate() {
                let arg_ty = self.infer(&mut call.arguments[i])?;
                self.unify(&arg_ty, param_ty)?;
            }
            
            let result_type = *ret_ty;
            call.inferred_type = Some(result_type.clone());
            Ok(result_type)
        } else {
            Err(TypeError::NotAFunction(func_ty))
        }
    }

    fn infer_lambda(&mut self, lambda: &mut LambdaExpr) -> Result<Type, TypeError> {
        // Save the current environment
        let saved_env = self.env.clone();
        
        // Create a new scope for parameter types
        let mut param_types = Vec::new();
        
        // Add parameter types to environment
        for param in &lambda.parameters {
            let param_ty = self.env.fresh_type_var();
            self.env.add_variable(param.clone(), param_ty.clone());
            param_types.push(param_ty);
        }
        
        // Infer body type
        let body_ty = self.infer(&mut lambda.body)?;
        
        // Create function type
        let fn_ty = Type::Function(param_types, Box::new(body_ty.clone()));
        lambda.inferred_type = Some(fn_ty.clone());
        
        // Restore the previous environment
        self.env = saved_env;
        
        Ok(fn_ty)
    }

    fn infer_data_source(&mut self, ds: &mut DataSourceExpr) -> Result<Type, TypeError> {
        if ds.schema.is_empty() {
            if let Ok(data_source) = DataSourceFactory::create_data_source(&ds.source) {
                if let Ok(schema_strings) = data_source.get_schema(&ds.source) {
                    // Convert string types to Type enum
                    for (field, type_str) in schema_strings {
                        let field_type = match type_str.as_str() {
                            "Int" => Type::Int,
                            "Float" => Type::Float,
                            "String" => Type::String,
                            "Boolean" => Type::Boolean,
                            "Array" => {
                                // For arrays without element type info, use a type variable
                                let elem_ty = self.env.fresh_type_var();
                                Type::Array(Box::new(elem_ty))
                            },
                            // Handle array types with element info (e.g., "Array<Int>")
                            s if s.starts_with("Array<") && s.ends_with(">") => {
                                let elem_type_str = &s[6..s.len()-1];
                                let elem_type = match elem_type_str {
                                    "Int" => Type::Int,
                                    "Float" => Type::Float,
                                    "String" => Type::String,
                                    "Boolean" => Type::Boolean,
                                    _ => Type::String, // Default for unknown element types
                                };
                                Type::Array(Box::new(elem_type))
                            },
                            "Record" => {
                                // For generic records, use an empty record type
                                Type::Record(HashMap::new())
                            },
                            // For unknown types, default to String
                            _ => Type::String,
                        };
                        ds.schema.insert(field, field_type);
                    }
                }
            }
        }
        
        // Create a record type from the schema
        let ty = Type::Record(ds.schema.clone());
        ds.inferred_type = Some(ty.clone());
        Ok(ty)
    }

    fn infer_filter(&mut self, filter: &mut FilterExpr) -> Result<Type, TypeError> {
        // Infer type based on the filter expression
        let input_ty = self.infer(&mut filter.input)?;
        
        // The predicate should be a function that takes a record and returns a boolean
        match filter.predicate.as_mut() {
            Expr::Lambda(lambda) => {
                // Save the current environment
                let saved_env = self.env.clone();
                
                // Ensure we have exactly one parameter
                if lambda.parameters.len() != 1 {
                    return Err(TypeError::WrongNumberOfArguments(lambda.parameters.len(), 1));
                }
                
                // Add the parameter to the environment with the input type
                let param_name = &lambda.parameters[0];
                self.env.add_variable(param_name.clone(), input_ty.clone());
                
                // Infer the body type
                let body_ty = self.infer(&mut lambda.body)?;
                
                // Ensure the body returns a boolean
                self.unify(&body_ty, &Type::Boolean)?;
                
                // Create the function type
                let fn_ty = Type::Function(vec![input_ty.clone()], Box::new(Type::Boolean));
                lambda.inferred_type = Some(fn_ty);
                
                // Restore the environment
                self.env = saved_env;
            },
            Expr::Variable(var) => {
                // For variables, first infer the variable's type
                let var_ty = self.infer_variable(var)?;
                
                // Then resolve any type variables through substitution
                let resolved_ty = self.apply_substitutions(var_ty.clone());
                
                match resolved_ty {
                    Type::Function(params, ret_ty) => {
                        // Check that the function returns a boolean
                        self.unify(&ret_ty, &Box::new(Type::Boolean))?;
                        
                        // Check that the function takes a single parameter
                        if params.len() != 1 {
                            return Err(TypeError::WrongNumberOfArguments(params.len(), 1));
                        }
                        
                        // Unify the parameter type with the input type
                        self.unify(&params[0], &input_ty)?;
                    },
                    // If it's a type variable, constrain it to be the right kind of function
                    Type::TypeVar(_) => {
                        // Create a function type that takes the input type and returns boolean
                        let fn_ty = Type::Function(vec![input_ty.clone()], Box::new(Type::Boolean));
                        
                        // Unify the variable's type with this function type
                        self.unify(&var_ty, &fn_ty)?;
                    },
                    _ => return Err(TypeError::NotAFunction(var_ty)),
                }
            },
            _ => {
                // For other expressions, infer normally and ensure it's a function
                let predicate_ty = self.infer(&mut filter.predicate)?;
                
                // Resolve any type variables through substitution
                let resolved_ty = self.apply_substitutions(predicate_ty.clone());
                
                match resolved_ty {
                    Type::Function(params, ret_ty) => {
                        // Check that the function returns a boolean
                        self.unify(&ret_ty, &Box::new(Type::Boolean))?;
                        
                        // Check that the function takes a single parameter
                        if params.len() != 1 {
                            return Err(TypeError::WrongNumberOfArguments(params.len(), 1));
                        }
                        
                        // Unify the parameter type with the input type
                        self.unify(&params[0], &input_ty)?;
                    },
                    // If it's a type variable, constrain it to be the right kind of function
                    Type::TypeVar(_) => {
                        // Create a function type that takes the input type and returns boolean
                        let fn_ty = Type::Function(vec![input_ty.clone()], Box::new(Type::Boolean));
                        
                        // Unify the predicate's type with this function type
                        self.unify(&predicate_ty, &fn_ty)?;
                    },
                    _ => return Err(TypeError::NotAFunction(predicate_ty)),
                }
            }
        }
        
        // The filter should return the type of its input
        filter.inferred_type = Some(input_ty.clone());
        Ok(input_ty)
    }

    fn infer_map(&mut self, map: &mut MapExpr) -> Result<Type, TypeError> {
        // Infer type based on the map expression
        let input_ty = self.infer(&mut map.input)?;
        
        // The transform should be a function that takes an element from the input
        let output_ty = match map.transform.as_mut() {
            Expr::Lambda(lambda) => {
                // Save the current environment
                let saved_env = self.env.clone();
                
                if lambda.parameters.len() != 1 {
                    return Err(TypeError::WrongNumberOfArguments(lambda.parameters.len(), 1));
                }
                
                let param_name = &lambda.parameters[0];
                self.env.add_variable(param_name.clone(), input_ty.clone());
                
                let body_ty = self.infer(&mut lambda.body)?;
                
                let fn_ty = Type::Function(vec![input_ty.clone()], Box::new(body_ty.clone()));
                lambda.inferred_type = Some(fn_ty);
                
                self.env = saved_env;
                
                body_ty
            },
            Expr::Variable(var) => {
                // For variables, first infer the variable's type
                let var_ty = self.infer_variable(var)?;
                
                // Then resolve any type variables through substitution
                let resolved_ty = self.apply_substitutions(var_ty.clone());
                
                match resolved_ty {
                    Type::Function(params, ret_ty) => {
                        // Check that the function takes a single parameter
                        if params.len() != 1 {
                            return Err(TypeError::WrongNumberOfArguments(params.len(), 1));
                        }
                        
                        // Unify the parameter type with the input type
                        self.unify(&params[0], &input_ty)?;
                        
                        // The return type is the output element type
                        *ret_ty
                    },
                    // If it's a type variable, constrain it to be the right kind of function
                    Type::TypeVar(_) => {
                        // Create a new type variable for the return type
                        let ret_ty = self.env.fresh_type_var();
                        
                        // Create a function type that takes the input type and returns the new type variable
                        let fn_ty = Type::Function(vec![input_ty.clone()], Box::new(ret_ty.clone()));
                        
                        // Unify the variable's type with this function type
                        self.unify(&var_ty, &fn_ty)?;
                        
                        ret_ty
                    },
                    _ => return Err(TypeError::NotAFunction(var_ty)),
                }
            },
            _ => {
                // For other expressions, infer normally and ensure it's a function
                let transform_ty = self.infer(&mut map.transform)?;
                
                // Resolve any type variables through substitution
                let resolved_ty = self.apply_substitutions(transform_ty.clone());
                
                match resolved_ty {
                    Type::Function(params, ret_ty) => {
                        // Check that the function takes a single parameter
                        if params.len() != 1 {
                            return Err(TypeError::WrongNumberOfArguments(params.len(), 1));
                        }
                        
                        // Unify the parameter type with the input type
                        self.unify(&params[0], &input_ty)?;
                        
                        // The return type is the output element type
                        *ret_ty
                    },
                    // If it's a type variable, constrain it to be the right kind of function
                    Type::TypeVar(_) => {
                        // Create a new type variable for the return type
                        let ret_ty = self.env.fresh_type_var();
                        
                        // Create a function type that takes the input type and returns the new type variable
                        let fn_ty = Type::Function(vec![input_ty.clone()], Box::new(ret_ty.clone()));
                        
                        // Unify the transform's type with this function type
                        self.unify(&transform_ty, &fn_ty)?;
                        
                        ret_ty
                    },
                    _ => return Err(TypeError::NotAFunction(transform_ty)),
                }
            }
        };
        
        // The map should return the type of its output
        map.inferred_type = Some(output_ty.clone());
        Ok(output_ty)
    }

    fn infer_group_by(&mut self, group_by: &mut GroupByExpr) -> Result<Type, TypeError> {
        // Infer type based on the group by expression
        let input_ty = self.infer(&mut group_by.input)?;
        let key_selector_ty = self.infer(&mut group_by.key_selector)?;
        
        // The group by should return a record with the key and the grouped data
        let mut fields = HashMap::new();
        fields.insert("key".to_string(), key_selector_ty.clone());
        fields.insert("data".to_string(), input_ty.clone());
        
        let result_type = Type::Record(fields);
        group_by.inferred_type = Some(result_type.clone());
        Ok(result_type)
    }

    fn infer_join(&mut self, join: &mut JoinExpr) -> Result<Type, TypeError> {
        // Infer type based on the join expression
        let left_ty = self.infer(&mut join.left)?;
        let right_ty = self.infer(&mut join.right)?;
        
        // The join should return a record with fields from both sides
        // This is a simplified implementation - handle field conflicts!!
        // TODO: Implement this
        let mut fields = HashMap::new();
        
        // Add a key field with appropriate type
        let key_ty = self.env.fresh_type_var(); // Or use a more specific type if known
        fields.insert("key".to_string(), key_ty);
        
        // Add fields for left and right sides
        fields.insert("left".to_string(), left_ty);
        fields.insert("right".to_string(), right_ty);
        
        let result_type = Type::Record(fields);
        join.inferred_type = Some(result_type.clone());
        Ok(result_type)
    }

    fn infer_aggregate(&mut self, agg: &mut AggregateExpr) -> Result<Type, TypeError> { 
        // Infer type based on the aggregate expression
        let input_ty = self.infer(&mut agg.input)?;
        let aggregator_ty = self.infer(&mut agg.aggregator)?;
        
        // The aggregate should return the type of its output
        agg.inferred_type = Some(aggregator_ty.clone());
        
        Ok(aggregator_ty)
    }
    
    fn infer_field_access(&mut self, field: &mut FieldAccessExpr) -> Result<Type, TypeError> {
        // Infer type based on the field access expression
        let object_ty = self.infer(&mut field.object)?;
        
        // Apply substitutions to get the current type
        let resolved_ty = self.apply_substitutions(object_ty.clone());
        
        match resolved_ty {
            // If it's already a record type, look up the field
            Type::Record(fields) => {
                match fields.get(&field.field) {
                    Some(field_ty) => {
                        let result_type = field_ty.clone();
                        field.inferred_type = Some(result_type.clone());
                        Ok(result_type)
                    },
                    None => Err(TypeError::UndefinedField(field.field.clone())),
                }
            },
            // If it's a type variable, constrain it to be a record with this field
            Type::TypeVar(id) => {  
                // Create a new type variable for the field's type
                let field_ty = self.env.fresh_type_var();
                
                // Create a record type with just this field
                let mut record_fields = HashMap::new();
                record_fields.insert(field.field.clone(), field_ty.clone());
                let record_ty = Type::Record(record_fields);
                
                // Unify the object type with this record type
                self.unify(&object_ty, &record_ty)?;
                
                field.inferred_type = Some(field_ty.clone());
                Ok(field_ty)
            },
            // Other types can't have fields
            _ => Err(TypeError::NotARecord(object_ty)),
        }
    }

    fn infer_method_call(&mut self, method: &mut MethodCallExpr) -> Result<Type, TypeError> {
        // Infer type based on the method call expression
        let object_ty = self.infer(&mut method.object)?;
        
        // Apply substitutions to get the current type
        let resolved_ty = self.apply_substitutions(object_ty.clone());
        
        match resolved_ty {
            // If it's already a record type, look up the method
            Type::Record(fields) => {
                // Look up the method in the record
                match fields.get(&method.method) {
                    Some(method_ty) => {
                        // Method should be a function type
                        if let Type::Function(_, ret_ty) = method_ty {
                            let result_type = (**ret_ty).clone();
                            method.inferred_type = Some(result_type.clone());
                            Ok(result_type)
                        } else {
                            Err(TypeError::NotAFunction(method_ty.clone()))
                        }
                    },
                    None => Err(TypeError::UndefinedField(method.method.clone())),
                }
            },
            // If it's a type variable, constrain it to be a record with this method
            Type::TypeVar(id) => {
                // Create a new type variable for the method's return type
                let ret_ty = self.env.fresh_type_var();
                
                // Create function type for the method
                let mut param_types = Vec::new();
                for arg in &mut method.arguments {
                    let arg_ty = self.infer(arg)?;
                    param_types.push(arg_ty);
                }
                let method_ty = Type::Function(param_types, Box::new(ret_ty.clone()));
                
                // Create a record type with just this method
                let mut record_fields = HashMap::new();
                record_fields.insert(method.method.clone(), method_ty);
                let record_ty = Type::Record(record_fields);
                
                // Unify the object type with this record type
                self.unify(&object_ty, &record_ty)?;
                
                method.inferred_type = Some(ret_ty.clone());
                Ok(ret_ty)
            },
            // Other types can't have methods
            _ => Err(TypeError::NotARecord(object_ty)),
        }
    }

    fn infer_binary_op(&mut self, binary_op: &mut BinaryOpExpr) -> Result<Type, TypeError> {
        let left_ty = self.infer(&mut binary_op.left)?;
        let right_ty = self.infer(&mut binary_op.right)?;
        
        // For comparison operators, result is Boolean
        match binary_op.operator {
            Token::Greater | Token::Less | Token::GreaterEqual | 
            Token::LessEqual | Token::Equal | Token::NotEqual |
            Token::And | Token::Or => {
                // For And and Or, both operands should be boolean
                if binary_op.operator == Token::And || binary_op.operator == Token::Or {
                    self.unify(&left_ty, &Type::Boolean)?;
                    self.unify(&right_ty, &Type::Boolean)?;
                } else {
                    // For other comparison operators, operands should have the same type
                    self.unify(&left_ty, &right_ty)?;
                }
                
                let result_type = Type::Boolean;
                binary_op.inferred_type = Some(result_type.clone());
                Ok(result_type)
            },
            // For arithmetic operators, result has same type as operands
            Token::Plus | Token::Minus | Token::Multiply | Token::Divide => {
                // Ensure operands have compatible types
                self.unify(&left_ty, &right_ty)?;
                
                let result_type = left_ty.clone();
                binary_op.inferred_type = Some(result_type.clone());
                Ok(result_type)
            },
            _ => Err(TypeError::Other(format!("Unsupported binary operator: {:?}", binary_op.operator))),
        }
    }

    fn infer_record_literal(&mut self, record: &mut RecordLiteralExpr) -> Result<Type, TypeError> {
        // Infer type based on the record literal expression
        let mut fields = HashMap::new();
        
        // Infer types for each field
        for (name, value) in &mut record.fields {
            let value_ty = self.infer(value)?;
            fields.insert(name.clone(), value_ty);
        }
        
        let result_type = Type::Record(fields);
        record.inferred_type = Some(result_type.clone());
        Ok(result_type)
    }
    
}