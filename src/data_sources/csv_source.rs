use std::collections::HashMap;
use std::fs::File;
use csv::ReaderBuilder;
use crate::ast::expressions::Value;
use super::{DataSource, DataSourceError};

pub struct CsvDataSource;

impl DataSource for CsvDataSource {
    fn load(&self, path: &str) -> Result<Vec<Value>, DataSourceError> {
        let file = File::open(path).map_err(|e| 
            DataSourceError::FileNotFound(format!("Could not open {}: {}", path, e))
        )?;
        
        let mut reader = ReaderBuilder::new()
            .has_headers(true)
            .from_reader(file);
            
        let headers = reader.headers()
            .map_err(|e| DataSourceError::ParseError(e.to_string()))?
            .clone();
            
        let mut records = Vec::new();
        
        for result in reader.records() {
            let record = result.map_err(|e| 
                DataSourceError::ParseError(format!("Error reading CSV record: {}", e))
            )?;
            
            let mut field_map = HashMap::new();
            
            for (i, field) in record.iter().enumerate() {
                if i < headers.len() {
                    let header = headers[i].to_string();
                    
                    // Try to parse as different types
                    if let Ok(int_val) = field.parse::<i64>() {
                        field_map.insert(header, Value::Int(int_val));
                    } else if let Ok(float_val) = field.parse::<f64>() {
                        field_map.insert(header, Value::Float(float_val));
                    } else if field.eq_ignore_ascii_case("true") {
                        field_map.insert(header, Value::Boolean(true));
                    } else if field.eq_ignore_ascii_case("false") {
                        field_map.insert(header, Value::Boolean(false));
                    } else {
                        field_map.insert(header, Value::String(field.to_string()));
                    }
                }
            }
            
            records.push(Value::Record(field_map));
        }
        
        Ok(records)
    }
    
    fn get_schema(&self, path: &str) -> Result<HashMap<String, String>, DataSourceError> {
        let file = File::open(path).map_err(|e| 
            DataSourceError::FileNotFound(format!("Could not open {}: {}", path, e))
        )?;
        
        // Create a new reader just for headers
        let mut header_reader = ReaderBuilder::new()
            .has_headers(true)
            .from_reader(file);
            
        let headers = header_reader.headers()
            .map_err(|e| DataSourceError::ParseError(e.to_string()))?;
    
        // Create a new reader for records
        let file = File::open(path).map_err(|e| 
            DataSourceError::FileNotFound(format!("Could not open {}: {}", path, e))
        )?;
        
        let mut reader = ReaderBuilder::new()
            .has_headers(true)
            .from_reader(file);
    
        // Sample the first few records to infer types
        let mut schema = HashMap::new();
        
        // Process records directly in a single pass
        for result in reader.records().take(10) {
            let record = result.map_err(|e| 
                DataSourceError::ParseError(format!("Error reading CSV record: {}", e))
            )?;
            
            for (i, field) in record.iter().enumerate() {
                if i < headers.len() {
                    let header = headers[i].to_string();
                    
                    // Convert the field to a Value for type inference
                    let value = if let Ok(int_val) = field.parse::<i64>() {
                        Value::Int(int_val)
                    } else if let Ok(float_val) = field.parse::<f64>() {
                        Value::Float(float_val)
                    } else if field.eq_ignore_ascii_case("true") {
                        Value::Boolean(true)
                    } else if field.eq_ignore_ascii_case("false") {
                        Value::Boolean(false)
                    } else {
                        Value::String(field.to_string())
                    };
                    
                    // Map the Value type to a type string
                    let field_type = match value {
                        Value::Int(_) => "Int",
                        Value::Float(_) => "Float",
                        Value::Boolean(_) => "Boolean",
                        Value::String(_) => "String",
                        Value::Array(_) => "Array",
                        Value::Record(_) => "Record",
                        Value::Function(_) => "Function",
                        Value::Null => "Null",
                    };
                    
                    schema.insert(header, field_type.to_string());
                }
            }
        }
        
        Ok(schema)
    }
}