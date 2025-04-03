use std::collections::HashMap;
use crate::ast::expressions::Value;

pub struct DataSourceFactory;

pub mod csv_source;

#[derive(Debug)]
#[allow(dead_code)]
pub enum DataSourceError {
    FileNotFound(String),
    ParseError(String),
    UnsupportedFormat(String),
    WriteError(String),
}

pub trait DataSource {
    fn load(&self, path: &str) -> Result<Vec<Value>, DataSourceError>;
    fn get_schema(&self, path: &str) -> Result<HashMap<String, String>, DataSourceError>;
    fn write(&self, path: &str, records: &[Value]) -> Result<(), DataSourceError>;
}

impl DataSourceFactory {
    pub fn create_data_source(path: &str) -> Result<Box<dyn DataSource>, DataSourceError> {
        // Check if the file exists
        if let Err(e) = std::fs::metadata(path) {
            return Err(DataSourceError::FileNotFound(
                format!("File not found: {}, error: {}", path, e)
            ));
        }

        // Determine the file format based on extension
        if let Some(extension) = path.split('.').last() {
            match extension.to_lowercase().as_str() {
                "csv" => {
                    // Use the CSV data source for .csv files
                    Ok(Box::new(csv_source::CsvDataSource))
                },
                // Add support for other formats here as needed
                _ => Err(DataSourceError::UnsupportedFormat(
                    format!("Unsupported file format: .{}", extension)
                )),
            }
        } else {
            Err(DataSourceError::UnsupportedFormat(
                "File has no extension".to_string()
            ))
        }
    }
}