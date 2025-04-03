mod ast;
mod type_system;
mod parser;
mod interpreter;
mod data_sources; 

use type_system::*;
use parser::*;
use interpreter::*;
use crate::data_sources::DataSource;
use crate::ast::expressions::Value;

fn main() {
    println!("Type-Inferred DSL for Data Processing");
    
    // Example of parsing a simple expression
    let input = r#"
        data_source("users.csv", {
            id: Int,
            name: String,
            age: Int,
            department: String
        })
        .filter(|user| user.age > 30 && user.department == "Engineering")
        .map(|user| {
            name: user.name,
            department: user.department
        })
    "#;
    
    match lex(input) {
        Ok(tokens) => {
            println!("Tokens: {:#?}", tokens);
            
            match parse(tokens) {
                Ok(mut expr) => {
                    println!("Parsed expression: {:#?}", expr);
                    
                    // Type inference
                    let mut type_inference = TypeInference::new();
                    match type_inference.infer(&mut expr) {
                        Ok(ty) => {
                            println!("Inferred type: {}", ty);
                            println!("Parsed expression with types: {:#?}", expr);
                            
                            // Execute the expression
                            let mut interpreter = Interpreter::new();
                            match interpreter.evaluate(&expr) {
                                Ok(result) => {
                                    println!("Execution result:");
                                    println!("{:#?}", result);
                                    // Write the result to a CSV file
                                    let output_path = "output.csv";
                                    // Then use that variable in the function call
                                    match data_sources::csv_source::CsvDataSource.write(output_path, &[result]) {
                                        Ok(_) => println!("Result written to {}", output_path),
                                        Err(e) => println!("Error writing result: {:#?}", e),
                                    }
                                },
                                Err(err) => println!("Execution error: {:#?}", err),
                            }
                        },
                        Err(err) => println!("Type error: {:#?}", err),
                    }
                },
                Err(err) => println!("Parse error: {:#?}", err),
            }
        },
        Err(err) => println!("Lex error: {:#?}", err),
    }
}