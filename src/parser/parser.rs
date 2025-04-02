use std::collections::HashMap;
use crate::ast::*;
use crate::type_system::types::Type;
use super::lexer::Token;
use crate::ast::operations::*;

#[derive(Debug)]
pub enum ParseError {
    ExpectedToken(String, Token),
    UnexpectedEOF,
    InvalidExpression,
    Other(String),
}

pub struct Parser {
    tokens: Vec<Token>,
    position: usize,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Parser {
            tokens,
            position: 0,
        }
    }

    pub fn parse(&mut self) -> Result<Expr, ParseError> {
        self.parse_expr()
    }

    fn current(&self) -> Option<&Token> {
        self.tokens.get(self.position)
    }

    fn advance(&mut self) -> Option<&Token> {
        if self.position < self.tokens.len() {
            self.position += 1;
        }
        self.tokens.get(self.position - 1)
    }

    fn expect(&mut self, expected: Token) -> Result<&Token, ParseError> {
        let current = self.current().cloned();
        match current {
            Some(token) if token == expected => {
                self.advance();
                Ok(self.tokens.get(self.position - 1).unwrap())
            }
            Some(token) => Err(ParseError::ExpectedToken(
                format!("{}", expected),
                token,
            )),
            None => Err(ParseError::UnexpectedEOF),
        }
    }

    fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        self.parse_binary_expr(0) // Start with lowest precedence
    }

    fn parse_binary_expr(&mut self, min_precedence: usize) -> Result<Expr, ParseError> {
        // First parse a primary expression
        let mut left = self.parse_primary_expr()?;
        
        // Then handle method chains
        while let Some(Token::Dot) = self.current() {
            self.advance(); // Consume the dot
            
            // Get method name
            let method_name = match self.current() {
                Some(Token::Identifier(name)) => {
                    let name = name.clone();
                    self.advance();
                    name
                },
                Some(Token::Filter) => {
                    self.advance();
                    "filter".to_string()
                },
                Some(Token::Map) => {
                    self.advance();
                    "map".to_string()
                },
                Some(Token::GroupBy) => {
                    self.advance();
                    "group_by".to_string()
                },
                Some(token) => return Err(ParseError::ExpectedToken(
                    "method name".to_string(),
                    token.clone(),
                )),
                None => return Err(ParseError::UnexpectedEOF),
            };
            
            // Check if this is a field access (no parentheses)
            if self.current() != Some(&Token::LeftParen) {
                left = Expr::FieldAccess(Box::new(FieldAccessExpr {
                    object: Box::new(left),
                    field: method_name,
                    inferred_type: None,
                }));
                continue;
            }
            
            // Otherwise, it's a method call...
            // Parse method arguments
            self.expect(Token::LeftParen)?;
            
            // Special handling for filter and map which expect lambda expressions
            left = match method_name.as_str() {
                "filter" => {
                    // Check if we have a lambda expression
                    if let Some(Token::Pipe) = self.current() {
                        let lambda = self.parse_lambda()?;
                        self.expect(Token::RightParen)?;
                        
                        Expr::Filter(Box::new(FilterExpr {
                            input: Box::new(left),
                            predicate: Box::new(lambda),
                            inferred_type: None,
                        }))
                    } else {
                        let predicate = Box::new(self.parse_expr()?);
                        self.expect(Token::RightParen)?;
                        
                        Expr::Filter(Box::new(FilterExpr {
                            input: Box::new(left),
                            predicate,
                            inferred_type: None,
                        }))
                    }
                },
                "map" => {
                    // Check if we have a lambda expression
                    if let Some(Token::Pipe) = self.current() {
                        let lambda = self.parse_lambda()?;
                        self.expect(Token::RightParen)?;
                        
                        Expr::Map(Box::new(MapExpr {
                            input: Box::new(left),
                            transform: Box::new(lambda),
                            inferred_type: None,
                        }))
                    } else {
                        let transform = Box::new(self.parse_expr()?);
                        self.expect(Token::RightParen)?;
                        
                        Expr::Map(Box::new(MapExpr {
                            input: Box::new(left),
                            transform,
                            inferred_type: None,
                        }))
                    }
                },
                "group_by" => {
                    let key_selector = Box::new(self.parse_expr()?);
                    self.expect(Token::RightParen)?;
                    
                    Expr::GroupBy(Box::new(GroupByExpr {
                        input: Box::new(left),
                        key_selector,
                        inferred_type: None,
                    }))
                },
                _ => return Err(ParseError::Other(format!("Unknown method: {}", method_name))),
            };
        }
        
        // Then handle binary operations with precedence
        while let Some(op) = self.current() {
            let (precedence, is_right_assoc) = match op {
                Token::And | Token::Or => (1, false),
                Token::Equal | Token::NotEqual => (2, false),
                Token::Less | Token::LessEqual | Token::Greater | Token::GreaterEqual => (3, false),
                Token::Plus | Token::Minus => (4, false),
                Token::Multiply | Token::Divide => (5, false),
                _ => break,
            };
            
            if precedence < min_precedence {
                break;
            }
            
            let operator = self.advance().unwrap().clone();
            
            // For right-associative operators, we use precedence+1
            let next_min_precedence = if is_right_assoc { precedence } else { precedence + 1 };
            
            let right = self.parse_binary_expr(next_min_precedence)?;
            
            left = Expr::BinaryOp(Box::new(BinaryOpExpr {
                left: Box::new(left),
                operator,
                right: Box::new(right),
                inferred_type: None,
            }));
        }
        
        Ok(left)
    }

    fn parse_primary_expr(&mut self) -> Result<Expr, ParseError> {
        match self.current() {
            Some(Token::DataSource) => self.parse_data_source(),
            Some(Token::Filter) => self.parse_filter(),
            Some(Token::Map) => self.parse_map(),
            Some(Token::GroupBy) => self.parse_group_by(),
            Some(Token::Join) => self.parse_join(),
            Some(Token::Aggregate) => self.parse_aggregate(),
            Some(Token::IntLiteral(_)) |
            Some(Token::FloatLiteral(_)) |
            Some(Token::StringLiteral(_)) |
            Some(Token::BooleanLiteral(_)) => self.parse_literal(),
            Some(Token::Identifier(_)) => self.parse_variable(),
            Some(Token::LeftParen) => self.parse_function_call(),
            Some(Token::Pipe) => self.parse_lambda(),
            Some(Token::LeftBrace) => self.parse_record_literal(),
            _ => Err(ParseError::InvalidExpression),
        }
    }

    fn parse_data_source(&mut self) -> Result<Expr, ParseError> {
        self.advance(); // Consume 'data_source'
        
        self.expect(Token::LeftParen)?; // Expect opening parenthesis
        
        // Parse source name
        let source_name = match self.advance() {
            Some(Token::StringLiteral(s)) => s.clone(),
            Some(token) => return Err(ParseError::ExpectedToken(
                "string literal".to_string(),
                token.clone(),
            )),
            None => return Err(ParseError::UnexpectedEOF),
        };
        
        self.expect(Token::Comma)?;
        let schema = self.parse_schema()?;
        
        self.expect(Token::RightParen)?; // Expect closing parenthesis
        
        Ok(Expr::DataSource(DataSourceExpr {
            source: source_name,
            schema,
            inferred_type: None,
            format: None,
        }))
    }

    fn parse_schema(&mut self) -> Result<HashMap<String, Type>, ParseError> {
        self.expect(Token::LeftBrace)?;
        
        let mut schema = HashMap::new();
        
        // Parse field: type pairs
        while let Some(token) = self.current() {
            if token == &Token::RightBrace {
                self.advance();
                break;
            }
            
            // Parse field name
            let field_name = match self.advance() {
                Some(Token::Identifier(name)) => name.clone(),
                Some(token) => return Err(ParseError::ExpectedToken(
                    "identifier".to_string(),
                    token.clone(),
                )),
                None => return Err(ParseError::UnexpectedEOF),
            };
            
            self.expect(Token::Colon)?;
            
            // Parse type
            let field_type = self.parse_type()?;
            
            schema.insert(field_name, field_type);
            
            // Check for comma or end of schema
            match self.current() {
                Some(Token::Comma) => { self.advance(); }
                Some(Token::RightBrace) => {}
                Some(token) => return Err(ParseError::ExpectedToken(
                    "comma or }".to_string(),
                    token.clone(),
                )),
                None => return Err(ParseError::UnexpectedEOF),
            }
        }
        
        Ok(schema)
    }

    fn parse_type(&mut self) -> Result<Type, ParseError> {
        match self.advance() {
            Some(Token::Identifier(type_name)) => {
                match type_name.as_str() {
                    "Int" => Ok(Type::Int),
                    "Float" => Ok(Type::Float),
                    "String" => Ok(Type::String),
                    "Boolean" => Ok(Type::Boolean),
                    "Array" => {
                        self.expect(Token::Less)?;
                        let elem_type = self.parse_type()?;
                        self.expect(Token::Greater)?;
                        Ok(Type::Array(Box::new(elem_type)))
                    },
                    _ => Err(ParseError::Other(format!("Unknown type: {}", type_name))),
                }
            },
            Some(token) => Err(ParseError::ExpectedToken(
                "type name".to_string(),
                token.clone(),
            )),
            None => Err(ParseError::UnexpectedEOF),
        }
    }

    fn parse_filter(&mut self) -> Result<Expr, ParseError> {
        self.advance(); // Consume 'filter'
        
        self.expect(Token::LeftParen)?;
        let input = Box::new(self.parse_expr()?);
        self.expect(Token::Comma)?;
        let predicate = Box::new(self.parse_expr()?);
        self.expect(Token::RightParen)?;
        
        Ok(Expr::Filter(Box::new(FilterExpr {
            input,
            predicate,
            inferred_type: None,
        })))
    }

    fn parse_map(&mut self) -> Result<Expr, ParseError> {
        self.advance(); // Consume 'map'
        
        self.expect(Token::LeftParen)?;
        let input = Box::new(self.parse_expr()?);
        self.expect(Token::Comma)?;
        let transform = Box::new(self.parse_expr()?);
        self.expect(Token::RightParen)?;
        
        Ok(Expr::Map(Box::new(MapExpr {
            input,
            transform,
            inferred_type: None,
        })))
    }

    fn parse_group_by(&mut self) -> Result<Expr, ParseError> {
        self.advance(); // Consume 'group_by'
        
        self.expect(Token::LeftParen)?;
        let input = Box::new(self.parse_expr()?);
        self.expect(Token::Comma)?;
        let key_selector = Box::new(self.parse_expr()?);
        self.expect(Token::RightParen)?;
        
        Ok(Expr::GroupBy(Box::new(GroupByExpr {
            input,
            key_selector,
            inferred_type: None,
        })))
    }

    fn parse_join(&mut self) -> Result<Expr, ParseError> {
        self.advance(); // Consume 'join'
        
        self.expect(Token::LeftParen)?;
        let left = Box::new(self.parse_expr()?);
        self.expect(Token::Comma)?;
        let right = Box::new(self.parse_expr()?);
        self.expect(Token::Comma)?;
        let left_key = Box::new(self.parse_expr()?);
        self.expect(Token::Comma)?;
        let right_key = Box::new(self.parse_expr()?);
        self.expect(Token::Comma)?;
        let result_selector = Box::new(self.parse_expr()?);
        self.expect(Token::RightParen)?;
        
        Ok(Expr::Join(Box::new(JoinExpr {
            left,
            right,
            left_key,
            right_key,
            result_selector,
            inferred_type: None,
        })))
    }

    fn parse_aggregate(&mut self) -> Result<Expr, ParseError> {
        self.advance(); // Consume 'aggregate'
        
        self.expect(Token::LeftParen)?;
        let input = Box::new(self.parse_expr()?);
        self.expect(Token::Comma)?;
        let aggregator = Box::new(self.parse_expr()?);
        self.expect(Token::RightParen)?;
        
        Ok(Expr::Aggregate(Box::new(AggregateExpr {
            input,
            aggregator,
            inferred_type: None,
        })))
    }

    fn parse_literal(&mut self) -> Result<Expr, ParseError> {
        match self.advance() {
            Some(Token::IntLiteral(i)) => Ok(Expr::Literal(LiteralExpr {
                value: Value::Int(*i),
                inferred_type: None,
            })),
            Some(Token::FloatLiteral(f)) => Ok(Expr::Literal(LiteralExpr {
                value: Value::Float(*f),
                inferred_type: None,
            })),
            Some(Token::StringLiteral(s)) => Ok(Expr::Literal(LiteralExpr {
                value: Value::String(s.clone()),
                inferred_type: None,
            })),
            Some(Token::BooleanLiteral(b)) => Ok(Expr::Literal(LiteralExpr {
                value: Value::Boolean(*b),
                inferred_type: None,
            })),
            Some(token) => Err(ParseError::ExpectedToken(
                "literal".to_string(),
                token.clone(),
            )),
            None => Err(ParseError::UnexpectedEOF),
        }
    }

    fn parse_variable(&mut self) -> Result<Expr, ParseError> {
        match self.advance() {
            Some(Token::Identifier(name)) => Ok(Expr::Variable(VariableExpr {
                name: name.clone(),
                inferred_type: None,
            })),
            Some(token) => Err(ParseError::ExpectedToken(
                "identifier".to_string(),
                token.clone(),
            )),
            None => Err(ParseError::UnexpectedEOF),
        }
    }

    fn parse_function_call(&mut self) -> Result<Expr, ParseError> {
        self.advance(); // Consume '('
        
        let function = Box::new(self.parse_expr()?);
        
        let mut arguments = Vec::new();
        
        while let Some(token) = self.current() {
            if token == &Token::RightParen {
                self.advance();
                break;
            }
            
            if !arguments.is_empty() {
                self.expect(Token::Comma)?;
            }
            
            arguments.push(self.parse_expr()?);
        }
        
        Ok(Expr::FunctionCall(Box::new(FunctionCallExpr {
            function,
            arguments,
            inferred_type: None,
        })))
    }

    fn parse_lambda(&mut self) -> Result<Expr, ParseError> {
        self.advance(); // Consume the opening pipe '|'
        
        // Parse parameter names
        let mut parameters = Vec::new();
        
        // Check if we have parameters
        if let Some(Token::Identifier(name)) = self.current().cloned() {
            parameters.push(name.clone());
            self.advance(); // Consume the parameter name
            
            // Parse additional parameters if any
            while let Some(Token::Comma) = self.current() {
                self.advance(); // Consume comma
                
                match self.current() {
                    Some(Token::Identifier(name)) => {
                        parameters.push(name.clone());
                        self.advance(); // Consume parameter name
                    },
                    _ => return Err(ParseError::ExpectedToken(
                        "parameter name".to_string(),
                        self.current().cloned().unwrap_or(Token::EOF),
                    )),
                }
            }
        }
        
        // Expect closing pipe
        match self.current() {
            Some(Token::Pipe) => {
                self.advance(); // Consume the closing pipe
            },
            _ => return Err(ParseError::ExpectedToken(
                "closing pipe '|'".to_string(),
                self.current().cloned().unwrap_or(Token::EOF),
            )),
        }
        
        // Parse the lambda body
        let body = self.parse_expr()?;
        
        Ok(Expr::Lambda(Box::new(LambdaExpr {
            parameters,
            body: Box::new(body),
            inferred_type: None,
        })))
    }

    fn parse_record_literal(&mut self) -> Result<Expr, ParseError> {
        self.advance(); // Consume '{'
        
        let mut fields = HashMap::new();
        
        // Parse field: value pairs
        while let Some(token) = self.current() {
            if token == &Token::RightBrace {
                self.advance();
                break;
            }
            
            // Parse field name
            let field_name = match self.advance() {
                Some(Token::Identifier(name)) => name.clone(),
                Some(token) => return Err(ParseError::ExpectedToken(
                    "identifier".to_string(),
                    token.clone(),
                )),
                None => return Err(ParseError::UnexpectedEOF),
            };
            
            self.expect(Token::Colon)?;
            
            // Parse field value
            let field_value = self.parse_expr()?;
            
            fields.insert(field_name, field_value);
            
            // Check for comma or end of record
            match self.current() {
                Some(Token::Comma) => { self.advance(); }
                Some(Token::RightBrace) => {}
                Some(token) => return Err(ParseError::ExpectedToken(
                    "comma or }".to_string(),
                    token.clone(),
                )),
                None => return Err(ParseError::UnexpectedEOF),
            }
        }
        
        Ok(Expr::RecordLiteral(RecordLiteralExpr {
            fields,
            inferred_type: None,
        }))
    }
}

pub fn parse(tokens: Vec<Token>) -> Result<Expr, ParseError> {
    let mut parser = Parser::new(tokens);
    parser.parse()
}
