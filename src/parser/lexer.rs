use std::fmt;
use nom::sequence::tuple;
use nom::combinator::opt;

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Keywords

    DataSource,
    Filter,
    Map,
    GroupBy,
    Join,
    Aggregate,

    // Literals
    IntLiteral(i64),
    FloatLiteral(f64),
    StringLiteral(String),
    BooleanLiteral(bool),
      
    // Identifiers
    Identifier(String),
      
    // Operators
    Plus,
    Minus,
    Multiply,
    Divide,
    Equal,
    NotEqual,
    Greater,
    Less,
    GreaterEqual,
    LessEqual,
    And,
    Or,
    Not,
      
    // Delimiters
    LeftParen,
    RightParen,
    LeftBrace,
    RightBrace,
    LeftBracket,
    RightBracket,
    Comma,
    Colon,
    Semicolon,
    Dot,
    Pipe,
      
    // Special
    Lambda,
    Arrow,
    Assign,
    EOF,
}

  impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Token::DataSource => write!(f, "data_source"),
            Token::Filter => write!(f, "filter"),
            Token::Map => write!(f, "map"),
            Token::GroupBy => write!(f, "group_by"),
            Token::Join => write!(f, "join"),
            Token::Aggregate => write!(f, "aggregate"),
            Token::IntLiteral(i) => write!(f, "{}", i),
            Token::FloatLiteral(fl) => write!(f, "{}", fl),
            Token::StringLiteral(s) => write!(f, "\"{}\"", s),
            Token::BooleanLiteral(b) => write!(f, "{}", b),
            Token::Identifier(id) => write!(f, "{}", id),
            Token::Plus => write!(f, "+"),
            Token::Minus => write!(f, "-"),
            Token::Multiply => write!(f, "*"),
            Token::Divide => write!(f, "/"),
            Token::Equal => write!(f, "=="),
            Token::NotEqual => write!(f, "!="),
            Token::Greater => write!(f, ">"),
            Token::Less => write!(f, "<"),
            Token::GreaterEqual => write!(f, ">="),
            Token::LessEqual => write!(f, "<="),
            Token::And => write!(f, "&&"),
            Token::Or => write!(f, "||"),
            Token::Not => write!(f, "!"),
            Token::LeftParen => write!(f, "("),
            Token::RightParen => write!(f, ")"),
            Token::LeftBrace => write!(f, "{{"),
            Token::RightBrace => write!(f, "}}"),
            Token::LeftBracket => write!(f, "["),
            Token::RightBracket => write!(f, "]"),
            Token::Comma => write!(f, ","),
            Token::Colon => write!(f, ":"),
            Token::Semicolon => write!(f, ";"),
            Token::Dot => write!(f, "."),
            Token::Pipe => write!(f, "|"),
            Token::Lambda => write!(f, "|"),
            Token::Arrow => write!(f, "=>"),
            Token::Assign => write!(f, "="),
            Token::EOF => write!(f, "EOF"),
        }
    }
}

#[derive(Debug)]
pub enum LexError {
    InvalidToken(String),
}

pub fn lex(input: &str) -> Result<Vec<Token>, LexError> {
    use nom::{
        branch::alt,
        bytes::complete::{tag, take_while, take_while1},
        character::complete::{char, digit1, multispace0, multispace1, one_of},
        combinator::{map, map_res, recognize, value},
        multi::many0,
        sequence::{delimited, pair, preceded, terminated},
        IResult,
    };

    // Helper functions for lexing
    fn parse_keyword(input: &str) -> IResult<&str, Token> {
        alt((
            value(Token::DataSource, tag("data_source")),
            value(Token::Filter, tag("filter")),
            value(Token::Map, tag("map")),
            value(Token::GroupBy, tag("group_by")),
            value(Token::Join, tag("join")),
            value(Token::Aggregate, tag("aggregate")),
        ))(input)
    }

    fn parse_identifier(input: &str) -> IResult<&str, Token> {
        let identifier_chars = |c: char| c.is_alphanumeric() || c == '_';
        let first_char = |c: char| c.is_alphabetic() || c == '_';
        
        map(
            recognize(pair(
                take_while1(first_char),
                take_while(identifier_chars),
            )),
            |s: &str| Token::Identifier(s.to_string()),
        )(input)
    }

    fn parse_int_literal(input: &str) -> IResult<&str, Token> {
        map_res(
            recognize(pair(
                opt(char('-')),
                digit1,
            )),
            |s: &str| s.parse::<i64>().map(Token::IntLiteral),
        )(input)
    }

    fn parse_float_literal(input: &str) -> IResult<&str, Token> {
        map_res(
            recognize(tuple((
                opt(char('-')),
                digit1,
                char('.'),
                digit1,
            ))),
            |s: &str| s.parse::<f64>().map(Token::FloatLiteral),
        )(input)
    }

    fn parse_string_literal(input: &str) -> IResult<&str, Token> {
        map(
            delimited(
                char('"'),
                take_while(|c| c != '"'),
                char('"'),
            ),
            |s: &str| Token::StringLiteral(s.to_string()),
        )(input)
    }

    fn parse_boolean_literal(input: &str) -> IResult<&str, Token> {
        alt((
            value(Token::BooleanLiteral(true), tag("true")),
            value(Token::BooleanLiteral(false), tag("false")),
        ))(input)
    }

    fn parse_operator(input: &str) -> IResult<&str, Token> {
        alt((
            value(Token::Plus, tag("+")),
            value(Token::Minus, tag("-")),
            value(Token::Multiply, tag("*")),
            value(Token::Divide, tag("/")),
            value(Token::Equal, tag("==")),
            value(Token::NotEqual, tag("!=")),
            value(Token::GreaterEqual, tag(">=")),
            value(Token::LessEqual, tag("<=")),
            value(Token::Greater, tag(">")),
            value(Token::Less, tag("<")),
            value(Token::And, tag("&&")),
            value(Token::Or, tag("||")),
            value(Token::Not, tag("!")),
            value(Token::Arrow, tag("=>")),
            value(Token::Assign, tag("=")),
        ))(input)
    }

    fn parse_delimiter(input: &str) -> IResult<&str, Token> {
        alt((
            value(Token::LeftParen, tag("(")),
            value(Token::RightParen, tag(")")),
            value(Token::LeftBrace, tag("{")),
            value(Token::RightBrace, tag("}")),
            value(Token::LeftBracket, tag("[")),
            value(Token::RightBracket, tag("]")),
            value(Token::Comma, tag(",")),
            value(Token::Colon, tag(":")),
            value(Token::Semicolon, tag(";")),
            value(Token::Dot, tag(".")),
            value(Token::Pipe, tag("|")),
        ))(input)
    }

    fn parse_token(input: &str) -> IResult<&str, Token> {
        preceded(
            multispace0,
            alt((
                parse_keyword,
                parse_boolean_literal,
                parse_float_literal,
                parse_int_literal,
                parse_string_literal,
                parse_operator,
                parse_delimiter,
                parse_identifier,
            )),
        )(input)
    }

    fn parse_tokens(input: &str) -> IResult<&str, Vec<Token>> {
        many0(parse_token)(input)
    }

    // Main lexing function
    let (remaining, tokens) = parse_tokens(input)
        .map_err(|_| LexError::InvalidToken(input.to_string()))?;

    if !remaining.trim().is_empty() {
        return Err(LexError::InvalidToken(remaining.to_string()));
    }

    Ok(tokens)
}