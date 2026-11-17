//! kasumi-lang: the formula language.
//!
//! A form is a few lines of math that place point `i` at time `t`:
//!
//! ```text
//! @points 20m
//! param k = 3
//! let a = u * tau
//! x = cos(a) * sin(a * k + t)
//! y = sin(a) * sin(a * k + t)
//! ```
//!
//! Pipeline: [`parse`] (hand-written pratt parser) -> [`ast::Form`] ->
//! [`ir::lower`] (ssa, with constant folding, cse, dead code removal and
//! hoisting of everything that does not depend on the point) ->
//! [`codegen`] (one fused cuda kernel) and [`interp`] (the cpu reference).

pub mod ast;
pub mod codegen;
pub mod error;
pub mod func;
pub mod interp;
pub mod ir;
pub mod lexer;
pub mod parser;
pub mod print;

pub use ast::{BinOp, Expr, Form, Output, Param, SetVal, Stmt, Target};
pub use error::LangError;
pub use func::Func;
pub use ir::{lower, Input, Ir, Op};
pub use parser::parse;
pub use print::{print_expr, print_form};

/// Parse and lower in one go.
pub fn compile(src: &str) -> Result<(Form, Ir), LangError> {
    let form = parse(src)?;
    let ir = lower(&form);
    Ok((form, ir))
}
