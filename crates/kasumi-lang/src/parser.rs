//! Hand-written line-oriented parser with a pratt expression parser.

use std::collections::HashSet;

use crate::ast::*;
use crate::error::{suggest, LangError};
use crate::func::Func;
use crate::lexer::{lex_line, strip_comment, Tok, Token};

pub fn parse(src: &str) -> Result<Form, LangError> {
    let mut form = Form::default();
    // Names a statement may reference: inputs, consts, params, earlier bindings.
    let mut scope: HashSet<String> = INPUTS.iter().map(|s| s.to_string()).collect();
    for (c, _) in CONSTS {
        scope.insert(c.to_string());
    }
    let mut seen_code = false;

    for (idx, raw) in src.lines().enumerate() {
        let lineno = idx as u32 + 1;
        let trimmed = raw.trim();
        if !seen_code {
            if let Some(rest) = trimmed.strip_prefix('#') {
                form.header.push(rest.trim().to_string());
                continue;
            }
        }
        let line = strip_comment(raw);
        if line.trim().is_empty() {
            continue;
        }
        seen_code = true;
        let lead = (line.len() - line.trim_start().len()) as u32;

        if let Some(rest) = line.trim_start().strip_prefix('@') {
            parse_setting(rest, lineno, lead + 2, &mut form)?;
            continue;
        }

        let toks = lex_line(line, lineno)?;
        let mut p = P { toks: &toks, pos: 0, line: lineno, scope: &scope };
        let first = p.peek().clone();
        match &first.tok {
            Tok::Ident(kw) if kw == "param" => {
                p.next();
                let (name, ncol) = p.ident("parameter name")?;
                check_new_name(&name, lineno, ncol, &scope)?;
                p.expect(Tok::Eq, "'=' after the parameter name")?;
                // Params are plain numbers so the explorer can nudge them.
                let empty = HashSet::new();
                let mut q = P { toks: p.toks, pos: p.pos, line: lineno, scope: &empty };
                let e = q.expr(0)?;
                q.end()?;
                let v = const_value(&e).ok_or_else(|| {
                    LangError::new(lineno, ncol, "a param must be a constant number")
                })?;
                form.params.push(Param { name: name.clone(), value: v });
                scope.insert(name);
            }
            Tok::Ident(kw) if kw == "let" => {
                p.next();
                let (name, ncol) = p.ident("variable name")?;
                if Output::from_name(&name).is_some() {
                    return Err(LangError::new(lineno, ncol, format!("'{name}' is an output; write `{name} = ...` without let")));
                }
                if INPUTS.contains(&name.as_str()) || CONSTS.iter().any(|(c, _)| *c == name) || Func::from_name(&name).is_some() {
                    return Err(LangError::new(lineno, ncol, format!("'{name}' is a built-in name and cannot be redefined")));
                }
                if form.params.iter().any(|q| q.name == name) {
                    return Err(LangError::new(lineno, ncol, format!("'{name}' is already a param")));
                }
                p.expect(Tok::Eq, "'=' after the variable name")?;
                let e = p.expr(0)?;
                p.end()?;
                form.stmts.push(Stmt { target: Target::Let(name.clone()), expr: e });
                scope.insert(name);
            }
            Tok::Ident(name) => {
                let name = name.clone();
                let col = first.col;
                p.next();
                let Some(out) = Output::from_name(&name) else {
                    if matches!(p.peek().tok, Tok::Eq) {
                        return Err(LangError::new(
                            lineno,
                            col,
                            format!("'{name}' is not an output (x y z b h); use `let {name} = ...`"),
                        ));
                    }
                    return Err(LangError::new(lineno, col, "expected `let`, `param`, `@setting` or an output like `x = ...`"));
                };
                p.expect(Tok::Eq, "'='")?;
                let e = p.expr(0)?;
                p.end()?;
                form.stmts.push(Stmt { target: Target::Out(out), expr: e });
                scope.insert(name);
            }
            _ => {
                return Err(LangError::new(lineno, first.col, "expected `let`, `param`, `@setting` or an output like `x = ...`"));
            }
        }
    }

    if !form.has_output(Output::X) || !form.has_output(Output::Y) {
        let line = src.lines().count().max(1) as u32;
        let missing = if !form.has_output(Output::X) { "x" } else { "y" };
        return Err(LangError::new(line, 1, format!("a form needs both x and y; `{missing} = ...` is missing")));
    }
    Ok(form)
}

fn check_new_name(name: &str, line: u32, col: u32, scope: &HashSet<String>) -> Result<(), LangError> {
    if scope.contains(name) || Func::from_name(name).is_some() || Output::from_name(name).is_some() {
        return Err(LangError::new(line, col, format!("'{name}' is already defined")));
    }
    Ok(())
}

fn const_value(e: &Expr) -> Option<f64> {
    match e {
        Expr::Num(v) => Some(*v),
        Expr::Neg(a) => const_value(a).map(|v| -v),
        Expr::Var(v) => CONSTS.iter().find(|(c, _)| c == v).map(|(_, x)| *x),
        Expr::Bin(op, a, b) => {
            let (a, b) = (const_value(a)?, const_value(b)?);
            Some(match op {
                BinOp::Add => a + b,
                BinOp::Sub => a - b,
                BinOp::Mul => a * b,
                BinOp::Div => a / b,
                BinOp::Mod => a - b * (a / b).floor(),
                BinOp::Pow => a.powf(b),
            })
        }
        Expr::Call(..) => None,
    }
}

fn parse_setting(rest: &str, line: u32, col: u32, form: &mut Form) -> Result<(), LangError> {
    let mut parts = rest.split_whitespace();
    let key = parts.next().ok_or_else(|| LangError::new(line, col, "empty setting"))?;
    if !SETTINGS.contains(&key) {
        let hint = suggest(key, SETTINGS.iter().copied()).map(|s| format!(" (did you mean @{s}?)")).unwrap_or_default();
        return Err(LangError::new(line, col, format!("unknown setting @{key}{hint}")));
    }
    let val = parts.next().ok_or_else(|| LangError::new(line, col, format!("@{key} needs a value")))?;
    if parts.next().is_some() {
        return Err(LangError::new(line, col, format!("@{key} takes one value")));
    }
    let v = parse_set_value(val).ok_or_else(|| LangError::new(line, col, format!("bad value '{val}' for @{key}")))?;
    form.set(key, v);
    Ok(())
}

/// Numbers may carry k / m / b suffixes: `20m` is twenty million.
pub fn parse_set_value(s: &str) -> Option<SetVal> {
    let lower = s.to_ascii_lowercase();
    let (body, mul) = match lower.chars().last()? {
        'k' => (&lower[..lower.len() - 1], 1e3),
        'm' => (&lower[..lower.len() - 1], 1e6),
        'b' => (&lower[..lower.len() - 1], 1e9),
        _ => (lower.as_str(), 1.0),
    };
    if let Ok(v) = body.parse::<f64>() {
        return Some(SetVal::Num(v * mul));
    }
    if s.chars().all(|c| c.is_alphanumeric() || c == '_' || c == '-') {
        return Some(SetVal::Word(s.to_string()));
    }
    None
}

struct P<'a> {
    toks: &'a [Token],
    pos: usize,
    line: u32,
    scope: &'a HashSet<String>,
}

impl<'a> P<'a> {
    fn peek(&self) -> &Token {
        &self.toks[self.pos]
    }
    fn next(&mut self) -> Token {
        let t = self.toks[self.pos].clone();
        if self.pos + 1 < self.toks.len() {
            self.pos += 1;
        }
        t
    }
    fn err(&self, col: u32, msg: impl Into<String>) -> LangError {
        LangError::new(self.line, col, msg)
    }
    fn ident(&mut self, what: &str) -> Result<(String, u32), LangError> {
        let t = self.next();
        match t.tok {
            Tok::Ident(s) => Ok((s, t.col)),
            _ => Err(self.err(t.col, format!("expected {what}"))),
        }
    }
    fn expect(&mut self, tok: Tok, what: &str) -> Result<(), LangError> {
        let t = self.next();
        if t.tok == tok {
            Ok(())
        } else {
            Err(self.err(t.col, format!("expected {what}")))
        }
    }
    fn end(&mut self) -> Result<(), LangError> {
        let t = self.peek().clone();
        if t.tok == Tok::End {
            Ok(())
        } else {
            Err(self.err(t.col, "unexpected token after the end of the expression"))
        }
    }

    fn infix_bp(tok: &Tok) -> Option<(BinOp, u8, u8)> {
        Some(match tok {
            Tok::Plus => (BinOp::Add, 10, 11),
            Tok::Minus => (BinOp::Sub, 10, 11),
            Tok::Star => (BinOp::Mul, 20, 21),
            Tok::Slash => (BinOp::Div, 20, 21),
            Tok::Percent => (BinOp::Mod, 20, 21),
            Tok::Caret => (BinOp::Pow, 41, 40),
            _ => return None,
        })
    }

    fn expr(&mut self, min_bp: u8) -> Result<Expr, LangError> {
        let t = self.next();
        let mut lhs = match t.tok {
            Tok::Num(v) => Expr::Num(v),
            Tok::Minus => {
                let e = self.expr(30)?;
                match e {
                    Expr::Num(v) => Expr::Num(-v),
                    e => Expr::Neg(Box::new(e)),
                }
            }
            Tok::Plus => self.expr(30)?,
            Tok::LParen => {
                let e = self.expr(0)?;
                let c = self.peek().col;
                if self.next().tok != Tok::RParen {
                    return Err(self.err(c, "expected ')'"));
                }
                e
            }
            Tok::Ident(name) => {
                if self.peek().tok == Tok::LParen {
                    self.call(name, t.col)?
                } else {
                    if !self.scope.contains(&name) {
                        let mut msg = format!("unknown name '{name}'");
                        if (name == "x" || name == "y") && self.scope.contains("px") {
                            msg.push_str(" (it is not assigned yet; use px / py for the previous point)");
                        } else if let Some(s) = suggest(&name, self.scope.iter().map(|s| s.as_str())) {
                            msg.push_str(&format!(" (did you mean '{s}'?)"));
                        } else if Func::from_name(&name).is_some() {
                            msg = format!("'{name}' is a function; call it like {name}(...)");
                        }
                        return Err(self.err(t.col, msg));
                    }
                    Expr::Var(name)
                }
            }
            Tok::End => return Err(self.err(t.col, "expected an expression")),
            _ => return Err(self.err(t.col, "expected a number, a name or '('")),
        };

        loop {
            let Some((op, lbp, rbp)) = Self::infix_bp(&self.peek().tok) else {
                match self.peek().tok {
                    Tok::End | Tok::RParen | Tok::Comma => break,
                    Tok::Num(_) | Tok::Ident(_) | Tok::LParen => {
                        let c = self.peek().col;
                        return Err(self.err(c, "missing operator (implicit multiplication is not supported)"));
                    }
                    _ => {
                        let c = self.peek().col;
                        return Err(self.err(c, "unexpected token"));
                    }
                }
            };
            if lbp < min_bp {
                break;
            }
            self.next();
            let rhs = self.expr(rbp)?;
            lhs = Expr::bin(op, lhs, rhs);
        }
        Ok(lhs)
    }

    fn call(&mut self, name: String, col: u32) -> Result<Expr, LangError> {
        let Some(f) = Func::from_name(&name) else {
            let hint = suggest(&name, Func::ALL.iter().map(|f| f.name()))
                .map(|s| format!(" (did you mean '{s}'?)"))
                .unwrap_or_default();
            return Err(self.err(col, format!("unknown function '{name}'{hint}")));
        };
        self.next(); // (
        let mut args = Vec::new();
        if self.peek().tok != Tok::RParen {
            loop {
                args.push(self.expr(0)?);
                match self.peek().tok {
                    Tok::Comma => {
                        self.next();
                    }
                    Tok::RParen => break,
                    _ => {
                        let c = self.peek().col;
                        return Err(self.err(c, "expected ',' or ')'"));
                    }
                }
            }
        }
        self.next(); // )
        if args.len() != f.arity() {
            return Err(self.err(
                col,
                format!("{}() takes {} argument{}, got {}", f.name(), f.arity(), if f.arity() == 1 { "" } else { "s" }, args.len()),
            ));
        }
        Ok(Expr::Call(f, args))
    }
}
