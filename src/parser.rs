//! Turns form text into a syntax tree. Statements never span lines, so each
//! line is lexed on its own and expressions use a Pratt parser.

use std::collections::BTreeSet;
use std::fmt;

use crate::ast::*;

/// An error at a 1-based line and column.
#[derive(Debug, Clone, PartialEq)]
pub struct LangError {
    pub line: u32,
    pub col: u32,
    pub msg: String,
}

impl LangError {
    fn new(line: u32, col: u32, msg: impl Into<String>) -> Self {
        Self { line, col, msg: msg.into() }
    }

    /// The message, the offending line and a caret under the column.
    pub fn render(&self, src: &str) -> String {
        let text = src.lines().nth(self.line.saturating_sub(1) as usize).unwrap_or("");
        let pad = " ".repeat(self.col.saturating_sub(1) as usize);
        format!("{}:{}: {}\n  {}\n  {}^", self.line, self.col, self.msg, text, pad)
    }
}

impl fmt::Display for LangError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}: {}", self.line, self.col, self.msg)
    }
}

impl std::error::Error for LangError {}

fn edit_distance(a: &str, b: &str) -> usize {
    let (a, b): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for i in 1..=a.len() {
        let mut cur = vec![i; b.len() + 1];
        for j in 1..=b.len() {
            let swap = prev[j - 1] + (a[i - 1] != b[j - 1]) as usize;
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(swap);
        }
        prev = cur;
    }
    prev[b.len()]
}

/// The closest known name, for "did you mean" hints.
fn suggest<'a>(name: &str, known: impl Iterator<Item = &'a str>) -> Option<&'a str> {
    known
        .map(|c| (edit_distance(name, c), c))
        .filter(|(d, c)| *d <= 2 && *d < c.len())
        .min_by_key(|(d, _)| *d)
        .map(|(_, c)| c)
}

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Num(f64),
    Ident(String),
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Caret,
    LParen,
    RParen,
    Comma,
    Eq,
    End,
}

#[derive(Clone, Debug)]
struct Token {
    tok: Tok,
    col: u32,
}

fn strip_comment(line: &str) -> &str {
    let end = [line.find('#'), line.find("//")].into_iter().flatten().min();
    &line[..end.unwrap_or(line.len())]
}

fn lex_number(chars: &[char], start: usize, line: u32) -> Result<(f64, usize), LangError> {
    let digit = |k: usize| chars.get(k).is_some_and(|c| c.is_ascii_digit());
    let mut k = start;
    while digit(k) || chars.get(k) == Some(&'.') {
        k += 1;
    }
    if matches!(chars.get(k), Some('e' | 'E')) {
        let sign = matches!(chars.get(k + 1), Some('+' | '-')) as usize;
        if digit(k + 1 + sign) {
            k += 1 + sign;
            while digit(k) {
                k += 1;
            }
        }
    }
    let text: String = chars[start..k].iter().collect();
    let col = start as u32 + 1;
    let v = text.parse().map_err(|_| LangError::new(line, col, format!("bad number '{text}'")))?;
    if let Some(&c) = chars.get(k).filter(|c| c.is_alphabetic() || **c == '_') {
        let msg = format!("unexpected '{c}' right after a number (write {text} * {c} ?)");
        return Err(LangError::new(line, k as u32 + 1, msg));
    }
    Ok((v, k))
}

fn lex_line(text: &str, line: u32) -> Result<Vec<Token>, LangError> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut k = 0;
    while k < chars.len() {
        let c = chars[k];
        let col = k as u32 + 1;
        if c.is_whitespace() {
            k += 1;
        } else if c.is_ascii_digit() || (c == '.' && chars.get(k + 1).is_some_and(|d| d.is_ascii_digit())) {
            let (v, end) = lex_number(&chars, k, line)?;
            out.push(Token { tok: Tok::Num(v), col });
            k = end;
        } else if c.is_alphabetic() || c == '_' {
            let start = k;
            while k < chars.len() && (chars[k].is_alphanumeric() || chars[k] == '_') {
                k += 1;
            }
            out.push(Token { tok: Tok::Ident(chars[start..k].iter().collect()), col });
        } else {
            let tok = match c {
                '+' => Tok::Plus,
                '-' => Tok::Minus,
                '*' => Tok::Star,
                '/' => Tok::Slash,
                '%' => Tok::Percent,
                '^' => Tok::Caret,
                '(' => Tok::LParen,
                ')' => Tok::RParen,
                ',' => Tok::Comma,
                '=' => Tok::Eq,
                _ => return Err(LangError::new(line, col, format!("unexpected character '{c}'"))),
            };
            out.push(Token { tok, col });
            k += 1;
        }
    }
    out.push(Token { tok: Tok::End, col: chars.len() as u32 + 1 });
    Ok(out)
}

const NOT_A_STATEMENT: &str = "expected `let`, `param`, `@setting` or an output like `x = ...`";

pub fn parse(src: &str) -> Result<Form, LangError> {
    let mut form = Form::default();
    // names a statement may read: inputs, constants, params, earlier bindings
    let mut scope: BTreeSet<String> = INPUTS.iter().map(|s| s.to_string()).collect();
    scope.extend(CONSTS.iter().map(|(c, _)| c.to_string()));
    let mut in_header = true;

    for (idx, raw) in src.lines().enumerate() {
        let line = idx as u32 + 1;
        if let (true, Some(rest)) = (in_header, raw.trim().strip_prefix('#')) {
            form.header.push(rest.trim().to_string());
            continue;
        }
        let text = strip_comment(raw);
        if text.trim().is_empty() {
            continue;
        }
        in_header = false;
        if let Some(rest) = text.trim_start().strip_prefix('@') {
            let lead = (text.len() - text.trim_start().len()) as u32;
            parse_setting(rest, line, lead + 2, &mut form)?;
            continue;
        }
        let toks = lex_line(text, line)?;
        let mut p = Parser { toks: &toks, pos: 0, line, scope: &scope };
        let name = p.statement(&mut form)?;
        scope.insert(name);
    }

    if let Some(missing) = [Output::X, Output::Y].into_iter().find(|o| !form.has_output(*o)) {
        let line = src.lines().count().max(1) as u32;
        let msg = format!("a form needs both x and y; `{} = ...` is missing", missing.name());
        return Err(LangError::new(line, 1, msg));
    }
    Ok(form)
}

/// The value of a constant expression, so a param stays a plain number.
fn const_value(e: &Expr) -> Option<f64> {
    match e {
        Expr::Num(v) => Some(*v),
        Expr::Neg(a) => const_value(a).map(|v| -v),
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
        Expr::Var(_) | Expr::Call(..) => None,
    }
}

fn parse_setting(rest: &str, line: u32, col: u32, form: &mut Form) -> Result<(), LangError> {
    let err = |msg: String| LangError::new(line, col, msg);
    let mut parts = rest.split_whitespace();
    let key = parts.next().ok_or_else(|| err("empty setting".into()))?;
    if !SETTINGS.contains(&key) {
        let hint = suggest(key, SETTINGS.iter().copied()).map(|s| format!(" (did you mean @{s}?)")).unwrap_or_default();
        return Err(err(format!("unknown setting @{key}{hint}")));
    }
    let val = parts.next().ok_or_else(|| err(format!("@{key} needs a value")))?;
    if parts.next().is_some() {
        return Err(err(format!("@{key} takes one value")));
    }
    let v = parse_set_value(val).ok_or_else(|| err(format!("bad value '{val}' for @{key}")))?;
    form.set(key, v);
    Ok(())
}

/// Numbers may carry a k, m or b suffix: `20m` is twenty million.
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
    s.chars().all(|c| c.is_alphanumeric() || c == '_' || c == '-').then(|| SetVal::Word(s.to_string()))
}

struct Parser<'a> {
    toks: &'a [Token],
    pos: usize,
    line: u32,
    scope: &'a BTreeSet<String>,
}

impl Parser<'_> {
    fn peek(&self) -> &Token {
        &self.toks[self.pos]
    }

    fn next(&mut self) -> Token {
        let t = self.toks[self.pos].clone();
        self.pos = (self.pos + 1).min(self.toks.len() - 1);
        t
    }

    fn err<T>(&self, col: u32, msg: impl Into<String>) -> Result<T, LangError> {
        Err(LangError::new(self.line, col, msg))
    }

    fn ident(&mut self, what: &str) -> Result<(String, u32), LangError> {
        let t = self.next();
        match t.tok {
            Tok::Ident(s) => Ok((s, t.col)),
            _ => self.err(t.col, format!("expected {what}")),
        }
    }

    fn expect_eq(&mut self, what: &str) -> Result<(), LangError> {
        let t = self.next();
        if t.tok == Tok::Eq {
            return Ok(());
        }
        self.err(t.col, format!("expected {what}"))
    }

    /// An expression that must run to the end of the line.
    fn whole_expr(&mut self) -> Result<Expr, LangError> {
        let e = self.expr(0)?;
        if self.peek().tok == Tok::End {
            return Ok(e);
        }
        self.err(self.peek().col, "unexpected token after the end of the expression")
    }

    /// Parses one line into `form` and returns the name it defines.
    fn statement(&mut self, form: &mut Form) -> Result<String, LangError> {
        let first = self.next();
        let Tok::Ident(word) = first.tok else { return self.err(first.col, NOT_A_STATEMENT) };
        match word.as_str() {
            "param" => self.param(form),
            "let" => self.binding(form),
            _ => self.output(form, word, first.col),
        }
    }

    fn param(&mut self, form: &mut Form) -> Result<String, LangError> {
        let (name, col) = self.ident("parameter name")?;
        if self.scope.contains(&name) || Func::from_name(&name).is_some() || Output::from_name(&name).is_some() {
            return self.err(col, format!("'{name}' is already defined"));
        }
        self.expect_eq("'=' after the parameter name")?;
        // params are plain numbers so the explorer can nudge them
        let empty = BTreeSet::new();
        let e = Parser { toks: self.toks, pos: self.pos, line: self.line, scope: &empty }.whole_expr()?;
        let Some(value) = const_value(&e) else { return self.err(col, "a param must be a constant number") };
        form.params.push(Param { name: name.clone(), value });
        Ok(name)
    }

    fn binding(&mut self, form: &mut Form) -> Result<String, LangError> {
        let (name, col) = self.ident("variable name")?;
        if Output::from_name(&name).is_some() {
            return self.err(col, format!("'{name}' is an output; write `{name} = ...` without let"));
        }
        if INPUTS.contains(&name.as_str()) || CONSTS.iter().any(|(c, _)| *c == name) || Func::from_name(&name).is_some() {
            return self.err(col, format!("'{name}' is a built-in name and cannot be redefined"));
        }
        if form.params.iter().any(|p| p.name == name) {
            return self.err(col, format!("'{name}' is already a param"));
        }
        self.expect_eq("'=' after the variable name")?;
        let expr = self.whole_expr()?;
        form.stmts.push(Stmt { target: Target::Let(name.clone()), expr });
        Ok(name)
    }

    fn output(&mut self, form: &mut Form, name: String, col: u32) -> Result<String, LangError> {
        let Some(out) = Output::from_name(&name) else {
            if self.peek().tok == Tok::Eq {
                return self.err(col, format!("'{name}' is not an output (x y z b h); use `let {name} = ...`"));
            }
            return self.err(col, NOT_A_STATEMENT);
        };
        self.expect_eq("'='")?;
        let expr = self.whole_expr()?;
        form.stmts.push(Stmt { target: Target::Out(out), expr });
        Ok(name)
    }

    fn infix(tok: &Tok) -> Option<(BinOp, u8, u8)> {
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

    fn expr(&mut self, min_power: u8) -> Result<Expr, LangError> {
        let mut lhs = self.prefix()?;
        loop {
            let t = self.peek().clone();
            let Some((op, left, right)) = Self::infix(&t.tok) else {
                return match t.tok {
                    Tok::End | Tok::RParen | Tok::Comma => Ok(lhs),
                    Tok::Num(_) | Tok::Ident(_) | Tok::LParen => {
                        self.err(t.col, "missing operator (implicit multiplication is not supported)")
                    }
                    _ => self.err(t.col, "unexpected token"),
                };
            };
            if left < min_power {
                return Ok(lhs);
            }
            self.next();
            lhs = Expr::bin(op, lhs, self.expr(right)?);
        }
    }

    fn prefix(&mut self) -> Result<Expr, LangError> {
        let t = self.next();
        match t.tok {
            Tok::Num(v) => Ok(Expr::Num(v)),
            Tok::Minus => Ok(match self.expr(30)? {
                Expr::Num(v) => Expr::Num(-v),
                e => Expr::Neg(Box::new(e)),
            }),
            Tok::Plus => self.expr(30),
            Tok::LParen => {
                let e = self.expr(0)?;
                let close = self.next();
                if close.tok != Tok::RParen {
                    return self.err(close.col, "expected ')'");
                }
                Ok(e)
            }
            Tok::Ident(name) if self.peek().tok == Tok::LParen => self.call(name, t.col),
            Tok::Ident(name) => self.name(name, t.col),
            Tok::End => self.err(t.col, "expected an expression"),
            _ => self.err(t.col, "expected a number, a name or '('"),
        }
    }

    fn name(&self, name: String, col: u32) -> Result<Expr, LangError> {
        if self.scope.contains(&name) {
            return Ok(Expr::Var(name));
        }
        let mut msg = format!("unknown name '{name}'");
        if (name == "x" || name == "y") && self.scope.contains("px") {
            msg.push_str(" (it is not assigned yet; use px / py for the previous point)");
        } else if let Some(s) = suggest(&name, self.scope.iter().map(|s| s.as_str())) {
            msg.push_str(&format!(" (did you mean '{s}'?)"));
        } else if Func::from_name(&name).is_some() {
            msg = format!("'{name}' is a function; call it like {name}(...)");
        }
        self.err(col, msg)
    }

    fn call(&mut self, name: String, col: u32) -> Result<Expr, LangError> {
        let Some(f) = Func::from_name(&name) else {
            let hint = suggest(&name, Func::ALL.iter().map(|f| f.name()));
            let hint = hint.map(|s| format!(" (did you mean '{s}'?)")).unwrap_or_default();
            return self.err(col, format!("unknown function '{name}'{hint}"));
        };
        self.next(); // the opening paren
        let mut args = Vec::new();
        if self.peek().tok != Tok::RParen {
            loop {
                args.push(self.expr(0)?);
                let t = self.peek().clone();
                match t.tok {
                    Tok::Comma => self.next(),
                    Tok::RParen => break,
                    _ => return self.err(t.col, "expected ',' or ')'"),
                };
            }
        }
        self.next(); // the closing paren
        if args.len() != f.arity() {
            let plural = if f.arity() == 1 { "" } else { "s" };
            return self.err(col, format!("{}() takes {} argument{plural}, got {}", f.name(), f.arity(), args.len()));
        }
        Ok(Expr::Call(f, args))
    }
}
