use crate::error::LangError;

#[derive(Clone, Debug, PartialEq)]
pub enum Tok {
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

#[derive(Clone, Debug, PartialEq)]
pub struct Token {
    pub tok: Tok,
    pub col: u32,
}

/// Strip a trailing `#` or `//` comment.
pub fn strip_comment(line: &str) -> &str {
    let a = line.find('#');
    let b = line.find("//");
    match (a, b) {
        (Some(a), Some(b)) => &line[..a.min(b)],
        (Some(a), None) => &line[..a],
        (None, Some(b)) => &line[..b],
        (None, None) => line,
    }
}

/// Tokenize one statement line. Statements never span lines.
pub fn lex_line(line: &str, lineno: u32) -> Result<Vec<Token>, LangError> {
    let chars: Vec<char> = line.chars().collect();
    let mut out = Vec::new();
    let mut k = 0usize;
    while k < chars.len() {
        let c = chars[k];
        let col = k as u32 + 1;
        if c.is_whitespace() {
            k += 1;
            continue;
        }
        if c.is_ascii_digit() || (c == '.' && chars.get(k + 1).is_some_and(|d| d.is_ascii_digit())) {
            let start = k;
            while k < chars.len() && (chars[k].is_ascii_digit() || chars[k] == '.') {
                k += 1;
            }
            if k < chars.len() && (chars[k] == 'e' || chars[k] == 'E') {
                let save = k;
                k += 1;
                if k < chars.len() && (chars[k] == '+' || chars[k] == '-') {
                    k += 1;
                }
                if k < chars.len() && chars[k].is_ascii_digit() {
                    while k < chars.len() && chars[k].is_ascii_digit() {
                        k += 1;
                    }
                } else {
                    k = save;
                }
            }
            let text: String = chars[start..k].iter().collect();
            let v: f64 = text
                .parse()
                .map_err(|_| LangError::new(lineno, col, format!("bad number '{text}'")))?;
            if k < chars.len() && (chars[k].is_alphabetic() || chars[k] == '_') {
                return Err(LangError::new(
                    lineno,
                    k as u32 + 1,
                    format!("unexpected '{}' right after a number (write {} * {} ?)", chars[k], text, chars[k]),
                ));
            }
            out.push(Token { tok: Tok::Num(v), col });
            continue;
        }
        if c.is_alphabetic() || c == '_' {
            let start = k;
            while k < chars.len() && (chars[k].is_alphanumeric() || chars[k] == '_') {
                k += 1;
            }
            out.push(Token { tok: Tok::Ident(chars[start..k].iter().collect()), col });
            continue;
        }
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
            _ => return Err(LangError::new(lineno, col, format!("unexpected character '{c}'"))),
        };
        out.push(Token { tok, col });
        k += 1;
    }
    out.push(Token { tok: Tok::End, col: chars.len() as u32 + 1 });
    Ok(out)
}
