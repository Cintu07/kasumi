use std::fmt;

/// An error with a source position (1-based line and column).
#[derive(Debug, Clone, PartialEq)]
pub struct LangError {
    pub line: u32,
    pub col: u32,
    pub msg: String,
}

impl LangError {
    pub fn new(line: u32, col: u32, msg: impl Into<String>) -> Self {
        Self { line, col, msg: msg.into() }
    }

    /// Multi-line rendering with the offending line and a caret.
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

/// Levenshtein distance, for "did you mean" hints.
pub fn edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for i in 1..=a.len() {
        let mut cur = vec![i; b.len() + 1];
        for j in 1..=b.len() {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        prev = cur;
    }
    prev[b.len()]
}

pub fn suggest<'a>(name: &str, candidates: impl Iterator<Item = &'a str>) -> Option<&'a str> {
    candidates
        .map(|c| (edit_distance(name, c), c))
        .filter(|(d, c)| *d <= 2 && *d < c.len())
        .min_by_key(|(d, _)| *d)
        .map(|(_, c)| c)
}
