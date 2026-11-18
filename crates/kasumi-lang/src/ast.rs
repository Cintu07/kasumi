use crate::func::Func;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
}

impl BinOp {
    pub fn sym(self) -> &'static str {
        match self {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::Mod => "%",
            BinOp::Pow => "^",
        }
    }
    /// Printing precedence (higher binds tighter).
    pub fn prec(self) -> u8 {
        match self {
            BinOp::Add | BinOp::Sub => 10,
            BinOp::Mul | BinOp::Div | BinOp::Mod => 20,
            BinOp::Pow => 40,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Num(f64),
    Var(String),
    Neg(Box<Expr>),
    Bin(BinOp, Box<Expr>, Box<Expr>),
    Call(Func, Vec<Expr>),
}

impl Expr {
    pub fn bin(op: BinOp, a: Expr, b: Expr) -> Expr {
        Expr::Bin(op, Box::new(a), Box::new(b))
    }
    pub fn var(s: &str) -> Expr {
        Expr::Var(s.to_string())
    }

    pub fn visit_vars(&self, f: &mut dyn FnMut(&str)) {
        match self {
            Expr::Num(_) => {}
            Expr::Var(v) => f(v),
            Expr::Neg(e) => e.visit_vars(f),
            Expr::Bin(_, a, b) => {
                a.visit_vars(f);
                b.visit_vars(f);
            }
            Expr::Call(_, args) => args.iter().for_each(|a| a.visit_vars(f)),
        }
    }

    pub fn uses(&self, name: &str) -> bool {
        let mut hit = false;
        self.visit_vars(&mut |v| hit |= v == name);
        hit
    }

    /// Number of nodes.
    pub fn size(&self) -> usize {
        match self {
            Expr::Num(_) | Expr::Var(_) => 1,
            Expr::Neg(e) => 1 + e.size(),
            Expr::Bin(_, a, b) => 1 + a.size() + b.size(),
            Expr::Call(_, args) => 1 + args.iter().map(|a| a.size()).sum::<usize>(),
        }
    }

    /// Pre-order node access by index (0 = self).
    pub fn node_mut(&mut self, idx: usize) -> Option<&mut Expr> {
        fn go<'a>(e: &'a mut Expr, idx: &mut usize) -> Option<&'a mut Expr> {
            if *idx == 0 {
                return Some(e);
            }
            *idx -= 1;
            match e {
                Expr::Num(_) | Expr::Var(_) => None,
                Expr::Neg(a) => go(a, idx),
                Expr::Bin(_, a, b) => {
                    if let Some(r) = go(a, idx) {
                        return Some(r);
                    }
                    go(b, idx)
                }
                Expr::Call(_, args) => {
                    for a in args.iter_mut() {
                        if let Some(r) = go(a, idx) {
                            return Some(r);
                        }
                    }
                    None
                }
            }
        }
        let mut i = idx;
        go(self, &mut i)
    }

    pub fn node(&self, idx: usize) -> Option<&Expr> {
        fn go<'a>(e: &'a Expr, idx: &mut usize) -> Option<&'a Expr> {
            if *idx == 0 {
                return Some(e);
            }
            *idx -= 1;
            match e {
                Expr::Num(_) | Expr::Var(_) => None,
                Expr::Neg(a) => go(a, idx),
                Expr::Bin(_, a, b) => go(a, idx).or_else(|| go(b, idx)),
                Expr::Call(_, args) => {
                    for a in args {
                        if let Some(r) = go(a, idx) {
                            return Some(r);
                        }
                    }
                    None
                }
            }
        }
        let mut i = idx;
        go(self, &mut i)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Output {
    X,
    Y,
    Z,
    B,
    H,
}

impl Output {
    pub const ALL: [Output; 5] = [Output::X, Output::Y, Output::Z, Output::B, Output::H];
    pub fn name(self) -> &'static str {
        match self {
            Output::X => "x",
            Output::Y => "y",
            Output::Z => "z",
            Output::B => "b",
            Output::H => "h",
        }
    }
    pub fn from_name(s: &str) -> Option<Output> {
        Output::ALL.iter().copied().find(|o| o.name() == s)
    }
    pub fn index(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Target {
    Let(String),
    Out(Output),
}

impl Target {
    pub fn name(&self) -> &str {
        match self {
            Target::Let(s) => s,
            Target::Out(o) => o.name(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Stmt {
    pub target: Target,
    pub expr: Expr,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Param {
    pub name: String,
    pub value: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum SetVal {
    Num(f64),
    Word(String),
}

/// A parsed form.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Form {
    /// Leading comment lines (without the `#`).
    pub header: Vec<String>,
    pub settings: Vec<(String, SetVal)>,
    pub params: Vec<Param>,
    pub stmts: Vec<Stmt>,
}

/// Per-point inputs. `n` and `t` are the same for every point in a frame.
pub const INPUTS: &[&str] = &["i", "n", "t", "u", "r1", "r2", "px", "py"];
pub const CONSTS: &[(&str, f64)] = &[("pi", std::f64::consts::PI), ("tau", std::f64::consts::TAU)];
pub const KEYWORDS: &[&str] = &["let", "param"];

/// Every `@setting` the language accepts.
pub const SETTINGS: &[&str] = &[
    "points", "iters", "warmup", "palette", "zoom", "cx", "cy", "gamma", "vibrancy", "heat",
    "exposure", "trail", "dof", "focus", "shutter", "fastmath", "bloom", "de",
];

impl Form {
    pub fn setting(&self, key: &str) -> Option<&SetVal> {
        self.settings.iter().rev().find(|(k, _)| k == key).map(|(_, v)| v)
    }
    pub fn num(&self, key: &str) -> Option<f64> {
        match self.setting(key) {
            Some(SetVal::Num(v)) => Some(*v),
            _ => None,
        }
    }
    pub fn num_or(&self, key: &str, default: f64) -> f64 {
        self.num(key).unwrap_or(default)
    }
    pub fn word(&self, key: &str) -> Option<&str> {
        match self.setting(key) {
            Some(SetVal::Word(w)) => Some(w),
            _ => None,
        }
    }
    pub fn flag(&self, key: &str) -> bool {
        match self.setting(key) {
            Some(SetVal::Word(w)) => w == "on" || w == "true" || w == "yes",
            Some(SetVal::Num(v)) => *v != 0.0,
            None => false,
        }
    }
    pub fn set(&mut self, key: &str, v: SetVal) {
        if let Some(slot) = self.settings.iter_mut().find(|(k, _)| k == key) {
            slot.1 = v;
        } else {
            self.settings.push((key.to_string(), v));
        }
    }

    /// A form that reads the previous point (`px`, `py`) is an iterated
    /// system (strange attractor): each gpu thread walks one chain.
    pub fn is_attractor(&self) -> bool {
        self.stmts.iter().any(|s| s.expr.uses("px") || s.expr.uses("py"))
    }

    pub fn has_output(&self, o: Output) -> bool {
        self.stmts.iter().any(|s| s.target == Target::Out(o))
    }
}
