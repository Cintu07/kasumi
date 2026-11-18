//! The syntax tree of a form.

use std::f64::consts::{PI, TAU};

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

    /// Higher binds tighter.
    pub fn prec(self) -> u8 {
        match self {
            BinOp::Add | BinOp::Sub => 10,
            BinOp::Mul | BinOp::Div | BinOp::Mod => 20,
            BinOp::Pow => 40,
        }
    }
}

macro_rules! funcs {
    ($($variant:ident $name:literal $arity:literal $cuda:literal,)*) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum Func { $($variant,)* }

        impl Func {
            pub const ALL: &'static [Func] = &[$(Func::$variant,)*];
            pub fn name(self) -> &'static str { match self { $(Func::$variant => $name,)* } }
            pub fn arity(self) -> usize { match self { $(Func::$variant => $arity,)* } }
            pub fn cuda(self) -> &'static str { match self { $(Func::$variant => $cuda,)* } }
        }
    };
}

funcs! {
    Sin "sin" 1 "sinf",
    Cos "cos" 1 "cosf",
    Tan "tan" 1 "tanf",
    Tanh "tanh" 1 "tanhf",
    Atan2 "atan2" 2 "atan2f",
    Sqrt "sqrt" 1 "sqrtf",
    Abs "abs" 1 "fabsf",
    Fract "fract" 1 "ks_fract",
    Floor "floor" 1 "floorf",
    Sign "sign" 1 "ks_sign",
    Min "min" 2 "fminf",
    Max "max" 2 "fmaxf",
    Mix "mix" 3 "ks_mix",
    Clamp "clamp" 3 "ks_clamp",
    Smoothstep "smoothstep" 3 "ks_smooth",
    Mag "mag" 2 "ks_mag",
    Exp "exp" 1 "expf",
    Log "log" 1 "logf",
    Pow "pow" 2 "powf",
    Noise "noise" 2 "ks_noise",
    Rnd "rnd" 1 "ks_rnd",
}

impl Func {
    pub fn from_name(s: &str) -> Option<Func> {
        Func::ALL.iter().copied().find(|f| f.name() == s)
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

    fn kids(&self) -> Vec<&Expr> {
        match self {
            Expr::Num(_) | Expr::Var(_) => vec![],
            Expr::Neg(a) => vec![a],
            Expr::Bin(_, a, b) => vec![a, b],
            Expr::Call(_, args) => args.iter().collect(),
        }
    }

    fn kids_mut(&mut self) -> Vec<&mut Expr> {
        match self {
            Expr::Num(_) | Expr::Var(_) => vec![],
            Expr::Neg(a) => vec![a],
            Expr::Bin(_, a, b) => vec![a, b],
            Expr::Call(_, args) => args.iter_mut().collect(),
        }
    }

    pub fn visit_vars(&self, f: &mut dyn FnMut(&str)) {
        if let Expr::Var(v) = self {
            f(v);
        }
        self.kids().iter().for_each(|k| k.visit_vars(f));
    }

    pub fn uses(&self, name: &str) -> bool {
        let mut hit = false;
        self.visit_vars(&mut |v| hit |= v == name);
        hit
    }

    /// Number of nodes in this tree.
    pub fn size(&self) -> usize {
        1 + self.kids().iter().map(|k| k.size()).sum::<usize>()
    }

    /// The node at a pre-order index, where 0 is this node.
    pub fn node(&self, idx: usize) -> Option<&Expr> {
        if idx == 0 {
            return Some(self);
        }
        let mut left = idx - 1;
        for k in self.kids() {
            if left < k.size() {
                return k.node(left);
            }
            left -= k.size();
        }
        None
    }

    pub fn node_mut(&mut self, idx: usize) -> Option<&mut Expr> {
        if idx == 0 {
            return Some(self);
        }
        let mut left = idx - 1;
        for k in self.kids_mut() {
            if left < k.size() {
                return k.node_mut(left);
            }
            left -= k.size();
        }
        None
    }
}

/// What a form can write: position, depth, brightness and hue.
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
        ["x", "y", "z", "b", "h"][self as usize]
    }

    pub fn from_name(s: &str) -> Option<Output> {
        Output::ALL.into_iter().find(|o| o.name() == s)
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

pub const INPUTS: &[&str] = &["i", "n", "t", "u", "r1", "r2", "px", "py"];
pub const CONSTS: &[(&str, f64)] = &[("pi", PI), ("tau", TAU)];
pub const SETTINGS: &[&str] = &[
    "points", "iters", "warmup", "palette", "zoom", "cx", "cy", "gamma", "vibrancy", "heat", "exposure", "trail", "dof",
    "focus", "shutter", "fastmath", "bloom", "de",
];

/// A parsed form.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Form {
    /// Leading comment lines, without the `#`.
    pub header: Vec<String>,
    pub settings: Vec<(String, SetVal)>,
    pub params: Vec<Param>,
    pub stmts: Vec<Stmt>,
}

impl Form {
    fn setting(&self, key: &str) -> Option<&SetVal> {
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
            Some(SetVal::Word(w)) => matches!(w.as_str(), "on" | "true" | "yes"),
            Some(SetVal::Num(v)) => *v != 0.0,
            None => false,
        }
    }

    pub fn set(&mut self, key: &str, v: SetVal) {
        match self.settings.iter_mut().find(|(k, _)| k == key) {
            Some(slot) => slot.1 = v,
            None => self.settings.push((key.to_string(), v)),
        }
    }

    /// A form that reads the previous point is an iterated system: each GPU
    /// thread walks one chain.
    pub fn is_attractor(&self) -> bool {
        self.stmts.iter().any(|s| s.expr.uses("px") || s.expr.uses("py"))
    }

    pub fn has_output(&self, o: Output) -> bool {
        self.stmts.iter().any(|s| s.target == Target::Out(o))
    }
}
