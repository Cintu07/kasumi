//! A small ssa ir. Every value is assigned once; values are hash-consed while
//! the ir is built, which gives common subexpression elimination for free,
//! and constant folding / algebraic simplification run at construction time.
//! After lowering: dead code removal, and every op that does not depend on the
//! point is marked uniform and hoisted out of the per-point kernel (the cpu
//! evaluates it once per frame and the kernel reads it from a small table).

use std::collections::HashMap;

use crate::ast::*;
use crate::func::Func;
use crate::interp::{eval_bin, eval_call};

pub type V = u32;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Input {
    I,
    N,
    T,
    U,
    R1,
    R2,
    Px,
    Py,
}

impl Input {
    pub fn from_name(s: &str) -> Option<Input> {
        Some(match s {
            "i" => Input::I,
            "n" => Input::N,
            "t" => Input::T,
            "u" => Input::U,
            "r1" => Input::R1,
            "r2" => Input::R2,
            "px" => Input::Px,
            "py" => Input::Py,
            _ => return None,
        })
    }
    /// Does this input change from point to point?
    pub fn varying(self) -> bool {
        !matches!(self, Input::N | Input::T)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Op {
    /// f32 bits, so ops can be hashed
    Const(u32),
    In(Input),
    Param(u32),
    Neg(V),
    Bin(BinOp, V, V),
    Call(Func, Vec<V>),
}

impl Op {
    pub fn operands(&self) -> Vec<V> {
        match self {
            Op::Const(_) | Op::In(_) | Op::Param(_) => vec![],
            Op::Neg(a) => vec![*a],
            Op::Bin(_, a, b) => vec![*a, *b],
            Op::Call(_, args) => args.clone(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct IrStats {
    pub ast_nodes: usize,
    pub folded: usize,
    pub cse_hits: usize,
    pub dead: usize,
    /// live ops that do not depend on the point: computed once per frame
    pub hoisted: usize,
    /// live ops computed per point on the gpu
    pub per_point: usize,
}

#[derive(Clone, Debug)]
pub struct Ir {
    pub ops: Vec<Op>,
    pub varying: Vec<bool>,
    pub live: Vec<bool>,
    /// x y z b h
    pub outs: [Option<V>; 5],
    pub params: Vec<(String, f32)>,
    /// uniform ops the gpu reads, in slot order
    pub slots: Vec<V>,
    pub slot_of: HashMap<V, usize>,
    pub attractor: bool,
    pub stats: IrStats,
}

impl Ir {
    pub fn has_b(&self) -> bool {
        self.outs[3].is_some()
    }
    pub fn has_h(&self) -> bool {
        self.outs[4].is_some()
    }
    pub fn has_z(&self) -> bool {
        self.outs[2].is_some()
    }
    pub fn const_val(&self, v: V) -> Option<f32> {
        match self.ops[v as usize] {
            Op::Const(b) => Some(f32::from_bits(b)),
            _ => None,
        }
    }
    pub fn param_values(&self) -> Vec<f32> {
        self.params.iter().map(|p| p.1).collect()
    }
}

struct Builder {
    ops: Vec<Op>,
    map: HashMap<Op, V>,
    stats: IrStats,
}

impl Builder {
    fn intern(&mut self, op: Op) -> V {
        if let Some(&v) = self.map.get(&op) {
            if !matches!(op, Op::Const(_) | Op::In(_) | Op::Param(_)) {
                self.stats.cse_hits += 1;
            }
            return v;
        }
        let v = self.ops.len() as V;
        self.ops.push(op.clone());
        self.map.insert(op, v);
        v
    }
    fn konst(&mut self, c: f32) -> V {
        // canonical NaN so all NaNs share one op
        let bits = if c.is_nan() { f32::NAN.to_bits() } else { c.to_bits() };
        self.intern(Op::Const(bits))
    }
    fn cval(&self, v: V) -> Option<f32> {
        match self.ops[v as usize] {
            Op::Const(b) => Some(f32::from_bits(b)),
            _ => None,
        }
    }
    fn fold(&mut self, c: f32) -> V {
        self.stats.folded += 1;
        self.konst(c)
    }
    fn simplified(&mut self, v: V) -> V {
        self.stats.folded += 1;
        v
    }

    fn neg(&mut self, a: V) -> V {
        if let Some(c) = self.cval(a) {
            return self.fold(-c);
        }
        if let Op::Neg(x) = self.ops[a as usize] {
            return self.simplified(x);
        }
        self.intern(Op::Neg(a))
    }

    fn bin(&mut self, op: BinOp, a: V, b: V) -> V {
        let (ca, cb) = (self.cval(a), self.cval(b));
        if let (Some(x), Some(y)) = (ca, cb) {
            return self.fold(eval_bin(op, x, y));
        }
        match op {
            BinOp::Add => {
                if cb == Some(0.0) {
                    return self.simplified(a);
                }
                if ca == Some(0.0) {
                    return self.simplified(b);
                }
            }
            BinOp::Sub => {
                if cb == Some(0.0) {
                    return self.simplified(a);
                }
                if ca == Some(0.0) {
                    self.stats.folded += 1;
                    return self.neg(b);
                }
            }
            BinOp::Mul => {
                if cb == Some(1.0) {
                    return self.simplified(a);
                }
                if ca == Some(1.0) {
                    return self.simplified(b);
                }
                if cb == Some(-1.0) {
                    self.stats.folded += 1;
                    return self.neg(a);
                }
                if ca == Some(-1.0) {
                    self.stats.folded += 1;
                    return self.neg(b);
                }
            }
            BinOp::Div => {
                if cb == Some(1.0) {
                    return self.simplified(a);
                }
                // x / 2^k  ->  x * 2^-k  (exact)
                if let Some(c) = cb {
                    let bits = c.to_bits();
                    if c.is_normal() && bits & 0x007f_ffff == 0 && (1.0 / c).is_normal() {
                        self.stats.folded += 1;
                        let r = self.konst(1.0 / c);
                        return self.bin(BinOp::Mul, a, r);
                    }
                }
            }
            BinOp::Pow => {
                if cb == Some(1.0) {
                    return self.simplified(a);
                }
                if cb == Some(2.0) {
                    self.stats.folded += 1;
                    return self.bin(BinOp::Mul, a, a);
                }
                if cb == Some(3.0) {
                    self.stats.folded += 1;
                    let sq = self.bin(BinOp::Mul, a, a);
                    return self.bin(BinOp::Mul, sq, a);
                }
                if cb == Some(0.5) {
                    self.stats.folded += 1;
                    return self.call(Func::Sqrt, vec![a]);
                }
            }
            BinOp::Mod => {}
        }
        let (a, b) = if matches!(op, BinOp::Add | BinOp::Mul) && a > b { (b, a) } else { (a, b) };
        self.intern(Op::Bin(op, a, b))
    }

    fn call(&mut self, f: Func, mut args: Vec<V>) -> V {
        let cs: Option<Vec<f32>> = args.iter().map(|&a| self.cval(a)).collect();
        if let Some(cs) = cs {
            return self.fold(eval_call(f, &cs));
        }
        if matches!(f, Func::Min | Func::Max | Func::Mag) && args[0] > args[1] {
            args.swap(0, 1);
        }
        self.intern(Op::Call(f, args))
    }
}

/// Lower a parsed form to ssa ir and run all passes.
pub fn lower(form: &Form) -> Ir {
    let mut b = Builder { ops: Vec::new(), map: HashMap::new(), stats: IrStats::default() };
    let params: HashMap<&str, u32> = form.params.iter().enumerate().map(|(k, p)| (p.name.as_str(), k as u32)).collect();
    let mut env: HashMap<String, V> = HashMap::new();
    let mut outs: [Option<V>; 5] = [None; 5];

    fn expr(b: &mut Builder, e: &Expr, env: &HashMap<String, V>, params: &HashMap<&str, u32>) -> V {
        b.stats.ast_nodes += 1;
        match e {
            Expr::Num(v) => b.konst(*v as f32),
            Expr::Var(name) => {
                if let Some(&v) = env.get(name) {
                    v
                } else if let Some(inp) = Input::from_name(name) {
                    b.intern(Op::In(inp))
                } else if let Some(&k) = params.get(name.as_str()) {
                    b.intern(Op::Param(k))
                } else if let Some((_, c)) = CONSTS.iter().find(|(c, _)| c == name) {
                    b.konst(*c as f32)
                } else {
                    // the parser rejects unknown names; be total anyway
                    b.konst(f32::NAN)
                }
            }
            Expr::Neg(a) => {
                let a = expr(b, a, env, params);
                b.neg(a)
            }
            Expr::Bin(op, x, y) => {
                let x = expr(b, x, env, params);
                let y = expr(b, y, env, params);
                b.bin(*op, x, y)
            }
            Expr::Call(f, args) => {
                let args = args.iter().map(|a| expr(b, a, env, params)).collect();
                b.call(*f, args)
            }
        }
    }

    for st in &form.stmts {
        let v = expr(&mut b, &st.expr, &env, &params);
        env.insert(st.target.name().to_string(), v);
        if let Target::Out(o) = st.target {
            outs[o.index()] = Some(v);
        }
    }

    let n = b.ops.len();
    // varying: depends on the point (directly or through an operand)
    let mut varying = vec![false; n];
    for v in 0..n {
        varying[v] = match &b.ops[v] {
            Op::In(inp) => inp.varying(),
            Op::Const(_) | Op::Param(_) => false,
            op => op.operands().iter().any(|&a| varying[a as usize]),
        };
    }
    // dead code removal: mark from the outputs backwards
    let mut live = vec![false; n];
    for o in outs.iter().flatten() {
        live[*o as usize] = true;
    }
    for v in (0..n).rev() {
        if live[v] {
            for a in b.ops[v].operands() {
                live[a as usize] = true;
            }
        }
    }
    // hoisting: uniform non-constant values that per-point code (or an
    // output) reads get a slot in the per-frame table
    let mut slots = Vec::new();
    let mut slot_of = HashMap::new();
    let need = |v: V, slots: &mut Vec<V>, slot_of: &mut HashMap<V, usize>, ops: &Vec<Op>| {
        if !matches!(ops[v as usize], Op::Const(_)) && !slot_of.contains_key(&v) {
            slot_of.insert(v, slots.len());
            slots.push(v);
        }
    };
    for v in 0..n {
        if live[v] && varying[v] {
            for a in b.ops[v].operands() {
                if !varying[a as usize] {
                    need(a, &mut slots, &mut slot_of, &b.ops);
                }
            }
        }
    }
    for o in outs.iter().flatten() {
        if !varying[*o as usize] {
            need(*o, &mut slots, &mut slot_of, &b.ops);
        }
    }

    let mut stats = b.stats;
    stats.dead = (0..n).filter(|&v| !live[v] && !matches!(b.ops[v], Op::Const(_))).count();
    stats.hoisted = (0..n)
        .filter(|&v| live[v] && !varying[v] && !matches!(b.ops[v], Op::Const(_) | Op::Param(_) | Op::In(_)))
        .count();
    stats.per_point = (0..n).filter(|&v| live[v] && varying[v] && !matches!(b.ops[v], Op::In(_))).count();

    Ir {
        ops: b.ops,
        varying,
        live,
        outs,
        params: form.params.iter().map(|p| (p.name.clone(), p.value as f32)).collect(),
        slots,
        slot_of,
        attractor: form.is_attractor(),
        stats,
    }
}
