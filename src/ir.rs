//! A small SSA IR. Values are hash-consed as they are built, so common
//! subexpressions merge on the way in, and constants fold at construction.
//! After lowering, dead code is removed and anything that does not depend on
//! the point is hoisted: the CPU computes it once per frame.

use std::collections::HashMap;

use crate::ast::*;
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
    fn from_name(s: &str) -> Option<Input> {
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
    fn varying(self) -> bool {
        !matches!(self, Input::N | Input::T)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Op {
    /// The bits of an f32, so ops can be hashed.
    Const(u32),
    In(Input),
    Param(u32),
    Neg(V),
    Bin(BinOp, V, V),
    Call(Func, Vec<V>),
}

impl Op {
    fn operands(&self) -> Vec<V> {
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
    /// Live ops computed once per frame.
    pub hoisted: usize,
    /// Live ops computed for every point.
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
    /// Per-frame values the kernel reads, in table order.
    pub slots: Vec<V>,
    pub slot_of: HashMap<V, usize>,
    pub attractor: bool,
    pub stats: IrStats,
}

impl Ir {
    pub fn has_b(&self) -> bool {
        self.outs[Output::B as usize].is_some()
    }

    pub fn has_h(&self) -> bool {
        self.outs[Output::H as usize].is_some()
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

#[derive(Default)]
struct Builder {
    ops: Vec<Op>,
    seen: HashMap<Op, V>,
    stats: IrStats,
}

impl Builder {
    fn intern(&mut self, op: Op) -> V {
        if let Some(&v) = self.seen.get(&op) {
            self.stats.cse_hits += !matches!(op, Op::Const(_) | Op::In(_) | Op::Param(_)) as usize;
            return v;
        }
        let v = self.ops.len() as V;
        self.ops.push(op.clone());
        self.seen.insert(op, v);
        v
    }

    fn constant(&mut self, c: f32) -> V {
        // one canonical NaN, so all NaNs share an op
        self.intern(Op::Const(if c.is_nan() { f32::NAN.to_bits() } else { c.to_bits() }))
    }

    fn const_of(&self, v: V) -> Option<f32> {
        match self.ops[v as usize] {
            Op::Const(b) => Some(f32::from_bits(b)),
            _ => None,
        }
    }

    fn folded(&mut self, c: f32) -> V {
        self.stats.folded += 1;
        self.constant(c)
    }

    fn simplified(&mut self, v: V) -> V {
        self.stats.folded += 1;
        v
    }

    fn neg(&mut self, a: V) -> V {
        if let Some(c) = self.const_of(a) {
            return self.folded(-c);
        }
        if let Op::Neg(x) = self.ops[a as usize] {
            return self.simplified(x);
        }
        self.intern(Op::Neg(a))
    }

    fn bin(&mut self, op: BinOp, a: V, b: V) -> V {
        use BinOp::*;
        let (ca, cb) = (self.const_of(a), self.const_of(b));
        if let (Some(x), Some(y)) = (ca, cb) {
            return self.folded(eval_bin(op, x, y));
        }
        // dividing by a power of two is exactly multiplying by its inverse
        let exact_inverse = |c: f32| c.is_normal() && c.to_bits() & 0x007f_ffff == 0 && (1.0 / c).is_normal();
        match (op, ca, cb) {
            (Add | Sub, _, Some(0.0)) | (Mul | Div | Pow, _, Some(1.0)) => return self.simplified(a),
            (Add, Some(0.0), _) | (Mul, Some(1.0), _) => return self.simplified(b),
            (Sub, Some(0.0), _) | (Mul, Some(-1.0), _) => {
                self.stats.folded += 1;
                return self.neg(b);
            }
            (Mul, _, Some(-1.0)) => {
                self.stats.folded += 1;
                return self.neg(a);
            }
            (Div, _, Some(c)) if exact_inverse(c) => {
                self.stats.folded += 1;
                let inverse = self.constant(1.0 / c);
                return self.bin(Mul, a, inverse);
            }
            (Pow, _, Some(2.0)) => {
                self.stats.folded += 1;
                return self.bin(Mul, a, a);
            }
            (Pow, _, Some(3.0)) => {
                self.stats.folded += 1;
                let square = self.bin(Mul, a, a);
                return self.bin(Mul, square, a);
            }
            (Pow, _, Some(0.5)) => {
                self.stats.folded += 1;
                return self.call(Func::Sqrt, vec![a]);
            }
            _ => {}
        }
        let (a, b) = if matches!(op, Add | Mul) && a > b { (b, a) } else { (a, b) };
        self.intern(Op::Bin(op, a, b))
    }

    fn call(&mut self, f: Func, mut args: Vec<V>) -> V {
        let constants: Option<Vec<f32>> = args.iter().map(|&a| self.const_of(a)).collect();
        if let Some(c) = constants {
            return self.folded(eval_call(f, &c));
        }
        if matches!(f, Func::Min | Func::Max | Func::Mag) && args[0] > args[1] {
            args.swap(0, 1);
        }
        self.intern(Op::Call(f, args))
    }

    fn var(&mut self, name: &str, env: &HashMap<String, V>, params: &HashMap<&str, u32>) -> V {
        if let Some(&v) = env.get(name) {
            return v;
        }
        if let Some(input) = Input::from_name(name) {
            return self.intern(Op::In(input));
        }
        if let Some(&k) = params.get(name) {
            return self.intern(Op::Param(k));
        }
        let value = CONSTS.iter().find(|(c, _)| *c == name).map_or(f32::NAN, |(_, c)| *c as f32);
        self.constant(value)
    }

    fn expr(&mut self, e: &Expr, env: &HashMap<String, V>, params: &HashMap<&str, u32>) -> V {
        self.stats.ast_nodes += 1;
        match e {
            Expr::Num(v) => self.constant(*v as f32),
            Expr::Var(name) => self.var(name, env, params),
            Expr::Neg(a) => {
                let a = self.expr(a, env, params);
                self.neg(a)
            }
            Expr::Bin(op, a, b) => {
                let a = self.expr(a, env, params);
                let b = self.expr(b, env, params);
                self.bin(*op, a, b)
            }
            Expr::Call(f, args) => {
                let args = args.iter().map(|a| self.expr(a, env, params)).collect();
                self.call(*f, args)
            }
        }
    }
}

/// Lower a parsed form to SSA and run every pass.
pub fn lower(form: &Form) -> Ir {
    let mut b = Builder::default();
    let params: HashMap<&str, u32> = form.params.iter().enumerate().map(|(k, p)| (p.name.as_str(), k as u32)).collect();
    let mut env = HashMap::new();
    let mut outs = [None; 5];
    for st in &form.stmts {
        let v = b.expr(&st.expr, &env, &params);
        env.insert(st.target.name().to_string(), v);
        if let Target::Out(o) = st.target {
            outs[o as usize] = Some(v);
        }
    }
    let ops = b.ops;
    let n = ops.len();

    // varying: depends on the point, directly or through an operand
    let mut varying = vec![false; n];
    for v in 0..n {
        varying[v] = match &ops[v] {
            Op::In(input) => input.varying(),
            op => op.operands().iter().any(|&a| varying[a as usize]),
        };
    }

    // live: reachable backwards from the outputs
    let mut live = vec![false; n];
    outs.iter().flatten().for_each(|&o| live[o as usize] = true);
    for v in (0..n).rev() {
        if live[v] {
            ops[v].operands().iter().for_each(|&a| live[a as usize] = true);
        }
    }

    // slots: per-frame values read by per-point code or by an output
    let read_per_point = (0..n).filter(|&v| live[v] && varying[v]).flat_map(|v| ops[v].operands());
    let read_by_outputs = outs.iter().flatten().copied();
    let mut slots = Vec::new();
    let mut slot_of = HashMap::new();
    for v in read_per_point.chain(read_by_outputs) {
        if !varying[v as usize] && !matches!(ops[v as usize], Op::Const(_)) && !slot_of.contains_key(&v) {
            slot_of.insert(v, slots.len());
            slots.push(v);
        }
    }

    let count = |keep: &dyn Fn(usize) -> bool| (0..n).filter(|&v| keep(v)).count();
    let stats = IrStats {
        dead: count(&|v| !live[v] && !matches!(ops[v], Op::Const(_))),
        hoisted: count(&|v| live[v] && !varying[v] && !matches!(ops[v], Op::Const(_) | Op::Param(_) | Op::In(_))),
        per_point: count(&|v| live[v] && varying[v] && !matches!(ops[v], Op::In(_))),
        ..b.stats
    };
    let params = form.params.iter().map(|p| (p.name.clone(), p.value as f32)).collect();
    Ir { ops, varying, live, outs, params, slots, slot_of, attractor: form.is_attractor(), stats }
}
