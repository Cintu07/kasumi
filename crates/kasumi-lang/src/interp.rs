//! The cpu reference interpreter. Every op is evaluated in f32 with the same
//! formulas the cuda prelude uses, so compiled kernels can be checked
//! point-for-point against it.

use crate::ast::BinOp;
use crate::func::Func;
use crate::ir::{Input, Ir, Op};

pub const R1_SALT: u32 = 0xA511_E9B3;
pub const R2_SALT: u32 = 0x63D8_3595;
pub const SEED_X: u32 = 0x2C1B_3C6D;
pub const SEED_Y: u32 = 0x297A_2D39;

/// lowbias32 integer hash (same constants as the cuda prelude).
#[inline]
pub fn hash32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

#[inline]
pub fn u01(x: u32) -> f32 {
    (hash32(x) >> 8) as f32 * (1.0 / 16_777_216.0)
}

/// Starting point of an attractor chain.
pub fn chain_seed(c: u32) -> (f32, f32) {
    (u01(c ^ SEED_X) * 2.0 - 1.0, u01(c ^ SEED_Y) * 2.0 - 1.0)
}

pub fn eval_bin(op: BinOp, a: f32, b: f32) -> f32 {
    match op {
        BinOp::Add => a + b,
        BinOp::Sub => a - b,
        BinOp::Mul => a * b,
        BinOp::Div => a / b,
        BinOp::Mod => a - b * (a / b).floor(),
        BinOp::Pow => a.powf(b),
    }
}

fn grad(h: u32, x: f32, y: f32) -> f32 {
    match h & 7 {
        0 => x + y,
        1 => -x + y,
        2 => x - y,
        3 => -x - y,
        4 => x,
        5 => -x,
        6 => y,
        _ => -y,
    }
}

fn corner(i: i32, j: i32) -> u32 {
    hash32((i as u32).wrapping_mul(73_856_093) ^ (j as u32).wrapping_mul(19_349_663))
}

/// 2d simplex noise with hashed gradients, roughly in [-1, 1].
pub fn noise2(x: f32, y: f32) -> f32 {
    const F2: f32 = 0.366_025_42;
    const G2: f32 = 0.211_324_87;
    let s = (x + y) * F2;
    let fi = (x + s).floor();
    let fj = (y + s).floor();
    let t = (fi + fj) * G2;
    let x0 = x - (fi - t);
    let y0 = y - (fj - t);
    let i1 = if x0 > y0 { 1.0f32 } else { 0.0 };
    let j1 = 1.0 - i1;
    let x1 = x0 - i1 + G2;
    let y1 = y0 - j1 + G2;
    let x2 = x0 - 1.0 + 2.0 * G2;
    let y2 = y0 - 1.0 + 2.0 * G2;
    let ii = fi as i32;
    let jj = fj as i32;
    let h0 = corner(ii, jj);
    let h1 = corner(ii.wrapping_add(i1 as i32), jj.wrapping_add(j1 as i32));
    let h2 = corner(ii.wrapping_add(1), jj.wrapping_add(1));
    let mut n = 0.0f32;
    let mut tt = 0.5 - x0 * x0 - y0 * y0;
    if tt > 0.0 {
        tt *= tt;
        n += tt * tt * grad(h0, x0, y0);
    }
    tt = 0.5 - x1 * x1 - y1 * y1;
    if tt > 0.0 {
        tt *= tt;
        n += tt * tt * grad(h1, x1, y1);
    }
    tt = 0.5 - x2 * x2 - y2 * y2;
    if tt > 0.0 {
        tt *= tt;
        n += tt * tt * grad(h2, x2, y2);
    }
    70.0 * n
}

pub fn eval_call(f: Func, a: &[f32]) -> f32 {
    match f {
        Func::Sin => a[0].sin(),
        Func::Cos => a[0].cos(),
        Func::Tan => a[0].tan(),
        Func::Tanh => a[0].tanh(),
        Func::Atan2 => a[0].atan2(a[1]),
        Func::Sqrt => a[0].sqrt(),
        Func::Abs => a[0].abs(),
        Func::Fract => a[0] - a[0].floor(),
        Func::Floor => a[0].floor(),
        Func::Sign => {
            if a[0] > 0.0 {
                1.0
            } else if a[0] < 0.0 {
                -1.0
            } else {
                0.0
            }
        }
        Func::Min => a[0].min(a[1]),
        Func::Max => a[0].max(a[1]),
        Func::Mix => a[0] + (a[1] - a[0]) * a[2],
        Func::Clamp => a[0].max(a[1]).min(a[2]),
        Func::Smoothstep => {
            let t = ((a[2] - a[0]) / (a[1] - a[0])).max(0.0).min(1.0);
            t * t * (3.0 - 2.0 * t)
        }
        Func::Mag => (a[0] * a[0] + a[1] * a[1]).sqrt(),
        Func::Exp => a[0].exp(),
        Func::Log => a[0].ln(),
        Func::Pow => a[0].powf(a[1]),
        Func::Noise => noise2(a[0], a[1]),
        Func::Rnd => u01(a[0].to_bits()),
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Pt {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub b: f32,
    pub h: f32,
}

/// Per-frame values: every uniform op evaluated once on the cpu.
#[derive(Clone, Debug)]
pub struct Uniforms {
    /// value of every op (varying ops are NaN)
    pub all: Vec<f32>,
    /// the values the gpu kernel reads, in slot order
    pub slots: Vec<f32>,
}

pub fn eval_uniforms(ir: &Ir, t: f32, n: f32, params: &[f32]) -> Uniforms {
    let mut all = vec![f32::NAN; ir.ops.len()];
    for (v, op) in ir.ops.iter().enumerate() {
        if ir.varying[v] || !ir.live[v] {
            continue;
        }
        all[v] = match op {
            Op::Const(b) => f32::from_bits(*b),
            Op::In(Input::N) => n,
            Op::In(Input::T) => t,
            Op::In(_) => unreachable!("varying input marked uniform"),
            Op::Param(k) => params.get(*k as usize).copied().unwrap_or(ir.params[*k as usize].1),
            Op::Neg(a) => -all[*a as usize],
            Op::Bin(o, a, b) => eval_bin(*o, all[*a as usize], all[*b as usize]),
            Op::Call(f, args) => {
                let xs: Vec<f32> = args.iter().map(|a| all[*a as usize]).collect();
                eval_call(*f, &xs)
            }
        };
    }
    let slots = ir.slots.iter().map(|&v| all[v as usize]).collect();
    Uniforms { all, slots }
}

/// Evaluate one point on the cpu. `scratch` is reused between calls.
pub fn eval_point(ir: &Ir, uni: &Uniforms, i: u32, inv_n: f32, px: f32, py: f32, scratch: &mut Vec<f32>) -> Pt {
    scratch.clear();
    scratch.extend_from_slice(&uni.all);
    let fi = i as f32;
    for (v, op) in ir.ops.iter().enumerate() {
        if !ir.varying[v] || !ir.live[v] {
            continue;
        }
        let val = match op {
            Op::In(Input::I) => fi,
            Op::In(Input::U) => fi * inv_n,
            Op::In(Input::R1) => u01(i ^ R1_SALT),
            Op::In(Input::R2) => u01(i ^ R2_SALT),
            Op::In(Input::Px) => px,
            Op::In(Input::Py) => py,
            Op::In(_) | Op::Const(_) | Op::Param(_) => unreachable!(),
            Op::Neg(a) => -scratch[*a as usize],
            Op::Bin(o, a, b) => eval_bin(*o, scratch[*a as usize], scratch[*b as usize]),
            Op::Call(f, args) => {
                let mut xs = [0.0f32; 3];
                for (k, a) in args.iter().enumerate() {
                    xs[k] = scratch[*a as usize];
                }
                eval_call(*f, &xs[..args.len()])
            }
        };
        scratch[v] = val;
    }
    let get = |o: Option<u32>, d: f32| o.map(|v| scratch[v as usize]).unwrap_or(d);
    Pt {
        x: get(ir.outs[0], 0.0),
        y: get(ir.outs[1], 0.0),
        z: get(ir.outs[2], 0.0),
        b: get(ir.outs[3], 1.0),
        h: get(ir.outs[4], 0.0),
    }
}
