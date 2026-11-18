//! The CPU reference interpreter. It evaluates every op in f32 with the same
//! formulas as `kernels/prelude.cu`, so compiled kernels can be checked point
//! for point against it.

use crate::ast::{BinOp, Func};
use crate::ir::{Input, Ir, Op};

const R1_SALT: u32 = 0xA511_E9B3;
const R2_SALT: u32 = 0x63D8_3595;
const SEED_X: u32 = 0x2C1B_3C6D;
const SEED_Y: u32 = 0x297A_2D39;

fn hash32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^ (x >> 16)
}

fn u01(x: u32) -> f32 {
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

fn gradient(h: u32, x: f32, y: f32) -> f32 {
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

/// 2D simplex noise with hashed gradients, roughly in [-1, 1].
pub fn noise2(x: f32, y: f32) -> f32 {
    const F2: f32 = 0.366_025_42;
    const G2: f32 = 0.211_324_87;
    let s = (x + y) * F2;
    let (fi, fj) = ((x + s).floor(), (y + s).floor());
    let t = (fi + fj) * G2;
    let (x0, y0) = (x - (fi - t), y - (fj - t));
    let i1 = if x0 > y0 { 1.0f32 } else { 0.0 };
    let j1 = 1.0 - i1;
    let (ii, jj) = (fi as i32, fj as i32);
    let corners = [
        (x0, y0, corner(ii, jj)),
        (x0 - i1 + G2, y0 - j1 + G2, corner(ii.wrapping_add(i1 as i32), jj.wrapping_add(j1 as i32))),
        (x0 - 1.0 + 2.0 * G2, y0 - 1.0 + 2.0 * G2, corner(ii.wrapping_add(1), jj.wrapping_add(1))),
    ];
    let mut n = 0.0;
    for (cx, cy, h) in corners {
        let falloff = 0.5 - cx * cx - cy * cy;
        if falloff > 0.0 {
            let squared = falloff * falloff;
            n += squared * squared * gradient(h, cx, cy);
        }
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
        Func::Sign if a[0] > 0.0 => 1.0,
        Func::Sign if a[0] < 0.0 => -1.0,
        Func::Sign => 0.0,
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
pub struct Point {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub b: f32,
    pub h: f32,
}

/// Every per-frame op, evaluated once.
#[derive(Clone, Debug)]
pub struct Uniforms {
    /// The value of every op. Per-point ops are NaN.
    pub all: Vec<f32>,
    /// The values the kernel reads, in slot order.
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
            Op::In(_) => unreachable!("per-point input marked per-frame"),
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

/// Evaluate one point. `values` is scratch space reused between calls.
pub fn eval_point(ir: &Ir, uni: &Uniforms, i: u32, inv_n: f32, px: f32, py: f32, values: &mut Vec<f32>) -> Point {
    values.clear();
    values.extend_from_slice(&uni.all);
    let fi = i as f32;
    for (v, op) in ir.ops.iter().enumerate() {
        if !ir.varying[v] || !ir.live[v] {
            continue;
        }
        values[v] = match op {
            Op::In(Input::I) => fi,
            Op::In(Input::U) => fi * inv_n,
            Op::In(Input::R1) => u01(i ^ R1_SALT),
            Op::In(Input::R2) => u01(i ^ R2_SALT),
            Op::In(Input::Px) => px,
            Op::In(Input::Py) => py,
            Op::In(_) | Op::Const(_) | Op::Param(_) => unreachable!(),
            Op::Neg(a) => -values[*a as usize],
            Op::Bin(o, a, b) => eval_bin(*o, values[*a as usize], values[*b as usize]),
            Op::Call(f, args) => {
                let mut xs = [0.0f32; 3];
                for (k, a) in args.iter().enumerate() {
                    xs[k] = values[*a as usize];
                }
                eval_call(*f, &xs[..args.len()])
            }
        };
    }
    let out = |k: usize, default: f32| ir.outs[k].map_or(default, |v| values[v as usize]);
    Point { x: out(0, 0.0), y: out(1, 0.0), z: out(2, 0.0), b: out(3, 1.0), h: out(4, 0.0) }
}
