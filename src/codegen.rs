//! Turns IR into CUDA. One form becomes one inlined point function plus the
//! splat, probe and optional count kernels from `kernels/form*.cu`. Several
//! forms can share a module, so the explorer compiles a batch in one NVRTC call.

use crate::args;
use crate::ast::BinOp;
use crate::ir::{Input, Ir, Op, V};

const PRELUDE: &str = include_str!("kernels/prelude.cu");
const FORM: &str = include_str!("kernels/form.cu");
const FORM_BIN: &str = include_str!("kernels/form_bin.cu");

fn literal(c: f32) -> String {
    if c.is_nan() {
        "__int_as_float(0x7fc00000)".into()
    } else if c.is_infinite() {
        format!("__int_as_float({})", if c > 0.0 { "0x7f800000" } else { "0xff800000" })
    } else if c < 0.0 {
        format!("({c:?}f)")
    } else {
        format!("{c:?}f")
    }
}

fn operand(ir: &Ir, v: V) -> String {
    if let Some(c) = ir.const_val(v) {
        return literal(c);
    }
    if !ir.varying[v as usize] {
        return format!("U[{}]", ir.slot_of[&v]);
    }
    match ir.ops[v as usize] {
        Op::In(Input::I) => "fi".into(),
        Op::In(Input::U) => "(fi * inv_n)".into(),
        Op::In(Input::R1) => "ks_u01(i ^ 0xA511E9B3u)".into(),
        Op::In(Input::R2) => "ks_u01(i ^ 0x63D83595u)".into(),
        Op::In(Input::Px) => "px".into(),
        Op::In(Input::Py) => "py".into(),
        _ => format!("v{v}"),
    }
}

fn op_expr(ir: &Ir, op: &Op) -> String {
    let o = |v: &V| operand(ir, *v);
    match op {
        Op::Neg(a) => format!("-{}", o(a)),
        Op::Bin(BinOp::Mod, a, b) => format!("ks_mod({}, {})", o(a), o(b)),
        Op::Bin(BinOp::Pow, a, b) => format!("powf({}, {})", o(a), o(b)),
        Op::Bin(op, a, b) => format!("{} {} {}", o(a), op.sym(), o(b)),
        Op::Call(f, args) => format!("{}({})", f.cuda(), args.iter().map(o).collect::<Vec<_>>().join(", ")),
        Op::Const(_) | Op::In(_) | Op::Param(_) => unreachable!(),
    }
}

/// The body of the inlined per-point function.
pub fn emit_body(ir: &Ir) -> String {
    let mut s = String::new();
    for (v, op) in ir.ops.iter().enumerate() {
        if ir.live[v] && ir.varying[v] && !matches!(op, Op::In(_)) {
            s.push_str(&format!("  const float v{v} = {};\n", op_expr(ir, op)));
        }
    }
    let outputs = [("ox", "0.f"), ("oy", "0.f"), ("oz", "0.f"), ("ob", "1.f"), ("oh", "0.f")];
    for (k, (name, default)) in outputs.iter().enumerate() {
        let value = ir.outs[k].map_or(default.to_string(), |v| operand(ir, v));
        s.push_str(&format!("  {name} = {value};\n"));
    }
    s.push_str("  (void)fi; (void)inv_n; (void)px; (void)py; (void)U;\n");
    s
}

fn emit_unit(ir: &Ir, id: usize, with_bin: bool) -> String {
    let source = if with_bin { format!("{FORM}{FORM_BIN}") } else { FORM.to_string() };
    source
        .replace("@BODY@", &emit_body(ir))
        .replace("@ID@", &id.to_string())
        .replace("@HB@", &ir.has_b().to_string())
        .replace("@HH@", &ir.has_h().to_string())
        .replace("@ATTR@", &ir.attractor.to_string())
}

/// CUDA source for a module of forms. `with_bin` adds the tile-binning kernel,
/// which the live window leaves out to compile faster.
pub fn module_source(irs: &[&Ir], with_bin: bool) -> String {
    let mut s = args::header() + PRELUDE;
    for (id, ir) in irs.iter().enumerate() {
        s.push_str(&emit_unit(ir, id, with_bin));
    }
    s
}
