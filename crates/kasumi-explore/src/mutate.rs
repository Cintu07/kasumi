//! Mutations on the syntax tree. Every result is printed and re-parsed, so a
//! mutant is always a valid form file.

use kasumi_lang::ast::*;
use kasumi_lang::{lower, parse, print_form, Func};

use crate::rng::Rng;

const PALETTES: &[&str] = &["mono", "ember", "ice", "sakura", "moss", "ink", "dusk", "jade"];
const MAX_NODES: usize = 90;

/// Round to 4 significant digits so mutated forms stay readable.
fn tidy(v: f64) -> f64 {
    if v == 0.0 || !v.is_finite() {
        return 0.0;
    }
    let mag = 10f64.powf(3.0 - v.abs().log10().floor());
    (v * mag).round() / mag
}

/// Names a statement may read: inputs, params, earlier bindings.
fn scope_at(form: &Form, stmt: usize) -> Vec<String> {
    let attractor = form.is_attractor();
    let mut s: Vec<String> = vec!["u".into(), "t".into(), "r1".into(), "r2".into()];
    if attractor {
        s.push("px".into());
        s.push("py".into());
    }
    s.extend(form.params.iter().map(|p| p.name.clone()));
    for st in &form.stmts[..stmt] {
        let n = st.target.name().to_string();
        if !s.contains(&n) {
            s.push(n);
        }
    }
    s
}

fn wave(rng: &mut Rng, scope: &[String]) -> Expr {
    // a * sin(f * v + w * t)
    let v = rng.pick(scope).clone();
    let f = if v == "u" { tidy(rng.range(1.0, 40.0)) } else { tidy(rng.range(0.5, 8.0)) };
    let w = tidy(rng.range(-0.8, 0.8));
    let a = tidy(rng.range(0.03, 0.3) * if rng.chance(0.5) { 1.0 } else { -1.0 });
    let func = if rng.chance(0.5) { Func::Sin } else { Func::Cos };
    let arg = Expr::bin(BinOp::Add, Expr::bin(BinOp::Mul, Expr::Num(f), Expr::Var(v)), Expr::bin(BinOp::Mul, Expr::Num(w), Expr::var("t")));
    Expr::bin(BinOp::Mul, Expr::Num(a), Expr::Call(func, vec![arg]))
}

fn count_where(e: &Expr, pred: &dyn Fn(&Expr) -> bool) -> Vec<usize> {
    (0..e.size()).filter(|&k| e.node(k).is_some_and(pred)).collect()
}

fn free_vars(e: &Expr) -> Vec<String> {
    let mut v = Vec::new();
    e.visit_vars(&mut |n| {
        if !v.iter().any(|x: &String| x == n) {
            v.push(n.to_string());
        }
    });
    v
}

/// One mutation step. Returns false if nothing applicable was found.
fn mutate_once(form: &mut Form, rng: &mut Rng, donor: Option<&Form>) -> bool {
    let roll = rng.f64();
    let ns = form.stmts.len();
    if ns == 0 {
        return false;
    }
    let si = rng.below(ns);
    let scope = scope_at(form, si);
    let big = form.stmts[si].expr.size() > MAX_NODES;

    if roll < 0.18 && !form.params.is_empty() {
        // nudge a param
        let p = rng.below(form.params.len());
        let v = form.params[p].value;
        let nv = if v == 0.0 { rng.normal() * 0.3 } else { v * (rng.normal() * 0.18).exp() };
        form.params[p].value = tidy(nv);
        return true;
    }
    if roll < 0.40 {
        // nudge a literal
        let e = &mut form.stmts[si].expr;
        let idx = count_where(e, &|n| matches!(n, Expr::Num(_)));
        if idx.is_empty() {
            return false;
        }
        let k = *rng.pick(&idx);
        if let Some(Expr::Num(v)) = e.node_mut(k) {
            *v = if *v == 0.0 { tidy(rng.normal() * 0.2) } else { tidy(*v * (rng.normal() * 0.2).exp()) };
        }
        return true;
    }
    if roll < 0.52 {
        // swap a function for a sibling
        let e = &mut form.stmts[si].expr;
        let idx = count_where(e, &|n| matches!(n, Expr::Call(f, _) if !f.siblings().is_empty()));
        if idx.is_empty() {
            return false;
        }
        let k = *rng.pick(&idx);
        if let Some(Expr::Call(f, _)) = e.node_mut(k) {
            *f = *rng.pick(f.siblings());
        }
        return true;
    }
    if roll < 0.68 && !big {
        // add a wave on top of some sub-expression
        let w = wave(rng, &scope);
        let e = &mut form.stmts[si].expr;
        let k = rng.below(e.size());
        let additive = rng.chance(0.75);
        if let Some(node) = e.node_mut(k) {
            let old = std::mem::replace(node, Expr::Num(0.0));
            // e + wave, or e * (1 + wave) so a small wave modulates instead of killing e
            *node = if additive {
                Expr::bin(BinOp::Add, old, w)
            } else {
                Expr::bin(BinOp::Mul, old, Expr::bin(BinOp::Add, Expr::Num(1.0), w))
            };
        }
        return true;
    }
    if roll < 0.80 {
        // remove a term
        let e = &mut form.stmts[si].expr;
        let idx = count_where(e, &|n| matches!(n, Expr::Bin(BinOp::Add | BinOp::Sub, ..)));
        if idx.is_empty() {
            return false;
        }
        let k = *rng.pick(&idx);
        let keep_left = rng.chance(0.6);
        if let Some(node) = e.node_mut(k) {
            if let Expr::Bin(_, a, b) = std::mem::replace(node, Expr::Num(0.0)) {
                *node = if keep_left { *a } else { *b };
            }
        }
        return true;
    }
    if roll < 0.92 {
        // crossover: graft a sub-expression from the donor (or self)
        let src = donor.unwrap_or(form).clone();
        if src.stmts.is_empty() {
            return false;
        }
        for _ in 0..12 {
            let ds = rng.below(src.stmts.len());
            let de = &src.stmts[ds].expr;
            let k = rng.below(de.size());
            let Some(piece) = de.node(k) else { continue };
            if piece.size() < 3 || piece.size() > 30 {
                continue;
            }
            if !free_vars(piece).iter().all(|v| scope.contains(v) || CONSTS.iter().any(|(c, _)| c == v)) {
                continue;
            }
            let piece = piece.clone();
            let e = &mut form.stmts[si].expr;
            let tk = rng.below(e.size());
            if let Some(node) = e.node_mut(tk) {
                *node = piece;
                return true;
            }
        }
        return false;
    }
    // look: palette or colour
    if !form.has_output(Output::H) && rng.chance(0.6) {
        let all = scope_at(form, ns);
        let v = if rng.chance(0.5) { "u".to_string() } else { rng.pick(&all).clone() };
        let k = tidy(rng.range(0.5, 6.0));
        let e = Expr::Call(Func::Fract, vec![Expr::bin(BinOp::Mul, Expr::Var(v), Expr::Num(k))]);
        form.stmts.push(Stmt { target: Target::Out(Output::H), expr: e });
    }
    let p = rng.pick(PALETTES).to_string();
    form.set("palette", SetVal::Word(p));
    true
}

/// A valid mutant of `form` with 1-3 mutations, or None after many failures.
pub fn mutate(form: &Form, rng: &mut Rng, donor: Option<&Form>) -> Option<Form> {
    for _ in 0..24 {
        let mut f = form.clone();
        let steps = 1 + rng.below(3);
        let mut changed = false;
        for _ in 0..steps {
            changed |= mutate_once(&mut f, rng, donor);
        }
        if !changed || f == *form {
            continue;
        }
        // drop lets nothing reads any more
        prune(&mut f);
        let text = print_form(&f);
        if let Ok(g) = parse(&text) {
            let ir = lower(&g);
            if ir.outs[0].is_some() && ir.outs[1].is_some() && ir.stats.per_point > 0 {
                return Some(g);
            }
        }
    }
    None
}

fn prune(f: &mut Form) {
    loop {
        let mut removed = false;
        for k in (0..f.stmts.len()).rev() {
            if let Target::Let(name) = &f.stmts[k].target {
                let used = f.stmts[k + 1..].iter().any(|s| s.expr.uses(name));
                if !used {
                    f.stmts.remove(k);
                    removed = true;
                    break;
                }
            }
        }
        if !removed {
            break;
        }
    }
}

/// Stable short id for a form's text.
pub fn form_id(text: &str) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in text.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    format!("{:012x}", h & 0xffff_ffff_ffff)
}
