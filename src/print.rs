//! Prints a form back to text. `parse(print_form(f)) == f`, which lets the
//! explorer mutate syntax trees and save them as ordinary form files.

use crate::ast::*;

pub fn fmt_num(v: f64) -> String {
    if v == v.trunc() && v.abs() < 1e15 {
        return format!("{}", v as i64);
    }
    let s = format!("{v}");
    if s.len() > 14 {
        format!("{v:e}")
    } else {
        s
    }
}

fn prec(e: &Expr) -> u8 {
    match e {
        Expr::Num(v) if v.is_sign_negative() => 30,
        Expr::Num(_) | Expr::Var(_) | Expr::Call(..) => 100,
        Expr::Neg(_) => 30,
        Expr::Bin(op, ..) => op.prec(),
    }
}

fn wrap(e: &Expr, paren: bool) -> String {
    if paren {
        format!("({})", print_expr(e))
    } else {
        print_expr(e)
    }
}

pub fn print_expr(e: &Expr) -> String {
    match e {
        Expr::Num(v) => fmt_num(*v),
        Expr::Var(v) => v.clone(),
        Expr::Neg(a) => format!("-{}", wrap(a, prec(a) <= 30)),
        Expr::Bin(BinOp::Pow, a, b) => {
            let p = BinOp::Pow.prec();
            format!("{}^{}", wrap(a, prec(a) <= p), wrap(b, prec(b) < p))
        }
        Expr::Bin(op, a, b) => {
            let p = op.prec();
            format!("{} {} {}", wrap(a, prec(a) < p), op.sym(), wrap(b, prec(b) <= p))
        }
        Expr::Call(f, args) => {
            let args: Vec<String> = args.iter().map(print_expr).collect();
            format!("{}({})", f.name(), args.join(", "))
        }
    }
}

fn print_setting(v: &SetVal) -> String {
    match v {
        SetVal::Word(w) => w.clone(),
        SetVal::Num(n) if *n >= 1e6 && (n / 1e6).fract() == 0.0 => format!("{}m", n / 1e6),
        SetVal::Num(n) if *n >= 1e3 && (n / 1e3).fract() == 0.0 => format!("{}k", n / 1e3),
        SetVal::Num(n) => fmt_num(*n),
    }
}

pub fn print_form(f: &Form) -> String {
    let mut s = String::new();
    for h in &f.header {
        s.push_str(&if h.is_empty() { "#\n".to_string() } else { format!("# {h}\n") });
    }
    for (k, v) in &f.settings {
        s.push_str(&format!("@{k} {}\n", print_setting(v)));
    }
    for p in &f.params {
        s.push_str(&format!("param {} = {}\n", p.name, fmt_num(p.value)));
    }
    for st in &f.stmts {
        let keyword = if matches!(st.target, Target::Let(_)) { "let " } else { "" };
        s.push_str(&format!("{keyword}{} = {}\n", st.target.name(), print_expr(&st.expr)));
    }
    s
}
