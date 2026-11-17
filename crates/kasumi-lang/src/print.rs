//! Pretty printer. `parse(print_form(f)) == f` for any form the parser
//! produced, which is what lets the explorer mutate syntax trees and write
//! them back out as ordinary form files.

use crate::ast::*;

pub fn fmt_num(v: f64) -> String {
    if v == v.trunc() && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        // shortest round-trip representation
        let s = format!("{v}");
        if s.len() > 14 {
            format!("{v:e}")
        } else {
            s
        }
    }
}

fn prec(e: &Expr) -> u8 {
    match e {
        Expr::Num(v) if *v < 0.0 || (*v == 0.0 && v.is_sign_negative()) => 30,
        Expr::Num(_) | Expr::Var(_) | Expr::Call(..) => 100,
        Expr::Neg(_) => 30,
        Expr::Bin(op, ..) => op.prec(),
    }
}

pub fn print_expr(e: &Expr) -> String {
    let mut s = String::new();
    go(e, &mut s);
    s
}

fn wrap(e: &Expr, paren: bool, out: &mut String) {
    if paren {
        out.push('(');
        go(e, out);
        out.push(')');
    } else {
        go(e, out);
    }
}

fn go(e: &Expr, out: &mut String) {
    match e {
        Expr::Num(v) => out.push_str(&fmt_num(*v)),
        Expr::Var(v) => out.push_str(v),
        Expr::Neg(a) => {
            out.push('-');
            wrap(a, prec(a) <= 30, out);
        }
        Expr::Bin(op, a, b) => {
            let p = op.prec();
            if *op == BinOp::Pow {
                wrap(a, prec(a) <= p, out);
                out.push('^');
                wrap(b, prec(b) < p, out);
            } else {
                wrap(a, prec(a) < p, out);
                out.push(' ');
                out.push_str(op.sym());
                out.push(' ');
                wrap(b, prec(b) <= p, out);
            }
        }
        Expr::Call(f, args) => {
            out.push_str(f.name());
            out.push('(');
            for (k, a) in args.iter().enumerate() {
                if k > 0 {
                    out.push_str(", ");
                }
                go(a, out);
            }
            out.push(')');
        }
    }
}

pub fn print_form(f: &Form) -> String {
    let mut s = String::new();
    for h in &f.header {
        if h.is_empty() {
            s.push_str("#\n");
        } else {
            s.push_str(&format!("# {h}\n"));
        }
    }
    for (k, v) in &f.settings {
        match v {
            SetVal::Num(n) => {
                let n = *n;
                let txt = if n >= 1e6 && (n / 1e6).fract() == 0.0 {
                    format!("{}m", n / 1e6)
                } else if n >= 1e3 && (n / 1e3).fract() == 0.0 {
                    format!("{}k", n / 1e3)
                } else {
                    fmt_num(n)
                };
                s.push_str(&format!("@{k} {txt}\n"));
            }
            SetVal::Word(w) => s.push_str(&format!("@{k} {w}\n")),
        }
    }
    for p in &f.params {
        s.push_str(&format!("param {} = {}\n", p.name, fmt_num(p.value)));
    }
    for st in &f.stmts {
        match &st.target {
            Target::Let(n) => s.push_str(&format!("let {} = {}\n", n, print_expr(&st.expr))),
            Target::Out(o) => s.push_str(&format!("{} = {}\n", o.name(), print_expr(&st.expr))),
        }
    }
    s
}
