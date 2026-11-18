use kasumi::codegen::module_source;
use kasumi::interp::{eval_point, eval_uniforms, noise2};
use kasumi::ir::lower;
use kasumi::mutate::mutate;
use kasumi::parser::parse;
use kasumi::print::{print_expr, print_form};
use kasumi::rng::Rng;

const SILK: &str = "# silk
@points 20m
@palette ember
param k = 3
let a = u * tau
let w = sin(a * k + t * 0.5)
x = cos(a) * (0.5 + 0.2 * w)
y = sin(a) * (0.5 + 0.2 * w)
h = fract(u * 4)
";

#[test]
fn parses_and_roundtrips() {
    let f = parse(SILK).unwrap();
    assert_eq!(f.header, vec!["silk"]);
    assert_eq!(f.num("points"), Some(20e6));
    assert_eq!(f.word("palette"), Some("ember"));
    let printed = print_form(&f);
    let g = parse(&printed).unwrap();
    assert_eq!(f, g, "printed:\n{printed}");
}

#[test]
fn precedence_and_printing() {
    let cases = [
        ("x = -a^2\ny = 0", "-a^2"),
        ("x = (-a)^2\ny = 0", "(-a)^2"),
        ("x = a - (q - r)\ny = 0", "a - (q - r)"),
        ("x = a - q - r\ny = 0", "a - q - r"),
        ("x = a^q^r\ny = 0", "a^q^r"),
        ("x = (a^q)^r\ny = 0", "(a^q)^r"),
        ("x = a * -0.5\ny = 0", "a * -0.5"),
        ("x = 2 * (a + q) % 3\ny = 0", "2 * (a + q) % 3"),
    ];
    for (src, want) in cases {
        let src = format!("param a = 1\nparam q = 2\nparam r = 3\n{src}");
        let f = parse(&src).unwrap();
        let got = print_expr(&f.stmts[0].expr);
        assert_eq!(got, want);
        let again = parse(&print_form(&f)).unwrap();
        assert_eq!(f, again);
    }
}

#[test]
fn good_errors() {
    let e = parse("x = sinn(u)\ny = 0").unwrap_err();
    assert_eq!((e.line, e.col), (1, 5));
    assert!(e.msg.contains("did you mean 'sin'"), "{}", e.msg);

    let e = parse("let a = u\nx = a + bb\ny = 0").unwrap_err();
    assert_eq!(e.line, 2);

    let e = parse("x = u\n").unwrap_err();
    assert!(e.msg.contains("y"));

    let e = parse("x = 2u\ny = 0").unwrap_err();
    assert!(e.msg.contains("after a number"));

    let e = parse("@pionts 2m\nx = u\ny = u").unwrap_err();
    assert!(e.msg.contains("@points"));

    let e = parse("x = sin(u, t)\ny = 0").unwrap_err();
    assert!(e.msg.contains("takes 1 argument"));
}

#[test]
fn passes_fold_cse_hoist() {
    let src = "param k = 2
let a = sin(t * 0.5) * k        # uniform: hoisted
let bb = (1 + 2) * u             # folds to 3 * u
let c = cos(bb) + cos(bb)          # cse
let dead = sin(u * 99)           # removed
x = c * a
y = bb ^ 2
";
    let ir = lower(&parse(src).unwrap());
    assert!(ir.stats.folded >= 2, "{:?}", ir.stats);
    assert!(ir.stats.cse_hits >= 1, "{:?}", ir.stats);
    assert!(ir.stats.dead >= 1, "{:?}", ir.stats);
    assert!(ir.stats.hoisted >= 2, "{:?}", ir.stats);
    // one uniform value (a) is read per point
    assert_eq!(ir.slots.len(), 1);
    let cu = module_source(&[&ir], true);
    assert!(cu.contains("ks_splat_0"));
    assert!(!cu.contains("99"), "dead code leaked into the kernel");
}

#[test]
fn interpreter_matches_hand_math() {
    let ir = lower(&parse(SILK).unwrap());
    let n = 1000.0f32;
    let t = 1.25f32;
    let uni = eval_uniforms(&ir, t, n, &ir.param_values());
    let mut scratch = Vec::new();
    let i = 137u32;
    let p = eval_point(&ir, &uni, i, 1.0 / n, 0.0, 0.0, &mut scratch);
    let u = i as f32 * (1.0 / n);
    let a = u * std::f32::consts::TAU;
    let w = (a * 3.0 + t * 0.5).sin();
    assert!((p.x - a.cos() * (0.5 + 0.2 * w)).abs() < 1e-5);
    assert!((p.y - a.sin() * (0.5 + 0.2 * w)).abs() < 1e-5);
    assert!((p.h - (u * 4.0).fract()).abs() < 1e-5);
    assert_eq!(p.b, 1.0);
}

#[test]
fn attractor_detection() {
    let f = parse("param a = -1.7\nx = sin(a * py) + cos(a * px)\ny = sin(px)").unwrap();
    assert!(f.is_attractor());
    let e = parse("x = sin(y)\ny = 0").unwrap_err();
    assert!(e.msg.contains("unknown name 'y'"));
}

#[test]
fn noise_is_bounded() {
    let mut lo = 0.0f32;
    let mut hi = 0.0f32;
    for k in 0..20000 {
        let x = (k as f32) * 0.0137 - 50.0;
        let y = (k as f32) * 0.0071 + 3.0;
        let v = noise2(x, y);
        lo = lo.min(v);
        hi = hi.max(v);
    }
    assert!(lo > -1.2 && hi < 1.2 && hi - lo > 0.8, "{lo} {hi}");
}

#[test]
fn mutants_are_valid_and_different() {
    let form = parse(SILK).unwrap();
    let mut rng = Rng::new(7);
    let mut distinct = std::collections::HashSet::new();
    for _ in 0..300 {
        let text = print_form(&mutate(&form, &mut rng, None).expect("mutation failed"));
        assert!(parse(&text).is_ok(), "{text}");
        distinct.insert(text);
    }
    assert!(distinct.len() > 250, "only {} distinct mutants", distinct.len());
}
