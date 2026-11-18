//! `check`, `bench`, `ptx`, `info`.

use std::fmt::Write as _;
use std::path::PathBuf;

use anyhow::{Context, Result};
use kasumi_gpu::{check, CompileOpts, FormModule, FrameReq, Gpu, Mode, Statics, Target};

use crate::common::*;

pub fn info() -> Result<()> {
    let gpu = Gpu::new()?;
    let (free, total) = gpu.mem_info();
    println!("gpu      {} (sm_{}{}, {} SMs)", gpu.name, gpu.cc.0, gpu.cc.1, gpu.sm_count);
    println!("memory   {:.1} GB free of {:.1} GB", free as f64 / 1e9, total as f64 / 1e9);
    println!("nvrtc    {}.{} ({})", gpu.nvrtc_version.0, gpu.nvrtc_version.1, gpu.nvrtc_path.display());
    let st = std::time::Instant::now();
    Statics::new(&gpu)?;
    println!("statics  compiled in {:.0} ms", st.elapsed().as_secs_f64() * 1e3);
    Ok(())
}

pub fn check_cmd(args: &Args) -> Result<()> {
    let gpu = Gpu::new()?;
    let samples = args.u32_or("samples", 10_000)?;
    let mut all_ok = true;
    for p in &args.pos {
        let l = load_form(&PathBuf::from(p))?;
        let points = l.form.num_or("points", 20e6) as u64;
        // strict: no fma contraction, the codegen itself is under test
        let strict = FormModule::build(&gpu, &[&l.prog.ir], CompileOpts { fast_math: false, strict: true })?;
        let r = check(&gpu, &strict.units[0], &l.prog, points, 1.5, samples)?;
        // as rendered: fma on (and fast math if the form asks for it)
        let real = build_one(&gpu, &l, args.has("fast"))?;
        let r2 = check(&gpu, &real.units[0], &l.prog, points, 1.5, samples)?;
        let tag = |r: &kasumi_gpu::CheckReport| if r.ok() { "ok" } else { "MISMATCH" };
        println!(
            "{:<24} strict {:>6.2}% {:<8} (max rel err {:.1e})   as-rendered {:>6.2}% {:<8} (max {:.1e} at i={})",
            stem(&l.path),
            r.frac() * 100.0,
            tag(&r),
            r.max_err,
            r2.frac() * 100.0,
            tag(&r2),
            r2.max_err,
            r2.worst_i
        );
        all_ok &= r.ok();
    }
    if !all_ok {
        anyhow::bail!("some forms did not match the cpu reference");
    }
    Ok(())
}

pub fn ptx(args: &Args) -> Result<()> {
    let path = PathBuf::from(args.pos.first().context("usage: kasumi ptx <form.ks>")?);
    let l = load_form(&path)?;
    let gpu = Gpu::new()?;
    let src = kasumi_lang::codegen::module_source(&[&l.prog.ir], true);
    let c = gpu.compile_with(&src, opts_for(&l.form, args.has("fast")), true)?;
    let cu = path.with_extension("cu");
    let ptx = path.with_extension("ptx");
    std::fs::write(&cu, &src)?;
    std::fs::write(&ptx, &c.image)?;
    let s = &l.prog.ir.stats;
    println!(
        "ir: {} ast nodes -> {} per-point ops | {} folded, {} cse hits, {} dead, {} hoisted to once-per-frame, {} uniform slots",
        s.ast_nodes,
        s.per_point,
        s.folded,
        s.cse_hits,
        s.dead,
        s.hoisted,
        l.prog.ir.slots.len()
    );
    println!("wrote {} and {} (nvrtc {:.0} ms)", cu.display(), ptx.display(), c.ms);
    println!("\n--- fused point function ---\n{}", kasumi_lang::codegen::emit_body(&l.prog.ir));
    Ok(())
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

pub fn bench(args: &Args) -> Result<()> {
    let gpu = Gpu::new()?;
    let st = Statics::new(&gpu)?;
    let mut forms: Vec<PathBuf> = args.pos.iter().map(PathBuf::from).collect();
    if forms.is_empty() {
        let mut v: Vec<PathBuf> = std::fs::read_dir("forms")?.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|e| e == "ks")).collect();
        v.sort();
        forms = v;
    }
    let (w, h) = parse_res(args.get("res").unwrap_or("4k"))?;
    let ss = args.u32_or("ss", 1)?;
    let points = args.count_or("points", 100_000_000)?;
    let reps = args.u32_or("reps", 5)?;
    let mut tgt = Target::new(&gpu, w, h, ss, 0)?;
    let mut md = String::new();
    writeln!(md, "# kasumi bench\n\n{} | {}x{} accumulation | {} points per frame | median of {reps}\n", gpu.name, w * ss, h * ss, human(points))?;
    writeln!(md, "| form | lit px | max density | naive Gpts/s | warp Gpts/s | bin Gpts/s | bin vs naive | bin == naive |")?;
    writeln!(md, "|---|---|---|---|---|---|---|---|")?;
    println!("{} | accumulation {}x{} | {} points | median of {reps}", gpu.name, w * ss, h * ss, human(points));
    println!("{:<18} {:>8} {:>10} {:>9} {:>9} {:>9} {:>8}  {}", "form", "lit", "max dens", "naive", "warp", "bin", "bin/nv", "match");
    for path in &forms {
        let l = load_form(path)?;
        let fm = build_one(&gpu, &l, false)?;
        let unit = fm.units[0];
        let mut res = Vec::new();
        let mut dens: Vec<Vec<f32>> = Vec::new();
        for mode in Mode::ALL {
            let req = FrameReq { t: 1.5, dt: 1.0 / 60.0, subframes: 1, points, mode, decay: 0.0 };
            tgt.accumulate(&gpu, &st, &unit, &l.prog, &l.look, &req)?; // warm up
            let mut times = Vec::new();
            let mut pts = 0;
            for _ in 0..reps {
                let s = tgt.accumulate(&gpu, &st, &unit, &l.prog, &l.look, &req)?;
                times.push(s.ms);
                pts = s.points;
            }
            res.push(gpts(pts, median(times)));
            dens.push(tgt.read_density(&gpu)?);
        }
        let lit = dens[0].iter().filter(|&&c| c > 0.0).count();
        let maxd = dens[0].iter().cloned().fold(0.0f32, f32::max);
        if !l.prog.ir.has_b() {
            let hits: f64 = dens[0].iter().map(|&c| c as f64).sum();
            println!("  {}: {:.1}% of points land on screen", stem(path), hits / points as f64 * 100.0);
        }
        let maxdiff = dens[0].iter().zip(&dens[2]).map(|(a, b)| (a - b).abs() / (1.0 + a.abs())).fold(0.0f32, f32::max);
        // unweighted counts are integers and must match exactly; weighted (b)
        // sums are added in a different order, so allow float rounding
        let tol = if l.prog.ir.has_b() { 1e-3 } else { 1e-6 };
        let same = if maxdiff <= tol {
            if maxdiff > 0.0 { format!("yes ({maxdiff:.0e}, float order)") } else { "yes".to_string() }
        } else {
            format!("NO ({maxdiff:.1e})")
        };
        let name = stem(path);
        println!(
            "{:<18} {:>7.1}% {:>10.0} {:>9.2} {:>9.2} {:>9.2} {:>7.2}x  {}",
            name,
            lit as f64 / dens[0].len() as f64 * 100.0,
            maxd,
            res[0],
            res[1],
            res[2],
            res[2] / res[0],
            same
        );
        writeln!(md, "| {name} | {:.1}% | {maxd:.0} | {:.2} | {:.2} | {:.2} | {:.2}x | {same} |", lit as f64 / dens[0].len() as f64 * 100.0, res[0], res[1], res[2], res[2] / res[0])?;
    }
    std::fs::create_dir_all("renders").ok();
    std::fs::write("renders/bench.md", &md)?;
    println!("\nwrote renders/bench.md");
    Ok(())
}
