//! `check`, `bench`, `ptx` and `info`.

use std::path::{Path, PathBuf};

use anyhow::{bail, Result};

use crate::cli::*;
use crate::codegen::{emit_body, module_source};
use crate::cuda::Gpu;
use crate::render::{check, compile, compile_one, Build, CheckReport, Engine, Format, Frame, Mode, Program, Target};

pub fn info() -> Result<()> {
    let gpu = Gpu::new()?;
    let (free, total) = gpu.memory();
    println!("gpu      {} ({} SMs)", gpu.name, gpu.sm_count);
    println!("memory   {:.1} GB free of {:.1} GB", free as f64 / 1e9, total as f64 / 1e9);
    println!("nvrtc    {}", gpu.nvrtc_path.display());
    Ok(())
}

/// Compare GPU output with the CPU interpreter, with fused multiply-add off
/// (testing the code generator) and on (as the form is really drawn).
pub fn check_forms(args: &Args) -> Result<()> {
    let gpu = Gpu::new()?;
    let samples = args.u32_or("samples", 10_000)?;
    let verdict = |r: &CheckReport| if r.ok() { "ok" } else { "MISMATCH" };
    let mut all_ok = true;
    for path in &args.pos {
        let p = Program::load(Path::new(path))?;
        let strict = compile(&gpu, &[&p], Build { strict: true, ..Build::default() })?.remove(0);
        let a = check(&gpu, &strict, &p, p.points, 1.5, samples)?;
        let drawn = compile_one(&gpu, &p, false)?;
        let b = check(&gpu, &drawn, &p, p.points, 1.5, samples)?;
        println!(
            "{:<24} strict {:>6.2}% {:<8} (max rel err {:.1e})   as-rendered {:>6.2}% {:<8} (max {:.1e} at i={})",
            stem(Path::new(path)), a.fraction() * 100.0, verdict(&a), a.max_err, b.fraction() * 100.0, verdict(&b), b.max_err, b.worst_i
        );
        all_ok &= a.ok();
    }
    if !all_ok {
        bail!("some forms did not match the cpu reference");
    }
    Ok(())
}

/// Write the generated CUDA and PTX next to a form.
pub fn ptx(args: &Args) -> Result<()> {
    let path = args.path(0, "kasumi ptx <form.ks>")?;
    let p = Program::load(&path)?;
    let source = module_source(&[&p.ir], true);
    let ptx = Gpu::new()?.compile(&source, p.form.flag("fastmath"), false)?;
    let (cu_path, ptx_path) = (path.with_extension("cu"), path.with_extension("ptx"));
    std::fs::write(&cu_path, &source)?;
    std::fs::write(&ptx_path, &ptx)?;
    let s = &p.ir.stats;
    println!(
        "ir: {} ast nodes -> {} per-point ops | {} folded, {} cse hits, {} dead, {} hoisted, {} uniform slots",
        s.ast_nodes, s.per_point, s.folded, s.cse_hits, s.dead, s.hoisted, p.ir.slots.len()
    );
    println!("wrote {} and {}\n\n--- point function ---\n{}", cu_path.display(), ptx_path.display(), emit_body(&p.ir));
    Ok(())
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// Time the three accumulation kernels and check that they agree.
pub fn bench(args: &Args) -> Result<()> {
    let engine = Engine::new()?;
    let forms: Vec<PathBuf> = if args.pos.is_empty() { forms_in(Path::new("forms"))? } else { args.pos.iter().map(PathBuf::from).collect() };
    let (w, h) = parse_res(args.get("res").unwrap_or("4k"))?;
    let ss = args.u32_or("ss", 1)?;
    let points = args.count_or("points", 100_000_000)?;
    let reps = args.u32_or("reps", 5)?;
    let mut target = Target::new(&engine.gpu, w, h, ss, Format::Window)?;
    println!("{} | accumulation {}x{} | {} points | median of {reps}", engine.gpu.name, w * ss, h * ss, human(points));
    println!("{:<18} {:>8} {:>9} {:>9} {:>9} {:>8}  match", "form", "lit", "naive", "warp", "bin", "bin/nv");
    for path in &forms {
        let p = Program::load(path)?;
        let kernels = compile_one(&engine.gpu, &p, true)?;
        let mut rates = Vec::new();
        let mut densities = Vec::new();
        for mode in Mode::ALL {
            let frame = Frame { t: 1.5, ..Frame::still(1.5, points, mode) };
            target.accumulate(&engine, &kernels, &p, &p.look, &frame)?; // warm up
            let mut stats = Vec::new();
            for _ in 0..reps {
                stats.push(target.accumulate(&engine, &kernels, &p, &p.look, &frame)?);
            }
            rates.push(gpts(stats[0].points, median(stats.iter().map(|s| s.ms).collect())));
            densities.push(target.read_density(&engine.gpu)?);
        }
        let (naive, binned) = (&densities[0], &densities[2]);
        let lit = naive.iter().filter(|&&c| c > 0.0).count() as f64 / naive.len() as f64;
        let diff = naive.iter().zip(binned).map(|(a, b)| (a - b).abs() / (1.0 + a.abs())).fold(0.0f32, f32::max);
        // counts are integers and must match exactly; weighted sums add in a different order
        let tolerance = if p.ir.has_b() { 1e-3 } else { 1e-6 };
        let same = if diff <= tolerance { "yes".to_string() } else { format!("NO ({diff:.1e})") };
        println!(
            "{:<18} {:>7.1}% {:>9.2} {:>9.2} {:>9.2} {:>7.2}x  {same}",
            stem(path), lit * 100.0, rates[0], rates[1], rates[2], rates[2] / rates[0]
        );
    }
    Ok(())
}
