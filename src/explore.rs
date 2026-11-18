//! Shared by `grid` and `evolve`: compile batches of forms and measure them.

use std::path::{Path, PathBuf};

use anyhow::Result;

use crate::cli::forms_in;
use crate::features::{chroma, features, luma, Feat};
use crate::render::{compile, Build, Engine, Frame, Kernels, Mode, Program, Target};

pub struct Compiled {
    pub program: Program,
    pub kernels: Kernels,
}

/// Compile a batch with one NVRTC call. If that fails, compile one by one and
/// drop the forms that do not build.
pub fn compile_batch(e: &Engine, programs: Vec<Program>) -> Vec<Compiled> {
    let refs: Vec<&Program> = programs.iter().collect();
    if let Ok(kernels) = compile(&e.gpu, &refs, Build::default()) {
        return programs.into_iter().zip(kernels).map(|(program, kernels)| Compiled { program, kernels }).collect();
    }
    let build_alone = |program: Program| {
        let kernels = compile(&e.gpu, &[&program], Build::default()).ok()?.remove(0);
        Some(Compiled { program, kernels })
    };
    programs.into_iter().filter_map(build_alone).collect()
}

pub fn draw(e: &Engine, target: &mut Target, c: &Compiled, points: u64, t: f32, smooth: bool) -> Result<Vec<u32>> {
    let p = &c.program;
    target.accumulate(e, &c.kernels, p, &p.look, &Frame::still(t, points, Mode::Warp))?;
    target.tonemap(e, &p.look, p.ir.has_h(), smooth)?;
    let mut px = Vec::new();
    target.read_pixels(&e.gpu, &mut px)?;
    Ok(px)
}

/// Draw a form at two moments and measure it.
pub fn evaluate(e: &Engine, target: &mut Target, c: &Compiled, points: u64) -> Result<(Feat, Vec<u32>)> {
    target.reset_exposure();
    let first = draw(e, target, c, points, 2.0, false)?;
    let later = draw(e, target, c, points, 2.6, true)?;
    let mut f = features(&luma(&first), &luma(&later), target.w as usize, target.h as usize);
    f.chroma = chroma(&first);
    Ok((f, first))
}

/// Seed forms from files and folders. Files that do not parse are skipped.
pub fn read_seeds(paths: &[String]) -> Result<Vec<Program>> {
    let paths: Vec<PathBuf> = if paths.is_empty() { vec!["forms".into()] } else { paths.iter().map(PathBuf::from).collect() };
    let mut seeds = Vec::new();
    for path in paths {
        let files = if path.is_dir() { forms_in(&path)? } else { vec![path] };
        for file in files {
            match Program::load(Path::new(&file)) {
                Ok(p) => seeds.push(p),
                Err(err) => eprintln!("skipping {err:#}"),
            }
        }
    }
    Ok(seeds)
}
