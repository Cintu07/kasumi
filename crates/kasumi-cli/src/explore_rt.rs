//! Shared by `grid` and `evolve`: compile many forms into one module and
//! render small thumbnails to score them.

use std::path::Path;
use std::rc::Rc;

use anyhow::Result;
use kasumi_explore::features::{features, luma, Feat};
use kasumi_explore::mutate::form_id;
use kasumi_explore::taste::Taste;
use kasumi_gpu::{CompileOpts, FormModule, FrameReq, Gpu, Look, Mode, Program, Statics, Target};
use kasumi_lang::{lower, print_form, Form};

#[derive(Clone)]
pub struct Cand {
    pub form: Form,
    pub text: String,
    pub id: String,
    pub prog: Program,
    pub look: Look,
}

impl Cand {
    pub fn new(form: Form) -> Cand {
        let text = print_form(&form);
        let ir = lower(&form);
        Cand { id: form_id(&text), prog: Program::new(&form, ir), look: Look::from_form(&form), text, form }
    }
}

/// A compiled candidate: which module, which unit.
#[derive(Clone)]
pub struct Live {
    pub cand: Cand,
    pub module: Rc<FormModule>,
    pub unit: usize,
}

/// Compile a batch with one nvrtc call. If the batch fails, fall back to
/// compiling one by one and drop the ones that fail.
pub fn compile(gpu: &Gpu, cands: Vec<Cand>) -> Vec<Live> {
    let irs: Vec<&kasumi_lang::Ir> = cands.iter().map(|c| &c.prog.ir).collect();
    if let Ok(m) = FormModule::build_with(gpu, &irs, CompileOpts::default(), false) {
        let m = Rc::new(m);
        return cands.into_iter().enumerate().map(|(k, c)| Live { cand: c, module: m.clone(), unit: k }).collect();
    }
    cands
        .into_iter()
        .filter_map(|c| {
            let m = FormModule::build_with(gpu, &[&c.prog.ir], CompileOpts::default(), false).ok()?;
            Some(Live { cand: c, module: Rc::new(m), unit: 0 })
        })
        .collect()
}

pub fn render_frame(gpu: &Gpu, st: &Statics, tgt: &mut Target, l: &Live, points: u64, t: f32, smooth: bool) -> Result<Vec<u32>> {
    let req = FrameReq { t, dt: 1.0 / 60.0, subframes: 1, points, mode: Mode::Warp, decay: 0.0 };
    tgt.accumulate(gpu, st, &l.module.units[l.unit], &l.cand.prog, &l.cand.look, &req)?;
    tgt.tonemap(gpu, st, &l.cand.look, l.cand.prog.ir.has_h(), smooth)?;
    let mut px = Vec::new();
    tgt.read_u32(gpu, &mut px)?;
    Ok(px)
}

/// Render at two moments and compute features (for filters / novelty / taste).
pub fn evaluate(gpu: &Gpu, st: &Statics, tgt: &mut Target, l: &Live, points: u64, t: f32) -> Result<(Feat, Vec<u32>)> {
    tgt.reset_exposure();
    let a = render_frame(gpu, st, tgt, l, points, t, false)?;
    let b = render_frame(gpu, st, tgt, l, points, t + 0.6, true)?;
    let f = features(&luma(&a), &luma(&b), tgt.w as usize, tgt.h as usize);
    Ok((f, a))
}

pub fn load_taste(path: &Path) -> Option<Taste> {
    std::fs::read_to_string(path).ok().and_then(|s| serde_json::from_str(&s).ok())
}

pub fn read_seeds(paths: &[String]) -> Result<Vec<Form>> {
    let mut files: Vec<std::path::PathBuf> = Vec::new();
    let list: Vec<String> = if paths.is_empty() { vec!["forms".into()] } else { paths.to_vec() };
    for p in list {
        let p = std::path::PathBuf::from(p);
        if p.is_dir() {
            let mut v: Vec<_> = std::fs::read_dir(&p)?.flatten().map(|e| e.path()).filter(|q| q.extension().is_some_and(|e| e == "ks")).collect();
            v.sort();
            files.extend(v);
        } else {
            files.push(p);
        }
    }
    let mut out = Vec::new();
    for f in files {
        let src = std::fs::read_to_string(&f)?;
        match kasumi_lang::parse(&src) {
            Ok(form) => out.push(form),
            Err(e) => eprintln!("skipping {}: {}", f.display(), e),
        }
    }
    Ok(out)
}
