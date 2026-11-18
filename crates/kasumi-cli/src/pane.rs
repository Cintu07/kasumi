//! `kasumi pane`: the right half of a live-coding video. Given the frames of a
//! recorded editing session (the session time each frame shows) and every
//! version of the form file saved during it, renders what the live window
//! would have shown: each save swaps the form a reload later, a save that
//! fails keeps the old form running with the error on screen.
//!
//!   kasumi pane plan.json out.mp4
//!
//! plan.json: { "fps": 60, "width": 1920, "height": 2160, "anim_t0": 2.0,
//!              "reload": 0.18, "frames": [vt, ...], "saves": [[vt, "text"], ...] }

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use anyhow::{bail, Context, Result};
use kasumi_gpu::{CompileOpts, FormModule, FrameReq, Gpu, Look, Mode, Program, Statics, Target};
use serde::Deserialize;

use crate::common::*;
use crate::hud;

#[derive(Deserialize)]
struct Plan {
    fps: f32,
    width: u32,
    height: u32,
    #[serde(default = "default_t0")]
    anim_t0: f32,
    #[serde(default = "default_reload")]
    reload: f32,
    #[serde(default = "default_ss")]
    ss: u32,
    /// draw kasumi's error text on the canvas (off when the video shows it elsewhere)
    #[serde(default = "default_true")]
    errors: bool,
    frames: Vec<f32>,
    saves: Vec<(f32, String)>,
}
fn default_t0() -> f32 {
    2.0
}
fn default_reload() -> f32 {
    0.18
}
fn default_ss() -> u32 {
    2
}
fn default_true() -> bool {
    true
}

struct Version {
    at: f32,
    good: Option<(FormModule, Program, Look, u64)>,
    error: Option<String>,
}

const BG: u32 = 0x090909;

pub fn pane(args: &Args) -> Result<()> {
    let plan_path = PathBuf::from(args.pos.first().context("usage: kasumi pane plan.json out.mp4")?);
    let out = PathBuf::from(args.pos.get(1).context("usage: kasumi pane plan.json out.mp4")?);
    let plan: Plan = serde_json::from_str(&std::fs::read_to_string(&plan_path)?)?;
    let gpu = Gpu::new()?;
    let st = Statics::new(&gpu)?;

    // every saved version, compiled once; a failing save carries the error and
    // keeps whatever was running before it
    let mut versions: Vec<Version> = Vec::new();
    for (vt, text) in &plan.saves {
        match kasumi_lang::parse(text) {
            Ok(form) => {
                let ir = kasumi_lang::lower(&form);
                let module = FormModule::build_with(&gpu, &[&ir], CompileOpts::default(), false)?;
                let points = form.num_or("points", 20e6) as u64;
                versions.push(Version { at: *vt, good: Some((module, Program::new(&form, ir), Look::from_form(&form), points)), error: None });
            }
            Err(e) => {
                let msg = e.render(text);
                versions.push(Version { at: *vt, good: None, error: Some(msg) });
            }
        }
    }
    if versions.first().is_none_or(|v| v.good.is_none()) {
        bail!("the first saved version must compile");
    }

    // the canvas itself: the editor and the window round it are drawn by the
    // video builder, this is laid over the canvas rectangle
    let (pw, ph) = (plan.width, plan.height);
    let scale = (ph / 420).max(1) as i32; // error text readable at full size
    let mut tgt = Target::new(&gpu, pw, ph, plan.ss.max(1), 0)?;

    let ff = find_ffmpeg().context("ffmpeg not found")?;
    let mut enc = Command::new(ff)
        .args(["-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr0", "-s", &format!("{pw}x{ph}"), "-r", &plan.fps.to_string(), "-i", "-"])
        .args(["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "17", "-b:v", "0", "-pix_fmt", "yuv420p"])
        .arg(&out)
        .stdin(Stdio::piped())
        .spawn()?;
    let stdin = enc.stdin.as_mut().unwrap();

    let mut frame = vec![BG; (pw * ph) as usize];
    let mut px = Vec::new();
    let n = plan.frames.len();
    for (k, &vt) in plan.frames.iter().enumerate() {
        // the newest save that has had time to reload, and the newest good one
        let live = versions.iter().rposition(|v| v.at + plan.reload <= vt).unwrap_or(0);
        let good = (0..=live).rev().find(|&i| versions[i].good.is_some()).unwrap();
        let error = if versions[live].good.is_none() { versions[live].error.as_deref() } else { None };
        let (module, prog, look, points) = versions[good].good.as_ref().unwrap();
        let t = plan.anim_t0 + k as f32 / plan.fps;
        let req = FrameReq { t, dt: 1.0 / plan.fps, subframes: 2, points: *points, mode: Mode::Warp, decay: look.trail };
        tgt.accumulate(&gpu, &st, &module.units[0], prog, look, &req)?;
        tgt.tonemap(&gpu, &st, look, prog.ir.has_h(), true)?;
        tgt.read_u32(&gpu, &mut px)?;

        let (w, h) = (pw as usize, ph as usize);
        // a p5 sketch's background(9): empty canvas is #090909, not pure black
        for (o, &p) in frame.iter_mut().zip(px.iter()) {
            let r = ((p >> 16) & 255).max(9);
            let g = ((p >> 8) & 255).max(9);
            let b = (p & 255).max(9);
            *o = (r << 16) | (g << 8) | b;
        }
        if let Some(e) = error.filter(|_| plan.errors) {
            let lines: Vec<&str> = e.lines().take(4).collect();
            let lh = 11 * scale;
            let top = h as i32 - lines.len() as i32 * lh - 14 * scale;
            hud::dim_rect(&mut frame, w, h, 0, top - 8 * scale, w as i32, h as i32);
            for (j, l) in lines.iter().enumerate() {
                hud::label(&mut frame, w, h, 12 * scale, top + j as i32 * lh, l, hud::RED, scale);
            }
        }
        let bytes: &[u8] = unsafe { std::slice::from_raw_parts(frame.as_ptr() as *const u8, frame.len() * 4) };
        stdin.write_all(bytes)?;
        if k % 120 == 0 {
            print!("\rpane {k}/{n}   ");
            std::io::stdout().flush().ok();
        }
    }
    drop(enc.stdin.take());
    let s = enc.wait()?;
    if !s.success() {
        bail!("ffmpeg failed");
    }
    println!("\rpane: {n} frames -> {}", out.display());
    Ok(())
}
