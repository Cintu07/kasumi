//! Offline rendering: a still png, or a video (ffmpeg + nvenc, hevc 10-bit)
//! with motion blur. Without ffmpeg, frames are written as pngs.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use kasumi_gpu::{FrameReq, Gpu, Mode, Statics, Target};

use crate::common::*;

pub fn render(args: &Args) -> Result<()> {
    let path = PathBuf::from(args.pos.first().context("usage: kasumi render <form.ks> [--out f.png|f.mp4] [--res 4k] [--points 200m] [--sub 8] [--seconds 10]")?);
    let mut l = load_form(&path)?;
    let gpu = Gpu::new()?;
    let st = Statics::new(&gpu)?;
    let fm = build_one(&gpu, &l, args.has("fast"))?;
    let unit = fm.units[0];
    let (w, h) = parse_res(args.get("res").unwrap_or("4k"))?;
    let ss = args.u32_or("ss", 2)?;
    let points = args.count_or("points", 200_000_000)?;
    let subs = args.u32_or("sub", 8)?;
    let fps = args.f32_or("fps", 60.0)?;
    let t0 = args.f32_or("t", 2.0)?;
    let mode_arg = args.get("mode").unwrap_or("auto");
    let fixed_mode = if mode_arg == "auto" { None } else { Some(Mode::parse(mode_arg).context("--mode auto|naive|warp|bin")?) };
    if let Some(p) = args.get("palette") {
        l.look.set_palette(p);
    }
    let seconds = args.f32_or("seconds", 0.0)?;
    let stem = stem(&path);

    println!("kasumi render | {} | {}x{} ss{} | {} pts x {} subframes | {}", path.display(), w, h, ss, human(points), subs, gpu.name);
    // pick the fastest accumulation kernel for this form at this size
    let mode = match fixed_mode {
        Some(m) => m,
        None => {
            let mut probe = Target::new(&gpu, w, h, ss, 2)?;
            let (m, all) = crate::live::autotune(&gpu, &st, &mut probe, &unit, &l.prog, &l.look, points.min(100_000_000));
            let s: Vec<String> = all.iter().map(|(m, ms)| format!("{} {:.1}", m.name(), gpts(points.min(100_000_000), *ms))).collect();
            println!("autotune Gpts/s: {} -> {}", s.join("  "), m.name());
            m
        }
    };

    if seconds <= 0.0 {
        let mut tgt = Target::new(&gpu, w, h, ss, 2)?;
        let req = FrameReq { t: t0, dt: 1.0 / fps, subframes: subs, points, mode, decay: 0.0 };
        let s = tgt.accumulate(&gpu, &st, &unit, &l.prog, &l.look, &req)?;
        tgt.tonemap(&gpu, &st, &l.look, l.prog.ir.has_h(), false)?;
        let mut rgb = Vec::new();
        tgt.read_bytes(&gpu, &mut rgb)?;
        let out = PathBuf::from(args.get("out").map(|s| s.to_string()).unwrap_or(format!("renders/{stem}.png")));
        write_png_rgb8(&out, w, h, &rgb)?;
        println!("{} points in {:.1} ms ({:.2} Gpts/s) -> {}", human(s.points), s.ms, gpts(s.points, s.ms), out.display());
        return Ok(());
    }

    let frames = (seconds * fps).round() as u32;
    let out = PathBuf::from(args.get("out").map(|s| s.to_string()).unwrap_or(format!("renders/{stem}.mp4")));
    if let Some(d) = out.parent() {
        std::fs::create_dir_all(d).ok();
    }
    let ffmpeg = find_ffmpeg();
    let mut tgt = Target::new(&gpu, w, h, ss, if ffmpeg.is_some() { 1 } else { 2 })?;
    let mut child = None;
    let frame_dir = out.with_extension("frames");
    if let Some(ff) = &ffmpeg {
        let codec = args.get("codec").unwrap_or("hevc_nvenc");
        let mut cmd = Command::new(ff);
        cmd.args(["-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb48le", "-s", &format!("{w}x{h}"), "-r", &fps.to_string(), "-i", "-"]);
        if codec.contains("nvenc") {
            cmd.args(["-c:v", codec, "-preset", "p6", "-rc", "vbr", "-cq", "16", "-b:v", "0", "-pix_fmt", "p010le", "-profile:v", "main10", "-tag:v", "hvc1"]);
        } else {
            cmd.args(["-c:v", codec, "-crf", "16", "-pix_fmt", "yuv420p10le"]);
        }
        cmd.arg(&out).stdin(Stdio::piped());
        child = Some(cmd.spawn().context("starting ffmpeg")?);
    } else {
        std::fs::create_dir_all(&frame_dir)?;
        println!("ffmpeg not found: writing png frames to {} (set KASUMI_FFMPEG or add ffmpeg to PATH for mp4)", frame_dir.display());
    }

    let started = Instant::now();
    let mut buf = Vec::new();
    for f in 0..frames {
        let t = t0 + f as f32 / fps;
        let req = FrameReq { t, dt: 1.0 / fps, subframes: subs, points, mode, decay: 0.0 };
        tgt.accumulate(&gpu, &st, &unit, &l.prog, &l.look, &req)?;
        tgt.tonemap(&gpu, &st, &l.look, l.prog.ir.has_h(), true)?;
        tgt.read_bytes(&gpu, &mut buf)?;
        if let Some(c) = child.as_mut() {
            c.stdin.as_mut().unwrap().write_all(&buf)?;
        } else {
            write_png_rgb8(&frame_dir.join(format!("{f:05}.png")), w, h, &buf)?;
        }
        let el = started.elapsed().as_secs_f64();
        print!("\rframe {}/{} | {:.2} fps | eta {:.0}s   ", f + 1, frames, (f + 1) as f64 / el, el / (f + 1) as f64 * (frames - f - 1) as f64);
        std::io::stdout().flush().ok();
    }
    println!();
    if let Some(mut c) = child {
        drop(c.stdin.take());
        let s = c.wait()?;
        if !s.success() {
            bail!("ffmpeg failed ({s}); try --codec libx265");
        }
        println!("wrote {}", out.display());
    } else {
        println!(
            "frames in {}\nencode: ffmpeg -framerate {fps} -i {}/%05d.png -c:v hevc_nvenc -preset p6 -cq 16 -pix_fmt p010le {}",
            frame_dir.display(),
            frame_dir.display(),
            out.display()
        );
    }
    Ok(())
}
