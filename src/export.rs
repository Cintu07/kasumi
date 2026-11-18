//! Offline rendering: a still PNG, or a video through ffmpeg and NVENC.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Instant;

use anyhow::{bail, Context, Result};

use crate::cli::*;
use crate::render::{compile_one, fastest_mode, Engine, Format, Frame, Mode, Program, Target};

const USAGE: &str = "kasumi render <form.ks> [--out f.png|f.mp4] [--res 4k] [--points 200m] [--sub 8] [--seconds 10]";

fn encoder(codec: &str) -> Result<&'static [&'static str]> {
    Ok(match codec {
        "h264" => &["-c:v", "h264_nvenc", "-preset", "p6", "-profile:v", "high", "-rc", "vbr", "-cq", "18", "-b:v", "0", "-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        "hevc" => &["-c:v", "hevc_nvenc", "-preset", "p6", "-rc", "vbr", "-cq", "16", "-b:v", "0", "-pix_fmt", "p010le", "-profile:v", "main10", "-tag:v", "hvc1"],
        other => bail!("--codec must be h264 or hevc, got '{other}'"),
    })
}

pub fn render(args: &Args) -> Result<()> {
    let path = args.path(0, USAGE)?;
    let mut program = Program::load(&path)?;
    let engine = Engine::new()?;
    let gpu = &engine.gpu;
    let kernels = compile_one(gpu, &program, true)?;
    let (w, h) = parse_res(args.get("res").unwrap_or("4k"))?;
    let ss = args.u32_or("ss", 2)?;
    let points = args.count_or("points", 200_000_000)?;
    let subframes = args.u32_or("sub", 8)?;
    let fps = args.f32_or("fps", 60.0)?;
    let start_t = args.f32_or("t", 2.0)?;
    let seconds = args.f32_or("seconds", 0.0)?;
    if let Some(p) = args.get("palette") {
        program.look.set_palette(p);
    }
    let look = program.look.clone();
    let has_h = program.ir.has_h();
    println!("kasumi render | {} | {w}x{h} ss{ss} | {} pts x {subframes} subframes | {}", path.display(), human(points), gpu.name);

    let mode = match args.get("mode").unwrap_or("auto") {
        "auto" => {
            let mut probe = Target::new(gpu, w, h, ss, Format::Rgb8)?;
            let sample = points.min(100_000_000);
            let (mode, timings) = fastest_mode(&engine, &mut probe, &kernels, &program, &look, sample);
            let rates: Vec<String> = timings.iter().map(|(m, ms)| format!("{} {:.1}", m.name(), gpts(sample, *ms))).collect();
            println!("autotune Gpts/s: {} -> {}", rates.join("  "), mode.name());
            mode
        }
        name => Mode::parse(name).context("--mode auto|naive|warp|bin")?,
    };
    let frame_at = |t: f32| Frame { t, dt: 1.0 / fps, subframes, points, mode, decay: 0.0 };
    let name = stem(&path);

    if seconds <= 0.0 {
        let mut target = Target::new(gpu, w, h, ss, Format::Rgb8)?;
        let stats = target.accumulate(&engine, &kernels, &program, &look, &frame_at(start_t))?;
        target.tonemap(&engine, &look, has_h, false)?;
        let mut rgb = Vec::new();
        target.read_bytes(gpu, &mut rgb)?;
        let out = args.get("out").map_or(PathBuf::from(format!("renders/{name}.png")), PathBuf::from);
        write_png(&out, w, h, &rgb)?;
        println!("{} points in {:.1} ms ({:.2} Gpts/s) -> {}", human(stats.points), stats.ms, gpts(stats.points, stats.ms), out.display());
        return Ok(());
    }

    let out = args.get("out").map_or(PathBuf::from(format!("renders/{name}.mp4")), PathBuf::from);
    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let ffmpeg = std::env::var("KASUMI_FFMPEG").unwrap_or("ffmpeg".into());
    let mut child = Command::new(ffmpeg)
        .args(["-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb48le", "-s", &format!("{w}x{h}"), "-r", &fps.to_string(), "-i", "-"])
        .args(encoder(args.get("codec").unwrap_or("h264"))?)
        .arg(&out)
        .stdin(Stdio::piped())
        .spawn()
        .context("starting ffmpeg. install it, or set KASUMI_FFMPEG to its path")?;
    let mut stdin = child.stdin.take().context("ffmpeg stdin")?;

    let mut target = Target::new(gpu, w, h, ss, Format::Rgb16)?;
    let frames = (seconds * fps).round() as u32;
    let started = Instant::now();
    let mut buf = Vec::new();
    for f in 0..frames {
        target.accumulate(&engine, &kernels, &program, &look, &frame_at(start_t + f as f32 / fps))?;
        target.tonemap(&engine, &look, has_h, true)?;
        target.read_bytes(gpu, &mut buf)?;
        stdin.write_all(&buf)?;
        let elapsed = started.elapsed().as_secs_f64();
        let done = (f + 1) as f64;
        print!("\rframe {}/{frames} | {:.2} fps | eta {:.0}s   ", f + 1, done / elapsed, elapsed / done * (frames as f64 - done));
        std::io::stdout().flush().ok();
    }
    println!();
    drop(stdin);
    let status = child.wait()?;
    if !status.success() {
        bail!("ffmpeg failed ({status})");
    }
    println!("wrote {}", out.display());
    Ok(())
}
