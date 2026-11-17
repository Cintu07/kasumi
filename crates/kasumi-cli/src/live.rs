//! The live window: renders the form every frame, recompiles on save
//! (the old kernel keeps running if the new one fails), and shows what the
//! engine is doing in a small hud.

use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime};

use anyhow::{Context, Result};
use kasumi_gpu::{check, FormModule, FrameReq, Gpu, Look, Mode, Program, Statics, Target, UnitFns, PALETTES};
use minifb::{Key, KeyRepeat, MouseButton, MouseMode, ScaleMode, Window, WindowOptions};

use crate::common::*;
use crate::hud::*;

struct Current {
    module: FormModule,
    prog: Program,
    look: Look,
    has_h: bool,
    points: u64,
    stats: kasumi_lang::ir::IrStats,
    check_pct: f64,
    check_ok: bool,
}

impl Current {
    fn unit(&self) -> &UnitFns {
        &self.module.units[0]
    }
}

fn load(gpu: &Gpu, path: &Path, fast: bool, with_bin: bool) -> Result<Current> {
    let t0 = Instant::now();
    let l = load_form(path)?;
    let t1 = Instant::now();
    let module = FormModule::build_with(gpu, &[&l.prog.ir], opts_for(&l.form, fast), with_bin)?;
    let t2 = Instant::now();
    let points = l.form.num_or("points", 20e6) as u64;
    let rep = check(gpu, &module.units[0], &l.prog, points, 1.0, 4096)?;
    if std::env::var_os("KASUMI_PROF").is_some() {
        eprintln!(
            "reload stages: parse {:.1} ms | nvrtc+jit {:.1} ms | cpu check {:.1} ms",
            (t1 - t0).as_secs_f64() * 1e3,
            (t2 - t1).as_secs_f64() * 1e3,
            t2.elapsed().as_secs_f64() * 1e3
        );
    }
    Ok(Current {
        has_h: l.prog.ir.has_h(),
        stats: l.prog.ir.stats.clone(),
        prog: l.prog,
        look: l.look,
        points,
        module,
        check_pct: rep.frac() * 100.0,
        check_ok: rep.ok(),
    })
}

fn mtime(p: &Path) -> Option<SystemTime> {
    std::fs::metadata(p).and_then(|m| m.modified()).ok()
}

/// Time every accumulation mode on this form and pick the fastest.
pub fn autotune(gpu: &Gpu, st: &Statics, tgt: &mut Target, unit: &UnitFns, prog: &Program, look: &Look, points: u64) -> (Mode, Vec<(Mode, f64)>) {
    let mut best = (Mode::Warp, f64::MAX);
    let mut all = Vec::new();
    for mode in Mode::ALL {
        let req = FrameReq { t: 1.0, dt: 1.0 / 60.0, subframes: 1, points, mode, decay: 0.0 };
        let mut ms = f64::MAX;
        for _ in 0..3 {
            match tgt.accumulate(gpu, st, unit, prog, look, &req) {
                Ok(s) => ms = ms.min(s.ms),
                Err(_) => break,
            }
        }
        all.push((mode, ms));
        if ms < best.1 {
            best = (mode, ms);
        }
    }
    (best.0, all)
}

pub fn live(args: &Args) -> Result<()> {
    let path = PathBuf::from(args.pos.first().context("usage: kasumi live <form.ks>")?);
    let gpu = Gpu::new()?;
    let st = Statics::new(&gpu)?;
    let (w, h) = parse_res(args.get("res").unwrap_or("1080p"))?;
    let ss = args.u32_or("ss", 1)?;
    let mut fast = args.has("fast");
    let mut auto = args.get("mode").is_none_or(|m| m == "auto");
    let mut mode = Mode::parse(args.get("mode").unwrap_or("warp")).unwrap_or(Mode::Warp);
    let points_override = args.get("points").map(parse_count).transpose()?;

    let mut tgt = Target::new(&gpu, w, h, ss, 0)?;
    let mut error: Option<String> = None;
    let mut cur = match load(&gpu, &path, fast, false) {
        Ok(c) => Some(c),
        Err(e) => {
            error = Some(format!("{e:#}"));
            None
        }
    };

    let (ww, wh) = if w > 1600 { (w * 2 / 3, h * 2 / 3) } else { (w, h) };
    let mut window = Window::new(
        &format!("kasumi - {}", path.display()),
        ww as usize,
        wh as usize,
        WindowOptions { resize: true, scale_mode: ScaleMode::AspectRatioStretch, ..WindowOptions::default() },
    )?;
    window.set_target_fps(60);

    let mut buf: Vec<u32> = vec![0; (w * h) as usize];
    let mut last_m = mtime(&path);
    let mut t = 0.0f32;
    let mut speed = 1.0f32;
    let mut paused = false;
    let mut show_hud = true;
    let mut show_help = false;
    let mut trail_on = false;
    let mut blur = 1u32;
    let mut points_mul = 1.0f64;
    let mut pal_ix: usize = 0;
    let mut pal_override: Option<usize> = None;
    let (mut zoom_mul, mut pan) = (1.0f32, (0.0f32, 0.0f32));
    let mut drag: Option<(f32, f32)> = None;
    let mut last = Instant::now();
    let mut fps = 60.0f64;
    let mut rate = 0.0f64;
    let mut reload_started: Option<Instant> = None;
    let mut reload_ms: Option<f64> = None;
    let mut toast: Option<(String, Instant)> = None;
    let mut tuned: Vec<(Mode, f64)> = Vec::new();
    let mut need_tune = auto;
    let mut tune_next = false;
    let max_frames = args.get("frames").map(|v| v.parse::<u64>()).transpose()?;
    let shot_path = args.get("shot").map(PathBuf::from);
    let mut frame_no = 0u64;

    while window.is_open() && !window.is_key_down(Key::Escape) {
        // ---- hot reload ----
        let m = mtime(&path);
        if m != last_m {
            last_m = m;
            let started = Instant::now();
            std::thread::sleep(std::time::Duration::from_millis(5)); // let the editor finish writing
            let empty = std::fs::read_to_string(&path).map(|s| s.trim().is_empty()).unwrap_or(true);
            if !empty {
                match load(&gpu, &path, fast, mode == Mode::Bin && !auto) {
                    Ok(c) => {
                        cur = Some(c);
                        error = None;
                        reload_started = Some(started);
                        tune_next = auto; // after the new frame is on screen
                    }
                    Err(e) => error = Some(format!("{e:#}")),
                }
            }
        }

        let now = Instant::now();
        let dt = now.duration_since(last).as_secs_f64();
        last = now;
        fps = fps * 0.9 + (1.0 / dt.max(1e-4)) * 0.1;
        if !paused {
            t += dt as f32 * speed;
        }

        if let Some(c) = cur.as_mut() {
            let mut look = c.look.clone();
            if let Some(p) = pal_override {
                look.set_palette(PALETTES[p].0);
            }
            look.zoom *= zoom_mul;
            look.cx += pan.0;
            look.cy += pan.1;
            let points = ((points_override.unwrap_or(c.points) as f64) * points_mul) as u64;
            let diffuse = tgt.lit as f64 > 0.35 * (tgt.aw as f64 * tgt.ah as f64);
            if (need_tune && diffuse || (!auto && mode == Mode::Bin)) && !c.module.with_bin {
                // tile binning only pays off on diffuse forms; compile it on demand
                if let Ok(full) = build_one(&gpu, &load_form(&path)?, fast) {
                    c.module = full;
                }
            }
            if need_tune {
                let (m, all) = autotune(&gpu, &st, &mut tgt, c.unit(), &c.prog, &look, points);
                mode = m;
                tuned = all;
                need_tune = false;
            }
            let trail = if trail_on { if look.trail > 0.0 { look.trail } else { 0.9 } } else { look.trail };
            let req = FrameReq { t, dt: 1.0 / 60.0, subframes: blur, points, mode, decay: trail };
            match tgt.accumulate(&gpu, &st, c.unit(), &c.prog, &look, &req) {
                Ok(s) => {
                    rate = rate * 0.8 + gpts(s.points, s.ms) * 0.2;
                    tgt.tonemap(&gpu, &st, &look, c.has_h, true)?;
                    tgt.read_u32(&gpu, &mut buf)?;
                }
                Err(e) => {
                    error = Some(format!("{e:#}"));
                    mode = Mode::Warp;
                }
            }

            if show_hud {
                let (bw, bh) = (w as usize, h as usize);
                let sc = if h >= 1000 { 2 } else { 1 };
                let lh = 10 * sc;
                let mode_s = if auto { format!("{} (auto)", mode.name()) } else { mode.name().to_string() };
                let l1 = format!(
                    "kasumi | {} | {} pts x{} | {} | {:.1} Gpts/s | {:.0} fps",
                    stem(&path),
                    human(points),
                    blur,
                    mode_s,
                    rate,
                    fps
                );
                let s = &c.stats;
                let mut l2 = format!(
                    "compile {:.0} ms | ir {} ops/pt, {} hoisted, {} folded, {} cse, {} dead | cpu ref {} {:.2}%",
                    c.module.compile_ms,
                    s.per_point,
                    s.hoisted,
                    s.folded,
                    s.cse_hits,
                    s.dead,
                    if c.check_ok { "ok" } else { "MISMATCH" },
                    c.check_pct
                );
                if let Some(r) = reload_ms {
                    l2.push_str(&format!(" | edit->screen {r:.0} ms"));
                }
                let l3 = format!(
                    "t {:.1}s{} | palette {} | trail {} | zoom {:.2}{}",
                    t,
                    if paused { " (paused)" } else { "" },
                    look.palette,
                    if trail > 0.0 { format!("{trail:.2}") } else { "off".into() },
                    look.zoom,
                    if fast { " | fast-math" } else { "" }
                );
                label(&mut buf, bw, bh, 12, 10, &l1, WHITE, sc);
                label(&mut buf, bw, bh, 12, 10 + lh + 4, &l2, GREY, sc);
                label(&mut buf, bw, bh, 12, 10 + 2 * (lh + 4), &l3, GREY, sc);
                if !tuned.is_empty() && auto {
                    let tl = tuned
                        .iter()
                        .map(|(m, ms)| if ms.is_finite() && *ms < 1e9 { format!("{} {:.1}", m.name(), gpts(points, *ms)) } else { format!("{} -", m.name()) })
                        .collect::<Vec<_>>()
                        .join("  ");
                    label(&mut buf, bw, bh, 12, 10 + 3 * (lh + 4), &format!("autotune Gpts/s: {tl}"), GREY, sc);
                }
            }
        } else {
            buf.iter_mut().for_each(|p| *p = 0x050505);
        }

        let (bw, bh) = (w as usize, h as usize);
        let sc = if h >= 1000 { 2 } else { 1 };
        if let Some(e) = &error {
            let lines: Vec<&str> = e.lines().take(8).collect();
            let top = bh as i32 - (lines.len() as i32 * 10 * sc + 24);
            dim_rect(&mut buf, bw, bh, 0, top - 8, bw as i32, bh as i32);
            for (k, line) in lines.iter().enumerate() {
                label(&mut buf, bw, bh, 16, top + k as i32 * 10 * sc, line, RED, sc);
            }
        }
        if let Some((msg, at)) = &toast {
            if at.elapsed().as_secs_f32() < 2.5 {
                label(&mut buf, bw, bh, 12, bh as i32 - 14 * sc - 30, msg, AMBER, sc);
            }
        }
        if show_help {
            let help = [
                "space pause   r reset   left/right speed   up/down points x2",
                "m accumulate mode (naive/warp/bin)   a auto-tune   f fast-math",
                "p palette   t trails   b motion blur   wheel zoom   drag pan",
                "s screenshot   e export 4k still   h hud   esc quit",
            ];
            let top = bh as i32 / 2 - 40;
            dim_rect(&mut buf, bw, bh, 0, top - 16, bw as i32, top + 4 * 12 * sc + 8);
            for (k, l) in help.iter().enumerate() {
                label(&mut buf, bw, bh, 40, top + k as i32 * 12 * sc, l, WHITE, sc);
            }
        }

        window.update_with_buffer(&buf, bw, bh)?;
        if let Some(s) = reload_started.take() {
            let ms = s.elapsed().as_secs_f64() * 1e3;
            reload_ms = Some(ms);
            println!("reload: edit -> screen {ms:.0} ms");
        }
        if tune_next {
            need_tune = true;
            tune_next = false;
        }
        frame_no += 1;
        if max_frames.is_some_and(|n| frame_no >= n) {
            if let Some(p) = &shot_path {
                write_png_rgb8(p, w, h, &u32_to_rgb8(&buf))?;
            }
            println!("live: {} frames, {:.0} fps, {:.1} Gpts/s, mode {}", frame_no, fps, rate, mode.name());
            break;
        }

        // ---- input ----
        for k in window.get_keys_pressed(KeyRepeat::No) {
            match k {
                Key::Space => paused = !paused,
                Key::R => {
                    t = 0.0;
                    zoom_mul = 1.0;
                    pan = (0.0, 0.0);
                    tgt.reset_exposure();
                }
                Key::Left => speed = (speed * 0.5).max(1.0 / 64.0),
                Key::Right => speed = (speed * 2.0).min(16.0),
                Key::Up => points_mul = (points_mul * 2.0).min(64.0),
                Key::Down => points_mul = (points_mul * 0.5).max(1.0 / 64.0),
                Key::M => {
                    auto = false;
                    mode = mode.next();
                }
                Key::A => {
                    auto = true;
                    need_tune = true;
                }
                Key::F => {
                    fast = !fast;
                    last_m = None; // force a recompile
                }
                Key::P => {
                    pal_ix = (pal_override.unwrap_or(pal_ix) + 1) % PALETTES.len();
                    pal_override = Some(pal_ix);
                }
                Key::T => trail_on = !trail_on,
                Key::B => blur = if blur == 1 { 4 } else { 1 },
                Key::H => show_hud = !show_hud,
                Key::Slash | Key::F1 => show_help = !show_help,
                Key::S => {
                    let out = PathBuf::from(format!("shots/{}_{}.png", stem(&path), timestamp()));
                    let rgb = u32_to_rgb8(&buf);
                    match write_png_rgb8(&out, w, h, &rgb) {
                        Ok(_) => toast = Some((format!("saved {}", out.display()), Instant::now())),
                        Err(e) => toast = Some((format!("screenshot failed: {e}"), Instant::now())),
                    }
                }
                Key::E => {
                    if let Some(c) = cur.as_ref() {
                        let mut look = c.look.clone();
                        if let Some(p) = pal_override {
                            look.set_palette(PALETTES[p].0);
                        }
                        look.zoom *= zoom_mul;
                        look.cx += pan.0;
                        look.cy += pan.1;
                        let out = PathBuf::from(format!("renders/{}_{}_4k.png", stem(&path), timestamp()));
                        let r = (|| -> Result<f64> {
                            let mut big = Target::new(&gpu, 3840, 2160, 2, 2)?;
                            let req = FrameReq { t, dt: 1.0 / 60.0, subframes: 8, points: 200_000_000, mode: Mode::Warp, decay: 0.0 };
                            let s = big.accumulate(&gpu, &st, c.unit(), &c.prog, &look, &req)?;
                            big.tonemap(&gpu, &st, &look, c.has_h, false)?;
                            let mut rgb = Vec::new();
                            big.read_bytes(&gpu, &mut rgb)?;
                            write_png_rgb8(&out, 3840, 2160, &rgb)?;
                            Ok(s.ms)
                        })();
                        toast = Some(match r {
                            Ok(ms) => (format!("4k still (1.6b points, {:.0} ms) -> {}", ms, out.display()), Instant::now()),
                            Err(e) => (format!("export failed: {e}"), Instant::now()),
                        });
                    }
                }
                _ => {}
            }
        }
        // zoom and pan
        if let Some((_, dy)) = window.get_scroll_wheel() {
            if dy != 0.0 {
                zoom_mul *= 1.1f32.powf(dy.signum());
            }
        }
        let (win_w, win_h) = window.get_size();
        let _ = win_w;
        if window.get_mouse_down(MouseButton::Left) {
            if let Some((mx, my)) = window.get_mouse_pos(MouseMode::Pass) {
                if let (Some((ox, oy)), Some(c)) = (drag, cur.as_ref()) {
                    let world_per_px = 2.0 / (c.look.zoom * zoom_mul * win_h.max(1) as f32);
                    pan.0 -= (mx - ox) * world_per_px;
                    pan.1 += (my - oy) * world_per_px;
                }
                drag = Some((mx, my));
            }
        } else {
            drag = None;
        }
    }
    Ok(())
}
