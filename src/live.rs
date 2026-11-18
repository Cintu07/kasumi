//! The live window: draws the form every frame and recompiles it on save. If
//! the new version fails, the old kernel keeps running with the error on screen.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

use anyhow::Result;
use minifb::{Key, KeyRepeat, MouseButton, MouseMode, ScaleMode, Window, WindowOptions};

use crate::cli::*;
use crate::ir::IrStats;
use crate::look::{Look, PALETTES};
use crate::render::{check, compile_one, fastest_mode, Engine, Format, Frame, Kernels, Mode, Program, Target};
use crate::ui::*;

const HELP: [&str; 4] = [
    "space pause   r reset   left/right speed   up/down points x2",
    "m accumulate mode (naive/warp/bin)   a auto-tune",
    "p palette   t trails   b motion blur   wheel zoom   drag pan",
    "s screenshot   h hud   esc quit",
];

struct Loaded {
    kernels: Kernels,
    program: Program,
    stats: IrStats,
    compile_ms: f64,
    check_percent: f64,
    check_ok: bool,
}

fn load(e: &Engine, path: &Path, with_bin: bool) -> Result<Loaded> {
    let program = Program::load(path)?;
    let start = Instant::now();
    let kernels = compile_one(&e.gpu, &program, with_bin)?;
    let compile_ms = start.elapsed().as_secs_f64() * 1e3;
    let report = check(&e.gpu, &kernels, &program, program.points, 1.0, 4096)?;
    Ok(Loaded {
        stats: program.ir.stats.clone(),
        kernels,
        program,
        compile_ms,
        check_percent: report.fraction() * 100.0,
        check_ok: report.ok(),
    })
}

fn modified(p: &Path) -> Option<SystemTime> {
    std::fs::metadata(p).and_then(|m| m.modified()).ok()
}

struct App {
    engine: Engine,
    path: PathBuf,
    target: Target,
    buf: Vec<u32>,
    current: Option<Loaded>,
    error: Option<String>,
    last_modified: Option<SystemTime>,
    reload_started: Option<Instant>,
    reload_ms: Option<f64>,
    toast: Option<(String, Instant)>,

    auto: bool,
    mode: Mode,
    timings: Vec<(Mode, f64)>,
    need_tune: bool,
    tune_next_frame: bool,

    t: f32,
    speed: f32,
    paused: bool,
    show_hud: bool,
    show_help: bool,
    trails: bool,
    subframes: u32,
    points_override: Option<u64>,
    points_scale: f64,
    palette: Option<usize>,
    zoom: f32,
    pan: (f32, f32),
    drag: Option<(f32, f32)>,
    fps: f64,
    rate: f64,
    points: u64,
}

impl App {
    fn reload_if_saved(&mut self) {
        let now = modified(&self.path);
        if now == self.last_modified {
            return;
        }
        self.last_modified = now;
        let started = Instant::now();
        std::thread::sleep(Duration::from_millis(5)); // let the editor finish writing
        if std::fs::read_to_string(&self.path).map_or(true, |s| s.trim().is_empty()) {
            return;
        }
        match load(&self.engine, &self.path, self.mode == Mode::Bin && !self.auto) {
            Ok(loaded) => {
                self.current = Some(loaded);
                self.error = None;
                self.reload_started = Some(started);
                self.tune_next_frame = self.auto;
            }
            Err(err) => self.error = Some(format!("{err:#}")),
        }
    }

    /// The form's look with the window's palette, zoom and pan applied.
    fn view(&self, base: &Look) -> Look {
        let mut look = base.clone();
        if let Some(p) = self.palette {
            look.set_palette(PALETTES[p].0);
        }
        look.zoom *= self.zoom;
        look.cx += self.pan.0;
        look.cy += self.pan.1;
        look
    }

    fn render(&mut self) -> Result<()> {
        let Some(mut current) = self.current.take() else {
            self.buf.fill(0x050505);
            return Ok(());
        };
        let look = self.view(&current.program.look);
        self.points = (self.points_override.unwrap_or(current.program.points) as f64 * self.points_scale) as u64;

        // binning only pays off on diffuse forms, so its kernel is compiled on demand
        let diffuse = self.target.lit as f64 > 0.35 * self.target.aw as f64 * self.target.ah as f64;
        if (self.need_tune && diffuse || (!self.auto && self.mode == Mode::Bin)) && !current.kernels.has_bin() {
            if let Ok(full) = compile_one(&self.engine.gpu, &current.program, true) {
                current.kernels = full;
            }
        }
        if self.need_tune {
            (self.mode, self.timings) = fastest_mode(&self.engine, &mut self.target, &current.kernels, &current.program, &look, self.points);
            self.need_tune = false;
        }

        let trail = if self.trails && look.trail <= 0.0 { 0.9 } else { look.trail };
        let frame = Frame { t: self.t, dt: 1.0 / 60.0, subframes: self.subframes, points: self.points, mode: self.mode, decay: trail };
        match self.target.accumulate(&self.engine, &current.kernels, &current.program, &look, &frame) {
            Ok(stats) => {
                self.rate = self.rate * 0.8 + gpts(stats.points, stats.ms) * 0.2;
                self.target.tonemap(&self.engine, &look, current.program.ir.has_h(), true)?;
                self.target.read_pixels(&self.engine.gpu, &mut self.buf)?;
            }
            Err(err) => {
                self.error = Some(format!("{err:#}"));
                self.mode = Mode::Warp;
            }
        }
        if self.show_hud {
            self.draw_hud(&current, &look, trail);
        }
        self.current = Some(current);
        Ok(())
    }

    fn draw_hud(&mut self, c: &Loaded, look: &Look, trail: f32) {
        let (w, h) = (self.target.w as usize, self.target.h as usize);
        let scale = if h >= 1000 { 2 } else { 1 };
        let step = 10 * scale + 4;
        let mode = if self.auto { format!("{} (auto)", self.mode.name()) } else { self.mode.name().to_string() };
        let s = &c.stats;
        let mut lines = vec![(
            format!("kasumi | {} | {} pts x{} | {mode} | {:.1} Gpts/s | {:.0} fps", stem(&self.path), human(self.points), self.subframes, self.rate, self.fps),
            WHITE,
        )];
        let verdict = if c.check_ok { "ok" } else { "MISMATCH" };
        let mut compiled = format!(
            "compile {:.0} ms | ir {} ops/pt, {} hoisted, {} folded, {} cse, {} dead | cpu ref {verdict} {:.2}%",
            c.compile_ms, s.per_point, s.hoisted, s.folded, s.cse_hits, s.dead, c.check_percent
        );
        if let Some(ms) = self.reload_ms {
            compiled.push_str(&format!(" | edit->screen {ms:.0} ms"));
        }
        lines.push((compiled, GREY));
        let paused = if self.paused { " (paused)" } else { "" };
        let trail = if trail > 0.0 { format!("{trail:.2}") } else { "off".into() };
        lines.push((format!("t {:.1}s{paused} | palette {} | trail {trail} | zoom {:.2}", self.t, look.palette, look.zoom), GREY));
        if self.auto && !self.timings.is_empty() {
            let rates: Vec<String> = self.timings.iter().map(|(m, ms)| {
                if *ms < 1e9 { format!("{} {:.1}", m.name(), gpts(self.points, *ms)) } else { format!("{} -", m.name()) }
            }).collect();
            lines.push((format!("autotune Gpts/s: {}", rates.join("  ")), GREY));
        }
        let mut canvas = Canvas { px: &mut self.buf, w, h };
        for (k, (line, color)) in lines.iter().enumerate() {
            canvas.label(12, 10 + k as i32 * step, line, *color, 12.0 * scale as f32);
        }
    }

    fn draw_messages(&mut self) {
        let (w, h) = (self.target.w as usize, self.target.h as usize);
        let scale = if h >= 1000 { 2 } else { 1 };
        let (right, bottom) = (w as i32, h as i32);
        let mut canvas = Canvas { px: &mut self.buf, w, h };
        if let Some(err) = &self.error {
            let lines: Vec<&str> = err.lines().take(8).collect();
            let top = bottom - (lines.len() as i32 * 10 * scale + 24);
            canvas.dim(0, top - 8, right, bottom);
            for (k, line) in lines.iter().enumerate() {
                canvas.label(16, top + k as i32 * 10 * scale, line, RED, 12.0 * scale as f32);
            }
        }
        if let Some((msg, _)) = self.toast.as_ref().filter(|(_, at)| at.elapsed().as_secs_f32() < 2.5) {
            canvas.label(12, bottom - 14 * scale - 30, msg, AMBER, 12.0 * scale as f32);
        }
        if self.show_help {
            let top = bottom / 2 - 40;
            canvas.dim(0, top - 16, right, top + 4 * 12 * scale + 8);
            for (k, line) in HELP.iter().enumerate() {
                canvas.label(40, top + k as i32 * 12 * scale, line, WHITE, 12.0 * scale as f32);
            }
        }
    }

    fn screenshot(&mut self) {
        let out = PathBuf::from(format!("shots/{}_{}.png", stem(&self.path), timestamp()));
        let saved = write_png(&out, self.target.w, self.target.h, &to_rgb(&self.buf));
        let msg = saved.map_or_else(|err| format!("screenshot failed: {err}"), |_| format!("saved {}", out.display()));
        self.toast = Some((msg, Instant::now()));
    }

    fn handle_keys(&mut self, window: &Window) {
        for key in window.get_keys_pressed(KeyRepeat::No) {
            match key {
                Key::Space => self.paused = !self.paused,
                Key::R => {
                    (self.t, self.zoom, self.pan) = (0.0, 1.0, (0.0, 0.0));
                    self.target.reset_exposure();
                }
                Key::Left => self.speed = (self.speed * 0.5).max(1.0 / 64.0),
                Key::Right => self.speed = (self.speed * 2.0).min(16.0),
                Key::Up => self.points_scale = (self.points_scale * 2.0).min(64.0),
                Key::Down => self.points_scale = (self.points_scale * 0.5).max(1.0 / 64.0),
                Key::M => (self.auto, self.mode) = (false, self.mode.next()),
                Key::A => (self.auto, self.need_tune) = (true, true),
                Key::P => self.palette = Some((self.palette.unwrap_or(0) + 1) % PALETTES.len()),
                Key::T => self.trails = !self.trails,
                Key::B => self.subframes = if self.subframes == 1 { 4 } else { 1 },
                Key::H => self.show_hud = !self.show_hud,
                Key::Slash | Key::F1 => self.show_help = !self.show_help,
                Key::S => self.screenshot(),
                _ => {}
            }
        }
    }

    fn handle_mouse(&mut self, window: &Window) {
        if let Some((_, dy)) = window.get_scroll_wheel().filter(|(_, dy)| *dy != 0.0) {
            self.zoom *= 1.1f32.powf(dy.signum());
        }
        if !window.get_mouse_down(MouseButton::Left) {
            self.drag = None;
            return;
        }
        let Some((x, y)) = window.get_mouse_pos(MouseMode::Pass) else { return };
        if let (Some((last_x, last_y)), Some(c)) = (self.drag, self.current.as_ref()) {
            let world_per_pixel = 2.0 / (c.program.look.zoom * self.zoom * window.get_size().1.max(1) as f32);
            self.pan.0 -= (x - last_x) * world_per_pixel;
            self.pan.1 += (y - last_y) * world_per_pixel;
        }
        self.drag = Some((x, y));
    }
}

pub fn live(args: &Args) -> Result<()> {
    let path = args.path(0, "kasumi live <form.ks>")?;
    let engine = Engine::new()?;
    let (w, h) = parse_res(args.get("res").unwrap_or("1080p"))?;
    let target = Target::new(&engine.gpu, w, h, args.u32_or("ss", 1)?, Format::Window)?;
    let auto = args.get("mode").is_none_or(|m| m == "auto");
    let (current, error) = match load(&engine, &path, false) {
        Ok(loaded) => (Some(loaded), None),
        Err(err) => (None, Some(format!("{err:#}"))),
    };
    let (window_w, window_h) = if w > 1600 { (w * 2 / 3, h * 2 / 3) } else { (w, h) };
    let options = WindowOptions { resize: true, scale_mode: ScaleMode::AspectRatioStretch, ..WindowOptions::default() };
    let mut window = Window::new(&format!("kasumi - {}", path.display()), window_w as usize, window_h as usize, options)?;
    window.set_target_fps(60);

    let mut app = App {
        last_modified: modified(&path),
        path,
        target,
        buf: vec![0; (w * h) as usize],
        current,
        error,
        reload_started: None,
        reload_ms: None,
        toast: None,
        auto,
        mode: args.get("mode").and_then(Mode::parse).unwrap_or(Mode::Warp),
        timings: Vec::new(),
        need_tune: auto,
        tune_next_frame: false,
        t: 0.0,
        speed: 1.0,
        paused: false,
        show_hud: true,
        show_help: false,
        trails: false,
        subframes: 1,
        points_override: args.get("points").map(parse_count).transpose()?,
        points_scale: 1.0,
        palette: None,
        zoom: 1.0,
        pan: (0.0, 0.0),
        drag: None,
        fps: 60.0,
        rate: 0.0,
        points: 0,
        engine,
    };
    let max_frames = args.get("frames").map(str::parse::<u64>).transpose()?;
    let mut last = Instant::now();
    let mut frames = 0;
    while window.is_open() && !window.is_key_down(Key::Escape) {
        app.reload_if_saved();
        let dt = last.elapsed().as_secs_f64();
        last = Instant::now();
        app.fps = app.fps * 0.9 + (1.0 / dt.max(1e-4)) * 0.1;
        if !app.paused {
            app.t += dt as f32 * app.speed;
        }
        app.render()?;
        app.draw_messages();
        window.update_with_buffer(&app.buf, w as usize, h as usize)?;
        if let Some(started) = app.reload_started.take() {
            let ms = started.elapsed().as_secs_f64() * 1e3;
            app.reload_ms = Some(ms);
            println!("reload: edit -> screen {ms:.0} ms");
        }
        app.need_tune |= std::mem::take(&mut app.tune_next_frame);
        frames += 1;
        if max_frames.is_some_and(|n| frames >= n) {
            if let Some(shot) = args.get("shot") {
                write_png(Path::new(shot), w, h, &to_rgb(&app.buf))?;
            }
            println!("live: {frames} frames, {:.0} fps, {:.1} Gpts/s, mode {}", app.fps, app.rate, app.mode.name());
            break;
        }
        app.handle_keys(&window);
        app.handle_mouse(&window);
    }
    Ok(())
}
