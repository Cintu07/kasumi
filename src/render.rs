//! Draws forms: splat points into a density buffer, then tone map it.

use std::path::Path;
use std::rc::Rc;
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};

use crate::args::*;
use crate::ast::Form;
use crate::codegen::module_source;
use crate::cuda::{Buf, Gpu, Kernel, Module};
use crate::interp::{chain_seed, eval_point, eval_uniforms};
use crate::ir::{lower, Ir};
use crate::look::Look;
use crate::mutate::form_id;
use crate::parser::parse;
use crate::print::print_form;

const TILE: u32 = 64;
const MAX_TILES: u32 = 8192;
const POST_CU: &str = include_str!("kernels/post.cu");

/// How points are added into the density buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// One global atomic per point.
    Naive,
    /// Lanes that hit the same pixel merge first, then one atomic.
    Warp,
    /// Sort points into tiles, then accumulate each tile in shared memory.
    Bin,
}

impl Mode {
    pub const ALL: [Mode; 3] = [Mode::Naive, Mode::Warp, Mode::Bin];

    pub fn name(self) -> &'static str {
        match self {
            Mode::Naive => "naive",
            Mode::Warp => "warp",
            Mode::Bin => "bin",
        }
    }

    pub fn parse(s: &str) -> Option<Mode> {
        Mode::ALL.into_iter().find(|m| m.name() == s)
    }

    pub fn next(self) -> Mode {
        Mode::ALL[(self as usize + 1) % Mode::ALL.len()]
    }
}

/// A form ready to compile and draw.
#[derive(Clone)]
pub struct Program {
    pub form: Form,
    pub text: String,
    pub id: String,
    pub ir: Ir,
    pub look: Look,
    pub points: u64,
    iters: u32,
    warmup: u32,
}

impl Program {
    pub fn new(form: Form) -> Program {
        let text = print_form(&form);
        let ir = lower(&form);
        let attractor = ir.attractor;
        Program {
            id: form_id(&text),
            look: Look::from_form(&form),
            points: form.num_or("points", 20e6) as u64,
            iters: if attractor { form.num_or("iters", 128.0).max(1.0) as u32 } else { 1 },
            warmup: if attractor { form.num_or("warmup", 16.0).max(0.0) as u32 } else { 0 },
            text,
            ir,
            form,
        }
    }

    pub fn load(path: &Path) -> Result<Program> {
        let src = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let form = parse(&src).map_err(|e| anyhow!("{}\n{}", path.display(), e.render(&src)))?;
        Ok(Program::new(form))
    }
}

/// The compiled kernels of one form.
#[derive(Clone)]
pub struct Kernels {
    _module: Rc<Module>,
    splat: Kernel,
    /// Tile-binning count pass, missing when built without it.
    count: Option<Kernel>,
    probe: Kernel,
}

impl Kernels {
    pub fn has_bin(&self) -> bool {
        self.count.is_some()
    }
}

#[derive(Clone, Copy, Default)]
pub struct Build {
    pub with_bin: bool,
    pub fast_math: bool,
    /// No fused multiply-add, so results match the CPU bit for bit.
    pub strict: bool,
}

/// Compile a batch of forms with one NVRTC call.
pub fn compile(gpu: &Gpu, programs: &[&Program], build: Build) -> Result<Vec<Kernels>> {
    let irs: Vec<&Ir> = programs.iter().map(|p| &p.ir).collect();
    let ptx = gpu.compile(&module_source(&irs, build.with_bin), build.fast_math, build.strict)?;
    let module = Rc::new(gpu.load(&ptx)?);
    (0..programs.len())
        .map(|id| {
            Ok(Kernels {
                splat: gpu.kernel(&module, &format!("ks_splat_{id}"))?,
                count: if build.with_bin { Some(gpu.kernel(&module, &format!("ks_count_{id}"))?) } else { None },
                probe: gpu.kernel(&module, &format!("ks_probe_{id}"))?,
                _module: module.clone(),
            })
        })
        .collect()
}

/// Compile one form the way it asks to be compiled.
pub fn compile_one(gpu: &Gpu, p: &Program, with_bin: bool) -> Result<Kernels> {
    let build = Build { with_bin, fast_math: p.form.flag("fastmath"), strict: false };
    Ok(compile(gpu, &[p], build)?.remove(0))
}

/// The GPU and the kernels that do not depend on a form.
pub struct Engine {
    pub gpu: Gpu,
    _module: Module,
    scan: Kernel,
    scatter: Kernel,
    accum: Kernel,
    hist: Kernel,
    tonemap: Kernel,
    decay: Kernel,
    glow_build: Kernel,
    glow_blur: Kernel,
    de_down: Kernel,
    de_resolve: Kernel,
}

impl Engine {
    pub fn new() -> Result<Engine> {
        let gpu = Gpu::new()?;
        let module = gpu.load(&gpu.compile(&(header() + POST_CU), false, false)?)?;
        let k = |name: &str| gpu.kernel(&module, name);
        Ok(Engine {
            scan: k("ks_scan")?,
            scatter: k("ks_scatter")?,
            accum: k("ks_accum")?,
            hist: k("ks_hist")?,
            tonemap: k("ks_tonemap")?,
            decay: k("ks_decay")?,
            glow_build: k("ks_bloom_build")?,
            glow_blur: k("ks_bloom_blur")?,
            de_down: k("ks_de_down")?,
            de_resolve: k("ks_de_resolve")?,
            _module: module,
            gpu,
        })
    }
}

/// What to draw for one frame.
#[derive(Clone, Copy, Debug)]
pub struct Frame {
    pub t: f32,
    /// Frame length in seconds. Motion blur spreads subframes over shutter * dt.
    pub dt: f32,
    pub subframes: u32,
    pub points: u64,
    pub mode: Mode,
    /// How much of the last frame stays: 0 clears, 1 keeps everything.
    pub decay: f32,
}

impl Frame {
    pub fn still(t: f32, points: u64, mode: Mode) -> Frame {
        Frame { t, dt: 1.0 / 60.0, subframes: 1, points, mode, decay: 0.0 }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct FrameStats {
    pub points: u64,
    pub ms: f64,
}

/// The pixel layout of a target's output.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// 0x00RRGGBB, for windows.
    Window = 0,
    /// 16-bit RGB, for video.
    Rgb16 = 1,
    /// 8-bit RGB, for PNG.
    Rgb8 = 2,
}

impl Format {
    fn bytes_per_pixel(self) -> usize {
        match self {
            Format::Window => 4,
            Format::Rgb16 => 6,
            Format::Rgb8 => 3,
        }
    }
}

struct BinBuffers {
    staged_tile: Buf,
    staged_key: Buf,
    staged_weight: Buf,
    keys: Buf,
    weights: Buf,
    tile_count: Buf,
    tile_cursor: Buf,
    tile_offset: Buf,
    slice_prefix: Buf,
    work: Buf,
    /// Points that fit in one pass.
    capacity: u64,
}

/// The mip chain and output of density estimation.
struct DeBuffers {
    levels: Vec<(Buf, Buf, u32, u32)>,
    density: Buf,
    hue: Buf,
}

/// A render target: density and hue at (w * ss) x (h * ss), plus the output image.
pub struct Target {
    pub w: u32,
    pub h: u32,
    ss: u32,
    /// Accumulation size.
    pub aw: u32,
    pub ah: u32,
    density: Buf,
    hue: Buf,
    out: Buf,
    format: Format,
    hist: Buf,
    uniforms: Buf,
    uniforms_capacity: usize,
    bin: Option<BinBuffers>,
    glow: Option<(Buf, Buf)>,
    de: Option<DeBuffers>,
    /// Smoothed reference density, so exposure does not flicker between frames.
    ref_log: Option<f32>,
    /// Pixels with any density in the last tone map.
    pub lit: u64,
}

impl Target {
    pub fn new(gpu: &Gpu, w: u32, h: u32, ss: u32, format: Format) -> Result<Target> {
        let (aw, ah) = (w * ss, h * ss);
        let pixels = aw as usize * ah as usize;
        let target = Target {
            w,
            h,
            ss,
            aw,
            ah,
            density: gpu.alloc(pixels * 4)?,
            hue: gpu.alloc(pixels * 4)?,
            out: gpu.alloc(w as usize * h as usize * format.bytes_per_pixel())?,
            format,
            hist: gpu.alloc(1024 * 4)?,
            uniforms: gpu.alloc(4096 * 4)?,
            uniforms_capacity: 4096,
            bin: None,
            glow: None,
            de: None,
            ref_log: None,
            lit: 0,
        };
        target.clear(gpu)?;
        Ok(target)
    }

    fn tiles(&self) -> (u32, u32) {
        (self.aw.div_ceil(TILE), self.ah.div_ceil(TILE))
    }

    fn clear(&self, gpu: &Gpu) -> Result<()> {
        gpu.zero(&self.density)?;
        gpu.zero(&self.hue)
    }

    pub fn reset_exposure(&mut self) {
        self.ref_log = None;
    }

    fn ensure_bin(&mut self, gpu: &Gpu) -> Result<()> {
        if self.bin.is_some() {
            return Ok(());
        }
        let (tx, ty) = self.tiles();
        let tiles = (tx * ty) as usize;
        if tx * ty > MAX_TILES {
            bail!("bin mode supports up to {MAX_TILES} tiles; {}x{} needs {tiles}", self.aw, self.ah);
        }
        // 20 bytes per point in flight, using at most a quarter of free memory
        let capacity = ((gpu.memory().0 / 4) / 20).clamp(1 << 20, 32 << 20);
        self.bin = Some(BinBuffers {
            staged_tile: gpu.alloc(capacity * 4)?,
            staged_key: gpu.alloc(capacity * 4)?,
            staged_weight: gpu.alloc(capacity * 4)?,
            keys: gpu.alloc(capacity * 4)?,
            weights: gpu.alloc(capacity * 4)?,
            tile_count: gpu.alloc(tiles * 4)?,
            tile_cursor: gpu.alloc(tiles * 4)?,
            tile_offset: gpu.alloc(tiles * 4)?,
            slice_prefix: gpu.alloc((tiles + 1) * 4)?,
            work: gpu.alloc(4)?,
            capacity: capacity as u64,
        });
        Ok(())
    }

    fn splat_args(&self, look: &Look) -> SplatArgs {
        let scale = look.zoom * self.ah as f32 * 0.5;
        let (tx, ty) = self.tiles();
        SplatArgs {
            density: self.density.ptr,
            hue: self.hue.ptr,
            w: self.aw as i32,
            h: self.ah as i32,
            scale_x: scale,
            scale_y: -scale,
            offset_x: self.aw as f32 * 0.5 - look.cx * scale,
            offset_y: self.ah as f32 * 0.5 + look.cy * scale,
            dof: look.dof,
            focus: look.focus,
            tiles_x: tx as i32,
            tiles: (tx * ty) as i32,
            stride: 1,
            ..Default::default()
        }
    }

    /// One row of per-frame values for each motion-blur subframe.
    fn upload_uniforms(&mut self, gpu: &Gpu, p: &Program, look: &Look, frame: &Frame, n: u64) -> Result<usize> {
        let subframes = frame.subframes.max(1);
        let row = p.ir.slots.len().max(1);
        let params = p.ir.param_values();
        let mut table = Vec::with_capacity(row * subframes as usize);
        for s in 0..subframes {
            let offset = if subframes == 1 { 0.0 } else { (s as f32 + 0.5) / subframes as f32 - 0.5 };
            let t = frame.t + offset * look.shutter * frame.dt;
            table.extend_from_slice(&eval_uniforms(&p.ir, t, n as f32, &params).slots);
            table.resize(row * (s as usize + 1), 0.0);
        }
        if table.len() > self.uniforms_capacity {
            self.uniforms = gpu.alloc(table.len() * 4)?;
            self.uniforms_capacity = table.len();
        }
        gpu.upload(&self.uniforms, &table)?;
        Ok(row)
    }

    /// Add one frame, with all its subframes, into the density buffer.
    pub fn accumulate(&mut self, e: &Engine, k: &Kernels, p: &Program, look: &Look, frame: &Frame) -> Result<FrameStats> {
        let gpu = &e.gpu;
        let iters = p.iters.max(1) as u64;
        let chains = (frame.points / iters).max(1);
        let n = chains * iters;
        let subframes = frame.subframes.max(1);
        let row = self.upload_uniforms(gpu, p, look, frame, n)?;

        if frame.decay <= 0.0 {
            self.clear(gpu)?;
        } else if frame.decay < 1.0 {
            let pixels = self.aw as u64 * self.ah as u64;
            let args = DecayArgs { density: self.density.ptr, hue: self.hue.ptr, n: pixels, factor: frame.decay };
            gpu.launch(e.decay, (gpu.sm_count * 8, 1, 1), (256, 1, 1), 0, &args)?;
        }

        let mut args = self.splat_args(look);
        args.uniforms = self.uniforms.ptr;
        args.n_uniforms = row as i32;
        args.inv_n = (1.0f64 / n as f64) as f32;
        args.iters = iters as i32;
        args.warmup = p.warmup as i32;
        args.mode = (frame.mode == Mode::Warp) as i32;

        gpu.sync()?;
        let start = Instant::now();
        if frame.mode == Mode::Bin {
            self.splat_binned(e, k, p, args, chains, subframes, row)?;
        } else {
            args.chains = chains;
            let blocks = chains.div_ceil(256).min(gpu.sm_count as u64 * 48).max(1) as u32;
            gpu.launch(k.splat, (blocks, subframes, 1), (256, 1, 1), 0, &args)?;
        }
        gpu.sync()?;
        Ok(FrameStats { points: n * subframes as u64, ms: start.elapsed().as_secs_f64() * 1e3 })
    }

    /// Tile binning: count points per tile, scan the counts into offsets,
    /// scatter the points into their tiles, then accumulate tile by tile.
    #[allow(clippy::too_many_arguments)]
    fn splat_binned(&mut self, e: &Engine, k: &Kernels, p: &Program, mut args: SplatArgs, chains: u64, subframes: u32, row: usize) -> Result<()> {
        let gpu = &e.gpu;
        let count = k.count.context("this module was built without the tile-binning kernels")?;
        self.ensure_bin(gpu)?;
        let b = self.bin.as_ref().context("bin buffers")?;
        let iters = p.iters.max(1) as u64;
        let tiles = args.tiles as u32;
        let grid = (gpu.sm_count * 6, 1, 1);
        let (has_b, has_h) = (p.ir.has_b() as i32, p.ir.has_h() as i32);
        let bin_args = BinArgs {
            keys: b.keys.ptr,
            weights: b.weights.ptr,
            tile_count: b.tile_count.ptr,
            tile_offset: b.tile_offset.ptr,
            tile_cursor: b.tile_cursor.ptr,
            slice_prefix: b.slice_prefix.ptr,
            work: b.work.ptr,
            density: self.density.ptr,
            hue: self.hue.ptr,
            w: self.aw as i32,
            h: self.ah as i32,
            tiles_x: args.tiles_x,
            tiles: args.tiles,
            has_b,
            has_h,
        };
        let mut scatter_args = ScatterArgs {
            staged_tile: b.staged_tile.ptr,
            staged_key: b.staged_key.ptr,
            staged_weight: b.staged_weight.ptr,
            keys: b.keys.ptr,
            weights: b.weights.ptr,
            tile_cursor: b.tile_cursor.ptr,
            n: 0,
            tiles: args.tiles,
            has_b,
        };
        args.staged_key = b.staged_key.ptr;
        args.staged_weight = b.staged_weight.ptr;
        args.staged_tile = b.staged_tile.ptr;
        args.tile_count = b.tile_count.ptr;
        let chunk = (b.capacity / iters).max(1);
        for s in 0..subframes {
            args.uniforms = self.uniforms.ptr + s as u64 * row as u64 * 4;
            args.subframe = s as i32;
            let mut done = 0;
            while done < chains {
                args.base = done;
                args.chains = chunk.min(chains - done);
                gpu.fill32(&b.tile_count, tiles as usize)?;
                gpu.launch(count, grid, (512, 1, 1), (tiles + TOUCH_COUNT) * 4, &args)?;
                gpu.launch(e.scan, (1, 1, 1), (1024, 1, 1), 0, &bin_args)?;
                scatter_args.n = args.chains * iters;
                gpu.launch(e.scatter, grid, (512, 1, 1), (tiles + 2 * TOUCH_SCATTER) * 4, &scatter_args)?;
                gpu.fill32(&b.work, 1)?;
                gpu.launch(e.accum, (gpu.sm_count * 8, 1, 1), (256, 1, 1), 0, &bin_args)?;
                done += args.chains;
            }
        }
        Ok(())
    }

    /// Blur sparse pixels more than dense ones, if the form asks for it.
    /// Returns the density and hue the tone map should read. The accumulation
    /// itself stays untouched, so trails keep working.
    fn density_estimate(&mut self, e: &Engine, look: &Look, has_h: bool) -> Result<(u64, u64)> {
        if look.de <= 0.0 {
            return Ok((self.density.ptr, self.hue.ptr));
        }
        let gpu = &e.gpu;
        if self.de.is_none() {
            let (mut w, mut h) = (self.aw, self.ah);
            let mut levels = Vec::new();
            for _ in 0..4 {
                (w, h) = (w.div_ceil(2).max(1), h.div_ceil(2).max(1));
                let bytes = w as usize * h as usize * 4;
                levels.push((gpu.alloc(bytes)?, gpu.alloc(bytes)?, w, h));
            }
            let bytes = self.aw as usize * self.ah as usize * 4;
            self.de = Some(DeBuffers { levels, density: gpu.alloc(bytes)?, hue: gpu.alloc(bytes)? });
        }
        let de = self.de.as_ref().context("density estimation buffers")?;
        let mut resolve = ResolveArgs {
            density: self.density.ptr,
            hue: self.hue.ptr,
            out_density: de.density.ptr,
            out_hue: de.hue.ptr,
            w: self.aw as i32,
            h: self.ah as i32,
            max_radius: look.de,
            curve: 0.4,
            has_h: has_h as i32,
            ..Default::default()
        };
        let mut src = (self.density.ptr, self.hue.ptr, self.aw, self.ah);
        for (k, (density, hue, w, h)) in de.levels.iter().enumerate() {
            let down = DownArgs {
                src_density: src.0,
                src_hue: src.1,
                dst_density: density.ptr,
                dst_hue: hue.ptr,
                src_w: src.2 as i32,
                src_h: src.3 as i32,
                dst_w: *w as i32,
                dst_h: *h as i32,
                has_h: has_h as i32,
            };
            gpu.launch(e.de_down, (w.div_ceil(16), h.div_ceil(16), 1), (16, 16, 1), 0, &down)?;
            resolve.level_density[k] = density.ptr;
            resolve.level_hue[k] = hue.ptr;
            resolve.level_w[k] = *w as i32;
            resolve.level_h[k] = *h as i32;
            src = (density.ptr, hue.ptr, *w, *h);
        }
        gpu.launch(e.de_resolve, (self.aw.div_ceil(16), self.ah.div_ceil(16), 1), (16, 16, 1), 0, &resolve)?;
        Ok((de.density.ptr, de.hue.ptr))
    }

    /// The log2 density that maps to full brightness: the 99.9th percentile of
    /// lit pixels, so a few hot pixels cannot darken the frame.
    fn reference_density(&mut self, e: &Engine, density: u64, look: &Look, smooth: bool) -> Result<f32> {
        let gpu = &e.gpu;
        gpu.zero(&self.hist)?;
        let n = self.aw as u64 * self.ah as u64;
        gpu.launch(e.hist, (gpu.sm_count * 4, 1, 1), (256, 1, 1), 0, &HistArgs { density, hist: self.hist.ptr, n })?;
        let mut hist = [0u32; 1024];
        gpu.download(&self.hist, &mut hist)?;
        self.lit = hist.iter().map(|&h| h as u64).sum();
        let skip = (self.lit as f64 * 0.001) as u64;
        let mut seen = 0;
        let mut reference = 1.0;
        for bin in (0..1024).rev() {
            seen += hist[bin] as u64;
            if seen > skip {
                reference = (bin as f32 + 1.0) / 32.0;
                break;
            }
        }
        let reference = (reference - look.exposure.max(0.01).log2()).max(0.5);
        let eased = match (smooth, self.ref_log) {
            (true, Some(prev)) => prev + (reference - prev) * if reference > prev { 0.35 } else { 0.12 },
            _ => reference,
        };
        self.ref_log = Some(eased);
        Ok(eased)
    }

    /// Soft glow: an eighth-resolution bright pass, blurred both ways.
    fn build_glow(&mut self, e: &Engine, look: &Look, density: u64, hue: u64, has_h: bool, inv_log_ref: f32) -> Result<u64> {
        if look.bloom <= 0.0 {
            return Ok(0);
        }
        let gpu = &e.gpu;
        let (w, h) = (self.w.div_ceil(8), self.h.div_ceil(8));
        if self.glow.is_none() {
            let bytes = w as usize * h as usize * 12;
            self.glow = Some((gpu.alloc(bytes)?, gpu.alloc(bytes)?));
        }
        let (blurred, bright) = self.glow.as_ref().context("glow buffers")?;
        let grid = (w.div_ceil(16), h.div_ceil(16), 1);
        let build = GlowArgs {
            density,
            hue,
            out: bright.ptr,
            w: self.aw as i32,
            h: self.ah as i32,
            cell: (8 * self.ss) as i32,
            glow_w: w as i32,
            glow_h: h as i32,
            has_h: has_h as i32,
            inv_log_ref,
            low: look.low,
            high: look.high,
        };
        gpu.launch(e.glow_build, grid, (16, 16, 1), 0, &build)?;
        let blur = |src: u64, dst: u64, dx, dy| BlurArgs { src, dst, w: w as i32, h: h as i32, dx, dy };
        gpu.launch(e.glow_blur, grid, (16, 16, 1), 0, &blur(bright.ptr, blurred.ptr, 1, 0))?;
        gpu.launch(e.glow_blur, grid, (16, 16, 1), 0, &blur(blurred.ptr, bright.ptr, 0, 1))?;
        Ok(bright.ptr)
    }

    /// Turn density into the output image. `smooth` eases exposure between frames.
    pub fn tonemap(&mut self, e: &Engine, look: &Look, has_h: bool, smooth: bool) -> Result<()> {
        let (density, hue) = self.density_estimate(e, look, has_h)?;
        let reference = self.reference_density(e, density, look, smooth)?;
        let inv_log_ref = 1.0 / (reference * std::f32::consts::LN_2);
        let glow = self.build_glow(e, look, density, hue, has_h, inv_log_ref)?;
        let args = ToneArgs {
            density,
            hue,
            out: self.out.ptr,
            w: self.aw as i32,
            h: self.ah as i32,
            ss: self.ss as i32,
            format: self.format as i32,
            inv_log_ref,
            gamma_inv: 1.0 / look.gamma.max(0.1),
            vibrancy: look.vibrancy,
            heat: look.heat,
            low: look.low,
            high: look.high,
            has_h: has_h as i32,
            glow,
            glow_w: self.w.div_ceil(8) as i32,
            glow_h: self.h.div_ceil(8) as i32,
            glow_gain: look.bloom,
        };
        e.gpu.launch(e.tonemap, (self.w.div_ceil(16), self.h.div_ceil(16), 1), (16, 16, 1), 0, &args)
    }

    pub fn read_pixels(&self, gpu: &Gpu, out: &mut Vec<u32>) -> Result<()> {
        assert!(self.format == Format::Window);
        out.resize(self.w as usize * self.h as usize, 0);
        gpu.download(&self.out, out)
    }

    pub fn read_bytes(&self, gpu: &Gpu, out: &mut Vec<u8>) -> Result<()> {
        out.resize(self.w as usize * self.h as usize * self.format.bytes_per_pixel(), 0);
        gpu.download(&self.out, out)
    }

    pub fn read_density(&self, gpu: &Gpu) -> Result<Vec<f32>> {
        let mut v = vec![0f32; self.aw as usize * self.ah as usize];
        gpu.download(&self.density, &mut v)?;
        Ok(v)
    }
}

/// Time every accumulation mode on this form and return the fastest, with all timings.
pub fn fastest_mode(e: &Engine, target: &mut Target, k: &Kernels, p: &Program, look: &Look, points: u64) -> (Mode, Vec<(Mode, f64)>) {
    let mut timings = Vec::new();
    for mode in Mode::ALL {
        let frame = Frame::still(1.0, points, mode);
        let runs = (0..3).map_while(|_| target.accumulate(e, k, p, look, &frame).ok());
        timings.push((mode, runs.map(|s| s.ms).fold(f64::MAX, f64::min)));
    }
    let best = timings.iter().fold((Mode::Warp, f64::MAX), |best, t| if t.1 < best.1 { *t } else { best });
    (best.0, timings)
}

/// The result of comparing GPU points with the CPU interpreter.
#[derive(Clone, Copy, Debug, Default)]
pub struct CheckReport {
    pub samples: u32,
    pub matched: u32,
    pub max_err: f32,
    pub worst_i: u32,
}

impl CheckReport {
    pub fn fraction(&self) -> f64 {
        self.matched as f64 / self.samples.max(1) as f64
    }

    pub fn ok(&self) -> bool {
        self.fraction() >= 0.995
    }
}

fn relative_error(cpu: f32, gpu: f32) -> f32 {
    if cpu.is_nan() && gpu.is_nan() {
        0.0
    } else if !cpu.is_finite() || !gpu.is_finite() {
        if cpu == gpu { 0.0 } else { f32::INFINITY }
    } else {
        (gpu - cpu).abs() / (1.0 + cpu.abs())
    }
}

/// Evaluate `samples` points on the GPU and on the CPU, and compare all five outputs.
pub fn check(gpu: &Gpu, k: &Kernels, p: &Program, points: u64, t: f32, samples: u32) -> Result<CheckReport> {
    let iters = p.iters.max(1) as u64;
    let n = (points / iters).max(1) * iters;
    let uniforms = eval_uniforms(&p.ir, t, n as f32, &p.ir.param_values());
    let mut slots = uniforms.slots.clone();
    slots.resize(slots.len().max(1), 0.0);
    let slots_buf = gpu.alloc(slots.len() * 4)?;
    gpu.upload(&slots_buf, &slots)?;
    let out = gpu.alloc(samples as usize * 5 * 4)?;
    let stride = (n / samples as u64).max(1);
    let inv_n = (1.0f64 / n as f64) as f32;
    let args = SplatArgs {
        uniforms: slots_buf.ptr,
        probe: out.ptr,
        chains: samples as u64,
        stride,
        inv_n,
        n_uniforms: slots.len() as i32,
        ..Default::default()
    };
    gpu.launch(k.probe, (samples.div_ceil(256), 1, 1), (256, 1, 1), 0, &args)?;
    let mut on_gpu = vec![0f32; samples as usize * 5];
    gpu.download(&out, &mut on_gpu)?;

    let mut values = Vec::new();
    let mut report = CheckReport { samples, ..Default::default() };
    for (s, row) in on_gpu.chunks_exact(5).enumerate() {
        let i = (s as u64 * stride) as u32;
        let (px, py) = chain_seed(i);
        let c = eval_point(&p.ir, &uniforms, i, inv_n, px, py, &mut values);
        let worst = [c.x, c.y, c.z, c.b, c.h].iter().zip(row).map(|(a, b)| relative_error(*a, *b)).fold(0.0, f32::max);
        report.matched += (worst <= 1e-3) as u32;
        if worst > report.max_err {
            (report.max_err, report.worst_i) = (worst, i);
        }
    }
    Ok(report)
}
