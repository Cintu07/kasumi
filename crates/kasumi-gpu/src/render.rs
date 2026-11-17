use std::time::Instant;

use anyhow::{bail, Result};
use kasumi_lang::codegen::{kernel_names, module_source};
use kasumi_lang::interp::{chain_seed, eval_point, eval_uniforms};
use kasumi_lang::{Form, Ir};

use crate::device::{Buf, CompileOpts, Func, Gpu, Module};
use crate::kernels::STATIC_SRC;

pub const TILE: u32 = 64;
pub const MAX_TILES: u32 = 8192;
const TOUCH_C: u32 = 2048;
const TOUCH_S: u32 = 1024;

/// How points are added into the image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// one global atomic per point
    Naive,
    /// same-pixel lanes merged with __match_any_sync, then one atomic
    Warp,
    /// tile binning: count, scan, scatter packed keys, shared-memory tiles
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
        Mode::ALL.iter().copied().find(|m| m.name() == s)
    }
    pub fn next(self) -> Mode {
        match self {
            Mode::Naive => Mode::Warp,
            Mode::Warp => Mode::Bin,
            Mode::Bin => Mode::Naive,
        }
    }
}

// ---- launch parameter structs, mirrored in cuda ----

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct KP {
    pub u: u64,
    pub acc_c: u64,
    pub acc_h: u64,
    pub pre_k: u64,
    pub pre_w: u64,
    pub tile_count: u64,
    pub pre_t: u64,
    pub probe: u64,
    pub base: u64,
    pub chains: u64,
    pub key_cap: u64,
    pub stride: u64,
    pub inv_n: f32,
    pub nu: i32,
    pub w: i32,
    pub h: i32,
    pub sx: f32,
    pub sy: f32,
    pub ox: f32,
    pub oy: f32,
    pub dof: f32,
    pub focus: f32,
    pub iters: i32,
    pub warmup: i32,
    pub tiles_x: i32,
    pub tiles: i32,
    pub s_base: i32,
    pub mode: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct AP {
    keys: u64,
    wts: u64,
    tile_count: u64,
    tile_offset: u64,
    tile_cursor: u64,
    slice_prefix: u64,
    work: u64,
    acc_c: u64,
    acc_h: u64,
    w: i32,
    h: i32,
    tiles_x: i32,
    tiles: i32,
    has_b: i32,
    has_h: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct SP {
    pre_t: u64,
    pre_k: u64,
    pre_w: u64,
    keys: u64,
    wts: u64,
    tile_cursor: u64,
    n: u64,
    tiles: i32,
    has_b: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct HP {
    acc: u64,
    hist: u64,
    n: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct TP {
    acc_c: u64,
    acc_h: u64,
    out: u64,
    w: i32,
    h: i32,
    ss: i32,
    fmt: i32,
    inv_logref: f32,
    gamma_inv: f32,
    vib: f32,
    heat: f32,
    a: [f32; 3],
    b: [f32; 3],
    has_h: i32,
    pad: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct DP {
    a: u64,
    b: u64,
    n: u64,
    f: f32,
    has_b: i32,
}

/// Form-independent kernels, compiled once.
pub struct Statics {
    _m: Module,
    scan: Func,
    scatter: Func,
    accum: Func,
    hist: Func,
    tonemap: Func,
    decay: Func,
}

impl Statics {
    pub fn new(gpu: &Gpu) -> Result<Statics> {
        let c = gpu.compile_with(STATIC_SRC, CompileOpts::default(), true)?;
        let m = gpu.load(&c.image)?;
        Ok(Statics {
            scan: gpu.func(&m, "ks_scan")?,
            scatter: gpu.func(&m, "ks_scatter")?,
            accum: gpu.func(&m, "ks_accum")?,
            hist: gpu.func(&m, "ks_hist")?,
            tonemap: gpu.func(&m, "ks_tonemap")?,
            decay: gpu.func(&m, "ks_decay")?,
            _m: m,
        })
    }
}

#[derive(Clone, Copy)]
pub struct UnitFns {
    pub splat: Func,
    /// tile-binning count kernel; None when built without bin support
    pub count: Option<Func>,
    pub probe: Func,
}

/// One or more compiled forms.
pub struct FormModule {
    _m: Module,
    pub units: Vec<UnitFns>,
    pub compile_ms: f64,
    pub source: String,
    pub with_bin: bool,
}

impl FormModule {
    pub fn build(gpu: &Gpu, irs: &[&Ir], opts: CompileOpts) -> Result<FormModule> {
        Self::build_with(gpu, irs, opts, true)
    }

    /// `with_bin = false` leaves out the tile-binning kernels (faster compile).
    pub fn build_with(gpu: &Gpu, irs: &[&Ir], opts: CompileOpts, with_bin: bool) -> Result<FormModule> {
        let source = module_source(irs, with_bin);
        // nvrtc -> ptx is ~13 ms; the driver jit (ptx -> sass) is 2-3x
        // faster than nvrtc's own cubin path, so hot reload goes via ptx
        let t0 = std::time::Instant::now();
        let c = gpu.compile_with(&source, opts, true)?;
        let t1 = std::time::Instant::now();
        let m = gpu.load(&c.image)?;
        let compile_ms = t0.elapsed().as_secs_f64() * 1e3;
        if std::env::var_os("KASUMI_PROF").is_some() {
            eprintln!("  nvrtc {:.1} ms, driver jit {:.1} ms", c.ms, t1.elapsed().as_secs_f64() * 1e3);
        }
        let mut units = Vec::new();
        for id in 0..irs.len() {
            let [s, c2, p] = kernel_names(id);
            let count = if with_bin { Some(gpu.func(&m, &c2)?) } else { None };
            units.push(UnitFns { splat: gpu.func(&m, &s)?, count, probe: gpu.func(&m, &p)? });
        }
        Ok(FormModule { _m: m, units, compile_ms, source, with_bin })
    }
}

/// Look settings, read from a form's @settings.
#[derive(Clone, Debug)]
pub struct Look {
    pub palette: String,
    pub a: [f32; 3],
    pub b: [f32; 3],
    pub gamma: f32,
    pub vibrancy: f32,
    pub heat: f32,
    pub exposure: f32,
    pub zoom: f32,
    pub cx: f32,
    pub cy: f32,
    pub dof: f32,
    pub focus: f32,
    pub trail: f32,
    pub shutter: f32,
}

pub const PALETTES: &[(&str, [f32; 3], [f32; 3])] = &[
    ("mono", [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]),
    ("ember", [1.0, 0.32, 0.08], [1.0, 0.82, 0.45]),
    ("ice", [0.22, 0.42, 1.0], [0.75, 0.95, 1.0]),
    ("sakura", [0.95, 0.38, 0.62], [1.0, 0.9, 0.93]),
    ("moss", [0.3, 0.72, 0.4], [0.95, 0.95, 0.7]),
    ("ink", [0.32, 0.36, 0.95], [0.98, 0.92, 0.84]),
    ("dusk", [0.55, 0.28, 0.95], [1.0, 0.6, 0.38]),
    ("jade", [0.08, 0.75, 0.65], [0.9, 1.0, 0.95]),
];

pub fn palette(name: &str) -> Option<([f32; 3], [f32; 3])> {
    PALETTES.iter().find(|p| p.0 == name).map(|p| (p.1, p.2))
}

impl Look {
    pub fn from_form(f: &Form) -> Look {
        let pal = f.word("palette").unwrap_or("mono").to_string();
        let (a, b) = palette(&pal).unwrap_or(([1.0; 3], [1.0; 3]));
        Look {
            palette: pal,
            a,
            b,
            gamma: f.num_or("gamma", 2.2) as f32,
            vibrancy: f.num_or("vibrancy", 1.0) as f32,
            heat: f.num_or("heat", 0.2) as f32,
            exposure: f.num_or("exposure", 1.0) as f32,
            zoom: f.num_or("zoom", 1.0) as f32,
            cx: f.num_or("cx", 0.0) as f32,
            cy: f.num_or("cy", 0.0) as f32,
            dof: f.num_or("dof", 0.0) as f32,
            focus: f.num_or("focus", 0.0) as f32,
            trail: f.num_or("trail", 0.0) as f32,
            shutter: f.num_or("shutter", 0.5) as f32,
        }
    }
    pub fn set_palette(&mut self, name: &str) {
        if let Some((a, b)) = palette(name) {
            self.palette = name.to_string();
            self.a = a;
            self.b = b;
        }
    }
}

/// A form ready to render: its ir, which unit of a module it is, and tunables.
#[derive(Clone)]
pub struct Program {
    pub ir: Ir,
    pub params: Vec<f32>,
    pub iters: u32,
    pub warmup: u32,
}

impl Program {
    pub fn new(form: &Form, ir: Ir) -> Program {
        let attractor = ir.attractor;
        let params = ir.param_values();
        Program {
            ir,
            params,
            iters: if attractor { form.num_or("iters", 128.0).max(1.0) as u32 } else { 1 },
            warmup: if attractor { form.num_or("warmup", 16.0).max(0.0) as u32 } else { 0 },
        }
    }
}

/// What to render for one frame.
#[derive(Clone, Copy, Debug)]
pub struct FrameReq {
    pub t: f32,
    /// frame duration in seconds; motion blur spreads subframes over shutter * dt
    pub dt: f32,
    pub subframes: u32,
    pub points: u64,
    pub mode: Mode,
    /// decay factor applied before accumulating (0 = clear)
    pub decay: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct FrameStats {
    pub points: u64,
    pub ms: f64,
}

struct BinScratch {
    pre_t: Buf,
    pre_k: Buf,
    pre_w: Buf,
    keys: Buf,
    wts: Buf,
    tile_count: Buf,
    tile_cursor: Buf,
    tile_offset: Buf,
    slice_prefix: Buf,
    work: Buf,
    cap: u64,
}

/// A render target: accumulation buffers at (w*ss) x (h*ss) plus output.
pub struct Target {
    pub w: u32,
    pub h: u32,
    pub ss: u32,
    pub aw: u32,
    pub ah: u32,
    pub acc_c: Buf,
    pub acc_h: Buf,
    out: Buf,
    out_fmt: i32,
    hist: Buf,
    uni: Buf,
    uni_cap: usize,
    bin: Option<BinScratch>,
    /// smoothed reference density (log domain) for stable exposure
    ref_log: Option<f32>,
    pub lit: u64,
}

impl Target {
    /// `fmt`: 0 = 0x00RRGGBB u32 (windows), 1 = rgb48 (video), 2 = rgb24 (png)
    pub fn new(gpu: &Gpu, w: u32, h: u32, ss: u32, fmt: i32) -> Result<Target> {
        let (aw, ah) = (w * ss, h * ss);
        let px = aw as usize * ah as usize;
        let out_bytes = (w as usize * h as usize) * if fmt == 0 { 4 } else if fmt == 1 { 6 } else { 3 };
        let t = Target {
            w,
            h,
            ss,
            aw,
            ah,
            acc_c: gpu.alloc(px * 4)?,
            acc_h: gpu.alloc(px * 4)?,
            out: gpu.alloc(out_bytes)?,
            out_fmt: fmt,
            hist: gpu.alloc(1024 * 4)?,
            uni: gpu.alloc(4096 * 4)?,
            uni_cap: 4096,
            bin: None,
            ref_log: None,
            lit: 0,
        };
        gpu.zero(&t.acc_c)?;
        gpu.zero(&t.acc_h)?;
        Ok(t)
    }

    pub fn tiles(&self) -> (u32, u32) {
        (self.aw.div_ceil(TILE), self.ah.div_ceil(TILE))
    }

    pub fn clear(&mut self, gpu: &Gpu) -> Result<()> {
        gpu.zero(&self.acc_c)?;
        gpu.zero(&self.acc_h)
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
        if tiles as u32 > MAX_TILES {
            bail!("bin mode supports up to {MAX_TILES} tiles; {}x{} needs {tiles}", self.aw, self.ah);
        }
        // 20 bytes per point in flight; up to 32m points per chunk using at
        // most ~1/4 of free memory
        let (free, _) = gpu.mem_info();
        let cap = ((free / 4) / 20).clamp(1 << 20, 32 << 20) as u64;
        self.bin = Some(BinScratch {
            pre_t: gpu.alloc(cap as usize * 4)?,
            pre_k: gpu.alloc(cap as usize * 4)?,
            pre_w: gpu.alloc(cap as usize * 4)?,
            keys: gpu.alloc(cap as usize * 4)?,
            wts: gpu.alloc(cap as usize * 4)?,
            tile_count: gpu.alloc(tiles * 4)?,
            tile_cursor: gpu.alloc(tiles * 4)?,
            tile_offset: gpu.alloc(tiles * 4)?,
            slice_prefix: gpu.alloc((tiles + 1) * 4)?,
            work: gpu.alloc(4)?,
            cap,
        });
        Ok(())
    }

    fn base_kp(&self, look: &Look) -> KP {
        let scale = look.zoom * self.ah as f32 * 0.5;
        let (tx, ty) = self.tiles();
        KP {
            acc_c: self.acc_c.ptr,
            acc_h: self.acc_h.ptr,
            w: self.aw as i32,
            h: self.ah as i32,
            sx: scale,
            sy: -scale,
            ox: self.aw as f32 * 0.5 - look.cx * scale,
            oy: self.ah as f32 * 0.5 + look.cy * scale,
            dof: look.dof,
            focus: look.focus,
            tiles_x: tx as i32,
            tiles: (tx * ty) as i32,
            stride: 1,
            ..Default::default()
        }
    }

    /// Add one frame (all motion-blur subframes) into the accumulation buffer.
    pub fn accumulate(&mut self, gpu: &Gpu, st: &Statics, unit: &UnitFns, prog: &Program, look: &Look, req: &FrameReq) -> Result<FrameStats> {
        let ir = &prog.ir;
        let iters = prog.iters.max(1) as u64;
        let chains = (req.points / iters).max(1);
        let n = chains * iters;
        let subs = req.subframes.max(1);
        let nu = ir.slots.len().max(1);

        // uniforms: one row per subframe, t spread over the shutter
        let mut table = Vec::with_capacity(nu * subs as usize);
        for s in 0..subs {
            let frac = if subs == 1 { 0.0 } else { (s as f32 + 0.5) / subs as f32 - 0.5 };
            let t = req.t + frac * look.shutter * req.dt;
            let u = eval_uniforms(ir, t, n as f32, &prog.params);
            table.extend_from_slice(&u.slots);
            table.resize(nu * (s as usize + 1), 0.0);
        }
        if table.len() > self.uni_cap {
            self.uni = gpu.alloc(table.len() * 4)?;
            self.uni_cap = table.len();
        }
        gpu.upload(&self.uni, &table)?;

        if req.decay <= 0.0 {
            self.clear(gpu)?;
        } else if req.decay < 1.0 {
            let px = self.aw as u64 * self.ah as u64;
            let p = DP { a: self.acc_c.ptr, b: self.acc_h.ptr, n: px, f: req.decay, has_b: 1 };
            gpu.launch(st.decay, (gpu.sm_count * 8, 1, 1), (256, 1, 1), 0, &p)?;
        }

        let mut kp = self.base_kp(look);
        kp.u = self.uni.ptr;
        kp.nu = nu as i32;
        kp.inv_n = (1.0f64 / n as f64) as f32;
        kp.iters = iters as i32;
        kp.warmup = prog.warmup as i32;
        kp.mode = if req.mode == Mode::Warp { 1 } else { 0 };

        gpu.sync()?;
        let t0 = Instant::now();
        match req.mode {
            Mode::Naive | Mode::Warp => {
                kp.base = 0;
                kp.chains = chains;
                let blocks = chains.div_ceil(256).min(gpu.sm_count as u64 * 48).max(1) as u32;
                gpu.launch(unit.splat, (blocks, subs, 1), (256, 1, 1), 0, &kp)?;
            }
            Mode::Bin => {
                let Some(count_fn) = unit.count else { bail!("this module was built without the tile-binning kernels") };
                self.ensure_bin(gpu)?;
                let b = self.bin.as_ref().unwrap();
                let tiles = kp.tiles as usize;
                let chunk = (b.cap / iters).max(1);
                let grid = gpu.sm_count * 6;
                let smem_c = (tiles as u32 + TOUCH_C) * 4;
                let smem_s = (tiles as u32 + 2 * TOUCH_S) * 4;
                let ap = AP {
                    keys: b.keys.ptr,
                    wts: b.wts.ptr,
                    tile_count: b.tile_count.ptr,
                    tile_offset: b.tile_offset.ptr,
                    tile_cursor: b.tile_cursor.ptr,
                    slice_prefix: b.slice_prefix.ptr,
                    work: b.work.ptr,
                    acc_c: self.acc_c.ptr,
                    acc_h: self.acc_h.ptr,
                    w: self.aw as i32,
                    h: self.ah as i32,
                    tiles_x: kp.tiles_x,
                    tiles: kp.tiles,
                    has_b: ir.has_b() as i32,
                    has_h: ir.has_h() as i32,
                };
                kp.pre_k = b.pre_k.ptr;
                kp.pre_w = b.pre_w.ptr;
                kp.pre_t = b.pre_t.ptr;
                kp.tile_count = b.tile_count.ptr;
                kp.key_cap = b.cap;
                let mut sp = SP {
                    pre_t: b.pre_t.ptr,
                    pre_k: b.pre_k.ptr,
                    pre_w: b.pre_w.ptr,
                    keys: b.keys.ptr,
                    wts: b.wts.ptr,
                    tile_cursor: b.tile_cursor.ptr,
                    n: 0,
                    tiles: kp.tiles,
                    has_b: ir.has_b() as i32,
                };
                let prof = std::env::var_os("KASUMI_PROF").is_some();
                let mut stage = [0f64; 4];
                let mut mark = Instant::now();
                let mut lap = |k: usize, gpu: &Gpu| -> Result<()> {
                    if prof {
                        gpu.sync()?;
                        stage[k] += mark.elapsed().as_secs_f64() * 1e3;
                        mark = Instant::now();
                    }
                    Ok(())
                };
                for s in 0..subs {
                    kp.u = self.uni.ptr + (s as u64 * nu as u64 * 4);
                    kp.s_base = s as i32;
                    let mut c0 = 0u64;
                    while c0 < chains {
                        kp.base = c0;
                        kp.chains = chunk.min(chains - c0);
                        gpu.fill32(b.tile_count.ptr, 0, tiles)?;
                        gpu.launch(count_fn, (grid, 1, 1), (512, 1, 1), smem_c, &kp)?;
                        lap(0, gpu)?;
                        gpu.launch(st.scan, (1, 1, 1), (1024, 1, 1), 0, &ap)?;
                        lap(1, gpu)?;
                        sp.n = kp.chains * iters;
                        gpu.launch(st.scatter, (grid, 1, 1), (512, 1, 1), smem_s, &sp)?;
                        lap(2, gpu)?;
                        gpu.fill32(b.work.ptr, 0, 1)?;
                        gpu.launch(st.accum, (gpu.sm_count * 8, 1, 1), (256, 1, 1), 0, &ap)?;
                        lap(3, gpu)?;
                        c0 += kp.chains;
                    }
                }
                if prof {
                    eprintln!("bin stages ms: count {:.2} scan {:.2} scatter {:.2} accum {:.2}", stage[0], stage[1], stage[2], stage[3]);
                }
            }
        }
        gpu.sync()?;
        Ok(FrameStats { points: n * subs as u64, ms: t0.elapsed().as_secs_f64() * 1e3 })
    }

    /// Histogram -> reference density -> tone map into the output buffer.
    /// `smooth` eases exposure changes between frames (live and video).
    pub fn tonemap(&mut self, gpu: &Gpu, st: &Statics, look: &Look, has_h: bool, smooth: bool) -> Result<()> {
        gpu.zero(&self.hist)?;
        let n = self.aw as u64 * self.ah as u64;
        gpu.launch(st.hist, (gpu.sm_count * 4, 1, 1), (256, 1, 1), 0, &HP { acc: self.acc_c.ptr, hist: self.hist.ptr, n })?;
        let mut hist = [0u32; 1024];
        gpu.download(&self.hist, &mut hist)?;
        let lit: u64 = hist.iter().map(|&h| h as u64).sum();
        self.lit = lit;
        // reference = 99.9th percentile density among lit pixels, so a few
        // hot pixels can't darken the whole frame
        let mut ref_log = 1.0f32;
        if lit > 0 {
            let skip = (lit as f64 * 0.001) as u64;
            let mut acc = 0u64;
            for b in (0..1024).rev() {
                acc += hist[b] as u64;
                if acc > skip {
                    ref_log = (b as f32 + 1.0) / 32.0; // log2(1 + density)
                    break;
                }
            }
        }
        ref_log = (ref_log - look.exposure.max(0.01).log2()).max(0.5);
        let r = match (smooth, self.ref_log) {
            (true, Some(prev)) => {
                let k = if ref_log > prev { 0.35 } else { 0.12 };
                prev + (ref_log - prev) * k
            }
            _ => ref_log,
        };
        self.ref_log = Some(r);
        let inv_logref = 1.0 / (r * std::f32::consts::LN_2); // 1 / ln(1 + ref)
        let p = TP {
            acc_c: self.acc_c.ptr,
            acc_h: self.acc_h.ptr,
            out: self.out.ptr,
            w: self.aw as i32,
            h: self.ah as i32,
            ss: self.ss as i32,
            fmt: self.out_fmt,
            inv_logref,
            gamma_inv: 1.0 / look.gamma.max(0.1),
            vib: look.vibrancy,
            heat: look.heat,
            a: look.a,
            b: look.b,
            has_h: has_h as i32,
            pad: 0,
        };
        gpu.launch(st.tonemap, (self.w.div_ceil(16), self.h.div_ceil(16), 1), (16, 16, 1), 0, &p)
    }

    pub fn read_u32(&self, gpu: &Gpu, out: &mut Vec<u32>) -> Result<()> {
        assert_eq!(self.out_fmt, 0);
        out.resize(self.w as usize * self.h as usize, 0);
        gpu.download(&self.out, out)
    }

    pub fn read_bytes(&self, gpu: &Gpu, out: &mut Vec<u8>) -> Result<()> {
        let per = match self.out_fmt {
            0 => 4,
            1 => 6,
            _ => 3,
        };
        out.resize(self.w as usize * self.h as usize * per, 0);
        gpu.download(&self.out, out)
    }

    pub fn read_density(&self, gpu: &Gpu) -> Result<Vec<f32>> {
        let mut v = vec![0f32; self.aw as usize * self.ah as usize];
        gpu.download(&self.acc_c, &mut v)?;
        Ok(v)
    }
}

/// Result of comparing gpu-evaluated points with the cpu interpreter.
#[derive(Clone, Copy, Debug, Default)]
pub struct CheckReport {
    pub samples: u32,
    pub matched: u32,
    pub max_err: f32,
    pub worst_i: u32,
}

impl CheckReport {
    pub fn frac(&self) -> f64 {
        self.matched as f64 / self.samples.max(1) as f64
    }
    pub fn ok(&self) -> bool {
        self.frac() >= 0.995
    }
}

/// Evaluate `samples` points spread across the frame on the gpu and on the
/// cpu reference interpreter, and compare all five outputs.
pub fn check(gpu: &Gpu, unit: &UnitFns, prog: &Program, points: u64, t: f32, samples: u32) -> Result<CheckReport> {
    let ir = &prog.ir;
    let iters = prog.iters.max(1) as u64;
    let n = (points / iters).max(1) * iters;
    let uni = eval_uniforms(ir, t, n as f32, &prog.params);
    let mut slots = uni.slots.clone();
    slots.resize(slots.len().max(1), 0.0);
    let ubuf = gpu.alloc(slots.len() * 4)?;
    gpu.upload(&ubuf, &slots)?;
    let out = gpu.alloc(samples as usize * 5 * 4)?;
    let stride = (n / samples as u64).max(1);
    let inv_n = (1.0f64 / n as f64) as f32;
    let kp = KP { u: ubuf.ptr, probe: out.ptr, base: 0, chains: samples as u64, stride, inv_n, nu: slots.len() as i32, ..Default::default() };
    gpu.launch(unit.probe, ((samples).div_ceil(256), 1, 1), (256, 1, 1), 0, &kp)?;
    let mut g = vec![0f32; samples as usize * 5];
    gpu.download(&out, &mut g)?;
    let mut scratch = Vec::new();
    let mut rep = CheckReport { samples, ..Default::default() };
    for k in 0..samples as usize {
        let i = (k as u64 * stride) as u32;
        let (px, py) = chain_seed(i);
        let c = eval_point(ir, &uni, i, inv_n, px, py, &mut scratch);
        let cpu = [c.x, c.y, c.z, c.b, c.h];
        let mut worst = 0f32;
        for (j, cv) in cpu.iter().enumerate() {
            let gv = g[k * 5 + j];
            let e = if cv.is_nan() && gv.is_nan() {
                0.0
            } else if !cv.is_finite() || !gv.is_finite() {
                if cv == &gv { 0.0 } else { f32::INFINITY }
            } else {
                (gv - cv).abs() / (1.0 + cv.abs())
            };
            worst = worst.max(e);
        }
        if worst <= 1e-3 {
            rep.matched += 1;
        }
        if worst > rep.max_err {
            rep.max_err = worst;
            rep.worst_i = i;
        }
    }
    Ok(rep)
}
