//! CUDA C code generation. One form becomes one fused device function plus
//! four kernels (splat, count, scatter, probe). Several forms can share one
//! module (the explorer compiles 16-40 forms with a single nvrtc call).

use crate::ast::BinOp;
use crate::func::Func;
use crate::ir::{Input, Ir, Op, V};

/// Shared by every form module: helpers mirrored exactly by `interp`.
pub const PRELUDE: &str = r#"
// ---------------- kasumi prelude ----------------
typedef unsigned int u32;
typedef unsigned long long u64;
#define KS_FULL 0xffffffffu
#define KS_HQ_MAX 1048575.0f
#define KS_TOUCH_C 2048

// Launch parameters. Mirrored by `KP` in kasumi-gpu (repr(C)).
struct KP {
  const float* U;
  float* acc_c;
  float* acc_h;
  u32* pre_k;        // bin: per-point packed key (pixel in tile | hue)
  float* pre_w;      // bin: per-point weight (forms with b)
  u32* tile_count;
  u32* pre_t;        // bin: per-point tile id, 0xffffffff = dropped
  float* probe;
  u64 base;
  u64 chains;
  u64 key_cap;
  u64 stride;
  float inv_n;
  int nu;
  int W, H;
  float sx, sy, ox, oy;
  float dof, focus;
  int iters, warmup;
  int tiles_x, tiles;
  int s_base;
  int mode;
};

__device__ __forceinline__ u32 ks_hash(u32 x) {
  x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x;
}
__device__ __forceinline__ float ks_u01(u32 x) { return (float)(ks_hash(x) >> 8) * (1.0f / 16777216.0f); }
__device__ __forceinline__ float ks_fract(float x) { return x - floorf(x); }
__device__ __forceinline__ float ks_mod(float a, float b) { return a - b * floorf(a / b); }
__device__ __forceinline__ float ks_sign(float x) { return x > 0.f ? 1.f : (x < 0.f ? -1.f : 0.f); }
__device__ __forceinline__ float ks_mix(float a, float b, float t) { return a + (b - a) * t; }
__device__ __forceinline__ float ks_clamp(float x, float lo, float hi) { return fminf(fmaxf(x, lo), hi); }
__device__ __forceinline__ float ks_smooth(float e0, float e1, float x) {
  float t = ks_clamp((x - e0) / (e1 - e0), 0.f, 1.f); return t * t * (3.f - 2.f * t);
}
__device__ __forceinline__ float ks_mag(float a, float b) { return sqrtf(a * a + b * b); }
__device__ __forceinline__ float ks_rnd(float x) { return ks_u01(__float_as_uint(x)); }
__device__ __forceinline__ float ks_grad(u32 h, float x, float y) {
  switch (h & 7u) {
    case 0: return x + y;  case 1: return -x + y; case 2: return x - y; case 3: return -x - y;
    case 4: return x;      case 5: return -x;     case 6: return y;     default: return -y;
  }
}
__device__ __forceinline__ u32 ks_corner(int i, int j) { return ks_hash((u32)i * 73856093u ^ (u32)j * 19349663u); }
__device__ float ks_noise(float x, float y) {
  const float F2 = 0.36602542f, G2 = 0.21132487f;
  float s = (x + y) * F2;
  float fi = floorf(x + s), fj = floorf(y + s);
  float t = (fi + fj) * G2;
  float x0 = x - (fi - t), y0 = y - (fj - t);
  float i1 = x0 > y0 ? 1.f : 0.f, j1 = 1.f - i1;
  float x1 = x0 - i1 + G2, y1 = y0 - j1 + G2;
  float x2 = x0 - 1.f + 2.f * G2, y2 = y0 - 1.f + 2.f * G2;
  int ii = (int)fi, jj = (int)fj;
  u32 h0 = ks_corner(ii, jj), h1 = ks_corner(ii + (int)i1, jj + (int)j1), h2 = ks_corner(ii + 1, jj + 1);
  float n = 0.f, tt;
  tt = 0.5f - x0 * x0 - y0 * y0; if (tt > 0.f) { tt *= tt; n += tt * tt * ks_grad(h0, x0, y0); }
  tt = 0.5f - x1 * x1 - y1 * y1; if (tt > 0.f) { tt *= tt; n += tt * tt * ks_grad(h1, x1, y1); }
  tt = 0.5f - x2 * x2 - y2 * y2; if (tt > 0.f) { tt *= tt; n += tt * tt * ks_grad(h2, x2, y2); }
  return 70.f * n;
}

__device__ __forceinline__ void ks_seed(u32 c, float& px, float& py) {
  px = ks_u01(c ^ 0x2C1B3C6Du) * 2.f - 1.f;
  py = ks_u01(c ^ 0x297A2D39u) * 2.f - 1.f;
}

// world -> accumulation pixel, with optional depth-of-field jitter
__device__ __forceinline__ void ks_project(const KP& P, float x, float y, float z, u32 i, u32 s, int& ix, int& iy) {
  if (P.dof > 0.f) {
    float r = P.dof * fabsf(z - P.focus);
    u32 hs = ks_hash(i * 0x9E3779B9u + s * 0x85EBCA6Bu + 0x01234567u);
    float a = (float)(hs & 0xffffu) * (6.2831853f / 65536.f);
    float rr = r * sqrtf((float)(hs >> 16) * (1.f / 65536.f));
    x += rr * __cosf(a); y += rr * __sinf(a);
  }
  float fx = x * P.sx + P.ox, fy = y * P.sy + P.oy;
  if (fx >= 0.f && fx < (float)P.W && fy >= 0.f && fy < (float)P.H) {
    ix = min((int)fx, P.W - 1); iy = min((int)fy, P.H - 1);
  } else { ix = -1; iy = -1; }
}

// naive (mode 0): one global atomic per point.
// warp  (mode 1): lanes that hit the same pixel are merged with
//                 __match_any_sync first; one atomic per distinct pixel.
__device__ __forceinline__ void ks_deposit(const KP& P, int pix, float w, float h, bool HB, bool HH,
                                           float* sw, float* sh) {
  if (P.mode == 0) {
    if (pix >= 0) {
      atomicAdd(P.acc_c + pix, HB ? w : 1.f);
      if (HH) atomicAdd(P.acc_h + pix, (HB ? w : 1.f) * h);
    }
    return;
  }
  const int lane = threadIdx.x & 31;
  const u32 m = __match_any_sync(KS_FULL, pix);
  const int leader = __ffs(m) - 1;
  if (!HB && !HH) {
    if (pix >= 0 && lane == leader) atomicAdd(P.acc_c + pix, (float)__popc(m));
    return;
  }
  const int wb = threadIdx.x & ~31;
  const float ww = HB ? w : 1.f;
  sw[threadIdx.x] = ww;
  if (HH) sh[threadIdx.x] = ww * h;
  __syncwarp();
  if (pix >= 0 && lane == leader) {
    float a = 0.f, c = 0.f; u32 mm = m;
    while (mm) { int l = __ffs(mm) - 1; mm &= mm - 1; a += sw[wb + l]; if (HH) c += sh[wb + l]; }
    atomicAdd(P.acc_c + pix, a);
    if (HH) atomicAdd(P.acc_h + pix, c);
  }
  __syncwarp();
}
"#;

/// Tile-binning pass 1 for one form; only compiled when bin mode is used.
const UNIT_BIN: &str = r#"
// tile binning, pass 1: evaluate every point once, stage its tile and packed
// key, and count points per 64x64 tile (block-local histogram in shared
// memory, flushed once per block through a list of touched tiles)
extern "C" __global__ void __launch_bounds__(512) ks_count_@ID@(KP P) {
  extern __shared__ u32 sm[];
  u32* sc = sm;
  u32* touched = sm + P.tiles;
  __shared__ u32 ntouch;
  __shared__ int overflow;
  for (int k = threadIdx.x; k < P.tiles; k += blockDim.x) sc[k] = 0;
  if (threadIdx.x == 0) { ntouch = 0; overflow = 0; }
  __syncthreads();
  const u32 s = (u32)P.s_base;
  const float* U = P.U;
  const int lane = threadIdx.x & 31;
  const u64 per = (P.chains + gridDim.x - 1) / gridDim.x;
  const u64 lo = (u64)blockIdx.x * per;
  const u64 hi = (lo + per < P.chains) ? lo + per : P.chains;
  for (u64 c0 = lo; c0 < hi; c0 += blockDim.x) {
    const u64 c = c0 + threadIdx.x;
    const bool valid = c < hi;
    const u64 chain = P.base + c;
    float px, py; ks_seed((u32)chain, px, py);
    u32 i = (u32)(chain * (u64)P.iters);
    for (int j = -P.warmup; j < P.iters; ++j) {
      int ix, iy; float w, h;
      ks_pt_@ID@(P, U, i, s, px, py, ix, iy, w, h);
      if (j < 0) continue;
      ++i;
      const int t = (valid && ix >= 0) ? (iy >> 6) * P.tiles_x + (ix >> 6) : -1;
      if (valid) {
        // j-major layout keeps these stores coalesced for attractor chains
        const u64 q = (u64)j * P.chains + c;
        P.pre_t[q] = t >= 0 ? (u32)t : 0xffffffffu;
        if (t >= 0) {
          u32 key = ((u32)(iy & 63) << 6) | (u32)(ix & 63);
          if (@HH@) key |= ((u32)(h * KS_HQ_MAX + 0.5f)) << 12;
          P.pre_k[q] = key;
          if (@HB@) P.pre_w[q] = w;
        }
      }
      const u32 m = __match_any_sync(KS_FULL, t);
      if (t >= 0 && lane == __ffs(m) - 1) {
        const u32 old = atomicAdd(&sc[t], (u32)__popc(m));
        if (old == 0) {
          const u32 slot = atomicAdd(&ntouch, 1u);
          if (slot < KS_TOUCH_C) touched[slot] = (u32)t; else overflow = 1;
        }
      }
    }
  }
  __syncthreads();
  if (overflow) {
    for (int k = threadIdx.x; k < P.tiles; k += blockDim.x) if (sc[k]) atomicAdd(P.tile_count + k, sc[k]);
  } else {
    for (u32 k = threadIdx.x; k < ntouch; k += blockDim.x) { const u32 t = touched[k]; atomicAdd(P.tile_count + t, sc[t]); }
  }
}

"#;

/// Per-form kernels. `@ID@`, `@HB@`, `@HH@`, `@ATTR@`, `@BODY@` are filled in.
const UNIT: &str = r#"
// ---------------- form @ID@ ----------------
__device__ __forceinline__ void ks_eval_@ID@(const float* __restrict__ U, u32 i, float inv_n, float px, float py,
    float& ox, float& oy, float& oz, float& ob, float& oh) {
  const float fi = (float)i;
@BODY@
}

__device__ __forceinline__ void ks_pt_@ID@(const KP& P, const float* __restrict__ U, u32 i, u32 s,
    float& px, float& py, int& ix, int& iy, float& w, float& h) {
  float x, y, z, b, hh;
  ks_eval_@ID@(U, i, P.inv_n, px, py, x, y, z, b, hh);
  if (@ATTR@) {
    if (fabsf(x) < 1e4f && fabsf(y) < 1e4f) { px = x; py = y; }
    else { ks_seed(i * 2654435761u, px, py); }   // chain blew up: restart it
  }
  ks_project(P, x, y, z, i, s, ix, iy);
  w = b;
  h = fminf(fmaxf(hh, 0.f), 1.f);
  if (@HB@ && !(w > 0.f)) { ix = -1; iy = -1; }
}

extern "C" __global__ void __launch_bounds__(256) ks_splat_@ID@(KP P) {
  __shared__ float sw[256], sh[256];
  const u32 s = (u32)P.s_base + blockIdx.y;
  const float* U = P.U + (u64)blockIdx.y * (u64)P.nu;
  for (u64 c0 = (u64)blockIdx.x * 256ull; c0 < P.chains; c0 += (u64)gridDim.x * 256ull) {
    const u64 c = c0 + threadIdx.x;
    const bool valid = c < P.chains;
    const u64 chain = P.base + c;
    float px, py; ks_seed((u32)chain, px, py);
    u32 i = (u32)(chain * (u64)P.iters);
    // one loop, one inlined copy of the form: warmup steps (j < 0) only
    // advance an attractor chain, they do not deposit
    for (int j = -P.warmup; j < P.iters; ++j) {
      int ix, iy; float w, h;
      ks_pt_@ID@(P, U, i, s, px, py, ix, iy, w, h);
      if (j >= 0) {
        const int pix = (valid && ix >= 0) ? iy * P.W + ix : -1;
        ks_deposit(P, pix, w, h, @HB@, @HH@, sw, sh);
        ++i;
      }
    }
  }
}

// correctness probe: raw outputs for sampled points (attractors: one step
// from the chain seed), compared against the cpu interpreter
extern "C" __global__ void ks_probe_@ID@(KP P) {
  const u64 k = (u64)blockIdx.x * 256ull + threadIdx.x;
  if (k >= P.chains) return;
  const u32 i = (u32)(P.base + k * P.stride);
  float px, py; ks_seed(i, px, py);
  float x, y, z, b, h;
  ks_eval_@ID@(P.U, i, P.inv_n, px, py, x, y, z, b, h);
  float* o = P.probe + k * 5;
  o[0] = x; o[1] = y; o[2] = z; o[3] = b; o[4] = h;
}
"#;

fn lit(c: f32) -> String {
    if c.is_nan() {
        "__int_as_float(0x7fc00000)".into()
    } else if c.is_infinite() {
        if c > 0.0 { "__int_as_float(0x7f800000)".into() } else { "__int_as_float(0xff800000)".into() }
    } else {
        let s = format!("{c:?}f");
        if c < 0.0 { format!("({s})") } else { s }
    }
}

fn operand(ir: &Ir, v: V) -> String {
    if let Some(c) = ir.const_val(v) {
        return lit(c);
    }
    if !ir.varying[v as usize] {
        return format!("U[{}]", ir.slot_of[&v]);
    }
    match ir.ops[v as usize] {
        Op::In(Input::I) => "fi".into(),
        Op::In(Input::U) => "(fi * inv_n)".into(),
        Op::In(Input::R1) => "ks_u01(i ^ 0xA511E9B3u)".into(),
        Op::In(Input::R2) => "ks_u01(i ^ 0x63D83595u)".into(),
        Op::In(Input::Px) => "px".into(),
        Op::In(Input::Py) => "py".into(),
        _ => format!("v{v}"),
    }
}

fn op_expr(ir: &Ir, op: &Op) -> String {
    let o = |v: &V| operand(ir, *v);
    match op {
        Op::Neg(a) => format!("-{}", o(a)),
        Op::Bin(BinOp::Mod, a, b) => format!("ks_mod({}, {})", o(a), o(b)),
        Op::Bin(BinOp::Pow, a, b) => format!("powf({}, {})", o(a), o(b)),
        Op::Bin(op, a, b) => format!("{} {} {}", o(a), op.sym(), o(b)),
        Op::Call(f, args) => {
            let a: Vec<String> = args.iter().map(o).collect();
            let name = match f {
                Func::Sin => "sinf",
                Func::Cos => "cosf",
                Func::Tan => "tanf",
                Func::Tanh => "tanhf",
                Func::Atan2 => "atan2f",
                Func::Sqrt => "sqrtf",
                Func::Abs => "fabsf",
                Func::Fract => "ks_fract",
                Func::Floor => "floorf",
                Func::Sign => "ks_sign",
                Func::Min => "fminf",
                Func::Max => "fmaxf",
                Func::Mix => "ks_mix",
                Func::Clamp => "ks_clamp",
                Func::Smoothstep => "ks_smooth",
                Func::Mag => "ks_mag",
                Func::Exp => "expf",
                Func::Log => "logf",
                Func::Pow => "powf",
                Func::Noise => "ks_noise",
                Func::Rnd => "ks_rnd",
            };
            format!("{}({})", name, a.join(", "))
        }
        Op::Const(_) | Op::In(_) | Op::Param(_) => unreachable!(),
    }
}

/// Body of the fused per-point function.
pub fn emit_body(ir: &Ir) -> String {
    let mut s = String::new();
    for (v, op) in ir.ops.iter().enumerate() {
        if !ir.live[v] || !ir.varying[v] || matches!(op, Op::In(_)) {
            continue;
        }
        s.push_str(&format!("  const float v{} = {};\n", v, op_expr(ir, op)));
    }
    let names = ["ox", "oy", "oz", "ob", "oh"];
    let defaults = ["0.f", "0.f", "0.f", "1.f", "0.f"];
    for k in 0..5 {
        let val = match ir.outs[k] {
            Some(v) => operand(ir, v),
            None => defaults[k].to_string(),
        };
        s.push_str(&format!("  {} = {};\n", names[k], val));
    }
    // silence unused-parameter warnings
    s.push_str("  (void)fi; (void)inv_n; (void)px; (void)py; (void)U;\n");
    s
}

pub fn emit_unit(ir: &Ir, id: usize, with_bin: bool) -> String {
    let b = |x: bool| if x { "true" } else { "false" };
    let mut src = UNIT.to_string();
    if with_bin {
        src.push_str(UNIT_BIN);
    }
    src.replace("@BODY@", &emit_body(ir))
        .replace("@ID@", &id.to_string())
        .replace("@HB@", b(ir.has_b()))
        .replace("@HH@", b(ir.has_h()))
        .replace("@ATTR@", b(ir.attractor))
}

/// A module source for one or more forms. `with_bin` adds the tile-binning
/// count kernels (left out of the live hot-reload path to compile faster).
pub fn module_source(irs: &[&Ir], with_bin: bool) -> String {
    let mut s = String::from(PRELUDE);
    for (k, ir) in irs.iter().enumerate() {
        s.push_str(&emit_unit(ir, k, with_bin));
    }
    s
}

pub fn kernel_names(id: usize) -> [String; 3] {
    [format!("ks_splat_{id}"), format!("ks_count_{id}"), format!("ks_probe_{id}")]
}
