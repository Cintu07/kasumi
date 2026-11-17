//! Kernels that do not depend on the form: the tile-binning scan and
//! accumulate passes, the density histogram, tone mapping and trail decay.

pub const STATIC_SRC: &str = r#"
typedef unsigned int u32;
typedef unsigned long long u64;
#define KS_FULL 0xffffffffu
#define KS_SLICE 32768u
#define KS_HQ_MAX 1048575.0f

// Mirrored by `AP` in render.rs
struct AP {
  const u32* keys;
  const float* wts;
  const u32* tile_count;
  u32* tile_offset;
  u32* tile_cursor;
  u32* slice_prefix;   // tiles + 1 entries; [tiles] = total slices
  u32* work;
  float* acc_c;
  float* acc_h;
  int W, H, tiles_x, tiles, has_b, has_h;
};

// exclusive scan of tile counts -> segment offsets, cursors and the work list
// of <=32k-point slices (so one hot tile is split across many blocks)
extern "C" __global__ void __launch_bounds__(1024) ks_scan(AP P) {
  __shared__ u32 a[1024], b[1024];
  const int T = P.tiles;
  const int per = (T + 1023) / 1024;
  const int lo = threadIdx.x * per;
  u32 sc = 0, ss = 0;
  for (int k = lo; k < lo + per && k < T; ++k) { u32 c = P.tile_count[k]; sc += c; ss += (c + KS_SLICE - 1) / KS_SLICE; }
  a[threadIdx.x] = sc; b[threadIdx.x] = ss;
  __syncthreads();
  for (int off = 1; off < 1024; off <<= 1) {
    u32 va = threadIdx.x >= off ? a[threadIdx.x - off] : 0u;
    u32 vb = threadIdx.x >= off ? b[threadIdx.x - off] : 0u;
    __syncthreads();
    a[threadIdx.x] += va; b[threadIdx.x] += vb;
    __syncthreads();
  }
  u32 ea = a[threadIdx.x] - sc, eb = b[threadIdx.x] - ss;
  for (int k = lo; k < lo + per && k < T; ++k) {
    u32 c = P.tile_count[k];
    P.tile_offset[k] = ea; P.tile_cursor[k] = ea; P.slice_prefix[k] = eb;
    ea += c; eb += (c + KS_SLICE - 1) / KS_SLICE;
  }
  if (threadIdx.x == 1023) P.slice_prefix[T] = b[1023];
}

// tile binning, pass 2: move each staged key into its tile's segment.
// form-independent; 4 points per thread per round, one global reservation
// per (block, tile) per round
struct SP { const u32* pre_t; const u32* pre_k; const float* pre_w; u32* keys; float* wts; u32* tile_cursor; u64 n; int tiles; int has_b; };

extern "C" __global__ void __launch_bounds__(512) ks_scatter(SP P) {
  extern __shared__ u32 sm[];
  u32* sc = sm;
  u32* touched = sm + P.tiles;
  u32* tbase = touched + 1024;
  __shared__ u32 ntouch;
  for (int k = threadIdx.x; k < P.tiles; k += blockDim.x) sc[k] = 0;
  if (threadIdx.x == 0) ntouch = 0;
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const u64 per = ((P.n + gridDim.x - 1) / gridDim.x + 1023) / 1024 * 1024;
  const u64 lo = (u64)blockIdx.x * per;
  const u64 hi = (lo + per < P.n) ? lo + per : P.n;
  for (u64 q0 = lo; q0 < hi; q0 += 1024) {
    int t[2]; u32 key[2]; float w[2]; u32 m[2]; u32 rank[2]; int lead[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const u64 q = q0 + r * 512 + threadIdx.x;
      const bool v = q < hi;
      const u32 tt = v ? P.pre_t[q] : 0xffffffffu;
      t[r] = tt == 0xffffffffu ? -1 : (int)tt;
      key[r] = t[r] >= 0 ? P.pre_k[q] : 0u;
      w[r] = (t[r] >= 0 && P.has_b) ? P.pre_w[q] : 0.f;
      m[r] = __match_any_sync(KS_FULL, t[r]);
      lead[r] = __ffs(m[r]) - 1;
      rank[r] = __popc(m[r] & ((1u << lane) - 1u));
      if (t[r] >= 0 && lane == lead[r]) {
        const u32 old = atomicAdd(&sc[t[r]], (u32)__popc(m[r]));
        if (old == 0) { const u32 slot = atomicAdd(&ntouch, 1u); touched[slot] = (u32)t[r]; }
      }
    }
    __syncthreads();
    const u32 nt = ntouch;
    for (u32 k = threadIdx.x; k < nt; k += blockDim.x) {
      const u32 tt = touched[k];
      tbase[k] = atomicAdd(P.tile_cursor + tt, sc[tt]);
      sc[tt] = k;
    }
    __syncthreads();
    if (threadIdx.x == 0) ntouch = 0;
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      if (t[r] >= 0) {
        const u32 slot = sc[t[r]];
        u32 pos = 0;
        if (lane == lead[r]) pos = atomicAdd(&tbase[slot], (u32)__popc(m[r]));
        pos = __shfl_sync(m[r], pos, lead[r]) + rank[r];
        P.keys[pos] = key[r];
        if (P.has_b) P.wts[pos] = w[r];
      }
    }
    __syncthreads();
    for (u32 k = threadIdx.x; k < nt; k += blockDim.x) sc[touched[k]] = 0;
    __syncthreads();
  }
}

// tile binning, pass 3: persistent blocks pull slices off a work counter,
// accumulate the slice into a 64x64 shared-memory tile, then write the tile
// out once (plain stores when the tile has one slice, atomics otherwise)
extern "C" __global__ void __launch_bounds__(256) ks_accum(AP P) {
  __shared__ float hc[4096];
  __shared__ float hh[4096];
  __shared__ u32 s_w;
  __shared__ int s_t;
  const int lane = threadIdx.x & 31;
  const u32 total = P.slice_prefix[P.tiles];
  for (;;) {
    if (threadIdx.x == 0) {
      const u32 w = atomicAdd(P.work, 1u);
      s_w = w;
      if (w < total) {
        int lo = 0, hi = P.tiles - 1;
        while (lo < hi) { int mid = (lo + hi + 1) >> 1; if (P.slice_prefix[mid] <= w) lo = mid; else hi = mid - 1; }
        s_t = lo;
      }
    }
    for (int k = threadIdx.x; k < 4096; k += 256) { hc[k] = 0.f; hh[k] = 0.f; }
    __syncthreads();
    const u32 w = s_w;
    if (w >= total) break;
    const int t = s_t;
    const u32 first = P.slice_prefix[t];
    const u32 nsl = P.slice_prefix[t + 1] - first;
    const u32 beg = P.tile_offset[t] + (w - first) * KS_SLICE;
    const u32 seg_end = P.tile_offset[t] + P.tile_count[t];
    const u32 end = (beg + KS_SLICE < seg_end) ? beg + KS_SLICE : seg_end;
    if (!P.has_b && !P.has_h) {
      for (u32 q0 = beg; q0 < end; q0 += 256) {
        const u32 q = q0 + threadIdx.x;
        const int loc = q < end ? (int)(P.keys[q] & 4095u) : -1;
        const u32 m = __match_any_sync(KS_FULL, loc);
        if (loc >= 0 && lane == __ffs(m) - 1) atomicAdd(&hc[loc], (float)__popc(m));
      }
    } else {
      for (u32 q = beg + threadIdx.x; q < end; q += 256) {
        const u32 key = P.keys[q];
        const int loc = (int)(key & 4095u);
        const float ww = P.has_b ? P.wts[q] : 1.f;
        atomicAdd(&hc[loc], ww);
        if (P.has_h) atomicAdd(&hh[loc], ww * (float)(key >> 12) * (1.f / KS_HQ_MAX));
      }
    }
    __syncthreads();
    const int tx0 = (t % P.tiles_x) * 64, ty0 = (t / P.tiles_x) * 64;
    for (int k = threadIdx.x; k < 4096; k += 256) {
      const float c = hc[k];
      if (c == 0.f) continue;
      const int gx = tx0 + (k & 63), gy = ty0 + (k >> 6);
      if (gx >= P.W || gy >= P.H) continue;
      const u64 idx = (u64)gy * (u64)P.W + (u64)gx;
      if (nsl == 1) { P.acc_c[idx] += c; if (P.has_h) P.acc_h[idx] += hh[k]; }
      else { atomicAdd(P.acc_c + idx, c); if (P.has_h) atomicAdd(P.acc_h + idx, hh[k]); }
    }
    __syncthreads();
  }
}

struct HP { const float* acc; u32* hist; u64 n; };

// 1024-bin histogram of log2(1 + density) over lit pixels
extern "C" __global__ void __launch_bounds__(256) ks_hist(HP P) {
  __shared__ u32 sh[1024];
  for (int k = threadIdx.x; k < 1024; k += 256) sh[k] = 0;
  __syncthreads();
  for (u64 k = (u64)blockIdx.x * 256ull + threadIdx.x; k < P.n; k += (u64)gridDim.x * 256ull) {
    const float c = P.acc[k];
    if (c > 0.f) { int b = (int)(log2f(1.f + c) * 32.f); atomicAdd(&sh[b > 1023 ? 1023 : b], 1u); }
  }
  __syncthreads();
  for (int k = threadIdx.x; k < 1024; k += 256) if (sh[k]) atomicAdd(P.hist + k, sh[k]);
}

// Mirrored by `TP` in render.rs
struct TP {
  const float* acc_c;
  const float* acc_h;
  void* out;
  int W, H, ss, fmt;
  float inv_logref, gamma_inv, vib, heat;
  float ar, ag, ab, br, bg, bb;
  int has_h, pad;
};

__device__ __forceinline__ float ks_tone1(float c, float al, float gi, float vib, float pa) {
  return vib * pa * c + (1.f - vib) * powf(c * al, gi);
}

// log-density tone map (fractal flame style) with vibrancy, a little white
// heat at the densest spots, and ss x ss supersample averaging
extern "C" __global__ void ks_tonemap(TP P) {
  const int OW = P.W / P.ss, OH = P.H / P.ss;
  const int ox = blockIdx.x * 16 + threadIdx.x, oy = blockIdx.y * 16 + threadIdx.y;
  if (ox >= OW || oy >= OH) return;
  float r = 0.f, g = 0.f, b = 0.f;
  for (int sy = 0; sy < P.ss; ++sy) for (int sx = 0; sx < P.ss; ++sx) {
    const u64 idx = (u64)(oy * P.ss + sy) * (u64)P.W + (u64)(ox * P.ss + sx);
    const float c = P.acc_c[idx];
    if (!(c > 0.f)) continue;
    const float al = fminf(log1pf(c) * P.inv_logref, 1.f);
    const float hv = P.has_h ? fminf(fmaxf(P.acc_h[idx] / c, 0.f), 1.f) : 0.f;
    float cr = P.ar + (P.br - P.ar) * hv, cg = P.ag + (P.bg - P.ag) * hv, cb = P.ab + (P.bb - P.ab) * hv;
    const float a2 = al * al, hot = P.heat * a2 * a2;
    cr += (1.f - cr) * hot; cg += (1.f - cg) * hot; cb += (1.f - cb) * hot;
    const float pa = powf(al, P.gamma_inv);
    r += ks_tone1(cr, al, P.gamma_inv, P.vib, pa);
    g += ks_tone1(cg, al, P.gamma_inv, P.vib, pa);
    b += ks_tone1(cb, al, P.gamma_inv, P.vib, pa);
  }
  const float inv = 1.f / (float)(P.ss * P.ss);
  r = fminf(fmaxf(r * inv, 0.f), 1.f); g = fminf(fmaxf(g * inv, 0.f), 1.f); b = fminf(fmaxf(b * inv, 0.f), 1.f);
  const u64 o = (u64)oy * (u64)OW + (u64)ox;
  if (P.fmt == 0) {
    ((u32*)P.out)[o] = ((u32)(r * 255.f + 0.5f) << 16) | ((u32)(g * 255.f + 0.5f) << 8) | (u32)(b * 255.f + 0.5f);
  } else if (P.fmt == 1) {
    unsigned short* q = (unsigned short*)P.out + o * 3;
    q[0] = (unsigned short)(r * 65535.f + 0.5f); q[1] = (unsigned short)(g * 65535.f + 0.5f); q[2] = (unsigned short)(b * 65535.f + 0.5f);
  } else {
    unsigned char* q = (unsigned char*)P.out + o * 3;
    q[0] = (unsigned char)(r * 255.f + 0.5f); q[1] = (unsigned char)(g * 255.f + 0.5f); q[2] = (unsigned char)(b * 255.f + 0.5f);
  }
}

struct DP { float* a; float* b; u64 n; float f; int has_b; };

// feedback trails: fade the accumulation instead of clearing it
extern "C" __global__ void ks_decay(DP P) {
  for (u64 k = (u64)blockIdx.x * 256ull + threadIdx.x; k < P.n; k += (u64)gridDim.x * 256ull) {
    P.a[k] *= P.f;
    if (P.has_b) P.b[k] *= P.f;
  }
}
"#;
