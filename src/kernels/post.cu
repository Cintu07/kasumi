// kernels that do not depend on the form: binning, histogram, tone map, glow, density estimation, trails
#define KS_FULL 0xffffffffu
#define KS_SLICE 32768u
#define KS_HQ_MAX 1048575.0f


// exclusive scan of tile counts -> segment offsets, cursors and the work list
// of <=32k-point slices (so one hot tile is split across many blocks)
extern "C" __global__ void __launch_bounds__(1024) ks_scan(BinArgs P) {
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

extern "C" __global__ void __launch_bounds__(512) ks_scatter(ScatterArgs P) {
  extern __shared__ u32 sm[];
  u32* sc = sm;
  u32* touched = sm + P.tiles;
  u32* tbase = touched + KS_TOUCH_S;
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
      const u32 tt = v ? P.staged_tile[q] : 0xffffffffu;
      t[r] = tt == 0xffffffffu ? -1 : (int)tt;
      key[r] = t[r] >= 0 ? P.staged_key[q] : 0u;
      w[r] = (t[r] >= 0 && P.has_b) ? P.staged_weight[q] : 0.f;
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
        if (P.has_b) P.weights[pos] = w[r];
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
extern "C" __global__ void __launch_bounds__(256) ks_accum(BinArgs P) {
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
        const float ww = P.has_b ? P.weights[q] : 1.f;
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
      if (gx >= P.w || gy >= P.h) continue;
      const u64 idx = (u64)gy * (u64)P.w + (u64)gx;
      if (nsl == 1) { P.density[idx] += c; if (P.has_h) P.hue[idx] += hh[k]; }
      else { atomicAdd(P.density + idx, c); if (P.has_h) atomicAdd(P.hue + idx, hh[k]); }
    }
    __syncthreads();
  }
}


// 1024-bin histogram of log2(1 + density) over lit pixels
extern "C" __global__ void __launch_bounds__(256) ks_hist(HistArgs P) {
  __shared__ u32 sh[1024];
  for (int k = threadIdx.x; k < 1024; k += 256) sh[k] = 0;
  __syncthreads();
  for (u64 k = (u64)blockIdx.x * 256ull + threadIdx.x; k < P.n; k += (u64)gridDim.x * 256ull) {
    const float c = P.density[k];
    if (c > 0.f) { int b = (int)(log2f(1.f + c) * 32.f); atomicAdd(&sh[b > 1023 ? 1023 : b], 1u); }
  }
  __syncthreads();
  for (int k = threadIdx.x; k < 1024; k += 256) if (sh[k]) atomicAdd(P.hist + k, sh[k]);
}


__device__ __forceinline__ float ks_tone1(float c, float al, float gi, float vib, float pa) {
  return vib * pa * c + (1.f - vib) * powf(c * al, gi);
}

// log-density tone map (fractal flame style) with vibrancy, a little white
// heat at the densest spots, and ss x ss supersample averaging
extern "C" __global__ void ks_tonemap(ToneArgs P) {
  const int OW = P.w / P.ss, OH = P.h / P.ss;
  const int ox = blockIdx.x * 16 + threadIdx.x, oy = blockIdx.y * 16 + threadIdx.y;
  if (ox >= OW || oy >= OH) return;
  float r = 0.f, g = 0.f, b = 0.f;
  for (int sy = 0; sy < P.ss; ++sy) for (int sx = 0; sx < P.ss; ++sx) {
    const u64 idx = (u64)(oy * P.ss + sy) * (u64)P.w + (u64)(ox * P.ss + sx);
    const float c = P.density[idx];
    if (!(c > 0.f)) continue;
    const float al = fminf(log1pf(c) * P.inv_log_ref, 1.f);
    const float hv = P.has_h ? fminf(fmaxf(P.hue[idx] / c, 0.f), 1.f) : 0.f;
    float cr = P.low.r + (P.high.r - P.low.r) * hv, cg = P.low.g + (P.high.g - P.low.g) * hv, cb = P.low.b + (P.high.b - P.low.b) * hv;
    const float a2 = al * al, hot = P.heat * a2 * a2;
    cr += (1.f - cr) * hot; cg += (1.f - cg) * hot; cb += (1.f - cb) * hot;
    const float pa = powf(al, P.gamma_inv);
    r += ks_tone1(cr, al, P.gamma_inv, P.vibrancy, pa);
    g += ks_tone1(cg, al, P.gamma_inv, P.vibrancy, pa);
    b += ks_tone1(cb, al, P.gamma_inv, P.vibrancy, pa);
  }
  const float inv = 1.f / (float)(P.ss * P.ss);
  r = fminf(fmaxf(r * inv, 0.f), 1.f); g = fminf(fmaxf(g * inv, 0.f), 1.f); b = fminf(fmaxf(b * inv, 0.f), 1.f);
  if (P.glow) {
    // bilinear fetch of the blurred glow, added with a screen blend so it cannot clip
    const float fx = fmaxf(((float)ox + 0.5f) * (float)P.glow_w / (float)OW - 0.5f, 0.f);
    const float fy = fmaxf(((float)oy + 0.5f) * (float)P.glow_h / (float)OH - 0.5f, 0.f);
    const int x0 = min((int)fx, P.glow_w - 1), y0 = min((int)fy, P.glow_h - 1);
    const int x1 = min(x0 + 1, P.glow_w - 1), y1 = min(y0 + 1, P.glow_h - 1);
    const float tx = fx - (float)x0, ty = fy - (float)y0;
    float glow[3];
    for (int c = 0; c < 3; ++c) {
      const float a = P.glow[(y0 * P.glow_w + x0) * 3 + c] * (1.f - tx) + P.glow[(y0 * P.glow_w + x1) * 3 + c] * tx;
      const float d = P.glow[(y1 * P.glow_w + x0) * 3 + c] * (1.f - tx) + P.glow[(y1 * P.glow_w + x1) * 3 + c] * tx;
      glow[c] = fminf((a * (1.f - ty) + d * ty) * P.glow_gain, 1.f);
    }
    r = 1.f - (1.f - r) * (1.f - glow[0]);
    g = 1.f - (1.f - g) * (1.f - glow[1]);
    b = 1.f - (1.f - b) * (1.f - glow[2]);
  }
  const u64 o = (u64)oy * (u64)OW + (u64)ox;
  if (P.format == 0) {
    ((u32*)P.out)[o] = ((u32)(r * 255.f + 0.5f) << 16) | ((u32)(g * 255.f + 0.5f) << 8) | (u32)(b * 255.f + 0.5f);
  } else if (P.format == 1) {
    unsigned short* q = (unsigned short*)P.out + o * 3;
    q[0] = (unsigned short)(r * 65535.f + 0.5f); q[1] = (unsigned short)(g * 65535.f + 0.5f); q[2] = (unsigned short)(b * 65535.f + 0.5f);
  } else {
    unsigned char* q = (unsigned char*)P.out + o * 3;
    q[0] = (unsigned char)(r * 255.f + 0.5f); q[1] = (unsigned char)(g * 255.f + 0.5f); q[2] = (unsigned char)(b * 255.f + 0.5f);
  }
}

// soft bloom on the brightest regions only: the part of the log density above
// a threshold goes into a low-res buffer, which is then blurred in two passes
#define KS_BLOOM_THRESHOLD 0.62f

extern "C" __global__ void ks_bloom_build(GlowArgs P) {
  const int lx = blockIdx.x * 16 + threadIdx.x, ly = blockIdx.y * 16 + threadIdx.y;
  if (lx >= P.glow_w || ly >= P.glow_h) return;
  const int step = P.cell > 8 ? P.cell / 8 : 1;
  float r = 0.f, g = 0.f, b = 0.f, n = 0.f;
  for (int y = 0; y < P.cell; y += step) for (int x = 0; x < P.cell; x += step) {
    const int sx = lx * P.cell + x, sy = ly * P.cell + y;
    n += 1.f;
    if (sx >= P.w || sy >= P.h) continue;
    const u64 idx = (u64)sy * (u64)P.w + (u64)sx;
    const float c = P.density[idx];
    if (!(c > 0.f)) continue;
    const float al = fminf(log1pf(c) * P.inv_log_ref, 1.f);
    const float k = fmaxf(al - KS_BLOOM_THRESHOLD, 0.f) / (1.f - KS_BLOOM_THRESHOLD);
    if (k <= 0.f) continue;
    const float hv = P.has_h ? fminf(fmaxf(P.hue[idx] / c, 0.f), 1.f) : 0.f;
    r += k * (P.low.r + (P.high.r - P.low.r) * hv); g += k * (P.low.g + (P.high.g - P.low.g) * hv); b += k * (P.low.b + (P.high.b - P.low.b) * hv);
  }
  float* o = P.out + (u64)(ly * P.glow_w + lx) * 3;
  o[0] = r / n; o[1] = g / n; o[2] = b / n;
}


// one direction of a 25-tap gaussian (sigma 6 texels)
extern "C" __global__ void ks_bloom_blur(BlurArgs P) {
  const int x = blockIdx.x * 16 + threadIdx.x, y = blockIdx.y * 16 + threadIdx.y;
  if (x >= P.w || y >= P.h) return;
  float s[3] = {0.f, 0.f, 0.f}, sum = 0.f;
  for (int k = -12; k <= 12; ++k) {
    const int sx = max(min(x + k * P.dx, P.w - 1), 0), sy = max(min(y + k * P.dy, P.h - 1), 0);
    const float g = expf(-(float)(k * k) / 72.f);
    const float* q = P.src + (u64)(sy * P.w + sx) * 3;
    s[0] += q[0] * g; s[1] += q[1] * g; s[2] += q[2] * g; sum += g;
  }
  float* o = P.dst + (u64)(y * P.w + x) * 3;
  o[0] = s[0] / sum; o[1] = s[1] / sum; o[2] = s[2] / sum;
}

// adaptive density estimation: sparse pixels are blurred more than dense ones,
// which removes grain where few points landed and leaves dense detail sharp.
// the blur comes from a small mip chain of the accumulation buffer

extern "C" __global__ void ks_de_down(DownArgs P) {
  const int x = blockIdx.x * 16 + threadIdx.x, y = blockIdx.y * 16 + threadIdx.y;
  if (x >= P.dst_w || y >= P.dst_h) return;
  float c = 0.f, h = 0.f;
  for (int j = 0; j < 2; ++j) for (int i = 0; i < 2; ++i) {
    const u64 q = (u64)min(y * 2 + j, P.src_h - 1) * (u64)P.src_w + (u64)min(x * 2 + i, P.src_w - 1);
    c += P.src_density[q];
    if (P.has_h) h += P.src_hue[q];
  }
  const u64 o = (u64)y * (u64)P.dst_w + (u64)x;
  P.dst_density[o] = c * 0.25f;
  if (P.has_h) P.dst_hue[o] = h * 0.25f;
}

__device__ __forceinline__ float ks_bil(const float* t, int w, int h, float x, float y) {
  const float fx = fmaxf(x - 0.5f, 0.f), fy = fmaxf(y - 0.5f, 0.f);
  const int x0 = min((int)fx, w - 1), y0 = min((int)fy, h - 1);
  const int x1 = min(x0 + 1, w - 1), y1 = min(y0 + 1, h - 1);
  const float tx = fx - (float)x0, ty = fy - (float)y0;
  const float a = t[(u64)y0 * (u64)w + (u64)x0] * (1.f - tx) + t[(u64)y0 * (u64)w + (u64)x1] * tx;
  const float b = t[(u64)y1 * (u64)w + (u64)x0] * (1.f - tx) + t[(u64)y1 * (u64)w + (u64)x1] * tx;
  return a * (1.f - ty) + b * ty;
}

// cubic b-spline sample from four bilinear fetches: the low levels of the chain
// are very coarse, and plain bilinear shows their texels as stair steps
__device__ __forceinline__ float ks_bic(const float* t, int w, int h, float x, float y) {
  const float tx = x - 0.5f, ty = y - 0.5f;
  const float ix = floorf(tx), iy = floorf(ty), fx = tx - ix, fy = ty - iy;
  const float x2 = fx * fx, x3 = x2 * fx, y2 = fy * fy, y3 = y2 * fy;
  const float a0 = (1.f - 3.f * fx + 3.f * x2 - x3) / 6.f, a1 = (4.f - 6.f * x2 + 3.f * x3) / 6.f;
  const float a2 = (1.f + 3.f * fx + 3.f * x2 - 3.f * x3) / 6.f, a3 = x3 / 6.f;
  const float b0 = (1.f - 3.f * fy + 3.f * y2 - y3) / 6.f, b1 = (4.f - 6.f * y2 + 3.f * y3) / 6.f;
  const float b2 = (1.f + 3.f * fy + 3.f * y2 - 3.f * y3) / 6.f, b3 = y3 / 6.f;
  const float gx0 = a0 + a1, gx1 = a2 + a3, gy0 = b0 + b1, gy1 = b2 + b3;
  const float px0 = ix + a1 / gx0 - 0.5f, px1 = ix + a3 / gx1 + 1.5f;
  const float py0 = iy + b1 / gy0 - 0.5f, py1 = iy + b3 / gy1 + 1.5f;
  return gy0 * (gx0 * ks_bil(t, w, h, px0, py0) + gx1 * ks_bil(t, w, h, px1, py0))
       + gy1 * (gx0 * ks_bil(t, w, h, px0, py1) + gx1 * ks_bil(t, w, h, px1, py1));
}


extern "C" __global__ void ks_de_resolve(ResolveArgs P) {
  const int x = blockIdx.x * 16 + threadIdx.x, y = blockIdx.y * 16 + threadIdx.y;
  if (x >= P.w || y >= P.h) return;
  const u64 o = (u64)y * (u64)P.w + (u64)x;
  const float px = (float)x + 0.5f, py = (float)y + 0.5f;
  // local density from the quarter-resolution level picks the blur width
  const float d = ks_bic(P.level_density.v[1], P.level_w.v[1], P.level_h.v[1], px * 0.25f, py * 0.25f);
  const float width = P.max_radius / powf(1.f + d, P.curve);
  const float L = fminf(log2f(fmaxf(width, 1.f)), 4.f);
  const int k0 = (int)L, k1 = min(k0 + 1, 4);
  const float f = L - (float)k0;
  float c[2], h[2];
  for (int n = 0; n < 2; ++n) {
    const int k = n == 0 ? k0 : k1;
    if (k == 0) { c[n] = P.density[o]; h[n] = P.has_h ? P.hue[o] : 0.f; continue; }
    const float s = 1.f / (float)(1 << k);
    c[n] = ks_bic(P.level_density.v[k - 1], P.level_w.v[k - 1], P.level_h.v[k - 1], px * s, py * s);
    h[n] = P.has_h ? ks_bic(P.level_hue.v[k - 1], P.level_w.v[k - 1], P.level_h.v[k - 1], px * s, py * s) : 0.f;
  }
  P.out_density[o] = c[0] + (c[1] - c[0]) * f;
  if (P.has_h) P.out_hue[o] = h[0] + (h[1] - h[0]) * f;
}


// feedback trails: fade the accumulation instead of clearing it
extern "C" __global__ void ks_decay(DecayArgs P) {
  for (u64 k = (u64)blockIdx.x * 256ull + threadIdx.x; k < P.n; k += (u64)gridDim.x * 256ull) {
    P.density[k] *= P.factor;
    P.hue[k] *= P.factor;
  }
}
