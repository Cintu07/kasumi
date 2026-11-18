// helpers shared by every form. interp.rs mirrors the math exactly.
#define KS_FULL 0xffffffffu
#define KS_HQ_MAX 1048575.0f

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
__device__ __forceinline__ void ks_project(const SplatArgs& P, float x, float y, float z, u32 i, u32 s, int& ix, int& iy) {
  if (P.dof > 0.f) {
    float r = P.dof * fabsf(z - P.focus);
    u32 hs = ks_hash(i * 0x9E3779B9u + s * 0x85EBCA6Bu + 0x01234567u);
    float a = (float)(hs & 0xffffu) * (6.2831853f / 65536.f);
    float rr = r * sqrtf((float)(hs >> 16) * (1.f / 65536.f));
    x += rr * __cosf(a); y += rr * __sinf(a);
  }
  float fx = x * P.scale_x + P.offset_x, fy = y * P.scale_y + P.offset_y;
  if (fx >= 0.f && fx < (float)P.w && fy >= 0.f && fy < (float)P.h) {
    ix = min((int)fx, P.w - 1); iy = min((int)fy, P.h - 1);
  } else { ix = -1; iy = -1; }
}

// naive (mode 0): one global atomic per point.
// warp  (mode 1): lanes that hit the same pixel are merged with
//                 __match_any_sync first; one atomic per distinct pixel.
__device__ __forceinline__ void ks_deposit(const SplatArgs& P, int pix, float w, float h, bool HB, bool HH,
                                           float* sw, float* sh) {
  if (P.mode == 0) {
    if (pix >= 0) {
      atomicAdd(P.density + pix, HB ? w : 1.f);
      if (HH) atomicAdd(P.hue + pix, (HB ? w : 1.f) * h);
    }
    return;
  }
  const int lane = threadIdx.x & 31;
  const u32 m = __match_any_sync(KS_FULL, pix);
  const int leader = __ffs(m) - 1;
  if (!HB && !HH) {
    if (pix >= 0 && lane == leader) atomicAdd(P.density + pix, (float)__popc(m));
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
    atomicAdd(P.density + pix, a);
    if (HH) atomicAdd(P.hue + pix, c);
  }
  __syncwarp();
}
