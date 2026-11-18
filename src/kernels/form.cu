// ---------------- form @ID@ ----------------
__device__ __forceinline__ void ks_eval_@ID@(const float* __restrict__ U, u32 i, float inv_n, float px, float py,
    float& ox, float& oy, float& oz, float& ob, float& oh) {
  const float fi = (float)i;
@BODY@
}

__device__ __forceinline__ void ks_pt_@ID@(const SplatArgs& P, const float* __restrict__ U, u32 i, u32 s,
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

extern "C" __global__ void __launch_bounds__(256) ks_splat_@ID@(SplatArgs P) {
  __shared__ float sw[256], sh[256];
  const u32 s = (u32)P.subframe + blockIdx.y;
  const float* U = P.uniforms + (u64)blockIdx.y * (u64)P.n_uniforms;
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
        const int pix = (valid && ix >= 0) ? iy * P.w + ix : -1;
        ks_deposit(P, pix, w, h, @HB@, @HH@, sw, sh);
        ++i;
      }
    }
  }
}

// correctness probe: raw outputs for sampled points (attractors: one step
// from the chain seed), compared against the cpu interpreter
extern "C" __global__ void ks_probe_@ID@(SplatArgs P) {
  const u64 k = (u64)blockIdx.x * 256ull + threadIdx.x;
  if (k >= P.chains) return;
  const u32 i = (u32)(P.base + k * P.stride);
  float px, py; ks_seed(i, px, py);
  float x, y, z, b, h;
  ks_eval_@ID@(P.uniforms, i, P.inv_n, px, py, x, y, z, b, h);
  float* o = P.probe + k * 5;
  o[0] = x; o[1] = y; o[2] = z; o[3] = b; o[4] = h;
}
