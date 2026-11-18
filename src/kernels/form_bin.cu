// tile binning, pass 1: evaluate every point once, stage its tile and packed
// key, and count points per 64x64 tile (block-local histogram in shared
// memory, flushed once per block through a list of touched tiles)
extern "C" __global__ void __launch_bounds__(512) ks_count_@ID@(SplatArgs P) {
  extern __shared__ u32 sm[];
  u32* sc = sm;
  u32* touched = sm + P.tiles;
  __shared__ u32 ntouch;
  __shared__ int overflow;
  for (int k = threadIdx.x; k < P.tiles; k += blockDim.x) sc[k] = 0;
  if (threadIdx.x == 0) { ntouch = 0; overflow = 0; }
  __syncthreads();
  const u32 s = (u32)P.subframe;
  const float* U = P.uniforms;
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
        P.staged_tile[q] = t >= 0 ? (u32)t : 0xffffffffu;
        if (t >= 0) {
          u32 key = ((u32)(iy & 63) << 6) | (u32)(ix & 63);
          if (@HH@) key |= ((u32)(h * KS_HQ_MAX + 0.5f)) << 12;
          P.staged_key[q] = key;
          if (@HB@) P.staged_weight[q] = w;
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

