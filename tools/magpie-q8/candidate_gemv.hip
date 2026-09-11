// magpie-q8/baseline.hip — current shipping kernel, self-timed.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

constexpr int TILE = 16;

__device__ __forceinline__ float fp16_to_float_u(uint16_t h) {
  uint32_t sign = (uint32_t(h & 0x8000u)) << 16;
  uint32_t exp = (h >> 10) & 0x1fu;
  uint32_t mant = h & 0x03ffu;
  uint32_t f;
  if (exp == 0) {
    if (mant == 0) f = sign;
    else {
      exp = 127 - 15 + 1;
      while ((mant & 0x0400u) == 0) { mant <<= 1; --exp; }
      mant &= 0x03ffu;
      f = sign | (exp << 23) | (mant << 13);
    }
  } else if (exp == 0x1fu) {
    f = sign | 0x7f800000u | (mant << 13);
  } else {
    f = sign | ((exp - 15 + 127) << 23) | (mant << 13);
  }
  return __int_as_float(f);
}

__device__ __forceinline__ float q8_load(const uint8_t* __restrict__ w, int64_t idx) {
  int64_t block = idx >> 5;
  int sub = (int)(idx & 31);
  const uint8_t* b = w + block * 34;
  uint16_t d;
  __builtin_memcpy(&d, b, 2);
  float scale = fp16_to_float_u(d);
  return scale * (float)((int8_t)b[2 + sub]);
}

__global__ void matmul_q8_kernel(const float* __restrict__ x, const uint8_t* __restrict__ w, int m, int n, int k,
                                 float* __restrict__ out) {
  int tx = threadIdx.x, ty = threadIdx.y;
  int row = blockIdx.y * TILE + ty;
  int col = blockIdx.x * TILE + tx;
  float acc = 0.0f;
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];
  for (int kt = 0; kt < (k + TILE - 1) / TILE; ++kt) {
    int ak = kt * TILE + tx;
    int row_abs = blockIdx.y * TILE + ty;
    As[ty][tx] = (row_abs < m && ak < k) ? x[(int64_t)row_abs * k + ak] : 0.0f;
    int br = blockIdx.x * TILE + tx;
    int bk = kt * TILE + ty;
    Bs[ty][tx] = (br < n && bk < k) ? q8_load(w, (int64_t)br * k + bk) : 0.0f;
    __syncthreads();
    for (int a = 0; a < TILE; ++a) acc += As[ty][a] * Bs[a][tx];
    __syncthreads();
  }
  if (row < m && col < n) out[(int64_t)row * n + col] = acc;
}

// Warp-per-row streaming GEMV for decode (small m): fully coalesced Q8_0 reads.
template <int M>
__global__ void q8_gemv_kernel(const float* __restrict__ x, const uint8_t* __restrict__ w, int n, int k,
                               float* __restrict__ out) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (warp >= n) return;
  const uint8_t* row = w + (int64_t)warp * ((k >> 5) * 34);
  const int nblk = k >> 5;
  float acc[M];
#pragma unroll
  for (int r = 0; r < M; ++r) acc[r] = 0.0f;
  for (int b = 0; b < nblk; ++b) {
    const uint8_t* blk = row + (int64_t)b * 34;
    uint16_t ds;
    __builtin_memcpy(&ds, blk, 2);
    const float scale = fp16_to_float_u(ds);
    const int8_t q = (int8_t)blk[2 + lane];
    const float wv = scale * (float)q;
#pragma unroll
    for (int r = 0; r < M; ++r)
      acc[r] = fmaf(wv, x[(int64_t)r * k + (b << 5) + lane], acc[r]);
  }
#pragma unroll
  for (int r = 0; r < M; ++r) {
    float v = acc[r];
    v += __shfl_down_sync(~0ull, v, 16, 32);
    v += __shfl_down_sync(~0ull, v, 8, 32);
    v += __shfl_down_sync(~0ull, v, 4, 32);
    v += __shfl_down_sync(~0ull, v, 2, 32);
    v += __shfl_down_sync(~0ull, v, 1, 32);
    if (lane == 0) out[(int64_t)r * n + warp] = v;
  }
}

void launch(const float* x, const uint8_t* w, int m, int n, int k, float* out) {
  if (m <= 8) {
    const int threads = 256;
    int blocks = (n * 32 + threads - 1) / threads;
    switch (m) {
      case 1: q8_gemv_kernel<1><<<blocks, threads>>>(x, w, n, k, out); break;
      case 2: q8_gemv_kernel<2><<<blocks, threads>>>(x, w, n, k, out); break;
      case 4: q8_gemv_kernel<4><<<blocks, threads>>>(x, w, n, k, out); break;
      default: {
        for (int r0 = 0; r0 < m; r0 += 4) {
          int rows = std::min(4, m - r0);
          int blocks2 = (n * 32 + threads - 1) / threads;
#define G(MM) q8_gemv_kernel<MM><<<blocks2, threads>>>(x + (int64_t)r0 * k, w, n, k, out + (int64_t)r0 * n)
          if (rows == 1) G(1); else if (rows == 2) G(2); else G(4);
#undef G
        }
      }
    }
    return;
  }
  dim3 threads(TILE, TILE);
  dim3 blocks((n + TILE - 1) / TILE, (m + TILE - 1) / TILE);
  matmul_q8_kernel<<<blocks, threads>>>(x, w, m, n, k, out);
}

// ---- host harness ----
static uint16_t f32_to_f16(float v) {
  uint32_t b; memcpy(&b, &v, 4);
  uint32_t sign = b >> 16 & 0x8000u;
  int32_t exp = ((b >> 23) & 0xffu) - 127 + 15;
  uint32_t mant = b & 0x7fffffu;
  if (exp <= 0) return (uint16_t)(sign | (mant ? 0x340u : 0u) | (mant >> 13));
  if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
  return (uint16_t)(sign | (exp << 10) | (mant >> 13));
}

static float fp16_to_f32_h(uint16_t h) {
  uint32_t sign = (uint32_t(h & 0x8000u)) << 16;
  uint32_t exp = (h >> 10) & 0x1fu;
  uint32_t mant = h & 0x03ffu;
  uint32_t f;
  if (exp == 0) {
    if (mant == 0) f = sign;
    else {
      exp = 127 - 15 + 1;
      while ((mant & 0x0400u) == 0) { mant <<= 1; --exp; }
      mant &= 0x03ffu;
      f = sign | (exp << 23) | (mant << 13);
    }
  } else if (exp == 0x1fu) {
    f = sign | 0x7f800000u | (mant << 13);
  } else {
    f = sign | ((exp - 15 + 127) << 23) | (mant << 13);
  }
  float out; memcpy(&out, &f, 4);
  return out;
}

int run_case(int m, int n, int k) {
  std::vector<float> hx((size_t)m * k), ref((size_t)m * n), got((size_t)m * n);
  std::vector<uint8_t> hw((size_t)n * k / 32 * 34);
  srand(1234 + n + k);
  for (auto& v : hx) v = (rand() / (float)RAND_MAX - 0.5f) * 2.0f;
  // random Q8_0 blocks
  for (size_t b = 0; b < hw.size(); b += 34) {
    float s = (rand() / (float)RAND_MAX) * 0.02f + 1e-3f;
    uint16_t sh = f32_to_f16(s);
    memcpy(&hw[b], &sh, 2);
    for (int i = 0; i < 32; ++i) hw[b + 2 + i] = (uint8_t)(rand() % 255);
  }
  auto q8_host = [&hw](int64_t idx) {
    int64_t block = idx >> 5;
    const uint8_t* p = &hw[block * 34];
    uint16_t d;
    memcpy(&d, p, 2);
    return fp16_to_f32_h(d) * (float)(int8_t)p[2 + (idx & 31)];
  };
  // CPU reference
  for (int r = 0; r < m; ++r)
    for (int j = 0; j < n; ++j) {
      double acc = 0;
      for (int t = 0; t < k; ++t)
        acc += (double)hx[(size_t)r * k + t] * (double)q8_host((int64_t)j * k + t);
      ref[(size_t)r * n + j] = (float)acc;
    }
  float *dx, *dout; uint8_t* dw;
  hipMalloc(&dx, sizeof(float) * m * k); hipMalloc(&dout, sizeof(float) * m * n); hipMalloc(&dw, hw.size());
  hipMemcpy(dx, hx.data(), sizeof(float) * m * k, hipMemcpyHostToDevice);
  hipMemcpy(dw, hw.data(), hw.size(), hipMemcpyHostToDevice);
  launch(dx, dw, m, n, k, dout);
  hipError_t le = hipGetLastError();
  if (le != hipSuccess) printf("LAUNCH ERR m=%d n=%d k=%d: %s\n", m, n, k, hipGetErrorString(le));
  hipMemcpy(got.data(), dout, sizeof(float) * m * n, hipMemcpyDeviceToHost);
  double maxrel = 0; size_t badidx = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    double d = fabs(ref[i] - got[i]);
    double r = d / (fabs(ref[i]) + 1e-6);
    if (r > maxrel) { maxrel = r; badidx = i; }
  }
  // timing
  hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
  for (int i = 0; i < 20; ++i) launch(dx, dw, m, n, k, dout);
  hipDeviceSynchronize();
  int iters = 200;
  hipEventRecord(e0);
  for (int i = 0; i < iters; ++i) launch(dx, dw, m, n, k, dout);
  hipEventRecord(e1); hipEventSynchronize(e1);
  float ms; hipEventElapsedTime(&ms, e0, e1);
  double us = ms * 1000.0 / iters;
  double gbs = (hw.size() + (double)sizeof(float) * m * k) / (us * 1e-6) / 1e9;
  printf("case m=%d n=%d k=%d: maxrel=%.2e %s  time=%.1fus  eff=%.0fGB/s\n",
         m, n, k, maxrel, maxrel < 0.05 ? "OK" : "BAD", us, gbs);
  hipFree(dx); hipFree(dout); hipFree(dw);
  return maxrel < 0.05 ? 0 : 1;
}

int main() {
  int rc = 0;
  rc |= run_case(1, 2048, 2560);     // qkv+o-ish
  rc |= run_case(1, 11008, 2048);    // gate/up
  rc |= run_case(1, 2048, 11008);    // down
  rc |= run_case(1, 151936, 2048);   // lm_head
  rc |= run_case(4, 11008, 2048);    // batched decode (bench m=4)
  printf("%s\n", rc ? "FAILED" : "PASSED");
  return rc;
}
