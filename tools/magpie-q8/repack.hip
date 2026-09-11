// magpie-q8/repack.hip — candidate C: split Q8_0 planes (int8 stream + fp16 scales).
// Requires k % 128 == 0. Repack happens ONCE in prep(); timing measures kernels only.
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

template <int M>
__global__ void q8_gemv2_kernel(const float* __restrict__ x, const uint32_t* __restrict__ w4,
                                const uint16_t* __restrict__ ws, int n, int k, float* __restrict__ out) {
  const int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (gw >= n) return;
  const uint32_t* row = w4 + (int64_t)gw * (k >> 2);
  const uint16_t* rows = ws + (int64_t)gw * (k >> 5);
  const int nit = k >> 7;  // 128 elements per iteration
  float acc[M];
#pragma unroll
  for (int r = 0; r < M; ++r) acc[r] = 0.0f;
  for (int i = 0; i < nit; ++i) {
    const uint32_t q = row[i * 32 + lane];
    const float s = fp16_to_float_u(rows[(i << 2) + (lane >> 3)]);
    const int64_t xb = (int64_t)i * 128 + lane * 4;
    const float q0 = s * (float)(int8_t)(q & 0xff);
    const float q1 = s * (float)(int8_t)((q >> 8) & 0xff);
    const float q2 = s * (float)(int8_t)((q >> 16) & 0xff);
    const float q3 = s * (float)(int8_t)(q >> 24);
#pragma unroll
    for (int r = 0; r < M; ++r) {
      const float4 xv = *reinterpret_cast<const float4*>(&x[(int64_t)r * k + xb]);
      acc[r] = fmaf(q0, xv.x, fmaf(q1, xv.y, fmaf(q2, xv.z, fmaf(q3, xv.w, acc[r]))));
    }
  }
  __shared__ float sred[8][M][32];
  const int wp = threadIdx.x >> 5;
#pragma unroll
  for (int r = 0; r < M; ++r) sred[wp][r][lane] = acc[r];
  __syncthreads();
  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < M; ++r) {
      float t = 0.0f;
      for (int l = 0; l < 32; ++l) t += sred[wp][r][l];
      out[(int64_t)r * n + gw] = t;
    }
  }
}

void repack(const std::vector<uint8_t>& hw, int n, int k, std::vector<uint32_t>& w4, std::vector<uint16_t>& ws) {
  w4.resize((size_t)n * k / 4);
  ws.resize((size_t)n * k / 32);
  for (int64_t j = 0; j < n; ++j)
    for (int64_t b = 0; b < k / 32; ++b) {
      const uint8_t* blk = &hw[((int64_t)j * k / 32 + b) * 34];
      memcpy(&ws[(size_t)j * k / 32 + b], blk, 2);
      memcpy(&w4[(size_t)j * k / 4 + b * 8], blk + 2, 32);
    }
}

struct Repacked { uint32_t* w4; uint16_t* ws; };

int run_case(int m, int n, int k) {
  std::vector<float> hx((size_t)m * k), ref((size_t)m * n), got((size_t)m * n);
  std::vector<uint8_t> hw((size_t)n * k / 32 * 34);
  srand(1234 + n + k);
  for (auto& v : hx) v = (rand() / (float)RAND_MAX - 0.5f) * 2.0f;
  for (size_t b = 0; b < hw.size(); b += 34) {
    float s = (rand() / (float)RAND_MAX) * 0.02f + 1e-3f;
    uint16_t sh;
    { uint32_t b32; float f = s; memcpy(&b32, &f, 4);
      uint32_t sign = b32 >> 16 & 0x8000u; int32_t exp = ((b32 >> 23) & 0xffu) - 127 + 15; uint32_t mant = b32 & 0x7fffffu;
      if (exp <= 0) sh = (uint16_t)(sign | 0x340u); else if (exp >= 31) sh = (uint16_t)(sign | 0x7c00u);
      else sh = (uint16_t)(sign | (exp << 10) | (mant >> 13)); }
    memcpy(&hw[b], &sh, 2);
    for (int i = 0; i < 32; ++i) hw[b + 2 + i] = (uint8_t)(rand() % 255);
  }
  auto q8_host = [&hw](int64_t idx) {
    int64_t block = idx >> 5;
    const uint8_t* p = &hw[block * 34];
    uint16_t d;
    memcpy(&d, p, 2);
    uint32_t sign = (uint32_t(d & 0x8000u)) << 16;
    uint32_t exp = (d >> 10) & 0x1fu, mant = d & 0x03ffu, f;
    if (exp == 0) f = sign;
    else if (exp == 0x1fu) f = sign | 0x7f800000u | (mant << 13);
    else f = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    float sf; memcpy(&sf, &f, 4);
    return sf * (float)(int8_t)p[2 + (idx & 31)];
  };
  for (int r = 0; r < m; ++r)
    for (int j = 0; j < n; ++j) {
      double acc = 0;
      for (int t = 0; t < k; ++t)
        acc += (double)hx[(size_t)r * k + t] * (double)q8_host((int64_t)j * k + t);
      ref[(size_t)r * n + j] = (float)acc;
    }
  float *dx, *dout;
  hipMalloc(&dx, sizeof(float) * m * k); hipMalloc(&dout, sizeof(float) * m * n);
  hipMemcpy(dx, hx.data(), sizeof(float) * m * k, hipMemcpyHostToDevice);
  const int threads = 256;
  dim3 blocks((n * 32 + threads - 1) / threads);
  Repacked rp{};
  auto launch_gemv = [&](int mm) {
#define G2(MM) q8_gemv2_kernel<MM><<<blocks, threads>>>(dx, rp.w4, rp.ws, n, k, dout)
    switch (mm) { case 1: G2(1); break; case 2: G2(2); break; case 4: G2(4); break; }
#undef G2
  };
  if (m <= 8 && k % 128 == 0) {
    std::vector<uint32_t> h4; std::vector<uint16_t> hs;
    repack(hw, n, k, h4, hs);
    hipMalloc(&rp.w4, h4.size() * 4); hipMalloc(&rp.ws, hs.size() * 2);
    hipMemcpy(rp.w4, h4.data(), h4.size() * 4, hipMemcpyHostToDevice);
    hipMemcpy(rp.ws, hs.data(), hs.size() * 2, hipMemcpyHostToDevice);
  }
  auto run = [&]() {
    if (rp.w4) launch_gemv(m);
    else {
      static uint8_t* dw = nullptr;
      if (!dw) { hipMalloc(&dw, hw.size()); hipMemcpy(dw, hw.data(), hw.size(), hipMemcpyHostToDevice); }
      dim3 bt(TILE, TILE); dim3 bg((n + TILE - 1) / TILE, (m + TILE - 1) / TILE);
      matmul_q8_kernel<<<bg, bt>>>(dx, dw, m, n, k, dout);
    }
  };
  run();
  hipMemcpy(got.data(), dout, sizeof(float) * m * n, hipMemcpyDeviceToHost);
  double maxrel = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    double d = fabs(ref[i] - got[i]);
    double r = d / (fabs(ref[i]) + 1e-6);
    if (r > maxrel) maxrel = r;
  }
  hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
  for (int i = 0; i < 20; ++i) run();
  hipDeviceSynchronize();
  int iters = 200;
  hipEventRecord(e0);
  for (int i = 0; i < iters; ++i) run();
  hipEventRecord(e1); hipEventSynchronize(e1);
  float ms; hipEventElapsedTime(&ms, e0, e1);
  double us = ms * 1000.0 / iters;
  double gbs = (hw.size() + (double)sizeof(float) * m * k) / (us * 1e-6) / 1e9;
  printf("case m=%d n=%d k=%d: maxrel=%.2e %s  time=%.1fus  eff=%.0fGB/s\n",
         m, n, k, maxrel, maxrel < 0.05 ? "OK" : "BAD", us, gbs);
  hipFree(dx); hipFree(dout); if (rp.w4) { hipFree(rp.w4); hipFree(rp.ws); }
  return maxrel < 0.05 ? 0 : 1;
}

int main() {
  int rc = 0;
  rc |= run_case(1, 2048, 2560);
  rc |= run_case(1, 11008, 2048);
  rc |= run_case(1, 2048, 11008);
  rc |= run_case(1, 151936, 2048);
  rc |= run_case(4, 11008, 2048);
  printf("%s\n", rc ? "FAILED" : "PASSED");
  return rc;
}
