#include "nanovllm/hip_ops.hpp"

#include <hip/amd_detail/amd_math_functions.h>

#include <cmath>
#include <cstdio>

constexpr int TILE = 16;
constexpr int RMS_THREADS = 256;
constexpr int ATTN_THREADS = 64;
constexpr int ATTN_CHUNK = 64;
constexpr int MAX_HEAD_DIM = 256;
constexpr int MAX_DIM_PER_THREAD = (MAX_HEAD_DIM + ATTN_THREADS - 1) / ATTN_THREADS;

__device__ __forceinline__ float fp16_to_float_u(uint16_t h) {
  unsigned int sign = (h & 0x8000u) << 16;
  unsigned int exp = (h >> 10) & 0x1fu;
  unsigned int mant = h & 0x3ffu;
  float f;
  if (exp == 0) {
    if (mant == 0) {
      f = __int_as_float(sign);
    } else {
      f = __int_as_float(sign | 0x7b800000u) * __int_as_float(1.0f);
    }
  } else if (exp == 0x1fu) {
    f = __int_as_float(sign | 0x7f800000u | (mant << 13));
  } else {
    f = __int_as_float(sign | ((exp - 15u + 127u) << 23) | (mant << 13));
  }
  return f;
}

__device__ __forceinline__ float bf16_to_float_u(uint16_t h) {
  return __int_as_float((unsigned)h << 16);
}

__device__ __forceinline__ float wt_load(float v, bool /*bf16*/) { return v; }
__device__ __forceinline__ float wt_load(uint16_t v, bool bf16) {
  return bf16 ? bf16_to_float_u(v) : fp16_to_float_u(v);
}

template <class TW>
__global__ void matmul_kernel(const float* __restrict__ x, const TW* __restrict__ w, int m, int n, int k,
                              float* __restrict__ out, bool w_bf16) {
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
    Bs[ty][tx] = (br < n && bk < k) ? wt_load(w[(int64_t)br * k + bk], w_bf16) : 0.0f;
    __syncthreads();
    for (int a = 0; a < TILE; ++a) acc += As[ty][a] * Bs[a][tx];
    __syncthreads();
  }
  if (row < m && col < n) out[(int64_t)row * n + col] = acc;
}

// Q8_0 stored as split planes: int8 weights (row-contiguous) + fp16 scales.
__global__ void matmul_q8_kernel(const float* __restrict__ x, const uint8_t* __restrict__ w8,
                                 const uint16_t* __restrict__ ws, int m, int n, int k, float* __restrict__ out) {
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
    int64_t idx = (int64_t)br * k + bk;
    Bs[ty][tx] = (br < n && bk < k) ? fp16_to_float_u(ws[idx >> 5]) * (float)(int8_t)w8[idx] : 0.0f;
    __syncthreads();
    for (int a = 0; a < TILE; ++a) acc += As[ty][a] * Bs[a][tx];
    __syncthreads();
  }
  if (row < m && col < n) out[(int64_t)row * n + col] = acc;
}

// QK_K=256 super-block inline dequant (native K-quants, zero VRAM bloat).
// Layouts mirror the GGUF spec: Q4_K = [fp16 d][fp16 dmin][12B scales][128B qs],
// Q6_K = [128B ql][64B qh][16 x int8 scales][fp16 d]. Element (b*256+i).
// K-quant block scalars (d, dmin) are frequently subnormal fp16; decode them with a
// mantissa-aware path (mirrors the GLSL shader + host reference), distinct from the
// legacy fp16_to_float_u used by the F16/Q8 matmul paths so those goldens are unchanged.
__device__ __forceinline__ float qk_fp16_to_float(uint16_t h) {
  unsigned int sign = (h & 0x8000u) << 16;
  unsigned int exp = (h >> 10) & 0x1fu;
  unsigned int mant = h & 0x3ffu;
  float f;
  if (exp == 0) {
    if (mant == 0) {
      f = __int_as_float(sign);
    } else {
      unsigned int e = 127 - 15 + 1, m = mant;
      while ((m & 0x400u) == 0u) { m <<= 1u; --e; }
      f = __int_as_float(sign | (e << 23) | ((m & 0x3FFu) << 13));
    }
  } else if (exp == 0x1fu) {
    f = __int_as_float(sign | 0x7f800000u | (mant << 13));
  } else {
    f = __int_as_float(sign | ((exp - 15u + 127u) << 23) | (mant << 13));
  }
  return f;
}
__device__ __forceinline__ void qk_scales_q4(const uint8_t* sc12, int j, float d, float dmin,
                                             float* s, float* mn) {
  uint8_t sc, m8;
  if (j < 4) {
    sc = sc12[j] & 63;
    m8 = sc12[j + 4] & 63;
  } else {
    sc = (sc12[j + 4] & 0xF) | ((sc12[j - 4] >> 6) << 4);
    m8 = (sc12[j + 4] >> 4) | ((sc12[j] >> 6) << 4);
  }
  *s = d * sc;
  *mn = dmin * m8;
}

__device__ __forceinline__ float qk_w_q4(const uint8_t* blk, int i) {
  uint16_t dh = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
  uint16_t dmh = (uint16_t)blk[2] | ((uint16_t)blk[3] << 8);
  float d = qk_fp16_to_float(dh), dmin = qk_fp16_to_float(dmh);
  int j = i >> 5, e = i & 31;
  float s, mn;
  qk_scales_q4(blk + 4, j, d, dmin, &s, &mn);
  uint8_t qb = blk[16 + j * 16 + (e & 15)];
  int q = (e < 16) ? (qb & 0xF) : (qb >> 4);
  return s * (float)q - mn;
}

__device__ __forceinline__ float qk_w_q5(const uint8_t* blk, int i) {
  int group = i >> 6, slot = (i & 63) >> 5, l = i & 31;
  int is = group * 2, j = is + slot;
  float d = qk_fp16_to_float((uint16_t)(blk[0] | (blk[1] << 8)));
  float dmin = qk_fp16_to_float((uint16_t)(blk[2] | (blk[3] << 8)));
  float s, mn;
  qk_scales_q4(blk + 4, j, d, dmin, &s, &mn);
  const uint8_t* ql = blk + 48;
  const uint8_t* qh = blk + 16;
  uint8_t qhl = qh[l];
  int q = (slot == 0) ? (ql[group * 32 + l] & 0xF) : (ql[group * 32 + l] >> 4);
  int qhbit = (slot == 0) ? (qhl & (1u << (group * 2))) : (qhl & (2u << (group * 2)));
  return s * (float)(q + (qhbit ? 16 : 0)) - mn;
}

__device__ __forceinline__ float qk_w_q6(const uint8_t* blk, int i) {
  int half = i >> 7, t = i & 127;
  const uint8_t* ql = blk + half * 64;
  const uint8_t* qh = blk + 128 + half * 32;
  const int8_t* sc = (const int8_t*)(blk + 192 + half * 8);
  uint16_t dh = (uint16_t)blk[208] | ((uint16_t)blk[209] << 8);
  float d = qk_fp16_to_float(dh);
  int l = t & 31, slot = t >> 5, is = l >> 4;
  uint8_t qhl = qh[l];
  int q;
  float s;
  if (slot == 0) {
    q = (ql[l] & 0xF) | (((qhl >> 0) & 3) << 4);
    s = (float)sc[is + 0];
  } else if (slot == 1) {
    q = (ql[l + 32] & 0xF) | (((qhl >> 2) & 3) << 4);
    s = (float)sc[is + 2];
  } else if (slot == 2) {
    q = (ql[l] >> 4) | (((qhl >> 4) & 3) << 4);
    s = (float)sc[is + 4];
  } else {
    q = (ql[l + 32] >> 4) | (((qhl >> 6) & 3) << 4);
    s = (float)sc[is + 6];
  }
  return d * s * ((float)q - 32.0f);
}

__device__ __forceinline__ float qk_w(const uint8_t* wq, int64_t idx, int kind, int bb) {
  int64_t b = idx >> 8;
   int i = (int)(idx & 255);
   const uint8_t* blk = wq + b * bb;
   float v;
   if (kind == 12) v = qk_w_q4(blk, i);
   else if (kind == 13) v = qk_w_q5(blk, i);
   else v = qk_w_q6(blk, i);
   return v;
}

// RDNA4 (gfx1201) has no signed V_DOT4_I32_I8 via inline asm; amd_mixed_dot
// emits signed x signed correctly (validated in tools/magpie-q8/prefill.hip).
__device__ __forceinline__ int32_t dot4_i8_signed(int32_t a, int32_t b, int32_t c) {
  char4 va, vb;
  *(int32_t*)&va = a;
  *(int32_t*)&vb = b;
  return amd_mixed_dot(va, vb, c, false);
}

// Per-row activation scale 127/max|x| for the INT8 DOT4 prefill kernel.
__global__ void q8_row_scale_kernel(const float* __restrict__ x, float* __restrict__ rs, int m, int k) {
  int r = blockIdx.x, t = threadIdx.x;
  __shared__ float red[256];
  float mx = 0.0f;
  for (int i = t; i < k; i += 256) mx = fmaxf(mx, fabsf(x[(int64_t)r * k + i]));
  red[t] = mx;
  __syncthreads();
  for (int s = 128; s > 0; s >>= 1) {
    if (t < s) red[t] = fmaxf(red[t], red[t + s]);
    __syncthreads();
  }
  if (t == 0) rs[r] = red[0] > 1e-8f ? 127.0f / red[0] : 0.0f;
}

// 64x16 tile INT8 V_DOT4 matmul over split planes (w8 row-major int8, ws fp16 per 32).
// Requires k % 16 == 0. Activations are int8-quantized per row (q8_row_scale_kernel).
__global__ __launch_bounds__(1024) void q8_tile64_dot4_kernel(
    const float* __restrict__ x, const uint8_t* __restrict__ w8, const uint16_t* __restrict__ ws,
    const float* __restrict__ arow, int m, int n, int k, float* __restrict__ out) {
  const int tx = threadIdx.x, ty = threadIdx.y;
  __shared__ int8_t As0[64][16], As1[64][16];
  __shared__ int8_t Bs0[16][16], Bs1[16][16];
  __shared__ float Sb0[16], Sb1[16];
  __shared__ float Ascl[64];
  __shared__ int use_buf;
  float acc = 0.0f;
  const int row0 = blockIdx.y * 64;
  const int col0 = blockIdx.x * 16;
  const int nblk = k / 32;
  if (tx == 0) Ascl[ty] = (row0 + ty < m) ? arow[row0 + ty] : 0.0f;
  use_buf = 0;
  __syncthreads();
  if (tx < 8) {
    int t2 = ty * 8 + tx;
    int ar = t2 >> 3, ac = (t2 & 7) * 2;
    float2 v = {0.0f, 0.0f};
    int row_abs = row0 + ar;
    if (row_abs < m && ac + 1 < k)
      v = *reinterpret_cast<const float2*>(&x[(int64_t)row_abs * k + ac]);
    float as = Ascl[ar];
    As0[ar][ac + 0] = (int8_t)fminf(127.0f, fmaxf(-127.0f, v.x * as));
    As0[ar][ac + 1] = (int8_t)fminf(127.0f, fmaxf(-127.0f, v.y * as));
  }
  if (ty < 16) {
    int col_global = col0 + tx;
    int8_t val = 0;
    if (col_global < n) val = (int8_t)w8[(int64_t)col_global * k + ty];
    Bs0[tx][ty] = val;
  }
  if (ty == 0) {
    int col_global = col0 + tx;
    Sb0[tx] = col_global < n ? fp16_to_float_u(ws[(int64_t)col_global * nblk]) : 0.0f;
  }
  __syncthreads();
  for (int kt = 0; kt < k / 16; ++kt) {
    int cur = use_buf, nxt = cur ^ 1;
    int8_t (*As)[16] = cur ? As1 : As0;
    int8_t (*As_n)[16] = cur ? As0 : As1;
    int8_t (*Bs)[16] = cur ? Bs1 : Bs0;
    int8_t (*Bs_n)[16] = cur ? Bs0 : Bs1;
    float* Sb = cur ? Sb1 : Sb0;
    float* Sb_n = cur ? Sb0 : Sb1;
    if (tx < 8) {
      int t2 = ty * 8 + tx;
      int ar = t2 >> 3, ac = (t2 & 7) * 2;
      float2 v = {0.0f, 0.0f};
      int row_abs = row0 + ar;
      if (row_abs < m && (kt + 1) * 16 + ac + 1 < k)
        v = *reinterpret_cast<const float2*>(&x[(int64_t)row_abs * k + (kt + 1) * 16 + ac]);
      float as = Ascl[ar];
      As_n[ar][ac + 0] = (int8_t)fminf(127.0f, fmaxf(-127.0f, v.x * as));
      As_n[ar][ac + 1] = (int8_t)fminf(127.0f, fmaxf(-127.0f, v.y * as));
    }
    if (ty < 16) {
      int col_global = col0 + tx;
      int8_t val = 0;
      if ((kt + 1) * 16 < k && col_global < n)
        val = (int8_t)w8[(int64_t)col_global * k + (kt + 1) * 16 + ty];
      Bs_n[tx][ty] = val;
    }
    if (ty == 0) {
      int col_global = col0 + tx;
      Sb_n[tx] = (col_global < n && (kt + 1) * 16 < k)
                     ? fp16_to_float_u(ws[(int64_t)col_global * nblk + (kt + 1) / 2])
                     : 0.0f;
    }
    int32_t c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    int32_t a0 = *(int32_t*)&As[ty][0];
    int32_t a1 = *(int32_t*)&As[ty][4];
    int32_t a2 = *(int32_t*)&As[ty][8];
    int32_t a3 = *(int32_t*)&As[ty][12];
    int32_t b0 = *(int32_t*)&Bs[tx][0];
    int32_t b1 = *(int32_t*)&Bs[tx][4];
    int32_t b2 = *(int32_t*)&Bs[tx][8];
    int32_t b3 = *(int32_t*)&Bs[tx][12];
    c0 = dot4_i8_signed(a0, b0, c0);
    c1 = dot4_i8_signed(a1, b1, c1);
    c2 = dot4_i8_signed(a2, b2, c2);
    c3 = dot4_i8_signed(a3, b3, c3);
    acc += Sb[tx] * (float)(c0 + c1 + c2 + c3);
    __syncthreads();
    use_buf = nxt;
  }
  int row = row0 + ty;
  int col = col0 + tx;
  float adq = Ascl[ty] > 0.0f ? 1.0f / Ascl[ty] : 0.0f;
  if (row < m && col < n) out[(int64_t)row * n + col] = acc * adq;
}

// Warp-per-row streaming GEMV for decode (small m): lane-owns-4-int8-activations
// + 4-int8-weight u32, reduced via one V_DOT4_I32_I8 (rank 3 INT8 DOT). Activations
// are int8-quantized per row (q8_row_scale_kernel, symmetric); dequant at write.
template <int M>
__global__ void q8_gemv_kernel(const float* __restrict__ x, const uint32_t* __restrict__ w4,
                               const uint16_t* __restrict__ ws, const float* __restrict__ arow,
                               int n, int k, float* __restrict__ out) {
  const int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (gw >= n) return;
  const uint32_t* row = w4 + (int64_t)gw * (k >> 2);
  const uint16_t* rows = ws + (int64_t)gw * (k >> 5);
  const int nit = k >> 7;
  float acc[M];
#pragma unroll
  for (int r = 0; r < M; ++r) acc[r] = 0.0f;
  for (int i = 0; i < nit; ++i) {
    const uint32_t q = row[i * 32 + lane];
    const float s = fp16_to_float_u(rows[(i << 2) + (lane >> 3)]);
    const int64_t xb = (int64_t)i * 128 + lane * 4;
    const int32_t qi = (int32_t)q;
#pragma unroll
    for (int r = 0; r < M; ++r) {
      float as = arow[r];
      if (as > 0.0f) {
        const float4 xv = *reinterpret_cast<const float4*>(&x[(int64_t)r * k + xb]);
        int8_t xa[4];
        xa[0] = (int8_t)fminf(127.0f, fmaxf(-127.0f, xv.x * as));
        xa[1] = (int8_t)fminf(127.0f, fmaxf(-127.0f, xv.y * as));
        xa[2] = (int8_t)fminf(127.0f, fmaxf(-127.0f, xv.z * as));
        xa[3] = (int8_t)fminf(127.0f, fmaxf(-127.0f, xv.w * as));
        int32_t xq = *reinterpret_cast<const int32_t*>(xa);
        acc[r] += s * (float)dot4_i8_signed(xq, qi, 0);
      }
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
      float as = arow[r];
      out[(int64_t)r * n + gw] = as > 0.0f ? t * (1.0f / as) : 0.0f;
    }
  }
}

__global__ void add_bias_kernel(float* __restrict__ x, const float* __restrict__ bias, int rows, int cols) {
  int row = blockIdx.x;
  int t = threadIdx.x;
  float* xr = x + (int64_t)row * cols;
  for (int c = t; c < cols; c += blockDim.x) xr[c] += bias[c];
}

__global__ void rms_norm_kernel(const float* __restrict__ x, const float* __restrict__ weight, float* __restrict__ out,
                                int hidden, float eps) {
  int r = blockIdx.x;
  int t = threadIdx.x;
  __shared__ float red[RMS_THREADS];
  float sum = 0.0f;
  const float* xr = x + (int64_t)r * hidden;
  for (int i = t; i < hidden; i += RMS_THREADS) {
    float v = xr[i];
    sum += v * v;
  }
  red[t] = sum;
  __syncthreads();
  for (int s = RMS_THREADS / 2; s > 0; s >>= 1) {
    if (t < s) red[t] += red[t + s];
    __syncthreads();
  }
  float inv = rsqrtf(red[0] / hidden + eps);
  float* orow = out + (int64_t)r * hidden;
  for (int i = t; i < hidden; i += RMS_THREADS) orow[i] = xr[i] * inv * weight[i];
}

__global__ void rms_norm_add_kernel(const float* __restrict__ x, float* __restrict__ residual,
                                    const float* __restrict__ weight, float* __restrict__ out, int hidden, float eps) {
  int r = blockIdx.x;
  int t = threadIdx.x;
  __shared__ float red[RMS_THREADS];
  const float* xr = x + (int64_t)r * hidden;
  float* rr = residual + (int64_t)r * hidden;
  float sum = 0.0f;
  for (int i = t; i < hidden; i += RMS_THREADS) {
    float v = xr[i] + rr[i];
    rr[i] = v;
    sum += v * v;
  }
  red[t] = sum;
  __syncthreads();
  for (int s = RMS_THREADS / 2; s > 0; s >>= 1) {
    if (t < s) red[t] += red[t + s];
    __syncthreads();
  }
  float inv = rsqrtf(red[0] / hidden + eps);
  float* orow = out + (int64_t)r * hidden;
  for (int i = t; i < hidden; i += RMS_THREADS) orow[i] = rr[i] * inv * weight[i];
}

__global__ void silu_and_mul_kernel(const float* __restrict__ gate_up, float* __restrict__ out, int rows,
                                    int intermediate) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows * intermediate;
  if (idx >= total) return;
  int r = idx / intermediate;
  int c = idx % intermediate;
  float gate = gate_up[(int64_t)r * (2 * intermediate) + c];
  float up = gate_up[(int64_t)r * (2 * intermediate) + intermediate + c];
  out[(int64_t)r * intermediate + c] = gate / (1.0f + expf(-gate)) * up;
}

__global__ void rope_kernel(float* __restrict__ data, const int64_t* __restrict__ positions, int tokens, int heads,
                            int head_dim, int64_t stride, const float* __restrict__ inv_freq) {
  int half = head_dim / 2;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = tokens * heads * half;
  if (idx >= total) return;
  int per_token = heads * half;
  int token = idx / per_token;
  int rem = idx % per_token;
  int head = rem / half;
  int d = rem % half;
  float pos = static_cast<float>(positions[token]);
  float freq = pos * inv_freq[d];
  float c = cosf(freq), s = sinf(freq);
  int64_t base = (int64_t)token * stride + (int64_t)head * head_dim;
  float x1 = data[base + d];
  float x2 = data[base + d + half];
  data[base + d] = x1 * c - x2 * s;
  data[base + d + half] = x2 * c + x1 * s;
}

__global__ void store_kv_kernel(const float* __restrict__ key, const float* __restrict__ value,
                                float* __restrict__ k_cache, float* __restrict__ v_cache,
                                const int32_t* __restrict__ slot_mapping, int kv_heads, int head_dim, int total) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  int d = kv_heads * head_dim;
  int token = idx / d;
  int off = idx % d;
  int32_t slot = slot_mapping[token];
  if (slot < 0) return;
  int64_t src = (int64_t)token * d + off;
  int64_t dst = (int64_t)slot * d + off;
  k_cache[dst] = key[src];
  v_cache[dst] = value[src];
}

__device__ inline const float* kv_ptr(const float* cache, int block_idx, int offset, int kv_heads, int head_dim,
                                      int block_size, int kv_head) {
  return cache + ((int64_t)(block_idx * block_size + offset) * kv_heads + kv_head) * head_dim;
}

__global__ void paged_attention_kernel(const float* __restrict__ q, float* __restrict__ out,
                                       const float* __restrict__ k_cache, const float* __restrict__ v_cache,
                                       const int32_t* __restrict__ query_seq, const int32_t* __restrict__ query_key_len,
                                       const int32_t* __restrict__ block_tables, int q_heads, int kv_heads,
                                       int head_dim, int block_size, int max_blocks, float scale) {
  if (head_dim > MAX_HEAD_DIM) return;
  int token = blockIdx.x;
  int head = blockIdx.y;
  int t = threadIdx.x;
  int seq = query_seq[token];
  int key_len = query_key_len[token];
  int kv_head = head / (q_heads / kv_heads);
  int q_stride = q_heads * head_dim;
  int kv_stride = kv_heads * head_dim;
  __shared__ float q_sh[MAX_HEAD_DIM];
  __shared__ float scores[ATTN_CHUNK];

  for (int d = t; d < head_dim; d += ATTN_THREADS) q_sh[d] = q[(int64_t)token * q_stride + head * head_dim + d];
  __syncthreads();

  if (key_len <= 0) {
    for (int d = t; d < head_dim; d += ATTN_THREADS) out[(int64_t)token * q_stride + head * head_dim + d] = 0.0f;
    return;
  }

  float m = -1e30f, s = 0.0f;
  float acc[MAX_DIM_PER_THREAD] = {};

  for (int key0 = 0; key0 < key_len; key0 += ATTN_CHUNK) {
    __syncthreads();
    int n = key_len - key0;
    if (n > ATTN_CHUNK) n = ATTN_CHUNK;
    float sc = -1e30f;
    if (t < n) {
      int key_idx = key0 + t;
      int block_idx = block_tables[(int64_t)seq * max_blocks + (key_idx / block_size)];
      int offset = key_idx % block_size;
      const float* kp = kv_ptr(k_cache, block_idx, offset, kv_heads, head_dim, block_size, kv_head);
      float dot = 0.0f;
      for (int d = 0; d < head_dim; ++d) dot += q_sh[d] * kp[d];
      sc = dot * scale;
    }
    scores[t] = sc;
    __syncthreads();

    float chunk_m = -1e30f;
    for (int i = 0; i < n; ++i) chunk_m = fmaxf(chunk_m, scores[i]);
    float sum_e = 0.0f;
    for (int i = 0; i < n; ++i) sum_e += expf(scores[i] - chunk_m);

    float new_m = fmaxf(m, chunk_m);
    float m_scale = expf(m - new_m);
    float chunk_scale = expf(chunk_m - new_m);
    s = s * m_scale + sum_e * chunk_scale;
    for (int a = 0; a < MAX_DIM_PER_THREAD; ++a) acc[a] *= m_scale;

    for (int local = 0; local < MAX_DIM_PER_THREAD; ++local) {
      int d = local * ATTN_THREADS + t;
      if (d >= head_dim) continue;
      float val = 0.0f;
      for (int j = 0; j < n; ++j) {
        int key_idx = key0 + j;
        int block_idx = block_tables[(int64_t)seq * max_blocks + (key_idx / block_size)];
        int offset = key_idx % block_size;
        const float* vp = kv_ptr(v_cache, block_idx, offset, kv_heads, head_dim, block_size, kv_head);
        val += expf(scores[j] - chunk_m) * vp[d];
      }
      acc[local] += chunk_scale * val;
    }
    m = new_m;
  }

  float inv_s = s > 0.0f ? 1.0f / s : 0.0f;
  for (int local = 0; local < MAX_DIM_PER_THREAD; ++local) {
    int d = local * ATTN_THREADS + t;
    if (d >= head_dim) continue;
    out[(int64_t)token * q_stride + head * head_dim + d] = acc[local] * inv_s;
  }
}

template <class TW>
__global__ void embedding_kernel(const int64_t* __restrict__ ids, const TW* __restrict__ table, int hidden,
                                 int total, float* __restrict__ out, bool w_bf16) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  int token = idx / hidden;
  int col = idx % hidden;
  int64_t id = ids[token];
  TW v = table[(int64_t)id * hidden + col];
  out[(int64_t)token * hidden + col] = wt_load(v, w_bf16);
}

__global__ void embedding_q8_kernel(const int64_t* __restrict__ ids, const uint8_t* __restrict__ w8,
                                    const uint16_t* __restrict__ ws, int hidden, int total, float* __restrict__ out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  int token = idx / hidden;
  int col = idx % hidden;
  int64_t id = ids[token];
  int64_t e = id * hidden + col;
  out[(int64_t)token * hidden + col] = fp16_to_float_u(ws[e >> 5]) * (float)(int8_t)w8[e];
}

// TILE=16 twin of matmul_q8_kernel over raw K-quant super-blocks (same grid math).
__global__ void matmul_qk_kernel(const float* __restrict__ x, const uint8_t* __restrict__ wq,
                                 int m, int n, int k, float* __restrict__ out, int kind, int bb) {
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
    int64_t idx = (int64_t)br * k + bk;
    Bs[ty][tx] = (br < n && bk < k) ? qk_w(wq, idx, kind, bb) : 0.0f;
    __syncthreads();
    for (int a = 0; a < TILE; ++a) acc += As[ty][a] * Bs[a][tx];
    __syncthreads();
  }
  if (row < m && col < n) out[(int64_t)row * n + col] = acc;
}

__global__ void embedding_qk_kernel(const int64_t* __restrict__ ids, const uint8_t* __restrict__ wq,
                                    int hidden, int total, float* __restrict__ out, int kind, int bb) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  int token = idx / hidden;
  int col = idx % hidden;
  int64_t id = ids[token];
  int64_t e = id * hidden + col;
  out[(int64_t)token * hidden + col] = qk_w(wq, e, kind, bb);
}

__global__ void gather_rows_kernel(const float* __restrict__ rows, const int32_t* __restrict__ indices, int cols,
                                   int total, float* __restrict__ out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  int out_row = idx / cols;
  int col = idx % cols;
  int32_t src = indices[out_row];
  out[(int64_t)idx] = rows[(int64_t)src * cols + col];
}

__global__ void copy_host_to_device_kernel(const float* /*unused*/, float* /*unused*/) {}

void hip_matmul(const float* x, const float* w, int m, int n, int k, float* out) {
  if (m <= 0 || n <= 0 || k <= 0) return;
  dim3 blocks((n + TILE - 1) / TILE, (m + TILE - 1) / TILE);
  dim3 threads(TILE, TILE);
  matmul_kernel<float><<<blocks, threads>>>(x, w, m, n, k, out, false);
  HIP_CHECK(hipGetLastError());
}

void hip_matmul_u16(const float* x, const uint16_t* w, int m, int n, int k, float* out, bool bf16) {
  if (m <= 0 || n <= 0 || k <= 0) return;
  dim3 blocks((n + TILE - 1) / TILE, (m + TILE - 1) / TILE);
  dim3 threads(TILE, TILE);
  matmul_kernel<uint16_t><<<blocks, threads>>>(x, w, m, n, k, out, bf16);
  HIP_CHECK(hipGetLastError());
}

void hip_matmul_q8(const float* x, const uint8_t* w8, const uint16_t* ws, int m, int n, int k, float* out) {
  if (m <= 0 || n <= 0 || k <= 0) return;
  if (m <= 8 && (k & 127) == 0) {
    static float* arow = nullptr;
    static int arow_cap = 0;
    if (m > arow_cap) {  // ponytail: grow-only scratch; engine is single-threaded
      if (arow) HIP_CHECK(hipFree(arow));
      HIP_CHECK(hipMalloc(&arow, (size_t)m * sizeof(float)));
      arow_cap = m;
    }
    q8_row_scale_kernel<<<m, 256>>>(x, arow, m, k);
    const int threads = 256;
    dim3 blocks((n * 32 + threads - 1) / threads);
#define NANO_Q8_GEMV(MM) q8_gemv_kernel<MM><<<blocks, threads>>>(x, reinterpret_cast<const uint32_t*>(w8), ws, arow, n, k, out)
    switch (m) {
      case 1: NANO_Q8_GEMV(1); break;
      case 2: NANO_Q8_GEMV(2); break;
      case 3: NANO_Q8_GEMV(3); break;
      case 4: NANO_Q8_GEMV(4); break;
      case 5: NANO_Q8_GEMV(5); break;
      case 6: NANO_Q8_GEMV(6); break;
      case 7: NANO_Q8_GEMV(7); break;
      default: NANO_Q8_GEMV(8); break;
    }
#undef NANO_Q8_GEMV
    HIP_CHECK(hipGetLastError());
    return;
  }
  if ((k & 15) == 0 && k >= 32) {
    static float* arow = nullptr;
    static int arow_cap = 0;
    if (m > arow_cap) {  // ponytail: grow-only scratch; engine is single-threaded
      if (arow) HIP_CHECK(hipFree(arow));
      HIP_CHECK(hipMalloc(&arow, (size_t)m * sizeof(float)));
      arow_cap = m;
    }
    q8_row_scale_kernel<<<m, 256>>>(x, arow, m, k);
    dim3 t64(16, 64);
    dim3 b64((n + 15) / 16, (m + 63) / 64);
    q8_tile64_dot4_kernel<<<b64, t64>>>(x, w8, ws, arow, m, n, k, out);
    HIP_CHECK(hipGetLastError());
    return;
  }
  dim3 blocks((n + TILE - 1) / TILE, (m + TILE - 1) / TILE);
  dim3 threads(TILE, TILE);
  matmul_q8_kernel<<<blocks, threads>>>(x, w8, ws, m, n, k, out);
  HIP_CHECK(hipGetLastError());
}

void hip_add_bias_inplace(float* x, const float* bias, int rows, int cols) {
  if (rows <= 0 || cols <= 0) return;
  add_bias_kernel<<<rows, 256>>>(x, bias, rows, cols);
  HIP_CHECK(hipGetLastError());
}

void hip_rms_norm(const float* x, const float* weight, float* out, int rows, int hidden, float eps) {
  if (rows <= 0) return;
  rms_norm_kernel<<<rows, RMS_THREADS>>>(x, weight, out, hidden, eps);
  HIP_CHECK(hipGetLastError());
}

void hip_rms_norm_add(const float* x, float* residual, const float* weight, float* out, int rows, int hidden,
                      float eps) {
  if (rows <= 0) return;
  rms_norm_add_kernel<<<rows, RMS_THREADS>>>(x, residual, weight, out, hidden, eps);
  HIP_CHECK(hipGetLastError());
}

void hip_silu_and_mul(const float* gate_up, float* out, int rows, int intermediate) {
  if (rows <= 0 || intermediate <= 0) return;
  int total = rows * intermediate;
  int block = 256;
  silu_and_mul_kernel<<<(total + block - 1) / block, block>>>(gate_up, out, rows, intermediate);
  HIP_CHECK(hipGetLastError());
}

void hip_rope_inplace(float* data, const int64_t* positions, int tokens, int heads, int head_dim, int64_t stride,
                      const float* inv_freq) {
  int total = tokens * heads * (head_dim / 2);
  if (total <= 0) return;
  int block = 256;
  rope_kernel<<<(total + block - 1) / block, block>>>(data, positions, tokens, heads, head_dim, stride, inv_freq);
  HIP_CHECK(hipGetLastError());
}

void hip_store_kv(const float* key, const float* value, float* k_cache, float* v_cache, const int32_t* slot_mapping,
                  int tokens, int kv_heads, int head_dim) {
  int d = kv_heads * head_dim;
  if (tokens <= 0 || d <= 0) return;
  int total = tokens * d;
  int block = 256;
  store_kv_kernel<<<(total + block - 1) / block, block>>>(key, value, k_cache, v_cache, slot_mapping, kv_heads,
                                                          head_dim, total);
  HIP_CHECK(hipGetLastError());
}

void hip_paged_attention(const float* q, float* out, const float* k_cache, const float* v_cache,
                         const int32_t* query_seq, const int32_t* query_key_len, const int32_t* block_tables,
                         int tokens, int q_heads, int kv_heads, int head_dim, int block_size, int max_blocks,
                         float scale) {
  if (tokens <= 0 || q_heads <= 0 || kv_heads <= 0) return;
  dim3 blocks(tokens, q_heads);
  paged_attention_kernel<<<blocks, ATTN_THREADS>>>(q, out, k_cache, v_cache, query_seq, query_key_len, block_tables,
                                                   q_heads, kv_heads, head_dim, block_size, max_blocks, scale);
  HIP_CHECK(hipGetLastError());
}

void hip_embedding(const int64_t* ids, const float* table, int tokens, int hidden, float* out) {
  if (tokens <= 0 || hidden <= 0) return;
  int total = tokens * hidden;
  int block = 256;
  embedding_kernel<float><<<(total + block - 1) / block, block>>>(ids, table, hidden, total, out, false);
  HIP_CHECK(hipGetLastError());
}

void hip_embedding_u16(const int64_t* ids, const uint16_t* table, int tokens, int hidden, float* out, bool bf16) {
  if (tokens <= 0 || hidden <= 0) return;
  int total = tokens * hidden;
  int block = 256;
  embedding_kernel<uint16_t><<<(total + block - 1) / block, block>>>(ids, table, hidden, total, out, bf16);
  HIP_CHECK(hipGetLastError());
}

void hip_embedding_q8(const int64_t* ids, const uint8_t* w8, const uint16_t* ws, int tokens, int hidden, float* out) {
  if (tokens <= 0 || hidden <= 0) return;
  int total = tokens * hidden;
  int block = 256;
  embedding_q8_kernel<<<(total + block - 1) / block, block>>>(ids, w8, ws, hidden, total, out);
  HIP_CHECK(hipGetLastError());
}

void hip_matmul_qk(const float* x, const uint8_t* wq, int m, int n, int k, float* out, int kind) {
  if (m <= 0 || n <= 0 || k <= 0) return;
  int bb = (kind == 12) ? 144 : (kind == 13 ? 176 : 210);
  dim3 blocks((n + TILE - 1) / TILE, (m + TILE - 1) / TILE);
  dim3 threads(TILE, TILE);
  matmul_qk_kernel<<<blocks, threads>>>(x, wq, m, n, k, out, kind, bb);
  HIP_CHECK(hipGetLastError());
}

void hip_embedding_qk(const int64_t* ids, const uint8_t* wq, int tokens, int hidden, float* out, int kind) {
  if (tokens <= 0 || hidden <= 0) return;
  int total = tokens * hidden;
  int block = 256;
  int bb = (kind == 12) ? 144 : (kind == 13 ? 176 : 210);
  embedding_qk_kernel<<<(total + block - 1) / block, block>>>(ids, wq, hidden, total, out, kind, bb);
  HIP_CHECK(hipGetLastError());
}

void hip_gather_rows(const float* rows, const int32_t* indices, int rows_count, int cols, int out_rows, float* out) {
  (void)rows_count;
  if (out_rows <= 0 || cols <= 0) return;
  int total = out_rows * cols;
  int block = 256;
  gather_rows_kernel<<<(total + block - 1) / block, block>>>(rows, indices, cols, total, out);
  HIP_CHECK(hipGetLastError());
}

void hip_copy_to_host(const float* dev, std::vector<float>& out, size_t n) {
  out.resize(n);
  if (n == 0) return;
  HIP_CHECK(hipMemcpy(out.data(), dev, n * sizeof(float), hipMemcpyDeviceToHost));
}

void hip_copy_from_host(const std::vector<float>& host, float* dev) {
  if (host.empty()) return;
  HIP_CHECK(hipMemcpy(dev, host.data(), host.size() * sizeof(float), hipMemcpyHostToDevice));
}
