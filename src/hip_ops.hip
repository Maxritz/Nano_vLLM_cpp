#include "nanovllm/hip_ops.hpp"

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
    if (mant == 0) f = __int_as_float(sign);
    else f = __int_as_float(sign | 0x7b800000u) * __int_as_float(1.0f);
  } else if (exp == 0x1fu) {
    f = __int_as_float(sign | 0x7f800000u | (mant << 13));
  } else {
    f = __int_as_float(sign | ((exp - 15 + 127) << 23) | (mant << 13));
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

// Warp-per-row streaming GEMV for decode (small m): lane-owns-u32 coalesced reads
// (128B/wavefront), float4 activation loads. Requires k % 128 == 0.
// Reduce via LDS so it is wave32/wave64 agnostic.
template <int M>
__global__ void q8_gemv_kernel(const float* __restrict__ x, const uint32_t* __restrict__ w4,
                               const uint16_t* __restrict__ ws, int n, int k, float* __restrict__ out) {
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
    const int threads = 256;
    dim3 blocks((n * 32 + threads - 1) / threads);
#define NANO_Q8_GEMV(MM) q8_gemv_kernel<MM><<<blocks, threads>>>(x, reinterpret_cast<const uint32_t*>(w8), ws, n, k, out)
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
