#include "nanovllm/hip_ops.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

static void test_attention();

int main() {

  std::vector<float> x = {1, 2, 3, 4, 5, 6};
  std::vector<float> w = {1, 0, 0, 1, 1, 1};
  DevVec<float> dx(x), dw(w);
  DeviceBuf<float> out_buf;
  out_buf.alloc(4);
  hip_matmul(dx.d, dw.d, 2, 2, 3, out_buf.ptr());
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> out;
  hip_copy_to_host(out_buf.ptr(), out, 4);
  assert(out[0] == 1.0f && out[1] == 6.0f && out[2] == 4.0f && out[3] == 15.0f);

  std::vector<float> xs = {1, 2, 3, 4};
  std::vector<float> wt = {1, 1, 1, 1};
  DevVec<float> dxs(xs), dwt(wt);
  DeviceBuf<float> norm;
  norm.alloc(4);
  hip_rms_norm(dxs.d, dwt.d, norm.ptr(), 1, 4, 1e-6f);
  hip_copy_to_host(norm.ptr(), out, 4);
  float expected0 = 1.0f / std::sqrt((1.0f + 4.0f + 9.0f + 16.0f) / 4.0f + 1e-6f);
  assert(out[0] > 0.0f && out[3] > out[0] && std::fabs(out[0] - expected0) < 1e-4f);
  std::printf("basic matmul/rmsnorm OK\n");
  test_attention();
}

static void test_attention() {
  // tokens=2, q_heads=1, kv_heads=1, head_dim=2, block_size=2, one block id=0
  std::vector<float> q = {1.0f, 0.0f, 0.0f, 1.0f};
  std::vector<float> k_cache = {1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  std::vector<float> v_cache = {1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  std::vector<int32_t> tables = {0};
  std::vector<int32_t> seqs = {0, 0};
  std::vector<int32_t> lens = {1, 2};
  DevVec<float> dq(q), dk(k_cache), dv(v_cache);
  DevVec<int32_t> dt(tables), ds(seqs), dl(lens);
  DeviceBuf<float> out;
  out.alloc(4);
  hip_paged_attention(dq.d, out.ptr(), dk.d, dv.d, ds.d, dl.d, dt.d, 2, 1, 1, 2, 2, 1, 1.0f);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> res;
  hip_copy_to_host(out.ptr(), res, 4);
  assert(std::fabs(res[0] - 1.0f) < 1e-4f && std::fabs(res[1] - 0.0f) < 1e-4f);
  float p0 = std::exp(-1.0f) / (1.0f + std::exp(-1.0f));
  assert(std::fabs(res[2] - p0) < 1e-4f && std::fabs(res[3] - (1.0f - p0)) < 1e-4f);
  std::printf("basic attention OK\n");
}

