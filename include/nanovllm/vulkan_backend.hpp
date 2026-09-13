#pragma once

// Vulkan backend for nano-vllm. Mirrors the hip_ops / DevVec / Qwen3Model
// surface, but backed by VAiT vkruntime (device-local allocator) + the GLSL
// kernels in shaders/vulkan/*.comp (SPIR-V embedded in spv.h). Uses
// VK_KHR_push_descriptor so there is ZERO per-dispatch descriptor-pool churn
// (this is what makes a many-op model forward feasible, unlike VAiT vkblas's
// fixed 64-set pool which leaks one set per vkblas_qgemm_* call).
//
// Perf reference port: AMD RDNA2/3/4 detection (apiVersion>=1.4 || devID 0x7550)
// and force-device-local-VRAM policy, matching vulkan-amd-rdna4-perf-fix.patch
// (disable_host_visible_vidmem=true for AMD dGPUs).

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <vulkan/vulkan.h>

#include "gguf.hpp"

#include "vkruntime/vkruntime.h"
#include "spv.h"

// Forward-declared handle types; backend owns the lifetime.
struct VBuf {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkDeviceSize nbytes = 0;
  VBuf() = default;
  VBuf(const VBuf&) = delete;
  VBuf& operator=(const VBuf&) = delete;
  VBuf(VBuf&& o) noexcept;
  VBuf& operator=(VBuf&& o) noexcept;
  ~VBuf();
  void alloc(VkDeviceSize bytes, VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  void upload(const void* host, VkDeviceSize bytes);
  void download(void* host, VkDeviceSize bytes) const;
  void free();
  uint32_t words() const; // for fp32 views
};

// Q8_0 weight block (ggml/nano-vllm format: 2-byte fp16 scale + 32-byte int8 = 34B).
// The backend splits these into (w8 int8 plane, qsc fp16 scale plane) so the
// matmul_q8 kernel matches hip_ops.hip bit-for-bit.
struct VMatrix {
  VBuf f16;     // uint16 storage (or float for f32 weights) + bf16 flag
  VBuf q8;      // int8, row-major, k per row
  VBuf qsc;     // fp16 scale, one per 32 cols (k/32 per row)
  VBuf qk;      // raw K-quant super-blocks (QK_K=256 elements each), row-major
  std::vector<QKSeg> qk_segs;  // per-segment (kind, byte_off, elem_off); singles hold one
  bool is_q8 = false, bf16 = false; // bf16 only meaningful when !is_q8
};

class VulkanBackend {
 public:
  // Creates instance + device + runtime. `force_device_local` applies the
  // llama-patch policy (ignore host-visible VRAM on AMD dGPUs).
  static std::unique_ptr<VulkanBackend> Create(bool force_device_local = true);
  ~VulkanBackend();
  VulkanBackend(const VulkanBackend&) = delete;
  VulkanBackend& operator=(const VulkanBackend&) = delete;

  VkDevice device() const { return dev_; }
  VkQueue queue() const { return q_; }
  VkRuntime* rt() const { return rt_; }
  uint32_t queue_family() const { return qfi_; }
  VkCommandBuffer cmd() const { return cmd_; }
  bool is_amd() const { return is_amd_; }
  bool is_rdna4() const { return is_rdna4_; }

  // Flush the batched command buffer (submit + fence wait + reset). Ops below
  // record lazily; call submit_wait() after the last op to execute the batch.
  void submit_wait();

  // Recorded ops (append to cmd_; each calls dispatch() which barrier's). Buffer
  // sub-ranges (row/col offsets) are passed as byte-element offsets so per-layer KV
  // slices and qkv_row splits work without shader changes (Vulkan descriptor offsets).
  void matmul(VBuf& x, VBuf& w, int m, int n, int k, VBuf& y, bool bf16, uint32_t w_off = 0);
  void matmul_q8(VBuf& x, VBuf& w8, VBuf& ws, int m, int n, int k, VBuf& y, uint32_t w_off = 0, uint32_t ws_off = 0);
  void matmul_qk(VBuf& x, VBuf& wq, int m, int n, int k, VBuf& y, int kind, VkDeviceSize wbyte_off = 0);
  void matmul(VBuf& x, VMatrix& w, int m, int n, int k, VBuf& y, uint32_t w_off = 0); // qk/q8/f16 dispatch
  void rms_norm(VBuf& x, VBuf& w, VBuf& y, int rows, int hidden, float eps);
  void rms_norm_add(VBuf& x, VBuf& residual, VBuf& w, VBuf& y, int rows, int hidden, float eps);
  void add_bias_inplace(VBuf& x, VBuf& bias, int rows, int cols, uint32_t row_off = 0);
  void silu_and_mul(VBuf& gup, VBuf& y, int rows, int inter);
  void rope(VBuf& data, VBuf& pos, VBuf& inv_freq, int tokens, int heads, int head_dim, int64_t stride);
  void store_kv(VBuf& key, VBuf& value, VBuf& k_cache, VBuf& v_cache, VBuf& slot_map,
                int kv_heads, int head_dim, int total_tokens, VkDeviceSize k_off = 0, VkDeviceSize v_off = 0);
  void paged_attention(VBuf& q, VBuf& y, VBuf& k_cache, VBuf& v_cache, VBuf& qseq, VBuf& qlen,
                       VBuf& block_tables, int tokens, int q_heads, int kv_heads, int head_dim,
                       int block_size, int max_blocks, float scale, VkDeviceSize k_off = 0, VkDeviceSize v_off = 0);
  void embedding(VBuf& ids, VMatrix& w, VBuf& y, int tokens, int hidden);
  void embedding_qk(VBuf& ids, VBuf& wq, VBuf& y, int tokens, int hidden, int kind);
  void gather_rows(VBuf& rows, VBuf& idx, VBuf& y, int cols, int total);

  template <class T>
  VBuf make(const std::vector<T>& host, VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) {
    VBuf b;
    if (!host.empty()) {
      b.alloc((VkDeviceSize)(host.size() * sizeof(T)), usage);
      b.upload(host.data(), (VkDeviceSize)(host.size() * sizeof(T)));
      submit_wait();
    }
    return b;
  }

 private:
  VulkanBackend(VkInstance inst, VkPhysicalDevice pd, VkPhysicalDeviceProperties props,
                VkDevice dev, VkRuntime* rt, uint32_t qfi, bool is_amd, bool is_rdna4);
  void ensure_pipeline(const char* spv_name);
  VkPipeline get_pipe(const char* name);
  struct BufBind { uint32_t binding; VkBuffer buffer; VkDeviceSize offset = 0; VkDeviceSize range = 0; };
  void begin_if_needed();
  void storage_barrier();
  void dispatch(const char* name, const void* pc, size_t pc_size,
                const BufBind* bufs, size_t nbufs, uint32_t gx, uint32_t gy, uint32_t gz);
  VkInstance inst_;
  VkPhysicalDevice pd_;
  VkPhysicalDeviceProperties props_;
  VkDevice dev_;
  VkRuntime* rt_;
  VkQueue q_;
  uint32_t qfi_;
  bool is_amd_, is_rdna4_;
  VkCommandPool pool_;
  VkCommandBuffer cmd_;
  VkFence fence_;
  VkPipelineCache pcache_;
  VkPipelineCacheHeaderVersion pcache_ver_;
  VkDescriptorSetLayout dset_layout_;
  VkPipelineLayout pipe_layout_;
  struct PipeEntry { VkPipeline pipe; };
  std::unordered_map<std::string, PipeEntry> pipes_;
   bool begun_ = false;
   PFN_vkCmdPushDescriptorSetKHR push_fn_ = nullptr;
};
