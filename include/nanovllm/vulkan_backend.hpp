#pragma once

// Vulkan backend for nano-vllm. Mirrors the hip_ops / DevVec / Qwen3Model
// surface, backed by the GLSL kernels in shaders/vulkan/*.comp (SPIR-V
// embedded in spv.h) and a minimal device-local allocator. Uses
// VK_KHR_push_descriptor so there is ZERO per-dispatch descriptor-pool churn
// (this is what makes a many-op model forward feasible, unlike VAiT vkblas's
// fixed 64-set pool which leaks one set per vkblas_qgemm_* call).
//
// Device extensions / features are enabled as available (shaderFloat16,
// shaderInt8 via VK_KHR_shader_float16_int8; VK_EXT_scalar_block_layout;
// VK_KHR_push_descriptor). Validation layers are best-effort via VK_LAYER_KHRONOS_validation.
// AMD-specific perf detection (apiVersion>=1.4 || vendor 0x1002/0x1023) gates
// future RDNA4-specific tuning; generic device selection is otherwise vendor-agnostic.

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <vulkan/vulkan.h>

#include "gguf.hpp"

#include "spv.h"

// Minimal runtime state, replacing the VAiT vkruntime dependency. Holds device
// handles and memory properties needed for buffer allocation / transfers.
struct VkRT {
    VkDevice device = VK_NULL_HANDLE;
    VkPhysicalDevice physDev = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memProps{};
    VkCommandPool xferPool = VK_NULL_HANDLE;
    VkCommandBuffer xferCmd = VK_NULL_HANDLE;
    VkFence xferFence = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t qfi = 0;
};

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
   void upload_at(VkDeviceSize offset, const void* host, VkDeviceSize bytes);
   void upload_at_async(VkDeviceSize offset, const void* host, VkDeviceSize bytes);
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
   VkPhysicalDevice physical_device() const { return pd_; }
   VkPhysicalDeviceProperties props() const { return props_; }
   VkQueue queue() const { return q_; }
  VkRT* rt() const { return rt_; }
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
  void matmul(VBuf& x, VBuf& w, int m, int n, int k, VBuf& y, bool bf16, uint32_t w_off = 0, bool barrier_after = true);
  void matmul_q8(VBuf& x, VBuf& w8, VBuf& ws, int m, int n, int k, VBuf& y, uint32_t w_off = 0, uint32_t ws_off = 0, bool barrier_after = true);
  void matmul_qk(VBuf& x, VBuf& wq, int m, int n, int k, VBuf& y, int kind, VkDeviceSize wbyte_off = 0, bool barrier_after = true);
  void matmul(VBuf& x, VMatrix& w, int m, int n, int k, VBuf& y, uint32_t w_off = 0, bool barrier_after = true); // qk/q8/f16 dispatch
  void rms_norm(VBuf& x, VBuf& w, VBuf& y, int rows, int hidden, float eps, bool barrier_after = true);
  void rms_norm_add(VBuf& x, VBuf& residual, VBuf& w, VBuf& y, int rows, int hidden, float eps);
  void add_bias_inplace(VBuf& x, VBuf& bias, int rows, int cols, uint32_t row_off = 0, bool barrier_after = true);
   void silu_and_mul(VBuf& gup, VBuf& y, int rows, int inter);
   void scale_sigmoid(VBuf& x, VBuf& s, int rows, int cols);
  void rope(VBuf& data, VBuf& pos, VBuf& inv_freq, int tokens, int heads, int head_dim, int64_t stride,
          float rope_factor = 1.0f, float rope_beta = 32.0f, bool barrier_after = true);
  void store_kv(VBuf& key, VBuf& value, VBuf& k_cache, VBuf& v_cache, VBuf& slot_map,
                int kv_heads, int head_dim, int total_tokens, VkDeviceSize k_off = 0, VkDeviceSize v_off = 0);
   void paged_attention(VBuf& q, VBuf& y, VBuf& k_cache, VBuf& v_cache, VBuf& qseq, VBuf& qlen,
                        VBuf& block_tables, int tokens, int q_heads, int kv_heads, int head_dim,
                        int block_size, int max_blocks, float scale, VkDeviceSize k_off = 0, VkDeviceSize v_off = 0);
   // Extended paged attention: sw_start masks keys with key_idx < sw_start
   // (sliding window; 0 = disabled), attn_softcap applies tanh-softcap
   // (0 = off). paged_attention() above forwards with (0, 0.0f).
   void paged_attention_ex(VBuf& q, VBuf& y, VBuf& k_cache, VBuf& v_cache, VBuf& qseq, VBuf& qlen,
                           VBuf& block_tables, int tokens, int q_heads, int kv_heads, int head_dim,
                           int block_size, int max_blocks, float scale, int sw_start, float attn_softcap,
                           VkDeviceSize k_off = 0, VkDeviceSize v_off = 0);
  void embedding(VBuf& ids, VMatrix& w, VBuf& y, int tokens, int hidden);
  void embedding_qk(VBuf& ids, VBuf& wq, VBuf& y, int tokens, int hidden, int kind);
   void gather_rows(VBuf& rows, VBuf& idx, VBuf& y, int cols, int total);

   // MoE (mirrors hip_ops.hip moe_route/moe_ffn).
   void moe_route(VBuf& logits, VBuf& idx, VBuf& score, int rows, int E, int K, bool norm);
   void moe_route_sigmoid(VBuf& logits, VBuf& idx, VBuf& score, VBuf& bias,
                          int rows, int E, int K, int n_group, int topk_group,
                          bool norm, float routed_scaling, int bias_off);
    void moe_ffn(VBuf& x, VBuf& idx, VBuf& score, VBuf& gu, VBuf& dn,
                 VBuf& slot_of, VBuf& out, int rows, int H, int I, int K, bool bf16);
    // Fused route+FFN (P2-4): single dispatch; experts accumulate into `out`
    // (overwrite). Host still reads idx/score back for slot placement; on a
    // slot miss it places then re-runs (see vulkan_model.cpp).
    void moe_route_ffn(VBuf& x, VBuf& logits, VBuf& idx, VBuf& score, VBuf& gu, VBuf& dn,
                       VBuf& slot_of, VBuf& out, int r0, int rows, int H, int I, int K, int E,
                       bool bf16, bool norm);
    void add_buf(VBuf& a, VBuf& b, int n);

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
                 VkDevice dev, VkRT* rt, uint32_t qfi, bool is_amd, bool is_rdna4);
  void ensure_pipeline(const char* spv_name);
  VkPipeline get_pipe(const char* name);
  struct BufBind { uint32_t binding; VkBuffer buffer; VkDeviceSize offset = 0; VkDeviceSize range = 0; };
  // Specialization-constant pipeline variant (e.g. moe_ffn SPEC_H/SPEC_I so
  // shared memory is sized to the model, not compile-time maxima). `key`
  // must uniquely identify (name + spec values); pipelines are cached by key.
  void ensure_pipeline_spec(const char* spv_name, const char* key,
                            const VkSpecializationMapEntry* entries, uint32_t n_entries,
                            const void* data, size_t data_size);
  VkPipeline get_pipe_spec(const char* spv_name, const char* key,
                           const VkSpecializationMapEntry* entries, uint32_t n_entries,
                           const void* data, size_t data_size);
  void dispatch_spec(const char* spv_name, const char* key,
                     const VkSpecializationMapEntry* entries, uint32_t n_entries,
                     const void* data, size_t data_size,
                     const void* pc, size_t pc_size,
                     const BufBind* bufs, size_t nbufs, uint32_t gx, uint32_t gy, uint32_t gz);
  void begin_if_needed();
  void storage_barrier();
  void dispatch(const char* name, const void* pc, size_t pc_size,
                const BufBind* bufs, size_t nbufs, uint32_t gx, uint32_t gy, uint32_t gz);
  // Same, but skips the trailing storage barrier. Only for runs of independent
  // dispatches (disjoint outputs); the last one must barrier (or use dispatch).
  void dispatch_nb(const char* name, const void* pc, size_t pc_size,
                   const BufBind* bufs, size_t nbufs, uint32_t gx, uint32_t gy, uint32_t gz);
  void dispatch_bound(VkPipeline p, const void* pc, size_t pc_size,
                      const BufBind* bufs, size_t nbufs, uint32_t gx, uint32_t gy, uint32_t gz,
                      bool barrier = true);
  VkInstance inst_;
  VkPhysicalDevice pd_;
  VkPhysicalDeviceProperties props_;
   VkDevice dev_;
   VkRT* rt_;
  VkQueue q_;
  uint32_t qfi_;
  bool is_amd_, is_rdna4_;
  VkCommandPool pool_;
  VkCommandBuffer cmd_;
  VkFence fence_;
   VkPipelineCache pcache_;
  VkDescriptorSetLayout dset_layout_;
  VkPipelineLayout pipe_layout_;
  struct PipeEntry { VkPipeline pipe; };
  std::unordered_map<std::string, PipeEntry> pipes_;
   bool begun_ = false;
   PFN_vkCmdPushDescriptorSetKHR push_fn_ = nullptr;
};
