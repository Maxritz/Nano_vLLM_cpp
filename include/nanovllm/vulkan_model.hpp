#pragma once

// Self-contained Vulkan Qwen3 model (mirrors Qwen3Model in model.hpp/model.cpp but
// backed by VulkanBackend + the GLSL kernels in shaders/vulkan/). Does NOT touch
// DevVec / hip_ops / common.hpp (HIP), so it compiles in a pure-C++ Vulkan target.
// Weight loading reuses nano-vllm's host loaders (gguf.hpp / safetensors.hpp).

#include <algorithm>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "config.hpp"
#include "edge0/stream.h"
#include "gguf.hpp"
#include "safetensors.hpp"
#include "vulkan_backend.hpp"

// Vulkan-native request context (host vectors only; no DevVec). Built by main.cpp
// from the decoded request before calling forward_logits.
struct VKContext {
    std::vector<int64_t> input_ids;
    std::vector<int64_t> positions;
    std::vector<int32_t> slot_mapping;
    std::vector<int32_t> query_seq;
    std::vector<int32_t> query_key_len;
    std::vector<int32_t> block_tables;     // [seq * max_blocks]
    std::vector<int32_t> last_indices;
    int32_t max_blocks = 0;
};

struct VLayerWeights {
    VMatrix qkv;        // [qkv_size, hidden]  (row-major, out-features rows)
    VBuf qkv_bias;      // [qkv_size] or empty
    VMatrix o;          // [hidden, q_size]
    VMatrix gate_up;    // [2*inter, hidden]
    VMatrix down;       // [hidden, inter]
    VBuf input_ln;      // [hidden]
    VBuf post_ln;       // [hidden]
    VBuf q_norm;        // [head_dim] or empty
    VBuf k_norm;        // [head_dim] or empty
    struct VulkanMoeState {
        int E = 0, K = 0, I = 0;
        int slots = 0;
        bool bf16 = false;
        bool stacked = false;
        int router_kind = 0;          // 0=SOFTMAX_TOPK (Qwen), 1=SIGMOID_GROUP (Llama/Bailing)
        int n_group = 0, topk_group = 0;
        float routed_scaling = 1.0f;
        bool norm_topk = true;
        VMatrix router;              // [E, H] device
        VMatrix sh_gu, sh_dn;        // shared expert
        VMatrix sh_gate;             // [1, H] sigmoid-gate (if present)
        VMatrix ex_bias_f16;      // [E] expert bias as fp16 (SIGMOID_GROUP)
        VBuf ex_bias;             // [E] expert bias as float (SIGMOID_GROUP)
        VBuf slot_gu;  // [slots, 2I*H] device
        VBuf slot_dn;  // [slots, I*H] device
        VBuf slot_of;  // [E] device (slot index per expert)
        // Host-side sources (pointers into mmap'd shards, valid for model lifetime).
        std::vector<const uint16_t*> f_g, f_u, f_d;  // per-expert [I*H]
        const uint16_t* s_gu = nullptr;  // stacked [E, 2I, H]
        const uint16_t* s_dn = nullptr;  // stacked [E, H, I]
        std::vector<int32_t> h_slot_of, h_slot_exp;  // mirrors
        std::vector<uint64_t> h_lru;
        uint64_t clock = 0;
        std::vector<uint16_t> stage;
        int evictions = 0, loads = 0;
    } moe;
};

class VulkanModel {
public:
    explicit VulkanModel(const Config& config);
    ~VulkanModel();

    void load_weights();
    void alloc_expert_slots();
    int allocate_kv_cache();
    int estimate_kv_cache_blocks() const;

    // Returns host logits [out_rows, vocab]. One batched Vulkan submission.
    std::vector<float> forward_logits(const VKContext& ctx);

    bool ready() const { return ready_; }

private:
    Config config_;
    std::unique_ptr<VulkanBackend> dev_;
    std::vector<VLayerWeights> layers_;
    VMatrix embed_;       // [vocab, hidden]
    VBuf final_norm_;     // [hidden]
    VMatrix lm_head_;     // [vocab, hidden] (only if !tied)
    VBuf inv_freq_;       // [head_dim/2]
    // KV cache is carved into per-layer k/v slices (each <maxBufferSize on AMD)
    // instead of one monolithic buffer that can exceed the 2GB VkBuffer limit.
    std::vector<VBuf> k_cache_;   // per-layer [num_blocks*block_size*kv_head_stride]
    std::vector<VBuf> v_cache_;
    int num_blocks_ = 0;
    int q_size_ = 0;
    int kv_size_ = 0;
    int qkv_size_ = 0;
    int kv_head_stride_ = 0;
    int block_size_ = 256;
    size_t kv_layer_stride_ = 0;
    bool tie_lm_head_ = true;
    bool ready_ = false;

    void load_matrix(const std::string& name, VMatrix& m, bool required, size_t n_elements);
    void load_matrix_host(const std::string& name, std::vector<uint16_t>& u16, bool& bf16,
                          std::vector<uint8_t>& q8, bool& is_q8, std::vector<uint8_t>& qk,
                          int& qk_kind, size_t n_elements, bool required);
    void upload_q8_block(const std::vector<uint8_t>& fused, VMatrix& m); // 34B->split
    void upload_qk_block(const std::vector<uint8_t>& raw, int kind, VMatrix& m); // raw super-blocks
    void upload_u16_block(const std::vector<uint16_t>& src, VMatrix& m);
    bool gguf_ = false;
    // Loader (GGUF or safetensors) + name map, inlined from model.cpp.
    GGUFLoader gg_loader_;
    SafetensorsLoader st_loader_;
    bool st_ = false;
    static std::string map_name(const std::string& n);
    bool contains(const std::string& st_name) const;
    bool load_float(const std::string& st_name, std::vector<float>& out) const;
    bool load_u16(const std::string& st_name, std::vector<uint16_t>& out, bool& bf16) const;
    bool load_q8(const std::string& st_name, std::vector<uint8_t>& out) const;
    bool load_qk(const std::string& st_name, int kind, size_t n_elements, std::vector<uint8_t>& out) const;
    bool load_optional(const std::string& st_name, std::vector<float>& out) const;
    void load_model_dir(const std::string& model_dir);
    void prefetch_experts();
    void do_prefetch_experts();  // host page-touch only; safe on a worker thread
    std::future<void> prefetch_fut_;
    mutable std::mutex prefetch_mu_;  // guards MoE slot mirrors vs prefetch worker
    std::string moe_key_prefix(int layer) const;
    void moe_place(VLayerWeights::VulkanMoeState& ms, int layer,
                   const std::vector<int32_t>& h_idx, int hidden, int I);
    e0::SharedExpertCache& expert_cache() { return expert_cache_; }
    e0::PrefetchBuffer& prefetch_buf() { return prefetch_buf_; }
    const std::vector<VLayerWeights>& layers() const { return layers_; }
    const Config& config() const { return config_; }
    VLayerWeights::VulkanMoeState* layer_moe(int l) {
        auto it = std::find_if(layers_.begin(), layers_.end(),
                               [](const VLayerWeights& lw) { return lw.moe.E > 0; });
        return it == layers_.end() ? nullptr : &it->moe;
    }
private:
    e0::SharedExpertCache expert_cache_;
    e0::PrefetchBuffer prefetch_buf_;
    std::string model_prefix_;  // "model." or "model.language_model."
};
