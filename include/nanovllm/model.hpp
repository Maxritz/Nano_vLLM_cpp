#pragma once

#include <memory>
#include <string>
#include <vector>

#include "common.hpp"
#include "config.hpp"
#include "context.hpp"
#include "gguf.hpp"
#include "hip_ops.hpp"
#include "safetensors.hpp"

// A weight matrix that lives on device in one of three encodings:
//  - half: DevVec<uint16_t> F16/BF16
//  - q8:   DevVec<uint8_t>  Q8_0 blocks (34 bytes per 32 elements) -- in-kernel dequant
//  - qk:   DevVec<uint8_t>  raw K-quant super-blocks (QK_K=256 elements each,
//            e.g. 144B for Q4_K, 210B for Q6_K) -- in-kernel dequant, zero bloat
struct Matrix {
  DevVec<uint16_t> f16;
  DevVec<uint8_t> q8;    // int8 weight plane (Q8_0, rows contiguous)
  DevVec<uint16_t> qsc;  // fp16 block scales plane, one per 32 elements
  DevVec<uint8_t> qk;    // raw K-quant blocks, row-major super-blocks
  std::vector<QKSeg> qk_segs;  // per-segment (kind, byte_off, elem_off); singles hold one
  bool bf16 = false;
  bool is_q8 = false;
};

// Streaming MoE state for one sparse layer. Expert weights stay on disk
// (mmap = sysmem page cache); a small per-layer set of VRAM slots holds the
// hot experts, LRU-managed. Uploads are raw f16/bf16 slices, native format.
struct MoeState {
  int E = 0, K = 0, I = 0;
  int slots = 0;
  bool bf16 = false;
  bool stacked = false;
  Matrix router;              // [E, H] device
  Matrix sh_gu, sh_dn;        // shared expert, device (empty if none)
  Matrix sh_gate;             // [1, H] sigmoid-scales shared output (if present)
  DeviceBuf<uint16_t> slot_gu; // [slots, 2I*H]
  DeviceBuf<uint16_t> slot_dn; // [slots, H*I]
  DevVec<int32_t> slot_of;    // [E] expert -> slot or -1 (use .d on device)
  // Host-side sources (pointers into mmap'd shards, valid for model lifetime).
  std::vector<const uint16_t*> f_g, f_u, f_d;  // per-expert [I*H], [I*H], [H*I]
  const uint16_t* s_gu = nullptr;              // stacked [E, 2I, H]
  const uint16_t* s_dn = nullptr;              // stacked [E, H, I]
  std::vector<int32_t> h_slot_of, h_slot_exp;  // mirrors
  std::vector<uint64_t> h_lru;
  uint64_t clock = 0;
  std::vector<uint16_t> stage;
  int evictions = 0, loads = 0;
};

struct LayerWeights {
  Matrix qkv;         // [(num_heads+2*num_kv_heads)*head_dim, hidden]
  DevVec<float> qkv_bias;   // [(num_heads+2*num_kv_heads)*head_dim] or empty
  Matrix o;           // [hidden, num_heads*head_dim]
  Matrix gate_up;     // [2*intermediate, hidden]
  Matrix down;        // [hidden, intermediate]
  DevVec<float> input_ln;   // [hidden]
  DevVec<float> post_ln;    // [hidden]
  DevVec<float> q_norm;     // [head_dim] or empty
  DevVec<float> k_norm;     // [head_dim] or empty
  MoeState moe;       // non-empty for sparse layers
};

class WeightLoader;  // defined in model.cpp; must outlive load_weights (mmap views)

class Qwen3Model {
 public:
  explicit Qwen3Model(const Config& config);
  ~Qwen3Model();

  void load_weights();
  void alloc_expert_slots();
  int allocate_kv_cache();
  int estimate_kv_cache_blocks() const;

  // Computes logits for the last token of each sequence (prefill) or every token (decode).
  // Returned vector is host memory [ctx.num_seqs_for_logits, vocab].
  std::vector<float> forward_logits(const Context& ctx);

  bool ready() const { return ready_; }

 private:
  Config config_;
  std::vector<LayerWeights> layers_;
  Matrix embed_;      // [vocab, hidden]
  DevVec<float> final_norm_; // [hidden]
  Matrix lm_head_;    // [vocab, hidden]
  DevVec<float> inv_freq_;
  DeviceBuf<float> kv_cache_;
  int num_blocks_ = 0;
  int q_size_ = 0;
  int kv_size_ = 0;
  int qkv_size_ = 0;
  int kv_head_stride_ = 0;
  int block_size_ = 256;
  size_t kv_layer_stride_ = 0;
  bool tie_lm_head_ = true;
  bool ready_ = false;
  std::unique_ptr<WeightLoader> loader_;
};
