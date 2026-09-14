#include "nanovllm/model.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <numeric>
#include <stdexcept>
#include <string>
#include <cstdio>

#include "nanovllm/common.hpp"
#include "nanovllm/gguf.hpp"
#include "nanovllm/safetensors.hpp"

// WeightLoader: unified view over safetensors or GGUF weights, keyed by the
// safetensors-style names used throughout this model (HF checkpoint naming).
class WeightLoader {
 public:
  void load_model_dir(const std::string& model_dir);
  bool contains(const std::string& st_name) const;
  bool load_float(const std::string& st_name, std::vector<float>& out) const;
  bool load_u16(const std::string& st_name, std::vector<uint16_t>& out, bool& bf16) const;
  bool load_q8(const std::string& st_name, std::vector<uint8_t>& out) const;
  bool load_qk(const std::string& st_name, int kind, size_t n_elements, std::vector<uint8_t>& out) const;
  bool gguf_mode() const { return gg_; }
  // Mapped (zero-copy) view of an F16/BF16 tensor; safetensors only.
  const uint16_t* mapped_u16(const std::string& name, size_t& elems, bool& bf16) const {
    if (!st_) throw std::runtime_error("mapped_u16 requires safetensors");
    size_t bytes = 0;
    bool f16 = false;
    const uint8_t* p = st_loader_.mapped(name, bytes, f16);
    if (!p) return nullptr;
    elems = bytes / 2;
    bf16 = !f16;
    return reinterpret_cast<const uint16_t*>(p);
  }
  // MoE weights would otherwise be silently ignored (garbage output). Fail loud.
  bool has_moe() const {
    auto names = st_ ? st_loader_.names() : gg_loader_.names();
    for (auto& n : names)
      if (n.find("exps") != std::string::npos || n.find(".experts.") != std::string::npos ||
          n.find("gate_inp") != std::string::npos || n.find("shared_expert") != std::string::npos ||
          n.find("moe") != std::string::npos)
        return true;
    return false;
  }

 private:
  static std::string map(const std::string& st_name);
  bool st_ = false;
  bool gg_ = false;
  SafetensorsLoader st_loader_;
  GGUFLoader gg_loader_;
};

void WeightLoader::load_model_dir(const std::string& model) {
  st_ = gg_ = false;
  std::filesystem::path p(model);
  if (std::filesystem::is_directory(p)) {
    // Directories: prefer safetensors shards (deterministic goldens), fall back to gguf.
    std::error_code ec;
    bool has_safetensors = false;
    for (const auto& ent : std::filesystem::directory_iterator(p, ec))
      if (ent.path().extension() == ".safetensors") has_safetensors = true;
    if (!has_safetensors) {
      std::string gguf = find_gguf_path(model);
      if (!gguf.empty()) {
        gg_loader_.add_file(gguf);
        gg_ = true;
        return;
      }
    }
    st_loader_.add_directory(model);
    st_ = true;
    return;
  }
  // Direct .gguf file path.
  if (p.extension() == ".gguf") {
    gg_loader_.add_file(p.string());
    gg_ = true;
    return;
  }
  st_loader_.add_directory(model);
  st_ = true;
}

// HF checkpoint name -> GGUF tensor name (LLaMA.cpp qwen2 naming).
std::string WeightLoader::map(const std::string& n) {
  std::string s = n;
  auto rep = [&](const std::string& a, const std::string& b, const std::string& c) {
    size_t p = s.find(a);
    if (p != std::string::npos) s.replace(p, a.size(), b);
  };
  rep("model.embed_tokens.weight", "token_embd.weight", "");
  rep(".self_attn.q_proj.weight", ".attn_q.weight", "");
  rep(".self_attn.k_proj.weight", ".attn_k.weight", "");
  rep(".self_attn.v_proj.weight", ".attn_v.weight", "");
  rep(".self_attn.o_proj.weight", ".attn_output.weight", "");
  rep(".self_attn.q_proj.bias", ".attn_q.bias", "");
  rep(".self_attn.k_proj.bias", ".attn_k.bias", "");
  rep(".self_attn.v_proj.bias", ".attn_v.bias", "");
  rep(".mlp.gate_proj.weight", ".ffn_gate.weight", "");
  rep(".mlp.up_proj.weight", ".ffn_up.weight", "");
  rep(".mlp.down_proj.weight", ".ffn_down.weight", "");
  rep(".mlp.gate_up_proj.weight", ".ffn_gate_up.weight", "");
  rep(".input_layernorm.weight", ".attn_norm.weight", "");
  rep(".post_attention_layernorm.weight", ".ffn_norm.weight", "");
  rep(".self_attn.q_norm.weight", ".attn_q_norm.weight", "");
  rep(".self_attn.k_norm.weight", ".attn_k_norm.weight", "");
  rep("model.norm.weight", "output_norm.weight", "");
  rep("lm_head.weight", "output.weight", "");
  rep("model.layers.", "blk.", "");
  return s;
}

bool WeightLoader::contains(const std::string& st_name) const {
  return st_ ? st_loader_.contains(st_name) : gg_loader_.contains(map(st_name));
}

bool WeightLoader::load_float(const std::string& st_name, std::vector<float>& out) const {
  return st_ ? st_loader_.load_float(st_name, out) : gg_loader_.load_float(map(st_name), out);
}

bool WeightLoader::load_u16(const std::string& st_name, std::vector<uint16_t>& out, bool& bf16) const {
  return st_ ? st_loader_.load_u16(st_name, out, bf16) : gg_loader_.load_u16(map(st_name), out, bf16);
}

bool WeightLoader::load_q8(const std::string& st_name, std::vector<uint8_t>& out) const {
  return gg_ && gg_loader_.load_q8_0(map(st_name), out);
}

bool WeightLoader::load_qk(const std::string& st_name, int kind, size_t n_elements,
                            std::vector<uint8_t>& out) const {
  return gg_ && gg_loader_.load_qk(map(st_name), kind, n_elements, out);
}

static bool load_optional(WeightLoader& loader, const std::string& name, std::vector<float>& out) {
  return loader.load_float(name, out);
}

// Loads a big matmul/embedding weight, keeping its on-disk encoding (F16/BF16,
// Q8_0, or raw K-quant super-blocks). n_elements must match the tensor.
static void load_matrix(WeightLoader& loader, const std::string& name, Matrix& m, bool required,
                        size_t n_elements);
static void upload_q8_split(Matrix& m, const std::vector<uint8_t>& fused);

static void load_matrix(WeightLoader& loader, const std::string& name, Matrix& m, bool required,
                        size_t n_elements) {
  std::vector<uint8_t> q8;
  if (loader.load_q8(name, q8)) {
    upload_q8_split(m, q8);
    m.f16.free();
    m.qk.free();
    m.qk_segs.clear();
    m.is_q8 = true;
    return;
  }
  for (int kind : {12, 13, 14}) {
    std::vector<uint8_t> qk;
    if (loader.load_qk(name, kind, n_elements, qk)) {
      m.qk.assign(qk);
      m.qk_segs = {{kind, 0, 0}};
      m.f16.free();
      m.q8.free();
      m.qsc.free();
      m.is_q8 = false;
      return;
    }
  }
  std::vector<uint16_t> u;
  bool bf16 = false;
  if (loader.load_u16(name, u, bf16)) {
    m.f16.assign(u);
    m.q8.free();
    m.qsc.free();
    m.qk.free();
    m.qk_segs.clear();
    m.is_q8 = false;
    m.bf16 = bf16;
    return;
  }
  if (required) throw std::runtime_error("missing required weight: " + name);
}

// Loads a tensor as host bytes for fuse-then-upload (q/k/v, gate/up). Exactly one
// of {u16, q8, qk} is filled; the others stay empty. kind carries the K-quant id.
static void load_matrix_host(WeightLoader& loader, const std::string& name, std::vector<uint16_t>& u16,
                             bool& bf16, std::vector<uint8_t>& q8, bool& is_q8, std::vector<uint8_t>& qk,
                             int& qk_kind, size_t n_elements, bool required) {
  u16.clear();
  q8.clear();
  qk.clear();
  bf16 = false;
  is_q8 = false;
  qk_kind = 0;
  if (loader.load_q8(name, q8)) {
    is_q8 = true;
    return;
  }
  for (int kind : {12, 13, 14}) {
    if (loader.load_qk(name, kind, n_elements, qk)) {
      qk_kind = kind;
      return;
    }
  }
  if (loader.load_u16(name, u16, bf16)) {
    is_q8 = false;
    return;
  }
  if (required) throw std::runtime_error("missing required weight: " + name);
}

static void concat_rows(std::vector<float>& dst, const std::vector<float>& a, const std::vector<float>& b,
                        const std::vector<float>& c) {
  dst.clear();
  dst.reserve(a.size() + b.size() + c.size());
  dst.insert(dst.end(), a.begin(), a.end());
  dst.insert(dst.end(), b.begin(), b.end());
  dst.insert(dst.end(), c.begin(), c.end());
}

// Row-concatenates three weight matrices that share the same contiguous dim (hidden).
// Represented either as F16/BF16 u16 values or Q8_0 byte blocks; in both layouts rows
// are stored back-to-back with the in-dim contiguous, so a plain append is row-concat.
static void concat_rows_u(std::vector<uint16_t>& dst, const std::vector<uint16_t>& a,
                          const std::vector<uint16_t>& b, const std::vector<uint16_t>& c) {
  dst.clear();
  dst.reserve(a.size() + b.size() + c.size());
  dst.insert(dst.end(), a.begin(), a.end());
  dst.insert(dst.end(), b.begin(), b.end());
  dst.insert(dst.end(), c.begin(), c.end());
}

// ---- streaming MoE: load (mmap slices), slot pool, per-step placement ----
static void upload_u16(DevVec<uint16_t>& dst, const std::vector<uint16_t>& src);
static void load_moe_layer(WeightLoader& loader, const HFConfig& hf, LayerWeights& lw, int layer) {
  std::string p = "model.layers." + std::to_string(layer) + ".mlp.";
  auto& ms = lw.moe;
  const int H = hf.hidden_size, E = hf.num_experts, K = hf.num_experts_per_tok, I = hf.moe_intermediate_size;
  ms.E = E; ms.K = K; ms.I = I;
  std::vector<uint16_t> rt;
  bool bf16 = false;
  if (!loader.load_u16(p + "gate.weight", rt, bf16) || rt.size() != (size_t)E * H)
    throw std::runtime_error("missing/short router weights in " + p);
  upload_u16(ms.router.f16, rt);
  ms.router.bf16 = bf16;
  ms.bf16 = bf16;
  const int SI = hf.shared_expert_intermediate_size;
  if (SI > 0) {
    std::vector<uint16_t> g, u, d;
    bool b1 = false, b2 = false, b3 = false;
    if (!loader.load_u16(p + "shared_expert.gate_proj.weight", g, b1) ||
        !loader.load_u16(p + "shared_expert.up_proj.weight", u, b2) ||
        !loader.load_u16(p + "shared_expert.down_proj.weight", d, b3))
      throw std::runtime_error("missing shared expert weights in " + p);
    std::vector<uint16_t> gu;
    gu.reserve(g.size() + u.size());
    gu.insert(gu.end(), g.begin(), g.end());
    gu.insert(gu.end(), u.begin(), u.end());
    upload_u16(ms.sh_gu.f16, gu);
    upload_u16(ms.sh_dn.f16, d);
    ms.sh_gu.bf16 = b1 && b2;
    ms.sh_dn.bf16 = b3;
    std::vector<uint16_t> gt;
    bool bg = false;
    if (loader.load_u16(p + "shared_expert_gate.weight", gt, bg) && gt.size() == (size_t)H)
      upload_u16(ms.sh_gate.f16, gt), ms.sh_gate.bf16 = bg;
  }
  ms.stacked = loader.contains(p + "experts.gate_up_proj") && loader.contains(p + "experts.down_proj");
  size_t ne = 0;
  bool b = false;
  if (ms.stacked) {
    ms.s_gu = loader.mapped_u16(p + "experts.gate_up_proj", ne, b);
    if (ne != (size_t)E * 2 * I * H) throw std::runtime_error("unexpected gate_up_proj shape in " + p);
    ms.s_dn = loader.mapped_u16(p + "experts.down_proj", ne, b);
    if (ne != (size_t)E * H * I) throw std::runtime_error("unexpected down_proj shape in " + p);
  } else {
    ms.f_g.resize(E);
    ms.f_u.resize(E);
    ms.f_d.resize(E);
    for (int e = 0; e < E; ++e) {
      std::string pe = p + "experts." + std::to_string(e) + ".";
      ms.f_g[e] = loader.mapped_u16(pe + "gate_proj.weight", ne, b);
      if (!ms.f_g[e] || ne != (size_t)I * H) throw std::runtime_error("missing expert gate " + pe);
      ms.f_u[e] = loader.mapped_u16(pe + "up_proj.weight", ne, b);
      if (!ms.f_u[e] || ne != (size_t)I * H) throw std::runtime_error("missing expert up " + pe);
      ms.f_d[e] = loader.mapped_u16(pe + "down_proj.weight", ne, b);
      if (!ms.f_d[e] || ne != (size_t)H * I) throw std::runtime_error("missing expert down " + pe);
    }
  }
}

void Qwen3Model::alloc_expert_slots() {
  auto& hf = config_.hf;
  if (hf.num_experts <= 0) return;
  int nl = 0;
  for (int l = 0; l < hf.num_hidden_layers; ++l)
    if (hf.layer_is_moe(l)) ++nl;
  if (!nl) return;
  const size_t per_expert = (size_t)3 * hf.moe_intermediate_size * hf.hidden_size * 2;
  const size_t total = (size_t)nl * hf.num_experts * per_expert;
  size_t free_b = 0, tot_b = 0;
  HIP_CHECK(hipMemGetInfo(&free_b, &tot_b));
  size_t budget = config_.expert_budget_gb > 0
                      ? (size_t)(config_.expert_budget_gb * 1e9)
                      : (size_t)(free_b * 0.45);
  if (budget > total) budget = total;
  int slots = (int)(budget / ((size_t)nl * per_expert));
  if (slots < hf.num_experts_per_tok + 2) slots = hf.num_experts_per_tok + 2;
  if (slots > hf.num_experts) slots = hf.num_experts;
  for (auto& lw : layers_) {
    auto& ms = lw.moe;
    if (!ms.E) continue;
    ms.slots = slots;
    ms.slot_gu.alloc((size_t)slots * 2 * ms.I * hf.hidden_size);
    ms.slot_dn.alloc((size_t)slots * ms.I * hf.hidden_size);
    ms.h_slot_of.assign(ms.E, -1);
    ms.h_slot_exp.assign(slots, -1);
    ms.h_lru.assign(slots, 0);
    ms.slot_of.assign(ms.h_slot_of);
    ms.stage.resize((size_t)3 * ms.I * hf.hidden_size);
  }
  std::fprintf(stderr, "MoE: %d sparse layers, %d experts (top-%d), %d VRAM slots/layer, %.2f GB budget\n",
               nl, hf.num_experts, hf.num_experts_per_tok, slots, budget / 1e9);
}

static void moe_place(MoeState& ms, const std::vector<int32_t>& h_idx, int hidden, int I) {
  std::vector<int32_t> uniq = h_idx;
  std::sort(uniq.begin(), uniq.end());
  uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
  if ((int)uniq.size() > ms.slots)
    throw std::runtime_error("moe_place: unique experts exceed slots (chunking invariant broken)");
  const size_t gu_n = (size_t)2 * I * hidden, dn_n = (size_t)I * hidden;
  bool dirty = false;
  for (int32_t e : uniq) {
    if (e < 0 || e >= ms.E) continue;
    if (ms.h_slot_of[e] >= 0) {
      ms.h_lru[ms.h_slot_of[e]] = ++ms.clock;
      continue;
    }
    int slot = -1;
    for (int s = 0; s < ms.slots; ++s)
      if (ms.h_slot_exp[s] < 0) { slot = s; break; }
    if (slot < 0) {
      slot = 0;
      for (int s = 1; s < ms.slots; ++s)
        if (ms.h_lru[s] < ms.h_lru[slot]) slot = s;
      ms.h_slot_of[ms.h_slot_exp[slot]] = -1;
      ms.evictions++;
    }
    uint16_t* sg = ms.stage.data();
    uint16_t* sd = sg + gu_n;
    if (ms.stacked) {
      std::memcpy(sg, ms.s_gu + (size_t)e * gu_n, gu_n * 2);
      std::memcpy(sd, ms.s_dn + (size_t)e * dn_n, dn_n * 2);
    } else {
      std::memcpy(sg, ms.f_g[e], dn_n * 2);
      std::memcpy(sg + dn_n, ms.f_u[e], dn_n * 2);
      std::memcpy(sd, ms.f_d[e], dn_n * 2);
    }
    // ponytail: synchronous copies; pageable-async from the shared stage buffer raced.
    // Ring buffer + hipStreamSynchronize if this shows up in decode profiles.
    HIP_CHECK(hipMemcpy(ms.slot_gu.ptr() + (size_t)slot * gu_n, sg, gu_n * 2, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(ms.slot_dn.ptr() + (size_t)slot * dn_n, sd, dn_n * 2, hipMemcpyHostToDevice));
    ms.h_slot_of[e] = slot;
    ms.h_slot_exp[slot] = e;
    ms.h_lru[slot] = ++ms.clock;
    ms.loads++;
    dirty = true;
  }
  if (dirty)
    HIP_CHECK(hipMemcpy(ms.slot_of.d, ms.h_slot_of.data(), ms.h_slot_of.size() * sizeof(int32_t),
                        hipMemcpyHostToDevice));
}

static void concat_rows_q8(std::vector<uint8_t>& dst, const std::vector<uint8_t>& a,
                           const std::vector<uint8_t>& b, const std::vector<uint8_t>& c) {
  dst.clear();
  dst.reserve(a.size() + b.size() + c.size());
  dst.insert(dst.end(), a.begin(), a.end());
  dst.insert(dst.end(), b.begin(), b.end());
  dst.insert(dst.end(), c.begin(), c.end());
}

static void upload(DevVec<float>& dst, const std::vector<float>& src) { dst.assign(src); }
static void upload_u16(DevVec<uint16_t>& dst, const std::vector<uint16_t>& src) { dst.assign(src); }

// Split fused 34-byte Q8_0 blocks into int8 + fp16-scale planes, upload both.
static void upload_q8_split(Matrix& m, const std::vector<uint8_t>& fused) {
  size_t nblk = fused.size() / 34;
  std::vector<uint8_t> w8(nblk * 32);
  std::vector<uint16_t> ws(nblk);
  for (size_t b = 0; b < nblk; ++b) {
    memcpy(&ws[b], &fused[b * 34], 2);
    memcpy(&w8[b * 32], &fused[b * 34 + 2], 32);
  }
  m.q8.assign(w8);
  m.qsc.assign(ws);
}

Qwen3Model::Qwen3Model(const Config& config) : config_(config) {
  config_.validate();
  const auto& hf = config_.hf;
  q_size_ = hf.num_attention_heads * hf.head_dim;
  kv_size_ = hf.num_key_value_heads * hf.head_dim;
  qkv_size_ = q_size_ + 2 * kv_size_;
  kv_head_stride_ = hf.num_key_value_heads * hf.head_dim;
  block_size_ = config_.kvcache_block_size;
  layers_.resize(hf.num_hidden_layers);

  std::vector<float> inv_freq(hf.head_dim / 2);
  for (int i = 0; i < hf.head_dim / 2; ++i) {
    inv_freq[i] = static_cast<float>(1.0 / std::pow(hf.rope_theta, (2.0 * i) / hf.head_dim));
  }
  inv_freq_.assign(inv_freq);
}

Qwen3Model::~Qwen3Model() = default;

void Qwen3Model::load_weights() {
  loader_.reset(new WeightLoader());
  WeightLoader& loader = *loader_;
  loader.load_model_dir(config_.model);
  if (loader.has_moe() && (loader.gguf_mode() || config_.hf.num_experts <= 0))
    throw std::runtime_error("MoE weights present but layout unsupported (safetensors Qwen MoE only)");
  if (!loader.contains("model.embed_tokens.weight")) throw std::runtime_error("model.embed_tokens.weight not found");

  auto& hf = config_.hf;
  int hidden = hf.hidden_size;
  int inter = hf.intermediate_size;

  load_matrix(loader, "model.embed_tokens.weight", embed_, true, size_t(hf.vocab_size) * hidden);

  for (int layer = 0; layer < hf.num_hidden_layers; ++layer) {
    std::string p = "model.layers." + std::to_string(layer);
    auto& lw = layers_[layer];

    // qkv: prefer fused, else separate q/k/v (works for both safetensors and GGUF names).
    if (loader.contains(p + ".self_attn.qkv_proj.weight")) {
      load_matrix(loader, p + ".self_attn.qkv_proj.weight", lw.qkv, true, size_t(qkv_size_) * hidden);
    } else {
      std::vector<uint16_t> q_u, k_u, v_u;
      bool q_bf16 = false, k_bf16 = false, v_bf16 = false;
      std::vector<uint8_t> q_q8, k_q8, v_q8;
      bool q_is_q8 = false, k_is_q8 = false, v_is_q8 = false;
      std::vector<uint8_t> q_qk, k_qk, v_qk;
      int q_kind = 0, k_kind = 0, v_kind = 0;
      load_matrix_host(loader, p + ".self_attn.q_proj.weight", q_u, q_bf16, q_q8, q_is_q8, q_qk, q_kind,
                       size_t(q_size_) * hidden, true);
      load_matrix_host(loader, p + ".self_attn.k_proj.weight", k_u, k_bf16, k_q8, k_is_q8, k_qk, k_kind,
                       size_t(kv_size_) * hidden, true);
      load_matrix_host(loader, p + ".self_attn.v_proj.weight", v_u, v_bf16, v_q8, v_is_q8, v_qk, v_kind,
                       size_t(kv_size_) * hidden, true);
      if (q_is_q8 || k_is_q8 || v_is_q8) {
        if (!(q_is_q8 && k_is_q8 && v_is_q8))
          throw std::runtime_error("mixed Q8/non-Q8 q/k/v weights are unsupported");
        std::vector<uint8_t> fused;
        concat_rows_q8(fused, q_q8, k_q8, v_q8);
        upload_q8_split(lw.qkv, fused);
        lw.qkv.f16.free();
        lw.qkv.qk.free();
        lw.qkv.qk_segs.clear();
        lw.qkv.is_q8 = true;
      } else if (q_kind || k_kind || v_kind) {
        if (!(q_kind && k_kind && v_kind))
          throw std::runtime_error("mixed K-quant/non-K-quant q/k/v weights are unsupported");
        size_t qe = size_t(q_size_) * hidden, ke = size_t(kv_size_) * hidden;
        if (qe % 256 || ke % 256)
          throw std::runtime_error("K-quant q/k/v split is not on a super-block boundary");
        size_t qb = qe / 256 * GGUFLoader::qk_block_bytes(q_kind);
        size_t kb = ke / 256 * GGUFLoader::qk_block_bytes(k_kind);
        std::vector<uint8_t> fused;
        fused.reserve(q_qk.size() + k_qk.size() + v_qk.size());
        fused.insert(fused.end(), q_qk.begin(), q_qk.end());
        fused.insert(fused.end(), k_qk.begin(), k_qk.end());
        fused.insert(fused.end(), v_qk.begin(), v_qk.end());
        lw.qkv.qk.assign(fused);
        lw.qkv.qk_segs = {{q_kind, 0, 0}, {k_kind, qb, qe}, {v_kind, qb + kb, qe + ke}};
        lw.qkv.f16.free();
        lw.qkv.q8.free();
        lw.qkv.qsc.free();
        lw.qkv.is_q8 = false;
      } else {
        std::vector<uint16_t> fused;
        concat_rows_u(fused, q_u, k_u, v_u);
        upload_u16(lw.qkv.f16, fused);
        lw.qkv.q8.free(); lw.qkv.qsc.free(); lw.qkv.qk.free(); lw.qkv.qk_segs.clear();
        lw.qkv.is_q8 = false;
        lw.qkv.bf16 = q_bf16 && k_bf16 && v_bf16;
      }
    }

    std::vector<float> qb_h, kb_h, vb_h;
    if (load_optional(loader, p + ".self_attn.q_proj.bias", qb_h) &&
        load_optional(loader, p + ".self_attn.k_proj.bias", kb_h) &&
        load_optional(loader, p + ".self_attn.v_proj.bias", vb_h)) {
      std::vector<float> qkv_bias_h;
      concat_rows(qkv_bias_h, qb_h, kb_h, vb_h);
      upload(lw.qkv_bias, qkv_bias_h);
    }

    load_matrix(loader, p + ".self_attn.o_proj.weight", lw.o, true, size_t(hidden) * q_size_);

    if (hf.num_experts > 0 && hf.layer_is_moe(layer)) {
      load_moe_layer(loader, hf, lw, layer);
    } else if (loader.contains(p + ".mlp.gate_up_proj.weight")) {
      load_matrix(loader, p + ".mlp.gate_up_proj.weight", lw.gate_up, true, size_t(2 * inter) * hidden);
    } else {
      std::vector<uint16_t> g_u, u_u;
      bool g_bf16 = false, u_bf16 = false;
      std::vector<uint8_t> g_q8, u_q8;
      bool g_is_q8 = false, u_is_q8 = false;
      std::vector<uint8_t> g_qk, u_qk;
      int g_kind = 0, u_kind = 0;
      load_matrix_host(loader, p + ".mlp.gate_proj.weight", g_u, g_bf16, g_q8, g_is_q8, g_qk, g_kind,
                       size_t(inter) * hidden, true);
      load_matrix_host(loader, p + ".mlp.up_proj.weight", u_u, u_bf16, u_q8, u_is_q8, u_qk, u_kind,
                       size_t(inter) * hidden, true);
      if (g_is_q8 || u_is_q8) {
        if (!(g_is_q8 && u_is_q8))
          throw std::runtime_error("mixed Q8/non-Q8 gate/up weights are unsupported");
        std::vector<uint8_t> fused;
        fused.reserve(g_q8.size() + u_q8.size());
        fused.insert(fused.end(), g_q8.begin(), g_q8.end());
        fused.insert(fused.end(), u_q8.begin(), u_q8.end());
        upload_q8_split(lw.gate_up, fused);
        lw.gate_up.f16.free();
        lw.gate_up.qk.free();
        lw.gate_up.qk_segs.clear();
        lw.gate_up.is_q8 = true;
      } else if (g_kind || u_kind) {
        if (!(g_kind && g_kind == u_kind))
          throw std::runtime_error("mixed K-quant gate/up needs split dispatch (unsupported)");
        if ((size_t(inter) * hidden) % 256)
          throw std::runtime_error("K-quant gate/up split is not on a super-block boundary");
        size_t gb = size_t(inter) * hidden / 256 * GGUFLoader::qk_block_bytes(g_kind);
        std::vector<uint8_t> fused;
        fused.reserve(g_qk.size() + u_qk.size());
        fused.insert(fused.end(), g_qk.begin(), g_qk.end());
        fused.insert(fused.end(), u_qk.begin(), u_qk.end());
        lw.gate_up.qk.assign(fused);
        lw.gate_up.qk_segs = {{g_kind, 0, 0}, {u_kind, gb, size_t(inter) * hidden}};
        lw.gate_up.f16.free();
        lw.gate_up.q8.free();
        lw.gate_up.qsc.free();
        lw.gate_up.is_q8 = false;
      } else {
        std::vector<uint16_t> fused;
        fused.reserve(g_u.size() + u_u.size());
        fused.insert(fused.end(), g_u.begin(), g_u.end());
        fused.insert(fused.end(), u_u.begin(), u_u.end());
        upload_u16(lw.gate_up.f16, fused);
        lw.gate_up.q8.free(); lw.gate_up.qsc.free(); lw.gate_up.qk.free(); lw.gate_up.qk_segs.clear();
        lw.gate_up.is_q8 = false;
        lw.gate_up.bf16 = g_bf16 && u_bf16;
      }
    }

    if (!(hf.num_experts > 0 && hf.layer_is_moe(layer)))
      load_matrix(loader, p + ".mlp.down_proj.weight", lw.down, true, size_t(hidden) * inter);

    std::vector<float> ln_h(hidden, 1.0f);
    load_optional(loader, p + ".input_layernorm.weight", ln_h);
    upload(lw.input_ln, ln_h);

    ln_h.assign(hidden, 1.0f);
    load_optional(loader, p + ".post_attention_layernorm.weight", ln_h);
    upload(lw.post_ln, ln_h);

    if (!hf.attention_bias) {
      std::vector<float> qn_h;
      if (load_optional(loader, p + ".self_attn.q_norm.weight", qn_h)) upload(lw.q_norm, qn_h);
      std::vector<float> kn_h;
      if (load_optional(loader, p + ".self_attn.k_norm.weight", kn_h)) upload(lw.k_norm, kn_h);
    }
  }

  std::vector<float> fn_h(hidden, 1.0f);
  load_optional(loader, "model.norm.weight", fn_h);
  upload(final_norm_, fn_h);

  if (hf.tie_word_embeddings) {
    tie_lm_head_ = true;
  } else {
    if (loader.contains("lm_head.weight")) {
      load_matrix(loader, "lm_head.weight", lm_head_, false, size_t(hf.vocab_size) * hidden);
      tie_lm_head_ = false;
    } else {
      tie_lm_head_ = true;
    }
  }

  alloc_expert_slots();
  ready_ = true;
}

int Qwen3Model::allocate_kv_cache() {
  int num_blocks = config_.num_kvcache_blocks > 0 ? config_.num_kvcache_blocks : estimate_kv_cache_blocks();
  if (num_blocks <= 0) throw std::runtime_error("cannot allocate KV cache: no free blocks");
  auto& hf = config_.hf;
  kv_layer_stride_ = static_cast<size_t>(num_blocks) * block_size_ * kv_head_stride_;
  num_blocks_ = num_blocks;
  config_.num_kvcache_blocks = num_blocks;
  kv_cache_.alloc(static_cast<size_t>(hf.num_hidden_layers) * 2 * kv_layer_stride_);
  const size_t bpb = 2ull * hf.num_hidden_layers * block_size_ * kv_head_stride_ * sizeof(float);
  std::fprintf(stderr, "KV pool: %d blocks x %.1f MB = %.2f GB (f32, bs=%d, ctx=%d)\n",
               num_blocks, bpb / 1e6, num_blocks * bpb / 1e9, block_size_, config_.max_model_len);
  return num_blocks;
}

int Qwen3Model::estimate_kv_cache_blocks() const {
  auto& hf = config_.hf;
  size_t free_mem = 0, total_mem = 0;
  HIP_CHECK(hipMemGetInfo(&free_mem, &total_mem));
  size_t bytes_per_block =
      2ull * hf.num_hidden_layers * block_size_ * kv_head_stride_ * sizeof(float);
  size_t usable = static_cast<size_t>(static_cast<double>(free_mem) * config_.gpu_memory_utilization);
  if (usable < bytes_per_block) return 0;
  int blocks = static_cast<int>(usable / bytes_per_block);
  int max_needed = (config_.max_model_len + block_size_ - 1) / block_size_ + 4;
  return std::min(blocks, max_needed);
}

static void matmul_dispatch(const float* x, const Matrix& w, int m, int n, int k, float* out) {
  if (!w.qk_segs.empty())
    hip_matmul_qk(x, w.qk.d, m, n, k, out, w.qk_segs[0].kind);
  else if (w.is_q8)
    hip_matmul_q8(x, w.q8.d, w.qsc.d, m, n, k, out);
  else
    hip_matmul_u16(x, w.f16.d, m, n, k, out, w.bf16);
}

static void matmul_row_dispatch(const float* x, const Matrix& w, size_t elem, int m, int n, int k, float* out) {
  if (!w.qk_segs.empty()) {
    const QKSeg* seg = &w.qk_segs[0];
    for (auto& s : w.qk_segs)
      if (elem >= s.elem_off) seg = &s;
    size_t bb = GGUFLoader::qk_block_bytes(seg->kind);
    if (!bb || (elem - seg->elem_off) % 256)
      throw std::runtime_error("K-quant row split off super-block boundary");
    hip_matmul_qk(x, w.qk.d + seg->byte_off + (elem - seg->elem_off) / 256 * bb, m, n, k, out, seg->kind);
  } else if (w.is_q8)
    hip_matmul_q8(x, w.q8.d + elem, w.qsc.d + elem / 32, m, n, k, out);
  else
    hip_matmul_u16(x, w.f16.d + elem, m, n, k, out, w.bf16);
}

static void embedding_dispatch(const int64_t* ids, const Matrix& w, int tokens, int hidden, float* out) {
  if (!w.qk_segs.empty())
    hip_embedding_qk(ids, w.qk.d, tokens, hidden, out, w.qk_segs[0].kind);
  else if (w.is_q8)
    hip_embedding_q8(ids, w.q8.d, w.qsc.d, tokens, hidden, out);
  else
    hip_embedding_u16(ids, w.f16.d, tokens, hidden, out, w.bf16);
}

std::vector<float> Qwen3Model::forward_logits(const Context& ctx) {
  if (!ready_) throw std::runtime_error("model weights not loaded");
  if (!kv_cache_.ptr()) throw std::runtime_error("kv cache not allocated");

  auto& hf = config_.hf;
  int rows = static_cast<int>(ctx.input_ids.size());
  int hidden = hf.hidden_size;
  int inter = hf.intermediate_size;
  int heads = hf.num_attention_heads;
  int kv_heads = hf.num_key_value_heads;
  int head_dim = hf.head_dim;
  float eps = static_cast<float>(hf.rms_norm_eps);
  float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  for (size_t i = 0; i < ctx.input_ids.size(); ++i) {
    if (ctx.input_ids[i] < 0 || ctx.input_ids[i] >= hf.vocab_size) {
      throw std::runtime_error("input token id out of vocab range: " + std::to_string(ctx.input_ids[i]));
    }
  }
  if (head_dim > 256) throw std::runtime_error("head_dim > 256 is not supported by the attention kernel");

  DeviceBuf<int64_t> d_ids;  d_ids.alloc(ctx.input_ids.size());
  HIP_CHECK(hipMemcpy(d_ids.ptr(), ctx.input_ids.data(), ctx.input_ids.size() * sizeof(int64_t), hipMemcpyHostToDevice));
  DeviceBuf<int64_t> d_pos;
  d_pos.alloc(ctx.positions.size());
  HIP_CHECK(hipMemcpy(d_pos.ptr(), ctx.positions.data(), ctx.positions.size() * sizeof(int64_t), hipMemcpyHostToDevice));
  DeviceBuf<int32_t> d_slot;
  d_slot.alloc(ctx.slot_mapping.size());
  if (!ctx.slot_mapping.empty())
    HIP_CHECK(hipMemcpy(d_slot.ptr(), ctx.slot_mapping.data(), ctx.slot_mapping.size() * sizeof(int32_t), hipMemcpyHostToDevice));
  DeviceBuf<int32_t> d_qseq;
  d_qseq.alloc(ctx.query_seq.size());
  if (!ctx.query_seq.empty())
    HIP_CHECK(hipMemcpy(d_qseq.ptr(), ctx.query_seq.data(), ctx.query_seq.size() * sizeof(int32_t), hipMemcpyHostToDevice));
  DeviceBuf<int32_t> d_qlen;
  d_qlen.alloc(ctx.query_key_len.size());
  if (!ctx.query_key_len.empty())
    HIP_CHECK(hipMemcpy(d_qlen.ptr(), ctx.query_key_len.data(), ctx.query_key_len.size() * sizeof(int32_t), hipMemcpyHostToDevice));
  DeviceBuf<int32_t> d_tables;
  d_tables.alloc(ctx.block_tables.size());
  if (!ctx.block_tables.empty())
    HIP_CHECK(hipMemcpy(d_tables.ptr(), ctx.block_tables.data(), ctx.block_tables.size() * sizeof(int32_t), hipMemcpyHostToDevice));
  DeviceBuf<int32_t> d_last;
  d_last.alloc(ctx.last_indices.size());
  if (!ctx.last_indices.empty())
    HIP_CHECK(hipMemcpy(d_last.ptr(), ctx.last_indices.data(), ctx.last_indices.size() * sizeof(int32_t), hipMemcpyHostToDevice));

  DeviceBuf<float> hidden_dev, residual_dev, norm_dev, q_dev, k_dev, v_dev, attn_dev, gate_dev, mlp_dev, selected_dev, logits_dev;
  hidden_dev.alloc(size_t(rows) * hidden);
  residual_dev.alloc(size_t(rows) * hidden);
  norm_dev.alloc(size_t(rows) * hidden);
  q_dev.alloc(size_t(rows) * q_size_);
  k_dev.alloc(size_t(rows) * kv_size_);
  v_dev.alloc(size_t(rows) * kv_size_);
  attn_dev.alloc(size_t(rows) * q_size_);
  gate_dev.alloc(size_t(rows) * 2 * inter);
  mlp_dev.alloc(size_t(rows) * inter);

  embedding_dispatch(d_ids.ptr(), embed_, rows, hidden, hidden_dev.ptr());

  for (int layer = 0; layer < hf.num_hidden_layers; ++layer) {
    auto& lw = layers_[layer];
    float* k_layer = kv_cache_.ptr() + static_cast<size_t>(layer) * 2 * kv_layer_stride_;
    float* v_layer = k_layer + kv_layer_stride_;
    hip_rms_norm_add(hidden_dev.ptr(), residual_dev.ptr(), lw.input_ln.d, norm_dev.ptr(), rows, hidden, eps);

    matmul_dispatch(norm_dev.ptr(), lw.qkv, rows, q_size_, hidden, q_dev.ptr());
    matmul_row_dispatch(norm_dev.ptr(), lw.qkv, size_t(q_size_) * hidden, rows, kv_size_, hidden, k_dev.ptr());
    matmul_row_dispatch(norm_dev.ptr(), lw.qkv, size_t(q_size_ + kv_size_) * hidden, rows, kv_size_, hidden,
                        v_dev.ptr());
    if (lw.qkv_bias.n) {
      hip_add_bias_inplace(q_dev.ptr(), lw.qkv_bias.d, rows, q_size_);
      hip_add_bias_inplace(k_dev.ptr(), lw.qkv_bias.d + q_size_, rows, kv_size_);
      hip_add_bias_inplace(v_dev.ptr(), lw.qkv_bias.d + q_size_ + kv_size_, rows, kv_size_);
    }

    if (lw.q_norm.n) hip_rms_norm(q_dev.ptr(), lw.q_norm.d, q_dev.ptr(), rows * heads, head_dim, eps);
    if (lw.k_norm.n) hip_rms_norm(k_dev.ptr(), lw.k_norm.d, k_dev.ptr(), rows * kv_heads, head_dim, eps);

    hip_rope_inplace(q_dev.ptr(), d_pos.ptr(), rows, heads, head_dim, q_size_, inv_freq_.d);
    hip_rope_inplace(k_dev.ptr(), d_pos.ptr(), rows, kv_heads, head_dim, kv_size_, inv_freq_.d);

    if (rows > 0 && !ctx.slot_mapping.empty())
      hip_store_kv(k_dev.ptr(), v_dev.ptr(), k_layer, v_layer, d_slot.ptr(), rows, kv_heads, head_dim);

    hip_paged_attention(q_dev.ptr(), attn_dev.ptr(), k_layer, v_layer, d_qseq.ptr(), d_qlen.ptr(),
                        d_tables.ptr(), rows, heads, kv_heads, head_dim, block_size_, ctx.max_blocks, scale);

     matmul_dispatch(attn_dev.ptr(), lw.o, rows, hidden, q_size_, hidden_dev.ptr());
     hip_rms_norm_add(hidden_dev.ptr(), residual_dev.ptr(), lw.post_ln.d, norm_dev.ptr(), rows, hidden, eps);

    if (lw.moe.E) {
      auto& ms = lw.moe;
      DeviceBuf<float> route_dev;
      DeviceBuf<int32_t> idx_dev;
      DeviceBuf<float> sc_dev;
      route_dev.alloc((size_t)rows * ms.E);
      idx_dev.alloc((size_t)rows * ms.K);
      sc_dev.alloc((size_t)rows * ms.K);
      matmul_dispatch(norm_dev.ptr(), ms.router, rows, ms.E, hidden, route_dev.ptr());
      hip_moe_route(route_dev.ptr(), rows, ms.E, ms.K, hf.norm_topk_prob, idx_dev.ptr(), sc_dev.ptr());
      std::vector<int32_t> h_idx((size_t)rows * ms.K);
      HIP_CHECK(hipMemcpy(h_idx.data(), idx_dev.ptr(), h_idx.size() * sizeof(int32_t), hipMemcpyDeviceToHost));
      const int SI = hf.shared_expert_intermediate_size;
      if (SI > 0 && !std::getenv("MOE_NO_SHARED")) {
        matmul_dispatch(norm_dev.ptr(), ms.sh_gu, rows, 2 * SI, hidden, gate_dev.ptr());
        hip_silu_and_mul(gate_dev.ptr(), mlp_dev.ptr(), rows, SI);
        matmul_dispatch(mlp_dev.ptr(), ms.sh_dn, rows, hidden, SI, hidden_dev.ptr());
        if (ms.sh_gate.f16.d) {
          DeviceBuf<float> shs;
          shs.alloc(rows);
          matmul_dispatch(norm_dev.ptr(), ms.sh_gate, rows, 1, hidden, shs.ptr());
          hip_scale_rows_sigmoid(hidden_dev.ptr(), shs.ptr(), rows, hidden);
        }
      } else {
        HIP_CHECK(hipMemsetAsync(hidden_dev.ptr(), 0, (size_t)rows * hidden * sizeof(float)));
      }
      // Chunk rows so one ffn launch never needs more unique experts than
      // slots (in-step eviction would leave slot_of[e]=-1 -> OOB reads).
      const int chunk = std::max(1, ms.slots / ms.K);
      for (int r0 = 0; r0 < rows; r0 += chunk) {
        int rc = std::min(chunk, rows - r0);
        std::vector<int32_t> sub(h_idx.begin() + (size_t)r0 * ms.K,
                                 h_idx.begin() + (size_t)(r0 + rc) * ms.K);
        moe_place(ms, sub, hidden, ms.I);
        hip_moe_ffn(norm_dev.ptr() + (size_t)r0 * hidden, idx_dev.ptr() + (size_t)r0 * ms.K,
                    sc_dev.ptr() + (size_t)r0 * ms.K, ms.slot_gu.ptr(), ms.slot_dn.ptr(),
                    ms.slot_of.d, rc, hidden, ms.I, ms.K, ms.bf16,
                    hidden_dev.ptr() + (size_t)r0 * hidden);
      }
    } else {
      matmul_dispatch(norm_dev.ptr(), lw.gate_up, rows, 2 * inter, hidden, gate_dev.ptr());
      hip_silu_and_mul(gate_dev.ptr(), mlp_dev.ptr(), rows, inter);
      matmul_dispatch(mlp_dev.ptr(), lw.down, rows, hidden, inter, hidden_dev.ptr());
    }
  }

  hip_rms_norm_add(hidden_dev.ptr(), residual_dev.ptr(), final_norm_.d, norm_dev.ptr(), rows, hidden, eps);

  int out_rows = static_cast<int>(ctx.last_indices.size());
  selected_dev.alloc(size_t(out_rows) * hidden);
  hip_gather_rows(norm_dev.ptr(), d_last.ptr(), rows, hidden, out_rows, selected_dev.ptr());

  const Matrix& lm = tie_lm_head_ ? embed_ : lm_head_;
  logits_dev.alloc(size_t(out_rows) * hf.vocab_size);
  matmul_dispatch(selected_dev.ptr(), lm, out_rows, hf.vocab_size, hidden, logits_dev.ptr());

  std::vector<float> out(static_cast<size_t>(out_rows) * hf.vocab_size);
  HIP_CHECK(hipMemcpy(out.data(), logits_dev.ptr(), out.size() * sizeof(float), hipMemcpyDeviceToHost));
  return out;
}
