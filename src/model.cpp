#include "nanovllm/model.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <numeric>
#include <stdexcept>
#include <string>

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
  bool gguf_mode() const { return gg_; }

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

static bool load_optional(WeightLoader& loader, const std::string& name, std::vector<float>& out) {
  return loader.load_float(name, out);
}

// Loads a big matmul/embedding weight, keeping its on-disk encoding (F16/BF16 or Q8_0).
static void load_matrix(WeightLoader& loader, const std::string& name, Matrix& m, bool required) {
  std::vector<uint8_t> q8;
  if (loader.load_q8(name, q8)) {
    m.q8.assign(q8);
    m.f16.free();
    m.is_q8 = true;
    return;
  }
  std::vector<uint16_t> u;
  bool bf16 = false;
  if (loader.load_u16(name, u, bf16)) {
    m.f16.assign(u);
    m.q8.free();
    m.is_q8 = false;
    m.bf16 = bf16;
    return;
  }
  if (required) throw std::runtime_error("missing required weight: " + name);
}

// Loads a tensor as host bytes for fuse-then-upload (q/k/v, gate/up). Exactly one
// of {u16, q8} is filled; the other stays empty.
static void load_matrix_host(WeightLoader& loader, const std::string& name, std::vector<uint16_t>& u16,
                             bool& bf16, std::vector<uint8_t>& q8, bool& is_q8, bool required) {
  u16.clear();
  q8.clear();
  bf16 = false;
  is_q8 = false;
  if (loader.load_q8(name, q8)) {
    is_q8 = true;
    return;
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
static void upload_q8(DevVec<uint8_t>& dst, const std::vector<uint8_t>& src) { dst.assign(src); }

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
  WeightLoader loader;
  loader.load_model_dir(config_.model);
  if (!loader.contains("model.embed_tokens.weight")) throw std::runtime_error("model.embed_tokens.weight not found");

  auto& hf = config_.hf;
  int hidden = hf.hidden_size;
  int inter = hf.intermediate_size;

  load_matrix(loader, "model.embed_tokens.weight", embed_, true);

  for (int layer = 0; layer < hf.num_hidden_layers; ++layer) {
    std::string p = "model.layers." + std::to_string(layer);
    auto& lw = layers_[layer];

    // qkv: prefer fused, else separate q/k/v (works for both safetensors and GGUF names).
    if (loader.contains(p + ".self_attn.qkv_proj.weight")) {
      load_matrix(loader, p + ".self_attn.qkv_proj.weight", lw.qkv, true);
    } else {
      std::vector<uint16_t> q_u, k_u, v_u;
      bool q_bf16 = false, k_bf16 = false, v_bf16 = false;
      std::vector<uint8_t> q_q8, k_q8, v_q8;
      bool q_is_q8 = false, k_is_q8 = false, v_is_q8 = false;
      load_matrix_host(loader, p + ".self_attn.q_proj.weight", q_u, q_bf16, q_q8, q_is_q8, true);
      load_matrix_host(loader, p + ".self_attn.k_proj.weight", k_u, k_bf16, k_q8, k_is_q8, true);
      load_matrix_host(loader, p + ".self_attn.v_proj.weight", v_u, v_bf16, v_q8, v_is_q8, true);
      if (q_is_q8 || k_is_q8 || v_is_q8) {
        if (!(q_is_q8 && k_is_q8 && v_is_q8))
          throw std::runtime_error("mixed Q8/non-Q8 q/k/v weights are unsupported");
        std::vector<uint8_t> fused;
        concat_rows_q8(fused, q_q8, k_q8, v_q8);
        upload_q8(lw.qkv.q8, fused);
        lw.qkv.f16.free();
        lw.qkv.is_q8 = true;
      } else {
        std::vector<uint16_t> fused;
        concat_rows_u(fused, q_u, k_u, v_u);
        upload_u16(lw.qkv.f16, fused);
        lw.qkv.q8.free();
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

    load_matrix(loader, p + ".self_attn.o_proj.weight", lw.o, true);

    if (loader.contains(p + ".mlp.gate_up_proj.weight")) {
      load_matrix(loader, p + ".mlp.gate_up_proj.weight", lw.gate_up, true);
    } else {
      std::vector<uint16_t> g_u, u_u;
      bool g_bf16 = false, u_bf16 = false;
      std::vector<uint8_t> g_q8, u_q8;
      bool g_is_q8 = false, u_is_q8 = false;
      load_matrix_host(loader, p + ".mlp.gate_proj.weight", g_u, g_bf16, g_q8, g_is_q8, true);
      load_matrix_host(loader, p + ".mlp.up_proj.weight", u_u, u_bf16, u_q8, u_is_q8, true);
      if (g_is_q8 || u_is_q8) {
        if (!(g_is_q8 && u_is_q8))
          throw std::runtime_error("mixed Q8/non-Q8 gate/up weights are unsupported");
        std::vector<uint8_t> fused;
        fused.reserve(g_q8.size() + u_q8.size());
        fused.insert(fused.end(), g_q8.begin(), g_q8.end());
        fused.insert(fused.end(), u_q8.begin(), u_q8.end());
        upload_q8(lw.gate_up.q8, fused);
        lw.gate_up.f16.free();
        lw.gate_up.is_q8 = true;
      } else {
        std::vector<uint16_t> fused;
        fused.reserve(g_u.size() + u_u.size());
        fused.insert(fused.end(), g_u.begin(), g_u.end());
        fused.insert(fused.end(), u_u.begin(), u_u.end());
        upload_u16(lw.gate_up.f16, fused);
        lw.gate_up.q8.free();
        lw.gate_up.is_q8 = false;
        lw.gate_up.bf16 = g_bf16 && u_bf16;
      }
    }

    load_matrix(loader, p + ".mlp.down_proj.weight", lw.down, true);

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
      load_matrix(loader, "lm_head.weight", lm_head_, false);
      tie_lm_head_ = false;
    } else {
      tie_lm_head_ = true;
    }
  }

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
  int max_needed = (config_.max_model_len + block_size_ - 1) / block_size_ + 64;
  return std::min(blocks, max_needed);
}

static void matmul_dispatch(const float* x, const Matrix& w, int m, int n, int k, float* out) {
  if (w.is_q8)
    hip_matmul_q8(x, w.q8.d, m, n, k, out);
  else
    hip_matmul_u16(x, w.f16.d, m, n, k, out, w.bf16);
}

static void matmul_row_dispatch(const float* x, const Matrix& w, size_t elem, int m, int n, int k, float* out) {
  if (w.is_q8)
    hip_matmul_q8(x, w.q8.d + (elem / 32) * 34, m, n, k, out);
  else
    hip_matmul_u16(x, w.f16.d + elem, m, n, k, out, w.bf16);
}

static void embedding_dispatch(const int64_t* ids, const Matrix& w, int tokens, int hidden, float* out) {
  if (w.is_q8)
    hip_embedding_q8(ids, w.q8.d, tokens, hidden, out);
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

    if (lw.q_norm.n) hip_rms_norm(q_dev.ptr(), lw.q_norm.d, q_dev.ptr(), rows, head_dim, eps);
    if (lw.k_norm.n) hip_rms_norm(k_dev.ptr(), lw.k_norm.d, k_dev.ptr(), rows, head_dim, eps);

    hip_rope_inplace(q_dev.ptr(), d_pos.ptr(), rows, heads, head_dim, q_size_, inv_freq_.d);
    hip_rope_inplace(k_dev.ptr(), d_pos.ptr(), rows, kv_heads, head_dim, kv_size_, inv_freq_.d);

    if (rows > 0 && !ctx.slot_mapping.empty())
      hip_store_kv(k_dev.ptr(), v_dev.ptr(), k_layer, v_layer, d_slot.ptr(), rows, kv_heads, head_dim);

    hip_paged_attention(q_dev.ptr(), attn_dev.ptr(), k_layer, v_layer, d_qseq.ptr(), d_qlen.ptr(),
                        d_tables.ptr(), rows, heads, kv_heads, head_dim, block_size_, ctx.max_blocks, scale);

    matmul_dispatch(attn_dev.ptr(), lw.o, rows, hidden, q_size_, hidden_dev.ptr());
    hip_rms_norm_add(hidden_dev.ptr(), residual_dev.ptr(), lw.post_ln.d, norm_dev.ptr(), rows, hidden, eps);

    matmul_dispatch(norm_dev.ptr(), lw.gate_up, rows, 2 * inter, hidden, gate_dev.ptr());
    hip_silu_and_mul(gate_dev.ptr(), mlp_dev.ptr(), rows, inter);
    matmul_dispatch(mlp_dev.ptr(), lw.down, rows, hidden, inter, hidden_dev.ptr());
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
