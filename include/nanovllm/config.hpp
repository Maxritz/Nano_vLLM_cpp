#pragma once

#include <algorithm>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>

#include "json.hpp"
#include "gguf.hpp"

struct HFConfig;

// Fill an HFConfig from GGUF metadata keys (llama.cpp naming, "<arch>.xxx").
// Only the prefill/attention/rope/ffn fields that nano-vllm needs are read.
static void fill_from_gguf(HFConfig& c, const GGUFLoader& g);

struct HFConfig {
  int hidden_size = 0;
  int intermediate_size = 0;
  int num_attention_heads = 0;
  int num_key_value_heads = 0;
  int num_hidden_layers = 0;
  int vocab_size = 0;
  int max_position_embeddings = 4096;
  int head_dim = 0;
  double rope_theta = 1000000.0;
  double rms_norm_eps = 1e-6;
  bool attention_bias = false;
  bool tie_word_embeddings = true;
  std::string hidden_act = "silu";
  std::string model_type;
  std::string dtype = "bfloat16";
  // MoE (Qwen2-5-MoE / Qwen3-MoE style). num_experts > 0 => sparse layers exist.
  int num_experts = 0;
  int num_experts_per_tok = 0;
  int moe_intermediate_size = 0;
  int shared_expert_intermediate_size = 0;
  int decoder_sparse_step = 1;
  bool norm_topk_prob = true;
  bool layer_is_moe(int l) const { return num_experts > 0 && ((l + 1) % decoder_sparse_step == 0); }

  static HFConfig from_json(const Json& j) {
    HFConfig c;
    if (j.contains("model_type")) c.model_type = j.at("model_type").as_string();
    c.tie_word_embeddings = (c.model_type != "qwen2" && c.model_type != "llama");
    c.hidden_size = j.at("hidden_size").as_int();
    c.intermediate_size = j.at("intermediate_size").as_int();
    c.num_attention_heads = j.at("num_attention_heads").as_int();
    c.num_key_value_heads = j.contains("num_key_value_heads") ? j.at("num_key_value_heads").as_int() : c.num_attention_heads;
    c.num_hidden_layers = j.at("num_hidden_layers").as_int();
    c.vocab_size = j.at("vocab_size").as_int();
    if (j.contains("max_position_embeddings")) c.max_position_embeddings = j.at("max_position_embeddings").as_int();
    if (j.contains("head_dim") && !j.at("head_dim").is_null()) c.head_dim = j.at("head_dim").as_int();
    if (c.head_dim == 0 && c.num_attention_heads) c.head_dim = c.hidden_size / c.num_attention_heads;
    if (j.contains("rope_theta")) c.rope_theta = j.at("rope_theta").as_number();
    if (j.contains("rms_norm_eps")) c.rms_norm_eps = j.at("rms_norm_eps").as_number();
    if (j.contains("attention_bias")) c.attention_bias = j.at("attention_bias").as_bool();
    if (j.contains("tie_word_embeddings")) c.tie_word_embeddings = j.at("tie_word_embeddings").as_bool();
    if (j.contains("hidden_act")) c.hidden_act = j.at("hidden_act").as_string();
    if (j.contains("num_experts")) c.num_experts = j.at("num_experts").as_int();
    if (j.contains("num_experts_per_tok")) c.num_experts_per_tok = j.at("num_experts_per_tok").as_int();
    if (j.contains("moe_intermediate_size")) c.moe_intermediate_size = j.at("moe_intermediate_size").as_int();
    if (j.contains("shared_expert_intermediate_size")) c.shared_expert_intermediate_size = j.at("shared_expert_intermediate_size").as_int();
    if (j.contains("decoder_sparse_step")) c.decoder_sparse_step = j.at("decoder_sparse_step").as_int();
    if (j.contains("norm_topk_prob")) c.norm_topk_prob = j.at("norm_topk_prob").as_bool();
    if (j.contains("torch_dtype")) c.dtype = j.at("torch_dtype").as_string();
    if (j.contains("dtype")) c.dtype = j.at("dtype").as_string();
    if (j.contains("rope_scaling") && j.at("rope_scaling").is_object()) {
      const auto& rs = j.at("rope_scaling");
      if (rs.contains("rope_theta")) c.rope_theta = rs.at("rope_theta").as_number();
    }
    return c;
  }
};

// Fill an HFConfig from GGUF metadata keys (llama.cpp naming, "<arch>.xxx").
// Only the prefill/attention/rope/ffn fields that nano-vllm needs are read.
static void fill_from_gguf(HFConfig& c, const GGUFLoader& g) {
  uint32_t u = 0;
  std::string arch;
  if (g.meta_str("general.architecture", arch)) c.model_type = arch;
  if (g.meta_u32(arch + ".block_count", u)) c.num_hidden_layers = static_cast<int>(u);
  if (g.meta_u32(arch + ".embedding_length", u)) c.hidden_size = static_cast<int>(u);
  if (g.meta_u32(arch + ".feed_forward_length", u)) c.intermediate_size = static_cast<int>(u);
  if (g.meta_u32(arch + ".attention.head_count", u)) c.num_attention_heads = static_cast<int>(u);
  if (g.meta_u32(arch + ".attention.head_count_kv", u)) c.num_key_value_heads = static_cast<int>(u);
  if (g.meta_u32(arch + ".attention.key_length", u)) c.head_dim = static_cast<int>(u);
  if (c.head_dim == 0 && c.num_attention_heads) c.head_dim = c.hidden_size / c.num_attention_heads;
  double d = 0;
  if (g.meta_f64(arch + ".attention.layer_norm_epsilon", d)) c.rms_norm_eps = d;
  if (g.meta_f64(arch + ".rope.freq_base", d)) c.rope_theta = d;
  if (g.meta_u32(arch + ".context_length", u)) c.max_position_embeddings = static_cast<int>(u);
  if (g.meta_u32(arch + ".vocab_size", u)) c.vocab_size = static_cast<int>(u);
  if (c.vocab_size <= 0) {
    std::vector<std::string> tokens;
    if (g.meta_str_array("tokenizer.ggml.tokens", tokens)) c.vocab_size = static_cast<int>(tokens.size());
  }
  c.attention_bias = g.contains("blk.0.attn_q.bias") || g.contains("blk.0.attn_k.bias");
  // GGUF always materializes a separate output tensor when the LM head is untied from
  // the embedding; its absence means the head is tied.
  c.tie_word_embeddings = !g.contains("output.weight");
}

struct Config {
  std::string model;
  int max_num_batched_tokens = 16384;
  int max_num_seqs = 512;
  int max_model_len = 4096;
  double gpu_memory_utilization = 0.9;
  int tensor_parallel_size = 1;
  bool enforce_eager = false;
  int eos = -1;
  int kvcache_block_size = 256;
  int num_kvcache_blocks = -1;
  double expert_budget_gb = 0.0;  // MoE hot-expert VRAM pool; 0 = auto (~50% of free after dense)

  HFConfig hf;

  void validate() const {
    if (model.empty()) throw std::runtime_error("model path is empty");
    if (kvcache_block_size % 256 != 0) throw std::runtime_error("kvcache_block_size must be a multiple of 256");
    if (tensor_parallel_size < 1 || tensor_parallel_size > 8) throw std::runtime_error("tensor_parallel_size must be in [1,8]");
    if (tensor_parallel_size != 1) throw std::runtime_error("tensor_parallel_size > 1 is not supported by this C++/ROCm rewrite");
    if (hf.head_dim <= 0) throw std::runtime_error("head_dim must be positive");
    if (hf.num_hidden_layers <= 0) throw std::runtime_error("num_hidden_layers must be positive");
  }

  void load_hf_config() {
    std::string gguf = find_gguf_path(model);
    if (model.find(".gguf") == std::string::npos) {
      std::string path = model + "/config.json";
      std::ifstream in(path, std::ios::binary);
      if (in) {
        std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        hf = HFConfig::from_json(JsonParser::parse(text));
        max_model_len = std::min(max_model_len, hf.max_position_embeddings);
        return;
      }
    }
    // No config.json: fall back to GGUF header metadata if present.
    if (!gguf.empty()) {
      GGUFLoader g;
      g.add_file(gguf);
      fill_from_gguf(hf, g);
      if (hf.num_hidden_layers <= 0)
        throw std::runtime_error("GGUF hparams missing block_count for " + gguf);
      max_model_len = std::min(max_model_len, hf.max_position_embeddings);
      return;
    }
    throw std::runtime_error("Cannot open " + model + "/config.json (no config.json and no .gguf)");
  }
};

struct SamplingParams {
  double temperature = 1.0;
  int max_tokens = 64;
  bool ignore_eos = false;

  void validate() const {
    if (temperature < 0) throw std::runtime_error("temperature cannot be negative");
    if (max_tokens < 1) throw std::runtime_error("max_tokens must be positive");
  }
};
