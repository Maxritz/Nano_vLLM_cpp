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
  int sliding_window = 0;               // 0 = full attention (Gemma SWA sets N)
  std::vector<char> sliding_window_pattern;  // per-layer SWA flag (empty = uniform)
  double attn_logit_softcapping = 0.0;  // in-attention tanh cap (Gemma-2/3)
  double final_logit_softcapping = 0.0;  // final-logits tanh cap (Gemma final_logit_softcapping)
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
    float routed_scaling = 1.0f;
    int n_group = 0;
    int topk_group = 0;
    bool layer_is_moe(int l) const { return num_experts > 0 && ((l + 1) % decoder_sparse_step == 0); }
    bool is_sigmoid_group() const { return n_group > 0 && topk_group > 0; }

  static HFConfig from_json(const Json& j) {
    HFConfig c;
    // Qwen3.5-MoE multi-modal configs nest text params in "text_config"; use it if present.
    const Json* cfg = &j;
    Json tc;
    if (j.contains("text_config") && j.at("text_config").is_object()) {
      tc = j.at("text_config");
      cfg = &tc;
    }
    if (cfg->contains("model_type")) c.model_type = cfg->at("model_type").as_string();
    else if (j.contains("model_type")) c.model_type = j.at("model_type").as_string();
    c.tie_word_embeddings = (c.model_type != "qwen2" && c.model_type != "llama");
    c.hidden_size = cfg->at("hidden_size").as_int();
    c.intermediate_size = cfg->at("intermediate_size").as_int();
    c.num_attention_heads = cfg->at("num_attention_heads").as_int();
    c.num_key_value_heads = cfg->contains("num_key_value_heads") ? cfg->at("num_key_value_heads").as_int() : c.num_attention_heads;
    c.num_hidden_layers = cfg->at("num_hidden_layers").as_int();
    c.vocab_size = cfg->at("vocab_size").as_int();
    if (cfg->contains("max_position_embeddings")) c.max_position_embeddings = cfg->at("max_position_embeddings").as_int();
    if (cfg->contains("head_dim") && !cfg->at("head_dim").is_null()) c.head_dim = cfg->at("head_dim").as_int();
    if (c.head_dim == 0 && c.num_attention_heads) c.head_dim = c.hidden_size / c.num_attention_heads;
    if (cfg->contains("rope_theta")) c.rope_theta = cfg->at("rope_theta").as_number();
    if (cfg->contains("rms_norm_eps")) c.rms_norm_eps = cfg->at("rms_norm_eps").as_number();
    if (cfg->contains("sliding_window") && !cfg->at("sliding_window").is_array())
      c.sliding_window = cfg->at("sliding_window").as_int();  // array (alternating SWA) unsupported: stays 0
    if (cfg->contains("attn_logit_softcapping"))
      c.attn_logit_softcapping = cfg->at("attn_logit_softcapping").as_number();
    if (cfg->contains("final_logit_softcapping"))
      c.final_logit_softcapping = cfg->at("final_logit_softcapping").as_number();
    if (cfg->contains("sliding_window_pattern") && cfg->at("sliding_window_pattern").is_array()) {
      bool any_t = false, any_f = false;
      for (const auto& e : cfg->at("sliding_window_pattern").as_array()) {
        char v = e.as_bool() ? 1 : 0;
        c.sliding_window_pattern.push_back(v);
        any_t = any_t || v; any_f = any_f || !v;
      }
      if (!any_t) c.sliding_window = 0;
    }
    if (cfg->contains("attention_bias")) c.attention_bias = cfg->at("attention_bias").as_bool();
    if (cfg->contains("tie_word_embeddings")) c.tie_word_embeddings = cfg->at("tie_word_embeddings").as_bool();
    if (cfg->contains("hidden_act")) c.hidden_act = cfg->at("hidden_act").as_string();
    if (cfg->contains("num_experts")) c.num_experts = cfg->at("num_experts").as_int();
    else if (cfg->contains("num_local_experts")) c.num_experts = cfg->at("num_local_experts").as_int();
    if (cfg->contains("num_experts_per_tok")) c.num_experts_per_tok = cfg->at("num_experts_per_tok").as_int();
    if (cfg->contains("moe_intermediate_size")) c.moe_intermediate_size = cfg->at("moe_intermediate_size").as_int();
    if (cfg->contains("shared_expert_intermediate_size")) c.shared_expert_intermediate_size = cfg->at("shared_expert_intermediate_size").as_int();
    if (cfg->contains("decoder_sparse_step")) c.decoder_sparse_step = cfg->at("decoder_sparse_step").as_int();
    if (cfg->contains("norm_topk_prob")) c.norm_topk_prob = cfg->at("norm_topk_prob").as_bool();
    if (cfg->contains("norm_topk_probs")) c.norm_topk_prob = cfg->at("norm_topk_probs").as_bool();
    if (cfg->contains("routed_scaling_factor")) c.routed_scaling = (float)cfg->at("routed_scaling_factor").as_number();
    if (cfg->contains("n_group")) c.n_group = cfg->at("n_group").as_int();
    if (cfg->contains("topk_group")) c.topk_group = cfg->at("topk_group").as_int();
    if (cfg->contains("torch_dtype")) c.dtype = cfg->at("torch_dtype").as_string();
    if (cfg->contains("dtype")) c.dtype = cfg->at("dtype").as_string();
    if (cfg->contains("rope_scaling") && cfg->at("rope_scaling").is_object()) {
      const auto& rs = cfg->at("rope_scaling");
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
  if (g.meta_u32(arch + ".attention.sliding_window", u)) c.sliding_window = static_cast<int>(u);
  if (g.meta_f64(arch + ".attn_logit_softcapping", d)) c.attn_logit_softcapping = d;
  if (g.meta_f64(arch + ".final_logit_softcapping", d)) c.final_logit_softcapping = d;
  {
    std::vector<int32_t> pat;
    if (g.meta_i32_array(arch + ".attention.sliding_window_pattern", pat) && !pat.empty()) {
      c.sliding_window_pattern.reserve(pat.size());
      for (int32_t v : pat) c.sliding_window_pattern.push_back(v ? 1 : 0);
      bool any_t = false, any_f = false;
      for (char v : c.sliding_window_pattern) { any_t = any_t || v; any_f = any_f || !v; }
      if (!any_t) c.sliding_window = 0;  // all-global: window unused
      // Dim-heterogeneous hybrids (per-layer head dims, shared KV) exceed the
      // uniform-layer engine: fail loud instead of silently generating garbage.
      if (any_t && any_f &&
          (g.meta(arch + ".attention.key_length_swa") || g.meta(arch + ".attention.value_length_swa") ||
           g.meta(arch + ".rope.dimension_count_swa") || g.meta(arch + ".attention.shared_kv_layers")))
        throw std::runtime_error("hybrid sliding-window architecture with per-layer head dims (e.g. gemma4-style) is unsupported");
    }
  }
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

  // HF-1: defaults from sibling generation_config.json (if present). Explicit
  // CLI flags always win; main_vulkan.cpp applies these only for unset flags.
  bool has_gen_eos = false;
  int gen_eos = -1;
  bool has_gen_temperature = false;
  double gen_temperature = 0.0;
  bool has_gen_top_p = false;
  double gen_top_p = 1.0;
  bool has_gen_max_tokens = false;
  int gen_max_tokens = 64;

  // Best-effort read of <model_dir>/generation_config.json (or the .gguf
  // file's directory). Missing/unparsable file or keys -> flags stay false.
  void load_generation_defaults() {
    std::string path;
    std::error_code ec;
    if (std::filesystem::is_directory(model, ec))
      path = (std::filesystem::path(model) / "generation_config.json").string();
    else
      path = (std::filesystem::path(model).parent_path() / "generation_config.json").string();
    std::ifstream in(path, std::ios::binary);
    if (!in) return;
    try {
      std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
      Json g = JsonParser::parse(text);
      if (g.contains("eos_token_id")) {
        const Json& e = g.at("eos_token_id");
        if (e.is_array() && !e.as_array().empty()) {
          gen_eos = e.as_array()[0].as_int();
          has_gen_eos = true;
        } else if (!e.is_null() && !e.is_array()) {
          gen_eos = e.as_int();
          has_gen_eos = true;
        }
      }
      if (g.contains("temperature")) {
        gen_temperature = g.at("temperature").as_number();
        has_gen_temperature = true;
      }
      if (g.contains("top_p")) {
        gen_top_p = g.at("top_p").as_number();
        has_gen_top_p = true;
      }
      if (g.contains("max_new_tokens") && g.at("max_new_tokens").as_int() >= 1) {
        gen_max_tokens = g.at("max_new_tokens").as_int();
        has_gen_max_tokens = true;
      }
    } catch (...) {
      // Defaults only: never fail model load on a bad generation config.
    }
  }

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
        load_generation_defaults();
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
      load_generation_defaults();
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
