// HIP per-position top-5 dump for a fixed token history (adjudication tool).
// Mirrors test/vk_golden_test.cpp but drives Qwen3Model directly.
// Usage: HIP_SEQ_MODEL=<model> HIP_SEQ_TOKENS="id,id,..." hip_seq_top5
// Prefills the prompt, then forces the given history and prints top-5 at each step.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "nanovllm/config.hpp"
#include "nanovllm/model.hpp"
#include "nanovllm/tokenizer.hpp"

static void top5(const float* row, int vocab) {
  std::vector<int> top(5, -1);
  std::vector<float> tv(5, -1e30f);
  for (int i = 0; i < vocab; ++i) {
    if (row[i] > tv[4]) {
      tv[4] = row[i];
      top[4] = i;
      for (int k = 3; k >= 0; --k)
        if (tv[k + 1] > tv[k]) {
          std::swap(tv[k], tv[k + 1]);
          std::swap(top[k], top[k + 1]);
        }
    }
  }
  std::printf("top5:");
  for (int k = 0; k < 5; ++k) std::printf(" %d(%.3f)", top[k], tv[k]);
}

int main() {
  const char* model = std::getenv("HIP_SEQ_MODEL");
  if (!model) {
    std::fprintf(stderr, "set HIP_SEQ_MODEL\n");
    return 2;
  }
  std::vector<int> hist;
  if (const char* t = std::getenv("HIP_SEQ_TOKENS"))
    for (const char* p = t; *p;) {
      hist.push_back(atoi(p));
      while (*p && *p != ',') ++p;
      if (*p == ',') ++p;
    }
  Config config;
  config.model = model;
  config.max_model_len = 512;
  config.load_hf_config();
  Tokenizer tok;
  tok.set_fallback_vocab_size(config.hf.vocab_size);
  tok.load(config.model);
  config.eos = tok.eos_token_id();
  Qwen3Model m(config);
  m.load_weights();
  m.allocate_kv_cache();
  auto& hf = config.hf;
  int bsize = config.kvcache_block_size, max_blocks = 64;
  const char* prompt = std::getenv("HIP_SEQ_PROMPT");
  if (!prompt) prompt = "The capital of France is";
  std::vector<int> ids = tok.encode_text(prompt);
  if (std::getenv("HIP_SEQ_CHAT")) {
    const char* cp = prompt;
    if (!std::getenv("HIP_SEQ_PROMPT")) cp = "Write a Python function to reverse a string";
    int im_start = -1, im_end = -1;
    // replicate LLMEngine::add_chat_request framing
    {
      // encode_special is const; probe via a fresh lookup path is unavailable here,
      // so resolve through the same Tokenizer API the engine uses.
      im_start = tok.encode_special("<|im_start|>");
      im_end = tok.encode_special("<|im_end|>");
    }
    std::vector<int> c;
    auto append = [&](const std::string& s) {
      auto part = tok.encode_text(s);
      c.insert(c.end(), part.begin(), part.end());
    };
    if (im_start >= 0 && im_end >= 0) {
      c.push_back(im_start);
      append(std::string("user\n") + cp);
      c.push_back(im_end);
      append("\n");
      c.push_back(im_start);
      append("assistant\n");
    } else {
      append(cp);
    }
    ids.swap(c);
  }
  std::printf("prompt ids=%d:", (int)ids.size());
  for (int v : ids) std::printf(" %d", v);
  std::printf("\n");
  // prefill
  {
    Context ctx;
    int n = (int)ids.size();
    for (int v : ids) ctx.input_ids.push_back(v);
    ctx.positions.resize(n);
    for (int i = 0; i < n; ++i) ctx.positions[i] = i;
    ctx.slot_mapping.resize(n);
    for (int i = 0; i < n; ++i) ctx.slot_mapping[i] = i;
    ctx.block_tables.assign(max_blocks, -1);
    for (int i = 0; i < n; ++i) ctx.block_tables[i / bsize] = i / bsize;
    ctx.query_seq.assign(n, 0);
    ctx.query_key_len.resize(n);
    for (int i = 0; i < n; ++i) ctx.query_key_len[i] = i + 1;
    ctx.last_indices = {n - 1};
    ctx.max_blocks = max_blocks;
    auto lg = m.forward_logits(ctx);
    std::printf("prefill ");
    top5(lg.data(), hf.vocab_size);
    std::printf("\n");
    fflush(stdout);
  }
  int pos = (int)ids.size();
  for (size_t s = 0; s < hist.size(); ++s) {
    Context ctx;
    ctx.input_ids = {(int64_t)hist[s]};
    ctx.positions = {pos};
    ctx.block_tables.assign(max_blocks, -1);
    for (int i = 0; i <= pos; ++i) ctx.block_tables[i / bsize] = i / bsize;
    ctx.slot_mapping = {pos};
    ctx.query_seq = {0};
    ctx.query_key_len = {pos + 1};
    ctx.last_indices = {0};
    ctx.max_blocks = max_blocks;
    auto lg = m.forward_logits(ctx);
    std::printf("pos%d-in%d ", pos, hist[s]);
    top5(lg.data(), hf.vocab_size);
    std::printf(" lg24=%.3f lg59=%.3f\n", lg[24], lg[59]);
    fflush(stdout);
    ++pos;
  }
  return 0;
}
