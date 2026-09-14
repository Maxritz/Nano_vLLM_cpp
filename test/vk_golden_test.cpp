#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "nanovllm/config.hpp"
#include "nanovllm/tokenizer.hpp"
#include "nanovllm/vulkan_model.hpp"

static int prefix_match(const std::vector<int>& a, const std::vector<int>& b,
                          const std::vector<int>& alt) {
  int n = (int)std::min(a.size(), b.size());
  for (int i = 0; i < n; ++i) {
    if (a[i] == b[i]) continue;
    if (i < (int)alt.size() && alt[i] >= 0 && a[i] == alt[i]) continue;
    return i;
  }
  return n;
}

static std::vector<int> greedy_vulkan_ids(const std::string& model_dir, const std::vector<int>& ids,
                                            int max_tokens, int max_model_len);

static std::vector<int> greedy_vulkan(const std::string& model_dir, const std::string& prompt,
                                      int max_tokens, int max_model_len) {
  Config config;
  config.model = model_dir; config.max_model_len = max_model_len; config.load_hf_config();
  Tokenizer tok;
  tok.set_fallback_vocab_size(config.hf.vocab_size);
  tok.load(config.model);
  return greedy_vulkan_ids(model_dir, tok.encode_text(prompt), max_tokens, max_model_len);
}

// Same chat framing as LLMEngine::add_chat_request (im_start/user/im_end/assistant).
static std::vector<int> chat_ids(Tokenizer& tok, const std::string& prompt) {
  int im_start = tok.encode_special("<|im_start|>");
  int im_end = tok.encode_special("<|im_end|>");
  std::vector<int> ids;
  auto append = [&](const std::string& s) {
    auto part = tok.encode_text(s);
    ids.insert(ids.end(), part.begin(), part.end());
  };
  if (im_start >= 0 && im_end >= 0) {
    if (tok.chat_auto_system()) {
      ids.push_back(im_start);
      append("system\nYou are a helpful assistant.");
      ids.push_back(im_end);
      append("\n");
    }
    ids.push_back(im_start);
    append("user\n" + prompt);
    ids.push_back(im_end);
    append("\n");
    ids.push_back(im_start);
    append("assistant\n");
  } else {
    append(prompt);
  }
  return ids;
}

static std::vector<int> greedy_vulkan_chat(const std::string& model_dir, const std::string& prompt,
                                           int max_tokens, int max_model_len) {
  Config config;
  config.model = model_dir; config.max_model_len = max_model_len; config.load_hf_config();
  Tokenizer tok;
  tok.set_fallback_vocab_size(config.hf.vocab_size);
  tok.load(config.model);
  return greedy_vulkan_ids(model_dir, chat_ids(tok, prompt), max_tokens, max_model_len);
}

// Parses "a,b|c,d" into want={a,b,d} + alt={-1,c,-1}: exact ties documented inline.
static void parse_ties(const char* t, std::vector<int>& want, std::vector<int>& alt) {
  want.clear();
  alt.clear();
  if (!t) return;
  for (const char* p = t; *p;) {
    want.push_back(atoi(p));
    while (*p && *p != ',' && *p != '|') ++p;
    if (*p == '|') {
      ++p;
      alt.push_back(atoi(p));
      while (*p && *p != ',') ++p;
    } else {
      alt.push_back(-1);
    }
    if (*p == ',') ++p;
  }
}
static std::vector<int> greedy_vulkan_ids(const std::string& model_dir, const std::vector<int>& ids,
                                            int max_tokens, int max_model_len) {
  Config config;
  config.model = model_dir; config.max_model_len = max_model_len; config.load_hf_config();
  Tokenizer tok;
  tok.set_fallback_vocab_size(config.hf.vocab_size);
  tok.load(config.model);
  config.eos = tok.eos_token_id();
  VulkanModel model(config);
  model.load_weights();
  int num_blocks = model.allocate_kv_cache();
  if (ids.empty()) throw std::runtime_error("empty prompt");
  auto& hf = config.hf;
  int bsize = config.kvcache_block_size, max_blocks = 64;
  std::vector<int> out;
  // prefill
  {
    VKContext ctx;
    int n = (int)ids.size();
    ctx.input_ids.assign(ids.begin(), ids.end());
    ctx.positions.resize(n); for (int i = 0; i < n; ++i) ctx.positions[i] = i;
    ctx.slot_mapping.resize(n); for (int i = 0; i < n; ++i) ctx.slot_mapping[i] = i;
    std::vector<int32_t> bt(max_blocks, -1); for (int i = 0; i < n; ++i) bt[i/bsize] = i/bsize;
    ctx.block_tables = bt;
    ctx.query_seq = std::vector<int>(n, 0); ctx.query_key_len.resize(n);
    for (int i = 0; i < n; ++i) ctx.query_key_len[i] = i + 1;
    ctx.last_indices = {n-1}; ctx.max_blocks = max_blocks;
     auto lg = model.forward_logits(ctx);
     const float* row = &lg[0];
     int nxt = (int)(std::max_element(row, row+hf.vocab_size) - row);
     { std::vector<int> top(5, -1); std::vector<float> tv(5, -1e30f);
       for (int i = 0; i < hf.vocab_size; ++i) {
         if (row[i] > tv[4]) { tv[4]=row[i]; top[4]=i;
           for (int k=3;k>=0;--k) if (tv[k+1]>tv[k]) { std::swap(tv[k],tv[k+1]); std::swap(top[k],top[k+1]); } } }
       std::fprintf(stderr, "top5 idx:");
       for (int k=0;k<5;++k) std::fprintf(stderr, " %d(%.2f)", top[k], tv[k]);
       std::fprintf(stderr, "\n"); std::fflush(stderr); }
     std::fprintf(stderr, "prefill argmax=%d\n", nxt); std::fflush(stderr);
     out.push_back(nxt);
    if (nxt == config.eos) return out;
  }
  int pos = (int)ids.size();
  while ((int)out.size() < max_tokens) {
    VKContext d;
    d.input_ids = {out.back()}; d.positions = {pos};
    std::vector<int32_t> bt(max_blocks,-1); for (int i=0;i<=pos;++i) bt[i/bsize]=i/bsize;
    d.block_tables = bt; d.slot_mapping = {pos};
    d.query_seq={0}; d.query_key_len={pos+1}; d.last_indices={0}; d.max_blocks=max_blocks;
    auto lg = model.forward_logits(d);
    const float* row = &lg[0];
    int t = (int)(std::max_element(row, row+hf.vocab_size) - row);
    std::fprintf(stderr, "decode pos=%d argmax=%d lg[0]=%g\n", pos, t, lg[0]); std::fflush(stderr);
    if (t == config.eos) break;
    out.push_back(t); ++pos;
  }
  return out;
}

static int check_prefix(const std::string& name, const std::vector<int>& got, const std::vector<int>& want,
                          const std::vector<int>& alt, int min_match) {
  int n = prefix_match(got, want, alt);
  if (n >= min_match) { std::printf("PASS %s (match %d/%d)\n", name.c_str(), n, (int)want.size()); return 0; }
  std::printf("FAIL %s (match %d/%d) want:", name.c_str(), n, (int)want.size());
  for (size_t i = 0; i < want.size(); ++i) {
    if (i < alt.size() && alt[i] >= 0) std::printf(" %d|%d", want[i], alt[i]);
    else std::printf(" %d", want[i]);
  }
  std::printf("\n  got:"); for (int t : got) std::printf(" %d", t); std::printf("\n");
  return 1;
}

int main(int argc, char** argv) {
  const char* model = std::getenv("GOLDEN_MODEL");
  if (!model) model = "G:/VLLM-Models/Qwen2.5-7B-Instruct";
  // Expected completion tokens (completion-only, like HIP). Override for the
  // model under test, e.g. GOLDEN_TOKENS="29051,29051,29051,29051".
  std::vector<int> want, alt, want_chat, alt_chat;
  parse_ties(std::getenv("GOLDEN_TOKENS"), want, alt);
  if (want.empty()) { want = {12095, 13}; alt = {-1, -1}; }
  parse_ties(std::getenv("GOLDEN_TOKENS_CHAT"), want_chat, alt_chat);
  if (want_chat.empty()) { want_chat = {95456, 0, 21193, 374, 264}; alt_chat = {-1, -1, -1, -1, -1}; }
  const char* chat_prompt = std::getenv("GOLDEN_CHAT_PROMPT");
  if (!chat_prompt) chat_prompt = "Write a Python function to reverse a string";
  int failures = 0;
  try {
    failures += check_prefix("qwen25_plain", greedy_vulkan(model, "The capital of France is", (int)want.size(), 512),
                              want, alt, (int)want.size());
    failures += check_prefix("qwen25_chat",
                              greedy_vulkan_chat(model, chat_prompt,
                                                 (int)want_chat.size(), 512),
                              want_chat, alt_chat, (int)want_chat.size());
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ERROR: %s\n", e.what()); failures++;
  }
  std::printf(failures == 0 ? "ALL VK GOLDEN TESTS PASS\n" : "VK GOLDEN TESTS FAILED\n");
  return failures;
}
