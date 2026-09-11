#pragma once

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "nanovllm/llm_engine.hpp"

static bool flag(const char* name) { return std::getenv(name) != nullptr; }

static std::vector<int> greedy_ids(const std::string& model_path, const std::vector<int>& prompt, int max_tokens,
                                   int max_model_len) {
  LLMEngine engine(model_path, max_model_len);
  SamplingParams sp;
  sp.temperature = 0;
  sp.max_tokens = max_tokens;
  auto out = engine.generate(std::vector<std::vector<int>>{prompt}, std::vector<SamplingParams>{sp});
  if (out.empty() || out[0].token_ids.empty()) throw std::runtime_error("no tokens generated");
  return out[0].token_ids;
}

static std::vector<int> greedy_ids_text(const std::string& model_path, const std::string& prompt, int max_tokens,
                                        int max_model_len) {
  LLMEngine engine(model_path, max_model_len);
  SamplingParams sp;
  sp.temperature = 0;
  sp.max_tokens = max_tokens;
  auto out = engine.generate(std::vector<std::string>{prompt}, sp);
  if (out.empty() || out[0].token_ids.empty()) throw std::runtime_error("no tokens generated");
  return out[0].token_ids;
}

static std::vector<int> greedy_ids_chat(const std::string& model_path, const std::string& prompt, int max_tokens,
                                        int max_model_len) {
  LLMEngine engine(model_path, max_model_len);
  SamplingParams sp;
  sp.temperature = 0;
  sp.max_tokens = max_tokens;
  auto out = engine.generate_chat(std::vector<std::string>{prompt}, std::vector<SamplingParams>{sp});
  if (out.empty() || out[0].token_ids.empty()) throw std::runtime_error("no tokens generated");
  return out[0].token_ids;
}

// Returns the first index where the two token sequences diverge, or N if identical.
static int prefix_match(const std::vector<int>& a, const std::vector<int>& b) {
  int n = (int)std::min(a.size(), b.size());
  for (int i = 0; i < n; ++i)
    if (a[i] != b[i]) return i;
  return n;
}